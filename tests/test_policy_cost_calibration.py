"""Tests for deterministic policy-frequency search-cost calibration."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import tools.policy_cost_calibration_v2 as calibration

from tools.policy_cost_calibration_v2 import (
    CalibrationError,
    DEFAULT_FOLD_SEED,
    DEFAULT_PLY_ANCHORS,
    DEFAULT_STANDARD_ERROR_FLOOR,
    FitConfig,
    PairObservation,
    PolicyCostSchedule,
    calibrate_policy_cost,
    derived_gap_threshold_table,
    make_group_folds,
    read_observation_jsonl,
    write_result_no_clobber,
)


ANCHORS = (0, 10, 20)


def observation(
    group: int,
    ply: int,
    core_ratio: float,
    draw_ratio: float,
    *,
    search_beta: float = 1.0,
    core_lambda: float = 1.5,
    draw_lambda: float = 0.5,
    search_delta: float = 0.0,
    noise: float = 0.0,
    state_suffix: str = "0",
    pair_suffix: str = "0",
    search_se: float = 0.2,
    truth_se: float = 0.2,
    search_panel_id: str = "primary",
) -> PairObservation:
    truth_delta = (
        search_beta * search_delta
        + core_lambda * math.log(core_ratio)
        + draw_lambda * math.log(draw_ratio)
        + noise
    )
    return PairObservation(
        source_match_id=f"match-{group:03d}",
        state_id=f"match-{group:03d}-ply-{ply}-{state_suffix}",
        pair_id=f"pair-{pair_suffix}",
        ply=ply,
        search_delta=search_delta,
        truth_delta=truth_delta,
        log_core_ratio=math.log(core_ratio),
        log_draw_ratio=math.log(draw_ratio),
        search_se=search_se,
        truth_se=truth_se,
        search_panel_id=search_panel_id,
        truth_panel_id="independent-truth-panel",
    )


def primary_fresh(row: PairObservation) -> list[PairObservation]:
    if row.search_panel_id != "primary":
        raise AssertionError("fixture row is not primary")
    return [row, replace(row, search_panel_id="fresh")]


def observation_mapping(row: PairObservation) -> dict:
    return {
        "source_match_id": row.source_match_id,
        "state_id": row.state_id,
        "pair_id": row.pair_id,
        "round": row.round_index,
        "ply": row.ply,
        "pair_type": row.pair_type,
        "search_delta": row.search_delta,
        "truth_delta": row.truth_delta,
        "log_core_ratio": row.log_core_ratio,
        "log_draw_ratio": row.log_draw_ratio,
        "search_se": row.search_se,
        "truth_se": row.truth_se,
        "search_panel_id": row.search_panel_id,
        "truth_panel_id": row.truth_panel_id,
        "orientation": row.orientation,
        "state_weight": row.state_weight,
    }
def synthetic_rows() -> list[PairObservation]:
    rows: list[PairObservation] = []
    for group in range(18):
        for ply in ANCHORS:
            core_ratio = (1.25, 2.0, 4.0)[(group + ply // 10) % 3]
            draw_ratio = (0.5, 1.5, 3.0)[(2 * group + ply // 10) % 3]
            noise = ((group % 5) - 2) * 0.002
            rows.extend(primary_fresh(observation(
                group, ply, core_ratio, draw_ratio,
                search_delta=((group % 7) - 3) * 0.17 + ply * 0.003,
                noise=noise,
            )))
    # A large truth-panel outlier exercises Huber downweighting without
    # changing the source-match grouping contract.
    rows.extend(primary_fresh(observation(
        17, 10, 8.0, 2.0, search_delta=0.37, noise=100.0,
        state_suffix="outlier",
    )))
    return rows


def campaign_rows(
    quota: int, *, round_effect: float = 0.0, gap_scale: float = 1.0,
) -> list[PairObservation]:
    rows: list[PairObservation] = []
    ratios = (1.1, 1.5, 3.0, 6.0, 16.0, 64.0)
    for round_index in range(3):
        for ply_bin, (lower, upper) in enumerate(calibration.CAMPAIGN_PLY_BINS):
            for ratio_bin, ratio in enumerate(ratios):
                for pair_type in calibration.PAIR_TYPES:
                    for member in range(quota):
                        ply = lower + member % (upper - lower)
                        source = (
                            f"campaign-r{round_index}-p{ply_bin}-g{ratio_bin}-"
                            f"t{pair_type}-q{member}"
                        )
                        if pair_type == "different_core":
                            log_core = math.log(ratio)
                            log_draw = math.log(
                                (0.5, 0.8, 1.0, 1.25, 1.5)[member % 5]
                            )
                        else:
                            log_core = 0.0
                            log_draw = math.log(ratio)
                        search = ((member % 5) - 2) * 0.03
                        truth = (
                            search
                            + gap_scale * (
                                (1.5 + round_effect * round_index) * log_core
                                + (0.5 + round_effect * round_index) * log_draw
                            )
                            + ((member % 3) - 1) * 0.004
                        )
                        for panel in ("primary", "fresh"):
                            rows.append(PairObservation(
                                source_match_id=source,
                                state_id=source,
                                pair_id="pair",
                                round_index=round_index,
                                pair_type=pair_type,
                                ply=ply,
                                search_delta=(
                                    search + (0.002 if panel == "fresh" else 0.0)
                                ),
                                truth_delta=truth,
                                log_core_ratio=log_core,
                                log_draw_ratio=log_draw,
                                search_se=0.2,
                                truth_se=0.2,
                                search_panel_id=panel,
                                truth_panel_id="independent-truth-panel",
                            ))
    return rows


def campaign_manifest(rows: list[PairObservation]) -> dict:
    primary = {
        row.source_match_id: row
        for row in rows if row.search_panel_id == "primary"
    }
    units = []
    for source, row in primary.items():
        round_index, ply_bin, ratio_bin, pair_type = (
            calibration._campaign_cell(row)
        )
        state_sha = hashlib.sha256(
            f"state/{row.state_id}".encode("utf-8")
        ).hexdigest()
        pair_sha = hashlib.sha256(
            f"pair/{row.state_id}/{row.pair_id}".encode("utf-8")
        ).hexdigest()
        priority = hashlib.sha256(
            f"priority/{source}".encode("utf-8")
        ).hexdigest()
        units.append({
            "source_match_id": source,
            "state_id": row.state_id,
            "pair_id": row.pair_id,
            "state_sha256": state_sha,
            "pair_sha256": pair_sha,
            "allocation_priority_sha256": priority,
            "round": round_index,
            "ply_bin": ply_bin,
            "ratio_bin": ratio_bin,
            "pair_type": pair_type,
        })
    units.sort(key=lambda unit: (
        unit["round"], unit["ply_bin"], unit["ratio_bin"],
        unit["pair_type"], unit["allocation_priority_sha256"],
        unit["state_sha256"], unit["source_match_id"], unit["state_id"],
        unit["pair_id"],
    ))
    payload = {
        "schema": calibration.TRAIN_ALLOCATION_SCHEMA,
        "source_reservoir_sha256": "a" * 64,
        "eligible_pair_commitment_sha256": "b" * 64,
        "allocation_rule_sha256": "c" * 64,
        "ply_bins": [list(pair) for pair in calibration.CAMPAIGN_PLY_BINS],
        "ratio_bins": list(calibration.RATIO_BAND_LABELS),
        "pair_types": list(calibration.PAIR_TYPES),
        "cell_quota": calibration.CAMPAIGN_CELL_QUOTA,
        "selected_units": units,
    }
    digest = hashlib.sha256((json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")).hexdigest()
    return {**payload, calibration.CANONICAL_PAYLOAD_SHA256: digest}


class PolicyCostCalibrationTests(unittest.TestCase):
    maxDiff = None

    def test_default_anchor_schedule_is_frozen(self) -> None:
        self.assertEqual(
            DEFAULT_PLY_ANCHORS,
            (0, 4, 8, 12, 16, 24, 32, 40, 48, 64),
        )
        self.assertEqual(DEFAULT_FOLD_SEED, "202612140101")
        self.assertEqual(DEFAULT_STANDARD_ERROR_FLOOR, 0.25)

        tail = PolicyCostSchedule(
            anchors=DEFAULT_PLY_ANCHORS,
            beta_search=tuple(1.0 for _ in range(10)),
            alpha_core=tuple(float(i) for i in range(10)),
            alpha_draw=tuple(0.0 for _ in range(10)),
        )
        self.assertEqual(tail.lambdas_at(64), tail.lambdas_at(299))
        table = derived_gap_threshold_table(tail)
        self.assertEqual(len(table), 300)
        self.assertEqual(table[64]["required_search_advantage"],
                         table[299]["required_search_advantage"])
        self.assertEqual(
            table[64][
                "ratio_band_lower_bound_required_search_advantage"
            ],
            table[299][
                "ratio_band_lower_bound_required_search_advantage"
            ],
        )
        self.assertEqual(
            table[0]["ratio_band_lower_bound_required_search_advantage"]
            ["semantic_core"]["[1,1.25)"],
            0.0,
        )

    def test_scalar_thresholds_are_transitive(self) -> None:
        schedule = PolicyCostSchedule(
            anchors=ANCHORS,
            beta_search=(1.0, 1.0, 1.0),
            alpha_core=(2.0, 2.0, 2.0),
            alpha_draw=(0.75, 0.75, 0.75),
        )
        ab = schedule.required_search_advantage(
            ply=8,
            incumbent_core_probability=0.95,
            challenger_core_probability=0.04,
            incumbent_draw_probability=0.8,
            challenger_draw_probability=0.5,
        )
        bc = schedule.required_search_advantage(
            ply=8,
            incumbent_core_probability=0.04,
            challenger_core_probability=0.01,
            incumbent_draw_probability=0.5,
            challenger_draw_probability=0.2,
        )
        ac = schedule.required_search_advantage(
            ply=8,
            incumbent_core_probability=0.95,
            challenger_core_probability=0.01,
            incumbent_draw_probability=0.8,
            challenger_draw_probability=0.2,
        )
        self.assertAlmostEqual(ac, ab + bc, places=13)

    def test_policy_gap_examples_and_one_percent_must_beat_four(self) -> None:
        schedule = PolicyCostSchedule(
            anchors=ANCHORS,
            beta_search=(1.0, 1.0, 1.0),
            alpha_core=(2.0, 2.0, 2.0),
            alpha_draw=(0.0, 0.0, 0.0),
        )
        table = derived_gap_threshold_table(schedule, plies=(5,))[0]
        thresholds = table["required_search_advantage"]
        self.assertAlmostEqual(
            thresholds["55_to_45"], 2.0 * math.log(0.55 / 0.45)
        )
        self.assertAlmostEqual(
            thresholds["95_to_4"], 2.0 * math.log(0.95 / 0.04)
        )
        self.assertAlmostEqual(
            thresholds["95_to_1"], 2.0 * math.log(0.95 / 0.01)
        )
        self.assertAlmostEqual(
            thresholds["4_to_1"], 2.0 * math.log(4.0)
        )
        self.assertGreater(thresholds["95_to_4"],
                           10.0 * thresholds["55_to_45"])

        four_percent_q = 0.0
        just_short = thresholds["4_to_1"] - 1.0e-6
        just_over = thresholds["4_to_1"] + 1.0e-6
        self.assertEqual(schedule.choose(
            ply=5,
            search_q=(four_percent_q, just_short),
            core_probabilities=(0.04, 0.01),
        ), 0)
        self.assertEqual(schedule.choose(
            ply=5,
            search_q=(four_percent_q, just_over),
            core_probabilities=(0.04, 0.01),
        ), 1)

        # The 1% action clears the 95% action considered in isolation, but it
        # still cannot bypass a stronger 4% action.  A single global score,
        # rather than a candidate-zero-only test, enforces this automatically.
        q_four = thresholds["95_to_4"] + 0.2
        q_one = thresholds["95_to_1"] + 0.1
        self.assertGreater(q_one, thresholds["95_to_1"])
        self.assertLess(q_one - q_four, thresholds["4_to_1"])
        self.assertEqual(schedule.choose(
            ply=5,
            search_q=(0.0, q_four, q_one),
            core_probabilities=(0.95, 0.04, 0.01),
        ), 1)

    def test_conditional_draw_probability_is_a_separate_cost(self) -> None:
        schedule = PolicyCostSchedule(
            anchors=ANCHORS,
            beta_search=(1.0, 1.0, 1.0),
            alpha_core=(4.0, 4.0, 4.0),
            alpha_draw=(0.5, 0.5, 0.5),
        )
        threshold = schedule.required_search_advantage(
            ply=10,
            incumbent_core_probability=0.2,
            challenger_core_probability=0.2,
            incumbent_draw_probability=0.9,
            challenger_draw_probability=0.1,
        )
        self.assertAlmostEqual(threshold, 0.5 * math.log(9.0))

    def test_grouped_folds_are_balanced_fixed_and_never_leak(self) -> None:
        rows = synthetic_rows()
        assignment = make_group_folds(rows, folds=5, seed="fixed-seed")
        again = make_group_folds(list(reversed(rows)), folds=5,
                                 seed="fixed-seed")
        self.assertEqual(assignment, again)
        counts = [sum(fold == index for fold in assignment.values())
                  for index in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)
        for fold in range(5):
            training_groups = {
                row.source_match_id for row in rows
                if assignment[row.source_match_id] != fold
            }
            validation_groups = {
                row.source_match_id for row in rows
                if assignment[row.source_match_id] == fold
            }
            self.assertTrue(training_groups.isdisjoint(validation_groups))

    def test_conventional_one_se_uses_minimum_model_cluster_se(self) -> None:
        # Candidate-specific paired-difference SE=0.01 would reject the
        # smoother 1.0 row, whereas the frozen conventional threshold uses
        # the minimum model's 0.10 source-cluster SE and accepts it.
        selected = calibration._conventional_one_se_smoothness((
            (0.0, 1.00, 0.10),
            (1.0, 1.08, 0.01),
            (10.0, 1.11, 0.01),
        ))
        self.assertEqual(selected, 1.0)

    def test_variance_standardization_favors_precise_panels(self) -> None:
        precise_positive: list[PairObservation] = []
        precise_zero: list[PairObservation] = []
        for group in range(12):
            for ply in ANCHORS:
                precise_positive.extend(primary_fresh(observation(
                        group, ply, 2.0, 1.0, core_lambda=2.0,
                        state_suffix="precise-positive", search_se=0.05,
                        truth_se=0.05,
                    )))
                precise_positive.extend(primary_fresh(observation(
                        group, ply, 2.0, 1.0, core_lambda=0.0,
                        state_suffix="imprecise-zero", search_se=5.0,
                        truth_se=5.0,
                    )))
                precise_zero.extend(primary_fresh(observation(
                        group, ply, 2.0, 1.0, core_lambda=2.0,
                        state_suffix="imprecise-positive", search_se=5.0,
                        truth_se=5.0,
                    )))
                precise_zero.extend(primary_fresh(observation(
                        group, ply, 2.0, 1.0, core_lambda=0.0,
                        state_suffix="precise-zero", search_se=0.05,
                        truth_se=0.05,
                    )))
        config = FitConfig(
            anchors=ANCHORS, smoothness_grid=(0.0,), folds=3,
            fold_seed="variance-test",
        )
        positive = calibrate_policy_cost(precise_positive, config)
        zero = calibrate_policy_cost(precise_zero, config)
        self.assertTrue(all(value > 1.8
                            for value in positive.schedule.alpha_core))
        self.assertTrue(all(value < 0.2
                            for value in zero.schedule.alpha_core))
        self.assertNotIn("robust_residual_scale", positive.to_dict()["fit"])
        self.assertIn("sqrt(search_se^2 + truth_se^2 +",
                      positive.to_dict()["fit"]["variance_standardization"])

    def test_zero_panel_ses_use_the_substantive_frozen_floor(self) -> None:
        rows: list[PairObservation] = []
        for group in range(9):
            for ply in ANCHORS:
                rows.extend(primary_fresh(observation(
                    group, ply, 2.0, 1.0,
                    search_se=0.0, truth_se=0.0,
                )))
        result = calibrate_policy_cost(rows, FitConfig(
            anchors=ANCHORS, smoothness_grid=(0.0,), folds=3,
        ))
        self.assertEqual(result.standard_error_floor, 0.25)
        self.assertTrue(all(math.isfinite(value)
                            for value in result.schedule.alpha_core))

    def test_each_state_has_equal_total_weight_despite_pair_count(self) -> None:
        base: list[PairObservation] = []
        expanded: list[PairObservation] = []
        for group in range(12):
            coefficient = 2.0 if group % 2 == 0 else 0.0
            for ply in ANCHORS:
                row = observation(
                    group, ply, 2.0, 1.0, core_lambda=coefficient,
                    state_suffix="equal-state", search_se=10.0,
                    truth_se=10.0,
                )
                base.extend(primary_fresh(row))
                if coefficient == 2.0:
                    for pair in range(7):
                        expanded.extend(primary_fresh(observation(
                            group, ply, 2.0, 1.0,
                            core_lambda=coefficient,
                            state_suffix="equal-state",
                            pair_suffix=str(pair),
                            search_se=10.0, truth_se=10.0,
                        )))
                else:
                    expanded.extend(primary_fresh(row))
        config = FitConfig(
            anchors=ANCHORS, smoothness_grid=(0.0,), folds=3,
            fold_seed="state-equality",
        )
        base_result = calibrate_policy_cost(base, config)
        expanded_result = calibrate_policy_cost(expanded, config)
        for left, right in zip(base_result.schedule.alpha_core,
                               expanded_result.schedule.alpha_core):
            self.assertAlmostEqual(left, right, places=11)
        for left, right in zip(base_result.schedule.alpha_draw,
                               expanded_result.schedule.alpha_draw):
            self.assertAlmostEqual(left, right, places=11)

    def test_primary_and_fresh_may_share_one_independent_truth_panel(self) -> None:
        rows: list[PairObservation] = []
        for group in range(9):
            for ply in ANCHORS:
                rows.extend((
                    observation(
                        group, ply, 2.0, 1.0,
                        state_suffix="shared-truth",
                        search_panel_id="primary",
                    ),
                    observation(
                        group, ply, 2.0, 1.0,
                        state_suffix="shared-truth",
                        search_panel_id="fresh",
                    ),
                ))
        result = calibrate_policy_cost(rows, FitConfig(
            anchors=ANCHORS, smoothness_grid=(0.0,), folds=3,
        ))
        self.assertEqual(result.source_match_count, 9)
        self.assertEqual(result.observation_count, 54)

        contaminated = list(rows)
        row = contaminated[-1]
        contaminated[-1] = PairObservation(
            source_match_id=row.source_match_id,
            state_id=row.state_id,
            pair_id=row.pair_id,
            ply=row.ply,
            search_delta=row.search_delta,
            truth_delta=row.truth_delta,
            log_core_ratio=row.log_core_ratio,
            log_draw_ratio=row.log_draw_ratio,
            search_se=row.search_se,
            truth_se=row.truth_se,
            search_panel_id=row.truth_panel_id,
            truth_panel_id=row.truth_panel_id,
        )
        with self.assertRaises(CalibrationError):
            calibrate_policy_cost(contaminated, FitConfig(
                anchors=ANCHORS, smoothness_grid=(0.0,), folds=3,
            ))

        with self.assertRaisesRegex(CalibrationError, "search panels differ"):
            calibrate_policy_cost(rows[:-1], FitConfig(
                anchors=ANCHORS, smoothness_grid=(0.0,), folds=3,
            ))
        unexpected = list(rows)
        unexpected[-1] = replace(
            unexpected[-1], search_panel_id="secondary",
        )
        with self.assertRaisesRegex(CalibrationError, "search panels differ"):
            calibrate_policy_cost(unexpected, FitConfig(
                anchors=ANCHORS, smoothness_grid=(0.0,), folds=3,
            ))

    def test_primary_fresh_are_separate_half_weight_observations(self) -> None:
        primary, fresh = primary_fresh(observation(
            0, 10, 2.0, 1.0, search_delta=0.4,
        ))
        design, target, weights = calibration._design(
            (primary, fresh), ANCHORS, DEFAULT_STANDARD_ERROR_FLOOR
        )
        self.assertEqual(design.shape, (2, 9))
        self.assertEqual(target.shape, (2,))
        self.assertEqual(tuple(weights), (0.5, 0.5))

    def test_fit_is_nonnegative_constrained_and_recovers_signal(self) -> None:
        config = FitConfig(
            anchors=ANCHORS,
            smoothness_grid=(0.0, 0.01, 1.0),
            folds=3,
            fold_seed="constraint-test",
            min_core_alpha=0.1,
            min_draw_alpha=0.05,
        )
        result = calibrate_policy_cost(synthetic_rows(), config)
        self.assertTrue(all(value >= 0.1
                            for value in result.schedule.alpha_core))
        self.assertTrue(all(value >= 0.05
                            for value in result.schedule.alpha_draw))
        for value in result.schedule.alpha_core:
            self.assertAlmostEqual(value, 1.5, delta=0.2)
        for value in result.schedule.alpha_draw:
            self.assertAlmostEqual(value, 0.5, delta=0.12)
        for value in result.schedule.beta_search:
            self.assertAlmostEqual(value, 1.0, delta=0.2)
        self.assertIn("beta_search", result.to_dict()["model"]["score"])
        self.assertEqual(result.to_dict()["model"]["candidate_zero_bonus"],
                         "absent")

    def test_search_shrinkage_is_fitted_instead_of_fixed_to_one(self) -> None:
        rows: list[PairObservation] = []
        for group in range(18):
            for ply in ANCHORS:
                rows.extend(primary_fresh(observation(
                    group,
                    ply,
                    (1.25, 2.0, 4.0)[group % 3],
                    (0.7, 1.0, 2.0)[(group + ply // 10) % 3],
                    search_beta=0.4,
                    core_lambda=1.2,
                    draw_lambda=0.3,
                    search_delta=((group % 9) - 4) * 0.31 + 0.01 * ply,
                    noise=((group % 5) - 2) * 0.001,
                )))
        result = calibrate_policy_cost(rows, FitConfig(
            anchors=ANCHORS,
            smoothness_grid=(0.0,),
            folds=3,
            fold_seed="predictive-beta",
        ))
        for value in result.schedule.beta_search:
            self.assertAlmostEqual(value, 0.4, delta=0.03)
        for value in result.schedule.alpha_core:
            self.assertAlmostEqual(value, 1.2, delta=0.04)
        for value in result.schedule.alpha_draw:
            self.assertAlmostEqual(value, 0.3, delta=0.04)

    def test_noisy_search_learns_policy_shrinkage_old_residual_misses(self) -> None:
        rows: list[PairObservation] = []
        residual_core_cross_product = 0.0
        group = 0
        for core_ratio in (1.25, 2.0, 4.0):
            gap = math.log(core_ratio)
            for latent in (-1.0, 0.0, 1.0):
                truth = 2.0 * gap + latent
                for primary_error in (-2.0, 0.0, 2.0):
                    for fresh_error in (-2.0, 0.0, 2.0):
                        for ply in ANCHORS:
                            state = f"latent-{group}-ply-{ply}"
                            for panel, error in (
                                ("primary", primary_error),
                                ("fresh", fresh_error),
                            ):
                                search = truth + error
                                rows.append(PairObservation(
                                    source_match_id=f"latent-{group}",
                                    state_id=state,
                                    pair_id="pair",
                                    ply=ply,
                                    search_delta=search,
                                    truth_delta=truth,
                                    log_core_ratio=gap,
                                    log_draw_ratio=0.0,
                                    search_se=0.2,
                                    truth_se=0.2,
                                    search_panel_id=panel,
                                    truth_panel_id="independent-truth",
                                ))
                                residual_core_cross_product += (
                                    gap * (truth - search)
                                )
                        group += 1
        # The rejected fixed-beta residual target has exactly zero policy-gap
        # cross-product in this factorial design, despite informative policy.
        self.assertAlmostEqual(residual_core_cross_product, 0.0, places=12)
        result = calibrate_policy_cost(rows, FitConfig(
            anchors=ANCHORS,
            smoothness_grid=(0.0,),
            folds=3,
            fold_seed="latent-noisy-search",
        ))
        self.assertTrue(all(0.05 < value < 0.5
                            for value in result.schedule.beta_search))
        self.assertTrue(all(value > 1.0
                            for value in result.schedule.alpha_core))

    def test_raw_coefficients_interpolate_before_alpha_over_beta(self) -> None:
        schedule = PolicyCostSchedule(
            anchors=(0, 10),
            beta_search=(1.0, 3.0),
            alpha_core=(1.0, 9.0),
            alpha_draw=(0.0, 6.0),
        )
        beta, core, draw = schedule.coefficients_at(5)
        self.assertEqual((beta, core, draw), (2.0, 5.0, 3.0))
        self.assertEqual(schedule.lambdas_at(5), (2.5, 1.5))
        # Averaging the endpoint ratios would instead give (2, 1).
        self.assertNotEqual(schedule.lambdas_at(5), (2.0, 1.0))

    def test_adverse_signal_projects_to_zero(self) -> None:
        rows = [
            PairObservation(
                source_match_id=f"negative-{group}",
                state_id=f"negative-{group}-{ply}",
                pair_id="pair-0",
                ply=ply,
                search_delta=0.0,
                truth_delta=-math.log(2.0),
                log_core_ratio=math.log(2.0),
                log_draw_ratio=0.0,
                search_se=0.2,
                truth_se=0.2,
                search_panel_id=panel,
                truth_panel_id="truth",
            )
            for group in range(9) for ply in ANCHORS
            for panel in ("primary", "fresh")
        ]
        result = calibrate_policy_cost(rows, FitConfig(
            anchors=ANCHORS,
            smoothness_grid=(0.0, 1.0),
            folds=3,
        ))
        self.assertTrue(all(value <= 1.0e-9
                            for value in result.schedule.alpha_core))
        self.assertTrue(all(value <= 1.0e-9
                            for value in result.schedule.alpha_draw))

    def test_one_se_rule_prefers_smoother_exact_tie(self) -> None:
        rows = [
            PairObservation(
                source_match_id=f"flat-{group}",
                state_id=f"flat-{group}-{ply}",
                pair_id="pair-0",
                ply=ply,
                search_delta=0.0,
                truth_delta=0.0,
                log_core_ratio=0.0,
                log_draw_ratio=0.0,
                search_se=0.2,
                truth_se=0.2,
                search_panel_id=panel,
                truth_panel_id="truth",
            )
            for group in range(9) for ply in ANCHORS
            for panel in ("primary", "fresh")
        ]
        result = calibrate_policy_cost(rows, FitConfig(
            anchors=ANCHORS,
            smoothness_grid=(0.0, 0.5, 7.0),
            folds=3,
        ))
        self.assertEqual(result.selected_smoothness, 7.0)

    def test_calibration_is_byte_deterministic_and_order_invariant(self) -> None:
        config = FitConfig(
            anchors=ANCHORS,
            smoothness_grid=(0.0, 0.1),
            folds=3,
            fold_seed="determinism",
        )
        rows = synthetic_rows()
        first_result = calibrate_policy_cost(rows, config)
        second_result = calibrate_policy_cost(list(reversed(rows)), config)
        self.assertEqual(first_result.canonical_json(),
                         second_result.canonical_json())
        self.assertEqual(first_result.observation_input_sha256,
                         second_result.observation_input_sha256)
        changed = list(rows)
        changed[0] = replace(
            changed[0], search_delta=changed[0].search_delta + 0.001,
        )
        changed_result = calibrate_policy_cost(changed, config)
        self.assertNotEqual(first_result.observation_input_sha256,
                            changed_result.observation_input_sha256)
        self.assertEqual(
            first_result.to_dict()["observation_input_sha256"],
            first_result.observation_input_sha256,
        )
        solver = first_result.to_dict()["fit"]["solver"]
        self.assertEqual(solver["max_irls_iterations"], 500)
        self.assertTrue(solver["fail_closed_on_nonconvergence"])
        self.assertTrue(solver["final_fit_convergence"]["converged"])
        self.assertGreater(
            solver["final_fit_convergence"]["irls_iterations"], 0
        )
        self.assertTrue(all(
            fit["converged"]
            for row in first_result.to_dict()["fit"]["cross_validation"]
            for fit in row["fit_convergence"]
        ))

    def test_solver_iteration_cap_fails_closed(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "IRLS did not converge"):
            calibrate_policy_cost(
                synthetic_rows(),
                FitConfig(
                    anchors=ANCHORS,
                    smoothness_grid=(0.0,),
                    folds=3,
                    max_irls_iterations=1,
                ),
            )

    def test_strict_jsonl_and_exact_input_keys(self) -> None:
        primary, fresh = primary_fresh(observation(0, 0, 2.0, 1.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "\n".join(json.dumps(observation_mapping(row))
                            for row in (primary, fresh)) + "\n",
                encoding="utf-8",
            )
            parsed = read_observation_jsonl(path)
            self.assertEqual(parsed, [primary, fresh])

            row_text = json.dumps(observation_mapping(primary))
            duplicate = row_text.replace('"ply": 0', '"ply": 0, "ply": 0')
            path.write_text(duplicate + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "duplicate JSON key"):
                read_observation_jsonl(path)

            nonfinite = observation_mapping(primary)
            nonfinite["search_delta"] = math.nan
            path.write_text(json.dumps(nonfinite) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "non-finite JSON"):
                read_observation_jsonl(path)

        extra = observation_mapping(primary)
        extra["ignored"] = 1
        with self.assertRaisesRegex(CalibrationError, "keys differ"):
            PairObservation.from_mapping(extra)
        missing = observation_mapping(primary)
        del missing["truth_panel_id"]
        with self.assertRaisesRegex(CalibrationError, "keys differ"):
            PairObservation.from_mapping(missing)

    def test_output_publish_is_atomic_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            write_result_no_clobber(destination, b"first\n")
            self.assertEqual(destination.read_bytes(), b"first\n")
            with self.assertRaisesRegex(CalibrationError, "refusing to replace"):
                write_result_no_clobber(destination, b"second\n")
            self.assertEqual(destination.read_bytes(), b"first\n")
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*")), []
            )

    def test_invalid_or_non_independent_inputs_fail_closed(self) -> None:
        with self.assertRaises(CalibrationError):
            PairObservation.from_probabilities(
                source_match_id="m",
                state_id="s",
                pair_id="p",
                ply=0,
                search_delta=0.0,
                truth_delta=0.0,
                left_core_probability=0.0,
                right_core_probability=0.5,
                search_se=0.2,
                truth_se=0.2,
            )
        bad = PairObservation(
            source_match_id="m",
            state_id="s",
            pair_id="p",
            ply=0,
            search_delta=0.0,
            truth_delta=0.0,
            log_core_ratio=0.0,
            log_draw_ratio=0.0,
            search_se=0.2,
            truth_se=0.2,
            search_panel_id="same",
            truth_panel_id="same",
        )
        with self.assertRaises(CalibrationError):
            calibrate_policy_cost([bad], FitConfig(
                anchors=ANCHORS, folds=2,
            ))

        wrong_orientation = observation(0, 0, 2.0, 1.0)
        wrong_orientation = PairObservation(
            source_match_id=wrong_orientation.source_match_id,
            state_id=wrong_orientation.state_id,
            pair_id=wrong_orientation.pair_id,
            ply=wrong_orientation.ply,
            search_delta=wrong_orientation.search_delta,
            truth_delta=wrong_orientation.truth_delta,
            log_core_ratio=wrong_orientation.log_core_ratio,
            log_draw_ratio=wrong_orientation.log_draw_ratio,
            search_se=wrong_orientation.search_se,
            truth_se=wrong_orientation.truth_se,
            search_panel_id=wrong_orientation.search_panel_id,
            truth_panel_id=wrong_orientation.truth_panel_id,
            orientation="right-minus-left",
        )
        with self.assertRaises(CalibrationError):
            calibrate_policy_cost([wrong_orientation], FitConfig(
                anchors=ANCHORS, folds=2,
            ))

        zero_se = observation(0, 0, 2.0, 1.0, search_se=0.0)
        self.assertEqual(zero_se.validated(ANCHORS).search_se, 0.0)
        for invalid_se in (-1.0, math.inf, math.nan):
            invalid = observation(
                0, 0, 2.0, 1.0, search_se=invalid_se,
            )
            with self.assertRaises(CalibrationError):
                calibrate_policy_cost([invalid], FitConfig(
                    anchors=ANCHORS, folds=2,
                ))


class LockedCampaignDesignTests(unittest.TestCase):
    def test_campaign_cell_rotation_balances_all_fixed_folds(self) -> None:
        rows = campaign_rows(16)
        assignment = make_group_folds(
            rows, folds=5, seed=DEFAULT_FOLD_SEED,
            stratify_campaign_cells=True,
        )
        counts = [sum(value == fold for value in assignment.values())
                  for fold in range(5)]
        self.assertEqual(sum(counts), 3 * 24 * 6 * 2 * 16)
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertNotEqual(counts, [3456, 2592, 2592, 2592, 2592])

    def test_balanced_design_identifiability_and_model_checks_pass(self) -> None:
        self.assertEqual(calibration.CAMPAIGN_CELL_QUOTA, 16)
        self.assertEqual(len(calibration.CAMPAIGN_PLY_BINS), 24)
        self.assertEqual(calibration.CAMPAIGN_PLY_BINS[0], (0, 2))
        self.assertEqual(calibration.CAMPAIGN_PLY_BINS[-2], (44, 48))
        self.assertEqual(calibration.CAMPAIGN_PLY_BINS[-1], (48, 64))
        # Five sources per cell is a test-sized analogue of the locked 16;
        # patching only the quota preserves every cell and one source/fold.
        with mock.patch.object(calibration, "CAMPAIGN_CELL_QUOTA", 5):
            rows = campaign_rows(5)
            result = calibrate_policy_cost(
                rows,
                FitConfig(
                    smoothness_grid=(0.0,),
                    folds=3,
                    require_campaign_design=True,
                ),
                campaign_manifest(rows),
            )
        self.assertTrue(result.campaign_design["validated"])
        self.assertEqual(result.campaign_design["cells"], 864)
        self.assertTrue(
            result.campaign_design["design_identifiability"]["validated"]
        )
        for partition in result.campaign_design[
                "design_identifiability"]["partitions"]:
            self.assertEqual(partition["rank_after_column_scaling"], 30)
            self.assertLessEqual(
                partition["condition_number_after_column_scaling"], 1.0e8
            )
        self.assertTrue(result.model_adequacy["passed"])
        self.assertTrue(result.calibration_passed)
        self.assertIsNotNone(result.schedule)
        serialized = result.to_dict()
        self.assertEqual(serialized["status"], "passed")
        self.assertEqual(
            serialized["deployment"], {"permitted": True, "reason": None}
        )
        self.assertIn("schedule", serialized)
        self.assertIn("derived_gap_thresholds", serialized)
        self.assertTrue(
            result.campaign_design["allocation_binding"]["validated"]
        )
        with mock.patch.object(calibration, "CAMPAIGN_CELL_QUOTA", 5):
            substituted = campaign_manifest(rows)
            substituted["selected_units"][0]["state_id"] += "-unallocated"
            payload = dict(substituted)
            del payload[calibration.CANONICAL_PAYLOAD_SHA256]
            substituted[calibration.CANONICAL_PAYLOAD_SHA256] = hashlib.sha256(
                (json.dumps(
                    payload, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ) + "\n").encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(
                    CalibrationError, "differs from sealed allocation"):
                calibration._campaign_allocation_binding(rows, substituted)
        self.assertGreater(
            result.model_adequacy["gap_over_beta_only_comparison"][
                "one_sided_lcb_z_1_645"
            ],
            0.0,
        )
        self.assertTrue(
            result.model_adequacy[
                "pooled_beta_only_minus_gap_loss_reduction"
            ]["passed"]
        )

    def test_round_specific_model_lack_fails_closed(self) -> None:
        with mock.patch.object(calibration, "CAMPAIGN_CELL_QUOTA", 5):
            result = calibrate_policy_cost(
                campaign_rows(5, round_effect=1.5),
                FitConfig(
                    smoothness_grid=(0.0,),
                    folds=3,
                    require_campaign_design=True,
                ),
            )
        self.assertFalse(result.calibration_passed)
        self.assertIsNone(result.schedule)
        serialized = result.to_dict()
        self.assertEqual(serialized["status"], "failed_model_adequacy")
        self.assertEqual(serialized["deployment"], {
            "permitted": False,
            "reason": "authoritative_predictive_model_adequacy_gate_failed",
        })
        self.assertNotIn("schedule", serialized)
        self.assertNotIn("derived_gap_thresholds", serialized)
        self.assertFalse(serialized["model_adequacy"]["passed"])
        self.assertIn(
            "nested_outer_folds", serialized["model_adequacy"]
        )
        self.assertEqual(
            serialized["observation_input_sha256"],
            result.observation_input_sha256,
        )

    def test_gap_model_must_beat_beta_only_before_select(self) -> None:
        with mock.patch.object(calibration, "CAMPAIGN_CELL_QUOTA", 5):
            result = calibrate_policy_cost(
                campaign_rows(5, gap_scale=0.0),
                FitConfig(
                    smoothness_grid=(0.0,),
                    folds=3,
                    require_campaign_design=True,
                ),
            )
        self.assertFalse(result.calibration_passed)
        serialized = result.to_dict()
        self.assertFalse(serialized["calibration_passed"])
        self.assertFalse(
            serialized["model_adequacy"][
                "gap_over_beta_only_comparison"
            ]["passed"]
        )
        self.assertNotIn("schedule", serialized)

    def test_campaign_beta_design_rank_failure_is_authoritative(self) -> None:
        with mock.patch.object(calibration, "CAMPAIGN_CELL_QUOTA", 5):
            rows = [replace(
                row,
                search_delta=0.0,
            ) for row in campaign_rows(5)]
            with self.assertRaisesRegex(
                    CalibrationError, "unobserved design column"):
                calibrate_policy_cost(
                    rows,
                    FitConfig(
                        smoothness_grid=(0.0,),
                        folds=3,
                        require_campaign_design=True,
                    ),
                )

    def test_pair_metadata_is_canonical_and_strict(self) -> None:
        row = observation(0, 0, 2.0, 1.0)
        with self.assertRaisesRegex(CalibrationError, "round_index"):
            replace(row, round_index=3).validated(ANCHORS)
        with self.assertRaisesRegex(CalibrationError, "same-core"):
            replace(
                row,
                pair_type="same_core_draw",
                log_core_ratio=0.0,
                log_draw_ratio=-math.log(2.0),
            ).validated(ANCHORS)


if __name__ == "__main__":
    unittest.main()
