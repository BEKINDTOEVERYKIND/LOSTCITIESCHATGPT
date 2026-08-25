"""Synthetic regression contracts for standalone policy-cost inference."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tools.policy_cost_selection as selection_module

from tools.policy_cost_selection import (
    CONFIG_BY_ID,
    CONFIG_IDS,
    DIGEST_FIELD,
    InferenceError,
    MIN_SOURCE_MATCH_CLUSTERS,
    PLY_LOS,
    POLICY_FLOORS,
    SELECT_BOOTSTRAP_REPS,
    SELECT_BOOTSTRAP_SEED,
    SELECT_RESULT_SCHEMA,
    TEST_RESULT_SCHEMA,
    canonical_json_bytes,
    canonical_sha256,
    select_configuration,
    seal_result,
    test_selected_configuration,
    verify_result_digest,
    write_canonical_json,
)


BASELINE = "floor-0.02_ply-14"
MOST_AGGRESSIVE = "floor-0.01_ply-00"
FAKE_DISCOVERY_SHA = "0" * 64


def select_rows(
    means: dict[str, float] | None = None,
    *,
    clusters: int = 24,
    units_per_cluster: int = 1,
    noise: float = 0.03,
    duplicate_units: int = 1,
) -> list[dict]:
    means = means or {config_id: 0.0 for config_id in CONFIG_IDS}
    rng = np.random.default_rng(930_401)
    rows: list[dict] = []
    unit_weight = 1.0 / (clusters * units_per_cluster * duplicate_units)
    for cluster in range(clusters):
        for unit in range(units_per_cluster):
            common = float(rng.normal(0.0, noise))
            config_noise = {
                config_id: float(rng.normal(0.0, noise))
                for config_id in CONFIG_IDS
            }
            for duplicate in range(duplicate_units):
                unit_name = f"u{unit:02d}-d{duplicate:02d}"
                identity = f"m{cluster:03d}/{unit_name}".encode("ascii")
                state_sha = hashlib.sha256(b"state/" + identity).hexdigest()
                priority_sha = hashlib.sha256(
                    b"priority/" + identity
                ).hexdigest()
                for config_id in CONFIG_IDS:
                    rows.append(
                        {
                            "source_match": f"m{cluster:03d}",
                            "unit": unit_name,
                            "state_sha256": state_sha,
                            "allocation_priority_sha256": priority_sha,
                            "config": config_id,
                            "round": 0,
                            "ply_stratum": 0,
                            "frontier_present": False,
                            "allocation_slot": 0,
                            "master_width": 1,
                            "post_stratum": "r0:p00:f0:j0",
                            "discovery_census_sha256": FAKE_DISCOVERY_SHA,
                            "hybrid_gain": (
                                means.get(config_id, 0.0)
                                + common + config_noise[config_id]
                            ),
                            "weight": unit_weight,
                            "exact_valid": True,
                            "capped": 0,
                        }
                    )
    return rows


def monotone_means(step: float = 0.5) -> dict[str, float]:
    result: dict[str, float] = {}
    for config_id, config in CONFIG_BY_ID.items():
        ply_steps = PLY_LOS.index(int(config["ply_lo"]))
        floor_step = int(float(config["policy_floor"]) == 0.01)
        result[config_id] = step * (ply_steps + floor_step)
    return result


def baseline_selection() -> dict:
    return select_configuration(select_rows(noise=0.0))


def test_rows(
    *,
    hybrid_by_round: tuple[float, float, float] = (0.4, 0.4, 0.4),
    score_by_round: tuple[float, float, float] = (0.08, 0.08, 0.08),
    clusters: int = 24,
    selected: str = BASELINE,
    frontier: bool = True,
    exact_valid: bool = True,
    capped: int = 0,
    noise: float = 0.01,
) -> list[dict]:
    rng = np.random.default_rng(930_402)
    rows: list[dict] = []
    for cluster in range(clusters):
        round_index = cluster % 3
        identity = f"tm{cluster:03d}/tu{cluster:03d}".encode("ascii")
        rows.append(
            {
                "source_match": f"tm{cluster:03d}",
                "unit": f"tu{cluster:03d}",
                "state_sha256": hashlib.sha256(
                    b"state/" + identity
                ).hexdigest(),
                "allocation_priority_sha256": hashlib.sha256(
                    b"priority/" + identity
                ).hexdigest(),
                "config": selected,
                "post_stratum": f"r{round_index}:p00:f{int(frontier)}:j0",
                "round": round_index,
                "ply_stratum": 0,
                "allocation_slot": 0,
                "master_width": 1,
                "discovery_census_sha256": FAKE_DISCOVERY_SHA,
                "weight": 1.0 / clusters,
                "hybrid_gain": (
                    hybrid_by_round[round_index]
                    + float(rng.normal(0.0, noise))
                ),
                "match_score_gain": (
                    score_by_round[round_index]
                    + float(rng.normal(0.0, noise / 10.0))
                ),
                "frontier_present": frontier,
                "exact_valid": exact_valid,
                "capped": capped,
            }
        )
    return rows


WEIGHTS = {
    "r0:p00:f1:j0": 0.6,
    "r1:p00:f1:j0": 0.2,
    "r2:p00:f1:j0": 0.2,
}


class FrozenConfigurationTests(unittest.TestCase):
    def test_exact_twelve_configs(self) -> None:
        self.assertEqual(len(CONFIG_IDS), 12)
        self.assertEqual(
            {(row["policy_floor"], row["ply_lo"])
             for row in CONFIG_BY_ID.values()},
            {(floor, ply) for floor in POLICY_FLOORS for ply in PLY_LOS},
        )
        self.assertEqual(CONFIG_IDS[0], BASELINE)
        self.assertEqual(SELECT_BOOTSTRAP_REPS, 20_000)
        self.assertEqual(SELECT_BOOTSTRAP_SEED, 202611150101)


class SelectionTests(unittest.TestCase):
    def test_large_incremental_gains_select_most_aggressive(self) -> None:
        result = select_configuration(select_rows(monotone_means()))
        self.assertTrue(verify_result_digest(result))
        self.assertEqual(result["schema"], SELECT_RESULT_SCHEMA)
        self.assertEqual(result["stage"], "SELECT")
        self.assertEqual(result["selected"]["id"], MOST_AGGRESSIVE)
        self.assertEqual(len(result["eligible_config_ids"]), 12)
        self.assertEqual(
            result["simultaneous_inference"][
                "protected_directed_pairwise_contrasts"
            ],
            132,
        )
        self.assertGreater(
            result["simultaneous_inference"]["critical_value"], 1.645
        )
        self.assertEqual(
            result["runtime_dependencies"]["numpy_version"], np.__version__
        )

    def test_no_incremental_evidence_retains_conservative_baseline(self) -> None:
        result = baseline_selection()
        self.assertEqual(result["selected"]["id"], BASELINE)
        self.assertEqual(result["eligible_config_ids"], [BASELINE])
        self.assertEqual(
            result["statistically_tied_config_ids"], [BASELINE]
        )

    def test_performance_tie_prefers_later_ply(self) -> None:
        means = {config_id: -2.0 for config_id in CONFIG_IDS}
        means[BASELINE] = 0.0
        means["floor-0.02_ply-12"] = 1.0
        means["floor-0.01_ply-14"] = 1.0
        # The two incomparable one-step expansions are both eligible and tied;
        # later onset is the first conservative tie-break.
        result = select_configuration(select_rows(means, noise=0.01))
        self.assertIn("floor-0.02_ply-12", result["eligible_config_ids"])
        self.assertIn("floor-0.01_ply-14", result["eligible_config_ids"])
        self.assertEqual(result["selected"]["id"], "floor-0.01_ply-14")

    def test_units_within_source_match_do_not_fake_precision(self) -> None:
        means = monotone_means(0.08)
        single = select_configuration(
            select_rows(means, clusters=16, duplicate_units=1)
        )
        duplicated = select_configuration(
            select_rows(means, clusters=16, duplicate_units=20)
        )
        self.assertEqual(single["selected"], duplicated["selected"])
        self.assertAlmostEqual(
            single["simultaneous_inference"]["critical_value"],
            duplicated["simultaneous_inference"]["critical_value"],
            places=12,
        )
        first = single["simultaneous_inference"]["pair_intervals"][0]
        repeated = duplicated["simultaneous_inference"]["pair_intervals"][0]
        self.assertAlmostEqual(
            first["source_match_cluster_se"],
            repeated["source_match_cluster_se"],
            places=12,
        )

    def test_fixed_post_strata_are_centered_before_cluster_variance(self) -> None:
        rows = select_rows(clusters=24, noise=0.0)
        for row in rows:
            cluster = int(str(row["source_match"])[1:])
            stratum = cluster % 3
            row["round"] = stratum
            row["post_stratum"] = f"r{stratum}:p00:f0:j0"
            config_index = CONFIG_IDS.index(str(row["config"]))
            row["hybrid_gain"] = float(stratum * config_index)
        result = select_configuration(rows)
        # There is no within-post-stratum sampling variation in any contrast.
        # Fixed allocation across strata must therefore not masquerade as
        # source uncertainty.
        self.assertTrue(all(
            interval["source_match_cluster_se"] < 1.0e-14
            for interval in result["simultaneous_inference"][
                "pair_intervals"
            ]
        ))

    def test_selection_fails_closed_on_incomplete_or_invalid_units(self) -> None:
        incomplete = select_rows(clusters=MIN_SOURCE_MATCH_CLUSTERS)
        incomplete.pop()
        with self.assertRaisesRegex(InferenceError, "coverage differs"):
            select_configuration(incomplete)

        invalid = select_rows(clusters=MIN_SOURCE_MATCH_CLUSTERS)
        invalid[0]["exact_valid"] = False
        with self.assertRaisesRegex(InferenceError, "not exactly valid"):
            select_configuration(invalid)

        capped = select_rows(clusters=MIN_SOURCE_MATCH_CLUSTERS)
        capped[0]["capped"] = 1
        with self.assertRaisesRegex(InferenceError, "capped"):
            select_configuration(capped)

        bad_weights = select_rows(clusters=MIN_SOURCE_MATCH_CLUSTERS)
        for row in bad_weights:
            row["weight"] *= 0.5
        with self.assertRaisesRegex(InferenceError, "weights must sum to one"):
            select_configuration(bad_weights)


class TestGateTests(unittest.TestCase):
    def test_positive_weighted_test_passes(self) -> None:
        selection = baseline_selection()
        result = test_selected_configuration(test_rows(), WEIGHTS, selection)
        self.assertTrue(verify_result_digest(result))
        self.assertEqual(result["schema"], TEST_RESULT_SCHEMA)
        self.assertTrue(result["passed"])
        self.assertFalse(result["selection_performed_in_test"])
        self.assertGreater(result["hybrid_gain"]["lcb_z_1_645"], 0.0)
        self.assertGreater(
            result["match_score_gain"]["lcb_z_1_645"], 0.0
        )
        self.assertGreaterEqual(result["frontier_hybrid_gain"]["point"], 0.0)

    def test_discovery_weights_and_round_guard_are_effective(self) -> None:
        selection = baseline_selection()
        rows = test_rows(
            hybrid_by_round=(1.0, -1.0, 1.0),
            score_by_round=(0.2, 0.2, 0.2),
            noise=0.0,
        )
        weights = {
            "r0:p00:f1:j0": 0.8,
            "r1:p00:f1:j0": 0.1,
            "r2:p00:f1:j0": 0.1,
        }
        result = test_selected_configuration(rows, weights, selection)
        self.assertAlmostEqual(result["hybrid_gain"]["point"], 0.8)
        self.assertAlmostEqual(result["round_hybrid_points"]["1"], -1.0)
        self.assertFalse(
            result["criteria"]["each_round_hybrid_point_nonnegative"]
        )
        self.assertFalse(result["passed"])

    def test_zero_lcb_boundary_fails_but_point_guard_is_inclusive(self) -> None:
        selection = baseline_selection()
        rows = test_rows(
            hybrid_by_round=(0.0, 0.0, 0.0),
            score_by_round=(0.0, 0.0, 0.0),
            noise=0.0,
        )
        result = test_selected_configuration(rows, WEIGHTS, selection)
        self.assertEqual(result["hybrid_gain"]["lcb_z_1_645"], 0.0)
        self.assertEqual(result["frontier_hybrid_gain"]["point"], 0.0)
        self.assertFalse(
            result["criteria"]["hybrid_gain_lcb_strictly_positive"]
        )
        self.assertTrue(
            result["criteria"]["frontier_hybrid_point_nonnegative"]
        )
        self.assertFalse(result["passed"])

    def test_invalidity_and_caps_are_authoritative_failures(self) -> None:
        selection = baseline_selection()
        invalid = test_rows(exact_valid=False)
        invalid_result = test_selected_configuration(invalid, WEIGHTS, selection)
        self.assertFalse(invalid_result["criteria"]["exact_validity"])
        self.assertFalse(invalid_result["passed"])

        capped = test_rows(capped=1)
        capped_result = test_selected_configuration(capped, WEIGHTS, selection)
        self.assertEqual(
            capped_result["raw_validity"]["capped_continuations"], len(capped)
        )
        self.assertFalse(capped_result["criteria"]["zero_caps"])
        self.assertFalse(capped_result["passed"])

    def test_test_rejects_other_actor_bad_weights_and_empty_frontier(self) -> None:
        selection = baseline_selection()
        wrong_actor = test_rows()
        wrong_actor[0]["config"] = "floor-0.01_ply-00"
        with self.assertRaisesRegex(InferenceError, "unselected actor"):
            test_selected_configuration(wrong_actor, WEIGHTS, selection)

        with self.assertRaisesRegex(InferenceError, "sum to one"):
            test_selected_configuration(
                test_rows(), {"r0:p00:f1:j0": 0.6,
                              "r1:p00:f1:j0": 0.2,
                              "r2:p00:f1:j0": 0.1}, selection
            )

        with self.assertRaisesRegex(InferenceError, "frontier"):
            test_selected_configuration(
                test_rows(frontier=False),
                {
                    "r0:p00:f0:j0": 0.6,
                    "r1:p00:f0:j0": 0.2,
                    "r2:p00:f0:j0": 0.2,
                },
                selection,
            )


class CanonicalEvidenceTests(unittest.TestCase):
    def test_select_cli_validates_campaign_evidence_and_writes(self) -> None:
        evidence = {
            "raw_verified": True,
            "stage": "SELECT",
            "execution_sha256": "1" * 64,
            "evaluation_sha256": "2" * 64,
            "evaluation_header_sha256": "3" * 64,
            "allocation_sha256": "4" * 64,
            "calibration_sha256": "5" * 64,
            "policy_cost_sha256": "6" * 64,
            "policy_cost_content_fingerprint": "7" * 16,
            "selection_sha256": None,
            "actor_manifest_sha256": None,
        }
        result = selection_module.seal_result({
            "schema": SELECT_RESULT_SCHEMA,
            "stage": "SELECT",
            "campaign_discovery_binding": {
                "required": True, "validated": True,
            },
        })
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "select-input.json"
            output = Path(directory) / "select-result.json"
            source.write_bytes(canonical_json_bytes({
                "schema": selection_module.SELECT_INPUT_SCHEMA,
                "discovery_manifest": {},
                "rows": [],
                "campaign_evidence_binding": evidence,
            }))
            with mock.patch.object(
                    selection_module, "select_configuration",
                    return_value=result) as selected:
                self.assertEqual(selection_module.main([
                    "select", "--input", str(source), "--output", str(output),
                ]), 0)
            self.assertEqual(output.read_bytes(), canonical_json_bytes(result))
            self.assertEqual(selected.call_args.args[2], evidence)

    def test_digest_detects_mutation_and_writer_is_canonical(self) -> None:
        result = baseline_selection()
        self.assertEqual(result[DIGEST_FIELD], canonical_sha256({
            key: value for key, value in result.items() if key != DIGEST_FIELD
        }))
        changed = copy.deepcopy(result)
        changed["selected"]["ply_lo"] = 0
        self.assertFalse(verify_result_digest(changed))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            write_canonical_json(path, result)
            self.assertEqual(path.read_bytes(), canonical_json_bytes(result))
            self.assertEqual(
                [item.name for item in Path(directory).iterdir()],
                ["selection.json"],
            )
            parsed = json.loads(path.read_text())
            self.assertTrue(verify_result_digest(parsed))
            with self.assertRaisesRegex(InferenceError, "refusing to replace"):
                write_canonical_json(path, result)
            self.assertEqual(path.read_bytes(), canonical_json_bytes(result))
            self.assertEqual(
                [item.name for item in Path(directory).iterdir()],
                ["selection.json"],
            )

    def test_test_binds_exact_selection_digest(self) -> None:
        selection = baseline_selection()
        result = test_selected_configuration(test_rows(), WEIGHTS, selection)
        self.assertEqual(
            result["selection_payload_sha256"], selection[DIGEST_FIELD]
        )
        stale = copy.deepcopy(selection)
        stale["point_best_config_id"] = MOST_AGGRESSIVE
        with self.assertRaisesRegex(InferenceError, "digest"):
            test_selected_configuration(test_rows(), WEIGHTS, stale)


class CampaignDiscoveryBindingTests(unittest.TestCase):
    @staticmethod
    def manifest(stage: str) -> dict:
        cells = []
        selected_units = []
        aggregate = {str(width): 0 for width in range(1, 6)}
        for round_index in range(3):
            for frontier in (False, True):
                for slot in range(2):
                    census = 8 if slot == 0 else 0
                    quota = census
                    histogram = {
                        "1": census, "2": 0, "3": 0, "4": 0, "5": 0,
                    }
                    aggregate["1"] += census
                    cells.append({
                        "round": round_index,
                        "ply_stratum": 0,
                        "frontier_present": frontier,
                        "allocation_slot": slot,
                        "post_stratum": (
                            f"r{round_index}:p00:f{int(frontier)}:j{slot}"
                        ),
                        "census_count": census,
                        "allocation_quota": quota,
                        "master_width_histogram": histogram,
                    })
                    if slot == 0:
                        post_stratum = (
                            f"r{round_index}:p00:f{int(frontier)}:j0"
                        )
                        for member in range(8):
                            source = (
                                f"{stage.lower()}-r{round_index}-"
                                f"f{int(frontier)}-m{member}"
                            )
                            unit_id = f"unit-{member}"
                            identity = f"{source}/{unit_id}".encode("ascii")
                            selected_units.append({
                                "source_match": source,
                                "unit": unit_id,
                                "state_sha256": hashlib.sha256(
                                    b"state/" + identity
                                ).hexdigest(),
                                "allocation_priority_sha256": hashlib.sha256(
                                    b"priority/" + identity
                                ).hexdigest(),
                                "round": round_index,
                                "ply_stratum": 0,
                                "frontier_present": frontier,
                                "allocation_slot": 0,
                                "master_width": 1,
                                "post_stratum": post_stratum,
                            })
        selected_units.sort(key=lambda row: (
            row["post_stratum"], row["allocation_priority_sha256"],
            row["state_sha256"], row["source_match"], row["unit"],
        ))
        return seal_result({
            "schema": selection_module.DISCOVERY_MANIFEST_SCHEMA,
            "stage": stage,
            "ply_boundaries": [0, 2],
            "base_vector_quota": 8,
            "source_reservoir_sha256": "a" * 64,
            "source_net_sha256": "b" * 64,
            "source_exclusion_sha256": "c" * 64,
            "eligible_state_commitment_sha256": "b" * 64,
            "allocation_rule_sha256": "c" * 64,
            "total_eligible_states": 48,
            "master_width_histogram": aggregate,
            "cells": cells,
            "selected_units": selected_units,
        })

    @staticmethod
    def rows(stage: str, manifest: dict, selected: str = BASELINE) -> list[dict]:
        rows = []
        digest = manifest[DIGEST_FIELD]
        for round_index in range(3):
            for frontier in (False, True):
                post_stratum = (
                    f"r{round_index}:p00:f{int(frontier)}:j0"
                )
                for member in range(8):
                    source = (
                        f"{stage.lower()}-r{round_index}-"
                        f"f{int(frontier)}-m{member}"
                    )
                    unit_id = f"unit-{member}"
                    identity = f"{source}/{unit_id}".encode("ascii")
                    common = {
                        "source_match": source,
                        "unit": unit_id,
                        "state_sha256": hashlib.sha256(
                            b"state/" + identity
                        ).hexdigest(),
                        "allocation_priority_sha256": hashlib.sha256(
                            b"priority/" + identity
                        ).hexdigest(),
                        "round": round_index,
                        "ply_stratum": 0,
                        "frontier_present": frontier,
                        "allocation_slot": 0,
                        "master_width": 1,
                        "post_stratum": post_stratum,
                        "discovery_census_sha256": digest,
                        "weight": 1.0 / 48.0,
                        "exact_valid": True,
                        "capped": 0,
                    }
                    if stage == "SELECT":
                        for config_id in CONFIG_IDS:
                            rows.append({
                                **common,
                                "config": config_id,
                                "hybrid_gain": 0.0,
                            })
                    else:
                        rows.append({
                            **common,
                            "config": selected,
                            "hybrid_gain": 1.0,
                            "match_score_gain": 0.1,
                        })
        return rows

    def test_campaign_cli_inference_binds_exact_census_and_quota(self) -> None:
        with (
            mock.patch.object(selection_module, "PLY_BOUNDARIES", (0, 2)),
            mock.patch.object(selection_module, "BASE_VECTOR_QUOTA", 8),
            mock.patch.object(selection_module, "MAX_ALLOCATION_SLOTS", 2),
        ):
            select_manifest = self.manifest("SELECT")
            selection = select_configuration(
                self.rows("SELECT", select_manifest), select_manifest
            )
            self.assertTrue(
                selection["campaign_discovery_binding"]["validated"]
            )
            test_manifest = self.manifest("TEST")
            test_input_rows = self.rows(
                "TEST", test_manifest, selection["selected"]["id"]
            )
            weights = {
                f"r{round_index}:p00:f{int(frontier)}:j0": 1.0 / 6.0
                for round_index in range(3)
                for frontier in (False, True)
            }
            result = test_selected_configuration(
                test_input_rows, weights, selection, test_manifest
            )
            self.assertTrue(result["campaign_discovery_binding"]["validated"])
            self.assertTrue(result["passed"])

            substituted = copy.deepcopy(test_input_rows)
            substituted[0]["source_match"] += "-unallocated"
            substituted[0]["state_sha256"] = "1" * 64
            substituted[0]["allocation_priority_sha256"] = "2" * 64
            with self.assertRaisesRegex(
                    InferenceError, "not in the sealed allocation"):
                test_selected_configuration(
                    substituted, weights, selection, test_manifest
                )

            corrupted = copy.deepcopy(test_manifest)
            corrupted["cells"][0]["allocation_quota"] = 7
            corrupted = seal_result({
                key: value for key, value in corrupted.items()
                if key != DIGEST_FIELD
            })
            with self.assertRaisesRegex(InferenceError, "quota"):
                test_selected_configuration(
                    test_input_rows, weights, selection, corrupted
                )


if __name__ == "__main__":
    unittest.main()
