"""Focused fail-closed contracts for the belief-history accuracy reducer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import belief_history_reduce as reducer


NATIVE_STRUCTURAL_SMOKE_ROOT = 202_706_290_103


def metric(nll: float, brier: float, *, states: int, cards: int,
           positives: int) -> dict:
    return {
        "brier_sum": brier,
        "nll_sum": nll,
        "positive_count": positives,
        "state_count": states,
        "top_hits_sum": positives / 2,
        "uncertain_card_count": cards,
    }


def row(source_match_id: int, *, stage_root: int = reducer.FROZEN_TEST_ROOT) -> dict:
    all_states = {
        "base_262k_head": metric(
            10.5, 3.4, states=10, cards=40, positives=16),
        "matched_head_control": metric(
            10.0, 3.0, states=10, cards=40, positives=16),
        "incumbent_head": metric(
            11.0, 3.8, states=10, cards=40, positives=16),
        "history": metric(9.0, 2.0, states=10, cards=40, positives=16),
        "uniform_exact_k": metric(11.4, 3.6, states=10, cards=40,
                                  positives=16),
    }
    post = {
        "base_262k_head": metric(
            8.4, 2.7, states=8, cards=32, positives=12),
        "matched_head_control": metric(
            8.0, 2.4, states=8, cards=32, positives=12),
        "incumbent_head": metric(
            8.8, 3.0, states=8, cards=32, positives=12),
        "history": metric(7.2, 1.6, states=8, cards=32, positives=12),
        "uniform_exact_k": metric(9.6, 3.2, states=8, cards=32,
                                  positives=12),
    }
    return {
        "actor_fingerprint": "0123456789abcdef",
        "base_alpha": 1.0,
        "base_net_fingerprint": "1111111111111111",
        "matched_base_alpha": 1.0,
        "incumbent_alpha": 1.15,
        "incumbent_net_fingerprint": "0123456789abcdef",
        "matched_base_net_fingerprint": "2222222222222222",
        "capped_rounds": 0,
        "excluded_state_count": 0,
        "exclusion_manifest_count": 17,
        "exclusion_manifest_sha256": reducer.EXACT17_TEXT_SHA256,
        "history_model_fingerprint": "fedcba9876543210",
        "max_scored_ply": 300,
        "metrics": {
            "all_states": all_states,
            "post_opponent_action": post,
        },
        "rounds_completed": 3,
        "reviewed_ply_inputs_used": False,
        "schema": reducer.ROW_SCHEMA,
        "seed_root": stage_root,
        "source_match_id": source_match_id,
        "structural_contract": {
            "action_history_public_only": True,
            "current_view_truth_scrubbed": True,
            "opening_history_uniform": True,
            "playing_actor_changed": False,
            "public_transcript_complete": True,
            "residual_features_opponent_action_anchored": True,
            "reviewed_ply_orbit_exclusion_enabled": True,
            "suit_equivariant_features": True,
            "truth_read_after_prediction": True,
            "wager_identity_collapsed": True,
        },
        "symmetries": 20,
        "temperature": 0.03,
    }


def plan(*, matches: int = reducer.FROZEN_TEST_MATCHES,
         shards: int = reducer.FROZEN_TEST_SHARDS,
         root: int = reducer.FROZEN_TEST_ROOT) -> dict:
    splits = {"TEST": {"matches": matches, "shards": shards,
                       "root": str(root)},
              "TRAIN": {
                  "base_control_matches": 262144,
                  "base_control_root": "202706110102",
                  "base_control_shards": 4,
                  "history_matches": 65536,
                  "history_root": "202706100101",
                  "history_shards": 1,
                  "matched_control_additional_matches": 65536,
                  "matched_control_root": "202706100101",
                  "matched_control_shards": 1,
              }}
    stages = {
        "TEST": {
            "matches": matches,
            "one_frozen_candidate": True,
            "one_look": True,
            "second_test_or_top_up": False,
            "test_gate_applies": True,
        },
        "TRAIN": {
            "base_control_matches": 262144,
            "history_matches": 65536,
            "matched_control_additional_matches": 65536,
            "test_gate_applies": False,
        },
    }
    return {
        "artifact_kind": reducer.PLAN_ARTIFACT_KIND,
        "artifact_schemas": {
            "evaluation_identity": reducer.IDENTITY_SCHEMA,
            "raw_match_metrics": reducer.ROW_SCHEMA,
            "result": reducer.RESULT_SCHEMA,
        },
        "experiment": "belief-history-v1",
        "schema": reducer.PLAN_SCHEMA,
        "data": {
            "splits": splits,
            "source_actor": {
                "actor_net": {"sha256": reducer.FROZEN_ACTOR_SHA256},
                "temperature": 0.03,
                "trajectory_symmetries": 20,
            },
        },
        "evaluation": {
            "bootstrap": {
                "method": reducer.FROZEN_BOOTSTRAP_METHOD,
                "replicates": 20000,
                "seed": str(reducer.FROZEN_BOOTSTRAP_SEED),
                "simultaneous_familywise": {
                    "components": 9,
                    "confidence": 0.99,
                    "coverage_claim": reducer.FROZEN_COVERAGE_CLAIM,
                    "exact_finite_sample_coverage_claimed": False,
                    "method": reducer.FROZEN_SIMULTANEOUS_METHOD,
                    "studentization":
                        reducer.FROZEN_SIMULTANEOUS_STUDENTIZATION,
                    "zero_standard_error_policy":
                        reducer.FROZEN_ZERO_SE_POLICY,
                },
                "unit": "source_match",
            },
            "primary_gate": {
                "relative_nll_improvement_at_least": 0.0,
                "nll_lcb_strictly_above": 0.0,
                "point_gain_strictly_above": 0.0,
                "confidence": 0.99,
            },
            "history_gate": {
                "relative_nll_improvement_at_least": 0.0,
                "nll_lcb_strictly_above": 0.0,
                "point_gain_strictly_above": 0.0,
                "confidence": 0.99,
                "min_opponent_actions": 1,
            },
            "brier_gate": {
                "lcb_strictly_above": 0.0,
                "point_gain_strictly_above": 0.0,
                "confidence": 0.99,
            },
            "stages": stages,
            "terminal_artifact_selection": {
                "comparison_bundle": reducer.FROZEN_COMPARISON_BUNDLE,
                "rule": [
                    "retain history only if history passes directly against both matched_head_control and incumbent_head",
                    "otherwise retain matched_head_control only if it passes directly against incumbent_head",
                    "otherwise retain incumbent_head",
                ],
                "selected_artifact_is_playing_actor": False,
            },
        },
        "models": {
            "candidate": {"base_alpha": 1.0},
            "matched_head_only_control": {
                "base_alpha": 1.0,
                "primary_test_comparator": True,
            },
            "incumbent_head": {
                "alpha": 1.15,
                "replacement_fallback": True,
                "sha256": reducer.FROZEN_ACTOR_SHA256,
            },
        },
    }


def identity(stage: str, *, matches: int = reducer.FROZEN_TEST_MATCHES,
             root: int = reducer.FROZEN_TEST_ROOT) -> dict:
    bindings = {
        "actor_sha256": reducer.FROZEN_ACTOR_SHA256,
        "base_262k_head_sha256": "1" * 64,
        "exact17_exclusions_sha256": reducer.EXACT17_TEXT_SHA256,
        "execution_sha256": "d" * 64,
        "history_model_sha256": "e" * 64,
        "matched_head_control_sha256": "2" * 64,
        "native_structural_test_sha256": "f" * 64,
        "shared_train_source_manifest_sha256": "3" * 64,
        "test_generator_manifest_sha256": "0" * 64,
        "transport_sha256": "0" * 64,
    }
    row_contract = {
        "actor_fingerprint": "0123456789abcdef",
        "base_alpha": 1.0,
        "base_net_fingerprint": "1111111111111111",
        "history_model_fingerprint": "fedcba9876543210",
        "incumbent_alpha": 1.15,
        "incumbent_net_fingerprint": "0123456789abcdef",
        "matched_base_alpha": 1.0,
        "matched_base_net_fingerprint": "2222222222222222",
        "max_scored_ply": 300,
        "seed_root": root,
        "source_match_count": matches,
        "source_match_start": 0,
        "symmetries": 20,
        "temperature": 0.03,
    }
    bindings["test_generator_manifest_sha256"] = reducer.canonical_sha256(
        reducer.frozen_test_generator_manifest({
            "bindings": bindings,
            "row_contract": row_contract,
        }))
    return reducer.seal({
        "schema": reducer.IDENTITY_SCHEMA,
        "campaign_id": "belief-history-v1",
        "stage": stage,
        "bindings": bindings,
        "row_contract": row_contract,
    })


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.stage = "TEST"
        self.plan = plan(matches=8, shards=2)
        self.identity = identity(self.stage, matches=8)
        self.rows = [row(i) for i in range(8)]
        # Use realistic between-match heterogeneity so the cluster standard
        # errors are nonzero. The symmetric factors preserve every aggregate
        # fixture expectation and the opening-history uniform contract.
        for index, value in enumerate(self.rows):
            factor = 1.0 + (index - 3.5) / 100.0
            for group in reducer.GROUP_KEY_ORDER:
                for model in reducer.MODEL_KEY_ORDER:
                    for name in ("nll_sum", "brier_sum"):
                        value["metrics"][group][model][name] *= factor

    def write(self) -> tuple[dict, str, dict, str, list[Path]]:
        plan_path = self.root / "plan.json"
        identity_path = self.root / "identity.json"
        plan_path.write_text(json.dumps(self.plan), encoding="ascii")
        identity_path.write_bytes(reducer.canonical_bytes(self.identity))
        paths: list[Path] = []
        for shard in range(2):
            path = self.root / f"shard-{shard}.jsonl"
            path.write_bytes(b"".join(
                (json.dumps(value, separators=(",", ":"), ensure_ascii=True)
                 + "\n").encode("ascii")
                for value in self.rows[shard * 4:(shard + 1) * 4]
            ))
            paths.append(path)
        loaded_identity, identity_sha = reducer.load_identity(
            identity_path, self.stage)
        return (self.plan, reducer.canonical_sha256(self.plan),
                loaded_identity, identity_sha, paths)

    def reduce(self) -> dict:
        p, p_sha, i, i_sha, paths = self.write()
        with mock.patch.object(reducer, "FROZEN_TEST_MATCHES", 8), \
                mock.patch.object(reducer, "FROZEN_TEST_SHARDS", 2):
            return reducer.reduce_evidence(p, p_sha, i, i_sha,
                                           self.stage, paths)


class BeliefHistoryReducerTests(unittest.TestCase):
    def test_compiled_native_writer_row_is_accepted_directly(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tool = root / "bin" / "history_belief_train"
        actor = root / "data" / "champion.bin"
        exclusions = root / "data" / "experiments" / \
            "policy_cost_v7_exact17_exclusions.txt"
        with tempfile.TemporaryDirectory(prefix="lc-belief-native-row-") as tmp:
            temporary = Path(tmp)
            model = temporary / "history.bin"
            common = [
                "--actor-net", str(actor), "--base-net", str(actor),
                "--matches", "1", "--rounds", "3",
                "--max-ply", "300", "--symmetries", "20",
                "--temperature", "0.03", "--base-alpha", "1.0",
                "--exclusions", str(exclusions),
                "--exclusions-sha256", reducer.EXACT17_TEXT_SHA256,
            ]
            trained = subprocess.run(
                [str(tool), "train", "--out", str(model),
                 "--seed", str(NATIVE_STRUCTURAL_SMOKE_ROOT),
                 "--match-start", "0",
                 "--epochs", "1", "--lr", "0.0001", "--l2", "0.0000001"]
                + common,
                cwd=root, text=True, capture_output=True, check=False)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            match_jsonl = temporary / "native.jsonl"
            evaluated = subprocess.run(
                [str(tool), "eval", "--model", str(model),
                 "--matched-base-net", str(actor),
                 "--matched-base-alpha", "1.0",
                 "--incumbent-alpha", "1.15",
                 "--seed", str(NATIVE_STRUCTURAL_SMOKE_ROOT),
                 "--match-start", "0", "--match-jsonl", str(match_jsonl)]
                + common,
                cwd=root, text=True, capture_output=True, check=False)
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            raw = match_jsonl.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\r", raw)
        self.assertNotIn(b" ", raw)
        native = reducer._loads(raw, "compiled native row")
        row_identity = {
            "bindings": {
                "exact17_exclusions_sha256": reducer.EXACT17_TEXT_SHA256,
            },
            "row_contract": {
                "actor_fingerprint": native["actor_fingerprint"],
                "base_alpha": reducer._binary32(1.0),
                "base_net_fingerprint": native["base_net_fingerprint"],
                "history_model_fingerprint":
                    native["history_model_fingerprint"],
                "incumbent_alpha": reducer._binary32(1.15),
                "incumbent_net_fingerprint":
                    native["incumbent_net_fingerprint"],
                "matched_base_alpha": reducer._binary32(1.0),
                "matched_base_net_fingerprint":
                    native["matched_base_net_fingerprint"],
                "max_scored_ply": 300,
                "seed_root": NATIVE_STRUCTURAL_SMOKE_ROOT,
                "source_match_count": 1,
                "source_match_start": 0,
                "symmetries": 20,
                "temperature": reducer._binary32(0.03),
            },
        }
        checked = reducer._validate_row(
            native, "compiled native row", row_identity)
        self.assertEqual(checked["source_match_id"], 0)

    def test_repository_plan_and_all_frozen_test_constants_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-plan-") as tmp:
            identity_path = Path(tmp) / "identity.json"
            identity_path.write_bytes(reducer.canonical_bytes(identity("TEST")))
            loaded, _ = reducer.load_identity(identity_path, "TEST")
            repository_plan = json.loads(
                (Path(__file__).resolve().parents[1] / "data" / "experiments" /
                 "locked_belief_history_v1_plan.json").read_text(
                     encoding="ascii"))
            expected = reducer._validate_plan(repository_plan, loaded, "TEST")
        self.assertEqual(expected["matches"], 65_536)
        self.assertEqual(expected["shards"], 16)
        self.assertEqual(expected["bootstrap_seed"], 202_706_150_101)

        mutations = {
            "schema": lambda value: value.__setitem__("schema", "v2"),
            "result schema": lambda value: value["artifact_schemas"]
                .__setitem__("result", "other"),
            "matches": lambda value: value["data"]["splits"]["TEST"]
                .__setitem__("matches", 32768),
            "shards": lambda value: value["data"]["splits"]["TEST"]
                .__setitem__("shards", 8),
            "root": lambda value: value["data"]["splits"]["TEST"]
                .__setitem__("root", "202706100402"),
            "bootstrap seed": lambda value: value["evaluation"]["bootstrap"]
                .__setitem__("seed", "202706150102"),
            "bootstrap count": lambda value: value["evaluation"]["bootstrap"]
                .__setitem__("replicates", 19999),
            "primary threshold": lambda value: value["evaluation"]
                ["primary_gate"].__setitem__(
                    "point_gain_strictly_above", 0.01),
            "simultaneous family size": lambda value: value["evaluation"]
                ["bootstrap"]["simultaneous_familywise"].__setitem__(
                    "components", 8),
            "simultaneous method": lambda value: value["evaluation"]
                ["bootstrap"]["simultaneous_familywise"].__setitem__(
                    "method", "other"),
            "simultaneous studentization": lambda value: value["evaluation"]
                ["bootstrap"]["simultaneous_familywise"].__setitem__(
                    "studentization", "other"),
            "simultaneous coverage": lambda value: value["evaluation"]
                ["bootstrap"]["simultaneous_familywise"].__setitem__(
                    "coverage_claim", "exact"),
            "finite-sample coverage": lambda value: value["evaluation"]
                ["bootstrap"]["simultaneous_familywise"].__setitem__(
                    "exact_finite_sample_coverage_claimed", True),
            "zero-SE policy": lambda value: value["evaluation"]
                ["bootstrap"]["simultaneous_familywise"].__setitem__(
                    "zero_standard_error_policy", "pass"),
            "history definition": lambda value: value["evaluation"]
                ["history_gate"].__setitem__("min_opponent_actions", 2),
            "confidence": lambda value: value["evaluation"]["brier_gate"]
                .__setitem__("confidence", 0.95),
            "actor": lambda value: value["data"]["source_actor"]["actor_net"]
                .__setitem__("sha256", "0" * 64),
            "temperature": lambda value: value["data"]["source_actor"]
                .__setitem__("temperature", 0.04),
            "symmetries": lambda value: value["data"]["source_actor"]
                .__setitem__("trajectory_symmetries", 10),
            "candidate alpha": lambda value: value["models"]["candidate"]
                .__setitem__("base_alpha", 1.1),
            "matched training budget": lambda value: value["data"]["splits"]
                ["TRAIN"].__setitem__("matched_control_additional_matches", 1),
            "extra stage": lambda value: value["evaluation"]["stages"]
                .__setitem__("SELECT", {"matches": 1}),
            "comparison bundle": lambda value: value["evaluation"]
                ["terminal_artifact_selection"].__setitem__(
                    "comparison_bundle", "same gates, different contract"),
            "missing comparison bundle": lambda value: value["evaluation"]
                ["terminal_artifact_selection"].pop("comparison_bundle"),
            "extra selection field": lambda value: value["evaluation"]
                ["terminal_artifact_selection"].__setitem__(
                    "unfrozen_override", False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(repository_plan)
                mutate(changed)
                with self.assertRaises(reducer.ReductionError):
                    reducer._validate_plan(changed, loaded, "TEST")

    def test_splitmix_bootstrap_and_inverse_ecdf_have_golden_results(self) -> None:
        expected_indices = [
            [3, 5, 6, 3, 3, 5, 6],
            [3, 1, 5, 2, 4, 3, 3],
            [3, 1, 4, 5, 4, 6, 0],
            [0, 3, 0, 2, 0, 3, 4],
            [0, 6, 4, 4, 2, 3, 1],
            [3, 3, 5, 5, 4, 6, 4],
            [1, 4, 6, 5, 2, 1, 6],
            [6, 2, 0, 2, 6, 2, 0],
            [4, 1, 6, 6, 3, 0, 5],
            [3, 5, 1, 0, 0, 3, 0],
        ]
        self.assertEqual(
            reducer._splitmix_indices(1, 0, 10, 7).tolist(),
            expected_indices,
        )
        statistic = {
            "name": "heterogeneous",
            "numerator": np.asarray([1, 4, 2, 8, 3, 9, 5], dtype=np.float64),
            "denominator": np.asarray([1, 2, 1, 4, 2, 3, 5], dtype=np.float64),
        }
        expected_draws = np.asarray([
            1.8571428571428572, 2.1, 1.736842105263158,
            1.7142857142857142, 1.5294117647058822,
            1.9565217391304348, 1.6, 1.2,
            1.5909090909090908, 2.0,
        ])
        with mock.patch.object(reducer, "BOOTSTRAP_REPLICATES", 10), \
                mock.patch.object(reducer, "BOOTSTRAP_BATCH", 4):
            actual = reducer._bootstrap([statistic], 1)["heterogeneous"]
        np.testing.assert_allclose(actual, expected_draws, rtol=0, atol=1e-15)
        ordered = np.arange(20_000, dtype=np.float64)
        self.assertEqual(reducer._lower(ordered, 0.99), 199.0)
        self.assertEqual(reducer._upper(ordered, 0.99), 19_799.0)

    def test_simultaneous_max_standardized_error_is_deterministic(self) -> None:
        estimates = np.linspace(0.1, 0.9, 9)
        standard_errors = np.linspace(0.01, 0.09, 9)
        statistics = [
            {
                "name": f"component-{index}",
                "estimate": float(estimates[index]),
                "source_match_cluster_se": float(standard_errors[index]),
            }
            for index in range(9)
        ]
        # Shared, heterogeneous draws make component 8 determine the 99th
        # percentile maximum error. The zero-SE path is covered separately.
        base = np.linspace(-1.0, 1.0, 20_000)
        draws = {
            item["name"]: np.asarray(
                item["estimate"] + item["source_match_cluster_se"] *
                (base + index / 10.0), dtype=np.float64)
            for index, item in enumerate(statistics)
        }
        first = reducer._simultaneous_lower_bounds(
            statistics, draws, 0.99)
        second = reducer._simultaneous_lower_bounds(
            statistics, draws, 0.99)
        self.assertEqual(first, second)
        self.assertEqual(first["family_size"], 9)
        self.assertEqual(first["empirical_quantile_rank"], 19_799)
        expected_critical = base[19_799] + 0.8
        self.assertAlmostEqual(first["critical_value"], expected_critical)
        for item in statistics:
            self.assertAlmostEqual(
                first["lower_bounds"][item["name"]],
                item["estimate"] - expected_critical *
                item["source_match_cluster_se"])

        zero = [dict(item) for item in statistics]
        zero[0]["source_match_cluster_se"] = 0.0
        zero[0]["estimate"] = 1e-15
        zero_draws = dict(draws)
        zero_draws[zero[0]["name"]] = np.full(
            20_000, zero[0]["estimate"], dtype=np.float64)
        self.assertEqual(
            reducer._simultaneous_lower_bounds(zero, zero_draws, 0.99)
                ["lower_bounds"][zero[0]["name"]],
            None)
        zero_draws[zero[0]["name"]][0] = -1e-15
        crossing = reducer._simultaneous_lower_bounds(
            zero, zero_draws, 0.99)
        self.assertIsNone(crossing["lower_bounds"][zero[0]["name"]])
        self.assertIn(
            zero[0]["name"], crossing["zero_se_components_failed_closed"])
        zero_draws[zero[0]["name"]][0] = 1e-4
        with self.assertRaisesRegex(reducer.ReductionError, "zero-SE"):
            reducer._simultaneous_lower_bounds(zero, zero_draws, 0.99)

    def test_native_rows_pass_test_deterministically_and_never_promote(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-reduce-") as tmp:
            fixture = Fixture(Path(tmp))
            first = fixture.reduce()
            second = fixture.reduce()
        self.assertEqual(reducer.canonical_bytes(first),
                         reducer.canonical_bytes(second))
        self.assertTrue(reducer.verify_seal(first))
        self.assertEqual(first["schema"], reducer.RESULT_SCHEMA)
        self.assertTrue(first["verdict"]["accuracy_artifact_passed"])
        self.assertFalse(first["verdict"]["actor_promotion_authorized"])
        self.assertFalse(first["verdict"]["playing_strength_claimed"])
        self.assertTrue(first["gates"]["terminal_test_gate_applied"])
        self.assertIn("ece", first["not_evaluated_metrics"])
        self.assertEqual(first["inference"]["replicates"], 20000)
        self.assertEqual(first["inference"]["prng"],
                         "SplitMix64-counter-upper53")
        self.assertEqual(first["inference"]["familywise_coverage_claimed"],
                         "nominal_asymptotic")
        self.assertFalse(
            first["inference"]["exact_finite_sample_coverage_claimed"])
        self.assertEqual(first["inference"]["simultaneous_family_size"], 9)
        self.assertEqual(
            first["inference"]["simultaneous_critical_quantile_rank"],
            19_799)
        self.assertEqual(first["inference"]["simultaneous_components"], [
            "history_vs_matched_head_control.all_states_nll.absolute",
            "history_vs_matched_head_control.post_action_nll.absolute",
            "history_vs_matched_head_control.all_states_brier.absolute",
            "history_vs_incumbent_head.all_states_nll.absolute",
            "history_vs_incumbent_head.post_action_nll.absolute",
            "history_vs_incumbent_head.all_states_brier.absolute",
            "matched_head_control_vs_incumbent_head.all_states_nll.absolute",
            "matched_head_control_vs_incumbent_head.post_action_nll.absolute",
            "matched_head_control_vs_incumbent_head.all_states_brier.absolute",
        ])
        comparisons = first["metrics"]["pairwise_replacement_comparisons"]
        history_matched = comparisons["history_vs_matched_head_control"]
        primary = history_matched["all_states_joint_nll"]
        self.assertAlmostEqual(primary["candidate"]["estimate"], 0.9)
        self.assertAlmostEqual(
            primary["baseline"]["estimate"], 1.0)
        reports = first["metrics"][
            "per_model_scope_reports_not_additional_gates"]
        self.assertAlmostEqual(
            reports["all_states"]["history"]
                ["joint_nll_per_uncertain_card"]["estimate"], 9 / 40)
        self.assertAlmostEqual(
            primary["relative_improvement"]["estimate"], 0.1)
        self.assertGreater(
            primary["absolute_improvement"]["simultaneous_familywise_lcb"],
            0.0)
        self.assertLessEqual(
            primary["absolute_improvement"]["simultaneous_familywise_lcb"],
            primary["absolute_improvement"]["percentile_lcb"])
        self.assertTrue(history_matched["bundle_passed"])
        self.assertEqual(first["verdict"]["selected_belief_artifact"],
                         "history")
        self.assertEqual(
            first["verdict"]["selected_belief_artifact_binding"],
            {
                "sha256": "e" * 64,
                "alpha": 1.0,
                "fingerprint": "fedcba9876543210",
                "base_262k_head_sha256": "1" * 64,
                "base_net_fingerprint": "1111111111111111",
                "base_alpha": 1.0,
            })
        self.assertEqual(
            first["verdict"]["selected_belief_artifact_binding"]["sha256"],
            first["bindings"]["history_model_sha256"])
        top_k = first["metrics"]["top_k_recall_not_a_gate"]
        self.assertAlmostEqual(top_k["all_states"]["history"]["estimate"], 0.5)
        self.assertAlmostEqual(
            top_k["post_opponent_action"]
                ["matched_head_control"]["estimate"], 0.5)
        self.assertEqual(set(top_k["all_states"]), set(reducer.MODEL_KEYS))
        diagnostic = first["metrics"][
            "base_262k_head_candidate_deltas_diagnostic_only"]
        self.assertAlmostEqual(
            diagnostic["all_states_joint_nll"]
                ["relative_improvement"]["estimate"], 1.5 / 10.5)
        self.assertAlmostEqual(
            reports["post_opponent_action"]["history"]
                ["brier_per_uncertain_card"]["estimate"], 0.05)
        self.assertEqual(
            first["evidence"]["sample_counts"]["all_states"],
            {"state_count": 80, "uncertain_card_count": 320,
             "positive_count": 128},
        )
        self.assertEqual(
            primary["relative_improvement"]["observations"], 80)

    def test_zero_se_components_emit_verdict_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-zero-se-") as tmp:
            fixture = Fixture(Path(tmp))
            fixture.rows = [row(index) for index in range(8)]
            result = fixture.reduce()
        self.assertEqual(result["verdict"]["selected_belief_artifact"],
                         "incumbent_head")
        self.assertFalse(result["verdict"]["accuracy_artifact_passed"])
        comparison = result["metrics"]["pairwise_replacement_comparisons"] \
            ["history_vs_matched_head_control"]
        absolute = comparison["all_states_joint_nll"] \
            ["absolute_improvement"]
        self.assertIsNone(absolute["simultaneous_familywise_lcb"])
        self.assertFalse(absolute["simultaneous_inferentially_eligible"])
        self.assertFalse(comparison["bundle_passed"])
        self.assertEqual(
            len(result["inference"]["zero_se_components_failed_closed"]), 9)

    def test_zero_baseline_brier_never_requires_relative_ratio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-zero-brier-") as tmp:
            fixture = Fixture(Path(tmp))
            for value in fixture.rows:
                value["metrics"]["all_states"]["incumbent_head"] \
                    ["brier_sum"] = 0.0
                value["metrics"]["post_opponent_action"]["incumbent_head"] \
                    ["brier_sum"] = 0.0
            result = fixture.reduce()
        comparison = result["metrics"]["pairwise_replacement_comparisons"] \
            ["history_vs_incumbent_head"]
        self.assertIsNone(
            comparison["all_states_brier"]["relative_improvement"])
        self.assertFalse(comparison["metric_gates"]["all_states_brier"])
        self.assertEqual(result["verdict"]["selected_belief_artifact"],
                         "incumbent_head")

    def test_valid_capped_prefix_is_retained_and_reported(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-cap-") as tmp:
            fixture = Fixture(Path(tmp))
            fixture.rows[0]["capped_rounds"] = 1
            fixture.rows[0]["rounds_completed"] = 2
            fixture.rows[0]["excluded_state_count"] = 4
            result = fixture.reduce()
        self.assertTrue(result["verdict"]["accuracy_artifact_passed"])
        self.assertEqual(result["evidence"]["capped_rounds"], 1)
        self.assertEqual(result["evidence"]["rounds_completed"], 23)
        self.assertEqual(result["evidence"]["excluded_state_count"], 4)
        self.assertTrue(result["evidence"]["capped_prefix_metrics_retained"])

    def test_native_binary32_provenance_serialization_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-f32-") as tmp:
            fixture = Fixture(Path(tmp))
            for value in fixture.rows:
                value["temperature"] = 0.0299999993
            result = fixture.reduce()
        self.assertTrue(result["verdict"]["accuracy_artifact_passed"])

    def test_history_failure_falls_back_to_passing_matched_control(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-fail-") as tmp:
            fixture = Fixture(Path(tmp))
            for index, value in enumerate(fixture.rows):
                factor = 1.0 + (index - 3.5) / 100.0
                value["metrics"]["all_states"]["history"]["nll_sum"] = \
                    9.95 * factor
                value["metrics"]["post_opponent_action"]["history"] \
                    ["nll_sum"] = 8.15 * factor
            result = fixture.reduce()
        pair = result["metrics"]["pairwise_replacement_comparisons"] \
            ["history_vs_matched_head_control"]
        self.assertTrue(pair["metric_gates"]["all_states_joint_nll"])
        self.assertFalse(
            pair["metric_gates"]["post_opponent_action_joint_nll"])
        self.assertFalse(pair["bundle_passed"])
        self.assertEqual(result["verdict"]["selected_belief_artifact"],
                         "matched_head_control")
        self.assertEqual(
            result["verdict"]["selected_belief_artifact_binding"],
            {
                "sha256": "2" * 64,
                "alpha": 1.0,
                "fingerprint": "2222222222222222",
            })
        self.assertEqual(
            result["verdict"]["selected_belief_artifact_binding"]["sha256"],
            result["bindings"]["matched_head_control_sha256"])
        self.assertTrue(result["verdict"]["accuracy_artifact_passed"])
        self.assertFalse(result["verdict"]["actor_promotion_authorized"])

    def test_history_and_brier_each_fail_independently(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-history-fail-") as tmp:
            fixture = Fixture(Path(tmp))
            for index, value in enumerate(fixture.rows):
                factor = 1.0 + (index - 3.5) / 100.0
                value["metrics"]["all_states"]["history"]["nll_sum"] = \
                    9.8 * factor
                value["metrics"]["post_opponent_action"]["history"] \
                    ["nll_sum"] = 8.0 * factor
            result = fixture.reduce()
        pair = result["metrics"]["pairwise_replacement_comparisons"] \
            ["history_vs_matched_head_control"]
        self.assertTrue(pair["metric_gates"]["all_states_joint_nll"])
        self.assertFalse(
            pair["metric_gates"]["post_opponent_action_joint_nll"])
        self.assertEqual(result["verdict"]["selected_belief_artifact"],
                         "matched_head_control")

        with tempfile.TemporaryDirectory(prefix="lc-belief-brier-fail-") as tmp:
            fixture = Fixture(Path(tmp))
            for index, value in enumerate(fixture.rows):
                factor = 1.0 + (index - 3.5) / 100.0
                value["metrics"]["all_states"]["history"]["brier_sum"] = \
                    3.0 * factor
                value["metrics"]["post_opponent_action"]["history"] \
                    ["brier_sum"] = 2.6 * factor
            result = fixture.reduce()
        pair = result["metrics"]["pairwise_replacement_comparisons"] \
            ["history_vs_matched_head_control"]
        self.assertTrue(pair["metric_gates"]["all_states_joint_nll"])
        self.assertTrue(
            pair["metric_gates"]["post_opponent_action_joint_nll"])
        self.assertFalse(pair["metric_gates"]["all_states_brier"])
        self.assertEqual(result["verdict"]["selected_belief_artifact"],
                         "matched_head_control")

    def test_gate_uses_simultaneous_bound_not_marginal_percentile(self) -> None:
        original = reducer._simultaneous_lower_bounds

        def force_one_simultaneous_failure(statistics, draws, confidence):
            result = original(statistics, draws, confidence)
            target = next(
                item["name"] for item in statistics
                if item["name"] ==
                "history_vs_matched_head_control.all_states_nll.absolute")
            result["lower_bounds"][target] = -1e-12
            return result

        with tempfile.TemporaryDirectory(prefix="lc-belief-sim-gate-") as tmp, \
                mock.patch.object(
                    reducer, "_simultaneous_lower_bounds",
                    side_effect=force_one_simultaneous_failure):
            result = Fixture(Path(tmp)).reduce()
        pair = result["metrics"]["pairwise_replacement_comparisons"] \
            ["history_vs_matched_head_control"]
        absolute = pair["all_states_joint_nll"]["absolute_improvement"]
        self.assertGreater(absolute["percentile_lcb"], 0.0)
        self.assertLess(absolute["simultaneous_familywise_lcb"], 0.0)
        self.assertFalse(pair["metric_gates"]["all_states_joint_nll"])
        self.assertFalse(pair["bundle_passed"])
        self.assertEqual(result["verdict"]["selected_belief_artifact"],
                         "matched_head_control")

    def test_incumbent_fallback_and_nontransitive_edge_are_explicit(self) -> None:
        self.assertEqual(reducer._select_belief_artifact({
            "history_vs_matched_head_control": False,
            "history_vs_incumbent_head": True,
            "matched_head_control_vs_incumbent_head": False,
        }), "incumbent_head")
        with tempfile.TemporaryDirectory(prefix="lc-belief-incumbent-") as tmp:
            fixture = Fixture(Path(tmp))
            for value in fixture.rows:
                value["metrics"]["all_states"]["incumbent_head"].update({
                    "nll_sum": 8.0,
                    "brier_sum": 1.8,
                })
                value["metrics"]["post_opponent_action"] \
                    ["incumbent_head"].update({
                        "nll_sum": 6.4,
                        "brier_sum": 1.4,
                    })
            result = fixture.reduce()
        self.assertEqual(result["verdict"]["selected_belief_artifact"],
                         "incumbent_head")
        self.assertEqual(
            result["verdict"]["selected_belief_artifact_binding"],
            {
                "sha256": reducer.FROZEN_ACTOR_SHA256,
                "alpha": 1.15,
                "fingerprint": "0123456789abcdef",
            })
        self.assertEqual(
            result["verdict"]["selected_belief_artifact_binding"]["sha256"],
            result["bindings"]["actor_sha256"])
        self.assertFalse(result["verdict"]["stage_passed"])
        self.assertFalse(result["verdict"]["accuracy_artifact_passed"])

    def test_rejects_order_provenance_probe_and_nonfinite_drift(self) -> None:
        mutations = {
            "duplicate source ID": lambda rows: rows[3].__setitem__(
                "source_match_id", 2),
            "actor provenance": lambda rows: rows[0].__setitem__(
                "actor_fingerprint", "1111111111111111"),
            "reviewed ply": lambda rows: rows[0].__setitem__(
                "reviewed_ply_inputs_used", True),
            "uniform structural": lambda rows: rows[0]["structural_contract"]
                .__setitem__("opening_history_uniform", False),
            "exclusion manifest": lambda rows: rows[0].__setitem__(
                "exclusion_manifest_count", 16),
            "paired counts": lambda rows: rows[0]["metrics"]["all_states"]
                ["history"].__setitem__("state_count", 9),
            "round count": lambda rows: rows[0].__setitem__(
                "rounds_completed", 4),
            "impossible cap count": lambda rows: rows[0].__setitem__(
                "capped_rounds", 2),
            "uncapped incomplete": lambda rows: rows[0].__setitem__(
                "rounds_completed", 2),
            "post exceeds all": lambda rows: rows[0]["metrics"]
                ["post_opponent_action"]["history"].__setitem__(
                    "nll_sum", 10.0),
            "impossible exact-K support": lambda rows: [
                rows[0]["metrics"]["all_states"][model].update({
                    "positive_count": 9,
                    "top_hits_sum": 4.5,
                }) for model in reducer.MODEL_KEYS
            ],
            "opening uniform mismatch": lambda rows: rows[0]["metrics"]
                ["all_states"]["uniform_exact_k"].__setitem__(
                    "nll_sum", 11.5),
            "nonfinite": lambda rows: rows[0]["metrics"]["all_states"]
                ["history"].__setitem__("nll_sum", float("nan")),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                    prefix="lc-belief-invalid-") as tmp:
                fixture = Fixture(Path(tmp))
                mutate(fixture.rows)
                with self.assertRaises(reducer.ReductionError):
                    fixture.reduce()

    def test_rejects_non_native_crlf_and_field_order(self) -> None:
        for name, transform in (
            ("crlf", lambda raw, _: raw.replace(b"\n", b"\r\n")),
            ("sorted", lambda raw, value:
                reducer.canonical_bytes(value) + raw.split(b"\n", 1)[1]),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                    prefix="lc-belief-wire-") as tmp:
                fixture = Fixture(Path(tmp))
                p, p_sha, i, i_sha, paths = fixture.write()
                raw = paths[0].read_bytes()
                paths[0].write_bytes(transform(raw, fixture.rows[0]))
                with mock.patch.object(reducer, "FROZEN_TEST_MATCHES", 8), \
                        mock.patch.object(reducer, "FROZEN_TEST_SHARDS", 2), \
                        self.assertRaises(reducer.ReductionError):
                    reducer.reduce_evidence(
                        p, p_sha, i, i_sha, fixture.stage, paths)

    def test_reduce_rejects_unverified_or_stale_identity_digest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-id-digest-") as tmp:
            fixture = Fixture(Path(tmp))
            p, p_sha, i, _, paths = fixture.write()
            with mock.patch.object(reducer, "FROZEN_TEST_MATCHES", 8), \
                    mock.patch.object(reducer, "FROZEN_TEST_SHARDS", 2), \
                    self.assertRaisesRegex(
                        reducer.ReductionError, "identity SHA-256"):
                reducer.reduce_evidence(
                    p, p_sha, i, "0" * 64, fixture.stage, paths)
            forged = dict(i)
            forged[reducer.VERIFIED_IDENTITY_SHA256_FIELD] = "0" * 64
            with mock.patch.object(reducer, "FROZEN_TEST_MATCHES", 8), \
                    mock.patch.object(reducer, "FROZEN_TEST_SHARDS", 2), \
                    self.assertRaisesRegex(
                        reducer.ReductionError, "identity SHA-256"):
                reducer.reduce_evidence(
                    p, p_sha, forged, "not-a-sha", fixture.stage, paths)

    def test_identity_is_canonical_sealed_and_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-identity-") as tmp:
            root = Path(tmp)
            value = identity("TEST")
            path = root / "identity.json"
            path.write_text(json.dumps(value, indent=2), encoding="ascii")
            with self.assertRaisesRegex(reducer.ReductionError,
                                        "not canonical"):
                reducer.load_identity(path, "TEST")
            changed = copy.deepcopy(value)
            changed["row_contract"]["seed_root"] += 1
            path.write_bytes(reducer.canonical_bytes(changed))
            with self.assertRaisesRegex(reducer.ReductionError, "seal"):
                reducer.load_identity(path, "TEST")
            extra = copy.deepcopy(value)
            extra["bindings"]["unexpected_sha256"] = "9" * 64
            extra = reducer.seal({
                key: item for key, item in extra.items()
                if key != reducer.DIGEST_FIELD
            })
            path.write_bytes(reducer.canonical_bytes(extra))
            with self.assertRaisesRegex(reducer.ReductionError,
                                        "binding fields drift"):
                reducer.load_identity(path, "TEST")
            wrong_generator = copy.deepcopy(value)
            wrong_generator["bindings"]["test_generator_manifest_sha256"] = \
                "9" * 64
            wrong_generator = reducer.seal({
                key: item for key, item in wrong_generator.items()
                if key != reducer.DIGEST_FIELD
            })
            path.write_bytes(reducer.canonical_bytes(wrong_generator))
            loaded, _ = reducer.load_identity(path, "TEST")
            with self.assertRaisesRegex(reducer.ReductionError,
                                        "generator manifest"):
                reducer._validate_plan(plan(), loaded, "TEST")

    def test_rejects_empty_post_action_evidence_row(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-empty-post-") as tmp:
            fixture = Fixture(Path(tmp))
            for model in reducer.MODEL_KEY_ORDER:
                fixture.rows[0]["metrics"]["post_opponent_action"][model] = \
                    metric(0.0, 0.0, states=0, cards=0, positives=0)
            with self.assertRaisesRegex(reducer.ReductionError,
                                        "at least one scored state"):
                fixture.reduce()

    def test_canonical_no_clobber_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-belief-write-") as tmp:
            root = Path(tmp)
            result = Fixture(root).reduce()
            output = root / "verdict.json"
            reducer.write_no_clobber(output, result)
            self.assertEqual(output.read_bytes(), reducer.canonical_bytes(result))
            with self.assertRaisesRegex(reducer.ReductionError,
                                        "refusing to replace"):
                reducer.write_no_clobber(output, result)


if __name__ == "__main__":
    unittest.main()
