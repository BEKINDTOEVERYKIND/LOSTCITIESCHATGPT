"""Fail-closed contracts for the fresh-seed phase-selective policy-cost v17."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import yaml
from tools import policy_cost_allocate_v17 as allocator
from tools import policy_cost_calibration_v17 as calibration
from tools import policy_cost_campaign_v17 as campaign


ROOT = Path(__file__).resolve().parents[1]
V11_LAUNCH = "a33cf258187a61a517df5fbc8d02bbf6445fbcd7"
V11_PARENT = "0bde020326288d2220a03d598755fa017aeac6de"
V11_LAUNCH_TREE = "b6b00d3cf953ed379f154a9975da2b0bcfc6d546"
V11_PLAN = ROOT / "data/experiments/locked_policy_cost_v11_plan.json"
V16_PLAN = ROOT / "data/experiments/locked_policy_cost_v16_plan.json"
V17_PLAN = ROOT / "data/experiments/locked_policy_cost_v17_plan.json"
V17_EXECUTION = ROOT / "data/experiments/locked_policy_cost_v17_execution.json"
V17_WORKFLOW = ROOT / ".github/workflows/policy-cost-v17.yml"
V17_DEFINITION = ROOT / ".github/workflows/policy-cost-v17-definition.yml"
V11_FAILURE = ROOT / "data/experiments/policy_cost_v11_run_33313785880_failure.json"
V11_RETAIN = ROOT / "data/experiments/policy_cost_v11_infrastructure_retain.json"
V16_FAILURE = ROOT / "data/experiments/policy_cost_v16_run_33473566182_failure.json"
V16_GATE = ROOT / "data/experiments/policy_cost_v16_run_33473566182_calibration_gate.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_roots(plan: dict) -> set[str]:
    roots = {
        plan["calibration"]["schedule_seed"],
        plan["selection"]["bootstrap"]["seed"],
        *plan["reservoirs"]["discovery"]["seeds"].values(),
        *plan["safety"]["seeds"].values(),
        *plan["final"]["seeds"].values(),
    }
    search = plan["reservoirs"]["search_and_truth"]
    for family in ("primary", "fresh", "maintained_actor_decision"):
        if family in search:
            roots.update(search[family][split]
                         for split in ("TRAIN", "SELECT", "TEST"))
    roots.update(search["truth"][split]["seed"]
                 for split in ("TRAIN", "SELECT", "TEST"))
    return roots


def without(value: dict, *keys: str) -> dict:
    result = copy.deepcopy(value)
    for key in keys:
        result.pop(key, None)
    return result


def seed_agnostic(value):
    """Remove only fixed-root identities while preserving every design byte."""
    if isinstance(value, dict):
        return {key: seed_agnostic(item) for key, item in value.items()}
    if isinstance(value, list):
        return [seed_agnostic(item) for item in value]
    if isinstance(value, str) and re.fullmatch(r"20\d{10}", value):
        return "<fresh-seed>"
    return value


def exact_seed_values(value) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(exact_seed_values(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(exact_seed_values(item) for item in value))
    if isinstance(value, str) and re.fullmatch(r"20\d{10}", value):
        return {value}
    return set()


class PolicyCostV17RecoveryTests(unittest.TestCase):
    @staticmethod
    def _fringe_policy(bits: list[int], serialized_core: float) -> dict:
        draws = (0, 1, 5)
        legal = []
        for index, (word, draw) in enumerate(zip(bits, draws)):
            probability = float(struct.unpack("<f", struct.pack("<I", word))[0])
            legal.append({
                "index": index,
                "move_pack": 46 + 120 * draw,
                "semantic_move_pack": 46 + 120 * draw,
                "card": 46,
                "discard": 0,
                "draw": draw,
                "probability": probability,
                "probability_bits": f"{word:08x}",
                "semantic_action_probability": serialized_core,
                "conditional_draw_probability": 0.0,
            })
        raw = sum(item["probability"] for item in legal)
        for item in legal:
            item["conditional_draw_probability"] = item["probability"] / raw
        return {
            "legal": legal,
            "legal_count": 3,
            "symmetries": 20,
            "exact_group_average": True,
            "literal_argmax_index": max(
                range(3), key=lambda index: legal[index]["probability"]
            ),
        }

    def test_binary32_semantic_mass_fringe_is_canonical_but_material_overshoot_fails(self) -> None:
        witness = self._fringe_policy(
            [0x3DAE27A8, 0x3EE3CD47, 0x3EF0A8D0], 1.0
        )
        raw = sum(item["probability"] for item in witness["legal"])
        self.assertEqual(raw, 1.0000000298023224)
        campaign._verify_full_policy(witness)

        material = self._fringe_policy([0x3EAAAAC1] * 3, 1.0)
        with self.assertRaisesRegex(
            campaign.EvidenceError, "semantic core exceeds binary32 tolerance"
        ):
            campaign._verify_full_policy(material)

    def test_v11_terminal_record_is_complete_and_non_promotable(self) -> None:
        failure, retain = load(V11_FAILURE), load(V11_RETAIN)
        self.assertEqual(failure["schema"],
                         "lc-policy-cost-v11-binary32-semantic-mass-roundoff-failure-v1")
        self.assertEqual(failure["status"], "complete_terminal_non_promotable")
        self.assertFalse(failure["promotion_efficacy_data_observed"])
        self.assertFalse(failure["locked_validation_relaxed"])
        self.assertEqual(failure["github_run"], {
            "attempt": 1, "conclusion": "failure", "event": "push",
            "head_sha": V11_LAUNCH, "head_tree": V11_LAUNCH_TREE,
            "id": 33313785880,
            "url": "https://github.com/BEKINDTOEVERYKIND/"
                   "LOSTCITIESCHATGPT/actions/runs/33313785880",
        })
        cause = failure["cause"]
        self.assertEqual(cause["classification"],
                         "binary32_semantic_action_mass_roundoff_overshoot")
        self.assertFalse(cause["efficacy_related"])
        self.assertEqual(cause["stage"]["job_id"], 99319526486)
        self.assertIn("1.0000000298023224", cause["mechanism"])
        self.assertIn("before calibration fitting", cause["mechanism"])
        self.assertEqual(failure["job_disposition"], {
            "failure": 1, "skipped": 9, "success": 63, "total": 73,
            "train_evaluate_success": 54,
        })
        self.assertTrue(
            failure["fixed_seed_execution"]["all_policy_cost_v11_fixed_roots_retired"])
        self.assertEqual(failure["fixed_seed_execution"]["retired_fixed_root_count"], 21)
        self.assertTrue(failure["fixed_seed_execution"]["retry_forbidden"])
        self.assertEqual(failure["execution"]["sha256"],
                         digest(ROOT / failure["execution"]["path"]))
        self.assertEqual(retain["artifact_count"], 66)
        self.assertEqual(retain["train_evaluation_artifacts"]["batch_count"], 54)
        self.assertEqual(retain["train_evaluation_artifacts"]["logical_slice_count"], 216)
        self.assertEqual(retain["train_evaluation_artifacts"]["missing_indices"], [])
        self.assertTrue(retain["complete_retention_job"])
        self.assertTrue(retain["raw_stage_artifacts_available"])
        self.assertTrue(retain["train_evaluation_artifacts"]["outer_sha256_verified"])
        self.assertTrue(retain["train_evaluation_artifacts"]["sidecar_sha256_verified"])
        self.assertEqual(failure["invalid_probability_witness"]["invalid_row_count"], 1)
        self.assertEqual(failure["invalid_probability_witness"]["allocation_id"], 1223)
        self.assertFalse(retain["promotion_gate_passed"])
        self.assertFalse(retain["maintained_actor_changed"])

    def test_v11_launch_is_the_unique_direct_addendum(self) -> None:
        parent = subprocess.check_output(
            ["git", "rev-parse", f"{V11_LAUNCH}^"], cwd=ROOT, text=True).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{V11_LAUNCH}^{{tree}}"], cwd=ROOT,
            text=True).strip()
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", V11_LAUNCH],
            cwd=ROOT, text=True).strip()
        self.assertEqual(parent, V11_PARENT)
        self.assertEqual(tree, V11_LAUNCH_TREE)
        self.assertEqual(changed,
                         "A\tdata/experiments/locked_policy_cost_v11_execution.json")

    def test_every_v11_campaign_byte_is_immutable(self) -> None:
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", V11_LAUNCH],
            cwd=ROOT, text=True).splitlines()
        prefixes = (
            ".github/workflows/policy-cost-v", "data/experiments/locked_policy_cost_v",
            "data/experiments/policy_cost_v", "src/policy_cost",
            "tests/test_policy_cost", "tools/policy_cost",
        )
        frozen = [path for path in paths if path.startswith(prefixes) or
                  path == "tools/build_policy_cost.c" or
                  re.fullmatch(r"POLICY_COST_V(?:[1-9]|1[01])\.md", path)]
        self.assertIn("tools/policy_cost_dataset_v11.c", frozen)
        for relative in frozen:
            current = ROOT / relative
            self.assertTrue(current.is_file(), relative)
            expected = subprocess.check_output(
                ["git", "show", f"{V11_LAUNCH}:{relative}"], cwd=ROOT)
            self.assertEqual(current.read_bytes(), expected, relative)

    def test_all_twenty_one_v17_roots_are_fresh_and_distinct(self) -> None:
        plans = [load(ROOT / f"data/experiments/locked_policy_cost_v{i}_plan.json")
                 for i in range(1, 18)]
        roots = active_roots(plans[-1])
        previous = set().union(*(exact_seed_values(plan) for plan in plans[:-1]))
        self.assertEqual(len(roots), 21)
        self.assertTrue(all(root.startswith("202804") for root in roots))
        self.assertFalse(roots & previous)
        burned = plans[-1]["reservoirs"]["burned_source_seeds"]
        self.assertIn("all policy-cost-v11 fixed seeds", burned)
        self.assertIn("every 20271029 feasibility-smoke seed", burned)
        self.assertIn("all policy-cost-v12 fixed seeds", burned)
        self.assertIn("every 20271129 feasibility-smoke seed", burned)
        self.assertIn("all policy-cost-v14 fixed seeds", burned)
        self.assertIn("every 20280129 feasibility-smoke seed", burned)
        self.assertIn("all policy-cost-v15 fixed seeds", burned)
        self.assertIn("every 20280229 feasibility-smoke seed", burned)
        self.assertIn("all policy-cost-v16 fixed seeds", burned)
        self.assertIn("every 20280329 feasibility-smoke seed", burned)

    def test_native_evaluator_and_python_consumer_share_burned_seed_literal(self) -> None:
        source = (ROOT / "tools/policy_cost_dataset_v17.c").read_text(encoding="utf-8")
        start = source.rindex('"\\\"burned_source_deal_seeds\\\":\\\"')
        end = source.index('"\\\"burned_seed_intersection\\\":0', start)
        fragments = [
            line.strip()[1:-1]
            for line in source[start:end].splitlines()
            if line.strip().startswith('"') and line.strip().endswith('"')
        ]
        rendered = "".join(
            bytes(fragment, "ascii").decode("unicode_escape")
            for fragment in fragments
        )
        prefix = '"burned_source_deal_seeds":"'
        self.assertTrue(rendered.startswith(prefix), repr(rendered[:160]))
        self.assertTrue(rendered.endswith('",'))
        self.assertEqual(rendered[len(prefix):-2], campaign.BURNED_SOURCE_DEAL_SEEDS)

    def test_v17_changes_only_preregistered_controller_semantics_and_fit(self) -> None:
        v16, v17 = load(V16_PLAN), load(V17_PLAN)
        self.assertEqual(v16["calibration"]["anchors"], v17["calibration"]["anchors"])
        self.assertEqual(v16["calibration"]["solver"], v17["calibration"]["solver"])
        self.assertEqual(v16["calibration"]["smoothness_grid"],
                         v17["calibration"]["smoothness_grid"])
        self.assertEqual(v16["calibration"]["huber_delta"],
                         v17["calibration"]["huber_delta"])
        self.assertEqual(v16["calibration"]["standard_error_floor"],
                         v17["calibration"]["standard_error_floor"])
        self.assertEqual(v16["calibration"]["negative_evidence_contract"],
                         v17["calibration"]["negative_evidence_contract"])
        self.assertIn("fixed exactly zero for 16 <= ply < 40",
                      v17["calibration"]["coefficient_constraints"])
        self.assertIn("unrestricted_full_spline",
                      next(key for key in v17["calibration"]["model_lack"]
                           if key.startswith("reject_if_")))
        ctl16 = copy.deepcopy(v16["controller"])
        ctl17 = copy.deepcopy(v17["controller"])
        ctl16["artifact_binding"]["source_seed"] = ctl17["artifact_binding"]["source_seed"]
        ctl16["adjusted_score"] = ctl17["adjusted_score"]
        ctl16["decision"] = ctl17["decision"]
        self.assertEqual(ctl16, ctl17)
        self.assertEqual(v16["multiplicity"], v17["multiplicity"])
        self.assertEqual(v16["prerequisite"], v17["prerequisite"])
        self.assertEqual(v16["test"], v17["test"])
        for stage in ("safety", "final"):
            self.assertEqual(without(v16[stage], "seeds"), without(v17[stage], "seeds"))
        self.assertEqual(without(v16["selection"], "bootstrap"),
                         without(v17["selection"], "bootstrap"))
        self.assertEqual(without(v16["selection"]["bootstrap"], "seed"),
                         without(v17["selection"]["bootstrap"], "seed"))
        for key in ("allocation", "candidate_masks",
                    "native_reservoir_origin_proof"):
            self.assertEqual(v16["reservoirs"][key], v17["reservoirs"][key])
        self.assertEqual(seed_agnostic(v16["reservoirs"]["search_and_truth"]),
                         seed_agnostic(v17["reservoirs"]["search_and_truth"]))
        self.assertEqual(
            without(v16["reservoirs"]["discovery"], "seeds", "sealed_smokes"),
            without(v17["reservoirs"]["discovery"], "seeds", "sealed_smokes"))

    def test_v16_terminal_record_and_phase_hypothesis_are_fail_closed(self) -> None:
        failure, gate = load(V16_FAILURE), load(V16_GATE)
        self.assertEqual(
            failure["schema"],
            "lc-policy-cost-v16-terminal-train-model-adequacy-and-retention-failure-v1",
        )
        self.assertEqual(failure["status"], "complete_terminal_non_promotable")
        self.assertFalse(failure["partial_efficacy_inspected"])
        self.assertFalse(failure["promotion_efficacy_data_observed"])
        self.assertFalse(failure["maintained_actor_changed"])
        self.assertEqual(failure["github_run"]["id"], 33473566182)
        self.assertEqual(failure["github_run"]["attempt"], 1)
        self.assertEqual(gate["status"], "failed_model_adequacy")
        self.assertFalse(gate["calibration_passed"])
        failed = failure["authoritative_train_gate"]["failed_requirement"]
        self.assertGreater(failed["ply_early_0_15"], 0.0)
        self.assertLess(failed["ply_mid_16_39"], 0.0)
        self.assertGreater(failed["ply_late_40_63"], 0.0)
        self.assertTrue(
            failure["fixed_seed_execution"]["all_policy_cost_v16_fixed_roots_retired"]
        )
        self.assertTrue(failure["fixed_seed_execution"]["retry_forbidden"])
        self.assertEqual(failure["immutable_calibration_artifact"]["train_evaluation_batch_count"], 54)
        self.assertEqual(failure["immutable_calibration_artifact"]["train_logical_slice_count"], 216)
        self.assertEqual(
            failure["execution_blob"]["sha256"],
            digest(ROOT / failure["execution_blob"]["path"]),
        )
        self.assertEqual(calibration.FitConfig().max_irls_iterations, 2000)

    def test_phase_selective_basis_is_exactly_zero_in_midgame(self) -> None:
        anchors = calibration.DEFAULT_PLY_ANCHORS
        active = calibration.policy_active_anchor_indices(anchors)
        self.assertEqual(active, (0, 1, 2, 3, 7, 8, 9))
        for ply in range(16, 40):
            basis = calibration.phase_selective_policy_basis(ply, anchors)
            self.assertEqual(tuple(basis.tolist()), (0.0,) * len(active))
        expanded = calibration.expand_phase_selective_policy_coefficients(
            [1.0] * len(active), anchors
        )
        self.assertEqual([expanded[i] for i in (4, 5, 6)], [0.0, 0.0, 0.0])

    def test_cell_saturated_different_core_allows_only_optional_zero_draw(self) -> None:
        config = calibration.FitConfig(
            anchors=calibration.DEFAULT_PLY_ANCHORS,
            smoothness_grid=calibration.DEFAULT_SMOOTHNESS_GRID,
            folds=5,
            fold_seed=calibration.DEFAULT_FOLD_SEED,
            huber_delta=1.345,
            min_search_beta=calibration.DEFAULT_MIN_SEARCH_BETA,
            min_core_alpha=0.0,
            min_draw_alpha=0.0,
            standard_error_floor=calibration.DEFAULT_STANDARD_ERROR_FLOOR,
            require_campaign_design=False,
            model_lack_max_relative_improvement=(
                calibration.MODEL_LACK_MAX_RELATIVE_IMPROVEMENT
            ),
        )
        observations = tuple(
            calibration.PairObservation(
                source_match_id=f"TRAIN-{index:012d}",
                state_id=f"state-{index}", pair_id=f"pair-{index}",
                ply=0, search_delta=search, truth_delta=0.0,
                log_core_ratio=core, log_draw_ratio=0.0,
                search_se=1.0, truth_se=1.0, round_index=0,
                pair_type="different_core",
            )
            for index, (search, core) in enumerate(
                ((1.0, 0.8), (-1.0, 1.0), (2.0, 1.2), (-2.0, 1.4))
            )
        )
        diagnostic = calibration._cell_design_diagnostic(
            (0, 0, 2, "different_core"), observations, config
        )
        self.assertEqual(
            diagnostic["active_columns"], ["search_delta", "log_core_ratio"]
        )
        self.assertEqual(
            diagnostic["inactive_structural_zero_columns"], ["log_draw_ratio"]
        )
        self.assertEqual(diagnostic["expected_rank"], 2)

        invalid = tuple(
            calibration.PairObservation(
                source_match_id=item.source_match_id,
                state_id=item.state_id,
                pair_id=item.pair_id,
                ply=item.ply,
                search_delta=item.search_delta,
                truth_delta=item.truth_delta,
                log_core_ratio=0.0,
                log_draw_ratio=0.0,
                search_se=item.search_se,
                truth_se=item.truth_se,
                round_index=item.round_index,
                pair_type=item.pair_type,
            )
            for item in observations
        )
        with self.assertRaisesRegex(calibration.CalibrationError,
                                    "zero required columns"):
            calibration._cell_design_diagnostic(
                (0, 0, 2, "different_core"), invalid, config
            )

    def test_batching_reduces_runner_exposure_without_changing_logical_slices(self) -> None:
        workflow = yaml.safe_load(V17_WORKFLOW.read_text(encoding="utf-8"))
        jobs = workflow["jobs"]
        self.assertEqual(jobs["train_evaluate"]["strategy"]["matrix"]["batch"],
                         list(range(54)))
        self.assertEqual(jobs["select_evaluate"]["strategy"]["matrix"]["batch"],
                         list(range(48)))
        self.assertEqual(jobs["test_evaluate"]["strategy"]["matrix"]["batch"],
                         list(range(48)))
        text = V17_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("FIRST=$(( ${{ matrix.batch }} * 4 ))", text)
        self.assertIn("test \"$(find train-slices -maxdepth 1 -type f -name '*.jsonl' | wc -l)\" = 4", text)
        self.assertIn('test "${#SLICES[@]}" = 216', text)
        self.assertEqual(text.count('test "${#SLICES[@]}" = 192'), 2)

    def test_full_train_manifest_is_canonical_but_bindings_stay_scheduled(self) -> None:
        lines = [
            "LCPOLICYCOST-TRAIN-ALLOCATION-V5", "split\tTRAIN",
            "purpose\tcampaign", "discovery_sha256\t" + "a" * 64,
            "reservoir_sha256\t" + "b" * 64,
            "source_net_sha256\t" + "c" * 64,
            "source_exclusion_sha256\t" + "d" * 64,
            "eligible_pair_commitment_sha256\t" + "e" * 64,
            "allocation_rule_sha256\t" + campaign.TRAIN_ALLOCATION_RULE_SHA256,
            "quota_per_cell\t16", "eligible_units\t13824",
            "retained_reservoir_units\t13824", "probe_orbit_rejections\t0",
            "pooled_ge64_observed\t0", "records\t13824",
            "columns\t" + "\t".join(campaign.TRAIN_ALLOCATION_COLUMNS),
        ]
        for allocation_id in range(campaign.TRAIN_RECORDS):
            rd, ply_bin, ratio_bin, pair_code = campaign._train_scheduled_cell(
                allocation_id
            )
            state = bytearray(174)
            state[0] = 1
            state[157:165] = allocation_id.to_bytes(8, "little")
            state_hex = state.hex()
            state_sha = hashlib.sha256(state).hexdigest()
            pair_sha = hashlib.sha256(
                bytes.fromhex(state_sha) + b"\x0a\x00\x0b\x00"
            ).hexdigest()
            source = f"TRAIN-{allocation_id:012d}"
            lines.append("\t".join((
                str(allocation_id), str(allocation_id), "0", source,
                source + ":s000", "00010-00011",
                f"r{rd}.p{ply_bin}.g{ratio_bin}.t{pair_code}", str(rd),
                str(ply_bin), str(ratio_bin), str(pair_code), "10", "11",
                f"{allocation_id + 1:064x}", state_sha, pair_sha,
                f"{allocation_id + 2:064x}", "d" * 64, "e" * 64,
                "f" * 64, state_hex,
            )))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train-allocation.tsv"
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            manifest, bindings = campaign.train_allocation_manifest(path)
        self.assertEqual(
            [row["allocation_id"] for row in bindings],
            list(range(campaign.TRAIN_RECORDS)),
        )
        order = [(
            row["round"], row["ply_bin"], row["ratio_bin"], row["pair_type"],
            row["allocation_priority_sha256"], row["state_sha256"],
            row["source_match_id"], row["state_id"], row["pair_id"],
        ) for row in manifest["selected_units"]]
        self.assertEqual(order, sorted(order))
        ratio_logs = tuple(
            math.log(value) for value in calibration.RATIO_BAND_LOWER_BOUNDS
        )
        observations = [calibration.PairObservation(
            source_match_id=row["source_match_id"], state_id=row["state_id"],
            pair_id=row["pair_id"], ply=campaign.PLY_STRATA[row["ply_bin"]][0],
            search_delta=0.0, truth_delta=0.0,
            log_core_ratio=(
                ratio_logs[row["ratio_bin"]]
                if row["pair_type"] == "different_core" else 0.0
            ),
            log_draw_ratio=(
                ratio_logs[row["ratio_bin"]]
                if row["pair_type"] == "same_core_draw" else 0.0
            ),
            search_se=0.0, truth_se=0.0,
            round_index=row["round"], pair_type=row["pair_type"],
        ) for row in manifest["selected_units"]]
        bound = calibration._campaign_allocation_binding(observations, manifest)
        self.assertTrue(bound["validated"])
        self.assertEqual(bound["selected_units"], campaign.TRAIN_RECORDS)

    def test_producer_boundary_still_rejects_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.jsonl"
            invalid = root / "invalid.jsonl"
            valid.write_text('{"conditional_draw_probability":0.0}\n',
                             encoding="ascii")
            invalid.write_text('{"conditional_draw_probability":-nan}\n',
                               encoding="ascii")
            self.assertEqual(campaign.strict_jsonl(valid), [
                {"conditional_draw_probability": 0.0}
            ])
            with self.assertRaisesRegex(campaign.EvidenceError,
                                        "invalid JSONL line 1"):
                campaign.strict_jsonl(invalid)

    def test_definition_is_inert_and_launch_is_fail_closed(self) -> None:
        self.assertFalse(V17_EXECUTION.exists())
        workflow = yaml.safe_load(V17_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(workflow[True]["push"]["paths"],
                         ["data/experiments/locked_policy_cost_v17_execution.json"])
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])
        text = V17_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("workflow_dispatch", text)
        self.assertIn("test \"$GITHUB_RUN_ATTEMPT\" = 1", text)
        self.assertIn("tools/policy_cost_dataset_v17.c", text)
        self.assertIn("tools/policy_cost_campaign_v17.py", text)
        self.assertIn("locked_policy_cost_v16_execution.json", text)
        self.assertIn("policy_cost_v16_run_33473566182_failure.json", text)
        self.assertIn("policy_cost_v16_run_33473566182_calibration_gate.json", text)
        self.assertIn("--no-index --no-deps --target ../python-runtime", text)
        self.assertIn("assert numpy.__version__ == \"2.3.5\"", text)
        self.assertIn("! -name 'test_policy_cost_v*_recovery.py'", text)
        self.assertNotIn("! -name 'test_policy_cost_v9_recovery.py'", text)
        self.assertIn("bin/policy_cost_dataset self-test", text)
        definition = V17_DEFINITION.read_text(encoding="utf-8")
        self.assertIn("tools/policy_cost_dataset_v17.c", definition)
        self.assertIn("tests/test_policy_cost_v17_recovery.py", definition)
        versioned_native = definition.split(
            "- name: Compile the exact versioned native runtime", 1
        )[1].split(
            "- name: Validate the complete definition", 1
        )[0]
        self.assertIn("bin/test_policy_cost_v17", versioned_native)
        self.assertNotIn("bin/test_policy_cost\n", versioned_native)


if __name__ == "__main__":
    unittest.main()
