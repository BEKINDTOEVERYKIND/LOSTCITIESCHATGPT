"""Fail-closed contracts for the fresh-seed policy-cost-v8 recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import yaml
from tools import policy_cost_allocate_v8 as allocator
from tools import policy_cost_calibration_v8 as calibration
from tools import policy_cost_campaign_v8 as campaign


ROOT = Path(__file__).resolve().parents[1]
V7_LAUNCH = "9b83d6ad8a92ecf518f63935dbe39b4d937bf15a"
V7_PARENT = "3e49c14a2e4249358177ef9b59c02df07c0c740f"
V7_LAUNCH_TREE = "e46e2a2322d351a5b1c210c7b2b4bacc478be5d6"
V7_PLAN = ROOT / "data/experiments/locked_policy_cost_v7_plan.json"
V8_PLAN = ROOT / "data/experiments/locked_policy_cost_v8_plan.json"
V8_EXECUTION = ROOT / "data/experiments/locked_policy_cost_v8_execution.json"
V8_WORKFLOW = ROOT / ".github/workflows/policy-cost-v8.yml"
V8_DEFINITION = ROOT / ".github/workflows/policy-cost-v8-definition.yml"
V7_FAILURE = ROOT / "data/experiments/policy_cost_v7_run_33237653855_failure.json"
V7_RETAIN = ROOT / "data/experiments/policy_cost_v7_infrastructure_retain.json"


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


class PolicyCostV8RecoveryTests(unittest.TestCase):
    def test_v7_terminal_record_is_complete_and_non_promotable(self) -> None:
        failure, retain = load(V7_FAILURE), load(V7_RETAIN)
        self.assertEqual(failure["schema"],
                         "lc-policy-cost-v7-train-allocation-order-failure-v1")
        self.assertEqual(failure["status"], "complete_terminal_non_promotable")
        self.assertFalse(failure["efficacy_data_observed"])
        self.assertFalse(failure["locked_validation_relaxed"])
        self.assertEqual(failure["github_run"], {
            "attempt": 1, "conclusion": "failure", "event": "push",
            "head_sha": V7_LAUNCH, "head_tree": V7_LAUNCH_TREE,
            "id": 33237653855,
            "url": "https://github.com/BEKINDTOEVERYKIND/"
                   "LOSTCITIESCHATGPT/actions/runs/33237653855",
        })
        cause = failure["cause"]
        self.assertEqual(cause["classification"],
                         "train_allocation_canonical_order_binding_defect")
        self.assertFalse(cause["efficacy_related"])
        self.assertEqual(cause["stage"]["job_id"], 99103597866)
        self.assertEqual(cause["train_allocation_artifact"]["id"], 9711357261)
        self.assertIn("allocation-id schedule", cause["mechanism"])
        self.assertIn("lexicographic", cause["mechanism"])
        self.assertEqual(failure["job_disposition"], {
            "failure": 1, "skipped": 9, "success": 225,
            "total": 235, "train_evaluate_success": 216,
        })
        self.assertTrue(
            failure["fixed_seed_execution"]["all_policy_cost_v7_fixed_roots_retired"])
        self.assertEqual(failure["fixed_seed_execution"]["retired_fixed_root_count"], 21)
        self.assertEqual(failure["execution"]["sha256"],
                         digest(ROOT / failure["execution"]["path"]))
        self.assertEqual(retain["artifact_count"], 228)
        self.assertEqual(retain["train_evaluation_artifacts"]["count"], 216)
        self.assertFalse(retain["promotion_gate_passed"])
        self.assertFalse(retain["maintained_actor_changed"])

    def test_v7_launch_is_the_unique_direct_addendum(self) -> None:
        parent = subprocess.check_output(
            ["git", "rev-parse", f"{V7_LAUNCH}^"], cwd=ROOT, text=True).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{V7_LAUNCH}^{{tree}}"], cwd=ROOT,
            text=True).strip()
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", V7_LAUNCH],
            cwd=ROOT, text=True).strip()
        self.assertEqual(parent, V7_PARENT)
        self.assertEqual(tree, V7_LAUNCH_TREE)
        self.assertEqual(changed,
                         "A\tdata/experiments/locked_policy_cost_v7_execution.json")

    def test_every_v7_campaign_byte_is_immutable(self) -> None:
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", V7_LAUNCH],
            cwd=ROOT, text=True).splitlines()
        prefixes = (
            ".github/workflows/policy-cost-v", "data/experiments/locked_policy_cost_v",
            "data/experiments/policy_cost_v", "src/policy_cost",
            "tests/test_policy_cost", "tools/policy_cost",
        )
        frozen = [path for path in paths if path.startswith(prefixes) or
                  path == "tools/build_policy_cost.c" or
                  re.fullmatch(r"POLICY_COST_V[1-7]\.md", path)]
        self.assertIn("tools/policy_cost_dataset_v7.c", frozen)
        for relative in frozen:
            current = ROOT / relative
            self.assertTrue(current.is_file(), relative)
            expected = subprocess.check_output(
                ["git", "show", f"{V7_LAUNCH}:{relative}"], cwd=ROOT)
            self.assertEqual(current.read_bytes(), expected, relative)

    def test_all_twenty_one_v8_roots_are_fresh_and_distinct(self) -> None:
        plans = [load(ROOT / f"data/experiments/locked_policy_cost_v{i}_plan.json")
                 for i in range(1, 9)]
        roots = active_roots(plans[-1])
        previous = set().union(*(exact_seed_values(plan) for plan in plans[:-1]))
        self.assertEqual(len(roots), 21)
        self.assertTrue(all(root.startswith("202707") for root in roots))
        self.assertFalse(roots & previous)
        burned = plans[-1]["reservoirs"]["burned_source_seeds"]
        self.assertIn("all policy-cost-v7 fixed seeds", burned)
        self.assertIn("every 20270529 feasibility-smoke seed", burned)

    def test_v8_preserves_every_efficacy_design_and_gate(self) -> None:
        v7, v8 = load(V7_PLAN), load(V8_PLAN)
        cal7, cal8 = copy.deepcopy(v7["calibration"]), copy.deepcopy(v8["calibration"])
        cal7["schedule_seed"] = cal8["schedule_seed"]
        cal7["cross_validation"]["seed"] = cal8["cross_validation"]["seed"]
        self.assertEqual(cal7, cal8)
        ctl7, ctl8 = copy.deepcopy(v7["controller"]), copy.deepcopy(v8["controller"])
        ctl7["artifact_binding"]["source_seed"] = ctl8["artifact_binding"]["source_seed"]
        self.assertEqual(ctl7, ctl8)
        self.assertEqual(v7["multiplicity"], v8["multiplicity"])
        self.assertEqual(v7["prerequisite"], v8["prerequisite"])
        self.assertEqual(v7["test"], v8["test"])
        for stage in ("safety", "final"):
            self.assertEqual(without(v7[stage], "seeds"), without(v8[stage], "seeds"))
        self.assertEqual(without(v7["selection"], "bootstrap"),
                         without(v8["selection"], "bootstrap"))
        self.assertEqual(without(v7["selection"]["bootstrap"], "seed"),
                         without(v8["selection"]["bootstrap"], "seed"))
        for key in ("allocation", "candidate_masks",
                    "native_reservoir_origin_proof"):
            self.assertEqual(v7["reservoirs"][key], v8["reservoirs"][key])
        self.assertEqual(seed_agnostic(v7["reservoirs"]["search_and_truth"]),
                         seed_agnostic(v8["reservoirs"]["search_and_truth"]))
        self.assertEqual(
            without(v7["reservoirs"]["discovery"], "seeds", "sealed_smokes"),
            without(v8["reservoirs"]["discovery"], "seeds", "sealed_smokes"))

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
        self.assertFalse(V8_EXECUTION.exists())
        workflow = yaml.safe_load(V8_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(workflow[True]["push"]["paths"],
                         ["data/experiments/locked_policy_cost_v8_execution.json"])
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])
        text = V8_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("workflow_dispatch", text)
        self.assertIn("test \"$GITHUB_RUN_ATTEMPT\" = 1", text)
        self.assertIn("tools/policy_cost_dataset_v8.c", text)
        self.assertIn("tools/policy_cost_campaign_v8.py", text)
        self.assertIn("locked_policy_cost_v7_execution.json", text)
        self.assertIn("policy_cost_v7_run_33237653855_failure.json", text)
        self.assertIn("bin/policy_cost_dataset self-test", text)
        definition = V8_DEFINITION.read_text(encoding="utf-8")
        self.assertIn("tools/policy_cost_dataset_v8.c", definition)
        self.assertIn("tests/test_policy_cost_v8_recovery.py", definition)


if __name__ == "__main__":
    unittest.main()
