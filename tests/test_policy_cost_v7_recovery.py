"""Fail-closed contracts for the fresh-seed policy-cost-v7 recovery."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import yaml
from tools import policy_cost_allocate_v7 as allocator
from tools import policy_cost_campaign_v7 as campaign


ROOT = Path(__file__).resolve().parents[1]
V6_LAUNCH = "5175504127c0dd836e07ed60c10541956ceea349"
V6_PARENT = "f7b500776c4284c83c9fc3a69e7220ee0b02a0aa"
V6_LAUNCH_TREE = "802bd64af71ed651a003aeb2faa23c0a383bfa5b"
V6_PLAN = ROOT / "data/experiments/locked_policy_cost_v6_plan.json"
V7_PLAN = ROOT / "data/experiments/locked_policy_cost_v7_plan.json"
V7_EXECUTION = ROOT / "data/experiments/locked_policy_cost_v7_execution.json"
V7_WORKFLOW = ROOT / ".github/workflows/policy-cost-v7.yml"
V7_DEFINITION = ROOT / ".github/workflows/policy-cost-v7-definition.yml"
V6_FAILURE = ROOT / "data/experiments/policy_cost_v6_run_33213087155_failure.json"
V6_RETAIN = ROOT / "data/experiments/policy_cost_v6_infrastructure_retain.json"


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


class PolicyCostV7RecoveryTests(unittest.TestCase):
    def test_v6_terminal_record_is_complete_and_non_promotable(self) -> None:
        failure, retain = load(V6_FAILURE), load(V6_RETAIN)
        self.assertEqual(failure["schema"],
                         "lc-policy-cost-v6-zero-mass-serialization-failure-v1")
        self.assertEqual(failure["status"], "complete_terminal_non_promotable")
        self.assertFalse(failure["efficacy_data_observed"])
        self.assertFalse(failure["locked_validation_relaxed"])
        self.assertEqual(failure["github_run"], {
            "attempt": 1, "conclusion": "failure", "event": "push",
            "head_sha": V6_LAUNCH, "head_tree": V6_LAUNCH_TREE,
            "id": 33213087155,
            "url": "https://github.com/BEKINDTOEVERYKIND/"
                   "LOSTCITIESCHATGPT/actions/runs/33213087155",
        })
        cause = failure["cause"]
        self.assertEqual(cause["classification"],
                         "zero_policy_mass_json_serialization_defect")
        self.assertFalse(cause["efficacy_related"])
        self.assertEqual(cause["producer"]["artifact_id"], 9707005417)
        self.assertEqual(cause["producer"]["invalid_line"], 2)
        self.assertEqual(cause["producer"]["invalid_column"], 10668)
        self.assertIn("0/0", cause["mechanism"])
        self.assertEqual(failure["job_disposition"], {
            "failure": 1, "skipped": 9, "success": 225,
            "total": 235, "train_evaluate_success": 216,
        })
        self.assertTrue(
            failure["fixed_seed_execution"]["all_policy_cost_v6_fixed_roots_retired"])
        self.assertEqual(failure["fixed_seed_execution"]["retired_fixed_root_count"], 21)
        self.assertEqual(failure["execution"]["sha256"],
                         digest(ROOT / failure["execution"]["path"]))
        self.assertEqual(retain["artifact_count"], 228)
        self.assertEqual(retain["train_evaluation_artifacts"]["count"], 216)
        self.assertFalse(retain["promotion_gate_passed"])
        self.assertFalse(retain["maintained_actor_changed"])

    def test_v6_launch_is_the_unique_direct_addendum(self) -> None:
        parent = subprocess.check_output(
            ["git", "rev-parse", f"{V6_LAUNCH}^"], cwd=ROOT, text=True).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{V6_LAUNCH}^{{tree}}"], cwd=ROOT,
            text=True).strip()
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", V6_LAUNCH],
            cwd=ROOT, text=True).strip()
        self.assertEqual(parent, V6_PARENT)
        self.assertEqual(tree, V6_LAUNCH_TREE)
        self.assertEqual(changed,
                         "A\tdata/experiments/locked_policy_cost_v6_execution.json")

    def test_every_v6_campaign_byte_is_immutable(self) -> None:
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", V6_LAUNCH],
            cwd=ROOT, text=True).splitlines()
        prefixes = (
            ".github/workflows/policy-cost-v", "data/experiments/locked_policy_cost_v",
            "data/experiments/policy_cost_v", "src/policy_cost",
            "tests/test_policy_cost", "tools/policy_cost",
        )
        frozen = [path for path in paths if path.startswith(prefixes) or
                  path == "tools/build_policy_cost.c" or
                  re.fullmatch(r"POLICY_COST_V[1-6]\.md", path)]
        self.assertIn("tools/policy_cost_dataset_v6.c", frozen)
        for relative in frozen:
            current = ROOT / relative
            self.assertTrue(current.is_file(), relative)
            expected = subprocess.check_output(
                ["git", "show", f"{V6_LAUNCH}:{relative}"], cwd=ROOT)
            self.assertEqual(current.read_bytes(), expected, relative)

    def test_all_twenty_one_v7_roots_are_fresh_and_distinct(self) -> None:
        plans = [load(ROOT / f"data/experiments/locked_policy_cost_v{i}_plan.json")
                 for i in range(1, 8)]
        roots = active_roots(plans[-1])
        previous = set().union(*(exact_seed_values(plan) for plan in plans[:-1]))
        self.assertEqual(len(roots), 21)
        self.assertTrue(all(root.startswith("202705") for root in roots))
        self.assertFalse(roots & previous)
        burned = plans[-1]["reservoirs"]["burned_source_seeds"]
        self.assertIn("all policy-cost-v6 fixed seeds", burned)
        self.assertIn("every 20270429 feasibility-smoke seed", burned)

    def test_v7_preserves_every_efficacy_design_and_gate(self) -> None:
        v6, v7 = load(V6_PLAN), load(V7_PLAN)
        cal6, cal7 = copy.deepcopy(v6["calibration"]), copy.deepcopy(v7["calibration"])
        cal6["schedule_seed"] = cal7["schedule_seed"]
        cal6["cross_validation"]["seed"] = cal7["cross_validation"]["seed"]
        self.assertEqual(cal6, cal7)
        ctl6, ctl7 = copy.deepcopy(v6["controller"]), copy.deepcopy(v7["controller"])
        ctl6["artifact_binding"]["source_seed"] = ctl7["artifact_binding"]["source_seed"]
        self.assertEqual(ctl6, ctl7)
        self.assertEqual(v6["multiplicity"], v7["multiplicity"])
        self.assertEqual(v6["prerequisite"], v7["prerequisite"])
        self.assertEqual(v6["test"], v7["test"])
        for stage in ("safety", "final"):
            self.assertEqual(without(v6[stage], "seeds"), without(v7[stage], "seeds"))
        self.assertEqual(without(v6["selection"], "bootstrap"),
                         without(v7["selection"], "bootstrap"))
        self.assertEqual(without(v6["selection"]["bootstrap"], "seed"),
                         without(v7["selection"]["bootstrap"], "seed"))
        for key in ("allocation", "candidate_masks",
                    "native_reservoir_origin_proof"):
            self.assertEqual(v6["reservoirs"][key], v7["reservoirs"][key])
        self.assertEqual(seed_agnostic(v6["reservoirs"]["search_and_truth"]),
                         seed_agnostic(v7["reservoirs"]["search_and_truth"]))
        self.assertEqual(
            without(v6["reservoirs"]["discovery"], "seeds", "sealed_smokes"),
            without(v7["reservoirs"]["discovery"], "seeds", "sealed_smokes"))

    def test_zero_mass_repair_is_narrow_and_has_exact_native_regression(self) -> None:
        old = (ROOT / "tools/policy_cost_dataset_v6.c").read_text(encoding="ascii")
        new = (ROOT / "tools/policy_cost_dataset_v7.c").read_text(encoding="ascii")
        self.assertIn("*action=a;*draw=j/a;", old)
        self.assertNotIn("*action=a;*draw=j/a;", new)
        self.assertIn("*action=a;*draw=a>0.0?j/a:0.0;", new)
        self.assertIn("zero_prob[2]={0.0f,0.0f}", new)
        self.assertIn("validate-jsonl", V7_WORKFLOW.read_text(encoding="utf-8"))

    def test_producer_boundary_rejects_v6_nonfinite_json(self) -> None:
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
        self.assertFalse(V7_EXECUTION.exists())
        workflow = yaml.safe_load(V7_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(workflow[True]["push"]["paths"],
                         ["data/experiments/locked_policy_cost_v7_execution.json"])
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])
        text = V7_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("workflow_dispatch", text)
        self.assertIn("test \"$GITHUB_RUN_ATTEMPT\" = 1", text)
        self.assertIn("tools/policy_cost_dataset_v7.c", text)
        self.assertIn("tools/policy_cost_campaign_v7.py", text)
        self.assertIn("locked_policy_cost_v6_execution.json", text)
        self.assertIn("policy_cost_v6_run_33213087155_failure.json", text)
        self.assertIn("bin/policy_cost_dataset self-test", text)
        definition = V7_DEFINITION.read_text(encoding="utf-8")
        self.assertIn("tools/policy_cost_dataset_v7.c", definition)
        self.assertIn("tests/test_policy_cost_v7_recovery.py", definition)


if __name__ == "__main__":
    unittest.main()
