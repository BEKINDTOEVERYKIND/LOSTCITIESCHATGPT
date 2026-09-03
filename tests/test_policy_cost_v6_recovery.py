"""Fail-closed contracts for the fresh-seed policy-cost-v6 recovery."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest

import yaml
from tools import policy_cost_allocate_v6 as allocator


ROOT = Path(__file__).resolve().parents[1]
V5_LAUNCH = "bf113e972c0e3f10af6e39e31dd8f19dea691812"
V5_PARENT = "10ad3e270b45cb4a85ab4f8c079af82d8a5c1416"
V5_LAUNCH_TREE = "3e256ac7ec3ea5b1cf0c03b87789535e25f46a06"
V5_PLAN = ROOT / "data/experiments/locked_policy_cost_v5_plan.json"
V6_PLAN = ROOT / "data/experiments/locked_policy_cost_v6_plan.json"
V6_EXECUTION = ROOT / "data/experiments/locked_policy_cost_v6_execution.json"
V6_WORKFLOW = ROOT / ".github/workflows/policy-cost-v6.yml"
V6_DEFINITION = ROOT / ".github/workflows/policy-cost-v6-definition.yml"
V5_FAILURE = ROOT / "data/experiments/policy_cost_v5_run_33176041169_failure.json"
V5_RETAIN = ROOT / "data/experiments/policy_cost_v5_infrastructure_retain.json"


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


class PolicyCostV6RecoveryTests(unittest.TestCase):
    def test_v5_terminal_record_is_complete_and_non_promotable(self) -> None:
        failure, retain = load(V5_FAILURE), load(V5_RETAIN)
        self.assertEqual(failure["schema"],
                         "lc-policy-cost-v5-numeric-serialization-failure-v1")
        self.assertEqual(failure["status"], "complete_terminal_non_promotable")
        self.assertFalse(failure["efficacy_data_observed"])
        self.assertFalse(failure["locked_validation_relaxed"])
        self.assertEqual(failure["github_run"], {
            "attempt": 1, "conclusion": "failure", "event": "push",
            "head_sha": V5_LAUNCH, "head_tree": V5_LAUNCH_TREE,
            "id": 33176041169,
            "url": "https://github.com/BEKINDTOEVERYKIND/"
                   "LOSTCITIESCHATGPT/actions/runs/33176041169",
        })
        cause = failure["cause"]
        self.assertEqual(cause["classification"],
                         "floating_point_variance_serialization_defect")
        self.assertFalse(cause["efficacy_related"])
        self.assertEqual(cause["producer"]["artifact_id"], 9690616686)
        self.assertEqual(cause["producer"]["invalid_line"], 17)
        self.assertEqual(cause["producer"]["invalid_column"], 8973)
        self.assertIn("-nan", cause["mechanism"])
        self.assertEqual(failure["job_disposition"], {
            "failure": 1, "skipped": 9, "success": 225,
            "total": 235, "train_evaluate_success": 216,
        })
        self.assertTrue(
            failure["fixed_seed_execution"]["all_policy_cost_v5_fixed_roots_retired"])
        self.assertEqual(failure["fixed_seed_execution"]["retired_fixed_root_count"], 21)
        self.assertEqual(failure["execution"]["sha256"],
                         digest(ROOT / failure["execution"]["path"]))
        self.assertEqual(retain["artifact_count"], 228)
        self.assertEqual(retain["train_evaluation_artifacts"]["count"], 216)
        self.assertFalse(retain["promotion_gate_passed"])
        self.assertFalse(retain["maintained_actor_changed"])

    def test_v5_launch_is_the_unique_direct_addendum(self) -> None:
        parent = subprocess.check_output(
            ["git", "rev-parse", f"{V5_LAUNCH}^"], cwd=ROOT, text=True).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{V5_LAUNCH}^{{tree}}"], cwd=ROOT,
            text=True).strip()
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", V5_LAUNCH],
            cwd=ROOT, text=True).strip()
        self.assertEqual(parent, V5_PARENT)
        self.assertEqual(tree, V5_LAUNCH_TREE)
        self.assertEqual(changed,
                         "A\tdata/experiments/locked_policy_cost_v5_execution.json")

    def test_every_v5_campaign_byte_is_immutable(self) -> None:
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", V5_LAUNCH],
            cwd=ROOT, text=True).splitlines()
        prefixes = (
            ".github/workflows/policy-cost-v", "data/experiments/locked_policy_cost_v",
            "data/experiments/policy_cost_v", "src/policy_cost",
            "tests/test_policy_cost", "tools/policy_cost",
        )
        frozen = [path for path in paths if path.startswith(prefixes) or
                  path == "tools/build_policy_cost.c" or
                  re.fullmatch(r"POLICY_COST_V[1-5]\.md", path)]
        self.assertIn("tools/policy_cost_dataset_v5.c", frozen)
        for relative in frozen:
            current = ROOT / relative
            self.assertTrue(current.is_file(), relative)
            expected = subprocess.check_output(
                ["git", "show", f"{V5_LAUNCH}:{relative}"], cwd=ROOT)
            self.assertEqual(current.read_bytes(), expected, relative)

    def test_all_twenty_one_v6_roots_are_fresh_and_distinct(self) -> None:
        plans = [load(ROOT / f"data/experiments/locked_policy_cost_v{i}_plan.json")
                 for i in range(1, 7)]
        roots = active_roots(plans[-1])
        previous = set().union(*(exact_seed_values(plan) for plan in plans[:-1]))
        self.assertEqual(len(roots), 21)
        self.assertTrue(all(root.startswith("202704") for root in roots))
        self.assertFalse(roots & previous)
        burned = plans[-1]["reservoirs"]["burned_source_seeds"]
        self.assertIn("all policy-cost-v5 fixed seeds", burned)
        self.assertIn("every 20270329 feasibility-smoke seed", burned)

    def test_v6_preserves_every_efficacy_design_and_gate(self) -> None:
        v5, v6 = load(V5_PLAN), load(V6_PLAN)
        cal5, cal6 = copy.deepcopy(v5["calibration"]), copy.deepcopy(v6["calibration"])
        cal5["schedule_seed"] = cal6["schedule_seed"]
        cal5["cross_validation"]["seed"] = cal6["cross_validation"]["seed"]
        self.assertEqual(cal5, cal6)
        ctl5, ctl6 = copy.deepcopy(v5["controller"]), copy.deepcopy(v6["controller"])
        ctl5["artifact_binding"]["source_seed"] = ctl6["artifact_binding"]["source_seed"]
        self.assertEqual(ctl5, ctl6)
        self.assertEqual(v5["multiplicity"], v6["multiplicity"])
        self.assertEqual(v5["prerequisite"], v6["prerequisite"])
        self.assertEqual(v5["test"], v6["test"])
        for stage in ("safety", "final"):
            self.assertEqual(without(v5[stage], "seeds"), without(v6[stage], "seeds"))
        self.assertEqual(without(v5["selection"], "bootstrap"),
                         without(v6["selection"], "bootstrap"))
        self.assertEqual(without(v5["selection"]["bootstrap"], "seed"),
                         without(v6["selection"]["bootstrap"], "seed"))
        for key in ("allocation", "candidate_masks",
                    "native_reservoir_origin_proof"):
            self.assertEqual(v5["reservoirs"][key], v6["reservoirs"][key])
        self.assertEqual(seed_agnostic(v5["reservoirs"]["search_and_truth"]),
                         seed_agnostic(v6["reservoirs"]["search_and_truth"]))
        self.assertEqual(
            without(v5["reservoirs"]["discovery"], "seeds", "sealed_smokes"),
            without(v6["reservoirs"]["discovery"], "seeds", "sealed_smokes"))

    def test_numeric_repair_is_narrow_and_has_exact_native_regression(self) -> None:
        old = (ROOT / "tools/policy_cost_dataset_v5.c").read_text(encoding="ascii")
        new = (ROOT / "tools/policy_cost_dataset_v6.c").read_text(encoding="ascii")
        self.assertIn("*sumsq-(double)n*(*mean)*(*mean)", old)
        self.assertNotIn("*sumsq-(double)n*(*mean)*(*mean)", new)
        self.assertIn("centered+=residual*residual", new)
        self.assertIn("-1.6298145055770874e-9", new)
        self.assertIn("constant[512]", new)
        self.assertIn("!isfinite(se)", new)

    def test_definition_is_inert_and_launch_is_fail_closed(self) -> None:
        self.assertFalse(V6_EXECUTION.exists())
        workflow = yaml.safe_load(V6_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(workflow[True]["push"]["paths"],
                         ["data/experiments/locked_policy_cost_v6_execution.json"])
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])
        text = V6_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("workflow_dispatch", text)
        self.assertIn("test \"$GITHUB_RUN_ATTEMPT\" = 1", text)
        self.assertIn("tools/policy_cost_dataset_v6.c", text)
        self.assertIn("tools/policy_cost_campaign_v6.py", text)
        self.assertIn("locked_policy_cost_v5_execution.json", text)
        self.assertIn("policy_cost_v5_run_33176041169_failure.json", text)
        self.assertIn("bin/policy_cost_dataset self-test", text)
        definition = V6_DEFINITION.read_text(encoding="utf-8")
        self.assertIn("tools/policy_cost_dataset_v6.c", definition)
        self.assertIn("tests/test_policy_cost_v6_recovery.py", definition)


if __name__ == "__main__":
    unittest.main()
