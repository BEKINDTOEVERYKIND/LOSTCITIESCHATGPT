"""Fail-closed contracts for the fresh-seed policy-cost-v3 recovery."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
V1_PLAN = ROOT / "data/experiments/locked_policy_cost_v1_plan.json"
V2_PLAN = ROOT / "data/experiments/locked_policy_cost_v2_plan.json"
V3_PLAN = ROOT / "data/experiments/locked_policy_cost_v3_plan.json"
V3_EXECUTION = ROOT / "data/experiments/locked_policy_cost_v3_execution.json"
V3_WORKFLOW = ROOT / ".github/workflows/policy-cost-v3.yml"
V3_DEFINITION_WORKFLOW = \
    ROOT / ".github/workflows/policy-cost-v3-definition.yml"
V2_FAILURE = \
    ROOT / "data/experiments/policy_cost_v2_run_33095561493_failure.json"
V2_RETAIN = ROOT / "data/experiments/policy_cost_v2_infrastructure_retain.json"

V2_LAUNCH = "cdb03279707ad927daa4159dfdaa65d281e8352d"
V2_PARENT = "e15cede7f8fa36c817728710872718df4c1235d4"
V2_PARENT_TREE = "38241f861fd6e8ee7be463db2588f4f7048c5a47"
V2_LAUNCH_TREE = "47da55ec4bd1d352a25edac0dbc663debd6edd6e"

# These are immutable bytes that existed before the v3 recovery was defined.
# V3 is additive: it may bind its predecessors, but may never repair them in
# place after their one-shot executions.
IMMUTABLE_PREDECESSOR_SHA256 = {
    ".github/workflows/policy-cost-v1.yml":
        "c051e1c7fcf55950ad21e33bdecbe5ca68b807b022c8e61397600af6c78edfab",
    ".github/workflows/policy-cost-v2.yml":
        "2cce0eda608cf36024757bf0839ff589694cf92889288c84fb463504f026e5f4",
    "data/experiments/locked_policy_cost_v1_plan.json":
        "ebde80f8ce87058d833b7456cee21046740c04a5930b8fb35ee8df085a4a9ec5",
    "data/experiments/locked_policy_cost_v1_execution.json":
        "7d12f55c8d8151189f1460f70c7835edd250d951ea40cecb6cb539aacae6e3de",
    "data/experiments/locked_policy_cost_v2_plan.json":
        "e165b26c6553068d07b79e627db994a376b2022b6b848d500d7e539ed73e6097",
    "data/experiments/locked_policy_cost_v2_execution.json":
        "9474f9302941392ff6e77aa7ad87bb484d9648fc5887a88d68a6b05c82c282ce",
    "tools/policy_cost_campaign.py":
        "95707a80018bcf3517c8a332dac13a0efa59510482d3236532c7e0c864c03228",
    "tools/policy_cost_campaign_v2.py":
        "815c60bda689b6ce55b22a663fa2d9b229d9390238de2525d13aef95a48da8ea",
    "tools/policy_cost_dataset.c":
        "451fa9db6ae39e66e1d2736ad9d27f74fa19fa1b33083cd4888c7e145bc3acb3",
    "tools/policy_cost_dataset_v2.c":
        "114e7e74effddf0a9b5586185b1bfb5e64637e5b678ef9c8d30cee7e43576142",
    "src/policy_cost.c":
        "5c8b2571a8b87b1ead0b1aee6cc8c7b19f3d5c7af6e8fdeee75b468e5c599d48",
    "src/policy_cost.h":
        "6d8fec0fbcd4dc08a389153209bdbd07c4d46c030f0b7c803e3ef9b2a51070e4",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_digest(relative: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT
    )
    return hashlib.sha256(payload).hexdigest()


def exact_seed_values(value: object) -> set[str]:
    """Collect seed fields without accidentally parsing retired-seed prose."""
    found: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            found.update(exact_seed_values(child))
    elif isinstance(value, list):
        for child in value:
            found.update(exact_seed_values(child))
    elif isinstance(value, str) and re.fullmatch(r"20\d{10}", value):
        found.add(value)
    return found


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
        roots.update(search[family][split]
                     for split in ("TRAIN", "SELECT", "TEST"))
    roots.update(search["truth"][split]["seed"]
                 for split in ("TRAIN", "SELECT", "TEST"))
    return roots


def without(mapping: dict, *keys: str) -> dict:
    value = copy.deepcopy(mapping)
    for key in keys:
        value.pop(key, None)
    return value


class PolicyCostV3RecoveryTests(unittest.TestCase):
    def test_live_v2_failure_record_is_complete_and_non_efficacy(self) -> None:
        failure = load(V2_FAILURE)
        retain = load(V2_RETAIN)
        self.assertEqual(failure["schema"],
                         "lc-policy-cost-v2-preflight-failure-v1")
        self.assertEqual(failure["status"], "complete_terminal_non_promotable")
        self.assertEqual(failure["artifact_count"], 0)
        self.assertFalse(failure["efficacy_data_observed"])
        self.assertFalse(failure["search_or_truth_label_jobs_started"])
        self.assertFalse(failure["locked_validation_relaxed"])
        self.assertEqual(failure["github_run"], {
            "attempt": 1,
            "conclusion": "failure",
            "event": "push",
            "head_sha": V2_LAUNCH,
            "head_tree": V2_LAUNCH_TREE,
            "id": 33095561493,
            "url": "https://github.com/BEKINDTOEVERYKIND/"
                   "LOSTCITIESCHATGPT/actions/runs/33095561493",
        })
        self.assertEqual(failure["source_parent"], {
            "commit": V2_PARENT, "tree": V2_PARENT_TREE,
        })
        computation = failure["campaign_computation"]
        self.assertTrue(computation["native_compilation_started"])
        self.assertTrue(computation["native_compilation_completed"])
        self.assertTrue(computation["exact17_export_started"])
        for key in (
            "arena_evaluation_started", "dataset_discovery_started",
            "efficacy_reducer_started", "policy_cost_build_started",
            "search_label_started", "transport_freeze_started",
            "truth_label_started",
        ):
            self.assertFalse(computation[key], key)
        cause = failure["cause"]
        self.assertEqual(
            cause["classification"],
            "host_dependent_compiled_binary_digest_in_canonical_exact17_manifest",
        )
        self.assertTrue(cause["immutable_exclusion_content_matched"])
        self.assertFalse(cause["source_semantics_or_exclusion_mismatch"])
        self.assertEqual(cause["offending_binding"],
                         "bindings.native_hash_probe.binary_sha256")
        retired = failure["fixed_seed_execution"]
        self.assertTrue(retired["all_policy_cost_v2_fixed_roots_retired"])
        self.assertFalse(retired["any_policy_cost_v2_fixed_seed_executed"])
        self.assertEqual(len(failure["job_disposition"]), 17)
        self.assertEqual(
            [(row["name"], row["conclusion"])
             for row in failure["job_disposition"] if row["conclusion"] != "skipped"],
            [("preflight", "failure")],
        )
        self.assertEqual(failure["execution"]["sha256"],
                         digest(ROOT / failure["execution"]["path"]))
        self.assertEqual(
            failure["maintained_actor_disposition"]["sha256"], digest(V2_RETAIN)
        )
        self.assertEqual(retain["schema"],
                         "lc-policy-cost-v2-infrastructure-retain-v1")
        self.assertEqual(retain["status"], "complete_terminal_non_promotable")
        self.assertFalse(retain["promotion_gate_passed"])
        self.assertFalse(retain["partial_evidence_retained"])
        self.assertEqual(retain["available_evidence"], [])

    def test_v2_launch_is_the_unique_direct_addendum_commit(self) -> None:
        parent = subprocess.check_output(
            ["git", "rev-parse", f"{V2_LAUNCH}^"], cwd=ROOT, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{V2_LAUNCH}^{{tree}}"], cwd=ROOT,
            text=True,
        ).strip()
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r",
             V2_LAUNCH], cwd=ROOT, text=True,
        ).strip()
        self.assertEqual(parent, V2_PARENT)
        self.assertEqual(tree, V2_LAUNCH_TREE)
        self.assertEqual(
            changed,
            "A\tdata/experiments/locked_policy_cost_v2_execution.json",
        )

    def test_v1_and_v2_campaign_bytes_are_immutable(self) -> None:
        for relative, expected in IMMUTABLE_PREDECESSOR_SHA256.items():
            with self.subTest(path=relative):
                # Campaign preflight deliberately overlays versioned v3 math
                # into canonical runtime paths in its detached worktree.  The
                # immutable authority is therefore the bound parent Git blob,
                # not that transient build-only overlay.
                self.assertEqual(git_blob_digest(relative), expected)

    def test_all_twenty_one_v3_roots_are_fresh_and_pairwise_distinct(self) -> None:
        v1, v2, v3 = load(V1_PLAN), load(V2_PLAN), load(V3_PLAN)
        roots = active_roots(v3)
        self.assertEqual(len(roots), 21)
        self.assertTrue(all(root.startswith("202701") for root in roots))
        self.assertFalse(roots & exact_seed_values(v1))
        self.assertFalse(roots & exact_seed_values(v2))
        self.assertEqual(
            roots,
            {root.replace("202612", "202701", 1)
             for root in active_roots(v2)},
        )

    def test_v3_preserves_the_v2_statistical_design_and_locked_gates(self) -> None:
        v2, v3 = load(V2_PLAN), load(V3_PLAN)
        calibration2 = copy.deepcopy(v2["calibration"])
        calibration3 = copy.deepcopy(v3["calibration"])
        calibration2["schedule_seed"] = calibration3["schedule_seed"]
        calibration2["cross_validation"]["seed"] = \
            calibration3["cross_validation"]["seed"]
        self.assertEqual(calibration3, calibration2)
        controller2 = copy.deepcopy(v2["controller"])
        controller3 = copy.deepcopy(v3["controller"])
        controller2["artifact_binding"]["source_seed"] = \
            controller3["artifact_binding"]["source_seed"]
        self.assertEqual(controller3, controller2)
        self.assertEqual(v3["multiplicity"], v2["multiplicity"])
        self.assertEqual(v3["prerequisite"], v2["prerequisite"])
        self.assertEqual(v3["test"], v2["test"])
        for stage in ("safety", "final"):
            self.assertEqual(without(v3[stage], "seeds"),
                             without(v2[stage], "seeds"))
        self.assertEqual(without(v3["selection"]["bootstrap"], "seed"),
                         without(v2["selection"]["bootstrap"], "seed"))
        self.assertEqual(without(v3["selection"], "bootstrap"),
                         without(v2["selection"], "bootstrap"))
        reservoirs2, reservoirs3 = v2["reservoirs"], v3["reservoirs"]
        for key in (
            "allocation", "allocation_output_order", "pre_efficacy_barrier",
            "native_reservoir_origin_proof", "candidate_masks", "ply_strata",
            "pooled_nply_ge_64", "ratio_bands", "select_and_test", "train",
        ):
            self.assertEqual(reservoirs3[key], reservoirs2[key], key)
        self.assertEqual(
            without(reservoirs3["discovery"], "seeds"),
            without(reservoirs2["discovery"], "seeds"),
        )
        search2 = reservoirs2["search_and_truth"]
        search3 = reservoirs3["search_and_truth"]
        for family in ("primary", "fresh"):
            self.assertEqual(search3[family]["worlds"], search2[family]["worlds"])
        for split in ("TRAIN", "SELECT", "TEST"):
            self.assertEqual(search3["truth"][split]["worlds"],
                             search2["truth"][split]["worlds"])
        self.assertEqual(search3["truth"]["controller"],
                         search2["truth"]["controller"])

    def test_definition_has_no_v3_execution_addendum(self) -> None:
        self.assertFalse(V3_EXECUTION.exists())
        template = ROOT / \
            "data/experiments/locked_policy_cost_v3_execution.template.json"
        self.assertTrue(template.is_file())
        self.assertNotEqual(template.name, V3_EXECUTION.name)

    def test_campaign_trigger_is_push_only_unique_addendum(self) -> None:
        text = V3_WORKFLOW.read_text(encoding="utf-8")
        workflow = yaml.load(text, Loader=yaml.BaseLoader)
        self.assertEqual(workflow["on"], {
            "push": {
                "branches": ["agent/correctness-and-policy-upgrade"],
                "paths": [
                    "data/experiments/locked_policy_cost_v3_execution.json"
                ],
            }
        })
        for forbidden in (
            "workflow_dispatch", "repository_dispatch", "pull_request:",
            "schedule:",
        ):
            self.assertNotIn(forbidden, text)
        for required in (
            'test "$GITHUB_EVENT_NAME" = push',
            'test "$GITHUB_RUN_ATTEMPT" = 1',
            'test "$FORCED" = false',
            'test "$BEFORE" = "$SOURCE_COMMIT"',
            "git diff-tree --no-commit-id --name-status -r HEAD",
            'test "$(git rev-list --all --count -- "$EXECUTION")" = 1',
            '! git cat-file -e "HEAD^:$EXECUTION"',
        ):
            self.assertIn(required, text)

    def test_dag_and_promotion_gates_are_unchanged_from_v2(self) -> None:
        v2 = yaml.safe_load((ROOT / ".github/workflows/policy-cost-v2.yml")
                            .read_text(encoding="utf-8"))["jobs"]
        v3 = yaml.safe_load(V3_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
        self.assertEqual(set(v3), set(v2))
        for name in v2:
            self.assertEqual(v3[name].get("needs"), v2[name].get("needs"), name)
        text = V3_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("--pairs-per-orientation 200", text)
        self.assertIn("--pairs-per-orientation 2500", text)
        self.assertGreaterEqual(text.count("--gate-z 1.645"), 2)
        self.assertIn("--mode safety", text)
        self.assertIn("--mode final", text)
        self.assertIn("--candidate-first-seed 202701210101", text)
        self.assertIn("--baseline-first-seed 202701210102", text)
        self.assertIn("--candidate-first-seed 202701220101", text)
        self.assertIn("--baseline-first-seed 202701220102", text)

    def test_runtime_is_compiled_once_then_all_downstream_jobs_are_source_free(self) -> None:
        workflow = yaml.safe_load(V3_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(workflow["env"]["POLICY_COST_COMPILE_ONCE"], "1")
        jobs = workflow["jobs"]
        self.assertEqual(
            sum("actions/checkout@" in str(step.get("uses", ""))
                for job in jobs.values() for step in job["steps"]), 1,
        )
        materialize = next(
            step["run"] for step in jobs["preflight"]["steps"]
            if step.get("name") == "Materialize and compile the bound parent once"
        )
        self.assertIn('git -C campaign worktree add --detach "$RUN_ROOT/source" HEAD^',
                      materialize)
        self.assertIn('cp "$POLICY_COST_SOURCE" src/policy_cost.c', materialize)
        self.assertIn('cp "$POLICY_COST_HEADER" src/policy_cost.h', materialize)
        self.assertEqual(materialize.count("make -j2 CC=gcc"), 1)
        self.assertIn(
            "gcc $CFLAGS_LOCKED -fno-fast-math -ffp-contract=off", materialize
        )
        for name, job in jobs.items():
            if name == "preflight":
                continue
            script = "\n".join(str(step.get("run", "")) for step in job["steps"])
            self.assertNotIn("actions/checkout", script, name)
            self.assertNotRegex(script, r"(^|[;&|]\s*)(gcc|make)\s", name)
        freeze = next(
            step["run"] for step in jobs["preflight"]["steps"]
            if step.get("name") ==
            "Freeze a source-free transport before any search or truth label"
        )
        self.assertIn("policy_cost_v2_infrastructure_retain.json", freeze)
        self.assertIn("policy_cost_v2_run_33095561493_failure.json", freeze)
        self.assertIn("locked_policy_cost_v2_execution.json", freeze)
        self.assertIn(
            "'cp src/policy_cost_v3.c src/policy_cost.c && cp "
            "src/policy_cost_v3.h src/policy_cost.h'", freeze,
        )
        self.assertIn("hashlib.sha256(path.read_bytes()).hexdigest()", freeze)
        self.assertIn("for relative in ('bin/arena', 'bin/build_policy_cost', "
                      "'bin/policy_cost_dataset')", freeze)

    def test_exact17_manifest_is_portable_and_fully_reproduced(self) -> None:
        evidence_path = ROOT / \
            "data/experiments/policy_cost_v3_exact17_exclusions.json"
        evidence = load(evidence_path)
        probe = evidence["bindings"]["native_hash_probe"]
        self.assertNotIn("binary_sha256", probe)
        self.assertEqual(
            probe["runtime_binary_binding"],
            "dynamically_sealed_in_build_identity_and_transport",
        )
        self.assertEqual(
            digest(evidence_path),
            "ad00b66446b5f20fe7bb9c108c3ec088c2d01a825e6b7d00c93ba06e59a6f1c0",
        )
        workflow = yaml.safe_load(V3_WORKFLOW.read_text(encoding="utf-8"))
        materialize = next(
            step["run"] for step in workflow["jobs"]["preflight"]["steps"]
            if step.get("name") == "Materialize and compile the bound parent once"
        )
        self.assertIn("policy_cost_exact17_v3.py", V3_WORKFLOW.read_text())
        self.assertIn("--hash-probe bin/policy_cost_dataset", materialize)
        self.assertIn(
            'cmp "$EXACT17_TMP/policy_cost_v3_exact17_exclusions.txt" "$EXCLUSIONS"',
            materialize,
        )
        self.assertIn(
            'cmp "$EXACT17_TMP/policy_cost_v3_exact17_exclusions.json" "$EXCLUSION_JSON"',
            materialize,
        )

    def test_definition_ci_uses_both_frozen_wheels_in_isolated_pythonpath(self) -> None:
        text = V3_DEFINITION_WORKFLOW.read_text(encoding="utf-8")
        workflow = yaml.safe_load(text)
        trigger = yaml.load(text, Loader=yaml.BaseLoader)["on"]
        execution = "data/experiments/locked_policy_cost_v3_execution.json"
        self.assertNotIn(execution, trigger["push"]["paths"])
        self.assertNotIn(execution, trigger["pull_request"]["paths"])
        steps = workflow["jobs"]["validate-definition"]["steps"]
        install = next(step["run"] for step in steps
                       if step.get("name") ==
                       "Materialize the exact frozen Python runtime")
        self.assertIn("--only-binary=:all: --require-hashes", install)
        self.assertIn("--no-deps --target python-runtime", install)
        self.assertIn("numpy-2.3.5-cp312", install)
        self.assertIn("pyyaml-6.0.3-cp312", install)
        self.assertIn(
            "0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28",
            install,
        )
        self.assertIn(
            "ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc",
            install,
        )
        validate = next(step["run"] for step in steps
                        if step.get("name") ==
                        "Validate the complete definition with the isolated runtime")
        self.assertIn('RUNTIME="$GITHUB_WORKSPACE/python-runtime"', validate)
        self.assertGreaterEqual(validate.count('PYTHONPATH="$RUNTIME"'), 5)
        self.assertIn("import numpy, yaml", validate)
        self.assertIn('yaml.__version__ == "6.0.3"', validate)


if __name__ == "__main__":
    unittest.main()
