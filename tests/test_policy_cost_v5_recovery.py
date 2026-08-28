"""Fail-closed contracts for the fresh-seed policy-cost-v5 recovery."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest

import yaml
from tools import policy_cost_allocate_v5 as allocator


ROOT = Path(__file__).resolve().parents[1]
V1_PLAN = ROOT / "data/experiments/locked_policy_cost_v1_plan.json"
V2_PLAN = ROOT / "data/experiments/locked_policy_cost_v2_plan.json"
V3_PLAN = ROOT / "data/experiments/locked_policy_cost_v3_plan.json"
V4_PLAN = ROOT / "data/experiments/locked_policy_cost_v4_plan.json"
V5_PLAN = ROOT / "data/experiments/locked_policy_cost_v5_plan.json"
V5_EXECUTION = ROOT / "data/experiments/locked_policy_cost_v5_execution.json"
V5_WORKFLOW = ROOT / ".github/workflows/policy-cost-v5.yml"
V5_DEFINITION_WORKFLOW = \
    ROOT / ".github/workflows/policy-cost-v5-definition.yml"
V4_FAILURE = \
    ROOT / "data/experiments/policy_cost_v4_run_33158323330_failure.json"
V4_RETAIN = ROOT / "data/experiments/policy_cost_v4_infrastructure_retain.json"

V4_LAUNCH = "6306dd8fba165c60529b7b2e7363cbbf3cc2a8f3"
V4_PARENT = "7b6317fabdc55b62841c7ef78c4cdb93c103876b"
V4_PARENT_TREE = "cd9068172153d5be294d2e2facd3fcd188b51ba3"
V4_LAUNCH_TREE = "7c82ba399e24ae6941a6cd81b833d681fffaaaf5"

# These are immutable bytes that existed before the v5 recovery was defined.
# V5 is additive: it may bind its predecessors, but may never repair them in
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
    ".github/workflows/policy-cost-v3.yml":
        "a5e283188910419ed607a9ec8a0deef3704525d8cc1704fc7748422f99b48dc2",
    ".github/workflows/policy-cost-v3-definition.yml":
        "03f34eb6ad5b0f113ad041b0983a04a94f03b71b11809b10f461767d61bd5048",
    "data/experiments/locked_policy_cost_v3_plan.json":
        "1706f0093e81625e9093647e5ced0b6f4f8eaa946235300023b3773f82ec9ce2",
    "data/experiments/locked_policy_cost_v3_execution.json":
        "5ffcf52909e9503b1e13fb44bdb7346e064e401715d9bcb560fcad5ae1119085",
    "tools/policy_cost_allocate_v3.py":
        "7702c3f673c0400e2dfbdc991a5e1ef41945ba72070365f7333bb5eec7c42d0c",
    "tools/policy_cost_campaign_v3.py":
        "13dc4b42486f5c8843b6a19da887a869ba1760ae206527b4911eb3b79c621c08",
    "tools/policy_cost_dataset_v3.c":
        "a35cfe810bbf6f9a6cc0039f1ea3164849c1441f1de73d0efcb7846b9953d62b",
    "src/policy_cost_v3.c":
        "283907c0e3b844385e7ec47d813e0a0c205de641afa03db06a9e0f24960cc4b0",
    "src/policy_cost_v3.h":
        "9a2aa11eb7373eb8a315bdaba6c29062cef28c2e276b253ce8512189cc9a012d",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_digest(relative: str) -> str:
    payload = subprocess.check_output(
        ["git", "show", f"{V4_LAUNCH}:{relative}"], cwd=ROOT
    )
    return hashlib.sha256(payload).hexdigest()


def frozen_predecessor_paths() -> list[str]:
    """Enumerate every policy-cost campaign byte at the v4 launch."""
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", V4_LAUNCH],
        cwd=ROOT, text=True,
    ).splitlines()
    prefixes = (
        ".github/workflows/policy-cost-v",
        "data/experiments/locked_policy_cost_v",
        "data/experiments/policy_cost_v",
        "src/policy_cost",
        "tests/test_policy_cost",
        "tools/policy_cost",
    )
    return sorted(path for path in paths if path.startswith(prefixes) or
                  path == "tools/build_policy_cost.c" or
                  re.fullmatch(r"POLICY_COST_V[1234]\.md", path))


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


class PolicyCostV5RecoveryTests(unittest.TestCase):
    def test_fixed_vector_groups_are_complete_transitive_and_not_slot_sparse(self) -> None:
        rows: list[dict[str, str]] = []
        census: dict[str, int] = {}
        source = 0
        for rd, pb, frontier in allocator.vector_bases():
            for group, quota in enumerate(allocator.VECTOR_GROUP_QUOTAS):
                cell = f"r{rd}:p{pb:02d}:f{frontier}:g{group}"
                census[cell] = quota
                for rank in range(quota):
                    # The pooled tail retains the exact hashed slot identity;
                    # it is not collapsed into one evaluated candidate.
                    slot = group if group < 2 else 2 + rank % 3
                    rows.append({
                        "cell": cell,
                        "source_match_index": str(source),
                        "source_state_index": "0",
                        "allocation_slot": str(slot),
                    })
                    source += 1
        selected, quotas = allocator.allocate_vectors(rows, census)
        self.assertEqual(len(selected), 9216)
        self.assertEqual(set(quotas.values()), {21, 22})
        self.assertEqual(sum(value == 22 for value in quotas.values()), 144)
        self.assertEqual(
            {int(row["allocation_slot"]) for row in selected},
            {0, 1, 2, 3, 4},
        )

    def test_live_v4_failure_record_is_complete_and_non_efficacy(self) -> None:
        failure = load(V4_FAILURE)
        retain = load(V4_RETAIN)
        self.assertEqual(failure["schema"],
                         "lc-policy-cost-v4-allocation-contract-failure-v1")
        self.assertEqual(failure["status"], "complete_terminal_non_promotable")
        self.assertEqual(failure["artifact_count"], 9)
        self.assertEqual(failure["raw_discovery_or_reservoir_artifact_count"], 3)
        self.assertFalse(failure["efficacy_data_observed"])
        self.assertFalse(failure["search_or_truth_label_jobs_started"])
        self.assertFalse(failure["locked_validation_relaxed"])
        self.assertEqual(failure["github_run"], {
            "attempt": 1,
            "conclusion": "failure",
            "event": "push",
            "head_sha": V4_LAUNCH,
            "head_tree": V4_LAUNCH_TREE,
            "id": 33158323330,
            "url": "https://github.com/BEKINDTOEVERYKIND/"
                   "LOSTCITIESCHATGPT/actions/runs/33158323330",
        })
        self.assertEqual(failure["source_parent"], {
            "commit": V4_PARENT, "tree": V4_PARENT_TREE,
        })
        cause = failure["cause"]
        self.assertEqual(
            cause["classification"],
            "sparse_random_slot_poststratification_contract",
        )
        self.assertFalse(cause["efficacy_related"])
        self.assertIn("r0:p00:f1:j3", cause["select"]["error"])
        self.assertIn("r0:p00:f0:j4", cause["test"]["error"])
        retired = failure["fixed_seed_execution"]
        self.assertTrue(retired["all_policy_cost_v4_fixed_roots_retired"])
        self.assertEqual(retired["physically_executed_fixed_root_count"], 3)
        self.assertEqual(retired["retired_fixed_root_count"], 21)
        self.assertEqual(len(failure["job_disposition"]), 20)
        self.assertEqual(
            [(row["name"], row["conclusion"])
             for row in failure["job_disposition"] if row["conclusion"] != "skipped"],
            [("preflight", "success"), ("select_discover", "success"),
             ("test_discover", "success"), ("train_discover", "success"),
             ("select_allocate", "failure"), ("test_allocate", "failure"),
             ("train_allocate", "success"),
             ("infrastructure_retain", "success")],
        )
        self.assertEqual(failure["execution"]["sha256"],
                         git_blob_digest(failure["execution"]["path"]))
        self.assertEqual(
            failure["maintained_actor_disposition"]["objective3_result_sha256"],
            digest(ROOT / failure["maintained_actor_disposition"]["objective3_result_path"])
        )
        self.assertEqual(retain["schema"],
                         "lc-policy-cost-v4-infrastructure-retain-v1")
        self.assertEqual(retain["status"], "complete_terminal_non_promotable")
        self.assertFalse(retain["promotion_gate_passed"])
        self.assertFalse(retain["efficacy_data_present"])
        self.assertEqual(retain["artifact_count"], 9)

    def test_v4_launch_is_the_unique_direct_addendum_commit(self) -> None:
        parent = subprocess.check_output(
            ["git", "rev-parse", f"{V4_LAUNCH}^"], cwd=ROOT, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{V4_LAUNCH}^{{tree}}"], cwd=ROOT,
            text=True,
        ).strip()
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r",
             V4_LAUNCH], cwd=ROOT, text=True,
        ).strip()
        self.assertEqual(parent, V4_PARENT)
        self.assertEqual(tree, V4_LAUNCH_TREE)
        self.assertEqual(
            changed,
            "A\tdata/experiments/locked_policy_cost_v4_execution.json",
        )

    def test_critical_predecessor_campaign_hashes_are_immutable(self) -> None:
        for relative, expected in IMMUTABLE_PREDECESSOR_SHA256.items():
            with self.subTest(path=relative):
                # Campaign preflight deliberately overlays versioned v5 math
                # into canonical runtime paths in its detached worktree.  The
                # immutable authority is therefore the bound parent Git blob,
                # not that transient build-only overlay.
                self.assertEqual(git_blob_digest(relative), expected)

    def test_every_v1_through_v4_campaign_byte_is_frozen(self) -> None:
        paths = frozen_predecessor_paths()
        self.assertGreaterEqual(len(paths), 80)
        self.assertIn("tools/policy_cost_artifact_v4.py", paths)
        self.assertIn("tools/policy_cost_calibration_v4.py", paths)
        self.assertIn("tools/policy_cost_selection_v4.py", paths)
        self.assertIn("tools/policy_cost_exact17_v4.py", paths)
        self.assertIn("tests/test_policy_cost_v4_recovery.py", paths)
        self.assertIn("POLICY_COST_V4.md", paths)
        self.assertIn(
            "data/experiments/locked_policy_cost_v4_execution.template.json",
            paths,
        )
        for relative in paths:
            with self.subTest(path=relative):
                path = ROOT / relative
                if path.is_file():
                    self.assertEqual(digest(path), git_blob_digest(relative))
                else:
                    self.fail(f"frozen predecessor byte is absent: {relative}")

    def test_all_twenty_one_v5_roots_are_fresh_and_pairwise_distinct(self) -> None:
        v1, v2, v3, v4, v5 = (load(V1_PLAN), load(V2_PLAN), load(V3_PLAN),
                              load(V4_PLAN), load(V5_PLAN))
        roots = active_roots(v5)
        self.assertEqual(len(roots), 21)
        self.assertTrue(all(root.startswith("202703") for root in roots))
        self.assertFalse(roots & exact_seed_values(v1))
        self.assertFalse(roots & exact_seed_values(v2))
        self.assertFalse(roots & exact_seed_values(v3))
        self.assertFalse(roots & exact_seed_values(v4))
        self.assertEqual(
            roots,
            {root.replace("202702", "202703", 1)
             for root in active_roots(v4)},
        )
        burned = v5["reservoirs"]["burned_source_seeds"]
        for binding in (
            "202701010101",
            "20270110/11/12/13/14/15/16/21/22",
            "every 20270129 feasibility-smoke seed",
            "202702010101",
            "20270210/11/12/13/14/15/16/21/22",
            "every 20270229 feasibility-smoke seed",
            "202703010101",
            "every 20270329 feasibility-smoke seed",
        ):
            self.assertIn(binding, burned)

    def test_v5_preserves_v4_efficacy_design_and_locked_gates(self) -> None:
        v4, v5 = load(V4_PLAN), load(V5_PLAN)
        calibration3 = copy.deepcopy(v4["calibration"])
        calibration4 = copy.deepcopy(v5["calibration"])
        calibration3["schedule_seed"] = calibration4["schedule_seed"]
        calibration3["cross_validation"]["seed"] = \
            calibration4["cross_validation"]["seed"]
        self.assertEqual(calibration4, calibration3)
        controller3 = copy.deepcopy(v4["controller"])
        controller4 = copy.deepcopy(v5["controller"])
        controller3["artifact_binding"]["source_seed"] = \
            controller4["artifact_binding"]["source_seed"]
        self.assertEqual(controller4, controller3)
        self.assertEqual(v5["multiplicity"], v4["multiplicity"])
        self.assertEqual(v5["prerequisite"], v4["prerequisite"])
        self.assertEqual(v5["test"], v4["test"])
        for stage in ("safety", "final"):
            self.assertEqual(without(v5[stage], "seeds"),
                             without(v4[stage], "seeds"))
        self.assertEqual(without(v5["selection"]["bootstrap"], "seed"),
                         without(v4["selection"]["bootstrap"], "seed"))
        self.assertEqual(without(v5["selection"], "bootstrap"),
                         without(v4["selection"], "bootstrap"))
        reservoirs3, reservoirs4 = v4["reservoirs"], v5["reservoirs"]
        for key in (
            "allocation_output_order", "pre_efficacy_barrier",
            "native_reservoir_origin_proof", "candidate_masks", "ply_strata",
            "pooled_nply_ge_64", "ratio_bands", "train",
        ):
            self.assertEqual(reservoirs4[key], reservoirs3[key], key)
        self.assertEqual(reservoirs4["select_and_test"]["records_per_split"],
                         reservoirs3["select_and_test"]["records_per_split"])
        self.assertEqual(reservoirs4["select_and_test"]["base_cells"], 144)
        self.assertIn("22/21/21", reservoirs4["select_and_test"]["allocation"])
        self.assertEqual(
            without(reservoirs4["discovery"], "seeds",
                    "atomic_persistence_before_allocation"),
            without(reservoirs3["discovery"], "seeds",
                    "atomic_persistence_before_allocation"),
        )
        search3 = reservoirs3["search_and_truth"]
        search4 = reservoirs4["search_and_truth"]
        for family in ("primary", "fresh"):
            self.assertEqual(search4[family]["worlds"], search3[family]["worlds"])
        for split in ("TRAIN", "SELECT", "TEST"):
            self.assertEqual(search4["truth"][split]["worlds"],
                             search3["truth"][split]["worlds"])
        self.assertEqual(search4["truth"]["controller"],
                         search3["truth"]["controller"])

    def test_definition_has_no_v5_execution_addendum(self) -> None:
        self.assertFalse(V5_EXECUTION.exists())
        template = ROOT / \
            "data/experiments/locked_policy_cost_v5_execution.template.json"
        self.assertTrue(template.is_file())
        self.assertNotEqual(template.name, V5_EXECUTION.name)

    def test_campaign_trigger_is_push_only_unique_addendum(self) -> None:
        text = V5_WORKFLOW.read_text(encoding="utf-8")
        workflow = yaml.load(text, Loader=yaml.BaseLoader)
        self.assertEqual(workflow["on"], {
            "push": {
                "branches": ["agent/correctness-and-policy-upgrade"],
                "paths": [
                    "data/experiments/locked_policy_cost_v5_execution.json"
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

    def test_semantic_dag_and_promotion_gates_are_unchanged_from_v4(self) -> None:
        v3 = yaml.safe_load((ROOT / ".github/workflows/policy-cost-v4.yml")
                            .read_text(encoding="utf-8"))["jobs"]
        v5 = yaml.safe_load(V5_WORKFLOW.read_text(encoding="utf-8"))["jobs"]

        allocate_to_discover: dict[str, str] = {}

        def normalized_needs(name: str) -> object:
            needs = v5[name].get("needs")
            if isinstance(needs, str):
                return allocate_to_discover.get(needs, needs)
            if isinstance(needs, list):
                result = []
                for dependency in needs:
                    dependency = allocate_to_discover.get(
                        dependency, dependency)
                    if dependency not in result:
                        result.append(dependency)
                return result
            return needs

        self.assertEqual(set(v5), set(v3))
        for allocation, discovery in allocate_to_discover.items():
            self.assertEqual(v5[allocation].get("needs"), discovery)
        for name in v3:
            self.assertEqual(normalized_needs(name), v3[name].get("needs"), name)

        text = V5_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("--pairs-per-orientation 200", text)
        self.assertIn("--pairs-per-orientation 2500", text)
        self.assertGreaterEqual(text.count("--gate-z 1.645"), 2)
        self.assertIn("--mode safety", text)
        self.assertIn("--mode final", text)
        self.assertIn("--candidate-first-seed 202703210101", text)
        self.assertIn("--baseline-first-seed 202703210102", text)
        self.assertIn("--candidate-first-seed 202703220101", text)
        self.assertIn("--baseline-first-seed 202703220102", text)

    def test_runtime_is_compiled_once_then_all_downstream_jobs_are_source_free(self) -> None:
        workflow = yaml.safe_load(V5_WORKFLOW.read_text(encoding="utf-8"))
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
        self.assertIn("policy_cost_v4_infrastructure_retain.json", freeze)
        self.assertIn("policy_cost_v4_run_33158323330_failure.json", freeze)
        self.assertIn("locked_policy_cost_v4_execution.json", freeze)
        self.assertIn(
            "'cp src/policy_cost_v5.c src/policy_cost.c && cp "
            "src/policy_cost_v5.h src/policy_cost.h'", freeze,
        )
        self.assertIn("hashlib.sha256(path.read_bytes()).hexdigest()", freeze)
        self.assertIn("for relative in ('bin/arena', 'bin/build_policy_cost', "
                      "'bin/policy_cost_dataset')", freeze)

    def test_exact17_manifest_is_portable_and_fully_reproduced(self) -> None:
        evidence_path = ROOT / \
            "data/experiments/policy_cost_v5_exact17_exclusions.json"
        evidence = load(evidence_path)
        plan = load(V5_PLAN)
        self.assertEqual(
            plan["v4_recovery"]["exact17_canonical_payload_sha256"],
            evidence["canonical_payload_sha256"],
        )
        probe = evidence["bindings"]["native_hash_probe"]
        self.assertNotIn("binary_sha256", probe)
        self.assertEqual(
            probe["runtime_binary_binding"],
            "dynamically_sealed_in_build_identity_and_transport",
        )
        self.assertEqual(
            digest(evidence_path),
            "703bbb7b927db893a091ec9435aad00fb9591f8d9618f78423f674432ab2b45d",
        )
        workflow = yaml.safe_load(V5_WORKFLOW.read_text(encoding="utf-8"))
        materialize = next(
            step["run"] for step in workflow["jobs"]["preflight"]["steps"]
            if step.get("name") == "Materialize and compile the bound parent once"
        )
        self.assertIn("policy_cost_exact17_v5.py", V5_WORKFLOW.read_text())
        self.assertIn("--hash-probe bin/policy_cost_dataset", materialize)
        self.assertIn(
            'cmp "$EXACT17_TMP/policy_cost_v5_exact17_exclusions.txt" "$EXCLUSIONS"',
            materialize,
        )
        self.assertIn(
            'cmp "$EXACT17_TMP/policy_cost_v5_exact17_exclusions.json" "$EXCLUSION_JSON"',
            materialize,
        )

    def test_definition_ci_uses_both_frozen_wheels_in_isolated_pythonpath(self) -> None:
        text = V5_DEFINITION_WORKFLOW.read_text(encoding="utf-8")
        workflow = yaml.safe_load(text)
        trigger = yaml.load(text, Loader=yaml.BaseLoader)["on"]
        execution = "data/experiments/locked_policy_cost_v5_execution.json"
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

    def test_both_preflights_exercise_real_v5_producer_consumer_contract(self) -> None:
        for path in (V5_WORKFLOW, V5_DEFINITION_WORKFLOW):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                for split, seed in (
                    ("TRAIN", "202703290101"),
                    ("SELECT", "202703290201"),
                    ("TEST", "202703290301"),
                ):
                    self.assertIn(f"{split}:{seed}", text)
                self.assertIn("bin/policy_cost_dataset discover", text)
                self.assertIn(
                    "tools/policy_cost_allocate_v5.py --validate-contract-only",
                    text,
                )
                self.assertIn("six-bin-contract-discovery.jsonl", text)
                self.assertIn("unequal-contract-discovery.jsonl", text)
                self.assertIn("union_width_counts'][source]-=1", text)
                self.assertIn("union_width_counts'][target]+=1", text)
                self.assertIn(
                    "negative producer/consumer contract mutation was accepted",
                    text,
                )

    def test_discovery_and_allocation_attempts_persist_before_enforcement(self) -> None:
        workflow = yaml.safe_load(V5_WORKFLOW.read_text(encoding="utf-8"))
        jobs = workflow["jobs"]
        for split in ("train", "select", "test"):
            with self.subTest(split=split):
                discovery_steps = jobs[f"{split}_discover"]["steps"]
                capture = next(
                    step for step in discovery_steps
                    if step.get("name") ==
                    f"Capture the atomic {split.upper()} producer disposition"
                )
                raw_upload = next(
                    step for step in discovery_steps
                    if step.get("with", {}).get("name") ==
                    f"policy-cost-v5-{split}-raw-discovery"
                )
                enforce = next(
                    step for step in discovery_steps
                    if step.get("name") ==
                    f"Enforce successful atomic {split.upper()} generation after persistence"
                )
                self.assertTrue(capture["continue-on-error"])
                self.assertEqual(raw_upload["if"], "always()")
                self.assertEqual(enforce["if"], "always()")
                self.assertLess(discovery_steps.index(capture),
                                discovery_steps.index(raw_upload))
                self.assertLess(discovery_steps.index(raw_upload),
                                discovery_steps.index(enforce))
                capture_script = capture["run"]
                for field in (
                    "schema", "split", "fixed_root_seed", "requested_matches",
                    "github_run_id", "github_run_attempt", "head_sha",
                    "producer_exit_code", "atomic_pair_complete", "footer",
                    "sha256", "size",
                ):
                    self.assertIn(field, capture_script)

                allocation_steps = jobs[f"{split}_allocate"]["steps"]
                allocation_capture = next(
                    step for step in allocation_steps
                    if step.get("name") ==
                    f"Capture independent {split.upper()} allocation and origin-proof disposition"
                )
                attempt_upload = next(
                    step for step in allocation_steps
                    if step.get("with", {}).get("name") ==
                    f"policy-cost-v5-{split}-allocation-attempt"
                )
                allocation_enforce = next(
                    step for step in allocation_steps
                    if step.get("name") ==
                    f"Enforce successful {split.upper()} allocation after attempt persistence"
                )
                self.assertTrue(allocation_capture["continue-on-error"])
                self.assertEqual(attempt_upload["if"], "always()")
                self.assertEqual(allocation_enforce["if"], "always()")
                self.assertLess(allocation_steps.index(allocation_capture),
                                allocation_steps.index(attempt_upload))
                self.assertLess(allocation_steps.index(attempt_upload),
                                allocation_steps.index(allocation_enforce))
                allocation_script = allocation_capture["run"]
                for status in (
                    "complete_validated_allocation", "allocation_rejected",
                    "origin_proof_rejected",
                ):
                    self.assertIn(status, allocation_script)
                for field in (
                    "github_run_id", "github_run_attempt", "head_sha",
                    "allocation_exit_code", "origin_proof_exit_code",
                    "sha256", "size",
                ):
                    self.assertIn(field, allocation_script)

    def test_infrastructure_retain_classifies_all_six_persistence_stages(self) -> None:
        workflow = yaml.safe_load(V5_WORKFLOW.read_text(encoding="utf-8"))
        job = workflow["jobs"]["infrastructure_retain"]
        script = "\n".join(str(step.get("run", "")) for step in job["steps"])
        for split in ("train", "select", "test"):
            self.assertTrue(any(
                step.get("with", {}).get("name") ==
                f"policy-cost-v5-{split}-raw-discovery"
                for step in job["steps"]
            ))
            self.assertTrue(any(
                step.get("with", {}).get("name") ==
                f"policy-cost-v5-{split}-allocation-attempt"
                for step in job["steps"]
            ))
        self.assertIn("f'{split}-generation-disposition.json'", script)
        self.assertIn("f'{split}-allocation-disposition.json'", script)
        self.assertIn("complete_atomic_generation", script)
        self.assertIn("allocation_rejected", script)
        self.assertIn("origin_proof_rejected", script)


if __name__ == "__main__":
    unittest.main()
