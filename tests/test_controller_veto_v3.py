"""Fail-closed contracts for the locked controller-veto-v3 actor campaign.

The workflow and plan were committed before any actor efficacy was generated.
The one-file execution addendum launched the campaign; after completion it may
point to the immutable result, which retains the exact launch-addendum digest.
Every precommitted identity, execution rule, and negative disposition remains
machine-checked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/experiments/locked_controller_veto_v3_plan.json"
WORKFLOW = ROOT / ".github/workflows/controller-veto-v3.yml"
EXECUTION = (
    ROOT / "data/experiments/locked_controller_veto_v3_execution.json"
)
RESULT = ROOT / "data/experiments/controller_veto_v3_result.json"

SOURCE_COMMIT = "cda70a217d776bdfb2c1457bfe8c0f5f0dbfed22"
SOURCE_TREE = "d006e7b0c1030f30ac135cdc489e2ea64e934ac0"
ROOT_SHA = "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"
VETO_SHA = "2d06a78eb9f088d36787fa559c529e5f6c6c674c45a8d067063deb8e06b15f3a"
ARENA_SHA = "2612e14a3e3b5c4e32333b8949261061797153b26e7501a37ad306f73889c768"

BASELINE = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
# The maintained actor relies on defaults for fields 36..39.  The candidate
# makes those same values explicit so its three-checkpoint boundary and full
# 40-field rollout tail are unambiguous in every transport and raw row.
FULL_TAIL = BASELINE.split(":", 2)[2] + ":0:0:0:1"
CANDIDATE = (
    "rolloutu3:data/champion.bin:data/champion.bin:"
    "data/models/continuation_v2_o0_shared_soup.bin:" + FULL_TAIL
)

SAFETY_STARTS = list(range(0, 200, 20))
FINAL_STARTS = list(range(0, 2500, 100))
SAFETY_SEEDS = {
    "candidate_first": "202608280301",
    "baseline_first": "202608280302",
}
FINAL_SEEDS = {
    "candidate_first": "202608280401",
    "baseline_first": "202608280402",
}

ARTIFACTS = {
    "arena": {
        "path": "data/evaluators/controller-veto-v3/arena",
        "sha256": ARENA_SHA,
        "size_bytes": 366704,
        "mode": "0755",
    },
    "root": {
        "path": "data/champion.bin",
        "sha256": ROOT_SHA,
        "size_bytes": 2823748,
    },
    "veto_controller": {
        "path": "data/models/continuation_v2_o0_shared_soup.bin",
        "sha256": VETO_SHA,
        "size_bytes": 2823748,
    },
}

EVIDENCE_TOOLS = {
    "merge_arena": (
        "tools/merge_arena.py",
        "9cad23c9e6550ea36d7721acf8e64144a44058083ad4aeb5bb5613a3a79139fb",
    ),
    "gate_actor_panel": (
        "tools/gate_actor_panel.py",
        "50ae52c66698d04cd4cc13af35c3d0eaed72e37bd76657dae12aa10f09dcc2ed",
    ),
    "validate_actor_shards": (
        "tools/validate_actor_shards.py",
        "bca430a94af64180436c7fb60d29b2e86ec4b3567ab3aabb09984aabee054855",
    ),
}


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def strict_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked_bytes(path: str) -> bytes:
    """Read immutable committed bytes, not a build-overwritten worktree file."""
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def tracked_mode(path: str) -> str:
    line = subprocess.check_output(
        ["git", "ls-tree", "HEAD", "--", path], cwd=ROOT, text=True
    ).strip()
    if not line:
        raise AssertionError(f"{path} is not tracked at HEAD")
    return line.split(maxsplit=1)[0]


def workflow_env(text: str, name: str) -> str:
    match = re.search(
        rf"(?m)^  {re.escape(name)}:\s*['\"]?([^'\"\n]+?)['\"]?\s*$",
        text,
    )
    if match is None:
        raise AssertionError(f"workflow omits env {name}")
    return match.group(1)


class ControllerVetoV3ProtocolTests(unittest.TestCase):
    maxDiff = None

    def test_plan_locks_exact_single_method_and_source(self) -> None:
        plan = strict_json(PLAN)
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(
            plan["artifact_kind"], "locked_controller_veto_v3_actor_campaign"
        )
        self.assertEqual(
            plan["status"], "precommitted_before_controller_veto_actor_efficacy"
        )
        self.assertIsNone(plan["results"])
        self.assertEqual(plan["source"]["remote_commit"], SOURCE_COMMIT)
        self.assertEqual(plan["source"]["tree"], SOURCE_TREE)
        self.assertEqual(plan["source"]["threads_per_shard"], 4)

        method = plan["method"]
        self.assertEqual(method["candidate_tail_40_fields"], FULL_TAIL)
        self.assertEqual(
            len(method["candidate_tail_40_fields"].split(":")), 40
        )
        self.assertEqual(method["candidate_actor"], CANDIDATE)
        self.assertEqual(method["baseline_actor"], BASELINE)
        self.assertEqual(len(CANDIDATE.split(":")), 44)
        self.assertEqual(len(BASELINE.split(":")), 38)
        self.assertIn("cannot add, replace, reorder, or widen", method["veto_rule"])
        self.assertTrue(any(
            "every legal move" in rule for rule in method["forbidden"]
        ))
        self.assertEqual(plan["multiplicity"], {
            "candidate_methods": 1,
            "veto_models": 1,
            "safety_looks": 1,
            "final_looks": 1,
            "no_optional_stopping_or_unplanned_retry": True,
        })

    def test_plan_locks_fresh_seeds_shards_and_exact_gates(self) -> None:
        plan = strict_json(PLAN)
        firewall = plan["seed_firewall"]
        self.assertEqual(firewall["namespace"], "20260828")
        self.assertEqual(firewall["excluded_development_namespace"], "20260827")
        self.assertEqual(firewall["excluded_development_seeds"], ["202608270001"])
        self.assertEqual(firewall["safety"], SAFETY_SEEDS)
        self.assertEqual(firewall["final"], FINAL_SEEDS)

        safety = plan["safety_screen"]
        self.assertEqual(safety["rounds"], 3)
        self.assertEqual(safety["pairs_per_orientation"], 200)
        self.assertEqual(safety["matches_total"], 800)
        self.assertEqual(safety["shards_per_orientation"], 10)
        self.assertEqual(safety["pairs_per_shard"], 20)
        self.assertEqual(safety["pair_starts"], SAFETY_STARTS)
        self.assertEqual(safety["candidate_first_seed"], SAFETY_SEEDS["candidate_first"])
        self.assertEqual(safety["baseline_first_seed"], SAFETY_SEEDS["baseline_first"])
        self.assertEqual(safety["gate"], [
            "equal-weight reciprocal combined candidate match score >= 0.5",
            "equal-weight reciprocal combined candidate point margin > 0",
            "candidate match score in each reciprocal orientation after inversion >= 0.475",
            "zero capped rounds, gaps, overlaps, incomplete footers, malformed rows, hash failures, provenance drift, or operational errors",
        ])

        final = plan["final_promotion"]
        self.assertEqual(final["rounds"], 3)
        self.assertEqual(final["pairs_per_orientation"], 2500)
        self.assertEqual(final["matches_total"], 10000)
        self.assertEqual(final["shards_per_orientation"], 25)
        self.assertEqual(final["pairs_per_shard"], 100)
        self.assertEqual(final["pair_starts"], FINAL_STARTS)
        self.assertEqual(final["candidate_first_seed"], FINAL_SEEDS["candidate_first"])
        self.assertEqual(final["baseline_first_seed"], FINAL_SEEDS["baseline_first"])
        self.assertEqual(final["confidence_z"], 1.645)
        self.assertEqual(final["promotion_gate"], [
            "combined candidate match score - 1.645 * orientation-stratified pair-clustered SE > 0.5",
            "combined candidate point margin - 1.645 * orientation-stratified pair-clustered SE > 0",
            "candidate match-score point estimate > 0.5 in each reciprocal orientation after inversion",
            "zero capped rounds, gaps, overlaps, incomplete footers, malformed rows, hash failures, provenance drift, or operational errors",
        ])

    def test_plan_hashes_exact_tracked_transport_models_and_evidence(self) -> None:
        plan = strict_json(PLAN)
        for role, expected in ARTIFACTS.items():
            with self.subTest(role=role):
                actual = plan["locked_artifacts"][role]
                for field in ("path", "sha256", "size_bytes"):
                    self.assertEqual(actual[field], expected[field])
                data = tracked_bytes(expected["path"])
                self.assertEqual(len(data), expected["size_bytes"])
                self.assertEqual(sha256(data), expected["sha256"])
                if role == "arena":
                    self.assertEqual(actual["mode"], expected["mode"])
                    self.assertEqual(tracked_mode(expected["path"]), "100755")

        for name, (path, digest) in EVIDENCE_TOOLS.items():
            with self.subTest(tool=name):
                self.assertEqual(plan["evidence"]["tools"][name], {
                    "path": path,
                    "sha256": digest,
                })
                self.assertEqual(sha256(tracked_bytes(path)), digest)

        provenance = plan["veto_controller_provenance"]
        prior_results = (
            ("continuation_v2_execution_result", "8ada5982e633daee3a5c9a2828484edddcf4ffe83c10e654a92560a30ace5c65"),
            ("screen_a_result", "4e1861488f89a13d0589cb6850781df52808cf9bb7e6eac02d5174fa8f9af1ce"),
            ("screen_b_result", "1b7c93c383c35049ea14e58ee05f26a8c9fd552a831c467a6e9643aaa22d8d0f"),
        )
        for name, digest in prior_results:
            entry = provenance[name]
            self.assertEqual(entry["sha256"], digest)
            self.assertEqual(sha256(tracked_bytes(entry["path"])), digest)

    def test_workflow_is_addendum_only_and_matches_locked_panel(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        execution_path = EXECUTION.relative_to(ROOT).as_posix()
        plan_path = PLAN.relative_to(ROOT).as_posix()
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("continue-on-error", text)
        self.assertIn("on:\n  push:", text)
        self.assertIn("agent/correctness-and-policy-upgrade", text)
        self.assertRegex(
            text,
            r"(?s)actions/checkout@v4\s*\n\s*with:\s*\n\s*fetch-depth: 0"
            r"\s*\n\s*path: campaign",
        )
        self.assertRegex(
            text,
            rf"(?s)paths:\s*\n\s*- {re.escape(execution_path)}(?:\n|$)",
        )
        self.assertEqual(workflow_env(text, "PLAN_PATH"), plan_path)
        self.assertEqual(workflow_env(text, "EXECUTION_PATH"), execution_path)
        self.assertEqual(workflow_env(text, "SOURCE_COMMIT"), SOURCE_COMMIT)
        self.assertEqual(workflow_env(text, "SOURCE_TREE"), SOURCE_TREE)
        self.assertEqual(workflow_env(text, "PLAN_SHA"), sha256(PLAN.read_bytes()))
        self.assertEqual(workflow_env(text, "ARENA_SHA"), ARENA_SHA)
        self.assertEqual(
            workflow_env(text, "ARENA_PATH"), ARTIFACTS["arena"]["path"]
        )
        self.assertEqual(workflow_env(text, "ARENA_SIZE"), "366704")
        self.assertEqual(workflow_env(text, "ROOT_SHA"), ROOT_SHA)
        self.assertEqual(
            workflow_env(text, "ROOT_PATH"), ARTIFACTS["root"]["path"]
        )
        self.assertEqual(workflow_env(text, "ROOT_SIZE"), "2823748")
        self.assertEqual(workflow_env(text, "VETO_SHA"), VETO_SHA)
        self.assertEqual(
            workflow_env(text, "VETO_PATH"),
            ARTIFACTS["veto_controller"]["path"],
        )
        self.assertEqual(workflow_env(text, "VETO_SIZE"), "2823748")
        self.assertEqual(
            workflow_env(text, "MERGER_SHA"), EVIDENCE_TOOLS["merge_arena"][1]
        )
        self.assertEqual(
            workflow_env(text, "GATE_SHA"),
            EVIDENCE_TOOLS["gate_actor_panel"][1],
        )
        self.assertEqual(
            workflow_env(text, "SHARD_VALIDATOR_SHA"),
            EVIDENCE_TOOLS["validate_actor_shards"][1],
        )
        self.assertEqual(workflow_env(text, "CANDIDATE"), CANDIDATE)
        self.assertEqual(workflow_env(text, "BASELINE"), BASELINE)
        self.assertEqual(
            workflow_env(text, "SAFETY_CANDIDATE_SEED"),
            "202608280301",
        )
        self.assertEqual(
            workflow_env(text, "SAFETY_BASELINE_SEED"),
            "202608280302",
        )
        self.assertEqual(
            workflow_env(text, "FINAL_CANDIDATE_SEED"),
            "202608280401",
        )
        self.assertEqual(
            workflow_env(text, "FINAL_BASELINE_SEED"),
            "202608280402",
        )

        self.assertIn('test "$GITHUB_EVENT_NAME" = push', text)
        self.assertIn('test "$GITHUB_RUN_ATTEMPT" = 1', text)
        self.assertIn(
            'test "$(git diff-tree --no-commit-id --name-only -r HEAD)" = "$EXECUTION_PATH"',
            text,
        )
        self.assertIn(
            "git diff-tree --no-commit-id --name-status -r HEAD",
            text,
        )
        self.assertIn("$(printf 'A\\t%s' \"$EXECUTION_PATH\")", text)
        self.assertIn('git cat-file -e "HEAD^:$EXECUTION_PATH"', text)
        self.assertIn(
            'test "$(git rev-list --all --count -- "$EXECUTION_PATH")" = 1',
            text,
        )
        self.assertNotRegex(text, r"(?m)^\s*(?:make|gcc|clang|cc)\s")
        self.assertIn(str(SAFETY_STARTS).replace(" ", ""), text.replace(" ", ""))
        self.assertIn(str(FINAL_STARTS).replace(" ", ""), text.replace(" ", ""))
        self.assertIn('-n 20 -t 4 -s "$SEED" -r 3', text)
        self.assertIn('-n 100 -t 4 -s "$SEED" -r 3', text)
        self.assertIn("--expect-pairs 200", text)
        self.assertIn("--expect-pairs 2500", text)
        self.assertIn("--mode safety", text)
        self.assertIn("--mode final", text)
        self.assertIn("needs.safety_merge.outputs.passed == 'true'", text)
        self.assertIn(CANDIDATE, text)
        self.assertIn(BASELINE, text)

    def test_execution_addendum_binds_every_identity_when_present(self) -> None:
        if not EXECUTION.exists():
            self.skipTest("execution addendum intentionally not committed yet")
        execution = strict_json(EXECUTION)
        completed = RESULT.exists()
        parent = execution.get("source_parent_commit")
        self.assertIsInstance(parent, str)
        self.assertRegex(parent, r"^[0-9a-f]{40}$")
        workflow_text = WORKFLOW.read_text(encoding="utf-8")
        inspection_rule = strict_json(PLAN)["evidence"]["inspection_rule"]
        expected = {
            "schema_version": 1,
            "artifact_kind": "locked_controller_veto_v3_execution",
            "status": (
                "complete_rejected_at_predeclared_safety_gate"
                if completed
                else "launch_bound_before_actor_efficacy"
            ),
            "source_parent_commit": parent,
            "workflow": {
                "path": WORKFLOW.relative_to(ROOT).as_posix(),
                "sha256": sha256(WORKFLOW.read_bytes()),
            },
            "plan": {
                "path": PLAN.relative_to(ROOT).as_posix(),
                "sha256": sha256(PLAN.read_bytes()),
            },
            "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
            "transport": {
                "binding": (
                    "transport exact prebuilt artifacts; compile nothing "
                    "on evaluation runners"
                ),
                "arena": {**ARTIFACTS["arena"]},
                "root": {**ARTIFACTS["root"]},
                "veto": {
                    "path": ARTIFACTS["veto_controller"]["path"],
                    "sha256": VETO_SHA,
                    "size_bytes": ARTIFACTS["veto_controller"]["size_bytes"],
                },
            },
            "upstream_provenance": {
                "continuation_v2_execution_result_sha256": (
                    "8ada5982e633daee3a5c9a2828484edddcf4ffe83c10e654a92560a30ace5c65"
                ),
                "continuation_v2_screen_a_result_sha256": (
                    "4e1861488f89a13d0589cb6850781df52808cf9bb7e6eac02d5174fa8f9af1ce"
                ),
                "continuation_v2_screen_b_result_sha256": (
                    "1b7c93c383c35049ea14e58ee05f26a8c9fd552a831c467a6e9643aaa22d8d0f"
                ),
            },
            "evidence_tools": {
                "merge_arena_sha256": EVIDENCE_TOOLS["merge_arena"][1],
                "gate_actor_panel_sha256": EVIDENCE_TOOLS[
                    "gate_actor_panel"
                ][1],
                "validate_actor_shards_sha256": EVIDENCE_TOOLS[
                    "validate_actor_shards"
                ][1],
            },
            "evaluator": {
                "compiler": "gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0",
                "cflags": (
                    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
                    "-Wall -Wextra -std=c11"
                ),
                "ldflags": "-lm -pthread",
                "threads_per_shard": 4,
            },
            "actors": {
                "candidate": CANDIDATE,
                "baseline": BASELINE,
                "candidate_tail_fields": 40,
                "baseline_literal_lc_champion_tail_fields": 36,
                "safety_provenance": workflow_env(
                    workflow_text, "SAFETY_PROVENANCE"
                ),
                "final_provenance": workflow_env(
                    workflow_text, "FINAL_PROVENANCE"
                ),
            },
            "seed_firewall": {
                "excluded_development_namespace": "20260827",
                "eligible_namespace": "20260828",
            },
            "safety": {
                "pairs_per_orientation": 200,
                "pairs_per_shard": 20,
                "starts": SAFETY_STARTS,
                "candidate_first_seed": SAFETY_SEEDS["candidate_first"],
                "baseline_first_seed": SAFETY_SEEDS["baseline_first"],
                "gate": (
                    "score>=0.5; margin>0; each orientation candidate "
                    "score>=0.475; zero caps; exact validity"
                ),
            },
            "final": {
                "execute_if_safety_passes": True,
                "pairs_per_orientation": 2500,
                "pairs_per_shard": 100,
                "starts": FINAL_STARTS,
                "candidate_first_seed": FINAL_SEEDS["candidate_first"],
                "baseline_first_seed": FINAL_SEEDS["baseline_first"],
                "gate_z": 1.645,
                "gate": (
                    "score-z*SE>0.5; margin-z*SE>0; each orientation "
                    "candidate score>0.5; zero caps; exact validity"
                ),
            },
            "inspection_rule": inspection_rule,
            "results": (
                {
                    "path": RESULT.relative_to(ROOT).as_posix(),
                    "sha256": sha256(RESULT.read_bytes()),
                    "workflow_run_id": 32589655623,
                    "evidence_artifact_id": 9480365159,
                    "safety_gate_passed": False,
                    "match_score": 0.485625,
                    "margin_per_match": -1.07875,
                    "orientation_match_scores": [0.50125, 0.47],
                    "locked_final_executed": False,
                    "reserved_final_seeds_consumed": [],
                    "candidate_promoted": False,
                    "maintained_actor_unchanged": True,
                }
                if completed
                else None
            ),
        }
        self.assertEqual(execution, expected)

        # In the launch state, independently prove the same one-path mutation
        # that the workflow's guard requires.  The completed state is instead
        # bound to the archived launch digest and immutable result above.
        if completed:
            return
        try:
            committed = tracked_bytes(EXECUTION.relative_to(ROOT).as_posix())
        except subprocess.CalledProcessError:
            committed = None
        if committed is not None:
            self.assertEqual(committed, EXECUTION.read_bytes())
            changed = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
                cwd=ROOT,
                text=True,
            ).splitlines()
            self.assertEqual(changed, [EXECUTION.relative_to(ROOT).as_posix()])
            self.assertEqual(
                execution["source_parent_commit"],
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD^"], cwd=ROOT, text=True
                ).strip(),
            )

    def test_completed_result_is_bound_and_fail_closed(self) -> None:
        if not RESULT.exists():
            self.skipTest("locked campaign result not recorded yet")
        result = strict_json(RESULT)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(
            result["status"], "rejected_at_predeclared_safety_gate"
        )
        self.assertEqual(result["workflow"]["run_id"], 32589655623)
        self.assertEqual(
            result["workflow"]["launch_commit"],
            "0f172d1df8444164970dfa551fc9c07f8dae957d",
        )
        self.assertEqual(
            result["source_binding"]["locked_actor_source_commit"],
            SOURCE_COMMIT,
        )
        self.assertEqual(
            result["source_binding"]["locked_actor_source_tree"], SOURCE_TREE
        )
        self.assertEqual(
            result["evidence"]["github_artifact_digest"],
            "sha256:cb9fcaa8f766a9a718432a05168951d8b37230d2130096ea729f055251d7508b",
        )
        self.assertEqual(result["candidate_result"]["match_score"], 0.485625)
        self.assertEqual(
            result["candidate_result"]["margin_per_match"], -1.07875
        )
        self.assertEqual(
            result["candidate_result"]["orientation_match_scores"],
            [0.50125, 0.47],
        )
        self.assertFalse(result["locked_safety_gate"]["passed"])
        self.assertTrue(result["locked_safety_gate"]["raw_inputs_validated"])
        self.assertTrue(result["locked_safety_gate"]["zero_capped_rounds"])
        self.assertFalse(result["disposition"]["locked_final_executed"])
        self.assertEqual(result["disposition"]["final_seeds_consumed"], [])
        self.assertEqual(
            result["disposition"]["reserved_final_seeds"],
            [FINAL_SEEDS["candidate_first"], FINAL_SEEDS["baseline_first"]],
        )
        self.assertFalse(result["disposition"]["candidate_promoted"])
        self.assertTrue(
            result["independent_verification"]
            ["substantive_statistics_and_gate_reproduced_exactly"]
        )


if __name__ == "__main__":
    unittest.main()
