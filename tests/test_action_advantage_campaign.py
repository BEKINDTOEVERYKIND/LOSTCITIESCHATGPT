"""Fail-closed protocol tests for action-advantage-veto-v1.

The locked plan and workflow are committed before the one execution addendum.
These tests require the execution file to be absent in the source commit.  If
it is later present on the branch, they validate its unique historical
addendum-only launch topology.  They also check the campaign's data firewall,
one-candidate calibration, compile-once transport, and unchanged safety/final
promotion gates.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/experiments/locked_action_advantage_veto_v1_plan.json"
WORKFLOW = ROOT / ".github/workflows/action-advantage-veto-v1.yml"
EXECUTION = (
    ROOT / "data/experiments/locked_action_advantage_veto_v1_execution.json"
)
GENERATOR = ROOT / "tools/action_advantage.c"

BASELINE = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
FULL_TAIL = BASELINE.split(":", 2)[2] + ":0:0:0:1"
PREFIX = (
    "rolloutu4:data/champion.bin:data/champion.bin:"
    "data/models/action_advantage_veto_v1.bin:" + FULL_TAIL + ":"
)


def unique(items: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in items:
        if key in out:
            raise ValueError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def strict_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    return value


class ActionAdvantageCampaignTests(unittest.TestCase):
    maxDiff = None

    def test_plan_locks_general_one_way_method(self) -> None:
        plan = strict_json(PLAN)
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(
            plan["artifact_kind"],
            "locked_action_advantage_veto_v1_actor_campaign",
        )
        self.assertEqual(
            plan["status"],
            "precommitted_before_candidate_generation_or_actor_efficacy",
        )
        self.assertIsNone(plan["results"])
        method = plan["method"]
        self.assertEqual(method["baseline_actor"], BASELINE)
        self.assertEqual(
            method["candidate_template"], PREFIX + "{heldout_threshold}"
        )
        self.assertEqual(method["candidate_tail_fields_before_threshold"], 40)
        self.assertEqual(len(FULL_TAIL.split(":")), 40)
        joined = "\n".join(method["invariants"])
        self.assertIn("cannot add, replace, reorder, or widen", joined)
        self.assertIn("top-policy-moves-only", joined)
        self.assertIn("consumes no RNG", joined)
        self.assertIn("sanitized information view", joined)
        self.assertIn("trunk, value head, and belief head", joined)
        self.assertIn("indistinguishable physical wager copies", joined)
        self.assertEqual(plan["multiplicity"], {
            "rankers": 1,
            "thresholds_entering_safety": 1,
            "candidate_actors": 1,
            "safety_looks": 1,
            "final_looks": 1,
            "unplanned_retries": 0,
            "optional_stopping": False,
        })

    def test_plan_locks_generated_data_and_honest_teacher(self) -> None:
        plan = strict_json(PLAN)
        data = plan["development_data"]
        self.assertEqual(data["source"], "generated maintained-actor self-play only")
        self.assertIn("zero-cap", data["proposal_population"])
        self.assertIn("primary-plus-fresh", data["proposal_population"])
        self.assertEqual(data["source_matches"], 64)
        self.assertIn("finish all 64", data["completion_rule"])
        self.assertIn("no proposal cap", data["completion_rule"])
        self.assertEqual(data["label_worlds"], 256)
        self.assertGreaterEqual(data["label_worlds"], 256)
        self.assertEqual(data["label_threads"], 4)
        self.assertEqual(data["teacher_actor"], "policy:data/champion.bin:0:20")
        self.assertIn("exact 20-symmetry champion policy", data["teacher_declaration"])
        self.assertIn("not claimed to equal", data["teacher_declaration"])
        prohibited = "\n".join(data["prohibited_inputs"])
        self.assertIn("data/probes/*.state", prohibited)
        self.assertIn("user-commented", prohibited)
        self.assertIn("human labels", prohibited)
        self.assertIn("no repository checkout", data["enforcement"])

        generator = GENERATOR.read_text(encoding="utf-8")
        self.assertIn("Only generated matches are accepted", generator)
        self.assertNotRegex(generator, r'!strcmp\(a, "--(?:state|probe)"')
        self.assertIn("AA_MIN_LABEL_WORLDS", generator)
        self.assertIn("finish_remaining_match", generator)
        self.assertIn("ss.prefix_confirmed", generator)
        self.assertIn("ss.unfinished_cap_leaves == 0", generator)

    def test_grouped_split_and_heldout_threshold_are_frozen(self) -> None:
        plan = strict_json(PLAN)
        training = plan["training"]
        self.assertIn("source_match_id is an indivisible group", training["split"])
        self.assertEqual(training["validation_permille"], 250)
        self.assertEqual(training["max_validation_kl"], 0.01)
        self.assertEqual(training["max_state_kl"], 0.05)
        self.assertTrue(any("stored champion logit" in rule
                            for rule in training["fail_closed"]))
        calibration = plan["threshold_calibration"]
        self.assertEqual(calibration["data"], "heldout development records only")
        self.assertEqual(calibration["predeclared_grid"], [0, 0.1, 0.25, 0.5, 1])
        self.assertIn("exact float32 threshold", calibration["runtime_boundary_rule"])
        self.assertEqual(
            calibration["selection"],
            "among eligible grid points, lexicographically minimize (oracle_regret, mistakes, negative signed_hybrid_sum, negative threshold)",
        )
        self.assertIn("at least 20 percent", calibration["eligibility"])
        self.assertIn("strictly positive", calibration["eligibility"])
        self.assertIn("before any safety game", calibration["freeze"])

        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("validation_proposals < 20", text)
        self.assertIn('"source_matches_completed": 64', text)
        self.assertIn('"proposal_cap": 0', text)
        self.assertIn('fnv1a64(b"ABSENT\\0")', text)
        self.assertIn("champion_bytes[24:]", text)
        self.assertIn("generated-record header drift", text)
        self.assertIn('metrics.get("training_gate_passed") is not True', text)
        self.assertNotIn('metrics.get("promotion_gate_passed")', text)
        self.assertIn("not final_loss < initial_loss", text)
        self.assertIn('row["invalid_scores"] != 0', text)
        self.assertIn('row.get("runtime_threshold") != runtime_threshold', text)
        self.assertIn('row["retained"] * 5 >= validation_proposals', text)
        self.assertIn('row["signed_hybrid_sum"] > 0.0', text)
        self.assertIn('item[1]["oracle_regret"]', text)
        self.assertIn('-item[1]["threshold"]', text)
        self.assertIn("threshold_and_actor_frozen_before_safety", text)
        self.assertRegex(
            text,
            r"(?s)  safety_evaluate:.*?needs: train_and_freeze.*?"
            r"name: action-advantage-v1-evaluator",
        )

    def test_workflow_is_one_addendum_compile_once_transport(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("continue-on-error", text)
        self.assertIn("on:\n  push:", text)
        self.assertIn(
            "data/experiments/locked_action_advantage_veto_v1_execution.json",
            text,
        )
        self.assertIn(
            'test "$(git diff-tree --no-commit-id --name-status -r HEAD)"',
            text,
        )
        preflight_header = text.split("\n  preflight:", 1)[1].split(
            "\n    steps:", 1
        )[0]
        self.assertNotIn("if:", preflight_header)
        self.assertIn('test "$GITHUB_RUN_ATTEMPT" = 1', text)
        self.assertIn('git -C campaign archive HEAD^ | tar -x -C source', text)
        self.assertIn("compile exactly once in preflight", text)
        self.assertIn("(cd source && ./bin/test_action_ranker", text)
        self.assertIn("./bin/test_action_advantage)", text)
        self.assertIn("(cd transport && sha256sum -c SHA256SUMS.txt)", text)
        self.assertIn("(cd evaluator && sha256sum -c SHA256SUMS.txt)", text)
        self.assertIn("selected.json PREFLIGHT_BUILD_INFO.txt", text)
        safety_eval = text.split("\n  safety_evaluate:", 1)[1].split(
            "\n  safety_merge:", 1
        )[0]
        self.assertEqual(
            safety_eval.count('sha256sum "${OUT}.jsonl" > "${OUT}.sha256"'),
            1,
        )

        after_preflight = text.split("\n  train_and_freeze:", 1)[1]
        self.assertNotRegex(after_preflight, r"(?m)^\s*(?:make|gcc|cc|clang)\b")
        self.assertNotIn("actions/checkout", after_preflight)
        self.assertIn("test ! -e data/probes", after_preflight)
        self.assertIn("find . -name '*.state'", after_preflight)
        self.assertIn("--matches 64 --worlds 256 --label-threads 4", after_preflight)
        self.assertNotIn("--max-proposals", after_preflight)
        self.assertIn('--reroot-actor "$TEACHER"', after_preflight)
        if EXECUTION.exists():
            execution = strict_json(EXECUTION)
            source = execution.get("source_parent_commit")
            self.assertRegex(source, r"^[0-9a-f]{40}$")
            self.assertEqual(
                execution["source_parent_tree"],
                subprocess.check_output(
                    ["git", "rev-parse", f"{source}^{{tree}}"],
                    cwd=ROOT, text=True,
                ).strip(),
            )
            self.assertEqual(
                execution["plan"]["sha256"],
                hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                execution["workflow"]["sha256"],
                hashlib.sha256(WORKFLOW.read_bytes()).hexdigest(),
            )
            path = EXECUTION.relative_to(ROOT).as_posix()
            additions = subprocess.check_output(
                ["git", "log", "--all", "--format=%H", "--diff-filter=A",
                 "--", path],
                cwd=ROOT, text=True,
            ).splitlines()
            self.assertEqual(len(additions), 1)
            launch = additions[0]
            self.assertEqual(
                subprocess.check_output(
                    ["git", "rev-parse", f"{launch}^"], cwd=ROOT, text=True,
                ).strip(),
                source,
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "diff-tree", "--no-commit-id", "--name-status",
                     "-r", launch],
                    cwd=ROOT, text=True,
                ).strip(),
                f"A\t{path}",
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "show", f"{launch}:{path}"], cwd=ROOT,
                ),
                EXECUTION.read_bytes(),
            )

    def test_unchanged_actor_gates_and_fresh_namespaces(self) -> None:
        plan = strict_json(PLAN)
        firewall = plan["seed_firewall"]
        self.assertEqual(firewall["development_namespace"], "20260901")
        self.assertEqual(firewall["safety_final_namespace"], "20260902")
        seeds = [
            firewall["safety"]["candidate_first"],
            firewall["safety"]["baseline_first"],
            firewall["final"]["candidate_first"],
            firewall["final"]["baseline_first"],
        ]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(seed.startswith("20260902") for seed in seeds))

        safety = plan["safety_screen"]
        self.assertEqual(safety["pairs_per_orientation"], 200)
        self.assertEqual(safety["matches_total"], 800)
        self.assertEqual(safety["pairs_per_shard"], 20)
        self.assertEqual(safety["pair_starts"], list(range(0, 200, 20)))
        self.assertEqual(safety["gate"], [
            "equal-weight reciprocal combined candidate match score >= 0.5",
            "equal-weight reciprocal combined candidate point margin > 0",
            "candidate match score in each reciprocal orientation after inversion >= 0.475",
            "zero capped rounds, gaps, overlaps, incomplete footers, malformed rows, hash failures, provenance drift, or operational errors",
        ])

        final = plan["final_promotion"]
        self.assertTrue(final["execute_only_if_safety_passes"])
        self.assertEqual(final["pairs_per_orientation"], 2500)
        self.assertEqual(final["matches_total"], 10000)
        self.assertEqual(final["pairs_per_shard"], 100)
        self.assertEqual(final["pair_starts"], list(range(0, 2500, 100)))
        self.assertEqual(final["confidence_z"], 1.645)
        self.assertEqual(final["promotion_gate"], [
            "combined candidate match score - 1.645 * orientation-stratified pair-clustered SE > 0.5",
            "combined candidate point margin - 1.645 * orientation-stratified pair-clustered SE > 0",
            "candidate match-score point estimate > 0.5 in each reciprocal orientation after inversion",
            "zero capped rounds, gaps, overlaps, incomplete footers, malformed rows, hash failures, provenance drift, or operational errors",
        ])

        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("needs.safety_merge.outputs.passed == 'true'", text)
        self.assertIn("needs.safety_merge.result == 'success'", text)
        self.assertIn("needs.final_evaluate.result == 'success'", text)
        self.assertEqual(text.count("--mode safety"), 1)
        self.assertEqual(text.count("--mode final"), 1)
        self.assertIn("--pairs-per-orientation 200", text)
        self.assertIn("--pairs-per-orientation 2500", text)
        self.assertEqual(text.count("--gate-z 1.645"), 4)

    def test_complete_shards_precede_each_gate(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        safety = text.split("\n  safety_merge:", 1)[1].split(
            "\n  final_evaluate:", 1
        )[0]
        final = text.split("\n  final_merge:", 1)[1]
        for section, pairs, starts in (
            (safety, 200, "0,20,40,60,80,100,120,140,160,180"),
            (
                final,
                2500,
                "0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400",
            ),
        ):
            validate_at = section.index("validate_actor_shards.py")
            merge_at = section.index("merge_arena.py reciprocal")
            gate_at = section.index("gate_actor_panel.py")
            self.assertLess(validate_at, merge_at)
            self.assertLess(merge_at, gate_at)
            self.assertIn(f"--expect-pairs {pairs}", section)
            self.assertIn(f"--starts {starts}", section)


if __name__ == "__main__":
    unittest.main()
