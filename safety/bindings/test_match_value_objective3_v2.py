"""Regression contracts for the locked objective-3 match-value v2 campaign."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

from tools.match_value_objective3_v2 import (
    AUDIT_EVIDENCE_PATH,
    AUDIT_JSON_PATH,
    AUDIT_MARKDOWN_PATH,
    AUDIT_RESULT_PATH,
    CRITICAL_Z,
    DEFINITION_PARENT_COMMIT,
    DEFINITION_PARENT_TREE,
    DEFINITION_PATHS,
    DEVELOPMENT_PAIRS,
    DEVELOPMENT_SCHEMA,
    FINAL_PAIRS,
    EXECUTION_PATH,
    LOCK_PATH,
    MAINTAINED_ACTOR,
    MODEL_PATH,
    MODEL_SHA256,
    PLAN_PATH,
    PREDECESSOR_EXECUTION_PATH,
    PROJECTED_TABLE_PATH,
    RAW_TABLE_PATH,
    RESULT_SCHEMA,
    SAFETY_PAIRS,
    SMOKE_NAMESPACE,
    TABLE_CONTROLLER_WORDS,
    TABLE_ROLE_CYCLE,
    TABLE_SAMPLES,
    TIE_PRIORITY,
    VARIANTS,
    WORLD_CAP,
    WORKFLOW_PATH,
    EvidenceError,
    authoritative_audit_result,
    build_actors,
    development_selection,
    evidence_manifest,
    final_gate,
    inspect_table,
    safety_gate,
    table_manifest,
    terminal_result,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / PLAN_PATH
WORKFLOW = ROOT / WORKFLOW_PATH
SMOKE_TABLE_SEED = f"{SMOKE_NAMESPACE}0001"


def panel(
    pairs: int,
    first_quarters: int,
    second_quarters: int,
    first_margin: int,
    second_margin: int,
    *,
    score: float = 0.51,
    score_se: float = 0.004,
    margin: float = 0.5,
    margin_se: float = 0.1,
    orientations: tuple[float, float] = (0.51, 0.51),
    caps: int = 0,
) -> dict:
    def sufficient(quarters: int, margin_sum: int,
                   block_caps: int) -> dict:
        return {
            "pairs": pairs,
            "score_quarters_sum": quarters,
            "margin_sum": margin_sum,
            "capped_rounds": block_caps,
        }

    return {
        "blocks": [
            {"sufficient_statistics": sufficient(
                first_quarters, first_margin, caps)},
            {"sufficient_statistics": sufficient(
                second_quarters, second_margin, 0)},
        ],
        "raw_input_validation": {"status": "validated"},
        "candidate_result": {
            "match_score": score,
            "match_score_pair_clustered_se": score_se,
            "margin_per_game": margin,
            "margin_pair_clustered_se": margin_se,
            "orientation_match_scores": list(orientations),
            "capped_rounds": caps,
        },
    }


def fnv1a(snapshot: bytes) -> int:
    value = 1469598103934665603
    for byte in snapshot:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def table_bytes(projected: bool, *, seed: str = SMOKE_TABLE_SEED,
                corrupt_profile: bool = False) -> bytes:
    r1_count, r2_count = 2361, 4721
    header = bytearray(128)
    header[:8] = b"LCMVAL1\0"
    for offset, value in (
        (8, 1), (12, 128), (16, TABLE_SAMPLES), (20, r1_count),
        (24, r2_count), (28, 150),
    ):
        struct.pack_into("<I", header, offset, value)
    struct.pack_into("<Q", header, 32, 0x123456789ABCDEF0)
    for index, value in enumerate(TABLE_CONTROLLER_WORDS):
        struct.pack_into("<I", header, 40 + 4 * index, value)
    struct.pack_into("<Q", header, 84, int(seed))
    struct.pack_into("<d", header, 92, 0.25)
    struct.pack_into("<d", header, 100, 0.5)
    struct.pack_into("<I", header, 108, TABLE_ROLE_CYCLE)
    struct.pack_into("<I", header, 112, 1)
    struct.pack_into("<I", header, 116, int(projected))
    struct.pack_into("<I", header, 120, 1)
    struct.pack_into("<I", header, 124,
                     int("0030d23b", 16) + int(corrupt_profile))
    values: list[float] = []
    for length in (r1_count, r2_count):
        not_start = [20.0 * (index / (length - 1) - 0.5)
                     for index in range(length)]
        if not projected:
            not_start[length // 3] += 0.125
        starts = [-value for value in reversed(not_start)]
        values.extend(not_start)
        values.extend(starts)
    body = bytes(header) + struct.pack(f"<{len(values)}d", *values)
    return body + struct.pack("<Q", fnv1a(body))


def terminal_evidence(stage: str) -> list[dict[str, str]]:
    names = {
        "pre-efficacy-manifest.json",
        "transport/BUILD_INFO.txt",
        "transport/SHA256SUMS.txt",
        "transport/bindings/actors.json",
        "transport/bindings/definition-lock.json",
        "transport/bindings/execution.json",
        "transport/bindings/plan.json",
        "transport/bindings/pre-efficacy-manifest.json",
        "transport/bindings/table-manifest.json",
        f"transport/{RAW_TABLE_PATH}",
        f"transport/{PROJECTED_TABLE_PATH}",
        "development/merged/development-selection.json",
        "development/merged/RAW_ALL_PLY-reciprocal.json",
        "development/merged/PROJECTED_ALL_PLY-reciprocal.json",
    }
    for index in range(40):
        for suffix in ("jsonl", "sha256", "time"):
            names.add(f"development/downloads/shard-{index}.{suffix}")
    if stage == "development":
        names.update({"stages/safety-skipped.json",
                      "stages/final-skipped.json"})
    else:
        names.update({"safety/merged/safety-decision.json",
                      "safety/merged/reciprocal.json"})
        for index in range(20):
            for suffix in ("jsonl", "sha256", "time"):
                names.add(f"safety/downloads/shard-{index}.{suffix}")
        if stage == "safety":
            names.add("stages/final-skipped.json")
        else:
            names.update({"final/merged/final-decision.json",
                          "final/merged/reciprocal.json"})
            for index in range(50):
                for suffix in ("jsonl", "sha256", "time"):
                    names.add(f"final/downloads/shard-{index}.{suffix}")
    return [{"path": name, "sha256": "0" * 64} for name in sorted(names)]


class MatchValueObjective3V2Tests(unittest.TestCase):
    maxDiff = None

    def test_plan_is_exact_rich_contract_and_uses_no_probe_selection(self) -> None:
        value = json.loads(PLAN.read_text(encoding="utf-8"))
        validate_plan(value)
        self.assertEqual(value["definition_lock"]["definition_parent_commit"],
                         DEFINITION_PARENT_COMMIT)
        self.assertEqual(value["definition_lock"]["definition_parent_tree"],
                         DEFINITION_PARENT_TREE)
        self.assertEqual(value["definition_lock"]["definition_files"],
                         list(DEFINITION_PATHS))
        self.assertEqual(set(value["variants"]), set(VARIANTS))
        self.assertEqual(value["diagnostic_audit"]["selection_use"],
                         "forbidden")
        self.assertEqual(value["development"]["total_raw_shards"], 40)
        self.assertEqual(value["safety"]["pairs_per_shard"], 20)
        self.assertEqual(value["safety"]["total_raw_shards"], 20)
        self.assertEqual(value["final"]["total_raw_shards"], 50)
        self.assertEqual(
            value["final"]["promotion_gate"]
            ["combined_margin_pair_clustered_lcb_strictly_above"], 0.0)

        for mutation in (
            lambda item: item["variants"].pop("RAW_ALL_PLY"),
            lambda item: item["diagnostic_audit"].update(
                {"selection_use": "variant_selection"}),
            lambda item: item["safety"].update({"pairs_per_shard": 100}),
            lambda item: item["definition_lock"].update(
                {"definition_parent_commit": "0" * 40}),
        ):
            broken = json.loads(json.dumps(value))
            mutation(broken)
            with self.assertRaises(EvidenceError):
                validate_plan(broken)

    def test_authoritative_audit_is_complete_bound_and_selection_forbidden(self) -> None:
        value = authoritative_audit_result(ROOT)
        self.assertEqual(value["selection_use"], "forbidden")
        self.assertEqual(value["cases"], 17)
        self.assertEqual(value["raw_shards"], 17)
        self.assertEqual(value["run"]["attempt"], 1)
        self.assertEqual(value["run"]["conclusion"], "success")
        self.assertEqual(
            [value[key]["path"] for key in (
                "result", "canonical_json", "canonical_markdown", "evidence")],
            [AUDIT_RESULT_PATH, AUDIT_JSON_PATH, AUDIT_MARKDOWN_PATH,
             AUDIT_EVIDENCE_PATH],
        )

    def test_predecessor_execution_is_absent_in_filesystem_and_history(self) -> None:
        self.assertFalse((ROOT / PREDECESSOR_EXECUTION_PATH).exists())
        self.assertEqual(subprocess.check_output(
            ["git", "rev-list", "--all", "--count", "--",
             PREDECESSOR_EXECUTION_PATH], cwd=ROOT, text=True).strip(), "0")

    def test_workflow_is_push_only_first_attempt_read_only_and_pinned(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("pull_request", text)
        self.assertNotIn("continue-on-error", text)
        self.assertNotIn("git push", text)
        self.assertNotIn("contents: write", text)
        self.assertIn("permissions:\n  contents: read", text)
        trigger = text.split("paths:", 1)[1].split("permissions:", 1)[0]
        self.assertIn(EXECUTION_PATH, trigger)
        self.assertNotIn(LOCK_PATH, trigger)
        for token in (
            'test "$GITHUB_RUN_ATTEMPT" = 1',
            'test "$EVENT_FORCED" = false',
            'test "$EVENT_BEFORE" = "$SOURCE_COMMIT"',
            "git rev-list --parents -n 1 HEAD",
            "git diff-tree --no-commit-id --name-status -r HEAD",
            'git rev-list --all --count -- "$EXECUTION_PATH"',
            "git -C campaign archive HEAD^ | tar -x -C source",
        ):
            self.assertIn(token, text)
        pinned = {
            "actions/checkout":
                "11bd71901bbe5b1630ceea73d27597364c9af683",
            "actions/upload-artifact":
                "ea165f8d65b6e75b540449e92b4886f43607fa02",
            "actions/download-artifact":
                "d3f86a106a0bac45b974a628896c90dbdf5c8093",
        }
        uses = re.findall(
            r"(?m)^\s*- uses: (actions/[a-z-]+)@([0-9a-f]{40})$", text)
        self.assertEqual(len(uses), text.count("uses: actions/"))
        self.assertEqual({name for name, _ in uses}, set(pinned))
        for name, revision in uses:
            self.assertEqual(revision, pinned[name])

    def test_workflow_freezes_once_then_validates_complete_stage_barriers(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(text.count("./bin/build_match_value \\"), 1)
        self.assertIn("--samples 16000 --threads 8 --seed \"$TABLE_SEED\"", text)
        self.assertIn("--playout-symmetries 20", text)
        self.assertIn("python3 -m unittest tests/test_match_value_objective3_v2.py",
                      text)
        self.assertIn("source/bin/test_match_value", text)
        after_development = text.split("\n  development_evaluate:\n", 1)[1]
        self.assertNotIn("actions/checkout", after_development)
        self.assertNotRegex(after_development,
                            r"(?m)^\s*(?:make|gcc|cc|clang)\b")

        development = text.split("\n  development_select:\n", 1)[1].split(
            "\n  safety_evaluate:\n", 1)[0]
        safety = text.split("\n  safety_merge:\n", 1)[1].split(
            "\n  final_evaluate:\n", 1)[0]
        final = text.split("\n  final_merge:\n", 1)[1].split(
            "\n  terminal_no_challenge:\n", 1)[0]
        for section, pairs in ((development, 1000), (safety, 200),
                               (final, 2500)):
            self.assertLess(section.index("validate_actor_shards.py"),
                            section.index("merge_arena.py block"))
            self.assertIn(f"--expect-pairs {pairs}", section)
        self.assertEqual(development.count("select-development"), 1)
        self.assertEqual(safety.count("gate-safety"), 1)
        self.assertEqual(final.count("gate-final"), 1)
        self.assertIn("start: [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]",
                      text)
        self.assertIn("needs.safety_merge.outputs.passed == 'true'", text)
        self.assertEqual(text.count("terminal-result \\"), 3)

    def test_actor_construction_changes_only_declared_fields(self) -> None:
        value = build_actors(MAINTAINED_ACTOR, WORLD_CAP,
                             RAW_TABLE_PATH, PROJECTED_TABLE_PATH)
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.assertEqual(value["actors"]["legacy"], MAINTAINED_ACTOR)
        self.assertEqual(set(value["actors"]), {"legacy", *VARIANTS})
        for name in VARIANTS:
            actor = value["actors"][name]
            self.assertEqual(actor, plan["variants"][name]["actor"])
            fields = actor.split(":")
            self.assertEqual(fields[:3],
                             ["rolloutu2", MODEL_PATH, MODEL_PATH])
            tail = fields[3:]
            self.assertEqual(len(tail), 42)
            self.assertEqual(tail[0], "800")
            self.assertEqual(tail[5], "0")
            self.assertEqual(tail[8], "3")
            self.assertEqual(tail[19], "800")
            self.assertEqual(tail[36:41], ["0", "0", "0", "1", "0"])
            self.assertEqual(tail[41], plan["variants"][name]["table_path"])
        with self.assertRaises(EvidenceError):
            build_actors(MAINTAINED_ACTOR, 512,
                         RAW_TABLE_PATH, PROJECTED_TABLE_PATH)
        with self.assertRaises(EvidenceError):
            build_actors(MAINTAINED_ACTOR, WORLD_CAP,
                         "unlocked-raw", PROJECTED_TABLE_PATH)

    def test_table_pair_uses_parameterized_smoke_seed_and_shared_corpus(self) -> None:
        self.assertTrue(SMOKE_TABLE_SEED.startswith(SMOKE_NAMESPACE))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.lcmv"
            projected = root / "projected.lcmv"
            raw.write_bytes(table_bytes(False))
            projected.write_bytes(table_bytes(True))
            value = table_manifest(raw, projected, SMOKE_TABLE_SEED)
            self.assertEqual(value["status"],
                             "complete_valid_single_transition_generation")
            self.assertTrue(value["single_builder_invocation"])
            self.assertTrue(
                value["variants_share_identical_transition_histograms"])
            self.assertFalse(value["raw"]["isotonic_projected"])
            self.assertTrue(value["projected"]["isotonic_projected"])
            self.assertEqual(value["raw"]["controller"],
                             value["projected"]["controller"])
            with self.assertRaises(EvidenceError):
                inspect_table(raw, False, f"{SMOKE_NAMESPACE}0002")
            broken = root / "broken.lcmv"
            snapshot = bytearray(raw.read_bytes())
            snapshot[-1] ^= 1
            broken.write_bytes(snapshot)
            with self.assertRaises(EvidenceError):
                inspect_table(broken, False, SMOKE_TABLE_SEED)
            wrong_profile = root / "wrong-profile.lcmv"
            wrong_profile.write_bytes(table_bytes(False,
                                                   corrupt_profile=True))
            with self.assertRaises(EvidenceError):
                inspect_table(wrong_profile, False, SMOKE_TABLE_SEED)

    def test_development_requires_both_complete_panels_and_projected_tie_break(self) -> None:
        n = DEVELOPMENT_PAIRS
        passing = panel(n, 2 * n, 2 * n, 1, 0,
                        score=0.5, margin=0.001,
                        orientations=(0.475, 0.475))
        panels = {name: json.loads(json.dumps(passing)) for name in VARIANTS}
        actors = build_actors(
            MAINTAINED_ACTOR, WORLD_CAP,
            RAW_TABLE_PATH, PROJECTED_TABLE_PATH)["actors"]
        actors = {name: actors[name] for name in VARIANTS}
        result = development_selection(panels, actors)
        self.assertEqual(result["schema"], DEVELOPMENT_SCHEMA)
        self.assertEqual(result["selected_variant"], TIE_PRIORITY[0])
        self.assertEqual(result["selected_actor"],
                         actors["PROJECTED_ALL_PLY"])
        self.assertTrue(result["challenge_exists"])
        self.assertTrue(result["eligible"])
        self.assertFalse(result["promotion_claim"])

        with self.assertRaises(EvidenceError):
            development_selection({"RAW_ALL_PLY": passing}, actors)
        failed = panel(n, 2 * n - 1, 2 * n + 1, 1, 0,
                       score=0.49, orientations=(0.47, 0.49))
        no_challenge = development_selection(
            {name: failed for name in VARIANTS}, actors)
        self.assertIsNone(no_challenge["selected_variant"])
        self.assertIsNone(no_challenge["selected_actor"])
        self.assertFalse(no_challenge["challenge_exists"])
        self.assertFalse(no_challenge["eligible"])

    def test_safety_exact_boundaries_are_locked(self) -> None:
        n = SAFETY_PAIRS
        passing = panel(n, 2 * n, 2 * n, 1, 0,
                        score=0.5, margin=0.001,
                        orientations=(0.475, 0.475))
        self.assertTrue(safety_gate(passing)["passed"])
        zero_margin = panel(n, 2 * n, 2 * n, 0, 0,
                            score=0.5, margin=0.0,
                            orientations=(0.475, 0.475))
        self.assertFalse(safety_gate(zero_margin)["passed"])
        low_score = panel(n, 2 * n - 1, 2 * n,
                          1, 0, score=0.499,
                          orientations=(0.475, 0.475))
        self.assertFalse(safety_gate(low_score)["passed"])
        low_orientation = panel(n, 2 * n, 2 * n, 1, 0,
                                score=0.5, margin=0.001,
                                orientations=(0.474999, 0.525001))
        self.assertFalse(safety_gate(low_orientation)["passed"])
        capped = panel(n, 2 * n, 2 * n, 1, 0,
                       score=0.5, margin=0.001,
                       orientations=(0.5, 0.5), caps=1)
        self.assertFalse(safety_gate(capped)["passed"])

    def test_final_requires_both_strict_lcbs_and_each_orientation(self) -> None:
        n = FINAL_PAIRS
        passing = panel(
            n, 2 * n + 200, 2 * n - 200, 1000, 0,
            score=0.51, score_se=0.004,
            margin=0.5, margin_se=0.1,
            orientations=(0.51, 0.51),
        )
        decision = final_gate(passing)
        self.assertTrue(decision["promotion_gate_passed"])
        self.assertAlmostEqual(
            decision["candidate_result"]["match_score_lower_bound"],
            0.51 - CRITICAL_Z * 0.004)
        self.assertAlmostEqual(
            decision["candidate_result"]["margin_lower_bound"],
            0.5 - CRITICAL_Z * 0.1)

        score_equality = panel(
            n, 2 * n + 200, 2 * n - 200, 1000, 0,
            score=0.5 + CRITICAL_Z * 0.004, score_se=0.004,
            margin=0.5, margin_se=0.1,
            orientations=(0.51, 0.51),
        )
        self.assertFalse(final_gate(score_equality)["promotion_gate_passed"])
        margin_equality = panel(
            n, 2 * n + 200, 2 * n - 200, 1000, 0,
            score=0.51, score_se=0.004,
            margin=CRITICAL_Z * 0.1, margin_se=0.1,
            orientations=(0.51, 0.51),
        )
        self.assertFalse(final_gate(margin_equality)["promotion_gate_passed"])
        orientation_equality = panel(
            n, 2 * n + 200, 2 * n - 200, 1000, 0,
            score=0.51, score_se=0.004,
            margin=0.5, margin_se=0.1,
            orientations=(0.5, 0.52),
        )
        self.assertFalse(
            final_gate(orientation_equality)["promotion_gate_passed"])
        with self.assertRaises(EvidenceError):
            final_gate(passing, math.nextafter(CRITICAL_Z, math.inf))

    def test_terminal_result_is_fail_closed_and_diagnostic_is_never_selection(self) -> None:
        baseline = {"spec": MAINTAINED_ACTOR}
        execution = {"subject": {"baseline": baseline}}
        no_challenge = {
            "schema": DEVELOPMENT_SCHEMA,
            "selected_actor": None,
            "eligible": False,
            "challenge_exists": False,
        }
        evidence = terminal_evidence("development")
        result = terminal_result(execution, no_challenge, None, None, evidence)
        self.assertEqual(result["schema"], RESULT_SCHEMA)
        self.assertEqual(result["winner_actor"], MAINTAINED_ACTOR)
        self.assertFalse(result["promotion_gate_passed"])
        self.assertFalse(result["locked_validation_relaxed"])
        self.assertFalse(result["diagnostic_audit_used_for_selection"])

        candidate = build_actors(
            MAINTAINED_ACTOR, WORLD_CAP,
            RAW_TABLE_PATH, PROJECTED_TABLE_PATH)["actors"]["RAW_ALL_PLY"]
        challenger = dict(no_challenge, selected_actor=candidate,
                          eligible=True, challenge_exists=True)
        safety = {
            "schema": "lc-match-value-objective3-v2-safety-gate-v1",
            "passed": True, "candidate": candidate,
            "baseline": MAINTAINED_ACTOR,
        }
        final = {
            "schema": "lc-match-value-objective3-v2-final-gate-v1",
            "promotion_gate_passed": True, "candidate": candidate,
            "baseline": MAINTAINED_ACTOR,
        }
        promoted = terminal_result(
            execution, challenger, safety, final, terminal_evidence("final"))
        self.assertEqual(promoted["winner_actor"], candidate)
        with self.assertRaises(EvidenceError):
            terminal_result(execution, challenger,
                            dict(safety, passed=False), final,
                            terminal_evidence("final"))

    def test_evidence_manifest_is_sorted_complete_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z").write_text("z", encoding="utf-8")
            (root / "a").write_text("a", encoding="utf-8")
            rows = evidence_manifest(root)
            self.assertEqual([row["path"] for row in rows], ["a", "z"])
            (root / "link").symlink_to(root / "a")
            with self.assertRaises(EvidenceError):
                evidence_manifest(root)

    def test_cli_help_exposes_all_frozen_operations(self) -> None:
        help_text = subprocess.check_output(
            ["python3", "tools/match_value_objective3_v2.py", "--help"],
            cwd=ROOT, text=True)
        for command in (
            "validate-plan", "prepare-definition-lock", "prepare-execution",
            "guard-execution", "table-manifest", "actors",
            "post-build-manifest", "select-development", "gate-safety",
            "gate-final", "terminal-result",
        ):
            self.assertIn(command, help_text)


if __name__ == "__main__":
    unittest.main()
