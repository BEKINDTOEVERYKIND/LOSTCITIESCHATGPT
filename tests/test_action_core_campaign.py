"""Fail-closed contracts for the inert action-core shortlist campaign."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tools.action_core_campaign import (
    ACTION_CORE_FIELD,
    CFLAGS,
    EXECUTION_PATH,
    FINAL_BASELINE_SEED,
    FINAL_CANDIDATE_SEED,
    FINAL_PAIRS,
    FINAL_STARTS,
    LDFLAGS,
    MODEL_PATH,
    PLAN_PATH,
    ROLLOUT_FIELDS,
    SAFETY_BASELINE_SEED,
    SAFETY_CANDIDATE_SEED,
    SAFETY_PAIRS,
    SAFETY_STARTS,
    SOURCE_FILES,
    WORKFLOW_PATH,
    WORLD_RESULT_PATH,
    build_actor_pair,
    expected_execution,
    guard_execution,
    normalized_rollout,
    strict_json,
    validate_plan,
)
from tools.match_value_campaign import BASELINE_512, CANDIDATE_800
from tools.merge_arena import EvidenceError


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / PLAN_PATH
WORKFLOW = ROOT / WORKFLOW_PATH
EXECUTION = ROOT / EXECUTION_PATH
TEMPLATE = ROOT / "data/experiments/locked_action_core_shortlist_execution.template.json"


def assert_workflow_mapping_keys_are_unique(test: unittest.TestCase,
                                            text: str) -> None:
    """Strictly parse the workflow's YAML subset and reject duplicate keys."""
    key_pattern = re.compile(
        r"^(?P<indent> *)(?P<list>- )?"
        r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*):(?P<value>.*)$"
    )
    stack: list[tuple[int, object]] = []
    seen: dict[object, set[str]] = {"root": set()}
    block_indent: int | None = None
    for number, raw in enumerate(text.splitlines(), 1):
        test.assertNotIn("\t", raw, f"workflow line {number} contains a tab")
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if block_indent is not None:
            if indent > block_indent:
                continue
            block_indent = None
        match = key_pattern.fullmatch(raw)
        if match is None and raw.lstrip().startswith("- "):
            while stack and stack[-1][0] >= indent:
                stack.pop()
            continue
        test.assertIsNotNone(match, f"unsupported YAML line {number}: {raw!r}")
        assert match is not None
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else "root"
        if match.group("list"):
            parent = ("item", number)
            seen[parent] = set()
            stack.append((indent, parent))
        key = match.group("key")
        test.assertNotIn(
            key, seen.setdefault(parent, set()),
            f"duplicate YAML key {key!r} on line {number}",
        )
        seen[parent].add(key)
        value = match.group("value").strip()
        if value in {"", "|", ">"}:
            node = ("mapping", number, key)
            seen[node] = set()
            stack.append((indent, node))
        if value in {"|", ">"}:
            block_indent = indent


class ActionCoreCampaignTests(unittest.TestCase):
    def test_plan_is_one_bounded_candidate_not_legal_move_enumeration(self) -> None:
        plan = strict_json(PLAN)
        validate_plan(plan)
        method = plan["method"]
        self.assertEqual(method["sole_candidate_action_core_count"], 3)
        self.assertEqual(method["root_width"], 5)
        self.assertEqual(method["complete_move_policy_floor"], 0.02)
        self.assertEqual(method["maximum_moves_evaluated"], 5)
        self.assertFalse(method["evaluates_all_legal_moves"])
        self.assertEqual(method["only_actor_field_changed"], "action_core_count")
        why = plan["why_three_is_the_only_confirmatory_candidate"]
        self.assertIn("five-move budget", why["structural_reason"])
        self.assertIn("not a claim", why["claim_limit"])
        self.assertTrue(all((ROOT / path).is_file() for path in SOURCE_FILES))
        # SOURCE_FILES is the immutable inventory frozen with the historical
        # action-core campaign.  rollout5's separately versioned policy-cost
        # ABIs and the standalone accuracy-only history-belief components were
        # added later and must neither be retroactively hashed into that
        # one-shot definition nor make its revalidation depend on the current
        # checkout's dynamic src glob.
        current_src = {
            str(path.relative_to(ROOT))
            for path in (ROOT / "src").glob("*.[ch]")
        }
        frozen_src = {
            path for path in SOURCE_FILES if path.startswith("src/")
        }
        self.assertEqual(
            current_src - frozen_src,
            {
                "src/policy_cost.c",
                "src/policy_cost.h",
                "src/policy_cost_v3.c",
                "src/policy_cost_v3.h",
                "src/policy_cost_v4.c",
                "src/policy_cost_v4.h",
                "src/policy_cost_v5.c",
                "src/policy_cost_v5.h",
                "src/policy_cost_v6.c",
                "src/policy_cost_v6.h",
                "src/policy_cost_v7.c",
                "src/policy_cost_v7.h",
                "src/history_belief_exclusion.c",
                "src/history_belief_exclusion.h",
                "src/history_belief_model.c",
                "src/history_belief_model.h",
            },
        )
        self.assertFalse(frozen_src - current_src)

    def test_no_user_position_or_training_data_enters_campaign(self) -> None:
        plan = strict_json(PLAN)
        firewall = plan["state_and_data_firewall"]
        self.assertFalse(firewall["user_commented_states_used"])
        self.assertFalse(firewall["position_specific_selection_used"])
        self.assertFalse(firewall["training_used"])
        self.assertEqual(
            firewall["match_source"],
            "fresh seeded random deals generated only by arena",
        )
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for forbidden in (
            "data/probes", "flagged_ply_probe", "commented_ply_eval",
            "showgame", "viewer",
        ):
            self.assertNotIn(forbidden, workflow)

    def test_actor_derivation_changes_exactly_field_35(self) -> None:
        for winner, worlds in ((BASELINE_512, 512), (CANDIDATE_800, 800)):
            with self.subTest(worlds=worlds):
                pair = build_actor_pair(winner)
                self.assertEqual(pair["baseline"], winner)
                self.assertEqual(pair["world_cap"], worlds)
                self.assertEqual(pair["rollout_field_count"], 42)
                self.assertEqual(pair["changed_fields"], ["action_core_count"])
                self.assertFalse(pair["candidate_limit"]["all_legal_moves_evaluated"])
                self.assertEqual(
                    pair["candidate_limit"]["maximum_complete_moves_evaluated"], 5)
                _, _, baseline = normalized_rollout(pair["baseline"])
                _, _, candidate = normalized_rollout(pair["candidate"])
                diffs = [i for i, values in enumerate(zip(baseline, candidate))
                         if values[0] != values[1]]
                self.assertEqual(diffs, [ACTION_CORE_FIELD])
                self.assertEqual(candidate[ACTION_CORE_FIELD], "3")
                self.assertEqual(len(baseline), len(ROLLOUT_FIELDS))
                self.assertEqual(pair["candidate"].split(":")[2], str(worlds))
                self.assertEqual(pair["candidate"].split(":")[3:5], ["5", "0.02"])

    def test_actor_derivation_rejects_extensions_or_prior_core_setting(self) -> None:
        changed = BASELINE_512.split(":")
        changed[2 + ACTION_CORE_FIELD] = "2"
        with self.assertRaisesRegex(EvidenceError, "already enables"):
            build_actor_pair(":".join(changed))
        with self.assertRaisesRegex(EvidenceError, "late-field extension"):
            build_actor_pair(BASELINE_512 + ":1")
        with self.assertRaisesRegex(EvidenceError, "expected uniform actor"):
            build_actor_pair(BASELINE_512.replace("rolloutu:", "rollout:", 1))

    def test_schedule_is_fresh_reciprocal_safety_then_one_final(self) -> None:
        plan = strict_json(PLAN)
        safety = plan["safety_screen"]
        final = plan["final_promotion"]
        self.assertEqual(safety["pairs_per_orientation"], SAFETY_PAIRS)
        self.assertEqual(safety["pair_starts"], SAFETY_STARTS)
        self.assertEqual(final["pairs_per_orientation"], FINAL_PAIRS)
        self.assertEqual(final["pair_starts"], FINAL_STARTS)
        self.assertTrue(final["execute_only_if_safety_passes"])
        seeds = {
            SAFETY_CANDIDATE_SEED, SAFETY_BASELINE_SEED,
            FINAL_CANDIDATE_SEED, FINAL_BASELINE_SEED,
        }
        self.assertEqual(len(seeds), 4)
        for path in list((ROOT / "data/experiments").glob("*.json")) + \
                list((ROOT / ".github/workflows").glob("*.yml")):
            if path in {PLAN, WORKFLOW, TEMPLATE, EXECUTION}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for seed in seeds:
                self.assertNotIn(seed, text, f"seed reused in {path}")

    def test_workflow_is_inert_addendum_only_and_compile_once(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert_workflow_mapping_keys_are_unique(self, text)
        trigger = text.split("permissions:", 1)[0]
        self.assertIn(EXECUTION_PATH, trigger)
        self.assertNotIn("workflow_dispatch", trigger)
        self.assertNotIn("schedule:", trigger)
        self.assertNotIn("pull_request", trigger)
        self.assertIn('test "$GITHUB_RUN_ATTEMPT" = 1', text)
        self.assertIn('test "$EVENT_FORCED" = false', text)
        self.assertIn("git -C campaign archive HEAD^ | tar -x -C source", text)
        self.assertIn("git diff-tree --no-commit-id --name-status -r HEAD", text)
        self.assertIn("guard-execution", text)
        self.assertIn("data/experiments/world800_result.json", text)
        self.assertEqual(text.count("make -C source"), 1)
        later = text.split("  safety_evaluate:", 1)[1]
        self.assertNotIn("actions/checkout@", later)
        self.assertNotIn("make -C", later)
        self.assertNotRegex(later, r"\b(?:gcc|clang|cc)\b.*(?:-o|-c)")
        self.assertIn("needs.safety_merge.outputs.passed == 'true'", text)
        self.assertIn("--pairs-per-orientation 2500", text)
        self.assertIn("--mode final", text)
        self.assertIn("--gate-z 1.645", text)
        self.assertNotIn("contents: write", text)

    def test_final_evidence_retains_raw_rows_timings_hashes_and_bindings(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        final_eval = text[text.index("  final_evaluate:"):
                          text.index("  final_merge:")]
        final_merge = text[text.index("  final_merge:"):]
        self.assertIn('OUT="raw/${ORIENTATION}-${PAIR_START}"', final_eval)
        self.assertIn('--raw-pairs "${OUT}.jsonl" --raw-only', final_eval)
        self.assertIn('-o "${OUT}.time"', final_eval)
        self.assertIn('sha256sum "${OUT}.jsonl" > "${OUT}.sha256"', final_eval)
        self.assertIn("name: action-core-shortlist-final-evidence", final_merge)
        artifact = final_merge.split(
            "name: action-core-shortlist-final-evidence", 1)[1]
        for directory in ("downloads", "merged", "bindings"):
            self.assertRegex(artifact, rf"(?m)^\s+{directory}$")
        self.assertIn("cp -r evaluator/bindings/. bindings/", final_merge)
        self.assertIn("EVALUATOR_SHA256SUMS.txt", final_merge)

    def test_workflow_waits_for_complete_stage_before_any_merge(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        safety_merge = text.index("  safety_merge:")
        final_eval = text.index("  final_evaluate:")
        final_merge = text.index("  final_merge:")
        self.assertGreater(safety_merge, text.index("  safety_evaluate:"))
        self.assertGreater(final_merge, final_eval)
        safety = text[safety_merge:final_eval]
        final = text[final_merge:]
        for section, expected in ((safety, 200), (final, 2500)):
            validator = section.index("validate_actor_shards.py")
            first_merge = section.index("merge_arena.py block")
            reciprocal = section.index("merge_arena.py reciprocal")
            gate = section.index("gate_actor_panel.py")
            self.assertLess(validator, first_merge)
            self.assertLess(first_merge, reciprocal)
            self.assertLess(reciprocal, gate)
            self.assertIn(f"--expect-pairs {expected}", section)
        self.assertIn("needs.safety_evaluate.result == 'success'", safety)
        self.assertIn("needs.final_evaluate.result == 'success'", final)

    def test_action_pins_and_shell_blocks_are_syntactically_valid(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        uses = re.findall(r"uses:\s*([^\s]+)", text)
        self.assertTrue(uses)
        for value in uses:
            self.assertRegex(value, r"^actions/(?:checkout|upload-artifact|download-artifact)@[0-9a-f]{40}$")
        lines = text.splitlines()
        blocks: list[str] = []
        index = 0
        while index < len(lines):
            match = re.match(r"^(\s*)run:\s*\|\s*$", lines[index])
            if not match:
                index += 1
                continue
            indent = len(match.group(1))
            index += 1
            body: list[str] = []
            while index < len(lines):
                line = lines[index]
                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    break
                body.append(line[indent + 2:] if len(line) >= indent + 2 else "")
                index += 1
            blocks.append("\n".join(body) + "\n")
        self.assertGreaterEqual(len(blocks), 9)
        for ordinal, block in enumerate(blocks):
            result = subprocess.run(
                ["bash", "-n"], input=block, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0,
                             f"shell block {ordinal}: {result.stderr}")

    def _fixture_root(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp = tempfile.TemporaryDirectory(prefix="action-core-campaign-")
        root = Path(temp.name)
        for relative in set(SOURCE_FILES) | {
                PLAN_PATH, WORKFLOW_PATH, MODEL_PATH, WORLD_RESULT_PATH}:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = ROOT / relative
            if source.exists():
                shutil.copyfile(source, target)
            else:
                target.write_text("synthetic world result\n", encoding="utf-8")
        return temp, root

    def test_expected_execution_mechanically_uses_world_winner(self) -> None:
        for passed, winner, worlds in (
                (False, BASELINE_512, 512), (True, CANDIDATE_800, 800)):
            temp, root = self._fixture_root()
            self.addCleanup(temp.cleanup)
            with mock.patch(
                    "tools.action_core_campaign._world_result",
                    return_value=({}, passed, winner, worlds)):
                expected = expected_execution(root, "1" * 40, "2" * 40)
            self.assertEqual(expected["actors"]["baseline"], winner)
            self.assertEqual(expected["actors"]["world_cap"], worlds)
            self.assertEqual(
                expected["authoritative_world800_result"]["promotion_gate_passed"],
                passed,
            )
            self.assertEqual(expected["build"]["cflags"], CFLAGS)
            self.assertEqual(expected["build"]["ldflags"], LDFLAGS)

    def test_execution_guard_is_exact_and_rejects_drift(self) -> None:
        temp, root = self._fixture_root()
        self.addCleanup(temp.cleanup)
        execution = root / EXECUTION_PATH
        execution.parent.mkdir(parents=True, exist_ok=True)
        patched = mock.patch(
            "tools.action_core_campaign._world_result",
            return_value=({}, False, BASELINE_512, 512),
        )
        with patched:
            expected = expected_execution(root, "3" * 40, "4" * 40)
            execution.write_text(json.dumps(expected), encoding="utf-8")
            validation = guard_execution(
                root, execution, "3" * 40, "4" * 40)
            self.assertTrue(validation["valid"])
            expected["actors"]["candidate_limit"]["root_width"] = 6
            execution.write_text(json.dumps(expected), encoding="utf-8")
            with self.assertRaisesRegex(EvidenceError, "does not exactly match"):
                guard_execution(root, execution, "3" * 40, "4" * 40)

    def test_template_cannot_launch(self) -> None:
        value = strict_json(TEMPLATE)
        self.assertEqual(value["status"], "TEMPLATE_ONLY_DOES_NOT_LAUNCH")
        self.assertNotEqual(TEMPLATE.name, Path(EXECUTION_PATH).name)
        if EXECUTION.exists():
            # Once legitimately launched, retain a durable history check rather
            # than making the test suite fail merely because evidence exists.
            launches = subprocess.check_output(
                ["git", "rev-list", "--all", "--", EXECUTION_PATH],
                cwd=ROOT, text=True,
            ).splitlines()
            self.assertEqual(len(launches), 1)


if __name__ == "__main__":
    unittest.main()
