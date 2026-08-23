"""Fail-closed contracts for the one-shot exact-17 audit launch."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import commented_ply_execution as execution


PLAN = ROOT / execution.PLAN_PATH
WORKFLOW = ROOT / execution.WORKFLOW_PATH


def workflow_keys_are_unique(test: unittest.TestCase, text: str) -> None:
    pattern = re.compile(
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
        match = pattern.fullmatch(raw)
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
        test.assertNotIn(key, seen.setdefault(parent, set()))
        seen[parent].add(key)
        value = match.group("value").strip()
        if value in {"", "|", ">"}:
            node = ("mapping", number, key)
            seen[node] = set()
            stack.append((indent, node))
        if value in {"|", ">"}:
            block_indent = indent


def workflow_shell_blocks_parse(test: unittest.TestCase, text: str) -> None:
    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)run:\s*\|\s*$", lines[index])
        if match is None:
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
    test.assertGreaterEqual(len(blocks), 6)
    for ordinal, block in enumerate(blocks):
        result = subprocess.run(
            ["bash", "-n"], input=block, text=True,
            capture_output=True, check=False,
        )
        test.assertEqual(
            result.returncode, 0,
            f"shell block {ordinal}: {result.stderr}",
        )


class CommentedPlyExecutionTests(unittest.TestCase):
    maxDiff = None

    @staticmethod
    def final_binding() -> dict:
        actor = {
            "spec": (
                "rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:"
                "3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:"
                "0:0:0:0:0:1"
            ),
            "checkpoints": [],
        }
        return {
            "path": execution.FINAL_RESULT_PATH,
            "sha256": "a" * 64,
            "selection_mode": "component_final",
            "source_commit": "2" * 40,
            "source_tree": "3" * 40,
            "decisive_result": {"path": "result.json", "sha256": "b" * 64},
            "authoritative_results": [{
                "path": "result.json", "sha256": "b" * 64,
                "role": "final_decision",
            }],
            "promotion_gate_passed": False,
            "reference": actor,
            "challenger": {**actor, "spec": actor["spec"] + ":challenger"},
            "winner": actor,
            "actor_assets": {"reference": [], "challenger": [], "winner": []},
            "no_change": True,
        }

    def test_plan_is_the_exact_ordered_17_case_definition(self) -> None:
        plan = execution.strict_json(PLAN)
        self.assertEqual(plan["schema"], "lc-commented-ply-audit-plan-v1")
        self.assertEqual(plan["case_definition_sha256"],
                         "c065a0d0e86f1db392b9e6e7382518cff947770be519417d899c21a965b223b5")
        cases, artifacts = execution._case_binding(ROOT, plan)
        self.assertEqual(cases["case_ids"], list(execution.CASE_IDS))
        self.assertEqual(cases["case_count"], 17)
        self.assertEqual(cases["action_panel_cases"], 17)
        self.assertEqual(cases["nominated_action_cases"], 16)
        self.assertEqual(cases["fixed_k_belief_cases"], 1)
        self.assertEqual(len(artifacts), 18)
        rows = plan["cases"]
        self.assertEqual(len(rows), 17)
        self.assertEqual(
            [(row["source_seed"], row["ply"]) for row in rows],
            [
                ("2214615196", 3), ("2214615196", 4),
                ("2214615196", 8), ("2214615196", 10),
                ("2214615196", 12), ("2214615196", 13),
                ("2214615196", 16), ("2214615196", 20),
                ("5726968372613385", 14), ("5726968372613385", 15),
                ("5726968372613385", 17), ("5726968372613385", 32),
                ("725402798", 21), ("725402798", 22),
                ("725402798", 23), ("725402798", 25),
                ("95647345759839", 44),
            ],
        )
        self.assertEqual(rows[3]["min_worlds"], 2048)
        self.assertTrue(all(
            row["min_worlds"] == (2048 if index == 3 else 1024)
            for index, row in enumerate(rows)
        ))
        self.assertEqual(rows[5]["belief_card"], "Y9")
        self.assertEqual(rows[5]["candidates"], [])

    def test_expected_execution_mechanically_binds_winner_and_assets(self) -> None:
        value = execution.expected_execution(
            ROOT, "4" * 40, "5" * 40,
            final_binding=self.final_binding(),
        )
        self.assertEqual(value["schema"], execution.SCHEMA)
        self.assertEqual(value["subject"]["actor"], self.final_binding()["winner"])
        self.assertEqual(value["continuation"]["actor"],
                         "policy:data/champion.bin:0:20")
        self.assertEqual(value["continuation"]["symmetries"], 20)
        self.assertEqual(value["continuation"]["scope"],
                         "full_remaining_three_round_match")
        self.assertEqual(value["audit"]["default_paired_worlds"], 1024)
        self.assertEqual(value["audit"]["ui_221_p10_paired_worlds"], 2048)
        self.assertTrue(value["audit"]["diagnostic_only"])
        self.assertIsNone(value["results"])
        bound_paths = {row["path"] for row in value["tools"]}
        self.assertIn("tools/audit_commented_plies.py", bound_paths)
        self.assertIn("tools/commented_ply_eval.c", bound_paths)
        self.assertNotIn("tools/flagged_ply_audit.py", bound_paths)

    def test_prepare_is_atomic_canonical_and_no_clobber(self) -> None:
        value = {"schema": "fixture"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / execution.EXECUTION_PATH
            with mock.patch.object(execution, "expected_execution",
                                   return_value=value):
                created = execution.prepare_execution(
                    root, output, "1" * 40, "2" * 40)
                self.assertEqual(created, value)
                self.assertEqual(json.loads(output.read_text()), value)
                snapshot = output.read_bytes()
                with self.assertRaises(execution.ExecutionError):
                    execution.prepare_execution(
                        root, output, "1" * 40, "2" * 40)
                self.assertEqual(output.read_bytes(), snapshot)
                with self.assertRaises(execution.ExecutionError):
                    execution.prepare_execution(
                        root, root / "wrong.json", "1" * 40, "2" * 40)

    def test_workflow_is_add_only_compile_once_and_exact_17(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        workflow_keys_are_unique(self, text)
        workflow_shell_blocks_parse(self, text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("continue-on-error", text)
        self.assertIn('test "$GITHUB_RUN_ATTEMPT" = 1', text)
        self.assertIn('test "$EVENT_FORCED" = false', text)
        self.assertIn("guard-execution", text)
        self.assertIn("git -C campaign archive HEAD^", text)
        self.assertEqual(text.count("make -C source"), 1)
        self.assertNotIn("flagged_ply_audit.py", text)
        self.assertNotIn("flagged-ply-audit.yml", text)
        matrix = text.split("        case_id:\n", 1)[1].split(
            "    steps:\n", 1)[0]
        actual = re.findall(r"^          - (.+)$", matrix, re.MULTILINE)
        self.assertEqual(actual, list(execution.CASE_IDS))
        self.assertIn("--case \"$CASE_ID\"", text)
        self.assertIn("--merge shards/*.json", text)
        self.assertIn("--worlds \"$DEFAULT_WORLDS\"", text)
        self.assertIn("EXACT_SYMMETRIES: '20'", text)
        self.assertIn("full_remaining_three_round_match", text)
        self.assertIn('contract.get("base_paired_worlds")', text)
        self.assertIn('contract.get("exact_policy_teacher")', text)
        self.assertIn('counterfactual.get("requested_worlds")', text)
        self.assertIn('counterfactual.get("completed_worlds")', text)
        self.assertIn('counterfactual.get("cap_hits")', text)
        self.assertIn('continuation.get("kind") != "exact_policy_argmax"', text)
        self.assertIn('continuation.get("checkpoint") != "data/champion.bin"', text)
        self.assertIn('counterfactual.get("policy_reference_candidates") != 2', text)
        self.assertNotIn('counterfactual.get("worlds"', text)
        self.assertNotIn('counterfactual.get("capped_matches")', text)
        self.assertIn("commented-ply-audit-complete-evidence", text)
        self.assertIn("find . -type f ! -name SHA256SUMS.txt", text)
        pinned = {
            "actions/checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
            "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
            "actions/download-artifact": "d3f86a106a0bac45b974a628896c90dbdf5c8093",
        }
        uses = re.findall(
            r"(?m)^\s*- uses: (actions/[a-z-]+)@([0-9a-f]{40})$", text)
        self.assertEqual(len(uses), text.count("uses: actions/"))
        for name, revision in uses:
            self.assertEqual(revision, pinned[name])


if __name__ == "__main__":
    unittest.main()
