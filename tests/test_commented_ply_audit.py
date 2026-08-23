"""Focused contracts for the explicit commented-ply diagnostic."""

from __future__ import annotations

import importlib.util
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "audit_commented_plies.py"
SPEC = importlib.util.spec_from_file_location("audit_commented_plies", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class CommentedPlyAuditTests(unittest.TestCase):
    POLICY_ACTOR = "policy:data/champion.bin:0:20"

    @staticmethod
    def with_complete_deck(text: str, *, reverse: bool = False) -> str:
        full = [
            f"{suit}{rank}"
            for suit in "YBWGR"
            for rank in ("x", "x", "x", "2", "3", "4", "5", "6",
                         "7", "8", "9", "10")
        ]
        remaining = Counter(full)
        for line in text.splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0].startswith("hand"):
                visible = fields[1:]
            elif fields[0] == "exp":
                visible = fields[3:]
            elif fields[0] == "pile":
                visible = fields[2:]
            else:
                continue
            for card in visible:
                remaining[card] -= 1
                if remaining[card] < 0:
                    raise AssertionError(f"invalid fixture card {card}")
        deck: list[str] = []
        for card in full:
            if remaining[card] > 0:
                deck.append(card)
                remaining[card] -= 1
        if reverse:
            deck.reverse()
        return text.rstrip() + "\ndeck " + " ".join(deck) + "\n"

    def helper(self, state: Path, *extra: str) -> dict:
        command = [
            str(ROOT / "bin" / "commented_ply_eval"),
            "--state", str(state),
            "--actor", self.POLICY_ACTOR,
            "--net", "data/champion.bin",
            "--seed", "991337",
            "--worlds", "2",
            "--symmetries", "20",
            *extra,
        ]
        result = subprocess.run(
            command, cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return json.loads(result.stdout)

    def test_inventory_is_exactly_the_17_explicit_comments(self) -> None:
        expected = {
            2214615196: {3, 4, 8, 10, 12, 13, 16, 20},
            5726968372613385: {14, 15, 17, 32},
            725402798: {21, 22, 23, 25},
            95647345759839: {44},
        }
        actual: dict[int, set[int]] = {}
        for case in audit.CASES:
            actual.setdefault(case.source_seed, set()).add(case.ply)
            self.assertTrue((ROOT / case.state).is_file())
            if case.candidates:
                self.assertGreaterEqual(
                    case.min_worlds, audit.MIN_PAIRED_WORLDS
                )
            self.assertNotIn("train", case.state.lower())
        self.assertEqual(len(audit.CASES), 17)
        self.assertEqual(actual, expected)

        p14 = next(case for case in audit.CASES if case.case_id == "showcase-572-p14")
        self.assertEqual(
            p14.candidates,
            ("R4 d deck", "G7 p deck", "B3 d deck"),
        )
        p10 = next(case for case in audit.CASES if case.case_id == "ui-221-p10")
        self.assertGreaterEqual(p10.min_worlds, 2048)

    def test_confidence_interval_that_spans_zero_is_inconclusive(self) -> None:
        self.assertEqual(audit.descriptive_signal(0.19, 1.17), "inconclusive")
        self.assertEqual(audit.descriptive_signal(2.0, 0.25), "alternative_ahead")
        self.assertEqual(audit.descriptive_signal(-2.0, 0.25), "reference_ahead")

    def test_helper_output_is_deterministic_and_information_safe(self) -> None:
        original = ROOT / "data" / "probes" / "ui_seed725402798_p21.state"
        text = original.read_text()
        self.assertIn("hand1 Yx Y3 Y10", text)
        # Y3 is hidden from the mover here. Swap it for the unseen Y2 while
        # preserving the opponent hand size; no decision-time result may move.
        mutated_text = text.replace("hand1 Yx Y3 Y10", "hand1 Yx Y2 Y10", 1)
        with tempfile.TemporaryDirectory(prefix="lc-commented-audit-") as tmp:
            completed = Path(tmp) / "complete.state"
            mutated = Path(tmp) / "mutated.state"
            completed.write_text(self.with_complete_deck(text))
            mutated.write_text(self.with_complete_deck(
                mutated_text, reverse=True
            ))
            args = (
                "--candidate", "Bx d deck",
                "--candidate", "G5 p deck",
            )
            first = self.helper(completed, *args)
            repeated = self.helper(completed, *args)
            changed_hidden_truth = self.helper(mutated, *args)
        self.assertEqual(first, repeated)
        self.assertEqual(first["state"]["input_deck_entries"], 24)
        self.assertEqual(changed_hidden_truth["state"]["input_deck_entries"], 24)
        first["state"]["path"] = "STATE"
        changed_hidden_truth["state"]["path"] = "STATE"
        self.assertEqual(first, changed_hidden_truth)
        continuation = first["counterfactual"]["continuation"]
        self.assertTrue(continuation["exact_group_average"])
        self.assertFalse(continuation["recursive_actor"])

    def test_p13_reports_exact_k_posterior_programmatically(self) -> None:
        state = ROOT / "data" / "probes" / "ui_seed2214615196_p13.state"
        result = self.helper(
            state,
            "--belief", "--belief-alpha", "1.15",
            "--belief-card", "Y9",
        )
        belief = result["belief"]
        self.assertTrue(belief["valid"])
        self.assertEqual(belief["kind"], "fixed_k")
        self.assertEqual(belief["need"], 8)
        self.assertAlmostEqual(belief["marginal_sum"], 8.0, places=5)
        self.assertAlmostEqual(belief["uniform_marginal"], 0.2, places=8)
        self.assertEqual(belief["target"]["card"], "Y9")
        self.assertGreater(belief["target"]["marginal"], 0.2)
        cards = {row["card"]: row for row in belief["cards"]}
        self.assertTrue(cards["B10"]["held"])
        self.assertTrue(cards["B9"]["held"])

    def test_public_driver_refuses_underpowered_disputed_panel(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1024"):
            audit.build_audit(
                helper=ROOT / "bin" / "commented_ply_eval",
                actor_spec=self.POLICY_ACTOR,
                net_path=ROOT / "data" / "champion.bin",
                worlds=1023,
            )

    def test_actor_selected_move_is_always_counterfactually_graded(self) -> None:
        state = ROOT / "data" / "probes" / "ui_seed95647345759839_p44.state"
        result = self.helper(state, "--candidate", "W10 p deck")
        moves = [row["move"] for row in result["counterfactual"]["candidates"]]
        self.assertEqual(moves[0], "W10 p deck")
        self.assertNotEqual(result["actor"]["selected"], "W10 p deck")
        self.assertIn(result["actor"]["selected"], moves)
        self.assertTrue(result["counterfactual"]["actor_selected_included"])


if __name__ == "__main__":
    unittest.main()
