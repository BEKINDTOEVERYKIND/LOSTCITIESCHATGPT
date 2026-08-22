"""Focused contracts for the actor-aware offline belief annotator."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/history_belief.py"
SPEC = importlib.util.spec_from_file_location("history_belief", MODULE_PATH)
assert SPEC and SPEC.loader
history_belief = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history_belief)


def fixture() -> dict:
    p0 = ["Yx", "Yx", "Y3", "Y9", "Bx", "G5", "R8", "R9"]
    p1 = ["Bx", "Wx", "W8", "W10", "G7", "R3", "R7", "R10"]
    return {
        "meta": {
            "actor": (
                "rolloutu:data/champion.bin:512:4:0.02:0:1:20:0:0:0:0:"
                "3.5:2:2:20:0:0:20:1:0:512:1"
            )
        },
        "plies": [
            {
                "n": 1,
                "round": 0,
                "round_ply": 0,
                "player": 0,
                "cum": [0, 0],
                "hands": [p0, p1],
                "move": {
                    "card": "Yx",
                    "act": "play",
                    "draw": "deck",
                    "drawn": "W5",
                },
            },
            {
                "n": 2,
                "round": 0,
                "round_ply": 1,
                "player": 1,
                "cum": [0, 0],
                "hands": [
                    ["Yx", "Y3", "Y9", "Bx", "W5", "G5", "R8", "R9"],
                    p1,
                ],
                "move": {
                    "card": "Wx",
                    "act": "play",
                    "draw": "deck",
                    "drawn": "R6",
                },
            },
            {
                "n": 3,
                "round": 0,
                "round_ply": 2,
                "player": 0,
                "cum": [0, 0],
                "hands": [
                    ["Yx", "Y3", "Y9", "Bx", "W5", "G5", "R8", "R9"],
                    ["Bx", "W8", "W10", "G7", "R3", "R6", "R7", "R10"],
                ],
                "move": {
                    "card": "Yx",
                    "act": "play",
                    "draw": "deck",
                    "drawn": "Y10",
                },
            },
            {
                "n": 4,
                "round": 0,
                "round_ply": 3,
                "player": 1,
                "cum": [0, 0],
                "hands": [
                    ["Y3", "Y9", "Y10", "Bx", "W5", "G5", "R8", "R9"],
                    ["Bx", "W8", "W10", "G7", "R3", "R6", "R7", "R10"],
                ],
                "move": {
                    "card": "Bx",
                    "act": "discard",
                    "draw": "deck",
                    "drawn": "W7",
                },
            },
        ],
    }


class HistoryBeliefTests(unittest.TestCase):
    maxDiff = None

    def test_scrubbed_view_excludes_hidden_and_future_information(self) -> None:
        original = fixture()
        mutated = copy.deepcopy(original)
        for record in mutated["plies"]:
            record["hands"][0] = ["Gx"] * 8
        # Opponent deck draws before the target are hidden from observer P1.
        mutated["plies"][0]["move"]["drawn"] = "B10"
        mutated["plies"][2]["move"]["drawn"] = "G9"
        # P1's draw on move four is after the target position.
        mutated["plies"][3]["move"]["drawn"] = "Y2"

        view_a = history_belief.build_view(original, 4, 1)
        view_b = history_belief.build_view(mutated, 4, 1)
        self.assertEqual(
            history_belief.canonical_view_bytes(view_a),
            history_belief.canonical_view_bytes(view_b),
        )
        self.assertEqual(
            history_belief.worker_wire(view_a),
            history_belief.worker_wire(view_b),
        )
        self.assertNotIn("own_draw", view_a["events"][0])
        self.assertEqual(view_a["events"][1]["own_draw"], [4, 6])
        self.assertNotIn("own_draw", view_a["events"][2])
        self.assertEqual(len(view_a["events"]), 3)

    def test_worker_rejects_opponent_hidden_draw_identity(self) -> None:
        view = history_belief.build_view(fixture(), 4, 1)
        view["events"][0]["own_draw"] = [2, 5]
        completed = subprocess.run(
            [str(ROOT / "bin/history_belief"), "-w", "1"],
            input=history_belief.worker_wire(view),
            text=True,
            capture_output=True,
            cwd=ROOT,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("opponent deck-draw identities are forbidden", completed.stderr)

    def test_prefix_validation_rejects_non_argmax_actors(self) -> None:
        with self.assertRaisesRegex(
            history_belief.ViewError, "stochastic policy sampling"
        ):
            history_belief.validate_actor_prefix(
                "policy:data/champion.bin:0.5:20",
                3,
                20,
                ROOT / "data/champion.bin",
            )
        with self.assertRaisesRegex(
            history_belief.ViewError, "planner-enabled policy"
        ):
            history_belief.validate_actor_prefix(
                "policy:data/champion.bin:0:20:16:12",
                3,
                20,
                ROOT / "data/champion.bin",
            )
        with self.assertRaisesRegex(
            history_belief.ViewError, "semantic-search-enabled"
        ):
            history_belief.validate_actor_prefix(
                (
                    "rolloutu:data/champion.bin:512:4:0.02:0:1:20:0:0:0:0:"
                    "3.5:2:2:20:0:0:20:1:0:512:1:0:0:1"
                ),
                3,
                20,
                ROOT / "data/champion.bin",
            )
        with self.assertRaisesRegex(
            history_belief.ViewError, "checkpoint does not match"
        ):
            history_belief.validate_actor_prefix(
                "policy:data/champion.bin:0:20",
                3,
                20,
                ROOT / "data/c8.bin",
            )

    def test_dual_actor_uses_root_checkpoint_and_shifted_rollout_tail(self) -> None:
        source = (
            "rolloutu2:data/champion.bin:data/c8.bin:512:4:0.02:0:1:20:"
            "0:0:0:0:3.5:2:2:20:0:0:20:1:0:512:1"
        )
        self.assertEqual(
            history_belief.validate_actor_prefix(
                source, 3, 20, ROOT / "data/champion.bin"
            ),
            source,
        )
        with self.assertRaisesRegex(
            history_belief.ViewError, "checkpoint does not match"
        ):
            history_belief.validate_actor_prefix(
                source, 3, 20, ROOT / "data/c8.bin"
            )

        searched_early = source.replace(":0:1:20:", ":0:1:2:")
        with self.assertRaisesRegex(
            history_belief.ViewError, "rollout-search action"
        ):
            history_belief.validate_actor_prefix(
                searched_early, 3, 20, ROOT / "data/champion.bin"
            )

    def test_dual_actor_requires_continuation_checkpoint_field(self) -> None:
        with self.assertRaisesRegex(
            history_belief.ViewError, "continuation checkpoint"
        ):
            history_belief.validate_actor_prefix(
                "rollout2:data/champion.bin",
                0,
                20,
                ROOT / "data/champion.bin",
            )

    def test_three_actor_uses_root_checkpoint_and_shifted_rollout_tail(
        self,
    ) -> None:
        source = (
            "rolloutu3:data/champion.bin:data/c8.bin:data/c8.bin:"
            "512:4:0.02:0:1:20:0:0:0:0:3.5:2:2:20:0:0:20:1:0:512:1"
        )
        self.assertEqual(
            history_belief.validate_actor_prefix(
                source, 3, 20, ROOT / "data/champion.bin"
            ),
            source,
        )
        with self.assertRaisesRegex(
            history_belief.ViewError, "checkpoint does not match"
        ):
            history_belief.validate_actor_prefix(
                source, 3, 20, ROOT / "data/c8.bin"
            )

        searched_early = source.replace(":0:1:20:", ":0:1:2:")
        with self.assertRaisesRegex(
            history_belief.ViewError, "rollout-search action"
        ):
            history_belief.validate_actor_prefix(
                searched_early, 3, 20, ROOT / "data/champion.bin"
            )

    def test_three_actor_requires_both_auxiliary_checkpoints(self) -> None:
        with self.assertRaisesRegex(
            history_belief.ViewError, "continuation checkpoint"
        ):
            history_belief.validate_actor_prefix(
                "rollout3:data/champion.bin",
                0,
                20,
                ROOT / "data/champion.bin",
            )
        with self.assertRaisesRegex(
            history_belief.ViewError, "veto checkpoint"
        ):
            history_belief.validate_actor_prefix(
                "rollout3:data/champion.bin:data/c8.bin",
                0,
                20,
                ROOT / "data/champion.bin",
            )

    def test_ply4_actor_aware_golden_and_cardinality(self) -> None:
        result = history_belief.annotate(
            fixture(),
            target_ply=4,
            observer=1,
            worlds=20000,
            seed=history_belief.DEFAULT_SEED,
            symmetries=20,
            net_path=ROOT / "data/champion.bin",
            worker_path=ROOT / "bin/history_belief",
        )
        self.assertEqual(result["accepted"], 886)
        by_name = {row["card"]: row for row in result["cards"]}
        self.assertGreater(
            by_name["Y9"]["expected_count"], by_name["Y4"]["expected_count"]
        )
        self.assertGreater(
            by_name["Y10"]["expected_count"], by_name["Y4"]["expected_count"]
        )
        self.assertAlmostEqual(
            by_name["Y4"]["expected_count"], 189 / 886, places=8
        )
        self.assertAlmostEqual(
            by_name["Y9"]["expected_count"], 203 / 886, places=8
        )
        self.assertAlmostEqual(
            by_name["Y10"]["expected_count"], 199 / 886, places=8
        )
        self.assertEqual(result["opponent_hand_size"], 8)
        self.assertAlmostEqual(result["marginal_sum"], 8.0, places=8)
        self.assertIn("opponent deck draws", result["provenance"]["information_contract"])


if __name__ == "__main__":
    unittest.main()
