"""End-to-end contracts for the deterministic exact-K held-out evaluator."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "bin" / "belief_eval"


class BeliefEvalToolTest(unittest.TestCase):
    def run_eval(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(TOOL),
                "--net",
                "data/champion.bin",
                "--games",
                "1",
                "--rounds",
                "1",
                "--seed",
                "9918273",
                "--symmetries",
                "1",
                "--alpha",
                "0",
                "--min-ply",
                "1",
                "--max-ply",
                "1",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_uniform_prior_and_repeatability(self) -> None:
        first = self.run_eval()
        second = self.run_eval()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)

        report = json.loads(first.stdout)
        self.assertEqual(report["schema"], "lc-belief-eval-v1")
        self.assertEqual(report["actor"]["type"], "exact_policy_argmax")
        self.assertFalse(report["actor"]["uses_belief"])
        self.assertTrue(report["actor"]["truth_scrubbed"])
        self.assertEqual(report["sample"]["states"], 1)
        self.assertGreater(report["sample"]["uncertain_cards"], 0)

        learned = report["exact_k"]
        prior = report["uniform_card_count_prior"]
        for key in (
            "nll_per_state",
            "nll_per_uncertain_card",
            "brier",
            "auc_within_state",
            "auc_pair_weighted",
            "top_k_recall",
        ):
            self.assertAlmostEqual(learned[key], prior[key], places=10)

    def test_rejects_malformed_numeric_options(self) -> None:
        cases = (
            ("--symmetries", "7"),
            ("--games", ""),
            ("--seed", "-1"),
            ("--seed", "12x"),
            ("--alpha", ""),
            ("--alpha", " nan"),
            ("--rounds", "4"),
        )
        for option, value in cases:
            with self.subTest(option=option, value=value):
                result = subprocess.run(
                    [str(TOOL), option, value],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("usage:", result.stderr)

    def test_candidate_policy_cannot_change_frozen_scoring_games(self) -> None:
        learned_command = [
            str(TOOL), "--net", "data/champion.bin", "--games", "1",
            "--rounds", "1", "--seed", "9918273", "--symmetries", "20",
            "--alpha", "1.15", "--min-ply", "1", "--max-ply", "12",
            "--json",
        ]
        baseline = subprocess.run(
            learned_command, cwd=ROOT, text=True, capture_output=True,
            check=False,
        )
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        baseline_report = json.loads(baseline.stdout)

        source = (ROOT / "data/champion.bin").read_bytes()
        altered = bytearray(source)
        _, feat_dim, h1, h2, nplay, _ = struct.unpack("=6I", source[:24])
        # Force a large, deterministic change in every policy bias while
        # leaving the evaluated belief head untouched. The default actor must
        # still be the separately loaded maintained champion, so the scored
        # multi-ply trajectory and learned-belief metrics remain identical.
        bplay_float = (
            feat_dim * h1 + h1 + h1 * h2 + h2 + h2 + 1
            + nplay * h2
        )
        bplay_offset = 24 + bplay_float * 4
        for i in range(nplay):
            struct.pack_into("=f", altered, bplay_offset + i * 4,
                             1000.0 * i)

        with tempfile.TemporaryDirectory(prefix="lc-belief-actor-") as tmp:
            candidate = Path(tmp) / "policy-altered.bin"
            candidate.write_bytes(altered)
            command = learned_command.copy()
            command[command.index("data/champion.bin")] = str(candidate)
            result = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )
            changed_actor_command = command.copy()
            changed_actor_command[3:3] = ["--actor-net", str(candidate)]
            changed_actor = subprocess.run(
                changed_actor_command, cwd=ROOT, text=True,
                capture_output=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["actor"]["model"], "data/champion.bin")
        self.assertEqual(
            report["actor"]["model_fingerprint"],
            baseline_report["actor"]["model_fingerprint"],
        )
        self.assertNotEqual(
            report["model_fingerprint"], baseline_report["model_fingerprint"]
        )
        self.assertEqual(report["sample"], baseline_report["sample"])
        self.assertEqual(report["exact_k"], baseline_report["exact_k"])
        self.assertEqual(changed_actor.returncode, 0, changed_actor.stderr)
        changed_actor_report = json.loads(changed_actor.stdout)
        self.assertNotEqual(
            changed_actor_report["exact_k"], baseline_report["exact_k"],
            "the deliberately altered policy must actually change the "
            "multi-ply scoring trajectory when explicitly selected as actor",
        )


if __name__ == "__main__":
    unittest.main()
