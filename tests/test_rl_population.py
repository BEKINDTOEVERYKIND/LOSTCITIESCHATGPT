#!/usr/bin/env python3
"""End-to-end contracts for conservative opponent-population training."""

from __future__ import annotations

import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PopulationTrainingTest(unittest.TestCase):
    def test_population_training_rejects_unbalanced_game_count(self) -> None:
        source = ROOT / "data" / "champion.bin"
        run = subprocess.run(
            [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--gen-opponent", f"policy:{source}:0:20",
                "--opponent-mix", "0.5",
                "--games", "3",
                "--eval", "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("requires an even --games count", run.stderr)

    def test_balanced_frozen_opponent_and_v6_only_bytes(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-test-") as tmp:
            candidate = Path(tmp) / "candidate.bin"
            command = [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--gen-opponent", f"policy:{source}:0:20",
                "--opponent-mix", "1",
                "--v6-only",
                "--iters", "1",
                "--games", "6",
                "--rounds", "1",
                "--threads", "2",
                "--epochs", "1",
                "--batch", "256",
                "--lr", "0.0001",
                "--vcoef", "0",
                "--bw", "0",
                "--ent", "0",
                "--wd", "0",
                "--eval", "0",
                "--seed", "20260802",
                "--out", str(candidate),
            ]
            run = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=True
            )
            match = re.search(
                r"frozen-opponent games (\d+), learner seats (\d+)/(\d+)",
                run.stdout,
            )
            self.assertIsNotNone(match, run.stdout)
            self.assertEqual(tuple(map(int, match.groups())), (6, 3, 3))

            before = source.read_bytes()
            after = candidate.read_bytes()
            self.assertEqual(len(before), len(after))
            magic, feat_dim, h1, h2, nplay, version = struct.unpack(
                "=6I", before[:24]
            )
            self.assertEqual((magic, version), (0x4C435651, 6))
            self.assertEqual((h1, h2, nplay), (512, 256, 120))

            float_bytes = 4
            legacy_dim = 556
            cards = 60
            draws = 6
            header = 24
            legacy_w1_end = header + legacy_dim * h1 * float_bytes
            b1_start = header + feat_dim * h1 * float_bytes
            middle_floats = (
                h1 + h1 * h2 + h2 + h2 + 1
                + nplay * h2 + nplay
                + draws * h2 + draws
                + cards * h2 + cards
            )
            wcomb_start = b1_start + middle_floats * float_bytes

            # v6-only may change appended pile rows and the complete-move
            # residual, but every inherited parameter must remain exact.
            self.assertEqual(before[:legacy_w1_end], after[:legacy_w1_end])
            self.assertEqual(before[b1_start:wcomb_start], after[b1_start:wcomb_start])
            self.assertNotEqual(before[legacy_w1_end:b1_start],
                                after[legacy_w1_end:b1_start])
            self.assertNotEqual(before[wcomb_start:], after[wcomb_start:])


if __name__ == "__main__":
    unittest.main()
