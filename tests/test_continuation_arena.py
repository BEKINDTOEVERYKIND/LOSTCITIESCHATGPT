"""Contracts for the fast continuation-role candidate screen."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "bin" / "continuation_arena"
CHAMPION = ROOT / "data" / "champion.bin"


class ContinuationArenaTest(unittest.TestCase):
    def run_screen(self, threads: int, raw: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(ARENA),
                "-a", str(CHAMPION),
                "-r", str(CHAMPION),
                "-b", str(CHAMPION),
                "-n", "10",
                "-t", str(threads),
                "-s", "202608229117",
                "--target-round", "0",
                "--raw-pairs", str(raw),
                "--provenance", "continuation-arena-unittest",
                "-q",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
            timeout=180,
        )

    def test_identical_checkpoint_and_thread_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            one = Path(td) / "one.jsonl"
            two = Path(td) / "two.jsonl"
            r1 = self.run_screen(1, one)
            r2 = self.run_screen(4, two)
            self.assertEqual(r1.stdout, r2.stdout)
            self.assertEqual(one.read_bytes(), two.read_bytes())

            fields = r1.stdout.split()
            self.assertEqual(len(fields), 8)
            self.assertEqual(float(fields[0]), 0.0)
            self.assertEqual(float(fields[2]), 0.5)
            self.assertEqual(int(fields[7]), 0)

            records = [json.loads(line) for line in one.read_text().splitlines()]
            self.assertEqual(
                records[0]["evidence_scope"],
                "candidate_screen_only_not_promotion",
            )
            self.assertEqual(records[-1], {"record": "complete", "pairs": 10})
            pairs = records[1:-1]
            self.assertEqual(len(pairs), 10)
            self.assertEqual(
                {mapping for row in pairs for mapping in row["player_mapping"]},
                set(range(20)),
            )
            for row in pairs:
                self.assertEqual(row["candidate_seat"][0] ^ 1,
                                 row["candidate_seat"][1])
                self.assertEqual(row["candidate_margin"][0],
                                 -row["candidate_margin"][1])
                self.assertEqual(row["capped"], [0, 0])
                self.assertEqual(len(row["cycle_forces"]), 2)


if __name__ == "__main__":
    unittest.main()
