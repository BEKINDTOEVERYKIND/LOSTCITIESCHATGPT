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
            self.assertEqual(records[0]["schema"], 2)
            self.assertEqual(records[0]["continuation_objective"], 0)
            self.assertEqual(records[0]["role_mapping_mode"],
                             "legacy-seat-balanced")
            self.assertEqual(records[-1], {"record": "complete", "pairs": 10})
            pairs = [row for row in records if row["record"] == "pair"]
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
                self.assertEqual(row["candidate_round_margin"],
                                 row["candidate_margin"])
                self.assertEqual(row["candidate_objective_target"],
                                 row["candidate_margin"])
                self.assertEqual(row["cumulative_before"], [0, 0])
                self.assertEqual(row["cum_before"],
                                 row["cumulative_before"])
                self.assertIsNone(row["candidate_final_match_margin"])
                self.assertIsNone(row["candidate_final_match_result"])
                self.assertIsNone(row["candidate_hybrid_target"])
                self.assertEqual(row["capped"], [0, 0])
                self.assertEqual(len(row["cycle_forces"]), 2)

            summary = next(
                row for row in records if row["record"] == "summary"
            )
            self.assertEqual(summary["continuation_objective"], 0)
            self.assertTrue(
                summary["configured_objective_aggregate_comparable"]
            )
            self.assertEqual(summary["configured_objective_per_leg"], 0.0)
            self.assertEqual(summary["rounds"][0]["pairs"], 10)
            self.assertEqual(summary["rounds"][1]["pairs"], 0)
            self.assertEqual(summary["rounds"][2]["pairs"], 0)

    def run_hybrid_screen(
        self, threads: int, raw: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(ARENA),
                "-a", str(CHAMPION),
                "-r", str(CHAMPION),
                "-b", str(CHAMPION),
                "-n", "6",
                "-t", str(threads),
                "-s", "202608259117",
                "--target-round", "2",
                "--continuation-objective", "2",
                "--continuation-role-mappings", "independent",
                "--raw-pairs", str(raw),
                "--provenance", "continuation-arena-hybrid-unittest",
                "-q",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
            timeout=180,
        )

    def test_hybrid_targets_role_schedule_and_thread_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            one = Path(td) / "hybrid-one.jsonl"
            many = Path(td) / "hybrid-many.jsonl"
            r1 = self.run_hybrid_screen(1, one)
            r2 = self.run_hybrid_screen(3, many)
            # Legacy quiet output remains an eight-field round diagnostic;
            # v2 selection metrics are the self-describing raw summary.
            self.assertEqual(r1.stdout, r2.stdout)
            self.assertEqual(len(r1.stdout.split()), 8)
            self.assertEqual(one.read_bytes(), many.read_bytes())

            records = [json.loads(line) for line in one.read_text().splitlines()]
            meta = records[0]
            self.assertEqual(meta["schema"], 2)
            self.assertEqual(meta["continuation_objective"], 2)
            self.assertEqual(meta["role_mapping_mode"], "independent")
            pairs = [row for row in records if row["record"] == "pair"]
            self.assertEqual(len(pairs), 6)
            root_role_mappings = set()
            for row in pairs:
                self.assertEqual(row["round"], 2)
                self.assertEqual(row["cum_before"],
                                 row["cumulative_before"])
                self.assertEqual(row["candidate_seat"][0] ^ 1,
                                 row["candidate_seat"][1])
                root_role_mappings.add(
                    row["player_mapping"][row["root_player"]]
                )
                self.assertEqual(
                    row["root_role_mapping"],
                    row["player_mapping"][row["root_player"]],
                )
                self.assertEqual(
                    row["opponent_role_mapping"],
                    row["player_mapping"][row["root_player"] ^ 1],
                )
                self.assertEqual(row["candidate_round_margin"][0],
                                 -row["candidate_round_margin"][1])
                self.assertEqual(row["candidate_final_match_margin"][0],
                                 -row["candidate_final_match_margin"][1])
                self.assertEqual(row["candidate_final_match_result"][0],
                                 -row["candidate_final_match_result"][1])
                for leg in range(2):
                    final_margin = row["candidate_final_match_margin"][leg]
                    result = (final_margin > 0) - (final_margin < 0)
                    expected = 0.05 * final_margin + 50.0 * result
                    self.assertAlmostEqual(
                        row["candidate_hybrid_target"][leg], expected
                    )
                    self.assertAlmostEqual(
                        row["candidate_objective_target"][leg], expected
                    )
            self.assertEqual(len(root_role_mappings), 6)

            summary = next(
                row for row in records if row["record"] == "summary"
            )
            final = summary["rounds"][2]
            self.assertEqual(final["pairs"], 6)
            self.assertEqual(final["selection_semantics"],
                             "final_match_hybrid")
            self.assertEqual(final["match_score"], 0.5)
            self.assertEqual(summary["configured_objective_per_leg"], 0.0)

    def test_new_cli_modes_fail_closed(self) -> None:
        cases = [
            (["--continuation-objective"],
             "--continuation-objective must be exactly"),
            (["--continuation-objective", "1"],
             "--continuation-objective must be exactly"),
            (["--continuation-role-mappings"],
             "--continuation-role-mappings must be exactly"),
            (["--continuation-role-mappings", "random"],
             "--continuation-role-mappings must be exactly"),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                run = subprocess.run(
                    [str(ARENA), "-a", str(CHAMPION), *args],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(expected, run.stderr)

    def test_mode2_cycle_keeps_early_round_margin_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "cycle.jsonl"
            subprocess.run(
                [
                    str(ARENA),
                    "-a", str(CHAMPION),
                    "-r", str(CHAMPION),
                    "-b", str(CHAMPION),
                    "-n", "3",
                    "-t", "2",
                    "-s", "202608259119",
                    "--continuation-objective", "2",
                    "--continuation-role-mappings", "independent",
                    "--raw-pairs", str(raw),
                    "--provenance", "continuation-arena-cycle-unittest",
                    "--raw-only",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
                timeout=180,
            )
            records = [json.loads(line) for line in raw.read_text().splitlines()]
            pairs = [row for row in records if row["record"] == "pair"]
            self.assertEqual([row["round"] for row in pairs], [0, 1, 2])
            for row in pairs[:2]:
                self.assertEqual(row["candidate_objective_target"],
                                 row["candidate_round_margin"])
                self.assertIsNone(row["candidate_final_match_margin"])
                self.assertIsNone(row["candidate_hybrid_target"])
            final = pairs[2]
            self.assertEqual(final["candidate_objective_target"],
                             final["candidate_hybrid_target"])
            summary = next(
                row for row in records if row["record"] == "summary"
            )
            self.assertEqual([row["pairs"] for row in summary["rounds"]],
                             [1, 1, 1])
            self.assertFalse(
                summary["configured_objective_aggregate_comparable"]
            )
            self.assertIsNone(summary["configured_objective_per_leg"])
            self.assertIsNone(
                summary["configured_objective_pair_clustered_se"]
            )
            self.assertEqual(
                [row["selection_semantics"] for row in summary["rounds"]],
                ["round_margin", "round_margin", "final_match_hybrid"],
            )


if __name__ == "__main__":
    unittest.main()
