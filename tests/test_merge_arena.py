import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.merge_arena import (
    EvidenceError, _result, _sufficient, _write_json,
    combine_reciprocal, merge_block,
)


ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "bin" / "arena"


class ArenaEvidenceTest(unittest.TestCase):
    def run_arena(self, output: Path, start: int, count: int, threads: int) -> None:
        subprocess.run([
            str(ARENA), "-a", "random", "-b", "heur", "-n", str(count),
            "-t", str(threads), "-s", "77331", "-r", "3",
            "--pair-start", str(start), "--raw-pairs", str(output),
            "--raw-only", "--provenance", "test-plan;arena-test;model-none",
        ], cwd=ROOT, check=True)

    def test_shards_equal_monolithic_rows_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            full = tmp / "full.jsonl"
            self.run_arena(full, 0, 19, 3)
            shards = []
            for ordinal, (start, count, threads) in enumerate(
                    ((0, 5, 1), (5, 7, 4), (12, 7, 2))):
                path = tmp / f"shard-{ordinal}.jsonl"
                self.run_arena(path, start, count, threads)
                shards.append(path)
            monolithic = merge_block([full], 0, 19)
            merged = merge_block(shards, 0, 19)
            self.assertEqual(monolithic["canonical_pair_rows_sha256"],
                             merged["canonical_pair_rows_sha256"])
            self.assertEqual(monolithic["sufficient_statistics"],
                             merged["sufficient_statistics"])
            self.assertEqual(monolithic["result"], merged["result"])

    def test_gap_duplicate_metadata_and_cap_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            first, second = tmp / "first", tmp / "second"
            self.run_arena(first, 0, 3, 1)
            self.run_arena(second, 4, 2, 1)
            with self.assertRaises(EvidenceError):
                merge_block([first, second], 0, 5)
            with self.assertRaises(EvidenceError):
                merge_block([first, first], 0, 6)

            records = [json.loads(line) for line in first.read_text().splitlines()]
            records[1]["capped_rounds"][0] = 1
            capped = tmp / "capped"
            capped.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            with self.assertRaises(EvidenceError):
                merge_block([capped], 0, 3)
            self.assertEqual(merge_block([capped], 0, 3, allow_caps=True)
                             ["result"]["capped_rounds"], 1)

            records = [json.loads(line) for line in second.read_text().splitlines()]
            records[0]["seed"] = "999"
            wrong = tmp / "wrong"
            wrong.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            with self.assertRaises(EvidenceError):
                merge_block([first, wrong], 0, 5, allow_caps=True)

            records[1]["index"] = 4.0
            malformed = tmp / "malformed"
            malformed.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n")
            with self.assertRaises(EvidenceError):
                merge_block([malformed], 4, 2, allow_caps=True)

    def test_output_refuses_to_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            _write_json(output, {"first": 1})
            with self.assertRaises(EvidenceError):
                _write_json(output, {"second": 2})
            self.assertEqual(json.loads(output.read_text()), {"first": 1})

    def test_reciprocal_inverts_second_orientation(self):
        def block(a, b, seed, rows):
            sufficient = _sufficient(rows)
            return {
                "schema_version": 1,
                "artifact_kind": "merged_arena_pair_evidence",
                "metadata": {"schema": 1, "seed": str(seed),
                             "agent_a": a, "agent_b": b, "rounds": 3,
                             "provenance": "locked"},
                "pair_start": "0", "pair_count": len(rows),
                "inputs": [{"path": "synthetic", "sha256": "1" * 64,
                            "pair_start": 0, "pair_count": len(rows)}],
                "canonical_pair_rows_sha256": "0" * 64,
                "sufficient_statistics": sufficient,
                "result": _result(sufficient),
            }
        def row(a, b):
            return {"score_a": a, "score_b": b,
                    "plies": [100, 100], "capped_rounds": [0, 0]}
        first = block("candidate", "baseline", 101, [
            row([10, 5], [0, 0]), row([0, 1], [5, 1]),
        ])
        second = block("baseline", "candidate", 102, [
            row([0, 0], [3, 2]), row([4, 0], [0, 0]),
        ])
        result = combine_reciprocal(first, second)["candidate_result"]
        self.assertAlmostEqual(result["margin_per_game"], 1.375)
        self.assertAlmostEqual(
            result["margin_pair_clustered_se"],
            math.hypot(first["result"]["margin_pair_clustered_se"],
                       second["result"]["margin_pair_clustered_se"]) / 2)
        self.assertAlmostEqual(result["match_score"], .625)
        self.assertEqual((result["wins"], result["losses"], result["draws"]),
                         (4, 2, 2))

        overlapping = json.loads(json.dumps(second))
        overlapping["metadata"]["seed"] = first["metadata"]["seed"]
        with self.assertRaises(EvidenceError):
            combine_reciprocal(first, overlapping)

        tampered = json.loads(json.dumps(first))
        tampered["result"]["match_score"] = 1.0
        with self.assertRaises(EvidenceError):
            combine_reciprocal(tampered, second)


if __name__ == "__main__":
    unittest.main()
