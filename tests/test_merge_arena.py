import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.merge_arena import (
    EvidenceError, _result, _sufficient, _write_json,
    combine_reciprocal, merge_block,
)


ROOT = Path(__file__).resolve().parents[1]
ARENA = Path(os.environ.get("LOSTCITIES_ARENA", ROOT / "bin" / "arena"))
MERGER = ROOT / "tools" / "merge_arena.py"


class ArenaEvidenceTest(unittest.TestCase):
    def run_arena(self, output: Path, start: int, count: int, threads: int,
                  agent_a: str = "random", agent_b: str = "heur",
                  seed: int = 77331) -> None:
        subprocess.run([
            str(ARENA), "-a", agent_a, "-b", agent_b, "-n", str(count),
            "-t", str(threads), "-s", str(seed), "-r", "3",
            "--pair-start", str(start), "--raw-pairs", str(output),
            "--raw-only", "--provenance", "test-plan;arena-test;model-none",
        ], cwd=ROOT, check=True)

    def test_arena_rejects_noncanonical_or_unsafe_ranges_without_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            cases = (
                ("plus", ["-n", "+1"]),
                ("space", ["-n", " 1"]),
                ("suffix", ["-t", "1x"]),
                ("negative", ["-s", "-1"]),
                ("seed-overflow", ["-s", str(1 << 64)]),
                ("pair-overflow", ["--pair-start", str((1 << 64) - 1),
                                   "-n", "2"]),
                ("count-overflow", ["-n", str(((1 << 31) - 1) // 2 + 1)]),
            )
            for label, extra in cases:
                with self.subTest(label=label):
                    output = tmp / f"{label}.jsonl"
                    command = [
                        str(ARENA), "-a", "random", "-b", "random",
                        "-n", "1", "-t", "1", "-s", "1", "-r", "1",
                        "--raw-pairs", str(output), "--raw-only",
                        "--provenance", "strict-test",
                    ] + extra
                    completed = subprocess.run(
                        command, cwd=ROOT, text=True, capture_output=True)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertFalse(output.exists())
                    self.assertEqual(list(tmp.glob(output.name + ".tmp.*")), [])

            no_provenance = tmp / "no-provenance.jsonl"
            completed = subprocess.run([
                str(ARENA), "-a", "random", "-b", "random", "-n", "1",
                "-t", "1", "--raw-pairs", str(no_provenance), "--raw-only",
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(no_provenance.exists())

    def test_uint64_final_pair_is_a_complete_mergeable_row(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "last.jsonl"
            self.run_arena(output, (1 << 64) - 1, 1, 1,
                           agent_b="random")
            merged = merge_block([output], (1 << 64) - 1, 1,
                                 allow_caps=True)
            self.assertEqual(merged["pair_start"], str((1 << 64) - 1))
            self.assertEqual(merged["pair_count"], 1)

    def test_raw_jsonl_rejects_truncation_duplicates_constants_and_blanks(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            original = tmp / "original.jsonl"
            self.run_arena(original, 0, 2, 1)
            text = original.read_text()
            variants = {
                "truncated": text.rstrip("\n"),
                "duplicate-key": text.replace(
                    '"schema":1', '"schema":1,"schema":1', 1),
                "nan": text.replace('"schema":1', '"schema":NaN', 1),
                "blank": text.replace("\n", "\n\n", 1),
                "padded": " " + text,
            }
            for label, content in variants.items():
                with self.subTest(label=label):
                    path = tmp / label
                    path.write_text(content)
                    with self.assertRaises(EvidenceError):
                        merge_block([path], 0, 2, allow_caps=True)

    def test_merger_cli_rejects_noncanonical_numeric_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            cases = (
                ("--expect-start", "+0"),
                ("--expect-start", str(1 << 64)),
                ("--expect-pairs", " 1"),
                ("--expect-pairs", str(((1 << 31) - 1) // 2 + 1)),
            )
            for ordinal, (option, value) in enumerate(cases):
                with self.subTest(option=option, value=value):
                    output = tmp / f"rejected-{ordinal}.json"
                    command = [
                        sys.executable, str(MERGER), "block",
                        "--expect-start", "0", "--expect-pairs", "1",
                        "--output", str(output), option, value,
                        str(tmp / "unused.jsonl"),
                    ]
                    completed = subprocess.run(
                        command, cwd=ROOT, text=True, capture_output=True)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertFalse(output.exists())

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

            records = [json.loads(line)
                       for line in first.read_text().splitlines()]
            records[1]["plies"][0] = 2
            too_short = tmp / "too-short"
            too_short.write_text(
                "\n".join(json.dumps(row) for row in records) + "\n")
            with self.assertRaises(EvidenceError):
                merge_block([too_short], 0, 3, allow_caps=True)

    def test_output_refuses_to_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            output = tmp / "result.json"
            _write_json(output, {"first": 1})
            with self.assertRaises(EvidenceError):
                _write_json(output, {"second": 2})
            self.assertEqual(json.loads(output.read_text()), {"first": 1})

            raw = tmp / "result.jsonl"
            raw.write_bytes(b"existing evidence\n")
            completed = subprocess.run([
                str(ARENA), "-a", "random", "-b", "random", "-n", "1",
                "-t", "1", "-s", "9", "-r", "1",
                "--raw-pairs", str(raw), "--raw-only",
                "--provenance", "no-clobber-test",
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(raw.read_bytes(), b"existing evidence\n")
            self.assertEqual(list(tmp.glob(raw.name + ".tmp.*")), [])

    def test_reciprocal_cli_hashes_the_exact_snapshots_it_parses(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            first_raw, second_raw = tmp / "first.jsonl", tmp / "second.jsonl"
            self.run_arena(first_raw, 0, 2, 1, seed=1901)
            self.run_arena(second_raw, 0, 2, 1, agent_a="heur",
                           agent_b="random", seed=1902)
            first = merge_block([first_raw], 0, 2, allow_caps=True)
            second = merge_block([second_raw], 0, 2, allow_caps=True)
            first_path, second_path = tmp / "first.json", tmp / "second.json"
            first_bytes = (json.dumps(first, indent=1) + "\n").encode()
            second_bytes = (json.dumps(second, separators=(",", ":")) +
                            "\n").encode()
            first_path.write_bytes(first_bytes)
            second_path.write_bytes(second_bytes)
            output = tmp / "reciprocal.json"
            subprocess.run([
                sys.executable, str(MERGER), "reciprocal",
                "--first", str(first_path), "--second", str(second_path),
                "--gate-z", "1.695", "--require-positive-margin",
                "--output", str(output),
            ], cwd=ROOT, check=True)
            merged = json.loads(output.read_text())
            self.assertEqual(merged["input_block_snapshots"], [
                {"path": str(first_path),
                 "sha256": hashlib.sha256(first_bytes).hexdigest()},
                {"path": str(second_path),
                 "sha256": hashlib.sha256(second_bytes).hexdigest()},
            ])
            self.assertEqual(merged["raw_input_validation"]["status"],
                             "validated")
            self.assertEqual(merged["promotion_gate_configuration"], {
                "critical_z": 1.695,
                "require_positive_margin": True,
                "require_each_orientation_above_half": True,
                "require_raw_input_validation": True,
            })
            self.assertEqual(
                [item["pair_count"]
                 for item in merged["raw_input_validation"]["blocks"]],
                [2, 2])

            fabricated = json.loads(json.dumps(first))
            n = fabricated["pair_count"]
            sufficient = fabricated["sufficient_statistics"]
            sufficient.update({
                "margin_sum": 0, "margin_sumsq": 0,
                "score_quarters_sum": 2 * n,
                "score_quarters_sumsq": 4 * n,
                "wins": 0, "losses": 0, "draws": 2 * n,
                "points_a_sum": 0, "points_b_sum": 0,
            })
            fabricated["result"] = _result(sufficient)
            fabricated_path = tmp / "fabricated.json"
            fabricated_path.write_text(json.dumps(fabricated) + "\n")
            fabricated_output = tmp / "fabricated-result.json"
            completed = subprocess.run([
                sys.executable, str(MERGER), "reciprocal",
                "--first", str(fabricated_path),
                "--second", str(second_path),
                "--output", str(fabricated_output),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("does not exactly match", completed.stderr)
            self.assertFalse(fabricated_output.exists())

            first_raw.write_bytes(first_raw.read_bytes() + b"\n")
            changed_raw_output = tmp / "changed-raw-result.json"
            completed = subprocess.run([
                sys.executable, str(MERGER), "reciprocal",
                "--first", str(first_path), "--second", str(second_path),
                "--output", str(changed_raw_output),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("recorded raw input validation failed",
                          completed.stderr)
            self.assertFalse(changed_raw_output.exists())

            duplicate = tmp / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":1,"schema_version":1}\n')
            rejected = tmp / "rejected.json"
            completed = subprocess.run([
                sys.executable, str(MERGER), "reciprocal",
                "--first", str(duplicate), "--second", str(second_path),
                "--output", str(rejected),
            ], cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate JSON key", completed.stderr)
            self.assertFalse(rejected.exists())

    def test_reciprocal_inverts_second_orientation(self):
        def block(a, b, seed, rows, digest):
            sufficient = _sufficient(rows)
            return {
                "schema_version": 1,
                "artifact_kind": "merged_arena_pair_evidence",
                "metadata": {"schema": 1, "seed": str(seed),
                             "agent_a": a, "agent_b": b, "rounds": 3,
                             "provenance": "locked"},
                "pair_start": "0", "pair_count": len(rows),
                "inputs": [{"path": "synthetic", "sha256": digest * 64,
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
        ], "1")
        second = block("baseline", "candidate", 102, [
            row([0, 0], [3, 2]), row([4, 0], [0, 0]),
        ], "2")
        reciprocal = combine_reciprocal(first, second)
        result = reciprocal["candidate_result"]
        self.assertAlmostEqual(result["margin_per_game"], 1.375)
        self.assertAlmostEqual(
            result["margin_pair_clustered_se"],
            math.hypot(first["result"]["margin_pair_clustered_se"],
                       second["result"]["margin_pair_clustered_se"]) / 2)
        self.assertAlmostEqual(result["match_score"], .625)
        self.assertEqual((result["wins"], result["losses"], result["draws"]),
                         (4, 2, 2))
        self.assertEqual(len(reciprocal["input_block_snapshots"]), 2)
        self.assertEqual(reciprocal["raw_input_validation"]["status"],
                         "not_performed")
        self.assertEqual(
            reciprocal["promotion_gate_configuration"],
            {"critical_z": 1.645, "require_positive_margin": False,
             "require_each_orientation_above_half": True,
             "require_raw_input_validation": True})

        overlapping = json.loads(json.dumps(second))
        overlapping["metadata"]["seed"] = first["metadata"]["seed"]
        with self.assertRaises(EvidenceError):
            combine_reciprocal(first, overlapping)

        adjacent = json.loads(json.dumps(second))
        adjacent["metadata"]["seed"] = first["metadata"]["seed"]
        adjacent["pair_start"] = "2"
        adjacent["inputs"][0]["pair_start"] = 2
        combine_reciprocal(first, adjacent)

        reused = json.loads(json.dumps(second))
        reused["inputs"][0]["sha256"] = first["inputs"][0]["sha256"]
        with self.assertRaises(EvidenceError):
            combine_reciprocal(first, reused)

        tampered = json.loads(json.dumps(first))
        tampered["result"]["match_score"] = 1.0
        with self.assertRaises(EvidenceError):
            combine_reciprocal(tampered, second)

        wrong_type = json.loads(json.dumps(first))
        wrong_type["result"]["pairs"] = True
        with self.assertRaises(EvidenceError):
            combine_reciprocal(wrong_type, second)

        inconsistent = json.loads(json.dumps(first))
        inconsistent["sufficient_statistics"]["score_quarters_sum"] += 1
        inconsistent["result"] = _result(
            inconsistent["sufficient_statistics"])
        with self.assertRaises(EvidenceError):
            combine_reciprocal(inconsistent, second)

        inconsistent = json.loads(json.dumps(first))
        inconsistent["sufficient_statistics"]["points_a_sum"] += 1
        inconsistent["result"] = _result(
            inconsistent["sufficient_statistics"])
        with self.assertRaises(EvidenceError):
            combine_reciprocal(inconsistent, second)

        inconsistent = json.loads(json.dumps(first))
        inconsistent["sufficient_statistics"]["plies_sum"] = 4
        inconsistent["result"] = _result(
            inconsistent["sufficient_statistics"])
        with self.assertRaises(EvidenceError):
            combine_reciprocal(inconsistent, second)

        negative_first = block("candidate", "baseline", 201, [
            row([1, 1], [0, 0]), row([1, -100], [0, 0]),
        ], "3")
        negative_second = block("baseline", "candidate", 202, [
            row([0, 0], [1, 1]), row([0, 0], [1, -100]),
        ], "4")
        permissive = combine_reciprocal(
            negative_first, negative_second, gate_z=0.1)
        directional = combine_reciprocal(
            negative_first, negative_second, gate_z=0.1,
            require_positive_margin=True)
        self.assertGreater(permissive["candidate_result"]["match_score"], .5)
        self.assertLess(permissive["candidate_result"]["margin_per_game"], 0)
        self.assertTrue(permissive["statistical_gate_passed"])
        self.assertFalse(permissive["promotion_gate_passed"])
        self.assertFalse(directional["statistical_gate_passed"])
        self.assertFalse(directional["promotion_gate_passed"])
        self.assertEqual(
            directional["promotion_gate_configuration"]["critical_z"], .1)
        self.assertTrue(directional["promotion_gate_configuration"]
                        ["require_positive_margin"])

        with self.assertRaises(EvidenceError):
            combine_reciprocal(first, second, gate_z=float("nan"))
        with self.assertRaises(EvidenceError):
            combine_reciprocal(first, second,
                               require_positive_margin=1)


if __name__ == "__main__":
    unittest.main()
