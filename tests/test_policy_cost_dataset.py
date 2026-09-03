from __future__ import annotations

import importlib.util
import hashlib
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "policy_cost_allocate_v2.py"
SPEC = importlib.util.spec_from_file_location("policy_cost_allocate", MODULE_PATH)
assert SPEC and SPEC.loader
allocate_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(allocate_module)


def train_rows(*, reuse_sources: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    serial = 0
    for cell in allocate_module.train_cells():
        for offset in range(allocate_module.TRAIN_QUOTA):
            rows.append({
                "cell": cell,
                "priority_sha256": f"{offset + 1:064x}",
                "source_match_index": str(offset if reuse_sources else serial),
            })
            serial += 1
    return rows


def vector_rows() -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    census: dict[str, int] = {}
    serial = 0
    for rd, ply, frontier in allocate_module.vector_bases():
        for slot in allocate_module.SLOTS:
            cell = f"r{rd}:p{ply:02d}:f{frontier}:j{slot}"
            census[cell] = allocate_module.VECTOR_QUOTA
            for priority in range(allocate_module.VECTOR_QUOTA):
                rows.append({
                    "cell": cell,
                    "priority_sha256": f"{priority + 1:064x}",
                    "source_match_index": str(serial),
                    "source_state_index": str(serial),
                })
                serial += 1
    return rows, census


class PolicyCostAllocationTests(unittest.TestCase):
    def _locked_discovery(
        self, split: str = "TRAIN"
    ) -> tuple[dict, dict, dict, dict, dict]:
        eligible, retained = ((10, 9) if split == "TRAIN" else (6, 5))
        reservoir_header = {
            "split": split, "seed": allocate_module.SEEDS[split],
            "net": "a" * 64, "exclusion": "b" * 64,
        }
        reservoir_footer = {
            "eligible": eligible, "retained": retained, "rejected": 1,
            "chain": "c" * 64, "pooled": 0,
        }
        header = allocate_module.campaign_discovery_header(
            split, reservoir_header
        )
        census = {
            "schema": "lc-policy-cost-discovery-v2", "record_type": "census",
            "state_commitment_chain_sha256": "c" * 64,
            "accepted_by_round": [2, 2, 2],
            "pooled_ge64_by_round": [0, 0, 0],
            "exact_terminal_preempted_by_round": [0, 0, 0],
            "mask_width_counts": [[6, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
            "union_width_counts": [6, 0, 0, 0, 0, 0],
            "eligible_master_width_counts": (
                [0, 0, 0, 0, 0] if split == "TRAIN" else [6, 0, 0, 0, 0]
            ),
            "allocation_cells": [],
        }
        footer = {
            "schema": "lc-policy-cost-discovery-v2", "record_type": "footer",
            "requested_matches": allocate_module.MATCHES[split],
            "completed_matches": allocate_module.MATCHES[split],
            "attempted_states": 7, "accepted_states": 6,
            "probe_orbit_rejections": 1, "cap_hits": 0,
            "eligible_units": eligible, "retained_units": retained,
            "units_rejected_by_bound": 1,
        }
        return header, census, footer, reservoir_header, reservoir_footer

    def test_locked_discovery_requires_complete_cap_free_exact_census(self) -> None:
        values = self._locked_discovery()
        allocate_module.validate_campaign_discovery(*values, "TRAIN")
        mutations = (
            (2, "completed_matches", allocate_module.MATCHES["TRAIN"] - 1,
             "incomplete"),
            (2, "cap_hits", 1, "cap hit"),
            (2, "attempted_states", 8, "algebra"),
            (2, "retained_units", 8, "footer count"),
            (0, "generator", "different_generator", "header mismatch"),
            (0, "burned_source_deal_seeds",
             "1..200, maintained-800 seed 1, 202612010101",
             "header mismatch"),
            (0, "burned_source_deal_seeds",
             "1..200, maintained-800 seed 1, and 202612290001..202612290100",
             "header mismatch"),
            (1, "accepted_by_round", [1, 2, 2], "accepted-state"),
        )
        for index, key, replacement, message in mutations:
            with self.subTest(key=key):
                altered = list(copy.deepcopy(values))
                altered[index][key] = replacement
                with self.assertRaisesRegex(
                    allocate_module.AllocationError, message
                ):
                    allocate_module.validate_campaign_discovery(
                        *altered, "TRAIN"
                    )

    def test_vector_discovery_partitions_every_accepted_state(self) -> None:
        values = self._locked_discovery("SELECT")
        allocate_module.validate_campaign_discovery(*values, "SELECT")
        altered = list(copy.deepcopy(values))
        altered[1]["exact_terminal_preempted_by_round"] = [1, 0, 0]
        with self.assertRaisesRegex(
                allocate_module.AllocationError, "partition census"):
            allocate_module.validate_campaign_discovery(*altered, "SELECT")

    def test_reservoir_rejects_source_outside_frozen_match_census(self) -> None:
        state_bytes = bytearray(174)
        state_bytes[0] = 1
        columns = "\t".join(("columns", *allocate_module.TRAIN_COLUMNS))
        def payload(
            source: int, state: int, encoded: bytes = bytes(state_bytes)
        ) -> bytes:
            state_hash = hashlib.sha256(encoded).hexdigest()
            row = {
                "cell": "r0.p0.g0.t0", "source_match_index": str(source),
                "source_state_index": str(state),
                "source_match_id": f"TRAIN-{source:012d}", "round": "0",
                "ply_bin": "0", "ratio_bin": "0", "pair_type": "0",
                "pair_move_a": "1", "pair_move_b": "2",
                "orbit_sha256": "1" * 64, "state_sha256": state_hash,
                "mask_001_sha256": "2" * 64, "mask_002_sha256": "3" * 64,
                "master_sha256": "4" * 64,
                "state_hex": encoded.hex(),
            }
            row["priority_sha256"] = allocate_module.train_priority(
                allocate_module.SEEDS["TRAIN"], row
            )
            body = "\t".join(row[field] for field in allocate_module.TRAIN_COLUMNS)
            return (
                f"{allocate_module.TRAIN_SCHEMA}\nsplit\tTRAIN\npurpose\tcampaign\n"
                f"seed\t{allocate_module.SEEDS['TRAIN']}\nnet_sha256\t{'a' * 64}\n"
                f"exclusion_sha256\t{'b' * 64}\nreservoir_per_subcell\t1024\n"
                f"{columns}\n{body}\nfooter\teligible_units\t1\tretained_units\t1\t"
                f"rejected_by_bound\t0\tstate_commitment_chain_sha256\t{'c' * 64}\t"
                "pooled_ge64_observed\t0\n"
            ).encode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reservoir.tsv"
            valid = payload(65535, 899)
            path.write_bytes(valid)
            _, rows, _ = allocate_module.reservoir(
                path, hashlib.sha256(valid).hexdigest()
            )
            self.assertEqual(rows[0]["source_state_index"], "899")
            outside_source = payload(65536, 899)
            path.write_bytes(outside_source)
            with self.assertRaisesRegex(
                    allocate_module.AllocationError, "outside frozen discovery"):
                allocate_module.reservoir(
                    path, hashlib.sha256(outside_source).hexdigest()
                )
            outside_state = payload(65535, 900)
            path.write_bytes(outside_state)
            with self.assertRaisesRegex(
                    allocate_module.AllocationError, "outside native match bound"):
                allocate_module.reservoir(
                    path, hashlib.sha256(outside_state).hexdigest()
                )
            short_state = payload(65535, 899, b"\x01")
            path.write_bytes(short_state)
            with self.assertRaisesRegex(
                    allocate_module.AllocationError, "exact native"):
                allocate_module.reservoir(
                    path, hashlib.sha256(short_state).hexdigest()
                )

    def test_train_cartesian_allocation_is_globally_source_unique(self) -> None:
        selected = allocate_module.allocate_train(train_rows())
        self.assertEqual(len(selected), 3 * 24 * 6 * 2 * 16)
        sources = [row["source_match_index"] for row in selected]
        self.assertEqual(len(sources), len(set(sources)))
        for start in range(0, len(selected), 64):
            cells = [tuple(int(part[1:]) for part in row["cell"].split("."))
                     for row in selected[start:start + 64]]
            self.assertEqual([len({cell[index] for cell in cells})
                              for index in range(4)], [3, 24, 6, 2])
        self.assertEqual(
            hashlib.sha256(allocate_module.TRAIN_RULE).hexdigest(),
            "cabca0624915c51a7d6a289baad6cbc054f9da073243dbb69031ba4b840899d1",
        )

    def test_cross_cell_source_reuse_fails_without_topup(self) -> None:
        with self.assertRaisesRegex(allocate_module.AllocationError, "sparse.*cell"):
            allocate_module.allocate_train(train_rows(reuse_sources=True))

    def test_holdout_quota_uses_lower_slot_for_remainder(self) -> None:
        rows, census = vector_rows()
        selected, quotas = allocate_module.allocate_vectors(rows, census)
        self.assertEqual(len(selected), 3 * 24 * 2 * 64)
        for rd, ply, frontier in allocate_module.vector_bases():
            found = [quotas[f"r{rd}:p{ply:02d}:f{frontier}:j{slot}"]
                     for slot in allocate_module.SLOTS]
            self.assertEqual(found, [13, 13, 13, 13, 12])
        for start in range(0, len(selected), 48):
            cells = [row["cell"].split(":")
                     for row in selected[start:start + 48]]
            rounds = {int(cell[0][1:]) for cell in cells}
            plies = {int(cell[1][1:]) for cell in cells}
            frontiers = {int(cell[2][1:]) for cell in cells}
            self.assertEqual(rounds, {0, 1, 2})
            self.assertEqual(len(plies), 8)
            self.assertEqual({ply // 8 for ply in plies}, {0, 1, 2})
            self.assertEqual(frontiers, {0, 1})
        self.assertEqual(
            hashlib.sha256(allocate_module.VECTOR_RULE).hexdigest(),
            "dbb7f1645883196c8453d98684c388c3e62cd76ceba77abc5c36887d5170be6c",
        )

    def test_allocator_source_has_no_efficacy_or_adaptive_paths(self) -> None:
        text = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("optional_stop", text)
        self.assertNotIn("--state", text)
        self.assertIn("TRAIN_QUOTA=16; VECTOR_QUOTA=64", text)
        self.assertIn("vector_slot", text)
        self.assertIn("source_minimum_per_positive_slot", text)


class PolicyCostDatasetCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temp.name) / "test_policy_cost_dataset"
        core = [
            "src/lc.c", "src/features.c", "src/net.c", "src/heuristic.c",
            "src/planner.c", "src/search.c", "src/rollout.c",
            "src/late_resolver.c", "src/match_value.c", "src/policy_cost.c",
            "src/agent.c", "src/match.c", "src/spec.c",
        ]
        command = [
            "gcc", "-O0", "-Wall", "-Wextra", "-std=c11",
            "-fno-fast-math", "-ffp-contract=off", "-o", str(cls.binary),
            "tests/test_policy_cost_dataset.c", *core, "-lm", "-pthread",
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_white_box_contracts(self) -> None:
        completed = subprocess.run(
            [str(self.binary)], cwd=ROOT, check=True, capture_output=True, text=True
        )
        self.assertIn("contract tests: ok", completed.stdout)

    def test_tool_self_test_and_audit_only_hash_probe(self) -> None:
        tool = Path(self.temp.name) / "policy_cost_dataset"
        core = [
            "src/lc.c", "src/features.c", "src/net.c", "src/heuristic.c",
            "src/planner.c", "src/search.c", "src/rollout.c",
            "src/late_resolver.c", "src/match_value.c", "src/policy_cost.c",
            "src/agent.c", "src/match.c", "src/spec.c",
        ]
        subprocess.run([
            "gcc", "-O0", "-Wall", "-Wextra", "-std=c11",
            "-fno-fast-math", "-ffp-contract=off", "-o", str(tool),
            "tools/policy_cost_dataset_v2.c", *core, "-lm", "-pthread",
        ], cwd=ROOT, check=True, capture_output=True, text=True)
        run = subprocess.run([str(tool), "self-test"], cwd=ROOT, check=True,
                             capture_output=True, text=True)
        self.assertIn("self-test: ok", run.stdout)
        help_text = subprocess.run([str(tool), "--help"], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout
        self.assertIn("hash-probe --state PATH", help_text)
        self.assertIn("--reservoir-sha256", help_text)
        probe = subprocess.run(
            [str(tool), "hash-probe", "--state", "data/probes/g424_p111.state"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        self.assertIn('"suit_orbit_information_view_sha256"', probe.stdout)
        self.assertIn('"information_view_sha256"', probe.stdout)
        source = (ROOT / "tools" / "policy_cost_dataset_v2.c").read_text(encoding="utf-8")
        self.assertIn("ROLLOUT_AUDIT_PANEL_PRIMARY", source)
        self.assertIn("ROLLOUT_AUDIT_PANEL_FRESH", source)
        self.assertIn("exact_terminal_preempted", source)
        self.assertIn("ROLLOUT_MAX_CANDIDATES", source)

    def test_real_native_discover_accepts_canonical_exclusion_manifest(self) -> None:
        tool = Path(self.temp.name) / "policy_cost_dataset_discover"
        core = [
            "src/lc.c", "src/features.c", "src/net.c", "src/heuristic.c",
            "src/planner.c", "src/search.c", "src/rollout.c",
            "src/late_resolver.c", "src/match_value.c", "src/policy_cost.c",
            "src/agent.c", "src/match.c", "src/spec.c",
        ]
        subprocess.run([
            "gcc", "-O0", "-Wall", "-Wextra", "-std=c11",
            "-fno-fast-math", "-ffp-contract=off", "-o", str(tool),
            "tools/policy_cost_dataset_v2.c", *core, "-lm", "-pthread",
        ], cwd=ROOT, check=True, capture_output=True, text=True)
        exclusions = ROOT / "data" / "experiments" / "policy_cost_v2_exact17_exclusions.txt"
        exclusion_sha = hashlib.sha256(exclusions.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "discovery.jsonl"
            reservoir = Path(directory) / "reservoir.tsv"
            completed = subprocess.run([
                str(tool), "discover", "--out", str(output),
                "--reservoir-out", str(reservoir), "--net", "data/champion.bin",
                "--split", "TRAIN", "--matches", "1", "--match-start", "0",
                "--reservoir-per-cell", "1", "--exclusions", str(exclusions),
                "--exclusions-sha256", exclusion_sha,
                "--smoke-seed", "202612290999",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertEqual(completed.stderr, "")
            self.assertTrue(output.is_file())
            self.assertTrue(reservoir.is_file())
            self.assertIn('"seed_domain":"20261229-smoke"',
                          output.read_text(encoding="utf-8"))
            self.assertIn(
                '"burned_source_deal_seeds":"1..200, maintained-800 seed 1, '
                '202611010101, all policy-cost-v1 fixed seeds in '
                '20261110/11/12/13/14/15/16/21/22, every 20261129 '
                'feasibility-smoke seed, 202612010101, and every 20261229 '
                'feasibility-smoke seed"',
                output.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
