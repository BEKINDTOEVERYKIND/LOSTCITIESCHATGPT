"""Mutation contracts for the sealed policy-cost holdout allocation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from tools import policy_cost_campaign as campaign
from tools import policy_cost_selection as selection


class MaintainedActorProfileTests(unittest.TestCase):
    OBJECTIVE0 = (
        "rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:3.5:2:"
        "4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1"
    )
    OBJECTIVE3 = (
        "rolloutu2:data/champion.bin:data/champion.bin:800:5:0.02:0:1:0:"
        "0:0:3:0:3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:"
        "0:0:0:0:0:1:0:0:0:1:0:"
        "data/models/match_value_objective3_v2_projected.lcmv"
    )

    def test_accepts_only_exact_disposition_profiles(self) -> None:
        objective0 = campaign.parse_maintained_actor(self.OBJECTIVE0)
        self.assertEqual(objective0["objective"], 0)
        self.assertEqual(objective0["kind"], "rolloutu")
        objective3 = campaign.parse_maintained_actor(self.OBJECTIVE3)
        self.assertEqual(objective3["objective"], 3)
        self.assertEqual(objective3["kind"], "rolloutu2")
        self.assertEqual(
            objective3["match_value_path"],
            "data/models/match_value_objective3_v2_projected.lcmv",
        )

    def test_rejects_crossed_objective_onset_and_controller_drift(self) -> None:
        mutations = (
            self.OBJECTIVE0.replace(":14:0:0:0:0:3.5", ":0:0:0:0:0:3.5"),
            self.OBJECTIVE3.replace(":0:0:0:3:0:3.5", ":14:0:0:3:0:3.5"),
            self.OBJECTIVE3.replace(":3.5:2:4:20", ":3.5:1:4:20"),
            self.OBJECTIVE3.replace("rolloutu2:", "rolloutu:", 1),
        )
        for actor in mutations:
            with self.subTest(actor=actor), self.assertRaises(
                    campaign.EvidenceError):
                campaign.parse_maintained_actor(actor)

    def test_rejects_actor_and_binding_paths_outside_canonical_root(self) -> None:
        for replacement in ("../data/champion.bin", "/tmp/champion.bin",
                            "data//champion.bin", "data\\champion.bin"):
            actor = self.OBJECTIVE0.replace(
                "data/champion.bin", replacement, 1
            )
            with self.subTest(path=replacement), self.assertRaisesRegex(
                    campaign.EvidenceError, "canonical relative path"):
                campaign.parse_maintained_actor(actor)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inside").write_text("bound", encoding="ascii")
            self.assertEqual(campaign.binding(root, "inside")["path"], "inside")
            outside = root.parent / f"{root.name}-outside"
            outside.write_text("escape", encoding="ascii")
            try:
                with self.assertRaisesRegex(
                        campaign.EvidenceError, "canonical relative path"):
                    campaign.binding(root, f"../{outside.name}")
                (root / "link").symlink_to(outside)
                with self.assertRaisesRegex(campaign.EvidenceError, "symlink"):
                    campaign.binding(root, "link")
            finally:
                outside.unlink()

    def test_authoritative_prerequisite_reopens_every_evidence_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/experiments").mkdir(parents=True)
            (root / "data").mkdir(exist_ok=True)
            (root / "data/champion.bin").write_bytes(b"champion")
            evidence_path = root / "data/experiments/o3-terminal.json"
            evidence_path.write_bytes(b"terminal evidence\n")
            result_path = root / campaign.PREREQUISITE_PATH

            def write_result(path: str, digest: str) -> None:
                value = {
                    "schema": "lc-match-value-objective3-v2-result-v1",
                    "artifact_kind":
                        "match_value_objective3_v2_authoritative_result",
                    "status": "complete",
                    "locked_validation_relaxed": False,
                    "diagnostic_audit_used_for_selection": False,
                    "promotion_gate_passed": False,
                    "baseline_actor": self.OBJECTIVE0,
                    "challenger_actor": self.OBJECTIVE3,
                    "winner_actor": self.OBJECTIVE0,
                    "disposition": "retained_baseline",
                    "evidence": [{"path": path, "sha256": digest}],
                }
                result_path.write_text(
                    json.dumps(value, sort_keys=True) + "\n", encoding="ascii"
                )

            write_result(
                "data/experiments/o3-terminal.json",
                hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(
                    campaign.EvidenceError, "omits required files"):
                campaign.authoritative_prerequisite(root)
            write_result("data/experiments/missing.json", "0" * 64)
            with self.assertRaisesRegex(campaign.EvidenceError, "absent"):
                campaign.authoritative_prerequisite(root)
            write_result("../escape.json", "0" * 64)
            with self.assertRaisesRegex(
                    campaign.EvidenceError, "canonical relative path"):
                campaign.authoritative_prerequisite(root)
            write_result("data/experiments/o3-terminal.json", "0" * 64)
            with self.assertRaisesRegex(campaign.EvidenceError, "SHA-256"):
                campaign.authoritative_prerequisite(root)
            link = root / "data/experiments/o3-link.json"
            link.symlink_to(evidence_path)
            write_result(
                "data/experiments/o3-link.json",
                hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(campaign.EvidenceError, "symlink"):
                campaign.authoritative_prerequisite(root)

    def test_objective3_completeness_requires_all_fixed_raw_triplets(self) -> None:
        names = {
            "pre-efficacy-manifest.json", "transport/BUILD_INFO.txt",
            "transport/SHA256SUMS.txt", "transport/bindings/actors.json",
            "transport/bindings/definition-lock.json",
            "transport/bindings/execution.json", "transport/bindings/plan.json",
            "transport/bindings/pre-efficacy-manifest.json",
            "transport/bindings/table-manifest.json",
            "transport/data/models/match_value_objective3_v2_raw.lcmv",
            "transport/data/models/match_value_objective3_v2_projected.lcmv",
            "development/merged/development-selection.json",
            "development/merged/RAW_ALL_PLY-reciprocal.json",
            "development/merged/PROJECTED_ALL_PLY-reciprocal.json",
            "stages/safety-skipped.json", "stages/final-skipped.json",
        }
        names.update(campaign._objective3_raw_triplets(False, False))
        evidence = [{"path": name, "sha256": "0" * 64}
                    for name in sorted(names)]
        value = {"challenger_actor": None, "safety": None, "final": None}
        campaign._validate_objective3_evidence_completeness(value, evidence)
        target = next(
            row for row in evidence
            if row["path"].endswith("candidate-first-0.jsonl")
        )
        target["path"] = "development/downloads/padded.jsonl"
        with self.assertRaises(campaign.EvidenceError):
            campaign._validate_objective3_evidence_completeness(value, evidence)

    def test_objective3_archive_checksum_manifest_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "a/first").write_bytes(b"first\n")
            (root / "second").write_bytes(b"second\n")
            manifest = root / "SHA256SUMS.txt"
            rows = []
            for name in ("a/first", "second"):
                rows.append(
                    f"{campaign.sha256(root / name)}  {name}\n"
                )
            manifest.write_text("".join(rows), encoding="ascii")
            campaign._verify_sha256sum_tree(root, manifest)
            (root / "extra").write_bytes(b"not sealed")
            with self.assertRaisesRegex(
                    campaign.EvidenceError, "member set"):
                campaign._verify_sha256sum_tree(root, manifest)

    def test_repository_objective3_promotion_is_raw_replayable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        prerequisite = campaign.authoritative_prerequisite(root)
        self.assertTrue(prerequisite["promotion_gate_passed"])
        self.assertEqual(prerequisite["disposition"], "final_passed")
        self.assertEqual(prerequisite["actor"]["spec"], self.OBJECTIVE3)
        self.assertEqual(prerequisite["terminal_evidence_files"], 550)

        from tools.flagged_ply_execution import authoritative_final_result

        final = authoritative_final_result(root)
        self.assertTrue(final["promotion_gate_passed"])
        self.assertEqual(final["selection_mode"], "component_final")
        self.assertEqual(final["winner"]["spec"], self.OBJECTIVE3)

    def test_finite_support_contract_distinguishes_census_from_sample(self) -> None:
        self.assertTrue(campaign._finite_support_worlds(137, 137, True))
        self.assertTrue(campaign._finite_support_worlds(800, 800, True))
        self.assertTrue(campaign._finite_support_worlds(800, 0, False))
        self.assertTrue(campaign._finite_support_worlds(800, 1200, False))
        self.assertFalse(campaign._finite_support_worlds(137, 200, True))
        self.assertFalse(campaign._finite_support_worlds(137, 0, False))
        self.assertFalse(campaign._finite_support_worlds(800, 800, False))
        self.assertFalse(campaign._finite_support_worlds(800, 0, None))

    def test_generated_actor_tails_match_native_optional_field_abi(self) -> None:
        objective0 = campaign._tail(
            floor=0.01, ply_lo=0, objective=0, match_value_path=None
        )
        objective3 = campaign._tail(
            floor=0.01, ply_lo=0, objective=3,
            match_value_path="data/models/table.lcmv",
        )
        self.assertEqual(len(objective0), 40)
        self.assertEqual(len(objective3), 42)
        self.assertEqual(objective3[-2:], ["0", "data/models/table.lcmv"])

    def test_full_policy_uses_complete_semantic_wager_move_pack(self) -> None:
        def bits(value: float) -> str:
            return f"{campaign.struct.unpack('<I', campaign.struct.pack('<f', value))[0]:08x}"

        def f32(value: float) -> float:
            return campaign.struct.unpack(
                "<f", campaign.struct.pack("<f", value)
            )[0]

        left, right = f32(0.6), f32(0.4)
        core = left + right
        legal = [
            {"index": 0, "move_pack": 1, "semantic_move_pack": 0,
             "card": 1, "discard": 0, "draw": 0, "probability": 0.6,
             "probability_bits": bits(0.6),
             "semantic_action_probability": core,
             "conditional_draw_probability": left / core},
            {"index": 1, "move_pack": 121, "semantic_move_pack": 120,
             "card": 1, "discard": 0, "draw": 1, "probability": 0.4,
             "probability_bits": bits(0.4),
             "semantic_action_probability": core,
             "conditional_draw_probability": right / core},
        ]
        policy = {"legal": legal, "legal_count": 2, "symmetries": 20,
                  "exact_group_average": True, "literal_argmax_index": 0}
        campaign._verify_full_policy(policy)
        self.assertEqual(campaign._semantic_core_from_pack(0),
                         campaign._semantic_core_from_pack(120))
        broken = copy.deepcopy(policy)
        broken["legal"][0]["semantic_move_pack"] = 1
        with self.assertRaisesRegex(campaign.EvidenceError, "packing"):
            campaign._verify_full_policy(broken)

    def test_full_policy_accepts_native_binary32_normalization_tolerance(self) -> None:
        def f32(value: float) -> float:
            return campaign.struct.unpack(
                "<f", campaign.struct.pack("<f", value)
            )[0]

        def bits(value: float) -> str:
            return f"{campaign.struct.unpack('<I', campaign.struct.pack('<f', value))[0]:08x}"

        left, right = f32(0.6), f32(0.4000001)
        self.assertGreater(abs((left + right) - 1.0), 2.0e-9)
        policy = {
            "legal": [
                {"index": 0, "move_pack": 3, "semantic_move_pack": 3,
                 "card": 3, "discard": 0, "draw": 0,
                 "probability": left, "probability_bits": bits(left),
                 "semantic_action_probability": left,
                 "conditional_draw_probability": 1.0},
                {"index": 1, "move_pack": 4, "semantic_move_pack": 4,
                 "card": 4, "discard": 0, "draw": 0,
                 "probability": right, "probability_bits": bits(right),
                 "semantic_action_probability": right,
                 "conditional_draw_probability": 1.0},
            ],
            "legal_count": 2, "symmetries": 20,
            "exact_group_average": True, "literal_argmax_index": 0,
        }
        campaign._verify_full_policy(policy)

    def test_full_policy_reconstructs_native_f32_before_summary_algebra(self) -> None:
        # Native prints an individual float with %.9g but accumulates P_A in
        # double from the exact binary32 value.  The printed decimal is enough
        # to recover the bits, but is not itself the value used in the sum.
        left_decimal, right_decimal = 0.333333343, 0.666666687
        left = campaign.struct.unpack(
            "<f", campaign.struct.pack("<f", left_decimal)
        )[0]
        right = campaign.struct.unpack(
            "<f", campaign.struct.pack("<f", right_decimal)
        )[0]
        self.assertGreater(abs(left_decimal - left), 2.0e-12)
        policy = {
            "legal": [
                {"index": 0, "move_pack": 3, "semantic_move_pack": 3,
                 "card": 3, "discard": 0, "draw": 0,
                 "probability": left_decimal,
                 "probability_bits": f"{campaign.struct.unpack('<I', campaign.struct.pack('<f', left_decimal))[0]:08x}",
                 "semantic_action_probability": left,
                 "conditional_draw_probability": 1.0},
                {"index": 1, "move_pack": 4, "semantic_move_pack": 4,
                 "card": 4, "discard": 0, "draw": 0,
                 "probability": right_decimal,
                 "probability_bits": f"{campaign.struct.unpack('<I', campaign.struct.pack('<f', right_decimal))[0]:08x}",
                 "semantic_action_probability": right,
                 "conditional_draw_probability": 1.0},
            ],
            "legal_count": 2,
            "symmetries": 20,
            "exact_group_average": True,
            "literal_argmax_index": 1,
        }
        campaign._verify_full_policy(policy)

    def test_runtime_mask_keeps_rank_four_literal_plus_true_top_three(self) -> None:
        def f32(value: float) -> float:
            return campaign.struct.unpack(
                "<f", campaign.struct.pack("<f", value)
            )[0]

        raw = [
            (3, 0, 0.20),
            (4, 0, 0.11), (4, 1, 0.11),
            (5, 0, 0.105), (5, 1, 0.105),
            (6, 0, 0.1025), (6, 1, 0.1025),
            (7, 0, 0.165),
        ]
        probabilities = [f32(item[2]) for item in raw]
        core_totals: dict[int, float] = {}
        for (card, _, _), probability in zip(raw, probabilities):
            core_totals[card] = core_totals.get(card, 0.0) + probability
        legal = []
        for index, ((card, draw, _), probability) in enumerate(
                zip(raw, probabilities)):
            legal.append({
                "index": index,
                "move_pack": card + 120 * draw,
                "semantic_move_pack": card + 120 * draw,
                "card": card, "discard": 0, "draw": draw,
                "probability": probability,
                "probability_bits":
                    f"{campaign.struct.unpack('<I', campaign.struct.pack('<f', probability))[0]:08x}",
                "semantic_action_probability": core_totals[card],
                "conditional_draw_probability": probability / core_totals[card],
            })
        def runtime_mask(floor: float, indices: list[int]) -> dict[str, object]:
            floor_bits = campaign.struct.unpack(
                "<I", campaign.struct.pack("<f", floor)
            )[0]
            packed = [legal[index]["semantic_move_pack"] for index in indices]
            encoded = bytes([
                len(packed),
                (floor_bits ^ (floor_bits >> 8) ^ (floor_bits >> 16) ^
                 (floor_bits >> 24)) & 0xff,
            ]) + b"".join(int(value).to_bytes(2, "little") for value in packed)
            selected_cores = {raw[index][0] for index in indices}
            return {
                "floor": floor, "floor_bits": f"{floor_bits:08x}",
                "count": len(indices), "core_candidates": len(indices),
                "draw_candidates": 0,
                "complete_move_mass": sum(probabilities[index] for index in indices),
                "semantic_core_mass": sum(core_totals[card] for card in selected_cores),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "legal_indices": indices,
            }

        # Literal index 0 is aggregate rank four.  Ranks one through three are
        # the representative indices 1, 3, and 5.
        first = runtime_mask(0.01, [0, 1, 3, 5])
        second = runtime_mask(0.02, [0, 1, 3, 5])
        policy = {
            "legal": legal, "legal_count": len(legal), "symmetries": 20,
            "exact_group_average": True, "literal_argmax_index": 0,
            "runtime_masks": [first, second],
        }
        campaign._verify_full_policy(policy)
        expected = {
            "mask_001_sha256": first["sha256"],
            "mask_002_sha256": second["sha256"],
            "master_sha256": first["sha256"],
        }
        self.assertEqual(
            campaign._verify_runtime_masks(policy, expected),
            [[3, 4, 5, 6], [3, 4, 5, 6]],
        )
        broken = copy.deepcopy(policy)
        broken["runtime_masks"][0]["legal_indices"] = [0, 1, 3]
        broken["runtime_masks"][0]["count"] = 3
        broken["runtime_masks"][0]["core_candidates"] = 3
        with self.assertRaisesRegex(
                campaign.EvidenceError, "candidate zero plus"):
            campaign._verify_runtime_masks(broken, expected)

    def test_native_train_panel_uses_u64_not_sha256_fingerprint(self) -> None:
        # Shape and number formatting mirror print_train_pair_panel(): the
        # hidden-world identity is an FNV-style uint64 printed with %016llx.
        panel = {
            "seed": "123", "requested_worlds": 800, "panel_role": 0,
            "hidden_world_fingerprint": "0123456789abcdef", "worlds": 800,
            "common_worlds_across_pair": True,
            "exact_hidden_support": False, "hidden_support": 0,
            "exact_terminal_leaves": 0, "unfinished_cap_leaves": 0,
            "cycle_breaks": 0, "cap_reserve_forces": 0,
            "actions": [
                {"position": 0, "legal_index": 3, "mean": 1.0,
                 "se": 0.0, "sum": 800.0, "sum_squares": 800.0},
                {"position": 1, "legal_index": 4, "mean": 0.0,
                 "se": 0.0, "sum": 0.0, "sum_squares": 0.0},
            ],
            "pair": {"delta_a_minus_b": 1.0, "paired_se": 0.0,
                     "sum_products": 0.0},
        }
        policy = {
            10: {"index": 3, "semantic_move_pack": 10},
            11: {"index": 4, "semantic_move_pack": 11},
        }
        campaign._verify_train_pair_panel(
            panel, seed="123", role=0, semantic_moves=[10, 11], policy=policy
        )
        panel["hidden_world_fingerprint"] = "a" * 64
        with self.assertRaisesRegex(campaign.EvidenceError, "protocol"):
            campaign._verify_train_pair_panel(
                panel, seed="123", role=0, semantic_moves=[10, 11],
                policy=policy,
            )


class VectorAllocationManifestTests(unittest.TestCase):
    @staticmethod
    def _allocation_state_hex(serial: int) -> str:
        state = bytearray(174)
        state[0] = 1
        state[157:165] = serial.to_bytes(8, "little")
        return state.hex()

    def test_plan_freezes_artifact_and_controller_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        campaign.validate_plan(campaign.strict_json(root / campaign.PLAN_PATH))

    def test_train_allocator_pair_digest_roundtrip_and_mutation(self) -> None:
        """Match the allocator's SHA(state_sha256 bytes || LE moves) rule."""

        lines = [
            "LCPOLICYCOST-TRAIN-ALLOCATION-V2", "split\tTRAIN",
            "purpose\tcampaign", "discovery_sha256\t" + "a" * 64,
            "reservoir_sha256\t" + "b" * 64,
            "source_net_sha256\t" + "c" * 64,
            "source_exclusion_sha256\t" + "d" * 64,
            "eligible_pair_commitment_sha256\t" + "e" * 64,
            "allocation_rule_sha256\t" + campaign.TRAIN_ALLOCATION_RULE_SHA256,
            "quota_per_cell\t16", "eligible_units\t13824",
            "retained_reservoir_units\t13824", "probe_orbit_rejections\t17",
            "pooled_ge64_observed\t0", "records\t13824",
            "columns\t" + "\t".join(campaign.TRAIN_ALLOCATION_COLUMNS),
        ]
        first_pair_sha = ""
        for allocation_id in range(13824):
            rd, ply, ratio, pair_type = campaign._train_scheduled_cell(
                allocation_id
            )
            state_hex = self._allocation_state_hex(allocation_id)
            state_sha = hashlib.sha256(bytes.fromhex(state_hex)).hexdigest()
            pair_sha = hashlib.sha256(
                bytes.fromhex(state_sha) + b"\x0a\x00\x0b\x00"
            ).hexdigest()
            if not first_pair_sha:
                first_pair_sha = pair_sha
            source = f"TRAIN-{allocation_id:012d}"
            lines.append("\t".join((
                str(allocation_id), str(allocation_id), "0", source,
                source + ":s000", "00010-00011",
                f"r{rd}.p{ply}.g{ratio}.t{pair_type}", str(rd),
                str(ply), str(ratio), str(pair_type), "10", "11",
                "f" * 64, state_sha, pair_sha,
                f"{allocation_id + 1:064x}", "d" * 64,
                "e" * 64, "d" * 64, state_hex,
            )))
        text = "\n".join(lines) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            allocation = Path(directory) / "train.tsv"
            allocation.write_text(text, encoding="ascii")
            manifest, _ = campaign.train_allocation_manifest(allocation)
            self.assertEqual(len(manifest["selected_units"]), 13824)
            allocation.write_text(
                text.replace(first_pair_sha, "0" * 64, 1), encoding="ascii"
            )
            with self.assertRaisesRegex(campaign.EvidenceError, "state/pair hash"):
                campaign.train_allocation_manifest(allocation)

    def _allocation(self, split: str = "SELECT") -> str:
        rule = campaign.VECTOR_ALLOCATION_RULE_SHA256
        lines = [
            "LCPOLICYCOST-VECTOR-ALLOCATION-V1", f"split\t{split}",
            "purpose\tcampaign", "discovery_sha256\t" + "a" * 64,
            "reservoir_sha256\t" + "b" * 64,
            "source_net_sha256\t" + "c" * 64,
            "source_exclusion_sha256\t" + "d" * 64,
            "eligible_state_commitment_sha256\t" + "c" * 64,
            "allocation_rule_sha256\t" + rule,
            "quota_per_base_cell\t64", "source_minimum_per_positive_slot\t8",
            "total_census\t9216", "retained_reservoir_vectors\t9216",
            "poststratum_cells\t720", "aggregate_master_width_histogram\t9216,0,0,0,0",
            "probe_orbit_rejections\t17", "pooled_ge64_observed\t0",
            "records\t9216",
        ]
        for rd in range(3):
            for ply in range(24):
                for frontier in range(2):
                    for slot in range(5):
                        name = f"r{rd}:p{ply:02d}:f{frontier}:j{slot}"
                        count = quota = 64 if slot == 0 else 0
                        widths = "64,0,0,0,0" if slot == 0 else "0,0,0,0,0"
                        lines.append(
                            f"poststratum\t{name}\t{count}\t{quota}\t{count}\t9216\t{widths}"
                        )
        lines.append("columns\t" + "\t".join(campaign.VECTOR_ALLOCATION_COLUMNS))
        for allocation_id in range(9216):
            rd, ply, frontier = campaign._vector_scheduled_base(allocation_id)
            name = f"r{rd}:p{ply:02d}:f{frontier}:j0"
            source = allocation_id
            state_hex = self._allocation_state_hex(allocation_id)
            state_hash = hashlib.sha256(bytes.fromhex(state_hex)).hexdigest()
            lines.append("\t".join((
                str(allocation_id), str(source), "0",
                f"{split}-{source:012d}", f"{split}-{source:012d}:s000",
                str(rd), str(ply), str(frontier), "0", name, "1",
                "64", "64", "64", str(64 * 9216), "d" * 64,
                state_hash, f"{allocation_id + 1:064x}", "e" * 64,
                "f" * 64, "e" * 64, state_hex, "a" * 64,
            )))
        return "\n".join(lines) + "\n"

    def _manifest(self, mutate=None, split: str = "SELECT"):
        text = self._allocation(split)
        if mutate:
            text = mutate(text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "select.tsv"
            path.write_text(text, encoding="ascii")
            return campaign.vector_allocation_manifest(path, split)

    def test_binds_exact_even_slots_and_componentwise_widths(self) -> None:
        manifest, rows, weights = self._manifest()
        self.assertEqual(len(rows), 9216)
        self.assertEqual(sum(weights.values()), 1.0)
        self.assertEqual(manifest["allocation_rule_sha256"],
                         campaign.VECTOR_ALLOCATION_RULE_SHA256)
        self.assertTrue(all(set(unit) == {
            "source_match", "unit", "state_sha256",
            "allocation_priority_sha256", "round", "ply_stratum",
            "frontier_present", "allocation_slot", "master_width",
            "post_stratum",
        } for unit in manifest["selected_units"]))

    def test_rejects_mutated_rule_digest(self) -> None:
        with self.assertRaisesRegex(campaign.EvidenceError, "digest"):
            self._manifest(lambda text: text.replace(
                campaign.VECTOR_ALLOCATION_RULE_SHA256, "0" * 64, 1))

    def test_rejects_mutated_width_aggregate(self) -> None:
        with self.assertRaisesRegex(campaign.EvidenceError, "width histogram"):
            self._manifest(lambda text: text.replace(
                "aggregate_master_width_histogram\t9216,0,0,0,0",
                "aggregate_master_width_histogram\t9215,1,0,0,0", 1))

    def test_rejects_zero_census_with_positive_quota(self) -> None:
        with self.assertRaisesRegex(campaign.EvidenceError, "census/quota"):
            self._manifest(lambda text: text.replace(
                "r0:p00:f0:j1\t0\t0\t0\t9216",
                "r0:p00:f0:j1\t0\t1\t0\t9216", 1))


class WorkflowLockTests(VectorAllocationManifestTests):
    def test_preflight_never_overwrites_retained_objective3_transport(self) -> None:
        root = Path(__file__).resolve().parents[1]
        jobs = yaml.safe_load((root / ".github/workflows" /
                               "policy-cost-v1.yml").read_text())["jobs"]
        steps = jobs["preflight"]["steps"]
        materialize = next(
            step["run"] for step in steps
            if step.get("name") == "Materialize and compile the bound parent once"
        )
        freeze = next(
            step["run"] for step in steps
            if step.get("name") ==
            "Freeze a source-free transport before any search or truth label"
        )
        upload = next(
            step["with"] for step in steps
            if step.get("with", {}).get("name") == "policy-cost-v1-transport"
        )

        # The source archive contains the authoritative O3 tree at
        # repo-root transport/.  The new campaign must be assembled elsewhere
        # until artifact upload, while reading O3 members from their original
        # repository-relative paths.
        self.assertIn("git -C campaign archive HEAD^ | tar -x", materialize)
        self.assertIn("mkdir -p policy-transport/", freeze)
        self.assertIn("source = relative", freeze)
        self.assertIn(
            "Path('policy-transport/bindings/objective3/repo') / relative",
            freeze,
        )
        self.assertNotIn("mkdir -p transport/", freeze)
        self.assertNotIn("Path('transport/bindings/objective3/repo')", freeze)
        self.assertEqual(upload["path"], "policy-transport")

    def test_push_only_source_free_staged_workflow(self) -> None:
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/policy-cost-v1.yml")
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("locked_policy_cost_v1_execution.json", text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("retry", text.lower())
        self.assertEqual(text.count("actions/checkout@"), 1)
        self.assertGreaterEqual(text.count("sha256sum -c SHA256SUMS"), 6)
        self.assertIn("--matches 65536", text)
        self.assertEqual(text.count("--matches 32768"), 2)
        self.assertEqual(text.count("verify-reservoir --out"), 3)
        for split in ("train", "select", "test"):
            self.assertIn(f"{split}-reservoir-proof.json", text)
        self.assertIn("test_once:", text)
        self.assertIn("needs: [select_configuration, reservoir_barrier]", text)
        self.assertGreaterEqual(text.count("merge-evaluation-slices"), 3)
        self.assertIn("--expect-pairs 200", text)
        self.assertIn("--expect-pairs 2500", text)
        self.assertNotIn("for PATH in", text)
        self.assertIn(
            "find . -type f ! -name SHA256SUMS -print0", text
        )
        self.assertEqual(text.count("cd transport;"),
                         text.count("cd transport; chmod 0755 "))
        self.assertIn("'mode': f'{path.stat().st_mode & 0o777:04o}'", text)
        for key, value in campaign.NUMERIC_RUNTIME_ENV.items():
            self.assertIn(f"{key}: '{value}'" if value.isdigit() else
                          f"{key}: {value}", text)
        self.assertGreaterEqual(text.count("--no-index --no-deps --target"), 4)
        self.assertIn("terminal_recommendation:\n", text)
        terminal = text.split("  terminal_recommendation:\n", 1)[1].split(
            "\n  infrastructure_retain:", 1
        )[0]
        self.assertIn("timeout-minutes: 360", terminal)

    def test_matrix_evaluators_use_only_sealed_lean_handoffs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        jobs = yaml.safe_load((root / ".github/workflows" /
                               "policy-cost-v1.yml").read_text())["jobs"]

        def downloads(job: str) -> list[str]:
            return [str(step.get("with", {}).get("name"))
                    for step in jobs[job]["steps"] if "uses" in step]

        self.assertIn("policy-cost-v1-calibration-handoff",
                      downloads("select_evaluate"))
        self.assertNotIn("policy-cost-v1-calibration",
                         downloads("select_evaluate"))
        self.assertIn("policy-cost-v1-selection-handoff",
                      downloads("test_evaluate"))
        self.assertNotIn("policy-cost-v1-selection",
                         downloads("test_evaluate"))
        workflow = (root / ".github/workflows/policy-cost-v1.yml").read_text()
        self.assertIn('RAW_PATH.sha256', workflow)
        self.assertGreaterEqual(workflow.count('test -f "$SIDE"'), 5)

    def test_all_reservoir_allocations_precede_every_evaluator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        jobs = yaml.safe_load((root / ".github/workflows" /
                               "policy-cost-v1.yml").read_text())["jobs"]
        evaluators = {"train_evaluate", "select_evaluate", "test_evaluate"}

        def upstream(name: str) -> set[str]:
            raw = jobs[name].get("needs", [])
            direct = {raw} if isinstance(raw, str) else set(raw)
            return direct | {node for parent in direct for node in upstream(parent)}

        barrier = upstream("train_evaluate")
        self.assertIn("reservoir_barrier", barrier)
        self.assertTrue({"train_discover", "select_discover", "test_discover"}
                        <= upstream("reservoir_barrier"))
        for discovery in ("train_discover", "select_discover", "test_discover"):
            self.assertFalse(upstream(discovery) & evaluators)

    def test_failed_safety_does_not_masquerade_as_incomplete_final(self) -> None:
        root = Path(__file__).resolve().parents[1]
        jobs = yaml.safe_load((root / ".github/workflows" /
                               "policy-cost-v1.yml").read_text())["jobs"]
        script = next(
            step["run"] for step in jobs["terminal_recommendation"]["steps"]
            if "run" in step
        )
        self.assertIn(
            'json.load(open("safety-gate.json"))["passed"]', script
        )

    def test_reservoir_freeze_is_verified_and_retained_through_terminal(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / ".github/workflows" / "policy-cost-v1.yml").read_text()
        self.assertGreaterEqual(text.count("policy-cost-v1-reservoir-freeze"), 7)
        self.assertGreaterEqual(text.count("verify-reservoir-freeze"), 6)
        self.assertIn("bindings/reservoir-freeze.json", text)


class EvaluationSliceMergeTests(VectorAllocationManifestTests):
    @staticmethod
    def _slices() -> list[list[dict]]:
        per_slice = campaign.HOLDOUT_RECORDS // 192
        header = {
            "schema": "lc-policy-cost-evaluation-v1", "record_type": "header",
            "split": "SELECT", "manifest_sha256": "a" * 64,
            "full_manifest_records": campaign.HOLDOUT_RECORDS,
            "reservoir_sha256": "b" * 64, "discovery_sha256": "c" * 64,
            "actor_spec": "rolloutu5:bound", "root_net_fingerprint": "1111",
            "continuation_net_fingerprint": "1111",
            "policy_cost_sha256": "d" * 64,
            "policy_cost_payload_fingerprint": "2222",
            "truth_net_sha256": "e" * 64,
            "primary": {"worlds": 800, "seed": campaign.PRIMARY_SEEDS["SELECT"]},
            "fresh": {"worlds": 800, "seed": campaign.FRESH_SEEDS["SELECT"]},
            "truth": {"worlds": 1024, "seed": campaign.TRUTH_SEEDS["SELECT"],
                      "controller": "exact_policy20_full_remaining_match"},
        }
        footer = {
            "schema": "lc-policy-cost-evaluation-v1", "record_type": "footer",
            "all_exact": True, "all_mask_overlaps_bit_exact": True,
            "primary_unfinished_cap_leaves": 0, "fresh_unfinished_cap_leaves": 0,
            "truth_cap_hits": 0, "maintained_unfinished_cap_leaves": 0,
            "exact_terminal_leaves": 1, "maintained_exact_terminal_leaves": 0,
        }
        result = []
        for ordinal in range(192):
            start = ordinal * per_slice
            result.append([
                {**header, "allocation_start": start, "allocation_count": per_slice},
                *[{"record_type": "allocation", "allocation_id": allocation_id}
                  for allocation_id in range(start, start + per_slice)],
                {**footer, "allocation_start": start, "records": per_slice},
            ])
        return result

    def test_rejects_any_noncoordinate_provenance_or_footer_drift(self) -> None:
        paths = [Path(f"{index}.jsonl") for index in range(192)]
        mutations = (
            (0, "header", "policy_cost_sha256", "f" * 64),
            (1, "header", "root_net_fingerprint", "3333"),
            (2, "header", "actor_spec", "rolloutu5:other"),
            (3, "header", "unexpected_provenance", "drift"),
            (4, "footer", "all_mask_overlaps_bit_exact", False),
        )
        for ordinal, kind, field, value in mutations:
            slices = self._slices()
            target = slices[ordinal][0 if kind == "header" else -1]
            target[field] = value
            with mock.patch.object(campaign, "strict_jsonl", side_effect=slices), \
                    mock.patch.object(campaign, "sha256", return_value="0" * 64):
                with self.assertRaisesRegex(campaign.EvidenceError, "slice (header|footer) binding"):
                    campaign.merge_evaluation_slices(paths, "SELECT")

    def test_merged_evaluation_binds_the_complete_burned_seed_namespace(
        self,
    ) -> None:
        header = {
            "schema": "lc-policy-cost-evaluation-v1",
            "record_type": "header", "split": "SELECT",
            "full_manifest_records": campaign.HOLDOUT_RECORDS,
            "allocation_count": campaign.HOLDOUT_RECORDS,
            "primary": {"worlds": 800,
                        "seed": campaign.PRIMARY_SEEDS["SELECT"]},
            "fresh": {"worlds": 800,
                      "seed": campaign.FRESH_SEEDS["SELECT"]},
            "truth": {
                "worlds": 1024, "seed": campaign.TRUTH_SEEDS["SELECT"],
                "controller": "exact_policy20_full_remaining_match",
            },
            "maintained_root_seed": campaign.MAINTAINED_SEEDS["SELECT"],
            "seed_domains_pairwise_disjoint": True,
            "burned_source_deal_seeds": campaign.BURNED_SOURCE_DEAL_SEEDS,
            "burned_seed_intersection": 0,
        }
        allocations = [
            {"record_type": "allocation", "allocation_id": index}
            for index in range(campaign.HOLDOUT_RECORDS)
        ]
        footer = {
            "schema": "lc-policy-cost-evaluation-v1",
            "record_type": "footer", "records": campaign.HOLDOUT_RECORDS,
            "primary_unfinished_cap_leaves": 0,
            "fresh_unfinished_cap_leaves": 0,
            "truth_cap_hits": 0,
            "maintained_unfinished_cap_leaves": 0,
            "all_exact": True,
        }
        rows = [header, *allocations, footer]
        with mock.patch.object(campaign, "strict_jsonl", return_value=rows):
            campaign._evaluation(Path("ignored.jsonl"), "SELECT")
        for field, replacement in (
            ("burned_source_deal_seeds",
             "1..200 and maintained-800 seed 1"),
            ("burned_seed_intersection", 1),
            ("seed_domains_pairwise_disjoint", False),
        ):
            altered = [dict(header), *allocations, footer]
            altered[0][field] = replacement
            with self.subTest(field=field), mock.patch.object(
                    campaign, "strict_jsonl", return_value=altered), \
                    self.assertRaisesRegex(
                        campaign.EvidenceError, "burned-seed evidence"):
                campaign._evaluation(Path("ignored.jsonl"), "SELECT")

    @staticmethod
    def _select_rows(manifest):
        rows = []
        digest = manifest["canonical_payload_sha256"]
        for unit in manifest["selected_units"]:
            for index, config in enumerate(selection.CONFIG_IDS):
                rows.append({
                    **unit,
                    "config": config,
                    "discovery_census_sha256": digest,
                    "hybrid_gain": 0.01 * index + (
                        0.001 if int(unit["round"]) % 2 else 0.0
                    ),
                    "weight": 1.0 / 9216.0,
                    "exact_valid": True,
                    "capped": 0,
                })
        return rows

    @staticmethod
    def _test_rows(manifest, config):
        digest = manifest["canonical_payload_sha256"]
        return [{
            **unit,
            "config": config,
            "discovery_census_sha256": digest,
            "hybrid_gain": 1.0 + (0.001 if int(unit["round"]) % 2 else 0.0),
            "match_score_gain": 1.0,
            "weight": 1.0 / 9216.0,
            "exact_valid": True,
            "capped": 0,
        } for unit in manifest["selected_units"]]

    def test_campaign_manifests_roundtrip_through_selection_and_test(self) -> None:
        """Campaign's minimal manifest schema is accepted at both stages."""

        select_manifest, _, _ = self._manifest()
        select_rows = self._select_rows(select_manifest)
        with mock.patch.object(selection, "_bootstrap_max_t", return_value=0.0):
            selected = selection.select_configuration(select_rows, select_manifest)
        self.assertTrue(selected["campaign_discovery_binding"]["validated"])

        test_manifest, _, test_weights = self._manifest(split="TEST")
        tested = selection.test_selected_configuration(
            self._test_rows(test_manifest, selected["selected"]["id"]),
            test_weights, selected, test_manifest,
        )
        self.assertTrue(tested["campaign_discovery_binding"]["validated"])

    def test_selection_rejects_enriched_selected_units(self) -> None:
        manifest, bindings, _ = self._manifest()
        manifest["selected_units"][0] = dict(bindings[0])
        payload = dict(manifest)
        del payload["canonical_payload_sha256"]
        manifest["canonical_payload_sha256"] = (
            campaign._canonical_payload_digest(payload)
        )
        with self.assertRaisesRegex(selection.InferenceError, "keys differ"):
            selection._discovery_manifest(manifest, stage="SELECT")

    def test_holdout_keeps_raw_discovery_digest_distinct_from_census_digest(self) -> None:
        """The evaluator binds the TSV's raw digest; inference gets its seal."""

        with tempfile.TemporaryDirectory() as directory:
            allocation = Path(directory) / "select.tsv"
            allocation.write_text(self._allocation(), encoding="ascii")
            manifest, bindings, _ = campaign.vector_allocation_manifest(
                allocation, "SELECT"
            )
            rows = []
            for binding in bindings:
                metrics = {
                    "full_match_hybrid": {"actions": [
                        {"position": 0, "mean": 1.0},
                        {"position": 1, "mean": 0.0},
                    ]},
                    "full_match_score": {"actions": [
                        {"position": 0, "mean": 1.0},
                        {"position": 1, "mean": 0.0},
                    ]},
                }
                rows.append({
                    "source_match_id": binding["source_match_id"],
                    "state_sha256": binding["state_sha256"],
                    "allocation_priority_sha256": (
                        binding["allocation_priority_sha256"]
                    ),
                    "allocation_id": binding["allocation_id"],
                    "source_match_index": binding["source_match_index"],
                    "source_state_index": binding["source_state_index"],
                    "orbit_sha256": binding["orbit_sha256"],
                    "round": binding["round"],
                    "ply_stratum": binding["ply_stratum"],
                    "frontier_present": binding["frontier_present"],
                    "allocation_slot": binding["allocation_slot"],
                    "master_width": binding["master_width"],
                    "post_stratum": binding["post_stratum"],
                    "census_count": binding["census_count"],
                    "allocation_quota": binding["allocation_quota"],
                    "weight_numerator": binding["weight_numerator"],
                    "weight_denominator": binding["weight_denominator"],
                    "discovery_sha256": binding["discovery_sha256"],
                    "unit": binding["unit"],
                    "nply": 0,
                    "policy": {
                        "literal_argmax_index": 0,
                        "runtime_masks": [
                            {"legal_indices": [0, 1]},
                            {"legal_indices": [0]},
                        ],
                    },
                    "production_decisions": {
                        "floor-0.01": {
                            "exact_valid": True,
                            "capped": 0,
                            "selected_legal_index": 0,
                        },
                        "floor-0.02": {
                            "exact_valid": True,
                            "capped": 0,
                            "selected_legal_index": 0,
                        },
                    },
                    "maintained_baseline": {
                        "actor_selected": True,
                        "information_view": True,
                        "unfinished_cap_leaves": 0,
                        "truth_position": 1,
                    },
                    "truth_support_legal_indices": [0, 1],
                    "truth": {"metrics": metrics},
                })
            select_header = {
                "manifest_sha256": campaign.sha256(allocation),
                "discovery_sha256": bindings[0]["discovery_sha256"],
                "reservoir_sha256": manifest["source_reservoir_sha256"],
            }
            with mock.patch.object(
                    campaign, "_evaluation",
                    return_value=(select_header, rows, {})):
                result = campaign.holdout_input(
                    Path("ignored.jsonl"), allocation, "SELECT"
                )
            self.assertNotEqual(
                bindings[0]["discovery_sha256"],
                manifest["canonical_payload_sha256"],
            )
            self.assertEqual(
                result["rows"][0]["discovery_census_sha256"],
                manifest["canonical_payload_sha256"],
            )

            test_allocation = Path(directory) / "test.tsv"
            test_allocation.write_text(self._allocation("TEST"), encoding="ascii")
            test_manifest, test_bindings, _ = campaign.vector_allocation_manifest(
                test_allocation, "TEST"
            )
            test_rows = []
            for row, binding in zip(rows, test_bindings, strict=True):
                copied = copy.deepcopy(row)
                for field in (
                    "source_match_id", "state_sha256",
                    "allocation_priority_sha256", "allocation_id",
                    "source_match_index", "source_state_index", "orbit_sha256",
                    "round", "ply_stratum", "frontier_present",
                    "allocation_slot", "master_width", "post_stratum",
                    "census_count", "allocation_quota", "weight_numerator",
                    "weight_denominator", "discovery_sha256", "unit",
                ):
                    copied[field] = binding[field]
                test_rows.append(copied)
                copied["config_decisions"] = None
            sealed_selection = selection.seal_result({
                "schema": "lc-policy-cost-select-result-v1",
                "stage": "SELECT",
                "selected": {
                    "id": "floor-0.01_ply-00",
                    "policy_floor": 0.01,
                    "ply_lo": 0,
                },
                "campaign_discovery_binding": {
                    "required": True,
                    "validated": True,
                },
                "selection_rule": {"test_evidence_used": False},
            })
            selection_path = Path(directory) / "selection.json"
            selection_path.write_text(
                json.dumps(sealed_selection), encoding="ascii"
            )
            test_header = {
                "manifest_sha256": campaign.sha256(test_allocation),
                "discovery_sha256": test_bindings[0]["discovery_sha256"],
                "reservoir_sha256":
                    test_manifest["source_reservoir_sha256"],
            }
            with mock.patch.object(
                    campaign, "_evaluation",
                    return_value=(test_header, test_rows, {})):
                test_input = campaign.holdout_input(
                    Path("ignored.jsonl"), test_allocation, "TEST", selection_path
                )
            self.assertTrue(all(
                row["weight"] == 1.0 / 9216.0
                for row in test_input["rows"]
            ))

            injected = copy.deepcopy(test_rows)
            injected[0]["config_decisions"] = {
                "floor-0.02_ply-00": {
                    "exact_valid": True,
                    "capped": 0,
                    "selected_legal_index": 0,
                },
            }
            with mock.patch.object(
                    campaign, "_evaluation",
                    return_value=(test_header, injected, {})):
                with self.assertRaisesRegex(
                        campaign.EvidenceError,
                        "configuration decision map must be JSON null"):
                    campaign.holdout_input(
                        Path("ignored.jsonl"), test_allocation, "TEST",
                        selection_path,
                    )

            sealed_selection["selected"]["id"] = "floor-0.02_ply-00"
            selection_path.write_text(
                json.dumps(sealed_selection), encoding="ascii"
            )
            with mock.patch.object(
                    campaign, "_evaluation",
                    return_value=(test_header, test_rows, {})):
                with self.assertRaisesRegex(campaign.EvidenceError, "digest"):
                    campaign.holdout_input(
                        Path("ignored.jsonl"), test_allocation, "TEST", selection_path
                    )

            rows[0]["policy"]["runtime_masks"][1]["legal_indices"] = [1, 0]
            with mock.patch.object(
                    campaign, "_evaluation",
                    return_value=(select_header, rows, {})):
                with self.assertRaisesRegex(campaign.EvidenceError, "no-refill"):
                    campaign.holdout_input(
                        Path("ignored.jsonl"), allocation, "SELECT"
                    )

class DiscardGuardReplayTests(unittest.TestCase):
    @staticmethod
    def _view_hex() -> str:
        state = bytearray(174)
        state[0] = 1
        # Mover 0 holds a live Y10-like card (23) and a publicly dead card
        # (3).  Both players' suit-0 expedition tops are five, making card 3
        # dead by the exact lc_dead_cards rule.
        state[4:12] = ((1 << 23) | (1 << 3)).to_bytes(8, "little")
        state[70] = 5
        state[75] = 5
        state[165] = 0
        state[166] = 0
        return state.hex()

    def test_discard_guard_is_recomputed_from_sealed_view(self) -> None:
        policy = {"legal": [
            {"index": 0, "card": 23, "discard": 0, "draw": 0},
            {"index": 1, "card": 23, "discard": 1, "draw": 0},
        ]}
        state_hex = self._view_hex()
        self.assertFalse(campaign._discard_dominated_from_view(
            state_hex, policy, 0
        ))
        self.assertTrue(campaign._discard_dominated_from_view(
            state_hex, policy, 1
        ))
        rejected = {
            "exact_valid": True, "capped": 0,
            "gate_reason": "discard_guard", "selected_position": 0,
            "selected_legal_index": 0,
        }
        campaign._verify_composed_decision(
            rejected, (1, True), (1, True), [0, 1], active=True,
            discard_rejected=True,
        )
        forged = dict(rejected)
        forged.update({
            "gate_reason": "selected", "selected_position": 1,
            "selected_legal_index": 1,
        })
        with self.assertRaisesRegex(campaign.EvidenceError, "gate drift"):
            campaign._verify_composed_decision(
                forged, (1, True), (1, True), [0, 1], active=True,
                discard_rejected=True,
            )


class TerminalDispositionTests(unittest.TestCase):
    """Exercise the authoritative retain-baseline branches from real paths."""

    def _paths(self, directory: Path, passed: bool):
        (directory / "bindings").mkdir(exist_ok=True)
        execution = directory / "bindings/execution.json"
        execution.write_bytes(campaign.canonical_json({"subject": {
            "maintained_actor": "baseline"}}))
        select_evidence = {
            "raw_verified": True, "stage": "SELECT",
            "execution_sha256": campaign.sha256(execution), "evaluation_sha256": "2" * 64,
            "evaluation_header_sha256": "3" * 64, "allocation_sha256": "4" * 64,
            "calibration_sha256": "5" * 64, "policy_cost_sha256": "6" * 64,
            "policy_cost_content_fingerprint": "7" * 16,
            "selection_sha256": None, "actor_manifest_sha256": None,
        }
        selected = selection.seal_result({
            "schema": "lc-policy-cost-select-result-v1", "stage": "SELECT",
            "selected": {"id": "floor-0.01_ply-00", "policy_floor": 0.01,
                         "ply_lo": 0},
            "campaign_discovery_binding": {"required": True, "validated": True},
            "selection_rule": {"test_evidence_used": False},
            "campaign_evidence_binding": select_evidence,
        })
        selection_path = directory / "select-result.json"
        selection_path.write_bytes(campaign.canonical_json(selected))
        artifact = directory / "policy-cost.lcpc"
        artifact.write_bytes(b"reconstructed terminal artifact\n")
        actors = {
            "schema": "lc-policy-cost-v1-actor-manifest-v1",
            "legacy_validation_relaxed": False, "results": None,
            "selected_configuration": selected["selected"],
            "selection_sha256": campaign.sha256(selection_path),
            "campaign_evidence_binding": select_evidence,
            "train_evidence_binding": {
                "schema": "lc-policy-cost-v1-train-evidence-binding-v1", "stage": "TRAIN",
                "raw_verified": True, "execution_sha256": campaign.sha256(execution),
                "evaluation_sha256": "8" * 64, "evaluation_header_sha256": "9" * 64,
                "allocation_sha256": "a" * 64, "train_input_sha256": "b" * 64,
            },
            "maintained_actor": "baseline", "candidate_actor": "candidate:policy-cost.lcpc",
            "policy_cost_artifact": {"path": "policy-cost.lcpc",
                "sha256": campaign.sha256(artifact), "size": artifact.stat().st_size,
                "content_fingerprint": "0" * 16},
            "preselect_policy_cost_artifact": {
                "sha256": select_evidence["policy_cost_sha256"],
                "content_fingerprint": select_evidence["policy_cost_content_fingerprint"],
            },
        }
        actors_path = directory / "actor-manifest.json"
        actors_path.write_bytes(campaign.canonical_json(actors))
        tested = selection.seal_result({
            "schema": "lc-policy-cost-test-result-v1", "stage": "TEST",
            "selection_payload_sha256": selected["canonical_payload_sha256"],
            "selected": selected["selected"], "passed": passed,
            "criteria": {},
            "campaign_discovery_binding": {"required": True, "validated": True},
            "campaign_evidence_binding": {
                **select_evidence, "stage": "TEST",
                "policy_cost_sha256": campaign.sha256(artifact),
                "policy_cost_content_fingerprint": "0" * 16,
                "selection_sha256": campaign.sha256(selection_path),
                "actor_manifest_sha256": campaign.sha256(actors_path),
            },
        })
        test_path = directory / "test-result.json"
        test_path.write_bytes(campaign.canonical_json(tested))
        return execution, selection_path, test_path, actors_path

    def test_terminal_dispositions_from_reconstructed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            execution, selected, test, actors = self._paths(root, False)
            patches = (
                mock.patch.object(campaign, "parse_maintained_actor",
                                  return_value={"root_path": "r", "continuation_path": "r", "objective": 0}),
                mock.patch.object(campaign, "policy_cost_actor", return_value="candidate:policy-cost.lcpc"),
                mock.patch.object(campaign, "_verify_terminal_efficacy_evidence"),
            )
            with patches[0], patches[1], patches[2]:
                failed_test = campaign.terminal_result(
                    execution=execution, selection=selected, test=test, actors=actors,
                    safety=None, final=None, evidence_root=root,
                    output=root / "policy-cost-v1-result.json",
                    retained_baseline_reason=
                        "TEST failed; safety and final not run.")
            self.assertFalse(failed_test["promotion_gate_passed"])
            self.assertEqual(failed_test["maintained_actor"], "baseline")
            self.assertEqual(
                failed_test["retained_baseline_reason"],
                "TEST failed; safety and final not run.",
            )

            execution, selected, test, actors = self._paths(root, True)
            (root / "safety-gate.json").write_bytes(campaign.canonical_json({}))
            (root / "final-gate.json").write_bytes(campaign.canonical_json({}))
            with patches[0], patches[1], patches[2], mock.patch.object(
                    campaign, "_panel_gate", return_value=(False, {"gate": False})):
                safety_failed = campaign.terminal_result(
                    execution=execution, selection=selected, test=test, actors=actors,
                    safety=root / "safety-gate.json", final=None, evidence_root=root,
                    output=root / "policy-cost-v1-result.json")
            self.assertFalse(safety_failed["promotion_gate_passed"])

            with patches[0], patches[1], patches[2]:
                incomplete_safety = campaign.terminal_result(
                    execution=execution, selection=selected, test=test, actors=actors,
                    safety=None, final=None, evidence_root=root,
                    output=root / "policy-cost-v1-result.json",
                    retained_baseline_reason="safety matrix incomplete")
            self.assertFalse(incomplete_safety["promotion_gate_passed"])
            self.assertEqual(incomplete_safety["retained_baseline_reason"],
                             "safety matrix incomplete")

            with patches[0], patches[1], patches[2], mock.patch.object(
                    campaign, "_panel_gate", return_value=(True, {"gate": True})):
                incomplete_final = campaign.terminal_result(
                    execution=execution, selection=selected, test=test, actors=actors,
                    safety=root / "safety-gate.json", final=None, evidence_root=root,
                    output=root / "policy-cost-v1-result.json",
                    retained_baseline_reason="final matrix incomplete")
            self.assertFalse(incomplete_final["promotion_gate_passed"])

            with patches[0], patches[1], patches[2], mock.patch.object(
                    campaign, "_panel_gate", return_value=(True, {"gate": True})):
                promoted = campaign.terminal_result(
                    execution=execution, selection=selected, test=test, actors=actors,
                    safety=root / "safety-gate.json", final=root / "final-gate.json",
                    evidence_root=root, output=root / "policy-cost-v1-result.json")
            self.assertTrue(promoted["promotion_gate_passed"])
        self.assertEqual(promoted["maintained_actor"], "candidate:policy-cost.lcpc")

    def test_terminal_unconditionally_propagates_raw_efficacy_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            execution, selected, test, actors = self._paths(root, False)
            with mock.patch.object(
                campaign, "parse_maintained_actor",
                return_value={"root_path": "r", "continuation_path": "r",
                              "objective": 0},
            ), mock.patch.object(
                campaign, "policy_cost_actor",
                return_value="candidate:policy-cost.lcpc",
            ), mock.patch.object(
                campaign, "_verify_terminal_efficacy_evidence",
                side_effect=campaign.EvidenceError(
                    "TRAIN raw evaluation shard binding drift"
                ),
            ) as verify:
                with self.assertRaisesRegex(
                        campaign.EvidenceError, "raw evaluation shard"):
                    campaign.terminal_result(
                        execution=execution, selection=selected, test=test,
                        actors=actors, safety=None, final=None,
                        evidence_root=root,
                        output=root / "policy-cost-v1-result.json",
                        retained_baseline_reason=
                            "TEST failed; safety and final not run.",
                    )
            verify.assert_called_once()


class RawEfficacyManifestTests(unittest.TestCase):
    def test_fixed_manifest_requires_every_raw_hash_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            raw_root = root / "evidence/train/raw"
            raw_root.mkdir(parents=True)
            files = []
            digest = hashlib.sha256(b"slice\n").hexdigest()
            for index in range(216):
                path = raw_root / f"{index}.jsonl"
                path.write_bytes(b"slice\n")
                (raw_root / f"{index}.jsonl.sha256").write_text(
                    digest + "\n", encoding="ascii"
                )
                files.append({"path": path.name, "sha256": digest,
                              "size": path.stat().st_size})
            manifest = {
                "schema": "lc-policy-cost-v1-raw-evaluation-manifest-v1",
                "stage": "TRAIN", "files": files,
            }
            manifest_path = root / "evidence/train/raw-shards.json"
            manifest_path.write_bytes(campaign.canonical_json(manifest))
            self.assertEqual(
                len(campaign._validate_raw_evaluation_manifest(root, "TRAIN")),
                216,
            )
            sidecar = raw_root / "17.jsonl.sha256"
            sidecar.write_text("0" * 64 + "\n", encoding="ascii")
            with self.assertRaisesRegex(campaign.EvidenceError, "binding"):
                campaign._validate_raw_evaluation_manifest(root, "TRAIN")
            sidecar.write_text(digest + "\n", encoding="ascii")
            (raw_root / "17.jsonl").unlink()
            with self.assertRaisesRegex(campaign.EvidenceError, "binding"):
                campaign._validate_raw_evaluation_manifest(root, "TRAIN")


if __name__ == "__main__":
    unittest.main()
