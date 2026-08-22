"""Synthetic contracts for strict continuation-v2 evidence selection."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import select_continuation_v2 as selector


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(pair_values: list[float]) -> tuple[float, float]:
    result = selector._sample_metric(pair_values)
    return float(result["estimate"]), float(result["pair_clustered_se"])


class SyntheticCampaign:
    def __init__(self, root: Path, pair_count: int = 12) -> None:
        self.root = root
        self.pair_count = pair_count
        self.plan = root / "locked-plan.json"
        self.arena = root / "continuation_arena"
        self.model = root / "champion.bin"
        self.plan.write_bytes(b"locked continuation v2 plan\n")
        self.arena.write_bytes(b"exact evaluator elf\n")
        self.model.write_bytes(b"exact checkpoint\n")
        self.cells = [
            {"cell": "o0-shared", "cell_order": 0,
             "objective": 0, "role_mapping": "shared"},
            {"cell": "o0-independent", "cell_order": 1,
             "objective": 0, "role_mapping": "independent"},
            {"cell": "o2-shared", "cell_order": 2,
             "objective": 2, "role_mapping": "shared"},
            {"cell": "o2-independent", "cell_order": 3,
             "objective": 2, "role_mapping": "independent"},
        ]
        self.replicates = ["r1", "r2", "r3"]

    def common(self, stage: str) -> dict:
        return {
            "schema": 1,
            "stage": stage,
            "plan_path": self.plan.name,
            "plan_id": sha(self.plan),
            "arena_path": self.arena.name,
            "arena_id": sha(self.arena),
            "root_model_id": sha(self.model),
            "baseline_model_id": sha(self.model),
            "root_checkpoint": self.model.name,
            "baseline_checkpoint": self.model.name,
            # Synthetic contract rows deliberately stay in the namespace that
            # the locked plan burns before any real campaign generation.
            "seed": "202608259701" if stage == "screen-a" else "202608259702",
            "pair_start": "0",
            "pair_count": self.pair_count,
            "target_round": "cycle_0_1_2",
            "cells": self.cells,
            "replicates": self.replicates,
        }

    def pair(self, index: int, seed: int, quality: int) -> dict:
        round_index = index % 3
        block = index // 3
        noise = -2 if block % 2 == 0 else 2
        margin = quality + noise
        root_player = round_index & 1
        mappings = selector._expected_mappings(
            seed, index, root_player, "independent"
        )
        seats = [root_player, root_player ^ 1]
        scores = []
        for seat in seats:
            leg_scores = [0, 0]
            leg_scores[seat] = margin
            scores.append(leg_scores)
        final_margin = [margin, margin]
        result = [(value > 0) - (value < 0) for value in final_margin]
        hybrid = [0.05 * value + 50.0 * sign
                  for value, sign in zip(final_margin, result)]
        return {
            "record": "pair",
            "index": str(index),
            "round": round_index,
            "root_player": root_player,
            "admitted": 2,
            "picked": (
                1 + (index // 2) % 1 if index & 1 else 0
            ),
            "root_move": index,
            "cum_before": [0, 0],
            "cumulative_before": [0, 0],
            "player_mapping": list(mappings),
            "root_role_mapping": mappings[root_player],
            "opponent_role_mapping": mappings[root_player ^ 1],
            "candidate_seat": seats,
            "score_by_seat": scores,
            "candidate_margin": [margin, margin],
            "candidate_round_margin": [margin, margin],
            "candidate_objective_target": (
                hybrid if round_index == 2 else [margin, margin]
            ),
            "candidate_final_match_margin": (
                final_margin if round_index == 2 else None
            ),
            "candidate_final_match_result": (
                result if round_index == 2 else None
            ),
            "candidate_hybrid_target": (
                hybrid if round_index == 2 else None
            ),
            "tail_plies": [20, 20],
            "capped": [0, 0],
            "exact_moves": [1, 1],
            "cap_forces": [0, 0],
            "cycle_forces": [0, 0],
        }

    def summary(self, rows: list[dict]) -> dict:
        rounds = []
        for round_index in range(3):
            selected = [row for row in rows if row["round"] == round_index]
            margin_values = [sum(row["candidate_round_margin"])
                             for row in selected]
            objective_values = [sum(row["candidate_objective_target"])
                                for row in selected]
            margin_mean, margin_se = metric(margin_values)
            objective_mean, objective_se = metric(objective_values)
            item = {
                "round": round_index,
                "pairs": len(selected),
                "selection_semantics": (
                    "final_match_hybrid" if round_index == 2
                    else "round_margin"
                ),
                "round_margin_per_leg": margin_mean,
                "round_margin_pair_clustered_se": margin_se,
                "configured_objective_per_leg": objective_mean,
                "configured_objective_pair_clustered_se": objective_se,
            }
            if round_index == 2:
                final_values = [sum(row["candidate_final_match_margin"])
                                for row in selected]
                score_values = [
                    sum(selector._result_score(value)
                        for value in row["candidate_final_match_margin"])
                    for row in selected
                ]
                final_mean, final_se = metric(final_values)
                score_mean, score_se = metric(score_values)
                results = [value for row in selected
                           for value in row["candidate_final_match_result"]]
                item.update({
                    "final_match_margin_per_leg": final_mean,
                    "final_match_margin_pair_clustered_se": final_se,
                    "match_score": score_mean,
                    "match_score_pair_clustered_se": score_se,
                    "match_wins": sum(value > 0 for value in results),
                    "match_losses": sum(value < 0 for value in results),
                    "match_draws": sum(value == 0 for value in results),
                })
            rounds.append(item)
        return {
            "record": "summary",
            "continuation_objective": 2,
            "configured_objective_aggregate_comparable": False,
            "configured_objective_per_leg": None,
            "configured_objective_pair_clustered_se": None,
            "rounds": rounds,
        }

    def raw(self, manifest: dict, entry: dict, name: str, quality: int) -> str:
        path = self.root / name
        seed = int(manifest["seed"])
        rows = [self.pair(i, seed, quality) for i in range(self.pair_count)]
        normalized = dict(entry)
        normalized["objective"] = 2
        normalized["role_mapping"] = "independent"
        meta = {
            "record": "meta",
            "schema": 2,
            "evidence_scope": "candidate_screen_only_not_promotion",
            "seed": manifest["seed"],
            "pair_start": manifest["pair_start"],
            "pair_count": self.pair_count,
            "target_round": "cycle_0_1_2",
            "continuation_objective": 2,
            "round_0_1_semantics": "round_margin",
            "round_2_mode_2_semantics":
                "0.05*final_match_margin+50*signed_match_result",
            "role_mapping_mode": "independent",
            "root_checkpoint": manifest["root_checkpoint"],
            "candidate_checkpoint": entry["candidate_checkpoint"],
            "baseline_checkpoint": manifest["baseline_checkpoint"],
            "root_ply": 14,
            "root_symmetries": 20,
            "root_width": 5,
            "root_floor": 0.02,
            "root_min": 1,
            "root_mix":
                "alternating_absolute_index_baseline_nonbaseline_with_singleton_fallback",
            "world_model": "uniform_mover_information_set",
            "continuation_policy":
                "greedy_fixed_player_mapping_affine20",
            "late_cycle":
                "production_semantic_information_tracker_deck_le_3",
            "pairing":
                "identical_post_root_world_controller_seat_swap",
            "provenance": selector._provenance(normalized, manifest),
        }
        records = [meta, *rows, self.summary(rows),
                   {"record": "complete", "pairs": self.pair_count}]
        path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n"
                    for record in records),
            encoding="utf-8",
        )
        return path.name

    def screen_a_manifest(self) -> tuple[Path, dict]:
        manifest = self.common("screen-a")
        manifest["checkpoints"] = list(selector.CHECKPOINTS)
        evidence = []
        base_name = None
        for cell in self.cells:
            for replicate in self.replicates:
                for checkpoint_order, checkpoint in enumerate(
                    selector.CHECKPOINTS
                ):
                    entry = {
                        "raw": "",
                        "raw_id": "",
                        "cell": cell["cell"],
                        "replicate": replicate,
                        "checkpoint": checkpoint,
                        "candidate_checkpoint": self.model.name,
                        "candidate_artifact": self.model.name,
                        "candidate_model_id": sha(self.model),
                    }
                    if checkpoint == "base":
                        if base_name is None:
                            base_name = self.raw(
                                manifest, entry, "screen-a-base.jsonl", 0
                            )
                        entry["raw"] = base_name
                    else:
                        # warm2 and full2 deliberately tie for first;
                        # the earlier locked checkpoint must win.
                        quality = 4 if checkpoint_order in (1, 2) else 1
                        filename = (
                            f"a-{cell['cell']}-{replicate}-{checkpoint}.jsonl"
                        )
                        entry["raw"] = self.raw(
                            manifest, entry, filename, quality
                        )
                    entry["raw_id"] = sha(self.root / entry["raw"])
                    evidence.append(entry)
        manifest["evidence"] = evidence
        path = self.root / "screen-a-manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest

    def screen_b_manifest(
        self, screen_a_path: Path, qualities: dict[str, int] | None = None
    ) -> tuple[Path, dict]:
        qualities = qualities or {
            "o0-shared": 4,
            "o0-independent": 5,
            "o2-shared": 7,
            "o2-independent": 7,
        }
        screen_a = json.loads(screen_a_path.read_text())
        selected = {
            (item["cell"], item["replicate"]): item["candidate_model_id"]
            for item in screen_a["selections"]
        }
        manifest = self.common("screen-b")
        manifest.update({
            "confidence_z": 1.645,
            "screen_a_result": screen_a_path.name,
            "screen_a_result_id": sha(screen_a_path),
        })
        evidence = []
        for cell in self.cells:
            cell_name = cell["cell"]
            components = [selected[(cell_name, replicate)]
                          for replicate in self.replicates]
            for variant in (*self.replicates, "soup"):
                entry = {
                    "raw": "",
                    "raw_id": "",
                    "cell": cell_name,
                    "variant": variant,
                    "candidate_checkpoint": self.model.name,
                    "candidate_artifact": self.model.name,
                    "candidate_model_id": sha(self.model),
                }
                if variant == "soup":
                    entry["components"] = components
                entry["raw"] = self.raw(
                    manifest, entry, f"b-{cell_name}-{variant}.jsonl",
                    qualities[cell_name],
                )
                entry["raw_id"] = sha(self.root / entry["raw"])
                evidence.append(entry)
        manifest["evidence"] = evidence
        path = self.root / "screen-b-manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest


class ContinuationV2SelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.campaign = SyntheticCampaign(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_screen_a(self) -> tuple[Path, dict]:
        manifest_path, _ = self.campaign.screen_a_manifest()
        output = self.root / "screen-a-result.json"
        self.assertEqual(selector.main([
            "screen-a", "--manifest", str(manifest_path),
            "--artifact-root", str(self.root), "--output", str(output),
        ]), 0)
        return output, json.loads(output.read_text())

    def test_screen_a_recomputes_and_uses_locked_checkpoint_tie_break(self) -> None:
        output, result = self.run_screen_a()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["selections"]), 12)
        self.assertEqual(
            {item["checkpoint"] for item in result["selections"]},
            {"warm2"},
        )
        self.assertTrue(all(
            item["checkpoint_order"] == 1 for item in result["selections"]
        ))
        self.assertEqual(
            {item["candidate_artifact"] for item in result["selections"]},
            {self.campaign.model.name},
        )
        validation = self.root / "screen-a-validation.json"
        self.assertEqual(selector.main([
            "validate", "--manifest", str(self.root / "screen-a-manifest.json"),
            "--artifact-root", str(self.root), "--output", str(validation),
        ]), 0)
        validation_result = json.loads(validation.read_text())
        self.assertTrue(all(
            item["candidate_artifact"] == self.campaign.model.name
            for item in validation_result["evidence"]
        ))
        second = self.root / "screen-a-result-copy.json"
        manifest = self.root / "screen-a-manifest.json"
        self.assertEqual(selector.main([
            "screen-a", "--manifest", str(manifest),
            "--artifact-root", str(self.root), "--output", str(second),
        ]), 0)
        self.assertEqual(output.read_bytes(), second.read_bytes())

    def test_screen_b_uses_soup_and_full_locked_tie_break(self) -> None:
        screen_a_path, _ = self.run_screen_a()
        manifest_path, _ = self.campaign.screen_b_manifest(screen_a_path)
        output = self.root / "screen-b-result.json"
        self.assertEqual(selector.main([
            "screen-b", "--manifest", str(manifest_path),
            "--artifact-root", str(self.root), "--output", str(output),
        ]), 0)
        result = json.loads(output.read_text())
        self.assertEqual(result["status"], "selected")
        # The last two cells have identical evidence, so fixed cell order wins.
        self.assertEqual(result["selected"]["cell"], "o2-shared")
        selected_cell = next(
            cell for cell in result["cells"]
            if cell["cell"] == result["selected"]["cell"]
        )
        self.assertEqual(
            result["selected"]["candidate_model_id"],
            selected_cell["soup"]["candidate_model_id"],
        )
        self.assertEqual(
            result["selected"]["candidate_artifact"],
            selected_cell["soup"]["candidate_artifact"],
        )
        self.assertEqual(selected_cell["replicate_positive_point_passes"], 3)
        self.assertTrue(selected_cell["soup_each_early_round_nonharm_upper_pass"])

    def test_screen_b_requires_two_positive_replicates_and_nontharm_third(self) -> None:
        screen_a_path, _ = self.run_screen_a()
        manifest_path, manifest = self.campaign.screen_b_manifest(screen_a_path)
        # Make every cell's r2/r3 evidence point-harmful.  Raw evidence remains
        # structurally valid, so the selector must return a valid no-winner
        # decision rather than confusing statistical failure with corruption.
        for entry in manifest["evidence"]:
            if entry["variant"] in ("r2", "r3"):
                entry["raw"] = self.campaign.raw(
                    manifest, entry,
                    f"harm-{entry['cell']}-{entry['variant']}.jsonl", -7,
                )
                entry["raw_id"] = sha(self.root / entry["raw"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output = self.root / "screen-b-none.json"
        self.assertEqual(selector.main([
            "screen-b", "--manifest", str(manifest_path),
            "--artifact-root", str(self.root), "--output", str(output),
        ]), 0)
        result = json.loads(output.read_text())
        self.assertEqual(result["status"], "no-eligible-cell")
        self.assertIsNone(result["selected"])

    def test_fail_closed_on_raw_corruption_and_model_drift(self) -> None:
        manifest_path, manifest = self.campaign.screen_a_manifest()
        raw_path = self.root / manifest["evidence"][1]["raw"]
        original = raw_path.read_text()
        mutations = {
            "duplicate": original.replace(
                '"round":0', '"round":0,"round":0', 1
            ),
            "nan": original.replace('"root_floor":0.02',
                                    '"root_floor":NaN', 1),
            "gap": original.replace('"index":"0"', '"index":"1"', 1),
            "provenance": original.replace(
                "continuation-v2|", "continuation-v2-drift|", 1
            ),
            "footer": original.replace(
                f'"pairs":{self.campaign.pair_count}}}',
                f'"pairs":{self.campaign.pair_count - 1}}}', 1,
            ),
        }
        for name, text in mutations.items():
            with self.subTest(name=name):
                raw_path.write_text(text, encoding="utf-8")
                for entry in manifest["evidence"]:
                    if entry["raw"] == raw_path.name:
                        entry["raw_id"] = sha(raw_path)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(selector.main([
                    "validate", "--manifest", str(manifest_path),
                    "--artifact-root", str(self.root),
                ]), 2)
                raw_path.write_text(original, encoding="utf-8")
                for entry in manifest["evidence"]:
                    if entry["raw"] == raw_path.name:
                        entry["raw_id"] = sha(raw_path)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.campaign.model.write_bytes(b"drifted checkpoint\n")
        self.assertEqual(selector.main([
            "validate", "--manifest", str(manifest_path),
            "--artifact-root", str(self.root),
        ]), 2)

    def test_fail_closed_on_mixed_evaluator_semantics(self) -> None:
        manifest_path, manifest = self.campaign.screen_a_manifest()
        raw_path = self.root / manifest["evidence"][1]["raw"]
        records = raw_path.read_text().splitlines()
        meta = json.loads(records[0])
        meta["continuation_objective"] = 0
        records[0] = json.dumps(meta, separators=(",", ":"))
        raw_path.write_text("\n".join(records) + "\n", encoding="utf-8")
        for entry in manifest["evidence"]:
            if entry["raw"] == raw_path.name:
                entry["raw_id"] = sha(raw_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(selector.main([
            "validate", "--manifest", str(manifest_path),
            "--artifact-root", str(self.root),
        ]), 2)

    def test_fail_closed_on_schedule_corruption(self) -> None:
        manifest_path, manifest = self.campaign.screen_a_manifest()
        raw_path = self.root / manifest["evidence"][1]["raw"]
        records = raw_path.read_text().splitlines()
        pair = json.loads(records[1])
        pair["root_player"] ^= 1
        records[1] = json.dumps(pair, separators=(",", ":"))
        raw_path.write_text("\n".join(records) + "\n", encoding="utf-8")
        for entry in manifest["evidence"]:
            if entry["raw"] == raw_path.name:
                entry["raw_id"] = sha(raw_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(selector.main([
            "validate", "--manifest", str(manifest_path),
            "--artifact-root", str(self.root),
        ]), 2)

    def test_fail_closed_on_traversal_and_parent_symlinks(self) -> None:
        manifest_path, original = self.campaign.screen_a_manifest()
        parent_link = self.root / "linked-parent"
        parent_link.symlink_to(self.root, target_is_directory=True)
        cases = []

        traversal = copy.deepcopy(original)
        traversal["evidence"][1]["candidate_artifact"] = "../champion.bin"
        cases.append(("traversal", traversal))

        plan_link = copy.deepcopy(original)
        plan_link["plan_path"] = f"{parent_link.name}/{self.campaign.plan.name}"
        cases.append(("plan-parent-symlink", plan_link))

        arena_link = copy.deepcopy(original)
        arena_link["arena_path"] = f"{parent_link.name}/{self.campaign.arena.name}"
        cases.append(("arena-parent-symlink", arena_link))

        checkpoint_link = copy.deepcopy(original)
        checkpoint_link["root_checkpoint"] = (
            f"{parent_link.name}/{self.campaign.model.name}"
        )
        cases.append(("checkpoint-parent-symlink", checkpoint_link))

        raw_link = copy.deepcopy(original)
        raw_link["evidence"][1]["raw"] = (
            f"{parent_link.name}/{original['evidence'][1]['raw']}"
        )
        cases.append(("raw-parent-symlink", raw_link))

        for name, manifest in cases:
            with self.subTest(name=name):
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(selector.main([
                    "validate", "--manifest", str(manifest_path),
                    "--artifact-root", str(self.root),
                ]), 2)

    def test_fail_closed_on_screen_a_link_parent_symlink(self) -> None:
        screen_a_path, _ = self.run_screen_a()
        manifest_path, manifest = self.campaign.screen_b_manifest(screen_a_path)
        parent_link = self.root / "result-parent-link"
        parent_link.symlink_to(self.root, target_is_directory=True)
        manifest["screen_a_result"] = (
            f"{parent_link.name}/{screen_a_path.name}"
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(selector.main([
            "validate", "--manifest", str(manifest_path),
            "--artifact-root", str(self.root),
        ]), 2)


if __name__ == "__main__":
    unittest.main()
