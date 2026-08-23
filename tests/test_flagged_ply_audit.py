from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/flagged_ply_audit.py"
SPEC = importlib.util.spec_from_file_location("flagged_ply_audit", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

EXECUTION_MODULE_PATH = ROOT / "tools/flagged_ply_execution.py"
EXECUTION_SPEC = importlib.util.spec_from_file_location(
    "flagged_ply_execution", EXECUTION_MODULE_PATH
)
assert EXECUTION_SPEC and EXECUTION_SPEC.loader
execution = importlib.util.module_from_spec(EXECUTION_SPEC)
EXECUTION_SPEC.loader.exec_module(execution)

ACTOR = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1"
)

EXPECTED = {
    2214615196: {3, 4, 8, 10, 12, 13, 16, 20},
    5726968372613385: {4, 7, 14, 15, 17, 25, 32},
    725402798: {
        1, 2, 3, 7, 14, 21, 22, 23, 25, 29,
        30, 31, 36, 40, 46, 47, 55, 62, 63, 64,
    },
    95647345759839: {44},
}


class CorpusTests(unittest.TestCase):
    def test_manifest_is_exact_literal_inventory_and_hashes_verify(self) -> None:
        manifest, digest = audit.load_manifest(
            ROOT / "data/user_reviewed_plies.json"
        )
        self.assertEqual(len(digest), 64)
        actual: dict[int, set[int]] = {}
        for case in manifest["cases"]:
            actual.setdefault(int(case["seed"]), set()).add(int(case["ply"]))
        self.assertEqual(actual, EXPECTED)
        self.assertEqual(
            {case["id"] for case in manifest["cases"] if case["kind"] == "belief"},
            {"ui221-p13", "showcase572-p4"},
        )
        p14 = next(
            case for case in manifest["cases"]
            if case["id"] == "showcase572-p14"
        )
        self.assertEqual(p14["preferred"], ["G7 p deck", "B3 d deck"])
        self.assertEqual(p14["criticized"], ["R4 d deck"])

    def test_failure_classification_separates_policy_omission_and_rollout(self) -> None:
        case = {"preferred": ["B10 p deck"], "criticized": ["Y10 p deck"]}
        self.assertEqual(
            audit.classify_move(case, "Y10 p deck", {"Y10 p deck"}),
            "preferred_move_missing_from_top_policy_union",
        )
        self.assertEqual(
            audit.classify_move(
                case, "Y10 p deck", {"Y10 p deck", "B10 p deck"}
            ),
            "flagged_move_selected_by_rollout_panel",
        )
        self.assertEqual(
            audit.classify_move(
                case, "B10 p deck", {"Y10 p deck", "B10 p deck"}
            ),
            "review_aligned",
        )

    def test_match_value_table_is_hashed_as_actor_provenance(self) -> None:
        tail = ["0"] * 41 + ["data/champion.bin"]
        provenance = audit.actor_provenance(
            ":".join(["rolloutu", "data/champion.bin", *tail])
        )
        self.assertEqual(
            provenance["match_value_table"]["sha256"],
            provenance["checkpoints"][0]["sha256"],
        )

        ranker_provenance = audit.actor_provenance(
            ":".join([
                "rolloutu4",
                "data/champion.bin",
                "data/champion.bin",
                "data/champion.bin",
                *tail,
            ])
        )
        self.assertEqual(len(ranker_provenance["checkpoints"]), 3)
        self.assertEqual(
            ranker_provenance["match_value_table"]["sha256"],
            ranker_provenance["checkpoints"][2]["sha256"],
        )

    def test_merge_rejects_mixed_source_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable = {
                "manifest_sha256": "a" * 64,
                "reference": {}, "candidate": {}, "decision_worlds": 2,
                "belief_alpha": 1.15, "history_worlds": 1,
                "base_seed": 1, "shard_count": 2,
                "candidate_rule": "top three", "world_model": "uniform",
                "selection": "all", "execution_sha256": None,
            }
            inputs = []
            for shard, commit in enumerate(("1" * 40, "2" * 40)):
                path = root / f"shard-{shard}.json"
                path.write_text(json.dumps({
                    "schema": "lc-flagged-ply-audit-v1",
                    "provenance": {
                        **stable, "source_commit": commit,
                        "shard_index": shard,
                    },
                    "errors": [], "cases": [],
                }))
                inputs.append(path)
            completed = subprocess.run(
                [
                    "python3", "tools/merge_flagged_ply_audit.py",
                    *(str(path) for path in inputs),
                    "--allow-partial", "--output", str(root / "merged.json"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("source_commit", completed.stderr)

    def test_one_shot_execution_template_is_inert_and_bound_to_manifest(self) -> None:
        template = ROOT / "data/flagged_ply_audit_execution.template.json"
        config = execution.load_execution(template, require_execute=False)
        self.assertFalse(config["execute"])
        self.assertEqual(config["decision_worlds_per_actor_per_case"], 16384)
        self.assertEqual(config["history_worlds"], 20000)
        self.assertEqual(config["shard_count"], 12)
        with self.assertRaises(execution.ExecutionError):
            execution.load_execution(template)
        self.assertFalse(
            (ROOT / "data/flagged_ply_audit_execution.json").exists(),
            "the real one-shot launch addendum must not be created early",
        )

    def test_workflow_has_guarded_addendum_push_not_template_push(self) -> None:
        workflow = (ROOT / ".github/workflows/flagged-ply-audit.yml").read_text()
        self.assertIn("push:", workflow)
        self.assertIn("data/flagged_ply_audit_execution.json", workflow)
        self.assertNotIn(
            "- data/flagged_ply_audit_execution.template.json", workflow
        )
        self.assertIn("--before \"$BEFORE\" --after \"$AFTER\"", workflow)
        self.assertIn("make -j2\n          CFLAGS=", workflow)
        self.assertIn(
            "bin/flagged_ply_probe bin/history_belief data/champion.bin",
            workflow,
        )
        self.assertNotIn("pull_request:", workflow)


@unittest.skipUnless(
    (ROOT / "bin/flagged_ply_probe").is_file()
    and (ROOT / "bin/history_belief").is_file(),
    "build bin/flagged_ply_probe and bin/history_belief for integration tests",
)
class ProbeIntegrationTests(unittest.TestCase):
    def probe(self, state: str, worlds: int) -> dict:
        completed = subprocess.run(
            [
                str(ROOT / "bin/flagged_ply_probe"),
                "-S", str(ROOT / state),
                "-a", ACTOR,
                "-b", ACTOR,
                "-w", str(worlds),
                "-s", "202608231799",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(completed.stdout)

    def test_identical_actors_get_identical_common_panel(self) -> None:
        result = self.probe(
            "data/probes/ui_seed725402798_p36.state", 32
        )
        self.assertLessEqual(result["evaluated_moves"], 5)
        self.assertLess(result["evaluated_moves"], result["legal_moves"])
        self.assertEqual(
            result["actors"][0]["rows"], result["actors"][1]["rows"]
        )
        self.assertEqual(
            result["actors"][0]["panel_selected"],
            result["actors"][1]["panel_selected"],
        )
        self.assertEqual(
            result["actors"][0]["deployed_selected"],
            result["actors"][1]["deployed_selected"],
        )
        self.assertEqual(result["actors"][0]["unfinished_cap_leaves"], 0)
        self.assertEqual(result["actors"][0]["objective_label"], "round_margin")
        self.assertEqual(result["actors"][0]["objective_units"], "round_points")

    def test_probe_distinguishes_final_hybrid_from_round_points(self) -> None:
        fields = ACTOR.split(":")
        # One-network rollout tail field 8 is the selection objective.
        fields[2 + 8] = "2"
        hybrid_actor = ":".join(fields)
        completed = subprocess.run(
            [
                str(ROOT / "bin/flagged_ply_probe"),
                "-S", str(ROOT / "data/probes/g424_p111.state"),
                "-a", ACTOR,
                "-b", hybrid_actor,
                "-w", "2",
                "-s", "202608231798",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        actors = json.loads(completed.stdout)["actors"]
        self.assertEqual(
            (actors[0]["objective_label"], actors[0]["objective_units"]),
            ("round_margin", "round_points"),
        )
        self.assertEqual(
            (actors[1]["objective_label"], actors[1]["objective_units"]),
            ("final_hybrid", "hybrid_match_utility_points"),
        )

    def test_decision_report_renders_objective_label_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            markdown = Path(directory) / "audit.md"
            subprocess.run(
                [
                    "python3", "tools/flagged_ply_audit.py",
                    "--reference", ACTOR,
                    "--candidate", ACTOR,
                    "--case", "ui725-p1",
                    "--worlds", "2",
                    "--output", str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(
                [
                    "python3", "tools/render_flagged_ply_audit.py",
                    str(output), "--output", str(markdown),
                ],
                cwd=ROOT,
                check=True,
            )
            report = markdown.read_text()
            self.assertIn("Panel objectives: reference", report)
            self.assertIn("`round_margin` in `round_points`", report)
            self.assertIn("only when both labels and units match", report)

    def test_deck_two_uses_complete_ordered_hidden_support(self) -> None:
        result = self.probe(
            "data/probes/ui_seed95647345759839_p43.state", 1000
        )
        for actor in result["actors"]:
            self.assertEqual(actor["hidden_support"], 90)
            self.assertEqual(actor["worlds"], 90)
            self.assertTrue(actor["exact_hidden_support"])
            self.assertEqual(actor["unfinished_cap_leaves"], 0)

    def test_deck_one_uses_complete_hidden_support(self) -> None:
        result = self.probe(
            "data/probes/ui_seed95647345759839_p44.state", 1000
        )
        admitted = {row["move"] for row in result["candidates"]}
        self.assertIn("G8 p deck", admitted)
        self.assertIn("G8 p W", admitted)
        self.assertIn("complete semantic policy moves", result["candidate_rule"])
        for actor in result["actors"]:
            self.assertEqual(actor["hidden_support"], 9)
            self.assertEqual(actor["worlds"], 9)
            self.assertTrue(actor["exact_hidden_support"])
            self.assertEqual(actor["unfinished_cap_leaves"], 0)
            belief = actor["belief"]
            cards = belief["cards"]
            self.assertEqual(
                len({row["card"] for row in cards}), len(cards),
                "indistinguishable wager copies must share one semantic row",
            )
            for row in cards:
                self.assertAlmostEqual(
                    row["head_minus_prior"],
                    row["estimate"] - row["prior"],
                    places=7,
                )
                if row["metric"] == "expected_count":
                    self.assertAlmostEqual(
                        row["prior"],
                        row["unseen_copies"]
                        * belief["unknown_hand"] / belief["unknown_pool"],
                        places=7,
                    )

    def test_runner_reports_belief_focus_without_large_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            subprocess.run(
                [
                    "python3", "tools/flagged_ply_audit.py",
                    "--reference", ACTOR,
                    "--candidate", ACTOR,
                    "--case", "ui221-p13",
                    "--worlds", "32",
                    "--history-worlds", "50",
                    "--output", str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            result = json.loads(output.read_text())
            case = result["cases"][0]
            self.assertEqual(case["kind"], "belief")
            self.assertTrue(case["probe"]["belief_only"])
            self.assertEqual(case["probe"]["evaluated_moves"], 0)
            self.assertEqual(case["probe"]["candidates"], [])
            self.assertFalse(case["probe"]["actors"][0]["action_panel"])
            self.assertNotIn("rows", case["probe"]["actors"][0])
            history = case["history_aware_belief"]
            self.assertEqual(
                history["provenance"]["view_sha256"],
                "a9ef8595235d5b1de3e168c13cbe57fe4943fd703cce665d1acee68b46944725",
            )
            self.assertEqual(
                history["status"], "insufficient_accepted_support"
            )
            focus = case["classifications"]["candidate"]["focus_cards"]
            self.assertEqual([row["card"] for row in focus], ["Y9"])
            belief = case["probe"]["actors"][1]["belief"]
            self.assertAlmostEqual(
                focus[0]["prior"],
                belief["unknown_hand"] / belief["unknown_pool"],
                places=7,
            )
            self.assertAlmostEqual(
                focus[0]["head_minus_prior"],
                focus[0]["probability"] - focus[0]["prior"],
                places=7,
            )
            markdown = Path(directory) / "audit.md"
            subprocess.run(
                [
                    "python3", "tools/render_flagged_ply_audit.py",
                    str(output), "--output", str(markdown),
                ],
                cwd=ROOT,
                check=True,
            )
            report = markdown.read_text()
            self.assertIn("## ui221-p13", report)
            self.assertIn("snapshot-only", report.lower())
            self.assertIn("insufficient_accepted_support", report)
            self.assertIn("No action or rollout-Q panel was run", report)
            self.assertNotIn("Admitted policy moves", report)
            self.assertNotIn("Panel selection", report)

    def test_history_aware_belief_consumes_frozen_public_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            subprocess.run(
                [
                    "python3", "tools/flagged_ply_audit.py",
                    "--reference", ACTOR,
                    "--candidate", ACTOR,
                    "--case", "showcase572-p4",
                    "--worlds", "2",
                    "--history-worlds", "1000",
                    "--output", str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            case = json.loads(output.read_text())["cases"][0]
            history = case["history_aware_belief"]
            self.assertEqual(history["status"], "ok")
            self.assertGreater(history["accepted"], 0)
            focus = case["classifications"]["history_aware"]["focus_cards"]
            self.assertEqual(
                [row["card"] for row in focus], ["Y4", "Y9", "Y10"]
            )
            self.assertEqual(
                history["provenance"]["checkpoint_sha256"],
                "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417",
            )


if __name__ == "__main__":
    unittest.main()
