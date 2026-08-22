"""Provenance contracts for the one-shot viewer generator."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/make_showcase.py"
SPEC = importlib.util.spec_from_file_location("make_showcase", MODULE_PATH)
assert SPEC and SPEC.loader
showcase = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(showcase)


def late_stub() -> dict:
    horizon = {
        "best_index": -1,
        "value": 0.0,
        "delta_vs_policy": 0.0,
        "nodes": 0,
        "improved_root_nodes": 0,
        "frozen_opponent_nodes": 0,
        "transitions": 0,
        "deviation_evaluations": 0,
        "exact_terminal_leaves": 0,
    }
    return {
        "enabled": False,
        "attempted": False,
        "completed": False,
        "method": "one_sided_bounded_particle_policy_improvement",
        "stable": False,
        "retained_policy": False,
        "override_authorized": False,
        "practical_gate_passed": False,
        "practical_threshold": 1.0,
        "used_to_select": False,
        "selection_reason": "not_attempted",
        "support": 0,
        "candidate_count": 0,
        "horizon2": dict(horizon),
        "horizon4": dict(horizon),
        "candidates": [],
    }


def challenger_late_stub(gate_passed: bool) -> dict:
    late = late_stub()
    late.update({
        "enabled": True,
        "attempted": True,
        "completed": True,
        "stable": True,
        "override_authorized": True,
        "practical_gate_passed": gate_passed,
        "used_to_select": True,
        "selection_reason": "challenger_override",
        "support": 90,
        "candidate_count": 2,
        "candidates": [{
            "card": "Y2", "act": "discard", "draw": "deck",
            "policy_prob": 0.7, "horizon2_q": 0.0, "horizon4_q": 0.0,
            "policy_baseline": True, "horizon2_best": False,
            "horizon4_best": False,
        }, {
            "card": "Y3", "act": "discard", "draw": "deck",
            "policy_prob": 0.3, "horizon2_q": 2.0, "horizon4_q": 2.0,
            "policy_baseline": False, "horizon2_best": True,
            "horizon4_best": True,
        }],
    })
    late["horizon2"]["best_index"] = 1
    late["horizon2"]["value"] = 2.0
    late["horizon2"]["delta_vs_policy"] = 2.0
    late["horizon4"]["best_index"] = 1
    late["horizon4"]["value"] = 2.0
    late["horizon4"]["delta_vs_policy"] = 2.0
    return late


class ShowcaseProvenanceTests(unittest.TestCase):
    ACTOR = "policy:data/c8.bin:0:20"
    TRACKED_AUDIT = (
        "rolloutu:data/champion.bin:2048:5:0.01:0:1:14:0:0:0:0:3.5:2:4:"
        "20:0:0:20:1:0:2048:1:0:0:0:0:0:0:3:1:0:0:2:1:0:3:1:0:0:1"
    )
    CHAMPION_SHA256 = (
        "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"
    )

    def test_default_actor_matches_locked_role_coherent_promotion(self) -> None:
        result = json.loads(
            (ROOT / "data/experiments/role_coherent_result.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(result["status"], "complete_valid_gate_passed")
        self.assertTrue(result["decision"]["promotion_gate_passed"])
        self.assertEqual(showcase.DEFAULT_ACTOR, result["candidate"]["spec"])

    @classmethod
    def analyzer_result(cls):
        move = {
            "card": "Y2",
            "act": "discard",
            "draw": "deck",
            "drawn": "Y3",
        }
        game = {
            "meta": {
                "actor": cls.ACTOR,
                "evaluator": showcase.DEFAULT_EVALUATOR,
                "seed": 1,
                "rounds": 3,
                "generated": "analyze",
                "plies": 1,
                "final": [0, 0],
            },
            "plies": [{
                "n": 1,
                "policy": [{**move, "prob": 1.0}],
                "move": move,
                "actor_decision": {"late_resolver": late_stub()},
                "analysis": {
                    "searched": False,
                    "late_resolver": late_stub(),
                },
                "search": [],
            }],
        }
        return type("Result", (), {"stdout": json.dumps(game)})()

    def run_main(self, *arguments: str):
        argv = ["make_showcase.py", *arguments, "--actor", self.ACTOR]
        with patch.object(sys, "argv", argv), patch.object(
            showcase.subprocess,
            "run",
            return_value=self.analyzer_result(),
        ) as run:
            showcase.main()
        return run

    def test_default_hash_path_comes_from_actor_spec(self) -> None:
        self.assertEqual(
            showcase.actor_model_path(showcase.DEFAULT_ACTOR).resolve(),
            (ROOT / "data/champion.bin").resolve(),
        )

    def test_dual_actor_paths_and_tail_boundary_are_unambiguous(self) -> None:
        root, continuation = showcase.actor_model_paths(
            "rolloutu2:data/champion.bin:data/c8.bin:512:5"
        )
        self.assertEqual(root.resolve(), (ROOT / "data/champion.bin").resolve())
        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertEqual(continuation.resolve(), (ROOT / "data/c8.bin").resolve())
        self.assertEqual(
            showcase.rollout_tail_start(
                "rolloutu2:data/champion.bin:data/c8.bin:512:5".split(":")
            ),
            3,
        )
        with self.assertRaisesRegex(RuntimeError, "continuation checkpoint"):
            showcase.actor_model_paths("rollout2:data/champion.bin")

    def test_three_actor_paths_and_tail_boundary_are_unambiguous(
        self,
    ) -> None:
        root, continuation, veto = showcase.actor_model_paths(
            "rolloutu3:data/champion.bin:data/c8.bin:data/champion.bin:512:5"
        )
        self.assertEqual(root.resolve(), (ROOT / "data/champion.bin").resolve())
        self.assertEqual(continuation.resolve(), (ROOT / "data/c8.bin").resolve())
        self.assertEqual(veto.resolve(), (ROOT / "data/champion.bin").resolve())
        self.assertEqual(
            showcase.rollout_tail_start(
                (
                    "rolloutu3:data/champion.bin:data/c8.bin:"
                    "data/champion.bin:512:5"
                ).split(":")
            ),
            4,
        )
        with self.assertRaisesRegex(RuntimeError, "continuation checkpoint"):
            showcase.actor_model_paths("rollout3:data/champion.bin")
        with self.assertRaisesRegex(RuntimeError, "veto checkpoint"):
            showcase.actor_model_paths(
                "rollout3:data/champion.bin:data/c8.bin"
            )

    def test_non_network_actor_cannot_claim_model_provenance(self) -> None:
        for spec in ("heur", "random", "policy:"):
            with self.subTest(spec=spec), self.assertRaises(RuntimeError):
                showcase.actor_model_path(spec)

    def test_checkpoint_change_during_analysis_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            checkpoint = parent / "model.bin"
            output = parent / "showcase.json"
            checkpoint.write_bytes(b"before")
            output.write_text("keep\n", encoding="utf-8")

            def analyzer_finished(*_args, **_kwargs):
                checkpoint.write_bytes(b"after")
                return self.analyzer_result()

            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--actor", self.ACTOR,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase, "actor_model_path", return_value=checkpoint,
            ), patch.object(
                showcase.subprocess, "run", side_effect=analyzer_finished,
            ), self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                showcase.main()
            self.assertEqual(output.read_text(encoding="utf-8"), "keep\n")

    def test_dual_actor_hashes_protect_both_roles_and_identify_match(self) -> None:
        tail = showcase.DEFAULT_ACTOR.split(":", 2)[2]
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "root.bin"
            continuation = parent / "continuation.bin"
            output = parent / "showcase.json"
            root.write_bytes(b"root checkpoint")
            continuation.write_bytes(b"continuation checkpoint A")
            actor = f"rolloutu2:{root}:{continuation}:{tail}"
            game = json.loads(self.analyzer_result().stdout)
            game["meta"]["actor"] = actor
            result = type("Result", (), {"stdout": json.dumps(game)})()
            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--actor", actor,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", return_value=result,
            ) as run:
                showcase.main()

            saved = json.loads(output.read_text(encoding="utf-8"))
            meta = saved["meta"]
            root_hash = hashlib.sha256(root.read_bytes()).hexdigest()
            continuation_hash = hashlib.sha256(
                continuation.read_bytes()
            ).hexdigest()
            self.assertEqual(meta["model_path"], str(root))
            self.assertEqual(meta["model_sha256"], root_hash)
            self.assertEqual(meta["root_model_path"], str(root))
            self.assertEqual(meta["root_model_sha256"], root_hash)
            self.assertEqual(meta["continuation_model_path"], str(continuation))
            self.assertEqual(
                meta["continuation_model_sha256"], continuation_hash
            )
            self.assertEqual(
                meta["match_id"], f"1-{root_hash}-{continuation_hash}"
            )
            self.assertIn("Root root.bin", meta["actor_label"])
            self.assertIn(
                "continuation continuation.bin", meta["actor_label"]
            )
            self.assertIn("root shortlist", meta["root_model_role"])
            self.assertIn("after", meta["continuation_model_role"])
            self.assertEqual(meta["actor_worlds"], 512)
            self.assertEqual(meta["actor_confirmation_worlds"], 512)
            self.assertEqual(meta["actor_root_width"], 5)
            self.assertEqual(meta["actor_search_from_round_ply"], 14)
            run.assert_called_once()
            self.assertEqual(
                run.call_args.args[0][-2:],
                ["-e", showcase.DEFAULT_EVALUATOR],
            )

            first_identity = meta["match_id"]
            continuation.write_bytes(b"continuation checkpoint B")
            second_hash = hashlib.sha256(continuation.read_bytes()).hexdigest()
            game["meta"]["actor"] = actor
            result = type("Result", (), {"stdout": json.dumps(game)})()
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", return_value=result,
            ):
                showcase.main()
            second_identity = json.loads(output.read_text())["meta"]["match_id"]
            self.assertEqual(second_identity, f"1-{root_hash}-{second_hash}")
            self.assertNotEqual(first_identity, second_identity)

    def test_dual_actor_continuation_change_during_analysis_is_rejected(self) -> None:
        tail = showcase.DEFAULT_ACTOR.split(":", 2)[2]
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "root.bin"
            continuation = parent / "continuation.bin"
            output = parent / "showcase.json"
            root.write_bytes(b"root")
            continuation.write_bytes(b"before")
            output.write_text("keep\n", encoding="utf-8")
            actor = f"rollout2:{root}:{continuation}:{tail}"
            game = json.loads(self.analyzer_result().stdout)
            game["meta"]["actor"] = actor
            result = type("Result", (), {"stdout": json.dumps(game)})()

            def analyzer_finished(*_args, **_kwargs):
                continuation.write_bytes(b"after")
                return result

            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--actor", actor,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", side_effect=analyzer_finished,
            ), self.assertRaisesRegex(
                RuntimeError, "continuation checkpoint changed"
            ):
                showcase.main()
            self.assertEqual(output.read_text(encoding="utf-8"), "keep\n")

    def test_three_actor_hashes_and_protects_every_checkpoint(self) -> None:
        tail = showcase.DEFAULT_ACTOR.split(":", 2)[2]
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "root.bin"
            continuation = parent / "continuation.bin"
            veto = parent / "veto.bin"
            output = parent / "showcase.json"
            root.write_bytes(b"root checkpoint")
            continuation.write_bytes(b"continuation checkpoint")
            veto.write_bytes(b"veto checkpoint")
            actor = f"rolloutu3:{root}:{continuation}:{veto}:{tail}"
            game = json.loads(self.analyzer_result().stdout)
            game["meta"]["actor"] = actor
            result = type("Result", (), {"stdout": json.dumps(game)})()
            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--actor", actor,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", return_value=result,
            ):
                showcase.main()

            meta = json.loads(output.read_text(encoding="utf-8"))["meta"]
            root_hash = hashlib.sha256(root.read_bytes()).hexdigest()
            continuation_hash = hashlib.sha256(
                continuation.read_bytes()
            ).hexdigest()
            veto_hash = hashlib.sha256(veto.read_bytes()).hexdigest()
            self.assertEqual(meta["root_model_path"], str(root))
            self.assertEqual(meta["root_model_sha256"], root_hash)
            self.assertEqual(meta["continuation_model_path"], str(continuation))
            self.assertEqual(
                meta["continuation_model_sha256"], continuation_hash
            )
            self.assertEqual(meta["veto_model_path"], str(veto))
            self.assertEqual(meta["veto_model_sha256"], veto_hash)
            self.assertEqual(
                meta["match_id"],
                f"1-{root_hash}-{continuation_hash}-{veto_hash}",
            )
            self.assertIn("veto veto.bin", meta["actor_label"])
            self.assertIn("conservative veto", meta["veto_model_role"])
            self.assertEqual(meta["actor_worlds"], 512)
            self.assertEqual(meta["actor_root_width"], 5)
            self.assertEqual(meta["actor_search_from_round_ply"], 14)

            output.write_text("keep\n", encoding="utf-8")

            def analyzer_finished(*_args, **_kwargs):
                veto.write_bytes(b"changed veto checkpoint")
                return result

            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", side_effect=analyzer_finished,
            ), self.assertRaisesRegex(
                RuntimeError, "veto checkpoint changed"
            ):
                showcase.main()
            self.assertEqual(output.read_text(encoding="utf-8"), "keep\n")

    def test_dual_actor_cannot_overwrite_continuation_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "root.bin"
            continuation = parent / "continuation.bin"
            root.write_bytes(b"root")
            continuation.write_bytes(b"continuation")
            actor = f"rolloutu2:{root}:{continuation}:512:5"
            argv = [
                "make_showcase.py", "--seed", "1", "--output",
                str(continuation), "--actor", actor,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run",
            ) as run, self.assertRaisesRegex(
                RuntimeError, "continuation checkpoint"
            ):
                showcase.main()
            run.assert_not_called()
            self.assertEqual(continuation.read_bytes(), b"continuation")

    def test_analyzer_provenance_mismatch_is_rejected(self) -> None:
        mutations = {
            "actor": "policy:data/other.bin",
            "evaluator": "rolloutu:data/other.bin",
            "seed": 2,
            "rounds": 2,
            "generated": "other",
            "plies": 2,
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                game = json.loads(self.analyzer_result().stdout)
                game["meta"][field] = value
                result = type("Result", (), {"stdout": json.dumps(game)})()
                output = Path(directory) / "showcase.json"
                argv = [
                    "make_showcase.py", "--seed", "1", "--output", str(output),
                    "--actor", self.ACTOR,
                ]
                with patch.object(sys, "argv", argv), patch.object(
                    showcase.subprocess, "run", return_value=result,
                ), self.assertRaisesRegex(RuntimeError, "attest"):
                    showcase.main()
                self.assertFalse(output.exists())

    def test_draw_repair_semantic_action_ignores_only_draw_source(self) -> None:
        base = {"card": "Y2", "act": "discard", "draw": "deck"}
        self.assertTrue(showcase.same_semantic_action(
            base, {"card": "Y2", "act": "discard", "draw": "W"}
        ))
        self.assertFalse(showcase.same_semantic_action(
            base, {"card": "Y3", "act": "discard", "draw": "W"}
        ))
        self.assertFalse(showcase.same_semantic_action(
            base, {"card": "Y2", "act": "play", "draw": "W"}
        ))

    def test_policy_draw_repair_has_attested_provenance(self) -> None:
        actor = "policy:data/c8.bin:0:20:0:0:4"
        raw = {"card": "Y2", "act": "discard", "draw": "deck"}
        played = {"card": "Y2", "act": "discard", "draw": "W"}
        game = {
            "meta": {
                "actor": actor,
                "evaluator": showcase.DEFAULT_EVALUATOR,
                "seed": 1,
                "rounds": 3,
                "generated": "analyze",
                "plies": 1,
                "final": [0, 0],
            },
            "plies": [{
                "n": 1,
                "policy": [{**raw, "prob": 1.0}],
                "move": played,
                "actor_decision": {
                    "baseline_source": "draw_source_planner",
                    "late_resolver": late_stub(),
                },
                "analysis": {
                    "searched": False,
                    "late_resolver": late_stub(),
                },
                "search": [],
            }],
        }
        result = type("Result", (), {"stdout": json.dumps(game)})()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "showcase.json"
            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--actor", actor,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", return_value=result,
            ):
                showcase.main()
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["meta"]["actor_draw_root_deck_max"], 4)
            self.assertIn("draw_repair", saved["meta"]["actor_method"])

    def test_same_output_and_viewer_is_rejected_before_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "viewer.html"
            original = (
                '<script type="application/json" id="game-data">'
                '{"old":true}</script>'
            )
            target.write_text(original, encoding="utf-8")
            argv = [
                "make_showcase.py",
                "--seed", "1",
                "--output", str(target),
                "--embed-viewer", str(target),
                "--actor", self.ACTOR,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run"
            ) as run, self.assertRaisesRegex(
                RuntimeError, "must differ"
            ):
                showcase.main()
            run.assert_not_called()
            self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_bad_viewer_is_rejected_before_output_or_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "showcase.json"
            viewer = parent / "viewer.html"
            output.write_text("keep\n", encoding="utf-8")
            viewer.write_text("no game marker", encoding="utf-8")
            argv = [
                "make_showcase.py",
                "--seed", "1",
                "--output", str(output),
                "--embed-viewer", str(viewer),
                "--actor", self.ACTOR,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run"
            ) as run, self.assertRaisesRegex(
                RuntimeError, "exactly one game-data script"
            ):
                showcase.main()
            run.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "keep\n")

    def test_unfinished_continuation_cannot_replace_artifacts(self) -> None:
        game = json.loads(self.analyzer_result().stdout)
        game["plies"][0]["actor_decision"] = {
            "deck2_replan": {"unfinished_continuation_leaves": 1},
            "late_resolver": late_stub(),
        }
        result = type("Result", (), {"stdout": json.dumps(game)})()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "showcase.json"
            viewer = parent / "viewer.html"
            output.write_text('{"old":true}\n', encoding="utf-8")
            viewer.write_text(
                'before<script type="application/json" id="game-data">'
                '{"old":true}</script>after',
                encoding="utf-8",
            )
            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--embed-viewer", str(viewer), "--actor", self.ACTOR,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", return_value=result,
            ), self.assertRaisesRegex(RuntimeError, "unfinished continuation"):
                showcase.main()
            self.assertEqual(output.read_text(encoding="utf-8"), '{"old":true}\n')
            self.assertIn('{"old":true}</script>', viewer.read_text())

    def test_bounded_policy_retention_needs_no_challenger_gain_gate(self) -> None:
        game = json.loads(self.analyzer_result().stdout)
        retained = late_stub()
        retained.update({
            "enabled": True,
            "attempted": True,
            "completed": True,
            "stable": True,
            "retained_policy": True,
            "used_to_select": True,
            "selection_reason": "baseline_best",
            "support": 90,
            "candidate_count": 1,
            "candidates": [{
                "card": "Y2", "act": "discard", "draw": "deck",
                "policy_prob": 1.0, "horizon2_q": 0.0,
                "horizon4_q": 0.0,
                "policy_baseline": True, "horizon2_best": True,
                "horizon4_best": True,
            }],
        })
        retained["horizon2"]["best_index"] = 0
        retained["horizon4"]["best_index"] = 0
        game["plies"][0]["actor_decision"]["late_resolver"] = retained
        result = type("Result", (), {"stdout": json.dumps(game)})()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "showcase.json"
            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--actor", self.ACTOR,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", return_value=result,
            ):
                showcase.main()
            self.assertTrue(json.loads(output.read_text())["plies"][0]
                            ["actor_decision"]["late_resolver"]
                            ["retained_policy"])

    def test_bounded_policy_retention_can_reject_a_stable_challenger(self) -> None:
        late = challenger_late_stub(True)
        late.update({
            "retained_policy": True,
            "override_authorized": False,
            "practical_gate_passed": False,
            "selection_reason": "below_practical_gain",
        })
        late["candidates"][1]["horizon2_q"] = 0.4
        late["candidates"][1]["horizon4_q"] = 0.7
        late["horizon2"]["value"] = 0.4
        late["horizon2"]["delta_vs_policy"] = 0.4
        late["horizon4"]["value"] = 0.7
        late["horizon4"]["delta_vs_policy"] = 0.7
        showcase.validate_late_resolver(late, 42, "analysis")
        self.assertTrue(late["used_to_select"])
        self.assertEqual(late["horizon4"]["best_index"], 1)

    def test_bounded_policy_retention_can_follow_horizon_disagreement(self) -> None:
        late = challenger_late_stub(True)
        late.update({
            "stable": False,
            "retained_policy": True,
            "override_authorized": False,
            "practical_gate_passed": False,
            "selection_reason": "horizon_disagreement",
        })
        late["horizon4"]["best_index"] = 0
        late["horizon4"]["value"] = 0.0
        late["horizon4"]["delta_vs_policy"] = 0.0
        late["candidates"][0]["horizon4_best"] = True
        late["candidates"][1]["horizon4_best"] = False
        showcase.validate_late_resolver(late, 42, "analysis")
        self.assertTrue(late["retained_policy"])
        self.assertNotEqual(
            late["horizon2"]["best_index"], late["horizon4"]["best_index"]
        )

    def test_bounded_challenger_without_gain_gate_is_rejected(self) -> None:
        game = json.loads(self.analyzer_result().stdout)
        invalid = challenger_late_stub(False)
        game["plies"][0]["actor_decision"]["late_resolver"] = invalid
        result = type("Result", (), {"stdout": json.dumps(game)})()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "showcase.json"
            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--actor", self.ACTOR,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", return_value=result,
            ), self.assertRaisesRegex(RuntimeError, "mislabels the challenger"):
                showcase.main()

    def test_bounded_challenger_with_gain_gate_is_accepted(self) -> None:
        game = json.loads(self.analyzer_result().stdout)
        game["plies"][0]["actor_decision"]["late_resolver"] = (
            challenger_late_stub(True)
        )
        result = type("Result", (), {"stdout": json.dumps(game)})()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "showcase.json"
            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--actor", self.ACTOR,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", return_value=result,
            ):
                showcase.main()
            late = json.loads(output.read_text())["plies"][0]
            late = late["actor_decision"]["late_resolver"]
            self.assertTrue(late["override_authorized"])
            self.assertTrue(late["practical_gate_passed"])

    def test_bounded_payload_matches_viewer_schema_limits(self) -> None:
        mutations = (
            lambda late: late["candidates"][0].update(card="invalid"),
            lambda late: late["horizon2"].update(value=1e10),
            lambda late: late["candidates"][0].update(q="bad"),
            lambda late: late.update(practical_threshold=-1.0),
            lambda late: late.update(practical_threshold=3.0),
            lambda late: late.update(stable=False),
            lambda late: late["candidates"][1].update(
                card=late["candidates"][0]["card"],
                act=late["candidates"][0]["act"],
                draw=late["candidates"][0]["draw"],
            ),
            lambda late: late["candidates"][1].update(horizon2_q=3.0),
            lambda late: late.update(selection_reason="baseline_best"),
            lambda late: late.update(used_to_select=False),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                late = challenger_late_stub(True)
                mutate(late)
                with self.assertRaises(RuntimeError):
                    showcase.validate_late_resolver(late, 1, "analysis")

    def test_success_installs_matching_validated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "showcase.json"
            viewer = parent / "viewer.html"
            output.write_text('{"old":true}\n', encoding="utf-8")
            viewer.write_text(
                "before"
                '<script type="application/json" id="game-data">'
                '{"old":true}</script>'
                "after",
                encoding="utf-8",
            )
            output.chmod(0o640)
            viewer.chmod(0o644)
            run = self.run_main(
                "--seed", "1",
                "--output", str(output),
                "--embed-viewer", str(viewer),
            )
            run.assert_called_once()
            standalone = json.loads(output.read_text(encoding="utf-8"))
            match = re.search(
                r'<script type="application/json" id="game-data">(.*?)</script>',
                viewer.read_text(encoding="utf-8"),
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            self.assertEqual(json.loads(match.group(1)), standalone)
            self.assertEqual(standalone["meta"]["actor"], self.ACTOR)
            self.assertEqual(
                standalone["meta"]["match_id"],
                "1-" + hashlib.sha256(
                    (ROOT / "data/c8.bin").read_bytes()
                ).hexdigest()[:12],
            )
            self.assertNotIn("root_model_path", standalone["meta"])
            self.assertNotIn("continuation_model_path", standalone["meta"])
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o640)
            self.assertEqual(os.stat(viewer).st_mode & 0o777, 0o644)

    def test_concurrent_viewer_fix_is_preserved_after_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "showcase.json"
            viewer = parent / "viewer.html"
            viewer.write_text(
                "old-before"
                '<script type="application/json" id="game-data">'
                '{"old":true}</script>'
                "old-after",
                encoding="utf-8",
            )

            def analyzer_finished(*_args, **_kwargs):
                viewer.write_text(
                    "fixed-before"
                    '<script type="application/json" id="game-data">'
                    '{"old":true}</script>'
                    "fixed-after",
                    encoding="utf-8",
                )
                return self.analyzer_result()

            argv = [
                "make_showcase.py", "--seed", "1", "--output", str(output),
                "--embed-viewer", str(viewer), "--actor", self.ACTOR,
            ]
            with patch.object(sys, "argv", argv), patch.object(
                showcase.subprocess, "run", side_effect=analyzer_finished,
            ):
                showcase.main()
            rendered = viewer.read_text(encoding="utf-8")
            self.assertTrue(rendered.startswith("fixed-before"))
            self.assertTrue(rendered.endswith("fixed-after"))
            self.assertNotIn("old-before", rendered)

    def test_tracked_viewer_payload_matches_attested_random_match(self) -> None:
        standalone_text = (ROOT / "data/analysis.json").read_text(
            encoding="utf-8"
        ).strip()
        viewer_bytes = (ROOT / "web/viewer.html").read_bytes()
        viewer_text = viewer_bytes.decode("utf-8")
        viewer_prefix = viewer_bytes[:1024].lower()
        charset_tag = b'<meta charset="utf-8">'
        first_non_ascii = next(
            i for i, byte in enumerate(viewer_bytes) if byte >= 0x80
        )
        self.assertTrue(viewer_text.lower().startswith("<!doctype html>"))
        self.assertIn(charset_tag, viewer_prefix)
        self.assertLess(
            viewer_prefix.index(charset_tag),
            first_non_ascii,
            "the UTF-8 declaration must precede non-ASCII viewer text",
        )
        self.assertIn(" → ", viewer_text)
        self.assertNotIn("â†’", viewer_text)
        self.assertNotIn(
            "items.push(", viewer_text,
            "viewer references an undefined diagnostics accumulator",
        )
        self.assertIn("failed_controller_veto", viewer_text)
        self.assertIn("controller_veto_evaluated", viewer_text)
        self.assertIn("This panel can cancel a move but can never introduce one.",
                      viewer_text)
        match = re.search(
            r'<script type="application/json" id="game-data">(.*?)</script>',
            viewer_text,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), standalone_text)
        tracked = json.loads(standalone_text)
        meta = tracked["meta"]
        self.assertEqual(meta["actor"], showcase.DEFAULT_ACTOR)
        self.assertEqual(meta["evaluator"], self.TRACKED_AUDIT)
        self.assertEqual(meta["seed"], 181292076812167)
        self.assertEqual(meta["match_id"], "181292076812167-af2b2c237d21")
        self.assertEqual(meta["plies"], 143)
        self.assertEqual(len(tracked["plies"]), meta["plies"])
        self.assertEqual(meta["final"], [191, 172])
        self.assertTrue(meta["actor_exact_terminal"])
        self.assertTrue(meta["actor_exact_terminal_continuations"])
        self.assertEqual(meta["actor_terminal_mode"], 1)
        self.assertEqual(meta["actor_deck2_replan_worlds"], 0)
        self.assertEqual(meta["actor_deck2_replan_cores"], 0)
        self.assertFalse(meta["actor_bounded_late_root"])
        self.assertTrue(meta["evaluator_bounded_late_root"])
        self.assertEqual(meta["model_sha256"], self.CHAMPION_SHA256)
        self.assertEqual(
            hashlib.sha256((ROOT / "data/champion.bin").read_bytes()).hexdigest(),
            self.CHAMPION_SHA256,
        )
        self.assertEqual(meta["selection"], "random_unfiltered")
        reserve_forces = 0
        replan_calls = 0
        replan_root_calls = 0
        bounded_attempts = 0
        bounded_completions = 0
        for ply in tracked["plies"]:
            for panel_name in ("actor_decision", "analysis"):
                work = ply[panel_name]["deck2_replan"]
                self.assertEqual(work["method"], "recursive_deck_2_to_3")
                self.assertIn("cycle_breaks", work)
                self.assertIn("cap_reserve_forces", work)
                for field in (
                    "root_calls", "root_worlds", "low_world_fallbacks",
                    "transposition_hits", "recursive_cycle_closures",
                    "max_recursive_depth", "max_stall_chain",
                ):
                    self.assertIn(field, work)
                self.assertEqual(work["unfinished_continuation_leaves"], 0)
                reserve_forces += work["cap_reserve_forces"]
                late = ply[panel_name]["late_resolver"]
                self.assertEqual(
                    late["method"],
                    "one_sided_bounded_particle_policy_improvement",
                )
                for field in (
                    "enabled", "attempted", "completed", "used_to_select", "stable",
                    "retained_policy", "override_authorized",
                    "practical_gate_passed",
                ):
                    self.assertIsInstance(late[field], bool)
                self.assertEqual(late["enabled"], panel_name == "analysis")
                self.assertIn("horizon2", late)
                self.assertIn("horizon4", late)
                self.assertIn("candidates", late)
                self.assertEqual(late["candidate_count"], len(late["candidates"]))
                self.assertFalse(
                    late["retained_policy"] and late["override_authorized"]
                )
                self.assertEqual(
                    late["practical_gate_passed"],
                    late["override_authorized"],
                )
                self.assertIn(late["selection_reason"], {
                    "not_attempted", "unavailable", "baseline_best",
                    "below_practical_gain", "horizon_disagreement",
                    "challenger_override",
                })
                self.assertEqual(late["used_to_select"], late["completed"])
                if late["used_to_select"]:
                    self.assertNotEqual(
                        late["retained_policy"], late["override_authorized"]
                    )
                bounded_attempts += int(late["attempted"])
                bounded_completions += int(late["completed"])
            audit = ply["analysis"]["deck2_replan"]
            self.assertFalse(audit["enabled"])
            self.assertEqual(audit["configured_worlds"], 0)
            self.assertEqual(audit["configured_cores"], 0)
            replan_calls += audit["calls"]
            replan_root_calls += audit["root_calls"]
        self.assertGreater(reserve_forces, 0)
        self.assertEqual(replan_calls, 0)
        self.assertEqual(replan_root_calls, 0)
        self.assertGreater(bounded_attempts, 0)
        self.assertGreater(bounded_completions, 0)


if __name__ == "__main__":
    unittest.main()
