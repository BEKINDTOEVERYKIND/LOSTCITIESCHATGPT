"""Provenance contracts for the one-shot viewer generator."""
from __future__ import annotations

import importlib.util
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


class ShowcaseProvenanceTests(unittest.TestCase):
    ACTOR = "policy:data/c8.bin:0:20"

    @classmethod
    def analyzer_result(cls):
        move = {
            "card": "Y2",
            "act": "discard",
            "draw": "deck",
            "drawn": "Y3",
        }
        game = {
            "meta": {"actor": cls.ACTOR, "plies": 1, "final": [0, 0]},
            "plies": [{
                "n": 1,
                "policy": [{**move, "prob": 1.0}],
                "move": move,
                "analysis": {"searched": False},
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

    def test_non_network_actor_cannot_claim_model_provenance(self) -> None:
        for spec in ("heur", "random", "policy:"):
            with self.subTest(spec=spec), self.assertRaises(RuntimeError):
                showcase.actor_model_path(spec)

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
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o640)
            self.assertEqual(os.stat(viewer).st_mode & 0o777, 0o644)


if __name__ == "__main__":
    unittest.main()
