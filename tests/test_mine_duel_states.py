"""Security contracts for opponent-proposed state mining."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/mine_duel_states.py"
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("mine_duel_states", MODULE_PATH)
assert SPEC and SPEC.loader
mine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mine)


class MineDuelStateSafetyTests(unittest.TestCase):
    def test_match_id_is_one_portable_component(self) -> None:
        self.assertEqual(mine.checked_match_id("p0001s0"), "p0001s0")
        for unsafe in (None, "", ".", "../escape", "/tmp/escape", "a/b", "a\\b", " x"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                mine.checked_match_id(unsafe)

    def test_atomic_install_never_replaces_destination(self) -> None:
        with tempfile.TemporaryDirectory() as parent_name:
            parent = Path(parent_name)
            source = parent / "source"
            destination = parent / "output"
            source.mkdir()
            (source / "manifest.json").write_text("new", encoding="utf-8")
            destination.mkdir()
            (destination / "keep").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                mine.install_directory_noreplace(source, destination)
            self.assertEqual((destination / "keep").read_text(encoding="utf-8"), "old")
            self.assertTrue(source.is_dir())

    def test_partial_competition_is_rejected(self) -> None:
        self.assertEqual(len(mine.checked_rounds([{}, {}, {}])), 3)
        for partial in (None, [], [{}], [{}, {}]):
            with self.subTest(partial=partial), self.assertRaises(ValueError):
                mine.checked_rounds(partial)


if __name__ == "__main__":
    unittest.main()
