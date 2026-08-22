import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.gate_actor_panel import evaluate_gate
from tools.merge_arena import EvidenceError
from tools.validate_actor_shards import validate_shards


ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "bin" / "arena"
WORKFLOW = ROOT / ".github" / "workflows" / "continuation-soup-v1.yml"
TRANSPORT_ARENA = ROOT / "bin" / "arena"
TRANSPORT_ROOT = ROOT / "data" / "champion.bin"


class LockedActorPanelTest(unittest.TestCase):
    def test_workflow_has_one_addendum_trigger_and_exact_locked_panels(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("workflow_dispatch", text)
        self.assertEqual(text.count(
            "data/experiments/locked_continuation_soup_v1_distributed_execution_retry.json"),
            4)
        self.assertNotIn(
            "data/experiments/locked_continuation_soup_v1_distributed_execution.json\n",
            text)
        self.assertIn(
            "start: [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]",
            text)
        self.assertIn(
            "start: [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, "
            "1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, "
            "1900, 2000, 2100, 2200, 2300, 2400]", text)
        self.assertIn("-n 20 -t 4 -s \"$SEED\" -r 3", text)
        self.assertIn("-n 100 -t 4 -s \"$SEED\" -r 3", text)
        self.assertIn("--expect-pairs 200", text)
        self.assertIn("--expect-pairs 2500", text)
        self.assertIn("--mode safety", text)
        self.assertIn("--mode final", text)
        self.assertIn("needs.safety_merge.outputs.passed == 'true'", text)
        self.assertIn(
            "e88c97912165165f200e7e1ebe9705971916c93083fa0049d63aa6eeba1adb8d",
            text)
        self.assertIn(
            "a65f73871e04ed0e21f2bc9920a235cbb5d29950b35bb91df37ae9dcc8801efa",
            text)
        self.assertIn(
            'test "$(sha256sum campaign/bin/arena | cut -d\' \' -f1)" = "$ARENA_SHA"',
            text)
        self.assertNotIn("make -C source", text)
        self.assertNotIn("path: source", text)
        self.assertEqual(TRANSPORT_ARENA.stat().st_size, 362592)
        self.assertEqual(TRANSPORT_ROOT.stat().st_size, 2823748)
        self.assertEqual(
            hashlib.sha256(TRANSPORT_ARENA.read_bytes()).hexdigest(),
            "a65f73871e04ed0e21f2bc9920a235cbb5d29950b35bb91df37ae9dcc8801efa")
        self.assertEqual(
            hashlib.sha256(TRANSPORT_ROOT.read_bytes()).hexdigest(),
            "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417")

    def _shard(self, directory: Path, orientation: str, start: int,
               count: int, seed: int) -> None:
        candidate, baseline = "random", "heur"
        agent_a, agent_b = ((candidate, baseline)
                            if orientation == "candidate-first"
                            else (baseline, candidate))
        stem = f"{orientation}-{start}"
        raw = directory / f"{stem}.jsonl"
        subprocess.run([
            str(ARENA), "-a", agent_a, "-b", agent_b,
            "-n", str(count), "-t", "2", "-s", str(seed), "-r", "3",
            "--pair-start", str(start), "--raw-pairs", str(raw),
            "--raw-only", "--provenance", "locked-panel-test",
        ], cwd=ROOT, check=True)
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        (directory / f"{stem}.sha256").write_text(
            f"{digest}  raw/{stem}.jsonl\n", encoding="ascii")
        (directory / f"{stem}.time").write_text(
            "wall_s=1.25 user_s=1.00 sys_s=0.01 max_rss_kb=1234 exit=0\n",
            encoding="ascii")

    def test_complete_shard_set_validates_without_estimating(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for orientation, seed in (("candidate-first", 101),
                                      ("baseline-first", 102)):
                for start in (0, 2):
                    self._shard(directory, orientation, start, 2, seed)
            value = validate_shards(
                directory, "random", "heur", "locked-panel-test",
                "101", "102", [0, 2], 2)
            self.assertEqual(value["status"],
                             "complete_structurally_valid_before_efficacy_merge")
            self.assertEqual(value["pairs_per_orientation"], 4)
            self.assertEqual(len(value["shards"]), 4)
            self.assertNotIn("result", value)

            sidecar = directory / "candidate-first-0.sha256"
            original = sidecar.read_text()
            sidecar.write_text("0" * 64 + original[64:])
            with self.assertRaises(EvidenceError):
                validate_shards(directory, "random", "heur",
                                "locked-panel-test", "101", "102",
                                [0, 2], 2)

    @staticmethod
    def _result(score=0.5, score_se=0.0, margin=0.25, margin_se=0.0,
                orientations=(0.525, 0.475), first_quarters=420,
                second_quarters=420, first_margin=1, second_margin=0,
                caps=0):
        return {
            "raw_input_validation": {"status": "validated"},
            "candidate_result": {
                "match_score": score,
                "match_score_pair_clustered_se": score_se,
                "margin_per_game": margin,
                "margin_pair_clustered_se": margin_se,
                "orientation_match_scores": list(orientations),
                "capped_rounds": caps,
            },
            "blocks": [
                {"sufficient_statistics": {
                    "pairs": 200, "score_quarters_sum": first_quarters,
                    "margin_sum": first_margin}},
                {"sufficient_statistics": {
                    "pairs": 200, "score_quarters_sum": second_quarters,
                    "margin_sum": second_margin}},
            ],
        }

    def test_safety_gate_uses_locked_inclusive_floors_and_exact_integers(self):
        decision = evaluate_gate(self._result(), "safety")
        self.assertTrue(decision["passed"])
        self.assertTrue(all(decision["requirements"].values()))

        self.assertFalse(evaluate_gate(
            self._result(first_margin=0, second_margin=0, margin=0.0),
            "safety")["passed"])
        self.assertFalse(evaluate_gate(
            self._result(first_quarters=419, second_quarters=420,
                         score=0.499375), "safety")["passed"])
        self.assertFalse(evaluate_gate(
            self._result(second_quarters=421, orientations=(0.525, 0.47375)),
            "safety")["passed"])
        self.assertFalse(evaluate_gate(self._result(caps=1), "safety")["passed"])

    def test_final_gate_requires_both_one_sided_bounds_and_orientations(self):
        passing = self._result(
            score=0.51, score_se=0.005, margin=1.0, margin_se=0.5,
            orientations=(0.51, 0.52))
        self.assertTrue(evaluate_gate(passing, "final")["passed"])

        weak_margin = self._result(
            score=0.51, score_se=0.005, margin=0.5, margin_se=0.5,
            orientations=(0.51, 0.52))
        self.assertFalse(evaluate_gate(weak_margin, "final")["passed"])
        weak_orientation = self._result(
            score=0.51, score_se=0.005, margin=1.0, margin_se=0.5,
            orientations=(0.5, 0.52))
        self.assertFalse(evaluate_gate(weak_orientation, "final")["passed"])
        with self.assertRaises(EvidenceError):
            evaluate_gate(passing, "final", float("nan"))


if __name__ == "__main__":
    unittest.main()
