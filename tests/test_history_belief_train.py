"""End-to-end contracts for causal history-belief training evidence."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "bin" / "history_belief_train"
ACTOR = ROOT / "data" / "champion.bin"
EXCLUSIONS = (
    ROOT / "data" / "experiments" /
    "policy_cost_v7_exact17_exclusions.txt"
)
EXCLUSIONS_SHA256 = (
    "10034cf8b83aadf24fa0775e4dad2712573e1d84cbf364568ce6136682ac254c"
)


def net_head_slices(payload: bytes) -> tuple[bytes, bytes, list[float]]:
    """Return immutable prefix/suffix and belief-head floats from v6 Net."""
    header = struct.unpack_from("<6I", payload, 0)
    self_magic, feat_dim, h1, h2, nplay, version = header
    if self_magic != 0x4C435651 or version != 6 or nplay != 120:
        raise ValueError("unexpected net artifact")
    ndraw = 6
    ncards = nplay // 2
    float_offset = (
        feat_dim * h1 + h1 + h1 * h2 + h2 + h2 + 1 +
        nplay * h2 + nplay + ndraw * h2 + ndraw
    )
    head_floats = ncards * h2 + ncards
    start = 24 + 4 * float_offset
    end = start + 4 * head_floats
    values = list(struct.unpack_from(f"<{head_floats}f", payload, start))
    return payload[:start], payload[end:], values


def strict_json(text: str) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token: {value}")

    return json.loads(text, parse_constant=reject_constant)


class HistoryBeliefTrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="lc-history-belief-test-"
        )
        cls.root = Path(cls.temporary.name)
        cls.model = cls.root / "model.bin"
        result = subprocess.run(
            [
                str(TOOL), "train",
                "--out", str(cls.model),
                "--actor-net", str(ACTOR),
                "--base-net", str(ACTOR),
                "--matches", "2",
                "--rounds", "1",
                "--seed", "71001",
                "--match-start", "0",
                "--max-ply", "4",
                "--symmetries", "1",
                "--temperature", "0",
                "--base-alpha", "1.15",
                "--lr", "0.004",
                "--l2", "0.0000003",
                "--exclusions", str(EXCLUSIONS),
                "--exclusions-sha256", EXCLUSIONS_SHA256,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        cls.train_report = strict_json(result.stdout)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_train_receipt_binds_optimizer_and_consumed_labels(self) -> None:
        report = self.train_report
        self.assertEqual(report["mode"], "train")
        self.assertAlmostEqual(report["training_learning_rate"],
                               0.004, places=8)
        self.assertAlmostEqual(report["training_l2"], 0.0000003, places=12)
        self.assertEqual(report["model_train_states"],
                         report["source_state_count"])
        self.assertGreater(report["model_train_states"], 0)
        self.assertIsNone(report["matched_base_net_fingerprint"])
        self.assertIsNone(report["matched_base_alpha"])
        self.assertIsNone(report["incumbent_alpha"])
        self.assertIsNone(report["incumbent_net_fingerprint"])
        self.assertRegex(report["output_sha256"], r"^[0-9a-f]{64}$")

    def run_eval(
        self, matches: int, match_start: int, output: Path,
        *, base_net: Path = ACTOR, temperature: str = "0",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(TOOL), "eval",
                "--model", str(self.model),
                "--actor-net", str(ACTOR),
                "--base-net", str(base_net),
                "--matches", str(matches),
                "--rounds", "1",
                "--seed", "72002",
                "--match-start", str(match_start),
                "--max-ply", "4",
                "--symmetries", "1",
                "--temperature", temperature,
                "--base-alpha", "1.15",
                "--incumbent-alpha", "1.15",
                "--exclusions", str(EXCLUSIONS),
                "--exclusions-sha256", EXCLUSIONS_SHA256,
                "--match-jsonl", str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_match_jsonl_is_paired_finite_and_shard_stable(self) -> None:
        monolithic_path = self.root / "monolithic.jsonl"
        first_path = self.root / "first.jsonl"
        second_path = self.root / "second.jsonl"
        monolithic = self.run_eval(2, 10, monolithic_path)
        first = self.run_eval(1, 10, first_path)
        second = self.run_eval(1, 11, second_path)
        self.assertEqual(monolithic.returncode, 0, monolithic.stderr)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(
            monolithic_path.read_bytes(),
            first_path.read_bytes() + second_path.read_bytes(),
        )

        report = strict_json(monolithic.stdout)
        raw = monolithic_path.read_bytes()
        self.assertEqual(
            report["match_jsonl_sha256"], hashlib.sha256(raw).hexdigest()
        )
        rows = [strict_json(line) for line in raw.decode().splitlines()]
        self.assertEqual([row["source_match_id"] for row in rows], [10, 11])
        self.assertEqual(report["model_fingerprint"],
                         rows[0]["history_model_fingerprint"])
        self.assertEqual(report["actor_fingerprint"],
                         rows[0]["actor_fingerprint"])
        self.assertEqual(report["base_net_fingerprint"],
                         rows[0]["base_net_fingerprint"])
        self.assertEqual(report["incumbent_net_fingerprint"],
                         rows[0]["incumbent_net_fingerprint"])
        self.assertAlmostEqual(report["incumbent_alpha"], 1.15, places=6)
        self.assertEqual(report["exclusion_manifest_sha256"],
                         EXCLUSIONS_SHA256)
        self.assertEqual(report["exclusion_manifest_count"], 17)
        for row in rows:
            self.assertFalse(row["reviewed_ply_inputs_used"])
            contract = row["structural_contract"]
            self.assertTrue(contract["opening_history_uniform"])
            self.assertTrue(contract["truth_read_after_prediction"])
            self.assertTrue(
                contract["residual_features_opponent_action_anchored"]
            )
            self.assertTrue(
                contract["reviewed_ply_orbit_exclusion_enabled"]
            )
            self.assertFalse(contract["playing_actor_changed"])
            for scope in ("all_states", "post_opponent_action"):
                metrics = row["metrics"][scope]
                history = metrics["history"]
                current = metrics["base_262k_head"]
                incumbent = metrics["incumbent_head"]
                uniform = metrics["uniform_exact_k"]
                for baseline in (current, incumbent, uniform):
                    self.assertEqual(history["state_count"],
                                     baseline["state_count"])
                    self.assertEqual(history["uncertain_card_count"],
                                     baseline["uncertain_card_count"])
                    self.assertEqual(history["positive_count"],
                                     baseline["positive_count"])

    def test_artifact_rejects_base_net_and_temperature_mismatch(self) -> None:
        changed = self.root / "changed-base.bin"
        payload = bytearray(ACTOR.read_bytes())
        payload[-1] ^= 1
        changed.write_bytes(payload)
        base_mismatch = self.run_eval(1, 20, self.root / "bad-base.jsonl",
                                      base_net=changed)
        temperature_mismatch = self.run_eval(
            1, 20, self.root / "bad-temperature.jsonl", temperature="0.1"
        )
        self.assertEqual(base_mismatch.returncode, 1)
        self.assertIn("provenance mismatch", base_mismatch.stderr)
        self.assertEqual(temperature_mismatch.returncode, 1)
        self.assertIn("provenance mismatch", temperature_mismatch.stderr)

    def test_base_alpha_is_order_independent_and_bound(self) -> None:
        first_path = self.root / "alpha-first.jsonl"
        last_path = self.root / "alpha-last.jsonl"
        common = [
            str(TOOL), "eval",
            "--model", str(self.model),
            "--actor-net", str(ACTOR),
            "--base-net", str(ACTOR),
            "--matches", "1", "--rounds", "1",
            "--seed", "72003", "--match-start", "30",
            "--max-ply", "4", "--symmetries", "1",
            "--temperature", "0",
            "--incumbent-alpha", "1.15",
            "--exclusions", str(EXCLUSIONS),
            "--exclusions-sha256", EXCLUSIONS_SHA256,
        ]
        first = subprocess.run(
            common[:2] + ["--base-alpha", "1.15"] + common[2:] +
            ["--match-jsonl", str(first_path)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        last = subprocess.run(
            common + ["--base-alpha", "1.15",
                      "--match-jsonl", str(last_path)],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(last.returncode, 0, last.stderr)
        self.assertEqual(first.stdout, last.stdout)
        self.assertEqual(first_path.read_bytes(), last_path.read_bytes())
        mismatch = subprocess.run(
            common + ["--base-alpha", "1.0",
                      "--match-jsonl", str(self.root / "alpha-bad.jsonl")],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(mismatch.returncode, 1)
        self.assertIn("provenance mismatch", mismatch.stderr)

    def test_match_jsonl_is_exclusive_no_clobber(self) -> None:
        output = self.root / "existing.jsonl"
        output.write_text("immutable\n")
        result = self.run_eval(1, 40, output)
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot open match JSONL", result.stderr)
        self.assertEqual(output.read_text(), "immutable\n")

    def test_training_rejects_match_jsonl(self) -> None:
        result = subprocess.run(
            [
                str(TOOL), "train",
                "--out", str(self.root / "forbidden.bin"),
                "--matches", "1",
                "--exclusions", str(EXCLUSIONS),
                "--exclusions-sha256", EXCLUSIONS_SHA256,
                "--match-jsonl", str(self.root / "forbidden.jsonl"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "forbidden.jsonl").exists())

    def test_exclusion_binding_is_mandatory_and_fail_closed(self) -> None:
        common = [
            str(TOOL), "train", "--matches", "1", "--rounds", "1",
            "--max-ply", "1", "--symmetries", "1",
            "--temperature", "0",
        ]
        cases = [
            ("missing", [] , 2),
            ("half", ["--exclusions", str(EXCLUSIONS)], 2),
            ("mismatch", [
                "--exclusions", str(EXCLUSIONS),
                "--exclusions-sha256", "0" * 64,
            ], 1),
        ]
        for name, extra, expected in cases:
            with self.subTest(name=name):
                artifact = self.root / f"{name}.bin"
                result = subprocess.run(
                    common + ["--out", str(artifact)] + extra,
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, expected)
                self.assertEqual(result.stdout, "")
                self.assertFalse(artifact.exists())

    def test_head_control_is_exclusion_bound_resumable_and_source_matched(
        self,
    ) -> None:
        control = self.root / "control.bin"
        state = self.root / "control.state"
        common = [
            "--actor-net", str(ACTOR), "--base-net", str(ACTOR),
            "--matches", "2", "--rounds", "1", "--seed", "73011",
            "--match-start", "0", "--max-ply", "4",
            "--symmetries", "1", "--temperature", "0",
            "--base-alpha", "1", "--epochs", "1",
            "--lr", "0.0001", "--l2", "0.0000001",
            "--exclusions", str(EXCLUSIONS),
            "--exclusions-sha256", EXCLUSIONS_SHA256,
        ]
        trained = subprocess.run(
            [str(TOOL), "train-control", "--out", str(control),
             "--control-state-out", str(state),
             "--control-batch-states", "3", "--control-finalize"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(trained.returncode, 0, trained.stderr)
        control_report = strict_json(trained.stdout)
        self.assertEqual(control_report["schema"],
                         "lc-history-belief-control-run-v1")
        self.assertTrue(control_report["control_changed_only_belief_head"])
        self.assertTrue(control_report["control_finalized"])
        self.assertEqual(control_report["pending_batch_states"], 0)
        self.assertEqual(control_report["exclusion_manifest_sha256"],
                         EXCLUSIONS_SHA256)
        self.assertTrue(control.exists())
        self.assertTrue(state.exists())
        actor_sha256 = hashlib.sha256(ACTOR.read_bytes()).hexdigest()
        control_sha256 = hashlib.sha256(control.read_bytes()).hexdigest()
        self.assertEqual(control_report["input_checkpoint_sha256"],
                         actor_sha256)
        self.assertEqual(control_report["output_sha256"], control_sha256)
        self.assertEqual(
            control_report["control_state_checkpoint_sha256"],
            control_sha256,
        )
        actor_prefix, actor_suffix, actor_head = net_head_slices(
            ACTOR.read_bytes()
        )
        control_prefix, control_suffix, control_head = net_head_slices(
            control.read_bytes()
        )
        self.assertEqual(actor_prefix, control_prefix)
        self.assertEqual(actor_suffix, control_suffix)
        self.assertNotEqual(actor_head, control_head)
        h2 = 256
        for suit in range(5):
            cards = [suit * 12 + rank for rank in range(3)]
            rows = [control_head[card * h2:(card + 1) * h2]
                    for card in cards]
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(rows[0], rows[2])
            biases = [control_head[60 * h2 + card] for card in cards]
            self.assertEqual(biases[0], biases[1])
            self.assertEqual(biases[0], biases[2])

        residual = self.root / "source-matched-model.bin"
        fitted = subprocess.run(
            [str(TOOL), "train", "--out", str(residual)] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(fitted.returncode, 0, fitted.stderr)
        residual_report = strict_json(fitted.stdout)
        self.assertEqual(control_report["source_manifest_sha256"],
                         residual_report["source_manifest_sha256"])
        self.assertEqual(control_report["source_state_count"],
                         residual_report["source_state_count"])

        no_clobber = subprocess.run(
            [str(TOOL), "train-control", "--out", str(control),
             "--control-state-out", str(self.root / "unused.state"),
             "--control-batch-states", "3"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(no_clobber.returncode, 1)
        self.assertIn("output already exists", no_clobber.stderr)

    def test_head_control_resume_binds_next_range_and_eval_comparator(
        self,
    ) -> None:
        first_net = self.root / "resume-first.bin"
        first_state = self.root / "resume-first.state"
        common = [
            "--actor-net", str(ACTOR), "--rounds", "1",
            "--seed", "73012", "--max-ply", "4", "--symmetries", "1",
            "--temperature", "0", "--base-alpha", "1", "--epochs", "1",
            "--lr", "0.0001", "--l2", "0.0000001",
            "--control-batch-states", "10",
            "--exclusions", str(EXCLUSIONS),
            "--exclusions-sha256", EXCLUSIONS_SHA256,
        ]
        first = subprocess.run(
            [str(TOOL), "train-control", "--out", str(first_net),
             "--control-state-out", str(first_state),
             "--base-net", str(ACTOR), "--matches", "1",
             "--match-start", "0"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertGreater(strict_json(first.stdout)["pending_batch_states"], 0)

        bad_net = self.root / "resume-bad.bin"
        bad = subprocess.run(
            [str(TOOL), "train-control", "--out", str(bad_net),
             "--control-state-in", str(first_state),
             "--control-state-out", str(self.root / "resume-bad.state"),
             "--base-net", str(first_net), "--matches", "1",
             "--match-start", "2"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(bad.returncode, 1)
        self.assertIn("resume provenance mismatch", bad.stderr)
        self.assertFalse(bad_net.exists())

        final_net = self.root / "resume-final.bin"
        final_state = self.root / "resume-final.state"
        resumed = subprocess.run(
            [str(TOOL), "train-control", "--out", str(final_net),
             "--control-state-in", str(first_state),
             "--control-state-out", str(final_state),
             "--base-net", str(first_net), "--matches", "1",
             "--match-start", "1", "--control-finalize"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(strict_json(resumed.stdout)["pending_batch_states"], 0)

        model = self.root / "matched-model.bin"
        trained_model = subprocess.run(
            [
                str(TOOL), "train", "--out", str(model),
                "--actor-net", str(ACTOR), "--base-net", str(final_net),
                "--matches", "1", "--rounds", "1", "--seed", "73013",
                "--match-start", "0", "--max-ply", "4",
                "--symmetries", "1", "--temperature", "0",
                "--base-alpha", "1", "--exclusions", str(EXCLUSIONS),
                "--exclusions-sha256", EXCLUSIONS_SHA256,
            ], cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(trained_model.returncode, 0, trained_model.stderr)
        rows = self.root / "matched-rows.jsonl"
        evaluated = subprocess.run(
            [
                str(TOOL), "eval", "--model", str(model),
                "--actor-net", str(ACTOR), "--base-net", str(final_net),
                "--matched-base-net", str(first_net),
                "--matched-base-alpha", "1", "--matches", "1",
                "--rounds", "1", "--seed", "73014", "--match-start", "0",
                "--max-ply", "4", "--symmetries", "1",
                "--temperature", "0", "--base-alpha", "1",
                "--incumbent-alpha", "1.15",
                "--match-jsonl", str(rows),
                "--exclusions", str(EXCLUSIONS),
                "--exclusions-sha256", EXCLUSIONS_SHA256,
            ], cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        row = strict_json(rows.read_text())
        self.assertIn("matched_head_control",
                      row["metrics"]["post_opponent_action"])

    def test_head_control_resume_is_monolithic_equivalent_and_strict(
        self,
    ) -> None:
        common = [
            "--actor-net", str(ACTOR), "--rounds", "1",
            "--seed", "73015", "--max-ply", "4", "--symmetries", "1",
            "--temperature", "0", "--base-alpha", "1", "--epochs", "1",
            "--lr", "0.0001", "--l2", "0.0000001",
            "--control-batch-states", "3",
            "--exclusions", str(EXCLUSIONS),
            "--exclusions-sha256", EXCLUSIONS_SHA256,
        ]
        mono_net = self.root / "mono.bin"
        mono_state = self.root / "mono.state"
        mono = subprocess.run(
            [str(TOOL), "train-control", "--out", str(mono_net),
             "--control-state-out", str(mono_state),
             "--base-net", str(ACTOR), "--matches", "2",
             "--match-start", "0", "--control-finalize"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(mono.returncode, 0, mono.stderr)
        mono_report = strict_json(mono.stdout)

        part_net = self.root / "part.bin"
        part_state = self.root / "part.state"
        part = subprocess.run(
            [str(TOOL), "train-control", "--out", str(part_net),
             "--control-state-out", str(part_state),
             "--base-net", str(ACTOR), "--matches", "1",
             "--match-start", "0"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(part.returncode, 0, part.stderr)
        part_report = strict_json(part.stdout)
        split_net = self.root / "split.bin"
        split_state = self.root / "split.state"
        split = subprocess.run(
            [str(TOOL), "train-control", "--out", str(split_net),
             "--control-state-in", str(part_state),
             "--control-state-out", str(split_state),
             "--base-net", str(part_net), "--matches", "1",
             "--match-start", "1", "--control-finalize"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(split.returncode, 0, split.stderr)
        split_report = strict_json(split.stdout)
        self.assertEqual(mono_net.read_bytes(), split_net.read_bytes())
        actor_sha256 = hashlib.sha256(ACTOR.read_bytes()).hexdigest()
        part_sha256 = hashlib.sha256(part_net.read_bytes()).hexdigest()
        final_sha256 = hashlib.sha256(mono_net.read_bytes()).hexdigest()
        self.assertEqual(mono_report["input_checkpoint_sha256"], actor_sha256)
        self.assertEqual(part_report["input_checkpoint_sha256"], actor_sha256)
        self.assertEqual(split_report["input_checkpoint_sha256"], part_sha256)
        for report in (mono_report, split_report):
            self.assertEqual(report["output_sha256"], final_sha256)
            self.assertEqual(report["control_state_checkpoint_sha256"],
                             final_sha256)
        self.assertEqual(part_report["output_sha256"], part_sha256)
        self.assertEqual(part_report["control_state_checkpoint_sha256"],
                         part_sha256)
        mono_state_bytes = mono_state.read_bytes()
        split_state_bytes = split_state.read_bytes()
        self.assertEqual(struct.unpack_from("<II", mono_state_bytes, 8),
                         (3, 224))
        self.assertEqual(struct.unpack_from("<II", split_state_bytes, 8),
                         (3, 224))
        self.assertEqual(mono_state_bytes[192:224],
                         hashlib.sha256(mono_net.read_bytes()).digest())
        self.assertEqual(split_state_bytes[192:224],
                         hashlib.sha256(split_net.read_bytes()).digest())
        # The v3 header deliberately records this invocation's public-source
        # digest at bytes 160..191. The bound output-checkpoint SHA at
        # bytes 192..223, optimizer payload, and every other field are exact.
        self.assertEqual(mono_state_bytes[:160], split_state_bytes[:160])
        self.assertEqual(mono_state_bytes[192:], split_state_bytes[192:])

        mismatch_cases = {
            "seed": ["--seed", "73016"],
            "start": ["--match-start", "2"],
            "rounds": ["--rounds", "2"],
            "maxply": ["--max-ply", "5"],
            "batch": ["--control-batch-states", "8"],
            "sym": ["--symmetries", "5"],
            "temp": ["--temperature", "0.1"],
            "alpha": ["--base-alpha", "1.1"],
            "lr": ["--lr", "0.0002"],
            "l2": ["--l2", "0.000001"],
        }
        for name, changed in mismatch_cases.items():
            with self.subTest(resume_mismatch=name):
                output = self.root / f"strict-{name}.bin"
                state = self.root / f"strict-{name}.state"
                command = [
                    str(TOOL), "train-control", "--out", str(output),
                    "--control-state-in", str(part_state),
                    "--control-state-out", str(state),
                    "--base-net", str(part_net), "--matches", "1",
                    "--match-start", "1",
                ] + common + changed
                failed = subprocess.run(
                    command, cwd=ROOT, text=True, capture_output=True,
                    check=False,
                )
                self.assertEqual(failed.returncode, 1)
                self.assertIn("resume provenance mismatch", failed.stderr)
                self.assertFalse(output.exists())

        wrong_input = self.root / "strict-input.bin"
        failed_input = subprocess.run(
            [str(TOOL), "train-control", "--out", str(wrong_input),
             "--control-state-in", str(part_state),
             "--control-state-out", str(self.root / "strict-input.state"),
             "--base-net", str(ACTOR), "--matches", "1",
             "--match-start", "1"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(failed_input.returncode, 1)
        self.assertIn("resume provenance mismatch", failed_input.stderr)
        self.assertFalse(wrong_input.exists())

        # Forge the legacy 64-bit net fingerprint to match the wrong input.
        # The v3 checkpoint SHA must independently reject this handoff.
        forged_state = bytearray(part_state.read_bytes())
        forged_state[24:32] = struct.pack(
            "<Q", int(part_report["input_net_fingerprint"], 16)
        )
        forged_state_path = self.root / "strict-forged-fingerprint.state"
        forged_state_path.write_bytes(forged_state)
        forged_output = self.root / "strict-forged-fingerprint.bin"
        forged = subprocess.run(
            [str(TOOL), "train-control", "--out", str(forged_output),
             "--control-state-in", str(forged_state_path),
             "--control-state-out",
             str(self.root / "strict-forged-fingerprint-out.state"),
             "--base-net", str(ACTOR), "--matches", "1",
             "--match-start", "1"] + common,
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(forged.returncode, 1)
        self.assertIn("resume provenance mismatch", forged.stderr)
        self.assertFalse(forged_output.exists())


if __name__ == "__main__":
    unittest.main()
