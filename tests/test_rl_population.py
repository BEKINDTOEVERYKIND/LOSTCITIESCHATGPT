#!/usr/bin/env python3
"""End-to-end contracts for conservative PPO generation modes."""

from __future__ import annotations

import re
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PopulationTrainingTest(unittest.TestCase):
    def test_continuation_state_start_contract_and_determinism(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-continuation-") as tmp:
            first = Path(tmp) / "first.bin"
            second = Path(tmp) / "second.bin"
            command = [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--out", str(first),
                "--continuation-start", "14",
                "--continuation-root", str(source),
                "--iters", "1",
                "--games", "2",
                "--rounds", "1",
                "--threads", "1",
                "--epochs", "1",
                "--batch", "32",
                "--vcoef", "1",
                "--ent", "0",
                "--wd", "0",
                "--seed", "20260822",
            ]
            run = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=True
            )
            contract = re.search(
                r"continuation roots (\d+): baseline (\d+), challenger "
                r"(\d+); first actor round ply (\d+); root actor rows (\d+); "
                r"exact deck-one moves (\d+); cycle-forced moves (\d+); "
                r"cap-reserve moves (\d+); "
                r"fingerprint ([0-9a-f]{16})",
                run.stdout,
            )
            self.assertIsNotNone(contract, run.stdout)
            (roots, baseline, challenger, first_ply, root_rows, exact,
             cycle_forces, cap_forces, fingerprint) = contract.groups()
            self.assertEqual(int(roots), 2)
            self.assertEqual(int(baseline) + int(challenger), 2)
            self.assertEqual(int(first_ply), 15)
            self.assertEqual(int(root_rows), 0)
            self.assertEqual(int(exact), 2)
            self.assertGreaterEqual(int(cycle_forces), 0)
            self.assertEqual(int(cap_forces), 0)
            self.assertRegex(run.stdout, r"lambda 1\.00")

            initial_guard = re.search(
                r"immutable champion fingerprint: ([0-9a-f]{16})", run.stdout
            )
            final_guard = re.search(
                r"immutable champion verified ([0-9a-f]{16})", run.stdout
            )
            self.assertIsNotNone(initial_guard, run.stdout)
            self.assertIsNotNone(final_guard, run.stdout)
            self.assertEqual(initial_guard.group(1), final_guard.group(1))

            repeated = command.copy()
            repeated[repeated.index(str(first))] = str(second)
            repeated.extend(["--lambda", "1"])
            second_run = subprocess.run(
                repeated, cwd=ROOT, text=True, capture_output=True, check=True
            )
            second_contract = re.search(
                r"continuation roots (\d+): baseline (\d+), challenger "
                r"(\d+); first actor round ply (\d+); root actor rows (\d+); "
                r"exact deck-one moves (\d+); cycle-forced moves (\d+); "
                r"cap-reserve moves (\d+); "
                r"fingerprint ([0-9a-f]{16})",
                second_run.stdout,
            )
            self.assertIsNotNone(second_contract, second_run.stdout)
            self.assertEqual(second_contract.groups(), contract.groups())
            self.assertEqual(second.read_bytes(), first.read_bytes())
            self.assertEqual(second_contract.group(9), fingerprint)

            # The learner may resume from an unrelated checkpoint without
            # changing the separately loaded root actor.
            pinned = Path(tmp) / "pinned.bin"
            pinned_command = command.copy()
            pinned_command[pinned_command.index(str(source))] = str(
                ROOT / "data" / "best.bin"
            )
            pinned_command[pinned_command.index(str(first))] = str(pinned)
            pinned_command[pinned_command.index("--epochs") + 1] = "0"
            pinned_run = subprocess.run(
                pinned_command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            pinned_guard = re.search(
                r"immutable champion fingerprint: ([0-9a-f]{16})",
                pinned_run.stdout,
            )
            self.assertIsNotNone(pinned_guard, pinned_run.stdout)
            self.assertEqual(pinned_guard.group(1), initial_guard.group(1))

    def test_explicitly_disabled_continuation_preserves_legacy_ppo(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-continuation-off-") as tmp:
            implicit = Path(tmp) / "implicit.bin"
            explicit = Path(tmp) / "explicit.bin"
            command = [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--out", str(implicit),
                "--iters", "1",
                "--games", "2",
                "--rounds", "1",
                "--threads", "1",
                "--epochs", "1",
                "--batch", "32",
                "--vcoef", "0",
                "--bw", "0",
                "--ent", "0",
                "--wd", "0",
                "--eval", "0",
                "--seed", "20260823",
            ]
            implicit_run = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=True
            )
            self.assertRegex(implicit_run.stdout, r"lambda 0\.85")
            explicit_command = command.copy()
            explicit_command[explicit_command.index(str(implicit))] = str(explicit)
            explicit_command.extend(
                ["--continuation-start", "0", "--lambda", "0.85"]
            )
            subprocess.run(
                explicit_command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(implicit.read_bytes(), explicit.read_bytes())

    def test_continuation_root_is_loaded_byte_exact(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-raw-root-") as tmp:
            asymmetric = Path(tmp) / "asymmetric.bin"
            payload = bytearray(source.read_bytes())
            magic, feat_dim, h1, h2, nplay, version = struct.unpack(
                "=6I", payload[:24]
            )
            self.assertEqual((magic, version), (0x4C435651, 6))
            bplay_float = (
                feat_dim * h1 + h1 + h1 * h2 + h2 + h2 + 1
                + nplay * h2
            )
            bplay_offset = 24 + bplay_float * 4
            value = struct.unpack_from("=f", payload, bplay_offset)[0]
            struct.pack_into("=f", payload, bplay_offset, value + 0.375)
            asymmetric.write_bytes(payload)

            fingerprint = 1469598103934665603
            for byte in payload[24:]:
                fingerprint ^= byte
                fingerprint = (fingerprint * 1099511628211) & ((1 << 64) - 1)

            output = Path(tmp) / "unused.bin"
            run = subprocess.run(
                [
                    str(ROOT / "bin" / "rl"),
                    "--init", str(source),
                    "--out", str(output),
                    "--continuation-start", "14",
                    "--continuation-root", str(asymmetric),
                    "--iters", "0",
                    "--games", "1",
                    "--rounds", "1",
                    "--threads", "1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            guard = re.search(
                r"immutable champion fingerprint: ([0-9a-f]{16})",
                run.stdout,
            )
            self.assertIsNotNone(guard, run.stdout)
            self.assertEqual(guard.group(1), f"{fingerprint:016x}")

    def test_continuation_output_cannot_alias_protected_checkpoints(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-alias-") as tmp:
            directory = Path(tmp)
            protected = directory / "protected.bin"
            anchor = directory / "anchor.bin"
            shutil.copyfile(source, protected)
            shutil.copyfile(source, anchor)
            original = protected.read_bytes()

            symlink = directory / "symlink.bin"
            symlink.symlink_to(protected)
            hardlink = directory / "hardlink.bin"
            os.link(protected, hardlink)
            generated_base = directory / "generated.bin"
            os.link(protected, Path(f"{generated_base}.it1"))
            subdir = directory / "subdir"
            subdir.mkdir()
            canonical_alias = subdir / ".." / protected.name

            cases = [
                (protected, None),
                (symlink, None),
                (hardlink, None),
                (generated_base, None),
                (canonical_alias, None),
                (anchor, anchor),
            ]
            for output, anchor_path in cases:
                with self.subTest(output=output, anchor=anchor_path):
                    command = [
                        str(ROOT / "bin" / "rl"),
                        "--init", str(protected),
                        "--out", str(output),
                        "--continuation-start", "14",
                        "--continuation-root", str(protected),
                        "--iters", "1",
                    ]
                    if anchor_path:
                        command.extend(["--anchor", str(anchor_path)])
                    run = subprocess.run(
                        command, cwd=ROOT, text=True, capture_output=True
                    )
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("aliases protected checkpoint", run.stderr)
            self.assertEqual(protected.read_bytes(), original)

    def test_continuation_rejects_nonproduction_start(self) -> None:
        source = ROOT / "data" / "champion.bin"
        run = subprocess.run(
            [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--continuation-start", "13",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("must be 0 or 14", run.stderr)

    def test_continuation_requires_independent_root_checkpoint(self) -> None:
        source = ROOT / "data" / "champion.bin"
        run = subprocess.run(
            [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--continuation-start", "14",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("requires --continuation-root PATH", run.stderr)

    def test_population_training_rejects_unbalanced_game_count(self) -> None:
        source = ROOT / "data" / "champion.bin"
        run = subprocess.run(
            [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--gen-opponent", f"policy:{source}:0:20",
                "--opponent-mix", "0.5",
                "--games", "3",
                "--eval", "0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("requires an even --games count", run.stderr)

    def test_balanced_frozen_opponent_and_v6_only_bytes(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-test-") as tmp:
            candidate = Path(tmp) / "candidate.bin"
            command = [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--gen-opponent", f"policy:{source}:0:20",
                "--opponent-mix", "1",
                "--v6-only",
                "--iters", "1",
                "--games", "6",
                "--rounds", "1",
                "--threads", "2",
                "--epochs", "1",
                "--batch", "256",
                "--lr", "0.0001",
                "--vcoef", "0",
                "--bw", "0",
                "--ent", "0",
                "--wd", "0",
                "--eval", "0",
                "--seed", "20260802",
                "--out", str(candidate),
            ]
            run = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=True
            )
            match = re.search(
                r"frozen-opponent games (\d+), learner seats (\d+)/(\d+)",
                run.stdout,
            )
            self.assertIsNotNone(match, run.stdout)
            self.assertEqual(tuple(map(int, match.groups())), (6, 3, 3))

            before = source.read_bytes()
            after = candidate.read_bytes()
            self.assertEqual(len(before), len(after))
            magic, feat_dim, h1, h2, nplay, version = struct.unpack(
                "=6I", before[:24]
            )
            self.assertEqual((magic, version), (0x4C435651, 6))
            self.assertEqual((h1, h2, nplay), (512, 256, 120))

            float_bytes = 4
            legacy_dim = 556
            cards = 60
            draws = 6
            header = 24
            legacy_w1_end = header + legacy_dim * h1 * float_bytes
            b1_start = header + feat_dim * h1 * float_bytes
            middle_floats = (
                h1 + h1 * h2 + h2 + h2 + 1
                + nplay * h2 + nplay
                + draws * h2 + draws
                + cards * h2 + cards
            )
            wcomb_start = b1_start + middle_floats * float_bytes

            # v6-only may change appended pile rows and the complete-move
            # residual, but every inherited parameter must remain exact.
            self.assertEqual(before[:legacy_w1_end], after[:legacy_w1_end])
            self.assertEqual(before[b1_start:wcomb_start], after[b1_start:wcomb_start])
            self.assertNotEqual(before[legacy_w1_end:b1_start],
                                after[legacy_w1_end:b1_start])
            self.assertNotEqual(before[wcomb_start:], after[wcomb_start:])

    def test_centered_critic_and_exact_k_belief_smoke(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-centered-test-") as tmp:
            candidate = Path(tmp) / "candidate.bin"
            command = [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--out", str(candidate),
                "--iters", "1",
                "--games", "2",
                "--rounds", "1",
                "--threads", "1",
                "--epochs", "1",
                "--batch", "32",
                "--vcoef", "1",
                "--bw", "1",
                "--ent", "0",
                "--wd", "0",
                "--eval", "0",
                "--seed", "4242",
            ]
            run = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            loss = re.search(
                r"belief exact-K nll/card ([0-9]+(?:\.[0-9]+)?)",
                run.stdout,
            )
            self.assertIsNotNone(loss, run.stdout)
            self.assertGreater(float(loss.group(1)), 0.0)

            before = source.read_bytes()
            after = candidate.read_bytes()
            _, feat_dim, h1, h2, _, _ = struct.unpack("=6I", before[:24])
            # b3 follows w1, b1, w2, b2 and w3 in the raw v6 payload.
            b3_offset = 24 + (
                feat_dim * h1 + h1 + h1 * h2 + h2 + h2
            ) * 4
            self.assertEqual(
                before[b3_offset:b3_offset + 4],
                after[b3_offset:b3_offset + 4],
                "the coupled +d/-d critic gradient must cancel common b3",
            )

            explicit_off = Path(tmp) / "explicit-off.bin"
            off_command = command.copy()
            off_command[off_command.index(str(candidate))] = str(explicit_off)
            off_command.extend(["--trajectory-symmetries", "0"])
            subprocess.run(off_command, cwd=ROOT, text=True,
                           capture_output=True, check=True)
            self.assertEqual(
                after,
                explicit_off.read_bytes(),
                "explicitly disabled trajectory augmentation changed defaults",
            )

    def test_augmented_population_keeps_opponent_policy_masked(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-augment-test-") as tmp:
            candidate = Path(tmp) / "candidate.bin"
            command = [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--out", str(candidate),
                "--gen-opponent", f"policy:{source}:0:20",
                "--opponent-mix", "1",
                "--trajectory-symmetries", "20",
                "--v6-only",
                "--iters", "1",
                "--games", "2",
                "--rounds", "1",
                "--threads", "2",
                "--epochs", "1",
                "--batch", "32",
                "--vcoef", "0",
                "--bw", "0",
                "--ent", "0",
                "--wd", "0",
                "--eval", "0",
                "--seed", "20260803",
            ]
            run = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("trajectory suit augmentation: exact group 20",
                          run.stdout)
            rows = re.search(r"policy-gradient rows (\d+)/(\d+)", run.stdout)
            self.assertIsNotNone(rows, run.stdout)
            actor_rows, all_rows = map(int, rows.groups())
            self.assertGreater(actor_rows, 0)
            # Every ply supplies two perspective rows.  If frozen-opponent
            # decisions leaked into PPO, actor_rows would equal all_rows/2;
            # correctly masked alternating learner decisions are near 1/4.
            self.assertLess(5 * actor_rows, 2 * all_rows)

            one_thread = Path(tmp) / "one-thread.bin"
            one_command = command.copy()
            one_command[one_command.index(str(candidate))] = str(one_thread)
            one_command[one_command.index("--threads") + 1] = "1"
            one_run = subprocess.run(
                one_command, cwd=ROOT, text=True, capture_output=True,
                check=True,
            )
            fingerprint = re.search(
                r"augmentation fingerprint ([0-9a-f]{16})", run.stdout
            )
            one_fingerprint = re.search(
                r"augmentation fingerprint ([0-9a-f]{16})", one_run.stdout
            )
            self.assertIsNotNone(fingerprint, run.stdout)
            self.assertIsNotNone(one_fingerprint, one_run.stdout)
            self.assertEqual(fingerprint.group(1), one_fingerprint.group(1))

    def test_belief_only_freezes_every_other_model_byte(self) -> None:
        source = ROOT / "data" / "champion.bin"
        with tempfile.TemporaryDirectory(prefix="lc-rl-belief-only-") as tmp:
            asymmetric = Path(tmp) / "asymmetric.bin"
            candidate = Path(tmp) / "candidate.bin"
            asymmetric_bytes = bytearray(source.read_bytes())
            # Deliberately break a non-belief wager-row symmetry.  Belief-only
            # setup must not silently project this trunk byte before training.
            first_w1 = struct.unpack_from("=f", asymmetric_bytes, 24)[0]
            struct.pack_into("=f", asymmetric_bytes, 24, first_w1 + 0.12345)
            asymmetric.write_bytes(asymmetric_bytes)
            run = subprocess.run(
                [
                    str(ROOT / "bin" / "rl"),
                    "--init", str(asymmetric),
                    "--out", str(candidate),
                    "--belief-only",
                    "--trajectory-symmetries", "20",
                    "--iters", "1",
                    "--games", "2",
                    "--rounds", "1",
                    "--threads", "1",
                    "--epochs", "1",
                    "--batch", "32",
                    "--bw", "1",
                    "--wd", "0.001",
                    "--eval", "0",
                    "--seed", "83003",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertRegex(
                run.stdout,
                r"belief-head-only updates .*exact-K nll/card [0-9.]+",
            )

            before = asymmetric.read_bytes()
            after = candidate.read_bytes()
            self.assertEqual(len(before), len(after))
            _, feat_dim, h1, h2, nplay, _ = struct.unpack("=6I", before[:24])
            cards, draws, header = 60, 6, 24
            prefix_floats = (
                feat_dim * h1 + h1 + h1 * h2 + h2 + h2 + 1
                + nplay * h2 + nplay + draws * h2 + draws
            )
            belief_start = header + prefix_floats * 4
            belief_end = belief_start + (cards * h2 + cards) * 4
            self.assertEqual(before[:belief_start], after[:belief_start])
            self.assertNotEqual(before[belief_start:belief_end],
                                after[belief_start:belief_end])
            self.assertEqual(before[belief_end:], after[belief_end:])

    def test_new_mode_validation(self) -> None:
        source = ROOT / "data" / "champion.bin"
        cases = [
            (["--trajectory-symmetries", "3"],
             "--trajectory-symmetries must be"),
            (["--trajectory-symmetries", "not-a-number"],
             "--trajectory-symmetries must be"),
            (["--belief-only", "--bw", "0"],
             "--belief-only requires --bw greater than zero"),
            (["--belief-only", "--v6-only"],
             "--belief-only and --v6-only are mutually exclusive"),
            (["--belief-only", "--anchor", str(source), "--kl", "0.1"],
             "--belief-only cannot optimize an anchor KL"),
            (["--lambda", "nan"],
             "--lambda must be finite and between zero and one"),
            (["--lambda", "-0.1"],
             "--lambda must be finite and between zero and one"),
            (["--lambda", "1.1"],
             "--lambda must be finite and between zero and one"),
        ]
        for args, error in cases:
            with self.subTest(args=args):
                run = subprocess.run(
                    [str(ROOT / "bin" / "rl"), "--init", str(source),
                     "--eval", "0", *args],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(error, run.stderr)

        continuation = subprocess.run(
            [
                str(ROOT / "bin" / "rl"),
                "--init", str(source),
                "--continuation-start", "14",
                "--continuation-root", str(source),
                "--lambda", "0.85",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(continuation.returncode, 0)
        self.assertIn("requires --lambda 1 exactly", continuation.stderr)


if __name__ == "__main__":
    unittest.main()
