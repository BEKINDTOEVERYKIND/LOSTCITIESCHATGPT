#!/usr/bin/env python3
"""Contracts for deterministic, safe Net checkpoint averaging."""

from __future__ import annotations

import itertools
import hashlib
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "bin" / "net_average"
SOURCE = ROOT / "data" / "champion.bin"


def checkpoint_with_parameter(source: Path, output: Path, value: float) -> None:
    checkpoint_with_parameters(source, output, {0: value})


def checkpoint_with_parameters(
    source: Path, output: Path, values: dict[int, float]
) -> None:
    data = bytearray(source.read_bytes())
    magic, _, _, _, _, version = struct.unpack_from("=6I", data, 0)
    if (magic, version) != (0x4C435651, 6):
        raise AssertionError("fixture requires a current v6 checkpoint")
    for parameter, value in values.items():
        struct.pack_into("=f", data, 24 + 4 * parameter, value)
    output.write_bytes(data)


class NetAverageTest(unittest.TestCase):
    def run_tool(self, output: Path, *inputs: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(TOOL), str(output), *(str(path) for path in inputs)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def test_identical_inputs_are_byte_exact_including_negative_zero(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-net-average-identical-") as tmp:
            directory = Path(tmp)
            negative_zero = directory / "negative-zero.bin"
            output = directory / "average.bin"
            checkpoint_with_parameter(SOURCE, negative_zero, -0.0)
            run = self.run_tool(output, negative_zero, negative_zero)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(output.read_bytes(), negative_zero.read_bytes())
            self.assertIn("identical_fast_path=1", run.stdout)
            self.assertIn("equal_weight=2", run.stdout)

    def test_double_average_is_bitwise_argument_permutation_invariant(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-net-average-order-") as tmp:
            directory = Path(tmp)
            paths = [directory / name for name in ("a.bin", "b.bin", "c.bin")]
            # Binary32 accumulation in canonical a,b,c order loses the middle
            # +1.  Binary64 retains it, yielding exactly binary32(1/3).
            subnormal = struct.unpack("=f", b"\x01\x00\x00\x00")[0]
            float_max = struct.unpack("=f", b"\xff\xff\x7f\x7f")[0]
            for path, values in zip(paths, (
                {0: 16777216.0, 1: subnormal, 2: float_max},
                {0: 1.0, 1: subnormal, 2: -float_max},
                {0: -16777216.0, 1: subnormal, 2: float_max},
            ), strict=True):
                checkpoint_with_parameters(SOURCE, path, values)

            reference: bytes | None = None
            reference_input_order: list[str] | None = None
            for number, permutation in enumerate(itertools.permutations(paths)):
                output = directory / f"average-{number}.bin"
                run = self.run_tool(output, *permutation)
                self.assertEqual(run.returncode, 0, run.stderr)
                data = output.read_bytes()
                if reference is None:
                    reference = data
                    self.assertEqual(
                        data[24:28], struct.pack("=f", 1.0 / 3.0)
                    )
                    self.assertEqual(data[28:32], struct.pack("=f", subnormal))
                    self.assertEqual(
                        data[32:36], struct.pack("=f", float_max / 3.0)
                    )
                    self.assertIn(
                        "ordering=model_fnv1a_then_bytes_then_path ", run.stdout
                    )
                    self.assertIn(
                        "accumulation=binary64_sequential", run.stdout
                    )
                    input_lines = re.findall(
                        r"^input\[\d+\]=([^ ]+).*$", run.stdout, re.MULTILINE
                    )
                    self.assertEqual(set(input_lines), {str(p.resolve()) for p in paths})
                    reference_input_order = input_lines
                    self.assertRegex(run.stdout, r"file_fnv1a=[0-9a-f]{16}")
                    self.assertRegex(run.stdout, r"model_fnv1a=[0-9a-f]{16}")
                    for path in paths:
                        expected = hashlib.sha256(path.read_bytes()).hexdigest()
                        line = next(
                            item for item in run.stdout.splitlines()
                            if re.match(r"input\[\d+\]=" + re.escape(str(path.resolve())) + r" ", item)
                        )
                        self.assertIn(f"sha256={expected}", line)
                    output_line = next(
                        item for item in run.stdout.splitlines()
                        if item.startswith("output=")
                    )
                    self.assertIn(
                        f"sha256={hashlib.sha256(data).hexdigest()}", output_line
                    )
                else:
                    self.assertEqual(data, reference)
                    input_lines = re.findall(
                        r"^input\[\d+\]=([^ ]+).*$", run.stdout, re.MULTILINE
                    )
                    self.assertEqual(input_lines, reference_input_order)
            self.assertEqual(list(directory.glob("*.tmp.*")), [])

    def test_rejects_nonfinite_and_malformed_inputs_without_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-net-average-invalid-") as tmp:
            directory = Path(tmp)
            valid = directory / "valid.bin"
            shutil.copyfile(SOURCE, valid)
            invalid: list[tuple[str, Path, str]] = []

            nan = directory / "nan.bin"
            checkpoint_with_parameter(SOURCE, nan, float("nan"))
            invalid.append(("nan", nan, "non-finite parameter 0"))
            infinity = directory / "infinity.bin"
            checkpoint_with_parameter(SOURCE, infinity, float("inf"))
            invalid.append(("infinity", infinity, "non-finite parameter 0"))
            negative_infinity = directory / "negative-infinity.bin"
            last_parameter = (len(SOURCE.read_bytes()) - 24) // 4 - 1
            checkpoint_with_parameters(
                SOURCE, negative_infinity, {last_parameter: float("-inf")}
            )
            invalid.append((
                "negative-infinity", negative_infinity,
                f"non-finite parameter {last_parameter}",
            ))
            bad_magic = directory / "bad-magic.bin"
            bad_magic_data = bytearray(SOURCE.read_bytes())
            struct.pack_into("=I", bad_magic_data, 0, 0)
            bad_magic.write_bytes(bad_magic_data)
            invalid.append(("bad-magic", bad_magic, "malformed checkpoint"))
            bad_version = directory / "bad-version.bin"
            bad_version_data = bytearray(SOURCE.read_bytes())
            struct.pack_into("=I", bad_version_data, 20, 99)
            bad_version.write_bytes(bad_version_data)
            invalid.append(("bad-version", bad_version, "malformed checkpoint"))
            truncated = directory / "truncated.bin"
            truncated.write_bytes(SOURCE.read_bytes()[:-1])
            invalid.append(("truncated", truncated, "malformed checkpoint"))
            trailing = directory / "trailing.bin"
            trailing.write_bytes(SOURCE.read_bytes() + b"x")
            invalid.append(("trailing", trailing, "malformed checkpoint"))

            for label, bad, message in invalid:
                with self.subTest(label=label):
                    output = directory / f"{label}-output.bin"
                    sentinel = f"keep-{label}".encode()
                    output.write_bytes(sentinel)
                    run = self.run_tool(output, valid, bad)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn(message, run.stderr)
                    self.assertEqual(output.read_bytes(), sentinel)
            self.assertEqual(list(directory.glob("*.tmp.*")), [])

    def test_rejects_direct_canonical_symlink_and_hardlink_output_aliases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-net-average-alias-") as tmp:
            directory = Path(tmp)
            source = directory / "source.bin"
            shutil.copyfile(SOURCE, source)
            original = source.read_bytes()

            symlink = directory / "symlink.bin"
            symlink.symlink_to(source)
            hardlink = directory / "hardlink.bin"
            os.link(source, hardlink)
            subdirectory = directory / "subdirectory"
            subdirectory.mkdir()
            canonical = subdirectory / ".." / source.name
            parent_link = directory / "parent-link"
            parent_link.symlink_to(directory, target_is_directory=True)
            symlinked_parent = parent_link / source.name

            for label, output in (
                ("direct", source),
                ("canonical", canonical),
                ("symlink", symlink),
                ("hardlink", hardlink),
                ("symlinked-parent", symlinked_parent),
            ):
                with self.subTest(label=label):
                    run = self.run_tool(output, source, source)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("aliases input checkpoint", run.stderr)
                    self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list(directory.glob("*.tmp.*")), [])

    def test_requires_at_least_two_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lc-net-average-usage-") as tmp:
            output = Path(tmp) / "average.bin"
            run = self.run_tool(output, SOURCE)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("INPUT1.bin INPUT2.bin", run.stderr)
            self.assertFalse(output.exists())
            zero = subprocess.run(
                [str(TOOL), str(output)], cwd=ROOT, text=True,
                capture_output=True,
            )
            self.assertNotEqual(zero.returncode, 0)
            self.assertIn("INPUT1.bin INPUT2.bin", zero.stderr)


if __name__ == "__main__":
    unittest.main()
