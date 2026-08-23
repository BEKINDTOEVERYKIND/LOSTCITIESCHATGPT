"""Synthetic, efficacy-free tests for the offline world800 archiver."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

from tools import archive_world800 as archive
from tools.merge_arena import (
    EvidenceError,
    _combine_reciprocal,
    _remerge_recorded_block,
    merge_block,
)
from tools.match_value_campaign import _world_result
from tools.validate_actor_shards import validate_shards


ROOT = Path(__file__).resolve().parents[1]


def encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def compact(value: dict) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode()


class SyntheticEvidence:
    """Construct a complete fake campaign transport with no measured games."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, bytes] = {}
        self.modes: dict[str, int] = {}
        self.arena = b"synthetic evaluator; never executed\n"
        self.arena_sha = hashlib.sha256(self.arena).hexdigest()
        self.provenance = archive._expected_provenance(self.arena_sha)
        self._bindings_and_evaluator()
        self._raw_shards()
        self._merged_results()
        self._sha_manifest()

    def _add(self, path: str, value: bytes, permission: int = 0o644) -> None:
        self.files[path] = value
        self.modes[path] = stat.S_IFREG | permission

    def _bindings_and_evaluator(self) -> None:
        for archive_path, (repository_path, _) in archive.BOUND_FILES.items():
            self._add(archive_path, repository_path.read_bytes())
        model = (ROOT / "data/champion.bin").read_bytes()
        self._add("evaluator/arena", self.arena, 0o755)
        self._add("evaluator/champion.bin", model)
        build = "\n".join([
            f"launch_commit={archive.LAUNCH_COMMIT}",
            f"source_parent_commit={archive.LAUNCH_PARENT}",
            f"execution_sha={archive.EXECUTION_SHA}",
            f"source_commit={archive.SOURCE_COMMIT}",
            f"source_tree={archive.SOURCE_TREE}",
            f"plan_sha={archive.PLAN_SHA}",
            f"parent_result_sha={archive.PARENT_RESULT_SHA}",
            "compiler=gcc (synthetic test compiler) 1.0",
            "cflags=-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
            "-Wall -Wextra -std=c11",
            "ldflags=-lm -pthread",
            "threads_per_shard=4",
            f"arena_size={len(self.arena)}",
            f"model_size={len(model)}",
            "Linux synthetic-runner 1.0 #1 SMP x86_64 GNU/Linux",
            f"{self.arena_sha}  evaluator/arena",
            f"{archive.MODEL_SHA}  evaluator/champion.bin",
        ]) + "\n"
        self._add("evaluator/BUILD_INFO.txt", build.encode("ascii"))

    def _raw_shards(self) -> None:
        for orientation in ("candidate-first", "baseline-first"):
            if orientation == "candidate-first":
                agent_a, agent_b = archive.CANDIDATE, archive.BASELINE
                seed = archive.CANDIDATE_SEED
                scores_a, scores_b = [1, 1], [0, 0]
            else:
                agent_a, agent_b = archive.BASELINE, archive.CANDIDATE
                seed = archive.BASELINE_SEED
                scores_a, scores_b = [0, 0], [1, 1]
            for start in archive.STARTS:
                stem = f"{orientation}-{start}"
                records = [{
                    "record": "meta", "schema": 1, "seed": seed,
                    "pair_start": str(start),
                    "pair_count": archive.PAIRS_PER_SHARD,
                    "rounds": archive.ROUNDS,
                    "agent_a": agent_a, "agent_b": agent_b,
                    "provenance": self.provenance,
                }]
                for index in range(start, start + archive.PAIRS_PER_SHARD):
                    records.append({
                        "record": "pair", "index": str(index),
                        "score_a": scores_a, "score_b": scores_b,
                        "plies": [3, 3], "capped_rounds": [0, 0],
                    })
                records.append({
                    "record": "complete", "pairs": archive.PAIRS_PER_SHARD,
                })
                raw = b"".join(compact(record) for record in records)
                digest = hashlib.sha256(raw).hexdigest()
                self._add(f"downloads/{stem}.jsonl", raw)
                self._add(
                    f"downloads/{stem}.sha256",
                    f"{digest}  raw/{stem}.jsonl\n".encode("ascii"),
                )
                self._add(
                    f"downloads/{stem}.time",
                    b"wall_s=1.00 user_s=2.00 sys_s=0.10 "
                    b"max_rss_kb=1024 exit=0\n",
                )

    def _merged_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="synthetic-world800-") as tmp:
            stage = Path(tmp)
            for name, value in self.files.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            previous = Path.cwd()
            os.chdir(stage)
            try:
                structural = validate_shards(
                    Path("downloads"), archive.CANDIDATE, archive.BASELINE,
                    self.provenance, archive.CANDIDATE_SEED,
                    archive.BASELINE_SEED, archive.STARTS,
                    archive.PAIRS_PER_SHARD, archive.ROUNDS,
                )
                first = merge_block([
                    Path(f"downloads/candidate-first-{start}.jsonl")
                    for start in archive.STARTS
                ], 0, archive.PAIRS_PER_ORIENTATION)
                second = merge_block([
                    Path(f"downloads/baseline-first-{start}.jsonl")
                    for start in archive.STARTS
                ], 0, archive.PAIRS_PER_ORIENTATION)
                first_bytes, second_bytes = encoded(first), encoded(second)
                raw_validation = {
                    "status": "validated",
                    "method": (
                        "reopened, SHA-256 checked, and exactly remerged "
                        "recorded raw inputs"),
                    "blocks": [
                        _remerge_recorded_block(first, "first"),
                        _remerge_recorded_block(second, "second"),
                    ],
                }
                reciprocal = _combine_reciprocal(
                    first, second, [
                        {"path": "merged/candidate-first.json",
                         "sha256": hashlib.sha256(first_bytes).hexdigest()},
                        {"path": "merged/baseline-first.json",
                         "sha256": hashlib.sha256(second_bytes).hexdigest()},
                    ], gate_z=archive.GATE_Z, require_positive_margin=True,
                    raw_input_validation=raw_validation,
                )
            finally:
                os.chdir(previous)
        for name, value in (
            ("merged/structural-validation.json", structural),
            ("merged/candidate-first.json", first),
            ("merged/baseline-first.json", second),
            ("merged/reciprocal.json", reciprocal),
        ):
            self._add(name, encoded(value))

    def _sha_manifest(self) -> None:
        lines = []
        for name in sorted(archive.EXPECTED_NAMES - {"merged/SHA256SUMS.txt"}):
            lines.append(f"{hashlib.sha256(self.files[name]).hexdigest()}  {name}\n")
        self._add("merged/SHA256SUMS.txt", "".join(lines).encode("ascii"))

    def zip_bytes(
            self, files: dict[str, bytes] | None = None,
            modes: dict[str, int] | None = None,
            duplicate: str | None = None) -> bytes:
        selected = self.files if files is None else files
        selected_modes = self.modes if modes is None else modes
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as stream:
            for name in sorted(selected):
                info = zipfile.ZipInfo(name, (2026, 8, 23, 0, 0, 0))
                info.create_system = 3
                info.external_attr = selected_modes[name] << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                stream.writestr(info, selected[name])
            if duplicate is not None:
                info = zipfile.ZipInfo(duplicate, (2026, 8, 23, 0, 0, 0))
                info.create_system = 3
                info.external_attr = selected_modes[duplicate] << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    stream.writestr(info, selected[duplicate])
        return output.getvalue()

    @staticmethod
    def refresh_manifest(files: dict[str, bytes]) -> None:
        lines = []
        for name in sorted(archive.EXPECTED_NAMES - {"merged/SHA256SUMS.txt"}):
            lines.append(f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n")
        files["merged/SHA256SUMS.txt"] = "".join(lines).encode("ascii")


class ArchiveWorld800Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_tmp = tempfile.TemporaryDirectory(prefix="world800-fixture-")
        cls.fixture = SyntheticEvidence(Path(cls.fixture_tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture_tmp.cleanup()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="world800-test-")
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_inputs(self, snapshot: bytes) -> tuple[Path, Path, Path]:
        zip_path = self.root / archive.INPUT_ZIP_NAME
        zip_path.write_bytes(snapshot)
        digest = hashlib.sha256(snapshot).hexdigest()
        run_path = self.root / "run.json"
        run_path.write_text(json.dumps({
            "repository": archive.REPOSITORY,
            "run_id": archive.RUN_ID,
            "run_attempt": archive.RUN_ATTEMPT,
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            "head_branch": archive.LAUNCH_BRANCH,
            "head_sha": archive.LAUNCH_COMMIT,
        }), encoding="utf-8")
        artifact_path = self.root / "artifact.json"
        artifact_path.write_text(json.dumps({
            "artifact_id": 9999999999,
            "name": archive.ARTIFACT_NAME,
            "size_in_bytes": len(snapshot),
            "digest": "sha256:" + digest,
            "expired": False,
            "workflow_run_id": archive.RUN_ID,
            "workflow_run_attempt": archive.RUN_ATTEMPT,
            "workflow_run_head_sha": archive.LAUNCH_COMMIT,
        }), encoding="utf-8")
        return zip_path, run_path, artifact_path

    def outputs(self) -> tuple[Path, Path, Path]:
        return (
            self.root / "world800_result.json",
            self.root / "world800_archive_manifest.json",
            self.root / "preserved.zip",
        )

    def invoke(self, snapshot: bytes) -> dict:
        zip_path, run_path, artifact_path = self.write_inputs(snapshot)
        result, manifest, preserved = self.outputs()
        return archive.archive_world800(
            zip_path, run_path, artifact_path, result, manifest, preserved)

    def test_accepts_complete_synthetic_archive_and_preserves_exact_bytes(self) -> None:
        locked_before = hashlib.sha256(
            (ROOT / "data/experiments/locked_world800_execution.json").read_bytes()
        ).hexdigest()
        snapshot = self.fixture.zip_bytes()
        manifest = self.invoke(snapshot)
        result, manifest_path, preserved = self.outputs()
        self.assertEqual(preserved.read_bytes(), snapshot)
        self.assertEqual(
            result.read_bytes(), self.fixture.files["merged/reciprocal.json"])
        _, passed, selected_actor, selected_worlds = _world_result(result)
        self.assertTrue(passed)
        self.assertEqual(selected_actor, archive.CANDIDATE)
        self.assertEqual(selected_worlds, 800)
        self.assertEqual(json.loads(manifest_path.read_text()), manifest)
        self.assertTrue(manifest["gate"]["promotion_gate_passed"])
        self.assertEqual(manifest["gate"]["selected_worlds"], 800)
        self.assertEqual(manifest["preserved_archive"]["regular_file_entries"], 164)
        self.assertEqual(
            manifest["preserved_archive"]["source_sha256_manifest_entries"],
            163,
        )
        self.assertTrue(
            manifest["verification"]
            ["all_four_json_outputs_independently_recomputed_byte_for_byte"]
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "data/experiments/locked_world800_execution.json")
                .read_bytes()
            ).hexdigest(),
            locked_before,
        )

    def test_rejects_duplicate_unsafe_missing_extra_and_symlink_entries(self) -> None:
        cases: list[tuple[str, bytes]] = []
        cases.append((
            "duplicate",
            self.fixture.zip_bytes(duplicate="evaluator/arena"),
        ))
        extra_files = dict(self.fixture.files)
        extra_modes = dict(self.fixture.modes)
        extra_files["unexpected.txt"] = b"extra\n"
        extra_modes["unexpected.txt"] = stat.S_IFREG | 0o644
        cases.append(("extra", self.fixture.zip_bytes(extra_files, extra_modes)))
        missing_files = dict(self.fixture.files)
        missing_modes = dict(self.fixture.modes)
        del missing_files["downloads/baseline-first-2400.time"]
        del missing_modes["downloads/baseline-first-2400.time"]
        cases.append((
            "missing", self.fixture.zip_bytes(missing_files, missing_modes),
        ))
        traversal_files = dict(self.fixture.files)
        traversal_modes = dict(self.fixture.modes)
        traversal_files["../escape"] = b"unsafe\n"
        traversal_modes["../escape"] = stat.S_IFREG | 0o644
        cases.append((
            "path traversal",
            self.fixture.zip_bytes(traversal_files, traversal_modes),
        ))
        symlink_modes = dict(self.fixture.modes)
        symlink_modes["evaluator/arena"] = stat.S_IFLNK | 0o777
        cases.append(("symlink", self.fixture.zip_bytes(modes=symlink_modes)))
        for label, snapshot in cases:
            with self.subTest(label=label), self.assertRaises(EvidenceError):
                self.invoke(snapshot)

    def test_rejects_validly_rehashed_but_false_recorded_merge(self) -> None:
        files = dict(self.fixture.files)
        value = json.loads(files["merged/reciprocal.json"])
        value["promotion_gate_passed"] = False
        files["merged/reciprocal.json"] = encoded(value)
        self.fixture.refresh_manifest(files)
        with self.assertRaisesRegex(EvidenceError, "byte-for-byte recomputation"):
            self.invoke(self.fixture.zip_bytes(files))

    def test_rejects_noncanonical_reciprocal_bytes(self) -> None:
        files = dict(self.fixture.files)
        files["merged/reciprocal.json"] += b" \n"
        self.fixture.refresh_manifest(files)
        with self.assertRaisesRegex(EvidenceError, "byte-for-byte recomputation"):
            self.invoke(self.fixture.zip_bytes(files))

    def test_rejects_validly_rehashed_unsuccessful_timing(self) -> None:
        files = dict(self.fixture.files)
        files["downloads/candidate-first-0.time"] = (
            b"wall_s=1.00 user_s=2.00 sys_s=0.10 "
            b"max_rss_kb=1024 exit=1\n"
        )
        self.fixture.refresh_manifest(files)
        with self.assertRaisesRegex(EvidenceError, "timing record"):
            self.invoke(self.fixture.zip_bytes(files))

    def test_rejects_internal_sha_manifest_mismatch(self) -> None:
        files = dict(self.fixture.files)
        files["downloads/candidate-first-0.time"] += b"tamper\n"
        with self.assertRaisesRegex(EvidenceError, "SHA256SUMS.txt mismatch"):
            self.invoke(self.fixture.zip_bytes(files))

    def test_rejects_artifact_digest_mismatch_and_existing_output(self) -> None:
        snapshot = self.fixture.zip_bytes()
        zip_path, run_path, artifact_path = self.write_inputs(snapshot)
        metadata = json.loads(artifact_path.read_text())
        metadata["digest"] = "sha256:" + "0" * 64
        artifact_path.write_text(json.dumps(metadata), encoding="utf-8")
        result, manifest, preserved = self.outputs()
        with self.assertRaisesRegex(EvidenceError, "does not bind"):
            archive.archive_world800(
                zip_path, run_path, artifact_path, result, manifest, preserved)
        metadata["digest"] = "sha256:" + hashlib.sha256(snapshot).hexdigest()
        artifact_path.write_text(json.dumps(metadata), encoding="utf-8")
        result.write_text("do not replace\n", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "refusing to replace"):
            archive.archive_world800(
                zip_path, run_path, artifact_path, result, manifest, preserved)
        self.assertEqual(result.read_text(), "do not replace\n")

    def test_rejects_run_metadata_inconsistency(self) -> None:
        snapshot = self.fixture.zip_bytes()
        zip_path, run_path, artifact_path = self.write_inputs(snapshot)
        metadata = json.loads(run_path.read_text())
        metadata["head_sha"] = "0" * 40
        run_path.write_text(json.dumps(metadata), encoding="utf-8")
        result, manifest, preserved = self.outputs()
        with self.assertRaisesRegex(EvidenceError, "locked successful run"):
            archive.archive_world800(
                zip_path, run_path, artifact_path, result, manifest, preserved)


if __name__ == "__main__":
    unittest.main()
