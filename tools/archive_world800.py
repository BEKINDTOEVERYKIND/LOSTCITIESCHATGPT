#!/usr/bin/env python3
"""Verify and immutably archive the completed locked world-800 campaign.

This is deliberately an offline, post-completion operation.  Its only evidence
input is the exact ``world800-final-evidence.zip`` downloaded from GitHub plus
two small, strict JSON records copied from the GitHub run and artifact APIs.
It validates the complete transport envelope before parsing any efficacy, then
uses the tools embedded in (and byte-identical to) the frozen evidence archive
to reconstruct all four generated JSON files byte for byte.

The authoritative result is an exact copy of ``merged/reciprocal.json``.  Run,
artifact, ZIP, file-manifest, and recomputation provenance live separately in
``world800_archive_manifest.json`` so downstream consumers can continue to
validate the original ``locked_reciprocal_arena_result`` schema unchanged.

Example (the three output paths below are also the defaults)::

    python3 tools/archive_world800.py \
      --zip world800-final-evidence.zip \
      --run-metadata /tmp/world800-run.json \
      --artifact-metadata /tmp/world800-artifact.json

The strict run record has exactly ``repository``, ``run_id``, ``run_attempt``,
``event``, ``status``, ``conclusion``, ``head_branch``, and ``head_sha``.  The
strict artifact record has exactly ``artifact_id``, ``name``,
``size_in_bytes``, ``digest``, ``expired``, ``workflow_run_id``,
``workflow_run_attempt``, and ``workflow_run_head_sha``.  The IDs and all run
identity fields are bound into the deterministic archive manifest; the size
and ``sha256:...`` digest must match the downloaded ZIP snapshot exactly.

No existing output is ever replaced and the locked execution addendum is never
opened for writing.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any, Iterator
import zipfile

if __package__:
    from tools.merge_arena import (
        EvidenceError,
        _combine_reciprocal,
        _remerge_recorded_block,
        merge_block,
    )
    from tools.validate_actor_shards import validate_shards
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.merge_arena import (  # type: ignore[no-redef]
        EvidenceError,
        _combine_reciprocal,
        _remerge_recorded_block,
        merge_block,
    )
    from tools.validate_actor_shards import (  # type: ignore[no-redef]
        validate_shards,
    )


ROOT = Path(__file__).resolve().parents[1]

REPOSITORY = "BEKINDTOEVERYKIND/LOSTCITIESCHATGPT"
RUN_ID = 32627863728
RUN_ATTEMPT = 1
LAUNCH_COMMIT = "04bbfda7f2dfcad134412bb0a0618df2c830e4bd"
LAUNCH_BRANCH = "agent/correctness-and-policy-upgrade"
LAUNCH_PARENT = "a6b1a92283f7eb4b1e10347d52599e5b3c996f5b"
ARTIFACT_NAME = "world800-final-evidence"
INPUT_ZIP_NAME = ARTIFACT_NAME + ".zip"

SOURCE_COMMIT = "08f9e1a5218e03c399b257b852efe20b0089c7b0"
SOURCE_TREE = "c70405a09b88919b228f96d19d84d83875d4fea4"
PLAN_SHA = "3f7d4e8b4be2c58268c9f85ade126a7f15357ab30bf146d71b3c6dc247e74e34"
EXECUTION_SHA = "d8b25f247f9a2e31488afb5b9fe96877972320c2e0a659e821f7169fc83f62cf"
PARENT_RESULT_SHA = (
    "9ae1caa83b9a2ffef715a6c90c3987e386795a00cd92bd19f000f8d2ca1811fb"
)
WORKFLOW_SHA = "61a1286c15fee40186d4d0b69967681ee891ef686bd6954f56d4c0cbc1f45294"
MERGER_SHA = "9cad23c9e6550ea36d7721acf8e64144a44058083ad4aeb5bb5613a3a79139fb"
VALIDATOR_SHA = (
    "bca430a94af64180436c7fb60d29b2e86ec4b3567ab3aabb09984aabee054855"
)
MODEL_SHA = "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"

CANDIDATE = (
    "rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
BASELINE = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
CANDIDATE_SEED = "202608221501"
BASELINE_SEED = "202608221502"
STARTS = list(range(0, 2500, 100))
PAIRS_PER_SHARD = 100
PAIRS_PER_ORIENTATION = 2500
ROUNDS = 3
THREADS = 4
GATE_Z = 1.645

RESULT_REPOSITORY_PATH = "data/experiments/world800_result.json"
MANIFEST_REPOSITORY_PATH = "data/experiments/world800_archive_manifest.json"
ARCHIVE_REPOSITORY_PATH = "data/experiments/world800_final_evidence.zip"

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_ENTRY_BYTES = 512 * 1024 * 1024
_SHA_LINE = re.compile(r"([0-9a-f]{64})  ([\x21-\x7e]+)\Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(token: str) -> None:
    raise EvidenceError(f"non-standard JSON constant {token}")


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(snapshot: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            snapshot.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvidenceError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label}: top-level JSON must be an object")
    return value


def _strict_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        return _strict_json_bytes(path.read_bytes(), label)
    except OSError as exc:
        raise EvidenceError(f"{label}: unreadable: {exc}") from exc


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or not 0 < value <= (1 << 63) - 1:
        raise EvidenceError(f"{label} must be a positive integer")
    return value


def _validate_run_metadata(value: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "repository", "run_id", "run_attempt", "event", "status",
        "conclusion", "head_branch", "head_sha",
    }
    if set(value) != expected_keys:
        raise EvidenceError("run metadata has missing or extra fields")
    if value != {
        "repository": REPOSITORY,
        "run_id": RUN_ID,
        "run_attempt": RUN_ATTEMPT,
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "head_branch": LAUNCH_BRANCH,
        "head_sha": LAUNCH_COMMIT,
    }:
        raise EvidenceError("run metadata does not identify the locked successful run")
    return value


def _validate_artifact_metadata(
        value: dict[str, Any], zip_size: int, zip_digest: str) -> dict[str, Any]:
    expected_keys = {
        "artifact_id", "name", "size_in_bytes", "digest", "expired",
        "workflow_run_id", "workflow_run_attempt", "workflow_run_head_sha",
    }
    if set(value) != expected_keys:
        raise EvidenceError("artifact metadata has missing or extra fields")
    _positive_integer(value.get("artifact_id"), "artifact id")
    expected = {
        "artifact_id": value["artifact_id"],
        "name": ARTIFACT_NAME,
        "size_in_bytes": zip_size,
        "digest": f"sha256:{zip_digest}",
        "expired": False,
        "workflow_run_id": RUN_ID,
        "workflow_run_attempt": RUN_ATTEMPT,
        "workflow_run_head_sha": LAUNCH_COMMIT,
    }
    if value != expected:
        raise EvidenceError("artifact metadata does not bind the exact run and ZIP")
    return value


def _expected_names() -> set[str]:
    names = {
        f"downloads/{orientation}-{start}{suffix}"
        for orientation in ("candidate-first", "baseline-first")
        for start in STARTS
        for suffix in (".jsonl", ".sha256", ".time")
    }
    names.update({
        "merged/structural-validation.json",
        "merged/candidate-first.json",
        "merged/baseline-first.json",
        "merged/reciprocal.json",
        "merged/SHA256SUMS.txt",
        "bindings/locked_world800_plan.json",
        "bindings/locked_world800_execution.json",
        "bindings/role_coherent_result.json",
        "bindings/world800.yml",
        "bindings/merge_arena.py",
        "bindings/validate_actor_shards.py",
        "evaluator/arena",
        "evaluator/champion.bin",
        "evaluator/BUILD_INFO.txt",
    })
    if len(names) != 164:  # Internal invariant, not evidence-dependent.
        raise AssertionError("world800 archive file-count invariant drift")
    return names


EXPECTED_NAMES = _expected_names()


def _validate_zip_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not name or "\x00" in name or "\\" in name or not name.isascii():
        raise EvidenceError(f"unsafe ZIP entry name {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise EvidenceError(f"unsafe ZIP entry path {name!r}")
    if info.is_dir() or name.endswith("/"):
        raise EvidenceError(f"unexpected directory ZIP entry {name!r}")
    if info.flag_bits & 0x1:
        raise EvidenceError(f"encrypted ZIP entry {name!r}")
    if info.create_system != 3:
        raise EvidenceError(f"ZIP entry lacks Unix regular-file metadata: {name!r}")
    mode = info.external_attr >> 16
    if not stat.S_ISREG(mode):
        kind = "symlink" if stat.S_ISLNK(mode) else "non-regular file"
        raise EvidenceError(f"{kind} ZIP entry rejected: {name!r}")
    if info.file_size < 0 or info.file_size > MAX_ENTRY_BYTES:
        raise EvidenceError(f"oversized ZIP entry {name!r}")


def _read_archive(snapshot: bytes) -> tuple[dict[str, bytes], dict[str, int]]:
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot), "r") as archive:
            if archive.comment:
                raise EvidenceError("ZIP archive comment is not permitted")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise EvidenceError("duplicate ZIP entry name")
            for info in infos:
                _validate_zip_member(info)
                if info.comment:
                    raise EvidenceError(f"ZIP entry comment is not permitted: {info.filename}")
            actual = set(names)
            if actual != EXPECTED_NAMES or len(infos) != 164:
                missing = sorted(EXPECTED_NAMES - actual)
                extra = sorted(actual - EXPECTED_NAMES)
                raise EvidenceError(
                    f"world800 ZIP file-set mismatch; missing={missing!r}, "
                    f"extra={extra!r}")
            total = sum(info.file_size for info in infos)
            if total > MAX_UNCOMPRESSED_BYTES:
                raise EvidenceError("world800 ZIP exceeds uncompressed size limit")
            files: dict[str, bytes] = {}
            for info in infos:
                try:
                    data = archive.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise EvidenceError(
                        f"cannot read ZIP entry {info.filename!r}: {exc}") from exc
                if len(data) != info.file_size:
                    raise EvidenceError(f"truncated ZIP entry {info.filename!r}")
                files[info.filename] = data
    except (OSError, zipfile.BadZipFile) as exc:
        raise EvidenceError(f"invalid world800 ZIP: {exc}") from exc
    return files, {"entries": len(files), "uncompressed_size_bytes": total}


def _parse_and_verify_sha_manifest(files: dict[str, bytes]) -> dict[str, str]:
    path = "merged/SHA256SUMS.txt"
    snapshot = files[path]
    try:
        text = snapshot.decode("ascii")
    except UnicodeDecodeError as exc:
        raise EvidenceError("SHA256SUMS.txt must be ASCII") from exc
    if not text.endswith("\n") or "\r" in text or not text:
        raise EvidenceError("SHA256SUMS.txt must be LF-terminated ASCII")
    expected_paths = sorted(EXPECTED_NAMES - {path})
    lines = text.splitlines()
    if len(lines) != 163:
        raise EvidenceError("SHA256SUMS.txt must contain exactly 163 entries")
    parsed: dict[str, str] = {}
    for line in lines:
        match = _SHA_LINE.fullmatch(line)
        if match is None:
            raise EvidenceError("malformed SHA256SUMS.txt entry")
        digest, name = match.groups()
        if name in parsed:
            raise EvidenceError("duplicate SHA256SUMS.txt path")
        parsed[name] = digest
    if list(parsed) != expected_paths:
        raise EvidenceError("SHA256SUMS.txt paths are not the exact sorted file set")
    for name, digest in parsed.items():
        if _sha256(files[name]) != digest:
            raise EvidenceError(f"SHA256SUMS.txt mismatch for {name}")
    return parsed


BOUND_FILES = {
    "bindings/locked_world800_plan.json": (
        ROOT / "data/experiments/locked_world800_plan.json", PLAN_SHA),
    "bindings/locked_world800_execution.json": (
        ROOT / "data/experiments/locked_world800_execution.json", EXECUTION_SHA),
    "bindings/role_coherent_result.json": (
        ROOT / "data/experiments/role_coherent_result.json", PARENT_RESULT_SHA),
    "bindings/world800.yml": (
        ROOT / ".github/workflows/world800.yml", WORKFLOW_SHA),
    "bindings/merge_arena.py": (ROOT / "tools/merge_arena.py", MERGER_SHA),
    "bindings/validate_actor_shards.py": (
        ROOT / "tools/validate_actor_shards.py", VALIDATOR_SHA),
}


def _validate_bindings(files: dict[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for archive_path, (repository_path, expected_sha) in BOUND_FILES.items():
        try:
            repository_snapshot = repository_path.read_bytes()
        except OSError as exc:
            raise EvidenceError(f"cannot read frozen repository binding: {exc}") from exc
        archive_snapshot = files[archive_path]
        if _sha256(repository_snapshot) != expected_sha or \
                _sha256(archive_snapshot) != expected_sha or \
                repository_snapshot != archive_snapshot:
            raise EvidenceError(f"frozen binding mismatch: {archive_path}")
        hashes[archive_path] = expected_sha

    execution = _strict_json_bytes(
        files["bindings/locked_world800_execution.json"], "locked execution")
    if execution.get("source_parent_commit") != LAUNCH_PARENT or \
            execution.get("source") != {
                "commit": SOURCE_COMMIT, "tree": SOURCE_TREE} or \
            execution.get("actors") != {
                "selection_rule": "role_coherent_parent_pass",
                "candidate": CANDIDATE, "baseline": BASELINE,
            } or execution.get("evidence_tools") != {
                "merge_arena_sha256": MERGER_SHA,
                "validate_actor_shards_sha256": VALIDATOR_SHA,
            } or execution.get("model") != {
                "path": "data/champion.bin", "sha256": MODEL_SHA,
            }:
        raise EvidenceError("locked execution identity drift")
    final = execution.get("final")
    if not isinstance(final, dict) or final.get("rounds") != ROUNDS or \
            final.get("threads_per_shard") != THREADS or \
            final.get("pairs_per_orientation") != PAIRS_PER_ORIENTATION or \
            final.get("shards_per_orientation") != len(STARTS) or \
            final.get("pairs_per_shard") != PAIRS_PER_SHARD or \
            final.get("starts") != STARTS or \
            final.get("candidate_first_seed") != CANDIDATE_SEED or \
            final.get("baseline_first_seed") != BASELINE_SEED or \
            final.get("gate_z") != GATE_Z:
        raise EvidenceError("locked execution schedule drift")
    return hashes


def _parse_build_info(files: dict[str, bytes]) -> dict[str, Any]:
    arena = files["evaluator/arena"]
    model = files["evaluator/champion.bin"]
    arena_sha = _sha256(arena)
    if _sha256(model) != MODEL_SHA:
        raise EvidenceError("evaluator model hash mismatch")
    try:
        text = files["evaluator/BUILD_INFO.txt"].decode("ascii")
    except UnicodeDecodeError as exc:
        raise EvidenceError("BUILD_INFO.txt must be ASCII") from exc
    if not text.endswith("\n") or "\r" in text:
        raise EvidenceError("BUILD_INFO.txt must be LF-terminated")
    lines = text.splitlines()
    if len(lines) != 16:
        raise EvidenceError("BUILD_INFO.txt has an unexpected line count")
    keys = [
        "launch_commit", "source_parent_commit", "execution_sha",
        "source_commit", "source_tree", "plan_sha", "parent_result_sha",
        "compiler", "cflags", "ldflags", "threads_per_shard", "arena_size",
        "model_size",
    ]
    values: dict[str, str] = {}
    for line, key in zip(lines[:13], keys):
        prefix = key + "="
        if not line.startswith(prefix) or not line[len(prefix):]:
            raise EvidenceError(f"BUILD_INFO.txt missing ordered {key}")
        values[key] = line[len(prefix):]
    expected = {
        "launch_commit": LAUNCH_COMMIT,
        "source_parent_commit": LAUNCH_PARENT,
        "execution_sha": EXECUTION_SHA,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "plan_sha": PLAN_SHA,
        "parent_result_sha": PARENT_RESULT_SHA,
        "cflags": (
            "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
            "-Wall -Wextra -std=c11"),
        "ldflags": "-lm -pthread",
        "threads_per_shard": str(THREADS),
        "arena_size": str(len(arena)),
        "model_size": str(len(model)),
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise EvidenceError("BUILD_INFO.txt frozen provenance mismatch")
    if not values["compiler"].startswith("gcc ") or \
            not lines[13].startswith("Linux ") or \
            any(ord(character) < 0x20 for character in lines[13]):
        raise EvidenceError("BUILD_INFO.txt compiler or runner identity is invalid")
    if lines[14] != f"{arena_sha}  evaluator/arena" or \
            lines[15] != f"{MODEL_SHA}  evaluator/champion.bin":
        raise EvidenceError("BUILD_INFO.txt evaluator hashes mismatch")
    return {
        "arena_sha256": arena_sha,
        "arena_size_bytes": len(arena),
        "model_sha256": MODEL_SHA,
        "model_size_bytes": len(model),
        "compiler": values["compiler"],
        "runner_uname": lines[13],
        "build_info_sha256": _sha256(files["evaluator/BUILD_INFO.txt"]),
    }


def _expected_provenance(arena_sha: str) -> str:
    return (
        f"stage=world800_final;plan={PLAN_SHA};execution={EXECUTION_SHA};"
        f"parent_result={PARENT_RESULT_SHA};source={SOURCE_COMMIT};"
        f"tree={SOURCE_TREE};arena={arena_sha};model={MODEL_SHA};threads=4"
    )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _materialize(files: dict[str, bytes], directory: Path) -> None:
    for name in sorted(files):
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(files[name])


def _recompute_all(
        files: dict[str, bytes], arena_sha: str) -> tuple[
            dict[str, Any], dict[str, dict[str, Any]]]:
    provenance = _expected_provenance(arena_sha)
    with tempfile.TemporaryDirectory(prefix="world800-verify-") as temporary:
        root = Path(temporary)
        _materialize(files, root)
        with _working_directory(root):
            structural = validate_shards(
                Path("downloads"), CANDIDATE, BASELINE, provenance,
                CANDIDATE_SEED, BASELINE_SEED, STARTS, PAIRS_PER_SHARD,
                ROUNDS,
            )
            first_paths = [
                Path(f"downloads/candidate-first-{start}.jsonl")
                for start in STARTS
            ]
            second_paths = [
                Path(f"downloads/baseline-first-{start}.jsonl")
                for start in STARTS
            ]
            first = merge_block(first_paths, 0, PAIRS_PER_ORIENTATION)
            second = merge_block(second_paths, 0, PAIRS_PER_ORIENTATION)
            first_bytes = _canonical_json(first)
            second_bytes = _canonical_json(second)
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
                first, second,
                [
                    {"path": "merged/candidate-first.json",
                     "sha256": _sha256(first_bytes)},
                    {"path": "merged/baseline-first.json",
                     "sha256": _sha256(second_bytes)},
                ],
                gate_z=GATE_Z, require_positive_margin=True,
                raw_input_validation=raw_validation,
            )

    generated = {
        "merged/structural-validation.json": structural,
        "merged/candidate-first.json": first,
        "merged/baseline-first.json": second,
        "merged/reciprocal.json": reciprocal,
    }
    for path, value in generated.items():
        if _canonical_json(value) != files[path]:
            raise EvidenceError(f"{path} is not its byte-for-byte recomputation")
    return reciprocal, generated


def _validate_gate(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("artifact_kind") != "locked_reciprocal_arena_result" or \
            result.get("candidate") != CANDIDATE or \
            result.get("baseline") != BASELINE or \
            result.get("promotion_gate_configuration") != {
                "critical_z": GATE_Z,
                "require_positive_margin": True,
                "require_each_orientation_above_half": True,
                "require_raw_input_validation": True,
            } or result.get("raw_input_validation", {}).get("status") != \
            "validated":
        raise EvidenceError("reciprocal result identity or gate configuration drift")
    candidate = result.get("candidate_result")
    if not isinstance(candidate, dict):
        raise EvidenceError("reciprocal result omits candidate statistics")
    for field in (
            "match_score", "match_score_pair_clustered_se",
            "promotion_gate_lower_bound", "margin_per_game"):
        value = candidate.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isfinite(float(value)):
            raise EvidenceError(f"invalid reciprocal gate field {field}")
    orientations = candidate.get("orientation_match_scores")
    if not isinstance(orientations, list) or len(orientations) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) for value in orientations):
        raise EvidenceError("invalid reciprocal orientation scores")
    lower = candidate["match_score"] - GATE_Z * \
        candidate["match_score_pair_clustered_se"]
    if candidate["promotion_gate_lower_bound"] != lower or \
            candidate.get("capped_rounds") != 0:
        raise EvidenceError("reciprocal gate inputs are inconsistent")
    statistical = lower > 0.5 and orientations[0] > 0.5 and \
        orientations[1] > 0.5 and candidate["margin_per_game"] > 0.0
    if result.get("statistical_gate_passed") is not statistical or \
            result.get("promotion_gate_passed") is not statistical:
        raise EvidenceError("reciprocal promotion gate was not exactly recomputed")
    return {
        "critical_z": GATE_Z,
        "match_score": candidate["match_score"],
        "match_score_pair_clustered_se": (
            candidate["match_score_pair_clustered_se"]),
        "promotion_gate_lower_bound": lower,
        "orientation_match_scores": orientations,
        "margin_per_game": candidate["margin_per_game"],
        "capped_rounds": 0,
        "promotion_gate_passed": statistical,
        "selected_actor": CANDIDATE if statistical else BASELINE,
        "selected_worlds": 800 if statistical else 512,
    }


def _manifest(
        run: dict[str, Any], artifact: dict[str, Any], zip_digest: str,
        zip_size: int, zip_info: dict[str, int], sha_manifest: dict[str, str],
        bindings: dict[str, str], build: dict[str, Any],
        generated: dict[str, dict[str, Any]], gate: dict[str, Any],
        result_bytes: bytes) -> dict[str, Any]:
    merged_hashes = {
        path: _sha256(_canonical_json(value))
        for path, value in sorted(generated.items())
    }
    return {
        "schema_version": 1,
        "artifact_kind": "verified_world800_archive_manifest",
        "status": (
            "complete_valid_gate_passed" if gate["promotion_gate_passed"]
            else "complete_valid_gate_failed"),
        "github": {
            "repository": run["repository"],
            "run_id": run["run_id"],
            "run_attempt": run["run_attempt"],
            "event": run["event"],
            "conclusion": run["conclusion"],
            "head_branch": run["head_branch"],
            "head_sha": run["head_sha"],
            "artifact_id": artifact["artifact_id"],
            "artifact_name": artifact["name"],
            "artifact_digest": artifact["digest"],
            "artifact_size_bytes": artifact["size_in_bytes"],
        },
        "preserved_archive": {
            "path": ARCHIVE_REPOSITORY_PATH,
            "sha256": zip_digest,
            "size_bytes": zip_size,
            "regular_file_entries": zip_info["entries"],
            "uncompressed_size_bytes": zip_info["uncompressed_size_bytes"],
            "source_sha256_manifest": "merged/SHA256SUMS.txt",
            "source_sha256_manifest_sha256": _sha256(
                _canonical_or_raw_sha_manifest(sha_manifest)),
            "source_sha256_manifest_entries": len(sha_manifest),
            "all_source_manifest_hashes_verified": True,
        },
        "locked_bindings": bindings,
        "evaluator": build,
        "verification": {
            "raw_shards": 50,
            "raw_shards_per_orientation": 25,
            "raw_pairs_per_shard": PAIRS_PER_SHARD,
            "raw_pairs_per_orientation": PAIRS_PER_ORIENTATION,
            "total_mirrored_pairs": 2 * PAIRS_PER_ORIENTATION,
            "timing_sidecars": 50,
            "sha256_sidecars": 50,
            "all_shards_hashes_timing_provenance_seeds_actors_and_coverage_validated": True,
            "all_four_json_outputs_independently_recomputed_byte_for_byte": True,
            "recomputed_json_sha256": merged_hashes,
        },
        "authoritative_result": {
            "path": RESULT_REPOSITORY_PATH,
            "sha256": _sha256(result_bytes),
            "byte_for_byte_source": "merged/reciprocal.json",
            "schema_version": 1,
            "artifact_kind": "locked_reciprocal_arena_result",
        },
        "gate": gate,
    }


def _canonical_or_raw_sha_manifest(parsed: dict[str, str]) -> bytes:
    """Reconstruct the workflow's exact sorted sha256sum manifest bytes."""
    return "".join(
        f"{digest}  {path}\n" for path, digest in parsed.items()
    ).encode("ascii")


def _write_exclusive_group(outputs: list[tuple[Path, bytes]]) -> None:
    resolved = [path.resolve(strict=False) for path, _ in outputs]
    if len(set(resolved)) != len(resolved):
        raise EvidenceError("output paths must be distinct")
    for path, _ in outputs:
        if path.exists() or path.is_symlink():
            raise EvidenceError(f"refusing to replace existing output {path}")
        if not path.parent.is_dir():
            raise EvidenceError(f"output directory does not exist: {path.parent}")

    staged: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for output, snapshot in outputs:
            temporary = output.with_name(
                output.name + f".tmp.{os.getpid()}.{len(staged)}")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(fd, "wb") as stream:
                    fd = -1
                    stream.write(snapshot)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)
            staged.append((temporary, output))
        for temporary, output in staged:
            os.link(temporary, output)
            installed.append(output)
        for directory in {output.parent for output, _ in outputs}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except (OSError, FileExistsError) as exc:
        for output in installed:
            try:
                output.unlink()
            except OSError:
                pass
        raise EvidenceError(f"cannot install immutable archive outputs: {exc}") from exc
    finally:
        for temporary, _ in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def archive_world800(
        zip_path: Path, run_metadata_path: Path, artifact_metadata_path: Path,
        result_path: Path, manifest_path: Path, archive_path: Path) -> dict[str, Any]:
    if zip_path.name != INPUT_ZIP_NAME:
        raise EvidenceError(f"input ZIP must be named exactly {INPUT_ZIP_NAME}")
    try:
        zip_size = zip_path.stat().st_size
        if not 0 < zip_size <= MAX_ARCHIVE_BYTES:
            raise EvidenceError("world800 ZIP size is invalid or exceeds limit")
        zip_snapshot = zip_path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read world800 ZIP: {exc}") from exc
    if len(zip_snapshot) != zip_size:
        raise EvidenceError("world800 ZIP changed while being read")
    zip_digest = _sha256(zip_snapshot)

    run = _validate_run_metadata(
        _strict_json_file(run_metadata_path, "run metadata"))
    artifact = _validate_artifact_metadata(
        _strict_json_file(artifact_metadata_path, "artifact metadata"),
        zip_size, zip_digest)

    # Complete transport structure and all 163 declared content hashes are
    # validated before any JSONL score rows or merged efficacy are parsed.
    files, zip_info = _read_archive(zip_snapshot)
    sha_manifest = _parse_and_verify_sha_manifest(files)
    bindings = _validate_bindings(files)
    build = _parse_build_info(files)

    result, generated = _recompute_all(files, build["arena_sha256"])
    gate = _validate_gate(result)
    result_bytes = files["merged/reciprocal.json"]
    manifest = _manifest(
        run, artifact, zip_digest, zip_size, zip_info, sha_manifest, bindings,
        build, generated, gate, result_bytes)
    manifest_bytes = _canonical_json(manifest)

    input_resolved = zip_path.resolve(strict=True)
    if archive_path.resolve(strict=False) == input_resolved:
        raise EvidenceError("preserved archive output must differ from input ZIP")
    _write_exclusive_group([
        (archive_path, zip_snapshot),
        (result_path, result_bytes),
        (manifest_path, manifest_bytes),
    ])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed offline verifier for world800-final-evidence")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--artifact-metadata", type=Path, required=True)
    parser.add_argument(
        "--output-result", type=Path,
        default=Path(RESULT_REPOSITORY_PATH))
    parser.add_argument(
        "--output-manifest", type=Path,
        default=Path(MANIFEST_REPOSITORY_PATH))
    parser.add_argument(
        "--output-archive", type=Path,
        default=Path(ARCHIVE_REPOSITORY_PATH))
    args = parser.parse_args()
    try:
        manifest = archive_world800(
            args.zip, args.run_metadata, args.artifact_metadata,
            args.output_result, args.output_manifest, args.output_archive)
    except (EvidenceError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": manifest["status"],
        "promotion_gate_passed": manifest["gate"]["promotion_gate_passed"],
        "selected_worlds": manifest["gate"]["selected_worlds"],
        "result_sha256": manifest["authoritative_result"]["sha256"],
        "archive_sha256": manifest["preserved_archive"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
