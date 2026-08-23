#!/usr/bin/env python3
"""Fail-closed orchestration helpers for the match-value actor campaign.

The GitHub workflow deliberately delegates every result-dependent operation to
this module.  Raw arena rows are still validated and merged by
``validate_actor_shards.py`` and ``merge_arena.py``.  This layer binds the
authoritative world-count result, validates the paired match-value artifacts,
constructs the four preregistered actors, and applies the exact staged
selection and promotion rules only to complete reciprocal panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any

if __package__:
    from tools.gate_actor_panel import _rebuild_reciprocal
    from tools.merge_arena import EvidenceError, _combine_reciprocal, _write_json
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gate_actor_panel import _rebuild_reciprocal  # type: ignore[no-redef]
    from merge_arena import (  # type: ignore[no-redef]
        EvidenceError,
        _combine_reciprocal,
        _write_json,
    )


PLAN_PATH = "data/experiments/match_value_variant_plan.json"
WORKFLOW_PATH = ".github/workflows/match-value-variant.yml"
EXECUTION_PATH = (
    "data/experiments/locked_match_value_variant_execution.json"
)
WORLD_RESULT_PATH = "data/experiments/world800_result.json"
MODEL_PATH = "data/champion.bin"
MODEL_SHA256 = (
    "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"
)
COMPILER = "gcc"
COMPILER_SEMANTIC_VERSION_COMMAND = "gcc -dumpfullversion -dumpversion"
REQUIRED_COMPILER_SEMANTIC_VERSION = "13.3.0"
CFLAGS = (
    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
    "-Wall -Wextra -std=c11"
)
LDFLAGS = "-lm -pthread"
BUILD_PROFILE_HEX = "0030d23b"
TABLE_SAMPLES = 16000
TABLE_THREADS = 8
TABLE_SEED = "7331001"
TABLE_ROLE_CYCLE = 400
TABLE_CONTROLLER_ABI = 1
TABLE_CONTROLLER_WORDS = [0, 20, 4, 1, 1, 0, 0, 0, 0, 0, 300]
WORLD800_PLAN_SHA256 = (
    "3f7d4e8b4be2c58268c9f85ade126a7f15357ab30bf146d71b3c6dc247e74e34"
)
WORLD800_EXECUTION_SHA256 = (
    "d8b25f247f9a2e31488afb5b9fe96877972320c2e0a659e821f7169fc83f62cf"
)
WORLD800_PARENT_RESULT_SHA256 = (
    "9ae1caa83b9a2ffef715a6c90c3987e386795a00cd92bd19f000f8d2ca1811fb"
)
WORLD800_SOURCE_COMMIT = "08f9e1a5218e03c399b257b852efe20b0089c7b0"
WORLD800_SOURCE_TREE = "c70405a09b88919b228f96d19d84d83875d4fea4"
WORLD800_FIRST_SEED = "202608221501"
WORLD800_SECOND_SEED = "202608221502"

BASELINE_512 = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
CANDIDATE_800 = (
    "rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
VARIANTS = ("R14", "P14", "R0", "P0")
TIE_PRIORITY = ("P14", "R14", "P0", "R0")
SOURCE_FILES = (
    "src/agent.c",
    "src/agent.h",
    "src/match.c",
    "src/match_value.c",
    "src/match_value.h",
    "src/rollout.c",
    "src/spec.c",
    "src/spec.h",
    "tools/arena.c",
    "tools/build_match_value.c",
    "tools/gate_actor_panel.py",
    "tools/match_value_campaign.py",
    "tools/merge_arena.py",
    "tools/validate_actor_shards.py",
)

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _reject_constant(token: str) -> None:
    raise EvidenceError(f"non-standard JSON constant {token}")


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise EvidenceError(f"duplicate JSON key {key}")
        value[key] = item
    return value


def strict_json(path: Path) -> dict[str, Any]:
    try:
        snapshot = path.read_bytes()
        value = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EvidenceError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path}: top-level JSON must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot hash: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha(value: Any, bits: int, field: str) -> str:
    pattern = _HEX40 if bits == 160 else _HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise EvidenceError(f"invalid {field}")
    return value


def _world_result(path: Path) -> tuple[dict[str, Any], bool, str, int]:
    """Validate the authoritative, completed world-800 reciprocal result."""
    value = strict_json(path)
    if value.get("artifact_kind") != "locked_reciprocal_arena_result":
        raise EvidenceError("wrong authoritative world800 artifact kind")
    blocks = value.get("blocks")
    snapshots = value.get("input_block_snapshots")
    raw_validation = value.get("raw_input_validation")
    config = value.get("promotion_gate_configuration")
    if not isinstance(blocks, list) or len(blocks) != 2 or \
            not isinstance(snapshots, list) or len(snapshots) != 2 or \
            not isinstance(raw_validation, dict) or \
            raw_validation.get("status") != "validated" or config != {
                "critical_z": 1.645,
                "require_positive_margin": True,
                "require_each_orientation_above_half": True,
                "require_raw_input_validation": True,
            }:
        raise EvidenceError("world800 result lacks the exact validated gate")
    rebuilt = _combine_reciprocal(
        blocks[0], blocks[1], snapshots, gate_z=1.645,
        require_positive_margin=True, raw_input_validation=raw_validation,
    )
    if rebuilt != value:
        raise EvidenceError("world800 result is not its exact embedded recomputation")
    if value.get("candidate") != CANDIDATE_800 or \
            value.get("baseline") != BASELINE_512:
        raise EvidenceError("world800 actor identity drift")
    result = value.get("candidate_result")
    if not isinstance(result, dict) or result.get("capped_rounds") != 0:
        raise EvidenceError("world800 result has invalid or capped evidence")
    provenance = value.get("provenance")
    expected_provenance = re.compile(
        "stage=world800_final;"
        f"plan={WORLD800_PLAN_SHA256};"
        f"execution={WORLD800_EXECUTION_SHA256};"
        f"parent_result={WORLD800_PARENT_RESULT_SHA256};"
        f"source={WORLD800_SOURCE_COMMIT};"
        f"tree={WORLD800_SOURCE_TREE};"
        r"arena=[0-9a-f]{64};"
        f"model={MODEL_SHA256};threads=4\\Z"
    )
    if not isinstance(provenance, str) or \
            expected_provenance.fullmatch(provenance) is None:
        raise EvidenceError("world800 result provenance drift")
    expected_blocks = (
        (CANDIDATE_800, BASELINE_512, WORLD800_FIRST_SEED),
        (BASELINE_512, CANDIDATE_800, WORLD800_SECOND_SEED),
    )
    for block, identity in zip(blocks, expected_blocks):
        metadata = block.get("metadata") if isinstance(block, dict) else None
        inputs = block.get("inputs") if isinstance(block, dict) else None
        if not isinstance(metadata, dict) or block.get("pair_start") != "0" or \
                block.get("pair_count") != 2500 or \
                metadata.get("agent_a") != identity[0] or \
                metadata.get("agent_b") != identity[1] or \
                metadata.get("seed") != identity[2] or \
                metadata.get("rounds") != 3 or \
                metadata.get("provenance") != provenance or \
                not isinstance(inputs, list) or len(inputs) != 25 or any(
                    item.get("pair_start") != index * 100 or
                    item.get("pair_count") != 100
                    for index, item in enumerate(inputs)
                    if isinstance(item, dict)
                ) or any(not isinstance(item, dict) for item in inputs):
            raise EvidenceError("world800 reciprocal block schedule drift")
    passed = value.get("promotion_gate_passed")
    if type(passed) is not bool:
        raise EvidenceError("world800 result has no authoritative boolean gate")
    expected_pass = bool(value.get("statistical_gate_passed"))
    if passed != expected_pass:
        raise EvidenceError("world800 promotion decision is inconsistent")
    return value, passed, CANDIDATE_800 if passed else BASELINE_512, \
        800 if passed else 512


def validate_plan(value: dict[str, Any]) -> None:
    """Check the high-value preregistration invariants used by the workflow."""
    if value.get("schema") != 3 or \
            value.get("experiment") != "match-value-factorial-development-panel" or \
            value.get("status") != "blocked_pending_add_only_world800_binding" or \
            value.get("scope") != "development_then_reserved_final_only":
        raise EvidenceError("unsupported match-value plan identity")
    build = value.get("artifact_build")
    if not isinstance(build, dict) or build.get("samples") != TABLE_SAMPLES or \
            build.get("threads") != TABLE_THREADS or \
            build.get("seed") != int(TABLE_SEED) or \
            build.get("compiler") != {
                "executable": COMPILER,
                "semantic_version_command":
                    COMPILER_SEMANTIC_VERSION_COMMAND,
                "required_semantic_version":
                    REQUIRED_COMPILER_SEMANTIC_VERSION,
                "build_info_records": [
                    "gcc --version first line",
                    "uname -a",
                    "ImageOS",
                    "ImageVersion",
                    "RUNNER_OS",
                    "RUNNER_ARCH",
                ],
            } or \
            build.get("command_template") != (
                "./bin/build_match_value --model data/champion.bin --out "
                "tables/winner-o0-16000-isotonic.lcmv --raw-out "
                "tables/winner-o0-16000-raw.lcmv --samples 16000 "
                "--threads 8 --seed 7331001 --playout-symmetries 20"
            ) or \
            build.get("required_role_balance") != \
            "complete 20x20 product cycles" or \
            build.get("required_controller_abi") != TABLE_CONTROLLER_ABI or \
            build.get("variants_share_identical_transition_histograms") is not True:
        raise EvidenceError("table-build preregistration drift")
    invariant = value.get("invariants")
    if not isinstance(invariant, dict) or invariant.get("rounds") != 3 or \
            invariant.get("world_cap") != \
            "freeze_to_world800_campaign_winner_in_execution_addendum" or \
            invariant.get("uniform_hidden_worlds") is not True or \
            invariant.get("playout_sample") != 4 or \
            invariant.get("playout_symmetries") != 20 or \
            invariant.get("playout_prune") != 1 or \
            invariant.get("exact_terminal") != 1 or \
            invariant.get("capped_rounds_must_be_zero") is not True:
        raise EvidenceError("actor invariant drift")
    candidates = value.get("factorial_candidates")
    expected = {
        "R14": (3, False, 14, "tables/winner-o0-16000-raw.lcmv"),
        "P14": (3, True, 14, "tables/winner-o0-16000-isotonic.lcmv"),
        "R0": (3, False, 0, "tables/winner-o0-16000-raw.lcmv"),
        "P0": (3, True, 0, "tables/winner-o0-16000-isotonic.lcmv"),
    }
    if not isinstance(candidates, dict) or set(candidates) != set(expected):
        raise EvidenceError("factorial candidate set drift")
    for name, (objective, projected, ply_lo, table) in expected.items():
        item = candidates[name]
        if not isinstance(item, dict) or item.get("objective") != objective or \
                item.get("isotonic_projected") is not projected or \
                item.get("table") != table or \
                item.get("phase") != {"ply_lo": ply_lo, "ply_hi": 0}:
            raise EvidenceError(f"factorial candidate {name} drift")
    stage1 = value.get("stage_1_factorial_screen")
    stage2 = value.get("stage_2_development_confirmation")
    final = value.get("locked_final_test_reservation")
    if not isinstance(stage1, dict) or stage1.get("total_mirrored_pairs") != 1600 or \
            stage1.get("comparisons") != [[name, "legacy"] for name in VARIANTS]:
        raise EvidenceError("stage-1 panel drift")
    if not isinstance(stage2, dict) or \
            stage2.get("total_mirrored_pairs") != 1000:
        raise EvidenceError("stage-2 panel drift")
    if not isinstance(final, dict) or final.get("total_mirrored_pairs") != 5000:
        raise EvidenceError("final reservation drift")
    expected_blocks = (
        (stage1, [(202608290101, 200, 2, 100),
                  (202608290102, 200, 2, 100)]),
        (stage2, [(202608290201, 500, 5, 100),
                  (202608290202, 500, 5, 100)]),
        (final, [(202608300101, 2500, 25, 100),
                 (202608300102, 2500, 25, 100)]),
    )
    for section, rows in expected_blocks:
        blocks = section.get("reciprocal_blocks")
        if not isinstance(blocks, list) or len(blocks) != 2:
            raise EvidenceError("reciprocal block count drift")
        for block, expected_row in zip(blocks, rows):
            values = (
                block.get("seed"),
                block.get("mirrored_pairs_per_candidate",
                          block.get("mirrored_pairs")),
                block.get("shards_per_candidate", block.get("shards")),
                block.get("pairs_per_shard"),
            )
            if values != expected_row:
                raise EvidenceError("reciprocal block schedule drift")


def expected_execution(
    root: Path,
    source_commit: str,
    source_tree: str,
    world_result_sha: str,
    world_passed: bool,
    winner: str,
    world_cap: int,
) -> dict[str, Any]:
    source_hashes = {
        path: sha256(root / path) for path in SOURCE_FILES
    }
    return {
        "schema_version": 1,
        "artifact_kind": "locked_match_value_variant_execution",
        "status": "launch_bound_after_world800_before_table_build_or_efficacy",
        "source_parent_commit": source_commit,
        "source_parent_tree": source_tree,
        "source_files_sha256": source_hashes,
        "workflow": {
            "path": WORKFLOW_PATH,
            "sha256": sha256(root / WORKFLOW_PATH),
        },
        "plan": {
            "path": PLAN_PATH,
            "sha256": sha256(root / PLAN_PATH),
        },
        "authoritative_world800_result": {
            "path": WORLD_RESULT_PATH,
            "sha256": world_result_sha,
            "candidate_actor": CANDIDATE_800,
            "baseline_actor": BASELINE_512,
            "promotion_gate_passed": world_passed,
            "selected_world_cap": world_cap,
            "selected_actor": winner,
        },
        "build": {
            "runner": "ubuntu-24.04",
            "compiler": COMPILER,
            "compiler_semantic_version_command":
                COMPILER_SEMANTIC_VERSION_COMMAND,
            "required_compiler_semantic_version":
                REQUIRED_COMPILER_SEMANTIC_VERSION,
            "cflags": CFLAGS,
            "ldflags": LDFLAGS,
            "expected_build_profile_hex": BUILD_PROFILE_HEX,
            "compile_once": ["bin/arena", "bin/build_match_value"],
            "model": {"path": MODEL_PATH, "sha256": MODEL_SHA256},
            "table": {
                "single_transition_generation": True,
                "samples": TABLE_SAMPLES,
                "threads": TABLE_THREADS,
                "seed": TABLE_SEED,
                "playout_symmetries": 20,
                "role_cycle_size": TABLE_ROLE_CYCLE,
                "role_balance_complete": True,
                "controller_abi": TABLE_CONTROLLER_ABI,
                "controller_words": TABLE_CONTROLLER_WORDS,
                "raw_path": "tables/winner-o0-16000-raw.lcmv",
                "projected_path": "tables/winner-o0-16000-isotonic.lcmv",
            },
        },
        "evaluation": {
            "plan_is_complete_command_seed_and_gate_binding": True,
            "partial_efficacy_inspection_forbidden": True,
            "stage1_selects_exactly_one_of": list(VARIANTS),
            "stage2_runs_only_selected_variant": True,
            "reserved_final_runs_only_after_exact_stage2_pass": True,
            "no_automatic_repository_promotion": True,
        },
        "results": None,
    }


def guard_execution(
    root: Path, execution_path: Path, source_commit: str,
    source_tree: str,
) -> dict[str, Any]:
    _canonical_sha(source_commit, 160, "source commit")
    _canonical_sha(source_tree, 160, "source tree")
    plan = strict_json(root / PLAN_PATH)
    validate_plan(plan)
    world_path = root / WORLD_RESULT_PATH
    _, passed, winner, world_cap = _world_result(world_path)
    world_sha = sha256(world_path)
    actual = strict_json(execution_path)
    expected = expected_execution(
        root, source_commit, source_tree, world_sha, passed, winner, world_cap,
    )
    if actual != expected:
        raise EvidenceError("execution addendum does not exactly match locked inputs")
    return expected


def _fnv1a(snapshot: bytes) -> int:
    value = 1469598103934665603
    for byte in snapshot:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def inspect_table(path: Path, projected: bool) -> dict[str, Any]:
    try:
        snapshot = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot read table: {exc}") from exc
    if len(snapshot) < 136 or snapshot[:8] != b"LCMVAL1\0":
        raise EvidenceError(f"{path}: invalid match-value header")
    u32 = lambda offset: struct.unpack_from("<I", snapshot, offset)[0]
    u64 = lambda offset: struct.unpack_from("<Q", snapshot, offset)[0]
    f64 = lambda offset: struct.unpack_from("<d", snapshot, offset)[0]
    version, header_bytes = u32(8), u32(12)
    samples, r1_count, r2_count, lead_limit = (
        u32(16), u32(20), u32(24), u32(28)
    )
    expected_size = 128 + 16 * (r1_count + r2_count) + 8
    if version != 1 or header_bytes != 128 or r1_count != 2361 or \
            r2_count != 4721 or lead_limit != 150 or \
            len(snapshot) != expected_size:
        raise EvidenceError(f"{path}: unsupported table dimensions")
    footer = u64(len(snapshot) - 8)
    fingerprint = _fnv1a(snapshot[:-8])
    if footer != fingerprint:
        raise EvidenceError(f"{path}: table payload fingerprint mismatch")
    words = [u32(40 + 4 * index) for index in range(11)]
    adjustments = [f64(92), f64(100)]
    controller_abi, build_profile = u32(120), u32(124)
    if samples != TABLE_SAMPLES or u64(84) != int(TABLE_SEED) or \
            u32(108) != TABLE_ROLE_CYCLE or u32(112) != 1 or \
            u32(116) != int(projected) or \
            controller_abi != TABLE_CONTROLLER_ABI or \
            f"{build_profile:08x}" != BUILD_PROFILE_HEX or \
            words != TABLE_CONTROLLER_WORDS or \
            not all(math.isfinite(value) and value >= 0.0
                    for value in adjustments):
        raise EvidenceError(f"{path}: frozen table metadata mismatch")
    count = 2 * (r1_count + r2_count)
    values = struct.unpack_from(f"<{count}d", snapshot, 128)
    if any(not math.isfinite(value) or abs(value) > 227.0 for value in values):
        raise EvidenceError(f"{path}: invalid table value")
    first = 0
    for length in (r1_count, r2_count):
        not_start = values[first:first + length]
        starts = values[first + length:first + 2 * length]
        if any(a != -b for a, b in zip(starts, reversed(not_start))):
            raise EvidenceError(f"{path}: table violates player-swap zero sum")
        if projected and any(a > b for a, b in zip(not_start, not_start[1:])):
            raise EvidenceError(f"{path}: projected table is nonmonotone")
        if projected and any(a > b for a, b in zip(starts, starts[1:])):
            raise EvidenceError(f"{path}: projected start table is nonmonotone")
        first += 2 * length
    return {
        "path": str(path),
        "sha256": hashlib.sha256(snapshot).hexdigest(),
        "size": len(snapshot),
        "version": version,
        "samples_per_policy_lead": samples,
        "source_seed": str(u64(84)),
        "role_cycle_size": u32(108),
        "role_balance_complete": bool(u32(112)),
        "isotonic_projected": projected,
        "max_isotonic_adjustment": adjustments,
        "payload_fingerprint": f"{fingerprint:016x}",
        "controller": {
            "net_fingerprint": f"{u64(32):016x}",
            "controller_words": words,
            "controller_abi": controller_abi,
            "build_profile_hex": f"{build_profile:08x}",
        },
    }


def table_manifest(raw_path: Path, projected_path: Path) -> dict[str, Any]:
    raw = inspect_table(raw_path, False)
    projected = inspect_table(projected_path, True)
    shared = (
        "samples_per_policy_lead", "source_seed", "role_cycle_size",
        "role_balance_complete", "max_isotonic_adjustment", "controller",
    )
    if any(raw[field] != projected[field] for field in shared):
        raise EvidenceError("raw and projected tables do not share one build corpus")
    if raw["sha256"] == projected["sha256"] or \
            raw["payload_fingerprint"] == projected["payload_fingerprint"]:
        raise EvidenceError("raw and projected table identities collapsed")
    return {
        "schema_version": 1,
        "artifact_kind": "controller_bound_match_value_table_pair",
        "status": "complete_valid_single_transition_generation",
        "variants_share_identical_transition_histograms": True,
        "raw": raw,
        "projected": projected,
    }


def build_actors(
    baseline: str, world_cap: int, raw_path: str, projected_path: str,
) -> dict[str, Any]:
    fields = baseline.split(":")
    if len(fields) != 38 or fields[0] != "rolloutu" or \
            fields[1] != MODEL_PATH or fields[2] != str(world_cap):
        raise EvidenceError("winner actor is not the frozen maintained controller")
    tail = fields[2:]
    if len(tail) != 36 or tail[19] != str(world_cap) or \
            tail[5] != "14" or tail[8] != "0" or tail[12] != "4" or \
            tail[13] != "20" or tail[16] != "20" or tail[20] != "1" or \
            tail[27] != "3" or tail[35] != "1":
        raise EvidenceError("winner actor controller fields drifted")
    # Explicit defaults for fields 36..40.  Field 39 defaults to one point;
    # it remains unreachable because the bounded-late resolver is disabled.
    tail.extend(["0", "0", "0", "1", "0"])
    if len(tail) != 41:
        raise AssertionError("rollout tail width changed")

    def actor(ply_lo: int, table: str) -> str:
        candidate = list(tail)
        candidate[5] = str(ply_lo)
        candidate[8] = "3"
        candidate.append(table)
        if len(candidate) != 42:
            raise AssertionError("match-value table field drift")
        return "rolloutu2:" + ":".join(
            [MODEL_PATH, MODEL_PATH, *candidate]
        )

    actors = {
        "legacy": baseline,
        "R14": actor(14, raw_path),
        "P14": actor(14, projected_path),
        "R0": actor(0, raw_path),
        "P0": actor(0, projected_path),
    }
    return {
        "schema_version": 1,
        "artifact_kind": "locked_match_value_variant_actors",
        "world_cap": world_cap,
        "model_path": MODEL_PATH,
        "actors": actors,
    }


def _panel_exact_statistics(value: dict[str, Any]) -> dict[str, int]:
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 2:
        raise EvidenceError("reciprocal panel lacks two blocks")
    sufficient = []
    for block in blocks:
        item = block.get("sufficient_statistics") if isinstance(block, dict) else None
        if not isinstance(item, dict):
            raise EvidenceError("reciprocal block lacks sufficient statistics")
        sufficient.append(item)
    a, b = sufficient
    fields = ("pairs", "score_quarters_sum", "margin_sum", "capped_rounds")
    if any(type(item.get(field)) is not int
           for item in sufficient for field in fields):
        raise EvidenceError("noninteger reciprocal sufficient statistics")
    if a["pairs"] <= 0 or a["pairs"] != b["pairs"]:
        raise EvidenceError("unbalanced reciprocal panel")
    n = a["pairs"]
    return {
        "pairs_per_orientation": n,
        "combined_score_numerator": a["score_quarters_sum"] +
            (4 * n - b["score_quarters_sum"]),
        "combined_score_denominator": 8 * n,
        "combined_margin_numerator": a["margin_sum"] - b["margin_sum"],
        "combined_margin_denominator": 4 * n,
        "candidate_first_score_quarters": a["score_quarters_sum"],
        "baseline_first_score_quarters": b["score_quarters_sum"],
        "capped_rounds": a["capped_rounds"] + b["capped_rounds"],
    }


def _validate_panel_identity(
    value: dict[str, Any], candidate: str, baseline: str, provenance: str,
    pairs: int, first_seed: str, second_seed: str,
) -> None:
    if value.get("artifact_kind") != "locked_reciprocal_arena_result" or \
            value.get("candidate") != candidate or \
            value.get("baseline") != baseline or \
            value.get("provenance") != provenance or \
            value.get("raw_input_validation", {}).get("status") != "validated":
        raise EvidenceError("locked reciprocal panel identity drift")
    result = value.get("candidate_result")
    if not isinstance(result, dict) or result.get("capped_rounds") != 0:
        raise EvidenceError("locked reciprocal panel has capped rounds")
    blocks = value.get("blocks")
    expected = (
        (candidate, baseline, first_seed),
        (baseline, candidate, second_seed),
    )
    if not isinstance(blocks, list) or len(blocks) != 2:
        raise EvidenceError("locked reciprocal panel lacks orientations")
    for block, identity in zip(blocks, expected):
        metadata = block.get("metadata") if isinstance(block, dict) else None
        if not isinstance(metadata, dict) or block.get("pair_start") != "0" or \
                block.get("pair_count") != pairs or \
                metadata.get("agent_a") != identity[0] or \
                metadata.get("agent_b") != identity[1] or \
                metadata.get("seed") != identity[2] or \
                metadata.get("rounds") != 3 or \
                metadata.get("provenance") != provenance:
            raise EvidenceError("locked reciprocal block identity drift")
    exact = _panel_exact_statistics(value)
    if exact["pairs_per_orientation"] != pairs or exact["capped_rounds"] != 0:
        raise EvidenceError("locked reciprocal panel size or cap drift")


def load_verified_panel(
    path: Path, candidate: str, baseline: str, provenance: str,
    pairs: int, first_seed: str, second_seed: str,
) -> tuple[dict[str, Any], str]:
    value, digest = _rebuild_reciprocal(path, 1.645)
    _validate_panel_identity(
        value, candidate, baseline, provenance, pairs,
        first_seed, second_seed,
    )
    return value, digest


def stage1_selection(
    panels: dict[str, dict[str, Any]], actors: dict[str, str],
    digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    if set(panels) != set(VARIANTS) or set(actors) != set(VARIANTS):
        raise EvidenceError("stage 1 requires all four and only four variants")
    if len(set(actors.values())) != 4:
        raise EvidenceError("stage-1 candidate actors are not distinct")
    digest_map = digests or {name: "0" * 64 for name in VARIANTS}
    if set(digest_map) != set(VARIANTS) or any(
            _HEX64.fullmatch(value) is None for value in digest_map.values()):
        raise EvidenceError("stage-1 result digest set is invalid")
    evidence: dict[str, Any] = {}
    for name in VARIANTS:
        exact = _panel_exact_statistics(panels[name])
        if exact["pairs_per_orientation"] != 200 or exact["capped_rounds"]:
            raise EvidenceError("stage-1 panel is incomplete or capped")
        result = panels[name].get("candidate_result")
        if not isinstance(result, dict):
            raise EvidenceError("stage-1 candidate result missing")
        evidence[name] = {
            **exact,
            "match_score": result.get("match_score"),
            "margin_per_game": result.get("margin_per_game"),
            "reciprocal_sha256": digest_map[name],
            "actor": actors[name],
        }
    rank = {name: len(TIE_PRIORITY) - TIE_PRIORITY.index(name)
            for name in TIE_PRIORITY}
    winner = max(
        VARIANTS,
        key=lambda name: (
            evidence[name]["combined_score_numerator"],
            evidence[name]["combined_margin_numerator"],
            rank[name],
        ),
    )
    return {
        "schema_version": 1,
        "artifact_kind": "match_value_stage1_mechanical_selection",
        "status": "complete_four_variant_reciprocal_panel",
        "selection_rule": (
            "maximize exact equal-weight reciprocal match score; then exact "
            "equal-weight margin; then fixed order P14,R14,P0,R0"
        ),
        "evidence": evidence,
        "selected_variant": winner,
        "selected_actor": actors[winner],
        "promotion_claim": False,
    }


def stage2_gate(value: dict[str, Any]) -> dict[str, Any]:
    exact = _panel_exact_statistics(value)
    if exact["pairs_per_orientation"] != 500:
        raise EvidenceError("stage-2 panel does not contain 500 pairs/orientation")
    n = exact["pairs_per_orientation"]
    requirements = {
        "complete_equal_reciprocal_blocks": True,
        "raw_inputs_validated":
            value.get("raw_input_validation", {}).get("status") == "validated",
        "zero_capped_rounds": exact["capped_rounds"] == 0,
        "combined_match_score_strictly_above_half":
            exact["combined_score_numerator"] >
            exact["combined_score_denominator"] // 2,
        "candidate_first_match_score_strictly_above_half":
            exact["candidate_first_score_quarters"] > 2 * n,
        "baseline_first_inverted_match_score_strictly_above_half":
            exact["baseline_first_score_quarters"] < 2 * n,
        "combined_margin_strictly_positive":
            exact["combined_margin_numerator"] > 0,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "match_value_stage2_development_gate",
        "status": "complete_development_confirmation",
        "exact_statistics": exact,
        "requirements": requirements,
        "passed": all(requirements.values()),
        "promotion_claim": False,
    }


def final_gate(value: dict[str, Any], critical_z: float = 1.645) -> dict[str, Any]:
    if not math.isfinite(critical_z) or critical_z != 1.645:
        raise EvidenceError(
            "final 95% one-sided critical value must remain exactly 1.645"
        )
    exact = _panel_exact_statistics(value)
    if exact["pairs_per_orientation"] != 2500:
        raise EvidenceError("final panel does not contain 2500 pairs/orientation")
    result = value.get("candidate_result")
    if not isinstance(result, dict):
        raise EvidenceError("final candidate result missing")
    numeric = (
        "match_score", "match_score_pair_clustered_se", "margin_per_game",
        "margin_pair_clustered_se",
    )
    if any(isinstance(result.get(field), bool) or
           not isinstance(result.get(field), (int, float)) or
           not math.isfinite(float(result[field])) for field in numeric):
        raise EvidenceError("final result has invalid numeric estimates")
    orientations = result.get("orientation_match_scores")
    if not isinstance(orientations, list) or len(orientations) != 2 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or
            not math.isfinite(float(item)) for item in orientations):
        raise EvidenceError("final result has invalid orientation estimates")
    score = float(result["match_score"])
    score_se = float(result["match_score_pair_clustered_se"])
    margin = float(result["margin_per_game"])
    score_lower = score - critical_z * score_se
    requirements = {
        "complete_equal_reciprocal_blocks": True,
        "raw_inputs_validated":
            value.get("raw_input_validation", {}).get("status") == "validated",
        "zero_capped_rounds": exact["capped_rounds"] == 0,
        "pair_clustered_orientation_stratified_score_lcb_above_half":
            score_lower > 0.5,
        "combined_match_score_point_estimate_above_half": score > 0.5,
        "each_reciprocal_orientation_strictly_above_half":
            all(float(item) > 0.5 for item in orientations),
        "combined_margin_strictly_positive": margin > 0.0,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "match_value_reserved_final_gate",
        "status": "complete_reserved_final_test",
        "critical_z": critical_z,
        "confidence_bound": (
            "95% one-sided lower bound; equivalently, the lower endpoint "
            "of a 90% two-sided confidence interval"
        ),
        "estimator": (
            "mirrored-pair clusters within orientation; independent "
            "reciprocal orientations combined with equal weight"
        ),
        "exact_statistics": exact,
        "candidate_result": {
            "match_score": score,
            "match_score_pair_clustered_se": score_se,
            "match_score_lower_bound": score_lower,
            "margin_per_game": margin,
            "margin_pair_clustered_se": float(
                result["margin_pair_clustered_se"]),
            "orientation_match_scores": [float(item) for item in orientations],
        },
        "requirements": requirements,
        "promotion_gate_passed": all(requirements.values()),
        "repository_promotion_performed": False,
    }


def _assignment(values: list[str], allowed: tuple[str, ...], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or name not in allowed or not item or name in result:
            raise EvidenceError(f"invalid {label} assignment {value!r}")
        result[name] = item
    if set(result) != set(allowed):
        raise EvidenceError(f"{label} assignments are incomplete")
    return result


def _github_output(path: Path | None, values: dict[str, str]) -> None:
    if path is None:
        return
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise EvidenceError("multiline GitHub output is forbidden")
            stream.write(f"{key}={value}\n")
        stream.flush()
        os.fsync(stream.fileno())


def post_build_manifest(
    root: Path, execution_path: Path, table_path: Path, actors_path: Path,
    builder_log: Path, build_info: Path, source_commit: str, source_tree: str,
) -> dict[str, Any]:
    execution = strict_json(execution_path)
    tables = strict_json(table_path)
    actors = strict_json(actors_path)
    if tables.get("status") != "complete_valid_single_transition_generation" or \
            actors.get("artifact_kind") != "locked_match_value_variant_actors":
        raise EvidenceError("post-build inputs are not complete")
    winner = execution.get("authoritative_world800_result", {})
    if actors.get("world_cap") != winner.get("selected_world_cap") or \
            actors.get("actors", {}).get("legacy") != winner.get("selected_actor"):
        raise EvidenceError("post-build actor does not match world800 winner")
    return {
        "schema_version": 1,
        "artifact_kind": "match_value_pre_efficacy_build_manifest",
        "status": "tables_and_actors_frozen_before_first_efficacy_match",
        "source": {"commit": source_commit, "tree": source_tree},
        "bindings": {
            PLAN_PATH: sha256(root / PLAN_PATH),
            WORKFLOW_PATH: sha256(root / WORKFLOW_PATH),
            str(execution_path): sha256(execution_path),
            WORLD_RESULT_PATH: sha256(root / WORLD_RESULT_PATH),
            str(table_path): sha256(table_path),
            str(actors_path): sha256(actors_path),
            str(builder_log): sha256(builder_log),
            str(build_info): sha256(build_info),
        },
        "build": execution["build"],
        "tables": tables,
        "actors": actors,
        "evaluation": execution["evaluation"],
        "results": None,
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="command", required=True)

    guard = command.add_parser("guard-execution")
    guard.add_argument("--root", type=Path, default=Path("."))
    guard.add_argument("--execution", type=Path, required=True)
    guard.add_argument("--source-commit", required=True)
    guard.add_argument("--source-tree", required=True)
    guard.add_argument("--output", type=Path)

    tables = command.add_parser("table-manifest")
    tables.add_argument("--raw", type=Path, required=True)
    tables.add_argument("--projected", type=Path, required=True)
    tables.add_argument("--output", type=Path, required=True)

    actors = command.add_parser("actors")
    actors.add_argument("--baseline", required=True)
    actors.add_argument("--world-cap", type=int, choices=(512, 800), required=True)
    actors.add_argument("--raw-path", required=True)
    actors.add_argument("--projected-path", required=True)
    actors.add_argument("--output", type=Path, required=True)

    post = command.add_parser("post-build-manifest")
    post.add_argument("--root", type=Path, required=True)
    post.add_argument("--execution", type=Path, required=True)
    post.add_argument("--tables", type=Path, required=True)
    post.add_argument("--actors", type=Path, required=True)
    post.add_argument("--builder-log", type=Path, required=True)
    post.add_argument("--build-info", type=Path, required=True)
    post.add_argument("--source-commit", required=True)
    post.add_argument("--source-tree", required=True)
    post.add_argument("--output", type=Path, required=True)

    select = command.add_parser("select-stage1")
    select.add_argument("--panel", action="append", default=[])
    select.add_argument("--actor", action="append", default=[])
    select.add_argument("--provenance", action="append", default=[])
    select.add_argument("--baseline", required=True)
    select.add_argument("--candidate-first-seed", required=True)
    select.add_argument("--baseline-first-seed", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--github-output", type=Path)

    for name in ("gate-stage2", "gate-final"):
        gate = command.add_parser(name)
        gate.add_argument("--reciprocal", type=Path, required=True)
        gate.add_argument("--candidate", required=True)
        gate.add_argument("--baseline", required=True)
        gate.add_argument("--provenance", required=True)
        gate.add_argument("--candidate-first-seed", required=True)
        gate.add_argument("--baseline-first-seed", required=True)
        gate.add_argument("--output", type=Path, required=True)
        gate.add_argument("--github-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    try:
        if args.command == "guard-execution":
            value = guard_execution(
                args.root, args.execution, args.source_commit, args.source_tree,
            )
            if args.output is not None:
                _write_json(args.output, value)
        elif args.command == "table-manifest":
            _write_json(args.output, table_manifest(args.raw, args.projected))
        elif args.command == "actors":
            _write_json(args.output, build_actors(
                args.baseline, args.world_cap, args.raw_path,
                args.projected_path,
            ))
        elif args.command == "post-build-manifest":
            _write_json(args.output, post_build_manifest(
                args.root, args.execution, args.tables, args.actors,
                args.builder_log, args.build_info,
                args.source_commit, args.source_tree,
            ))
        elif args.command == "select-stage1":
            paths = _assignment(args.panel, VARIANTS, "panel")
            actors = _assignment(args.actor, VARIANTS, "actor")
            provenances = _assignment(args.provenance, VARIANTS, "provenance")
            panels: dict[str, dict[str, Any]] = {}
            digests: dict[str, str] = {}
            for name in VARIANTS:
                panels[name], digests[name] = load_verified_panel(
                    Path(paths[name]), actors[name], args.baseline,
                    provenances[name], 200, args.candidate_first_seed,
                    args.baseline_first_seed,
                )
            value = stage1_selection(panels, actors, digests)
            _write_json(args.output, value)
            _github_output(args.github_output, {
                "selected_variant": value["selected_variant"],
                "selected_actor": value["selected_actor"],
                "selection_sha": sha256(args.output),
            })
        else:
            pairs = 500 if args.command == "gate-stage2" else 2500
            panel, reciprocal_sha = load_verified_panel(
                args.reciprocal, args.candidate, args.baseline,
                args.provenance, pairs, args.candidate_first_seed,
                args.baseline_first_seed,
            )
            value = stage2_gate(panel) if args.command == "gate-stage2" \
                else final_gate(panel)
            value.update({
                "candidate": args.candidate,
                "baseline": args.baseline,
                "provenance": args.provenance,
                "reciprocal_path": str(args.reciprocal),
                "reciprocal_sha256": reciprocal_sha,
                "seeds": {
                    "candidate_first": args.candidate_first_seed,
                    "baseline_first": args.baseline_first_seed,
                },
            })
            _write_json(args.output, value)
            passed = value.get("passed", value.get("promotion_gate_passed"))
            _github_output(args.github_output, {
                "passed": "true" if passed else "false",
                "decision_sha": sha256(args.output),
            })
    except (EvidenceError, OSError, ValueError, KeyError) as exc:
        print(f"match-value campaign: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
