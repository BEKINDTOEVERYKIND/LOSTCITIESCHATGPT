#!/usr/bin/env python3
"""Fail-closed helpers for the locked three-action-core actor campaign.

This campaign has no training or position-specific selection step.  It reads
the independently archived world-800 result, takes that result's mechanically
selected actor, and changes exactly rollout field 35 (``action_core_count``)
from zero to three.  The helper makes that narrow comparison machine-checkable
before any match may run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

if __package__:
    from tools.match_value_campaign import _world_result
    from tools.merge_arena import EvidenceError, _write_json
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from match_value_campaign import _world_result  # type: ignore[no-redef]
    from merge_arena import EvidenceError, _write_json  # type: ignore[no-redef]


PLAN_PATH = "data/experiments/locked_action_core_shortlist_plan.json"
WORKFLOW_PATH = ".github/workflows/action-core-shortlist.yml"
EXECUTION_PATH = (
    "data/experiments/locked_action_core_shortlist_execution.json"
)
WORLD_RESULT_PATH = "data/experiments/world800_result.json"
MODEL_PATH = "data/champion.bin"
MODEL_SHA256 = (
    "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"
)
MODEL_SIZE = 2823748
COMPILER = "gcc"
COMPILER_VERSION_COMMAND = "gcc -dumpfullversion -dumpversion"
COMPILER_VERSION = "13.3.0"
CFLAGS = (
    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
    "-Wall -Wextra -std=c11"
)
LDFLAGS = "-lm -pthread"
THREADS = 4
ROUNDS = 3
SAFETY_PAIRS = 200
SAFETY_PAIRS_PER_SHARD = 20
SAFETY_STARTS = list(range(0, SAFETY_PAIRS, SAFETY_PAIRS_PER_SHARD))
FINAL_PAIRS = 2500
FINAL_PAIRS_PER_SHARD = 100
FINAL_STARTS = list(range(0, FINAL_PAIRS, FINAL_PAIRS_PER_SHARD))
SAFETY_CANDIDATE_SEED = "202609040101"
SAFETY_BASELINE_SEED = "202609040102"
FINAL_CANDIDATE_SEED = "202609040201"
FINAL_BASELINE_SEED = "202609040202"
GATE_Z = 1.645

SOURCE_FILES = (
    "Makefile",
    "src/agent.c",
    "src/agent.h",
    "src/features.c",
    "src/features.h",
    "src/heuristic.c",
    "src/heuristic.h",
    "src/late_resolver.c",
    "src/late_resolver.h",
    "src/lc.c",
    "src/lc.h",
    "src/match.c",
    "src/match.h",
    "src/match_value.c",
    "src/match_value.h",
    "src/net.c",
    "src/net.h",
    "src/planner.c",
    "src/planner.h",
    "src/rollout.c",
    "src/search.c",
    "src/search.h",
    "src/spec.c",
    "src/spec.h",
    "tools/action_core_campaign.py",
    "tools/arena.c",
    "tools/gate_actor_panel.py",
    "tools/merge_arena.py",
    "tools/validate_actor_shards.py",
)

# These names deliberately mirror src/spec.c.  A deployed winner currently
# spells fields 1..36 and inherits defaults for fields 37..42.  We normalize
# all 42 so the one-field experiment is checked semantically as well as by
# string construction.
ROLLOUT_FIELDS = (
    "worlds", "root_width", "candidate_floor", "gate", "min_candidates",
    "ply_lo", "ply_hi", "eval_candidates", "objective", "root_prune",
    "override_k", "override_min", "playout_sample", "root_symmetries",
    "policy_mass", "batch_worlds", "playout_symmetries", "discard_guard",
    "deck_max", "confirm_worlds", "playout_prune", "plan_deck_max",
    "plan_block_gap", "semantic_candidates", "confirm_exact5",
    "draw_variant_cores", "draw_variant_deck_max", "policy_prefix_mode",
    "belief_alpha", "draw_root_deck_max", "draw_playout_deck_max",
    "prefix_confirm_k", "prefix_confirm_min", "confirm_temp",
    "action_core_count", "exact_terminal", "deck2_replan_worlds",
    "deck2_replan_cores", "bounded_late_root", "bounded_late_min",
    "action_ranker_min", "match_value_path",
)
ROLLOUT_DEFAULTS: tuple[str | None, ...] = (
    "128", "4", "0.02", "0", "1", "0", "0", "0", "0", "0",
    "0", "4", "0", "1", "0", "0", "1", "0", "0", "256", "-1",
    "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0",
    "0", "0", "1", "0", "0", "0", "1", "0", None,
)
ACTION_CORE_FIELD = 34  # zero based; rollout field 35

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _reject_constant(token: str) -> None:
    raise EvidenceError(f"non-standard JSON constant {token}")


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise EvidenceError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
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


def _hex(value: str, bits: int, field: str) -> str:
    pattern = _HEX40 if bits == 160 else _HEX64
    if pattern.fullmatch(value) is None:
        raise EvidenceError(f"invalid {field}")
    return value


def normalized_rollout(spec: str) -> tuple[str, str, list[str | None]]:
    """Return family, model, and all 42 semantic rollout fields."""
    parts = spec.split(":")
    if len(parts) < 3 or parts[0] != "rolloutu" or \
            parts[1] != MODEL_PATH:
        raise EvidenceError("authoritative winner is not the expected uniform actor")
    explicit = parts[2:]
    if len(explicit) > len(ROLLOUT_FIELDS):
        raise EvidenceError("authoritative winner has unsupported rollout fields")
    fields = list(ROLLOUT_DEFAULTS)
    for index, value in enumerate(explicit):
        if value == "":
            raise EvidenceError("empty rollout field")
        fields[index] = value
    return parts[0], parts[1], fields


def build_actor_pair(winner: str) -> dict[str, Any]:
    family, model, baseline_fields = normalized_rollout(winner)
    parts = winner.split(":")
    explicit = parts[2:]
    if len(explicit) <= ACTION_CORE_FIELD:
        raise EvidenceError("authoritative winner does not bind action_core_count")
    if baseline_fields[ACTION_CORE_FIELD] != "0":
        raise EvidenceError("authoritative winner already enables action cores")
    if baseline_fields[0] not in {"512", "800"} or \
            baseline_fields[1] != "5" or \
            baseline_fields[2] != "0.02":
        raise EvidenceError("authoritative winner width/floor/world identity drift")
    if baseline_fields[5] != "14" or baseline_fields[27] != "3" or \
            baseline_fields[35] != "1":
        raise EvidenceError("authoritative winner controller identity drift")
    expected_late = ["0", "0", "0", "1", "0", None]
    if baseline_fields[36:] != expected_late:
        raise EvidenceError("authoritative winner has a late-field extension")

    candidate_parts = list(parts)
    candidate_parts[2 + ACTION_CORE_FIELD] = "3"
    candidate = ":".join(candidate_parts)
    _, _, candidate_fields = normalized_rollout(candidate)
    changed = [
        ROLLOUT_FIELDS[index]
        for index, (before, after) in enumerate(
            zip(baseline_fields, candidate_fields))
        if before != after
    ]
    if changed != ["action_core_count"] or \
            candidate_fields[ACTION_CORE_FIELD] != "3":
        raise EvidenceError("candidate construction changed more than action cores")
    return {
        "schema_version": 1,
        "artifact_kind": "locked_action_core_shortlist_actor_pair",
        "baseline": winner,
        "candidate": candidate,
        "family": family,
        "model_path": model,
        "world_cap": int(str(baseline_fields[0])),
        "rollout_field_count": 42,
        "changed_fields": changed,
        "field_comparison": [
            {"index": index + 1, "name": name,
             "baseline": baseline_fields[index],
             "candidate": candidate_fields[index]}
            for index, name in enumerate(ROLLOUT_FIELDS)
        ],
        "candidate_limit": {
            "root_width": 5,
            "policy_floor": 0.02,
            "action_core_count": 3,
            "maximum_complete_moves_evaluated": 5,
            "all_legal_moves_evaluated": False,
        },
    }


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1 or \
            plan.get("artifact_kind") != \
            "locked_action_core_shortlist_actor_campaign" or \
            plan.get("status") != \
            "inert_pending_archived_world800_result_and_add_only_binding":
        raise EvidenceError("unsupported action-core plan identity")
    method = plan.get("method")
    if not isinstance(method, dict) or \
            method.get("sole_candidate_action_core_count") != 3 or \
            method.get("root_width") != 5 or \
            method.get("complete_move_policy_floor") != 0.02 or \
            method.get("maximum_moves_evaluated") != 5 or \
            method.get("evaluates_all_legal_moves") is not False or \
            method.get("only_actor_field_changed") != "action_core_count":
        raise EvidenceError("action-core method preregistration drift")
    exclusion = plan.get("state_and_data_firewall")
    if not isinstance(exclusion, dict) or \
            exclusion.get("user_commented_states_used") is not False or \
            exclusion.get("position_specific_selection_used") is not False or \
            exclusion.get("match_source") != \
            "fresh seeded random deals generated only by arena":
        raise EvidenceError("action-core state firewall drift")
    invariants = plan.get("actor_invariants")
    build = plan.get("build_and_transport")
    if not isinstance(invariants, dict) or \
            invariants.get("model_path") != MODEL_PATH or \
            invariants.get("model_sha256") != MODEL_SHA256 or \
            invariants.get("root_width") != 5 or \
            invariants.get("candidate_floor") != 0.02 or \
            invariants.get("ply_low") != 14 or \
            invariants.get("policy_prefix_mode") != 3 or \
            invariants.get("root_symmetries") != 20 or \
            invariants.get("playout_symmetries") != 20 or \
            invariants.get("exact_terminal") != 1 or \
            invariants.get("action_ranker") is not None or \
            invariants.get("match_value_table") is not None or \
            invariants.get("rounds") != ROUNDS or \
            invariants.get("threads_per_shard") != THREADS:
        raise EvidenceError("action-core actor invariant drift")
    if not isinstance(build, dict) or build.get("runner") != "ubuntu-24.04" or \
            build.get("compiler") != COMPILER or \
            build.get("compiler_semantic_version_command") != \
            COMPILER_VERSION_COMMAND or \
            build.get("required_compiler_semantic_version") != \
            COMPILER_VERSION or build.get("cflags") != CFLAGS or \
            build.get("ldflags") != LDFLAGS or \
            build.get("compile_once") != ["bin/arena"]:
        raise EvidenceError("action-core build preregistration drift")
    safety = plan.get("safety_screen")
    final = plan.get("final_promotion")
    if not isinstance(safety, dict) or not isinstance(final, dict) or \
            safety.get("pairs_per_orientation") != SAFETY_PAIRS or \
            safety.get("pairs_per_shard") != SAFETY_PAIRS_PER_SHARD or \
            safety.get("pair_starts") != SAFETY_STARTS or \
            safety.get("candidate_first_seed") != SAFETY_CANDIDATE_SEED or \
            safety.get("baseline_first_seed") != SAFETY_BASELINE_SEED or \
            safety.get("rounds") != ROUNDS or \
            final.get("pairs_per_orientation") != FINAL_PAIRS or \
            final.get("pairs_per_shard") != FINAL_PAIRS_PER_SHARD or \
            final.get("pair_starts") != FINAL_STARTS or \
            final.get("candidate_first_seed") != FINAL_CANDIDATE_SEED or \
            final.get("baseline_first_seed") != FINAL_BASELINE_SEED or \
            final.get("rounds") != ROUNDS or \
            final.get("gate_z") != GATE_Z or \
            final.get("execute_only_if_safety_passes") is not True:
        raise EvidenceError("action-core schedule preregistration drift")


def expected_execution(root: Path, source_commit: str,
                       source_tree: str) -> dict[str, Any]:
    _hex(source_commit, 160, "source parent commit")
    _hex(source_tree, 160, "source parent tree")
    plan_path = root / PLAN_PATH
    workflow_path = root / WORKFLOW_PATH
    world_path = root / WORLD_RESULT_PATH
    model_path = root / MODEL_PATH
    plan = strict_json(plan_path)
    validate_plan(plan)
    _, passed, winner, worlds = _world_result(world_path)
    actors = build_actor_pair(winner)
    if actors["world_cap"] != worlds:
        raise EvidenceError("world800 selected cap and actor disagree")
    if sha256(model_path) != MODEL_SHA256 or model_path.stat().st_size != MODEL_SIZE:
        raise EvidenceError("champion model identity drift")
    source_hashes = {name: sha256(root / name) for name in SOURCE_FILES}
    return {
        "schema_version": 1,
        "artifact_kind": "locked_action_core_shortlist_execution",
        "status": "launch_bound_before_any_action_core_efficacy",
        "source_parent_commit": source_commit,
        "source_parent_tree": source_tree,
        "source_files_sha256": source_hashes,
        "workflow": {"path": WORKFLOW_PATH, "sha256": sha256(workflow_path)},
        "plan": {"path": PLAN_PATH, "sha256": sha256(plan_path)},
        "authoritative_world800_result": {
            "path": WORLD_RESULT_PATH,
            "sha256": sha256(world_path),
            "promotion_gate_passed": passed,
            "selected_world_cap": worlds,
            "selected_actor": winner,
        },
        "actors": actors,
        "build": {
            "runner": "ubuntu-24.04",
            "compiler": COMPILER,
            "compiler_semantic_version_command": COMPILER_VERSION_COMMAND,
            "required_compiler_semantic_version": COMPILER_VERSION,
            "cflags": CFLAGS,
            "ldflags": LDFLAGS,
            "compile_once": ["bin/arena"],
            "model": {"path": MODEL_PATH, "sha256": MODEL_SHA256,
                      "size": MODEL_SIZE},
        },
        "safety": {
            "pairs_per_orientation": SAFETY_PAIRS,
            "pairs_per_shard": SAFETY_PAIRS_PER_SHARD,
            "starts": SAFETY_STARTS,
            "candidate_first_seed": SAFETY_CANDIDATE_SEED,
            "baseline_first_seed": SAFETY_BASELINE_SEED,
            "gate": "score>=0.5; margin>0; each orientation>=0.475; zero caps; exact validity",
        },
        "final": {
            "execute_if_safety_passes": True,
            "pairs_per_orientation": FINAL_PAIRS,
            "pairs_per_shard": FINAL_PAIRS_PER_SHARD,
            "starts": FINAL_STARTS,
            "candidate_first_seed": FINAL_CANDIDATE_SEED,
            "baseline_first_seed": FINAL_BASELINE_SEED,
            "gate_z": GATE_Z,
            "gate": "score-z*SE>0.5; margin-z*SE>0; each orientation>0.5; zero caps; exact validity",
        },
        "inspection_rule": (
            "do not parse any stage efficacy until every expected immutable "
            "raw shard and sidecar for that stage validates"
        ),
        "results": None,
    }


def guard_execution(root: Path, execution_path: Path,
                    source_commit: str, source_tree: str) -> dict[str, Any]:
    actual = strict_json(execution_path)
    expected = expected_execution(root, source_commit, source_tree)
    if actual != expected:
        raise EvidenceError("execution addendum does not exactly match the locked campaign")
    return {
        "schema_version": 1,
        "artifact_kind": "action_core_shortlist_execution_validation",
        "execution_sha256": sha256(execution_path),
        "source_parent_commit": source_commit,
        "source_parent_tree": source_tree,
        "world800_result_sha256": expected["authoritative_world800_result"]["sha256"],
        "actors": expected["actors"],
        "valid": True,
    }


def freeze_manifest(root: Path, execution_path: Path, arena_path: Path,
                    source_commit: str, source_tree: str) -> dict[str, Any]:
    validation = guard_execution(
        root, execution_path, source_commit, source_tree)
    execution = strict_json(execution_path)
    actors = execution["actors"]
    arena_sha = sha256(arena_path)
    execution_sha = sha256(execution_path)
    plan_sha = execution["plan"]["sha256"]
    common = (
        f"plan={plan_sha};execution={execution_sha};"
        f"world800={execution['authoritative_world800_result']['sha256']};"
        f"source={source_commit};tree={source_tree};arena={arena_sha};"
        f"model={MODEL_SHA256};worlds={actors['world_cap']};threads={THREADS}"
    )
    return {
        "schema_version": 1,
        "artifact_kind": "frozen_action_core_shortlist_actor_panel",
        "status": "actors_and_transport_frozen_before_safety",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "plan_sha256": plan_sha,
        "execution_sha256": execution_sha,
        "world800_result_sha256": execution["authoritative_world800_result"]["sha256"],
        "arena_sha256": arena_sha,
        "model_sha256": MODEL_SHA256,
        "baseline": actors["baseline"],
        "candidate": actors["candidate"],
        "world_cap": actors["world_cap"],
        "changed_fields": actors["changed_fields"],
        "safety_provenance": "stage=action_core_shortlist_safety;" + common,
        "final_provenance": "stage=action_core_shortlist_final;" + common,
        "execution_validation": validation,
        "results": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="command", required=True)
    expected = command.add_parser("expected-execution")
    expected.add_argument("--root", type=Path, required=True)
    expected.add_argument("--source-commit", required=True)
    expected.add_argument("--source-tree", required=True)
    expected.add_argument("--output", type=Path, required=True)
    guard = command.add_parser("guard-execution")
    guard.add_argument("--root", type=Path, required=True)
    guard.add_argument("--execution", type=Path, required=True)
    guard.add_argument("--source-commit", required=True)
    guard.add_argument("--source-tree", required=True)
    guard.add_argument("--output", type=Path, required=True)
    freeze = command.add_parser("freeze")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--execution", type=Path, required=True)
    freeze.add_argument("--arena", type=Path, required=True)
    freeze.add_argument("--source-commit", required=True)
    freeze.add_argument("--source-tree", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "expected-execution":
            value = expected_execution(
                args.root, args.source_commit, args.source_tree)
        elif args.command == "guard-execution":
            value = guard_execution(
                args.root, args.execution,
                args.source_commit, args.source_tree)
        else:
            value = freeze_manifest(
                args.root, args.execution, args.arena,
                args.source_commit, args.source_tree)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, value)
    except (EvidenceError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
