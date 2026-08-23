#!/usr/bin/env python3
"""Prepare and verify the inert action-advantage campaign launch binding.

The maintained rollout actor is not chosen by editing this campaign after the
world-count test.  Instead, the authoritative, raw-validated world800 result
is reopened through the same verifier used by the match-value campaign.  Its
boolean gate mechanically selects the exact 512- or 800-world actor, and this
module derives the ranker candidate without changing any other rollout field.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.match_value_campaign import (  # noqa: E402
    BASELINE_512,
    CANDIDATE_800,
    EvidenceError,
    _world_result,
    sha256,
    strict_json,
)


PLAN_PATH = "data/experiments/locked_action_advantage_veto_v1_plan.json"
WORKFLOW_PATH = ".github/workflows/action-advantage-veto-v1.yml"
EXECUTION_PATH = (
    "data/experiments/locked_action_advantage_veto_v1_execution.json"
)
WORLD_RESULT_PATH = "data/experiments/world800_result.json"
MODEL_PATH = "data/champion.bin"
MODEL_SHA256 = (
    "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"
)
MODEL_SIZE = 2823748
RANKER_PATH = "data/models/action_advantage_veto_v1.bin"
TEACHER = f"policy:{MODEL_PATH}:0:20"
COMPILER = "gcc"
COMPILER_SEMANTIC_VERSION_COMMAND = "gcc -dumpfullversion -dumpversion"
REQUIRED_COMPILER_SEMANTIC_VERSION = "13.3.0"
CFLAGS = (
    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
    "-Wall -Wextra -std=c11"
)
LDFLAGS = "-lm -pthread"
DEVELOPMENT_SEED = "202609030101"
SPLIT_SEED = "202609030201"
SAFETY_CANDIDATE_SEED = "202609020301"
SAFETY_BASELINE_SEED = "202609020302"
FINAL_CANDIDATE_SEED = "202609020401"
FINAL_BASELINE_SEED = "202609020402"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def candidate_prefix(winner: str) -> str:
    """Derive rollout4 from the exact selected maintained actor.

    Both world-count candidates have the historical 36-field tail.  Fields
    37--40 are the unchanged defaults for recursive deck-two replanning,
    bounded-late enablement, and bounded-late minimum.  The selected heldout
    threshold is appended later as field 41 (zero-based tail index 40).
    """
    try:
        kind, model, tail = winner.split(":", 2)
    except ValueError as exc:
        raise EvidenceError("world-count winner has malformed actor syntax") from exc
    if kind != "rolloutu" or model != MODEL_PATH or winner not in {
            BASELINE_512, CANDIDATE_800}:
        raise EvidenceError("world-count winner is not an authorized actor")
    fields = tail.split(":")
    if len(fields) != 36:
        raise EvidenceError("world-count winner does not have a 36-field tail")
    world = "800" if winner == CANDIDATE_800 else "512"
    if fields[0] != world or fields[19] != world:
        raise EvidenceError("selected world counts are not in both panel fields")
    padded = fields + ["0", "0", "0", "1"]
    if len(padded) != 40:
        raise EvidenceError("cannot construct the pre-threshold rollout tail")
    return (
        f"rolloutu4:{MODEL_PATH}:{MODEL_PATH}:{RANKER_PATH}:"
        + ":".join(padded) + ":"
    )


def authoritative_inputs(root: Path) -> dict[str, Any]:
    world_path = root / WORLD_RESULT_PATH
    _, passed, winner, world_cap = _world_result(world_path)
    if winner != (CANDIDATE_800 if passed else BASELINE_512) or \
            world_cap != (800 if passed else 512):
        raise EvidenceError("world-count result selected an inconsistent actor")
    return {
        "path": WORLD_RESULT_PATH,
        "sha256": sha256(world_path),
        "candidate_actor": CANDIDATE_800,
        "baseline_actor": BASELINE_512,
        "promotion_gate_passed": passed,
        "selected_world_cap": world_cap,
        "selected_actor": winner,
        "candidate_prefix": candidate_prefix(winner),
    }


def expected_execution(root: Path, source_commit: str,
                       source_tree: str,
                       authoritative: dict[str, Any] | None = None,
                       ) -> dict[str, Any]:
    if _HEX40.fullmatch(source_commit) is None or \
            _HEX40.fullmatch(source_tree) is None:
        raise EvidenceError("source commit/tree must be canonical SHA-1 values")
    bound = authoritative if authoritative is not None \
        else authoritative_inputs(root)
    required = {
        "path", "sha256", "candidate_actor", "baseline_actor",
        "promotion_gate_passed", "selected_world_cap", "selected_actor",
        "candidate_prefix",
    }
    if not isinstance(bound, dict) or set(bound) != required or \
            bound["path"] != WORLD_RESULT_PATH or \
            not isinstance(bound["sha256"], str) or \
            _HEX64.fullmatch(bound["sha256"]) is None or \
            bound["candidate_actor"] != CANDIDATE_800 or \
            bound["baseline_actor"] != BASELINE_512 or \
            type(bound["promotion_gate_passed"]) is not bool or \
            bound["selected_actor"] != (
                CANDIDATE_800 if bound["promotion_gate_passed"]
                else BASELINE_512) or \
            bound["selected_world_cap"] != (
                800 if bound["promotion_gate_passed"] else 512) or \
            bound["candidate_prefix"] != candidate_prefix(
                bound["selected_actor"]):
        raise EvidenceError("malformed authoritative world-count binding")
    world_binding = {key: bound[key] for key in (
        "path", "sha256", "candidate_actor", "baseline_actor",
        "promotion_gate_passed", "selected_world_cap", "selected_actor",
    )}
    return {
        "schema_version": 1,
        "artifact_kind": "locked_action_advantage_veto_v1_execution",
        "status": (
            "launch_bound_after_world800_before_candidate_generation_or_"
            "actor_efficacy"
        ),
        "source_parent_commit": source_commit,
        "source_parent_tree": source_tree,
        "workflow": {
            "path": WORKFLOW_PATH,
            "sha256": sha256(root / WORKFLOW_PATH),
        },
        "plan": {
            "path": PLAN_PATH,
            "sha256": sha256(root / PLAN_PATH),
        },
        "authoritative_world800_result": world_binding,
        "build": {
            "runner": "ubuntu-24.04",
            "compiler": COMPILER,
            "compiler_semantic_version_command":
                COMPILER_SEMANTIC_VERSION_COMMAND,
            "required_compiler_semantic_version":
                REQUIRED_COMPILER_SEMANTIC_VERSION,
            "cflags": CFLAGS,
            "ldflags": LDFLAGS,
            "binding": (
                "compile exactly once in preflight; transport and SHA-256 "
                "verify everywhere else"
            ),
        },
        "development": {
            "generator_seed": DEVELOPMENT_SEED,
            "split_seed": SPLIT_SEED,
            "source_matches": 64,
            "label_worlds": 256,
            "label_threads": 4,
            "teacher": TEACHER,
            "threshold_grid": [0.0, 0.1, 0.25, 0.5, 1.0],
        },
        "actors": {
            "baseline": bound["selected_actor"],
            "candidate_prefix": bound["candidate_prefix"],
            "candidate_count": 1,
        },
        "safety": {
            "pairs_per_orientation": 200,
            "pairs_per_shard": 20,
            "starts": list(range(0, 200, 20)),
            "candidate_first_seed": SAFETY_CANDIDATE_SEED,
            "baseline_first_seed": SAFETY_BASELINE_SEED,
            "gate": (
                "score>=0.5; margin>0; each orientation>=0.475; zero caps; "
                "exact validity"
            ),
        },
        "final": {
            "execute_if_safety_passes": True,
            "pairs_per_orientation": 2500,
            "pairs_per_shard": 100,
            "starts": list(range(0, 2500, 100)),
            "candidate_first_seed": FINAL_CANDIDATE_SEED,
            "baseline_first_seed": FINAL_BASELINE_SEED,
            "gate_z": 1.645,
            "gate": (
                "score-z*SE>0.5; margin-z*SE>0; each orientation>0.5; "
                "zero caps; exact validity"
            ),
        },
        "probe_exclusion": (
            "no repository checkout after preflight; data/probes and "
            "user-commented positions are absent from training and "
            "actor-panel transports"
        ),
        "results": None,
    }


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        # A same-filesystem hard link is both atomic and no-clobber: the
        # canonical name is never visible with partial contents, and an
        # existing addendum cannot be replaced even briefly.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise EvidenceError(f"{path} already exists") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def prepare_execution(root: Path, output: Path, source_commit: str,
                      source_tree: str) -> dict[str, Any]:
    expected_path = (root / EXECUTION_PATH).resolve()
    if output.resolve() != expected_path:
        raise EvidenceError("execution output must use the canonical path")
    value = expected_execution(root, source_commit, source_tree)
    _atomic_create(output, value)
    return value


def guard_execution(root: Path, execution: Path, source_commit: str,
                    source_tree: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bound = authoritative_inputs(root)
    expected = expected_execution(
        root, source_commit, source_tree, authoritative=bound)
    if strict_json(execution) != expected:
        raise EvidenceError(
            "execution addendum does not exactly match the authoritative "
            "world-count result and locked campaign"
        )
    return expected, bound


def _github_outputs(path: Path, value: dict[str, Any]) -> None:
    lines = {
        "baseline": value["selected_actor"],
        "candidate_prefix": value["candidate_prefix"],
        "world_cap": str(value["selected_world_cap"]),
        "world_result_sha": value["sha256"],
        "world_passed": "true" if value["promotion_gate_passed"] else "false",
    }
    if any("\n" in item or "\r" in item for item in lines.values()):
        raise EvidenceError("multiline GitHub output is forbidden")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        for key, item in lines.items():
            stream.write(f"{key}={item}\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-execution", "guard-execution"):
        command = sub.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--source-parent-commit", required=True)
        command.add_argument("--source-parent-tree", required=True)
        command.add_argument("--execution", type=Path, required=True)
        if name == "guard-execution":
            command.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare-execution":
            prepare_execution(
                args.root, args.execution, args.source_parent_commit,
                args.source_parent_tree)
        else:
            _, bound = guard_execution(
                args.root, args.execution, args.source_parent_commit,
                args.source_parent_tree)
            _github_outputs(args.github_output, bound)
    except (EvidenceError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
