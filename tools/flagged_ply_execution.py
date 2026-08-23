#!/usr/bin/env python3
"""Validate one-shot flagged-ply audit launches and emit workflow outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lc-flagged-ply-audit-execution-v1"
EXECUTION_PATH = Path("data/flagged_ply_audit_execution.json")
MANIFEST_PATH = ROOT / "data/user_reviewed_plies.json"
REQUIRED_KEYS = {
    "schema", "execute", "reference_actor", "winner_actor",
    "decision_worlds_per_actor_per_case", "belief_alpha", "base_seed",
    "history_worlds", "shard_count", "manifest_sha256",
}


class ExecutionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actor(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(
        ("rollout:", "rolloutu:", "rollout2:", "rolloutu2:",
         "rollout3:", "rolloutu3:", "rollout4:", "rolloutu4:")
    ):
        raise ExecutionError(f"{label} must be an explicit rollout actor spec")
    if any(ord(char) < 0x20 or char == "%" for char in value):
        raise ExecutionError(f"{label} contains an unsafe workflow character")
    return value


def load_execution(path: Path, *, require_execute: bool = True) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"cannot load execution addendum {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ExecutionError("unsupported execution-addendum schema")
    if set(value) != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - set(value))
        extra = sorted(set(value) - REQUIRED_KEYS)
        raise ExecutionError(f"execution fields differ (missing={missing}, extra={extra})")
    if not isinstance(value["execute"], bool):
        raise ExecutionError("execute must be a JSON boolean")
    if require_execute and value["execute"] is not True:
        raise ExecutionError("execution addendum is inert: execute is not true")
    reference = _actor(value["reference_actor"], "reference_actor")
    winner = _actor(value["winner_actor"], "winner_actor")
    if value["decision_worlds_per_actor_per_case"] != 16384:
        raise ExecutionError("the published campaign requires exactly 16384 worlds")
    if value["history_worlds"] != 20000:
        raise ExecutionError("the published campaign requires 20000 history worlds")
    if value["shard_count"] != 12:
        raise ExecutionError("the published campaign requires exactly 12 shards")
    seed = value["base_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise ExecutionError("base_seed must be an unsigned 64-bit integer")
    alpha = value["belief_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 5:
        raise ExecutionError("belief_alpha must be numeric and in [0,5]")
    manifest_sha = sha256(MANIFEST_PATH)
    if value["manifest_sha256"] != manifest_sha:
        raise ExecutionError(
            "execution addendum is bound to a different frozen corpus manifest"
        )
    return {
        **value,
        "reference_actor": reference,
        "winner_actor": winner,
        "execution_sha256": hashlib.sha256(raw).hexdigest(),
    }


def verify_one_shot_add(before: str, after: str) -> None:
    if len(before) != 40 or set(before) == {"0"}:
        raise ExecutionError("one-shot launch requires an existing parent commit")
    if len(after) != 40:
        raise ExecutionError("invalid launch commit")
    try:
        changed = subprocess.check_output(
            ["git", "diff", "--name-status", "--no-renames", before, after],
            cwd=ROOT, text=True, stderr=subprocess.STDOUT,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionError(f"cannot verify launch commit: {exc}") from exc
    expected = f"A\t{EXECUTION_PATH.as_posix()}"
    if changed != [expected]:
        raise ExecutionError(
            "push launch must add only data/flagged_ply_audit_execution.json; "
            f"observed {changed!r}"
        )


def emit_github_output(path: Path, config: dict[str, Any], mode: str) -> None:
    lines = {
        "reference_actor": config["reference_actor"],
        "winner_actor": config["winner_actor"],
        "worlds": str(config["decision_worlds_per_actor_per_case"]),
        "belief_alpha": str(config["belief_alpha"]),
        "history_worlds": str(config["history_worlds"]),
        "base_seed": str(config["base_seed"]),
        "shard_count": str(config["shard_count"]),
        "execution_sha256": config["execution_sha256"],
        "launch_mode": mode,
    }
    with path.open("a", encoding="utf-8") as handle:
        for key, value in lines.items():
            handle.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--allow-inert", action="store_true")
    args = parser.parse_args()
    try:
        if (args.before is None) != (args.after is None):
            raise ExecutionError("--before and --after must be supplied together")
        if args.before is not None:
            verify_one_shot_add(args.before, args.after)
        config = load_execution(
            args.execution, require_execute=not args.allow_inert
        )
        if args.github_output:
            emit_github_output(args.github_output, config, "push_addendum")
        else:
            print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    except ExecutionError as exc:
        print(f"flagged_ply_execution.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
