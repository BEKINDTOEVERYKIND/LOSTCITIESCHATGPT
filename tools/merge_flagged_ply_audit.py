#!/usr/bin/env python3
"""Merge the manual flagged-ply matrix without changing its evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class MergeError(RuntimeError):
    pass


def load(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        result = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"cannot load {path}: {exc}") from exc
    if result.get("schema") != "lc-flagged-ply-audit-v1":
        raise MergeError(f"{path}: wrong schema")
    return result, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        loaded = [(*load(path), path) for path in args.inputs]
        first = loaded[0][0]["provenance"]
        if not isinstance(first.get("source_commit"), str) or not first[
            "source_commit"
        ]:
            raise MergeError("shards must record a non-null source commit")
        stable_keys = (
            "source_commit", "manifest_sha256", "reference", "candidate", "decision_worlds",
            "belief_alpha", "history_worlds", "base_seed", "shard_count", "candidate_rule",
            "world_model", "selection", "execution_sha256",
        )
        shards: set[int] = set()
        cases: dict[str, dict[str, Any]] = {}
        artifacts: list[dict[str, str]] = []
        for result, digest, path in loaded:
            provenance = result.get("provenance", {})
            for key in stable_keys:
                if provenance.get(key) != first.get(key):
                    raise MergeError(f"{path}: provenance mismatch at {key}")
            shard = provenance.get("shard_index")
            if not isinstance(shard, int) or shard in shards:
                raise MergeError(f"{path}: duplicate/invalid shard index")
            shards.add(shard)
            if result.get("errors"):
                raise MergeError(f"{path}: shard contains worker errors")
            for case in result.get("cases", []):
                case_id = case.get("id")
                if not isinstance(case_id, str) or case_id in cases:
                    raise MergeError(f"{path}: duplicate/invalid case {case_id!r}")
                cases[case_id] = case
            artifacts.append({"path": str(path), "sha256": digest})
        shard_count = int(first["shard_count"])
        if not args.allow_partial and shards != set(range(shard_count)):
            raise MergeError(
                f"need shards 0..{shard_count - 1}, got {sorted(shards)}"
            )
        manifest = json.loads((ROOT / "data/user_reviewed_plies.json").read_text())
        order = [case["id"] for case in manifest["cases"]]
        unknown = set(cases) - set(order)
        if unknown:
            raise MergeError(f"unknown cases: {sorted(unknown)}")
        if not args.allow_partial and set(cases) != set(order):
            raise MergeError(
                f"complete merge needs 36 cases, got {len(cases)}"
            )
        merged_cases = [cases[case_id] for case_id in order if case_id in cases]
        counts: dict[str, Any] = {}
        for label in ("reference", "candidate"):
            counts[label] = {
                "policy": dict(Counter(
                    case["classifications"][label]["policy"]
                    for case in merged_cases
                )),
                "panel": dict(Counter(
                    case["classifications"][label]["panel"]
                    for case in merged_cases
                )),
                "deployed": dict(Counter(
                    case["classifications"][label]["deployed"]
                    for case in merged_cases
                )),
            }
        output = {
            "schema": "lc-flagged-ply-audit-merged-v1",
            "provenance": {
                **{key: first[key] for key in stable_keys},
                "source_commit": first.get("source_commit"),
                "merged_shards": sorted(shards),
                "source_artifacts": artifacts,
                "merge": "lossless case concatenation in frozen manifest order",
            },
            "completed_cases": len(merged_cases),
            "classification_counts": counts,
            "cases": merged_cases,
        }
        if args.output.exists() and not args.force:
            raise MergeError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
        return 0
    except MergeError as exc:
        print(f"merge_flagged_ply_audit.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
