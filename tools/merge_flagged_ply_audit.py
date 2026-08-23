#!/usr/bin/env python3
"""Merge the manual flagged-ply matrix without changing its evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class MergeError(RuntimeError):
    pass


_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def strict_json_bytes(raw: bytes) -> Any:
    value = json.loads(
        raw,
        object_pairs_hook=_unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {token}")
        ),
    )
    _finite(value)
    return value


def load(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        result = strict_json_bytes(raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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
        if _HEX40.fullmatch(str(first.get("source_commit"))) is None or \
                _HEX40.fullmatch(str(first.get("source_tree"))) is None:
            raise MergeError("shards must record canonical source commit/tree")
        stable_keys = (
            "source_commit", "source_tree", "manifest_sha256", "reference",
            "candidate", "decision_worlds", "belief_alpha", "history_worlds",
            "base_seed", "shard_count", "candidate_rule", "world_model",
            "selection", "execution_sha256", "evaluator_manifest_sha256",
            "authoritative_result_sha256", "launch_mode",
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
        manifest_path = ROOT / "data/user_reviewed_plies.json"
        manifest_raw = manifest_path.read_bytes()
        manifest = strict_json_bytes(manifest_raw)
        if hashlib.sha256(manifest_raw).hexdigest() != first["manifest_sha256"]:
            raise MergeError("shard manifest hash differs from the frozen manifest")
        order = [case["id"] for case in manifest["cases"]]
        unknown = set(cases) - set(order)
        if unknown:
            raise MergeError(f"unknown cases: {sorted(unknown)}")
        if not args.allow_partial and set(cases) != set(order):
            raise MergeError(
                f"complete merge needs 36 cases, got {len(cases)}"
            )
        merged_cases = [cases[case_id] for case_id in order if case_id in cases]
        if not args.allow_partial:
            if first.get("launch_mode") != "addendum_push" or any(
                    _HEX64.fullmatch(str(first.get(key))) is None for key in (
                        "execution_sha256", "evaluator_manifest_sha256",
                        "authoritative_result_sha256")):
                raise MergeError("complete merge requires fully bound launch provenance")
            if (first.get("decision_worlds"), first.get("history_worlds"),
                    first.get("belief_alpha"), first.get("base_seed"),
                    first.get("shard_count")) != (
                        16384, 20000, 1.15, 202608231701, 12):
                raise MergeError("complete merge settings differ from the locked plan")
            by_id = {case["id"]: case for case in manifest["cases"]}
            kinds = Counter()
            for case in merged_cases:
                frozen = by_id[case["id"]]
                kinds[case.get("kind")] += 1
                if case.get("state") != frozen.get("state") or \
                        case.get("state_sha256") != frozen.get("state_sha256"):
                    raise MergeError(f"{case['id']}: frozen state binding drift")
                probe = case.get("probe")
                if not isinstance(probe, dict) or \
                        probe.get("schema") != "lc-flagged-ply-probe-v1":
                    raise MergeError(f"{case['id']}: malformed probe evidence")
                candidates = probe.get("candidates")
                evaluated = probe.get("evaluated_moves")
                legal = probe.get("legal_moves")
                actors = probe.get("actors")
                if not isinstance(candidates, list) or len(candidates) != evaluated or \
                        type(evaluated) is not int or type(legal) is not int or \
                        evaluated > 5 or (legal > 5 and evaluated >= legal) or \
                        not isinstance(actors, list) or len(actors) != 2:
                    raise MergeError(f"{case['id']}: candidate-panel contract drift")
                labels = {actor.get("label"): actor for actor in actors}
                if set(labels) != {"reference", "candidate"} or \
                        labels["reference"].get("spec") != first["reference"]["spec"] or \
                        labels["candidate"].get("spec") != first["candidate"]["spec"]:
                    raise MergeError(f"{case['id']}: actor identity drift")
                if frozen["kind"] == "belief":
                    if evaluated != 0 or candidates or any(
                            actor.get("action_panel") is not False or "rows" in actor
                            for actor in actors):
                        raise MergeError(f"{case['id']}: belief-only action work detected")
                else:
                    if any(actor.get("requested_worlds") != 16384 or
                           actor.get("action_panel") is not True or
                           not isinstance(actor.get("rows"), list) or
                           len(actor["rows"]) != evaluated for actor in actors) or \
                            actors[0].get("worlds") != actors[1].get("worlds"):
                        raise MergeError(f"{case['id']}: common-world evidence drift")
            if kinds != Counter({"decision": 34, "belief": 2}):
                raise MergeError("complete evidence is not 34 decisions plus 2 beliefs")
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
