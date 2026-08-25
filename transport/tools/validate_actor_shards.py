#!/usr/bin/env python3
"""Fail-closed structural validation for a complete reciprocal shard set.

This validator intentionally emits no game estimate.  It checks the exact file
set, sidecars, timing exit status, JSONL structure, ranges, actor orientation,
seed, and provenance before the hardened merger is allowed to inspect the
complete panel.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
import sys
from typing import Any

if __package__:
    from tools.merge_arena import EvidenceError, _write_json, read_shard
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from merge_arena import EvidenceError, _write_json, read_shard  # type: ignore[no-redef]


UINT64_MAX = (1 << 64) - 1
_TIME_RE = re.compile(
    r"wall_s=[0-9]+(?:\.[0-9]+)? "
    r"user_s=[0-9]+(?:\.[0-9]+)? "
    r"sys_s=[0-9]+(?:\.[0-9]+)? "
    r"max_rss_kb=[0-9]+ exit=0\n")


def _decimal(text: str, field: str, maximum: int = UINT64_MAX) -> int:
    if not text or not text.isascii() or not text.isdigit() or \
            (len(text) > 1 and text[0] == "0"):
        raise argparse.ArgumentTypeError(f"{field} must be canonical decimal")
    value = int(text)
    if value > maximum:
        raise argparse.ArgumentTypeError(f"{field} is out of range")
    return value


def _starts(text: str) -> list[int]:
    values = [_decimal(item, "shard start") for item in text.split(",")]
    if not values or len(set(values)) != len(values) or values != sorted(values):
        raise argparse.ArgumentTypeError("starts must be unique and increasing")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shards(
        directory: Path, candidate: str, baseline: str, provenance: str,
        candidate_seed: str, baseline_seed: str, starts: list[int],
        pairs_per_shard: int, rounds: int = 3) -> dict[str, Any]:
    if not directory.is_dir() or not candidate or not baseline or \
            not provenance or candidate == baseline or not starts or \
            pairs_per_shard <= 0 or not 1 <= rounds <= 3:
        raise EvidenceError("invalid locked shard-set configuration")
    for seed, label in ((candidate_seed, "candidate seed"),
                        (baseline_seed, "baseline seed")):
        try:
            _decimal(seed, label)
        except argparse.ArgumentTypeError as exc:
            raise EvidenceError(str(exc)) from exc

    expected_stems = {
        f"{orientation}-{start}"
        for orientation in ("candidate-first", "baseline-first")
        for start in starts
    }
    expected_names = {
        stem + suffix
        for stem in expected_stems
        for suffix in (".jsonl", ".sha256", ".time")
    }
    actual_names = {
        item.name for item in directory.iterdir() if item.is_file()
    }
    nonfiles = [item.name for item in directory.iterdir() if not item.is_file()]
    if actual_names != expected_names or nonfiles:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names) + sorted(nonfiles)
        raise EvidenceError(
            f"locked shard file-set mismatch; missing={missing!r}, extra={extra!r}")

    records: list[dict[str, Any]] = []
    for orientation in ("candidate-first", "baseline-first"):
        agent_a, agent_b, seed = (
            (candidate, baseline, candidate_seed)
            if orientation == "candidate-first"
            else (baseline, candidate, baseline_seed)
        )
        for start in starts:
            stem = f"{orientation}-{start}"
            raw = directory / f"{stem}.jsonl"
            metadata, rows, digest = read_shard(raw)
            required = {
                "record": "meta",
                "schema": 1,
                "seed": seed,
                "pair_start": str(start),
                "pair_count": pairs_per_shard,
                "rounds": rounds,
                "agent_a": agent_a,
                "agent_b": agent_b,
                "provenance": provenance,
            }
            if metadata != required or len(rows) != pairs_per_shard:
                raise EvidenceError(f"{raw}: locked metadata mismatch")
            sidecar = directory / f"{stem}.sha256"
            expected_sidecar = f"{digest}  raw/{stem}.jsonl\n"
            try:
                sidecar_text = sidecar.read_text(encoding="ascii")
                timing = (directory / f"{stem}.time").read_text(
                    encoding="ascii")
            except (OSError, UnicodeError) as exc:
                raise EvidenceError(f"{stem}: unreadable sidecar: {exc}") from exc
            if sidecar_text != expected_sidecar or _sha256(raw) != digest:
                raise EvidenceError(f"{stem}: SHA-256 sidecar mismatch")
            if _TIME_RE.fullmatch(timing) is None:
                raise EvidenceError(f"{stem}: invalid or unsuccessful timing record")
            records.append({
                "orientation": orientation,
                "pair_start": start,
                "pair_count": pairs_per_shard,
                "raw_path": str(raw),
                "raw_sha256": digest,
                "sidecar_sha256": _sha256(sidecar),
                "timing_sha256": _sha256(directory / f"{stem}.time"),
            })

    return {
        "schema_version": 1,
        "artifact_kind": "validated_locked_actor_shard_set",
        "status": "complete_structurally_valid_before_efficacy_merge",
        "candidate": candidate,
        "baseline": baseline,
        "provenance": provenance,
        "rounds": rounds,
        "pairs_per_shard": pairs_per_shard,
        "shards_per_orientation": len(starts),
        "pairs_per_orientation": len(starts) * pairs_per_shard,
        "starts": starts,
        "seeds": {
            "candidate_first": candidate_seed,
            "baseline_first": baseline_seed,
        },
        "shards": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--candidate-first-seed", required=True)
    parser.add_argument("--baseline-first-seed", required=True)
    parser.add_argument("--starts", type=_starts, required=True)
    parser.add_argument("--pairs-per-shard", type=lambda value:
                        _decimal(value, "pairs per shard", (1 << 31) - 1),
                        required=True)
    parser.add_argument("--rounds", type=lambda value:
                        _decimal(value, "rounds", 3), default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = validate_shards(
            args.directory, args.candidate, args.baseline, args.provenance,
            args.candidate_first_seed, args.baseline_first_seed,
            args.starts, args.pairs_per_shard, args.rounds)
        _write_json(args.output, value)
    except (EvidenceError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
