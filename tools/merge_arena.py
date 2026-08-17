#!/usr/bin/env python3
"""Validate and merge exact, pair-addressed arena evidence.

Shard files are newline-delimited JSON written atomically by ``bin/arena``.
All estimates are recomputed from integer mirrored-pair outcomes; shard-level
means and standard errors are never averaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


class EvidenceError(ValueError):
    pass


UINT64_MAX = (1 << 64) - 1
LC_MAX_PLIES = 300


def _uint64_string(value: Any, field: str) -> int:
    if not isinstance(value, str) or not value or \
            (len(value) > 1 and value[0] == "0") or not value.isascii() or \
            not value.isdigit():
        raise EvidenceError(f"invalid canonical uint64 {field}")
    number = int(value)
    if number > UINT64_MAX:
        raise EvidenceError(f"out-of-range uint64 {field}")
    return number


def _integer(value: Any, field: str, low: int | None = None,
             high: int | None = None) -> int:
    if type(value) is not int:
        raise EvidenceError(f"invalid integer {field}")
    if low is not None and value < low or high is not None and value > high:
        raise EvidenceError(f"out-of-range integer {field}")
    return value


def read_shard(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    try:
        snapshot = path.read_bytes()
        records = [json.loads(line) for line in snapshot.decode("utf-8").splitlines()
                   if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{path}: unreadable shard: {exc}") from exc
    if len(records) < 2 or not isinstance(records[0], dict) or \
            not isinstance(records[-1], dict) or \
            records[0].get("record") != "meta" or \
            records[-1].get("record") != "complete":
        raise EvidenceError(f"{path}: missing metadata or completion footer")
    meta = records[0]
    if not isinstance(meta, dict):
        raise EvidenceError(f"{path}: invalid metadata")
    required_meta = {"record", "schema", "seed", "pair_start", "pair_count",
                     "rounds", "agent_a", "agent_b", "provenance"}
    if set(meta) != required_meta or meta.get("record") != "meta" or \
            _integer(meta.get("schema"), "schema") != 1:
        raise EvidenceError(f"{path}: unsupported or malformed metadata")
    _uint64_string(meta["seed"], "seed")
    start = _uint64_string(meta["pair_start"], "pair_start")
    count = _integer(meta["pair_count"], "pair_count", 1)
    rounds = _integer(meta["rounds"], "rounds", 1, 3)
    if start > UINT64_MAX - count:
        raise EvidenceError(f"{path}: overflowing pair range")
    if any(not isinstance(meta[field], str)
           for field in ("agent_a", "agent_b", "provenance")):
        raise EvidenceError(f"{path}: invalid string metadata")
    rows = records[1:-1]
    footer = records[-1]
    if not isinstance(footer, dict) or set(footer) != {"record", "pairs"} or \
            footer.get("record") != "complete" or \
            _integer(footer.get("pairs"), "complete pairs", 1) != len(rows) or \
            count != len(rows):
        raise EvidenceError(f"{path}: incomplete pair body")
    for offset, row in enumerate(rows):
        required_row = {"record", "index", "score_a", "score_b", "plies",
                        "capped_rounds"}
        if not isinstance(row, dict) or set(row) != required_row or \
                row.get("record") != "pair":
            raise EvidenceError(f"{path}: unexpected record in pair body")
        if _uint64_string(row["index"], "pair index") != start + offset:
            raise EvidenceError(f"{path}: non-contiguous or unordered rows")
        for field in ("score_a", "score_b", "plies", "capped_rounds"):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != 2 or \
                    any(type(value) is not int for value in values):
                raise EvidenceError(f"{path}: invalid {field} row")
        for value in row["score_a"] + row["score_b"]:
            _integer(value, "score", -(1 << 31), (1 << 31) - 1)
        for value in row["plies"]:
            _integer(value, "plies", 1, rounds * LC_MAX_PLIES)
        for value in row["capped_rounds"]:
            _integer(value, "capped_rounds", 0, rounds)
    return meta, rows, hashlib.sha256(snapshot).hexdigest()


def _sufficient(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    out = {
        "pairs": 0, "margin_sum": 0, "margin_sumsq": 0,
        "score_quarters_sum": 0, "score_quarters_sumsq": 0,
        "wins": 0, "losses": 0, "draws": 0,
        "points_a_sum": 0, "points_b_sum": 0, "plies_sum": 0,
        "capped_rounds": 0,
    }
    for row in rows:
        margin = sum(a - b for a, b in zip(row["score_a"], row["score_b"]))
        quarters = 0
        for a, b in zip(row["score_a"], row["score_b"]):
            if a > b:
                out["wins"] += 1
                quarters += 2
            elif a < b:
                out["losses"] += 1
            else:
                out["draws"] += 1
                quarters += 1
        out["pairs"] += 1
        out["margin_sum"] += margin
        out["margin_sumsq"] += margin * margin
        out["score_quarters_sum"] += quarters
        out["score_quarters_sumsq"] += quarters * quarters
        out["points_a_sum"] += sum(row["score_a"])
        out["points_b_sum"] += sum(row["score_b"])
        out["plies_sum"] += sum(row["plies"])
        out["capped_rounds"] += sum(row["capped_rounds"])
    return out


def _sample_se(n: int, total: int, total_sq: int, scale: float) -> float:
    if n <= 1:
        return 0.0
    variance = (total_sq - total * total / n) / (n - 1)
    return math.sqrt(max(0.0, variance) / n) / scale


def _result(s: dict[str, int]) -> dict[str, Any]:
    n = s["pairs"]
    if n <= 0:
        raise EvidenceError("empty evidence")
    games = 2 * n
    return {
        "pairs": n,
        "games": games,
        "margin_per_game": s["margin_sum"] / games,
        "margin_pair_clustered_se": _sample_se(
            n, s["margin_sum"], s["margin_sumsq"], 2.0),
        "match_score": s["score_quarters_sum"] / (4.0 * n),
        "match_score_pair_clustered_se": _sample_se(
            n, s["score_quarters_sum"],
            s["score_quarters_sumsq"], 4.0),
        "wins": s["wins"], "losses": s["losses"], "draws": s["draws"],
        "points_per_game_a": s["points_a_sum"] / games,
        "points_per_game_b": s["points_b_sum"] / games,
        "plies_per_game": s["plies_sum"] / games,
        "capped_rounds": s["capped_rounds"],
    }


def merge_block(paths: list[Path], expect_start: int, expect_pairs: int,
                allow_caps: bool = False) -> dict[str, Any]:
    if expect_start < 0 or expect_pairs <= 0 or not paths:
        raise EvidenceError("expected range and at least one shard are required")
    shards = []
    for path in paths:
        meta, rows, digest = read_shard(path)
        shards.append((_uint64_string(meta["pair_start"], "pair_start"),
                       path, meta, rows, digest))
    shards.sort(key=lambda item: item[0])
    stable_fields = ("schema", "seed", "rounds", "agent_a", "agent_b",
                     "provenance")
    reference = shards[0][2]
    if not reference.get("provenance"):
        raise EvidenceError("nonempty provenance is required")
    all_rows: list[dict[str, Any]] = []
    cursor = expect_start
    inputs = []
    for start, path, meta, rows, digest in shards:
        if any(meta.get(field) != reference.get(field) for field in stable_fields):
            raise EvidenceError(f"{path}: shard metadata mismatch")
        if start != cursor:
            raise EvidenceError(f"{path}: expected pair {cursor}, found {start}")
        all_rows.extend(rows)
        cursor += len(rows)
        inputs.append({"path": str(path), "sha256": digest,
                       "pair_start": start, "pair_count": len(rows)})
    if cursor != expect_start + expect_pairs:
        raise EvidenceError(
            f"expected {expect_pairs} pairs, merged {cursor - expect_start}")
    sufficient = _sufficient(all_rows)
    if sufficient["capped_rounds"] and not allow_caps:
        raise EvidenceError("cap-terminated round present in evidence")
    canonical = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in all_rows
    )
    return {
        "schema_version": 1,
        "artifact_kind": "merged_arena_pair_evidence",
        "metadata": {field: reference[field] for field in stable_fields},
        "pair_start": str(expect_start), "pair_count": expect_pairs,
        "inputs": inputs,
        "canonical_pair_rows_sha256": hashlib.sha256(canonical).hexdigest(),
        "sufficient_statistics": sufficient,
        "result": _result(sufficient),
    }


def _validated_block(block: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    required_block = {"schema_version", "artifact_kind", "metadata",
                      "pair_start", "pair_count", "inputs",
                      "canonical_pair_rows_sha256", "sufficient_statistics",
                      "result"}
    if not isinstance(block, dict) or set(block) != required_block or \
            type(block.get("schema_version")) is not int or \
            block.get("schema_version") != 1 or \
            block.get("artifact_kind") != "merged_arena_pair_evidence":
        raise EvidenceError("reciprocal inputs must be schema-1 merged blocks")
    metadata = block.get("metadata")
    stable_fields = {"schema", "seed", "rounds", "agent_a", "agent_b",
                     "provenance"}
    if not isinstance(metadata, dict) or set(metadata) != stable_fields or \
            metadata.get("schema") != 1:
        raise EvidenceError("malformed merged metadata")
    _uint64_string(metadata.get("seed"), "seed")
    _integer(metadata.get("rounds"), "rounds", 1, 3)
    if any(not isinstance(metadata.get(field), str)
           for field in ("agent_a", "agent_b", "provenance")) or \
            not metadata.get("provenance"):
        raise EvidenceError("malformed merged string metadata")
    start = _uint64_string(block.get("pair_start"), "pair_start")
    count = _integer(block.get("pair_count"), "pair_count", 1)
    if start > UINT64_MAX - count:
        raise EvidenceError("overflowing merged pair range")
    inputs = block.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise EvidenceError("missing merged input provenance")
    cursor = start
    for source in inputs:
        if not isinstance(source, dict) or set(source) != {
                "path", "sha256", "pair_start", "pair_count"} or \
                not isinstance(source.get("path"), str) or \
                type(source.get("pair_start")) is not int or \
                type(source.get("pair_count")) is not int or \
                source["pair_start"] != cursor or source["pair_count"] <= 0 or \
                not isinstance(source.get("sha256"), str) or \
                len(source["sha256"]) != 64 or any(
                    c not in "0123456789abcdef" for c in source["sha256"]):
            raise EvidenceError("malformed merged input provenance")
        cursor += source["pair_count"]
    if cursor != start + count:
        raise EvidenceError("merged input provenance does not cover range")
    sufficient = block.get("sufficient_statistics")
    keys = {"pairs", "margin_sum", "margin_sumsq", "score_quarters_sum",
            "score_quarters_sumsq", "wins", "losses", "draws",
            "points_a_sum", "points_b_sum", "plies_sum", "capped_rounds"}
    if not isinstance(sufficient, dict) or set(sufficient) != keys or \
            any(type(value) is not int for value in sufficient.values()):
        raise EvidenceError("malformed sufficient statistics")
    n = sufficient["pairs"]
    if n != count or n <= 0 or sufficient["wins"] < 0 or \
            sufficient["losses"] < 0 or sufficient["draws"] < 0 or \
            sufficient["wins"] + sufficient["losses"] + \
            sufficient["draws"] != 2 * n or \
            not 0 <= sufficient["score_quarters_sum"] <= 4 * n or \
            sufficient["margin_sumsq"] < 0 or \
            sufficient["score_quarters_sumsq"] < 0 or \
            n * sufficient["margin_sumsq"] < sufficient["margin_sum"] ** 2 or \
            n * sufficient["score_quarters_sumsq"] < \
            sufficient["score_quarters_sum"] ** 2 or \
            sufficient["plies_sum"] < 2 * n or \
            not 0 <= sufficient["capped_rounds"] <= \
            2 * n * metadata["rounds"]:
        raise EvidenceError("inconsistent sufficient statistics")
    recomputed = _result(sufficient)
    if block.get("result") != recomputed:
        raise EvidenceError("merged result does not match sufficient statistics")
    digest = block.get("canonical_pair_rows_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or \
            any(c not in "0123456789abcdef" for c in digest):
        raise EvidenceError("invalid canonical row digest")
    return metadata, recomputed


def combine_reciprocal(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    ma, ra = _validated_block(first)
    mb, rb = _validated_block(second)
    if ma["agent_a"] != mb["agent_b"] or ma["agent_b"] != mb["agent_a"]:
        raise EvidenceError("reciprocal agent orientations do not match")
    if ma["rounds"] != mb["rounds"] or ma["provenance"] != mb["provenance"]:
        raise EvidenceError("reciprocal provenance or round count mismatch")
    if first["pair_count"] != second["pair_count"]:
        raise EvidenceError("locked reciprocal blocks must have equal size")
    if ma["seed"] == mb["seed"]:
        a0, a1 = int(first["pair_start"]), \
                 int(first["pair_start"]) + first["pair_count"]
        b0, b1 = int(second["pair_start"]), \
                 int(second["pair_start"]) + second["pair_count"]
        if max(a0, b0) < min(a1, b1):
            raise EvidenceError("reciprocal blocks reuse overlapping RNG pairs")
    if ra["capped_rounds"] or rb["capped_rounds"]:
        raise EvidenceError("cap-terminated reciprocal evidence")
    margin = (ra["margin_per_game"] - rb["margin_per_game"]) / 2.0
    margin_se = math.hypot(ra["margin_pair_clustered_se"],
                           rb["margin_pair_clustered_se"]) / 2.0
    score_b_as_candidate = 1.0 - rb["match_score"]
    score = (ra["match_score"] + score_b_as_candidate) / 2.0
    score_se = math.hypot(ra["match_score_pair_clustered_se"],
                          rb["match_score_pair_clustered_se"]) / 2.0
    lower = score - 1.645 * score_se
    return {
        "schema_version": 1,
        "artifact_kind": "locked_reciprocal_arena_result",
        "provenance": ma["provenance"],
        "candidate": ma["agent_a"], "baseline": ma["agent_b"],
        "blocks": [first, second],
        "candidate_result": {
            "margin_per_game": margin,
            "margin_pair_clustered_se": margin_se,
            "match_score": score,
            "match_score_pair_clustered_se": score_se,
            "one_sided_95_lower_bound": lower,
            "orientation_match_scores": [ra["match_score"], score_b_as_candidate],
            "wins": ra["wins"] + rb["losses"],
            "losses": ra["losses"] + rb["wins"],
            "draws": ra["draws"] + rb["draws"],
            "capped_rounds": 0,
        },
        "promotion_gate_passed": lower > 0.5 and
            ra["match_score"] > 0.5 and score_b_as_candidate > 0.5,
        "estimator": "equal-weight reciprocal blocks; second orientation inverted",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    fd = -1
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise EvidenceError(f"{path}: refusing to replace existing artifact") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    block = sub.add_parser("block", help="merge shards for one orientation")
    block.add_argument("--expect-start", type=int, required=True)
    block.add_argument("--expect-pairs", type=int, required=True)
    block.add_argument("--allow-caps", action="store_true")
    block.add_argument("--output", type=Path, required=True)
    block.add_argument("shards", nargs="+", type=Path)
    reciprocal = sub.add_parser("reciprocal", help="combine two complete orientations")
    reciprocal.add_argument("--first", type=Path, required=True)
    reciprocal.add_argument("--second", type=Path, required=True)
    reciprocal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "block":
            value = merge_block(args.shards, args.expect_start,
                                args.expect_pairs, args.allow_caps)
        else:
            value = combine_reciprocal(json.loads(args.first.read_text()),
                                       json.loads(args.second.read_text()))
        _write_json(args.output, value)
    except (EvidenceError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
