#!/usr/bin/env python3
"""Fail-closed validation and selection for continuation-v2 screens.

The continuation arena's footer is useful as a human diagnostic, but model
selection never trusts it.  This tool validates every schema-2 row, binds the
raw evidence to exact artifact hashes through a canonical provenance string,
and recomputes all selection statistics from paired rows.

There are three intentionally small commands:

  validate   validate a Screen-A or Screen-B manifest and emit recomputed data
  screen-a   select one locked checkpoint in every (cell, replicate)
  screen-b   select one robust 2x2 cell, or report that none is eligible

All JSON inputs reject duplicate keys, non-finite numbers, unknown fields,
gaps, overlaps, and mixed provenance.  Output JSON is deterministic and never
contains NaN or Infinity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence


class EvidenceError(ValueError):
    """An operationally invalid evidence set."""


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
LABEL = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
UINT_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")
UINT64_MAX = (1 << 64) - 1
Z95 = 1.645
CHECKPOINTS = (
    "base",
    "warm2",
    "full2",
    "full4",
    "full6",
    "full8",
    "full10",
)

META_KEYS = {
    "record", "schema", "evidence_scope", "seed", "pair_start",
    "pair_count", "target_round", "continuation_objective",
    "round_0_1_semantics", "round_2_mode_2_semantics",
    "role_mapping_mode", "root_checkpoint", "candidate_checkpoint",
    "baseline_checkpoint", "root_ply", "root_symmetries", "root_width",
    "root_floor", "root_min", "root_mix", "world_model",
    "continuation_policy", "late_cycle", "pairing", "provenance",
}
PAIR_KEYS = {
    "record", "index", "round", "root_player", "admitted", "picked",
    "root_move", "cum_before", "cumulative_before", "player_mapping",
    "root_role_mapping", "opponent_role_mapping", "candidate_seat",
    "score_by_seat", "candidate_margin", "candidate_round_margin",
    "candidate_objective_target", "candidate_final_match_margin",
    "candidate_final_match_result", "candidate_hybrid_target",
    "tail_plies", "capped", "exact_moves", "cap_forces", "cycle_forces",
}
SUMMARY_KEYS = {
    "record", "continuation_objective",
    "configured_objective_aggregate_comparable",
    "configured_objective_per_leg",
    "configured_objective_pair_clustered_se", "rounds",
}
ROUND_KEYS = {
    "round", "pairs", "selection_semantics", "round_margin_per_leg",
    "round_margin_pair_clustered_se", "configured_objective_per_leg",
    "configured_objective_pair_clustered_se",
}
FINAL_ROUND_KEYS = ROUND_KEYS | {
    "final_match_margin_per_leg", "final_match_margin_pair_clustered_se",
    "match_score", "match_score_pair_clustered_se", "match_wins",
    "match_losses", "match_draws",
}
COMMON_MANIFEST_KEYS = {
    "schema", "stage", "plan_path", "plan_id", "arena_path", "arena_id",
    "root_model_id", "baseline_model_id", "root_checkpoint",
    "baseline_checkpoint", "seed", "pair_start", "pair_count",
    "target_round", "cells", "replicates", "evidence",
}


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise EvidenceError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _bad_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant: {value}")


def _loads_strict(text: str, source: str) -> Any:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_bad_constant,
        )
    except EvidenceError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise EvidenceError(f"{source}: malformed JSON: {exc}") from exc
    _finite_tree(value, source)
    return value


def _finite_tree(value: Any, where: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"{where}: non-finite number")
    if isinstance(value, list):
        for item in value:
            _finite_tree(item, where)
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item, where)


def _load_json(path: Path) -> Any:
    try:
        return _loads_strict(path.read_text(encoding="utf-8"), str(path))
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot read: {exc}") from exc


def _exact_keys(obj: Any, keys: set[str], where: str) -> dict[str, Any]:
    if type(obj) is not dict:
        raise EvidenceError(f"{where}: expected object")
    actual = set(obj)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise EvidenceError(
            f"{where}: field mismatch; missing={missing}, extra={extra}"
        )
    return obj


def _integer(value: Any, low: int, high: int, where: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise EvidenceError(f"{where}: expected integer in [{low}, {high}]")
    return value


def _number(value: Any, low: float, high: float, where: str) -> float:
    if type(value) not in (int, float):
        raise EvidenceError(f"{where}: expected finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise EvidenceError(f"{where}: number outside [{low}, {high}]")
    return result


def _string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise EvidenceError(f"{where}: expected nonempty string")
    return value


def _label(value: Any, where: str) -> str:
    text = _string(value, where)
    if not LABEL.fullmatch(text):
        raise EvidenceError(f"{where}: invalid canonical label")
    return text


def _hex_id(value: Any, where: str) -> str:
    text = _string(value, where)
    if not HEX64.fullmatch(text):
        raise EvidenceError(f"{where}: expected lowercase SHA-256")
    return text


def _uint_string(value: Any, where: str) -> int:
    text = _string(value, where)
    if not UINT_DECIMAL.fullmatch(text):
        raise EvidenceError(f"{where}: expected canonical unsigned decimal")
    result = int(text)
    if result > UINT64_MAX:
        raise EvidenceError(f"{where}: exceeds uint64")
    return result


def _array2(value: Any, where: str, low: int, high: int) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise EvidenceError(f"{where}: expected two-element array")
    return (
        _integer(value[0], low, high, f"{where}[0]"),
        _integer(value[1], low, high, f"{where}[1]"),
    )


def _float_array2(
    value: Any, where: str, low: float, high: float
) -> tuple[float, float]:
    if type(value) is not list or len(value) != 2:
        raise EvidenceError(f"{where}: expected two-element array")
    return (
        _number(value[0], low, high, f"{where}[0]"),
        _number(value[1], low, high, f"{where}[1]"),
    )


def _same_float(actual: Any, expected: float, where: str) -> None:
    got = _number(actual, -1e12, 1e12, where)
    # The C footer uses an ordered sum/sumsq variance while selectors use a
    # cancellation-resistant recomputation.  Permit only last-bit numerical
    # drift, never a material difference in the published statistic.
    tolerance = max(1e-10, abs(expected) * 1e-10)
    if abs(got - expected) > tolerance:
        raise EvidenceError(f"{where}: {got!r} != recomputed {expected!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot hash artifact: {exc}") from exc
    return digest.hexdigest()


def _canonical_relative(value: Any, where: str) -> str:
    literal = _string(value, where)
    path = Path(literal)
    if (
        path.is_absolute()
        or "\\" in literal
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != literal
    ):
        raise EvidenceError(f"{where}: expected canonical contained relative path")
    return literal


def _reject_symlink_chain(path: Path, where: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise EvidenceError(f"{where}: symlink in path chain")


def _regular_path(root: Path, literal: str, where: str) -> Path:
    literal = _canonical_relative(literal, where)
    _reject_symlink_chain(root, where)
    if not root.is_dir():
        raise EvidenceError(f"{where}: containment root is not a directory")
    path = root / literal
    _reject_symlink_chain(path, where)
    if not path.is_file():
        raise EvidenceError(f"{where}: artifact is absent or non-regular")
    root_resolved = root.resolve(strict=True)
    path_resolved = path.resolve(strict=True)
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise EvidenceError(f"{where}: path escapes containment root")
    return path


def _verify_artifact(root: Path, literal: str, expected: str, where: str) -> None:
    path = _regular_path(root, literal, where)
    actual = _sha256(path)
    if actual != expected:
        raise EvidenceError(f"{where}: SHA-256 {actual} != {expected}")


def _mix64(value: int) -> int:
    value &= UINT64_MAX
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & UINT64_MAX
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & UINT64_MAX
    return (value ^ (value >> 31)) & UINT64_MAX


def _rotl64(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & UINT64_MAX


def _expected_mappings(
    seed: int, index: int, root_player: int, mode: str
) -> tuple[int, int]:
    schedule = _mix64(seed ^ 0xA4093822299F31D0)
    fixed_offset = schedule % 20
    if mode == "shared":
        mapping = (fixed_offset + index % 20) % 20
        return mapping, mapping
    if mode != "independent":
        raise EvidenceError(f"unsupported v2 role mapping: {mode}")
    other_offset = _rotl64(schedule, 31) % 20
    fixed = (fixed_offset + index % 20) % 20
    other = (other_offset + (index // 20) % 20 + index % 20) % 20
    result = [0, 0]
    result[root_player] = fixed
    result[root_player ^ 1] = other
    return result[0], result[1]


def _sample_metric(pair_values: Sequence[float]) -> dict[str, float | int]:
    n = len(pair_values)
    if n == 0:
        raise EvidenceError("cannot summarize zero pairs")
    total = math.fsum(pair_values)
    mean = total / (2.0 * n)
    if n == 1:
        se = 0.0
    else:
        centered = math.fsum((value - total / n) ** 2 for value in pair_values)
        variance = max(0.0, centered / (n - 1))
        se = math.sqrt(variance / n) / 2.0
    if not math.isfinite(mean) or not math.isfinite(se):
        raise EvidenceError("non-finite recomputed statistic")
    return {"pairs": n, "estimate": mean, "pair_clustered_se": se}


def _c_sample_metric(pair_values: Sequence[float]) -> dict[str, float | int]:
    """Reproduce continuation_arena's ordered double footer arithmetic."""
    n = len(pair_values)
    if n == 0:
        raise EvidenceError("cannot summarize zero pairs")
    total = 0.0
    sumsq = 0.0
    for value in pair_values:
        total += value
        sumsq += value * value
    mean = total / (2.0 * n)
    if n == 1:
        se = 0.0
    else:
        variance = max(0.0, (sumsq - total * total / n) / (n - 1))
        se = math.sqrt(variance / n) / 2.0
    return {"pairs": n, "estimate": mean, "pair_clustered_se": se}


def _result_score(margin: int) -> float:
    return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)


@dataclass(frozen=True)
class RawMetrics:
    early_margin: dict[str, float | int]
    round_margin: tuple[dict[str, float | int], ...]
    round2_final_match_score: dict[str, float | int]
    round2_hybrid: dict[str, float | int]
    caps: int
    exact_moves: int
    cap_forces: int
    cycle_forces: int

    def json(self) -> dict[str, Any]:
        return {
            "early_margin": self.early_margin,
            "round_margin": list(self.round_margin),
            "round2_final_match_score": self.round2_final_match_score,
            "round2_hybrid": self.round2_hybrid,
            "caps": self.caps,
            "exact_moves": self.exact_moves,
            "cap_forces": self.cap_forces,
            "cycle_forces": self.cycle_forces,
        }


def _provenance(entry: dict[str, Any], manifest: dict[str, Any]) -> str:
    fields = [
        ("stage", manifest["stage"]),
        ("plan", manifest["plan_id"]),
        ("arena", manifest["arena_id"]),
        ("root", manifest["root_model_id"]),
        ("candidate", entry["candidate_model_id"]),
        ("baseline", manifest["baseline_model_id"]),
        ("cell", "all" if entry.get("checkpoint") == "base" else entry["cell"]),
    ]
    if manifest["stage"] == "screen-a":
        fields.extend([
            ("replicate", "all" if entry["checkpoint"] == "base" else entry["replicate"]),
            ("checkpoint", entry["checkpoint"]),
        ])
    else:
        fields.append(("variant", entry["variant"]))
        if entry["variant"] == "soup":
            fields.append(("components", ",".join(entry["components"])))
    return "continuation-v2|" + "|".join(f"{key}={value}" for key, value in fields)


def _validate_pair(
    row: Any,
    where: str,
    index: int,
    seed: int,
    objective: int,
    role_mapping: str,
) -> dict[str, Any]:
    row = _exact_keys(row, PAIR_KEYS, where)
    if row["record"] != "pair":
        raise EvidenceError(f"{where}: record must be 'pair'")
    if _uint_string(row["index"], f"{where}.index") != index:
        raise EvidenceError(f"{where}: index gap, overlap, or reordering")
    round_index = _integer(row["round"], 0, 2, f"{where}.round")
    if round_index != index % 3:
        raise EvidenceError(f"{where}: cycle round does not match absolute index")
    root_player = _integer(row["root_player"], 0, 1, f"{where}.root_player")
    if root_player != (round_index & 1):
        raise EvidenceError(f"{where}: root player does not match round/ply schedule")
    admitted = _integer(row["admitted"], 1, 5, f"{where}.admitted")
    picked = _integer(row["picked"], 0, admitted - 1, f"{where}.picked")
    expected_picked = (
        1 + (index // 2) % (admitted - 1)
        if index & 1 and admitted > 1 else 0
    )
    if picked != expected_picked:
        raise EvidenceError(f"{where}: root-candidate pick schedule mismatch")
    _integer(row["root_move"], 0, 719, f"{where}.root_move")
    cumulative = _array2(
        row["cumulative_before"], f"{where}.cumulative_before", -32768, 32767
    )
    alias = _array2(row["cum_before"], f"{where}.cum_before", -32768, 32767)
    if alias != cumulative:
        raise EvidenceError(f"{where}: cumulative aliases disagree")
    mappings = _array2(row["player_mapping"], f"{where}.player_mapping", 0, 19)
    if mappings != _expected_mappings(seed, index, root_player, role_mapping):
        raise EvidenceError(f"{where}: role mapping schedule mismatch")
    if _integer(row["root_role_mapping"], 0, 19, f"{where}.root_role_mapping") != mappings[root_player]:
        raise EvidenceError(f"{where}: root role mapping mismatch")
    if _integer(row["opponent_role_mapping"], 0, 19, f"{where}.opponent_role_mapping") != mappings[root_player ^ 1]:
        raise EvidenceError(f"{where}: opponent role mapping mismatch")
    seats = _array2(row["candidate_seat"], f"{where}.candidate_seat", 0, 1)
    if seats != (root_player, root_player ^ 1):
        raise EvidenceError(f"{where}: controller-seat schedule mismatch")
    scores_raw = row["score_by_seat"]
    if type(scores_raw) is not list or len(scores_raw) != 2:
        raise EvidenceError(f"{where}.score_by_seat: expected 2x2 array")
    scores = (
        _array2(scores_raw[0], f"{where}.score_by_seat[0]", -32768, 32767),
        _array2(scores_raw[1], f"{where}.score_by_seat[1]", -32768, 32767),
    )
    margins = _array2(row["candidate_margin"], f"{where}.candidate_margin", -65535, 65535)
    round_margins = _array2(row["candidate_round_margin"], f"{where}.candidate_round_margin", -65535, 65535)
    if margins != round_margins:
        raise EvidenceError(f"{where}: legacy and round margins disagree")
    for leg in range(2):
        seat = seats[leg]
        derived = scores[leg][seat] - scores[leg][seat ^ 1]
        if margins[leg] != derived:
            raise EvidenceError(f"{where}: candidate margin does not match scores")
    objectives = _float_array2(
        row["candidate_objective_target"],
        f"{where}.candidate_objective_target", -1e6, 1e6,
    )
    if round_index < 2:
        for name in (
            "candidate_final_match_margin", "candidate_final_match_result",
            "candidate_hybrid_target",
        ):
            if row[name] is not None:
                raise EvidenceError(f"{where}.{name}: must be null before round 2")
        expected_objectives = tuple(float(value) for value in margins)
        hybrid: tuple[float, float] | None = None
        final_margins: tuple[int, int] | None = None
        final_results: tuple[int, int] | None = None
    else:
        final_margins = _array2(
            row["candidate_final_match_margin"],
            f"{where}.candidate_final_match_margin", -131070, 131070,
        )
        final_results = _array2(
            row["candidate_final_match_result"],
            f"{where}.candidate_final_match_result", -1, 1,
        )
        hybrid = _float_array2(
            row["candidate_hybrid_target"],
            f"{where}.candidate_hybrid_target", -1e6, 1e6,
        )
        for leg in range(2):
            seat = seats[leg]
            expected_margin = (
                cumulative[seat] - cumulative[seat ^ 1] + margins[leg]
            )
            if final_margins[leg] != expected_margin:
                raise EvidenceError(f"{where}: final match margin mismatch")
            expected_result = (expected_margin > 0) - (expected_margin < 0)
            if final_results[leg] != expected_result:
                raise EvidenceError(f"{where}: final match result mismatch")
            expected_hybrid = 0.05 * expected_margin + 50.0 * expected_result
            if abs(hybrid[leg] - expected_hybrid) > 1e-12:
                raise EvidenceError(f"{where}: hybrid target mismatch")
        expected_objectives = (
            hybrid if objective == 2 else tuple(float(value) for value in margins)
        )
    for leg in range(2):
        if abs(objectives[leg] - expected_objectives[leg]) > 1e-12:
            raise EvidenceError(f"{where}: configured objective target mismatch")
    tail_plies = _array2(row["tail_plies"], f"{where}.tail_plies", 1, 300)
    capped = _array2(row["capped"], f"{where}.capped", 0, 1)
    exact_moves = _array2(row["exact_moves"], f"{where}.exact_moves", 0, 300)
    cap_forces = _array2(row["cap_forces"], f"{where}.cap_forces", 0, 300)
    cycle_forces = _array2(row["cycle_forces"], f"{where}.cycle_forces", 0, 300)
    for leg in range(2):
        if exact_moves[leg] > tail_plies[leg] or cap_forces[leg] > tail_plies[leg] or cycle_forces[leg] > tail_plies[leg]:
            raise EvidenceError(f"{where}: move diagnostics exceed tail length")
    if capped != (0, 0) or exact_moves != (1, 1):
        raise EvidenceError(
            f"{where}: expected uncapped tails and one exact deck-one move per leg"
        )
    return {
        "round": round_index,
        "margins": margins,
        "objectives": objectives,
        "final_margins": final_margins,
        "final_results": final_results,
        "hybrid": hybrid,
        "capped": capped,
        "exact_moves": exact_moves,
        "cap_forces": cap_forces,
        "cycle_forces": cycle_forces,
    }


def _validate_summary(
    summary: Any,
    rows: Sequence[dict[str, Any]],
    objective: int,
    where: str,
) -> None:
    summary = _exact_keys(summary, SUMMARY_KEYS, where)
    if summary["record"] != "summary":
        raise EvidenceError(f"{where}: record must be 'summary'")
    if summary["continuation_objective"] != objective:
        raise EvidenceError(f"{where}: objective mismatch")
    comparable = objective == 0
    if type(summary["configured_objective_aggregate_comparable"]) is not bool or summary["configured_objective_aggregate_comparable"] != comparable:
        raise EvidenceError(f"{where}: aggregate comparability mismatch")
    all_objectives = [math.fsum(row["objectives"]) for row in rows]
    if comparable:
        metric = _c_sample_metric(all_objectives)
        _same_float(summary["configured_objective_per_leg"], float(metric["estimate"]), f"{where}.configured_objective_per_leg")
        _same_float(summary["configured_objective_pair_clustered_se"], float(metric["pair_clustered_se"]), f"{where}.configured_objective_pair_clustered_se")
    elif summary["configured_objective_per_leg"] is not None or summary["configured_objective_pair_clustered_se"] is not None:
        raise EvidenceError(f"{where}: incomparable aggregate must be null")
    rounds = summary["rounds"]
    if type(rounds) is not list or len(rounds) != 3:
        raise EvidenceError(f"{where}.rounds: expected exactly three rounds")
    for round_index in range(3):
        item_where = f"{where}.rounds[{round_index}]"
        item = _exact_keys(
            rounds[round_index],
            FINAL_ROUND_KEYS if round_index == 2 else ROUND_KEYS,
            item_where,
        )
        if item["round"] != round_index:
            raise EvidenceError(f"{item_where}: round order mismatch")
        selected = [row for row in rows if row["round"] == round_index]
        if item["pairs"] != len(selected):
            raise EvidenceError(f"{item_where}: pair count mismatch")
        semantics = "final_match_hybrid" if round_index == 2 and objective == 2 else "round_margin"
        if item["selection_semantics"] != semantics:
            raise EvidenceError(f"{item_where}: selection semantics mismatch")
        margin_metric = _c_sample_metric([float(sum(row["margins"])) for row in selected])
        objective_metric = _c_sample_metric([sum(row["objectives"]) for row in selected])
        _same_float(item["round_margin_per_leg"], float(margin_metric["estimate"]), f"{item_where}.round_margin_per_leg")
        _same_float(item["round_margin_pair_clustered_se"], float(margin_metric["pair_clustered_se"]), f"{item_where}.round_margin_pair_clustered_se")
        _same_float(item["configured_objective_per_leg"], float(objective_metric["estimate"]), f"{item_where}.configured_objective_per_leg")
        _same_float(item["configured_objective_pair_clustered_se"], float(objective_metric["pair_clustered_se"]), f"{item_where}.configured_objective_pair_clustered_se")
        if round_index == 2:
            final_metric = _c_sample_metric([float(sum(row["final_margins"])) for row in selected])
            score_pairs = [sum(_result_score(value) for value in row["final_margins"]) for row in selected]
            score_metric = _c_sample_metric(score_pairs)
            _same_float(item["final_match_margin_per_leg"], float(final_metric["estimate"]), f"{item_where}.final_match_margin_per_leg")
            _same_float(item["final_match_margin_pair_clustered_se"], float(final_metric["pair_clustered_se"]), f"{item_where}.final_match_margin_pair_clustered_se")
            _same_float(item["match_score"], float(score_metric["estimate"]), f"{item_where}.match_score")
            _same_float(item["match_score_pair_clustered_se"], float(score_metric["pair_clustered_se"]), f"{item_where}.match_score_pair_clustered_se")
            wins = sum(value > 0 for row in selected for value in row["final_results"])
            losses = sum(value < 0 for row in selected for value in row["final_results"])
            draws = 2 * len(selected) - wins - losses
            if (item["match_wins"], item["match_losses"], item["match_draws"]) != (wins, losses, draws):
                raise EvidenceError(f"{item_where}: W/L/D mismatch")


def validate_raw(
    raw_path: Path,
    entry: dict[str, Any],
    manifest: dict[str, Any],
) -> RawMetrics:
    try:
        lines = raw_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvidenceError(f"{raw_path}: cannot read raw evidence: {exc}") from exc
    expected_lines = manifest["pair_count"] + 3
    if len(lines) != expected_lines or any(not line for line in lines):
        raise EvidenceError(
            f"{raw_path}: expected {expected_lines} nonempty JSONL records"
        )
    records = [_loads_strict(line, f"{raw_path}:{i + 1}") for i, line in enumerate(lines)]
    meta = _exact_keys(records[0], META_KEYS, f"{raw_path}:1")
    expected_meta = {
        "record": "meta", "schema": 2,
        "evidence_scope": "candidate_screen_only_not_promotion",
        "seed": manifest["seed"], "pair_start": manifest["pair_start"],
        "pair_count": manifest["pair_count"],
        "target_round": "cycle_0_1_2",
        "continuation_objective": entry["objective"],
        "round_0_1_semantics": "round_margin",
        "round_2_mode_2_semantics": "0.05*final_match_margin+50*signed_match_result",
        "role_mapping_mode": entry["role_mapping"],
        "root_checkpoint": manifest["root_checkpoint"],
        "candidate_checkpoint": entry["candidate_checkpoint"],
        "baseline_checkpoint": manifest["baseline_checkpoint"],
        "root_ply": 14, "root_symmetries": 20, "root_width": 5,
        "root_floor": 0.02, "root_min": 1,
        "root_mix": "alternating_absolute_index_baseline_nonbaseline_with_singleton_fallback",
        "world_model": "uniform_mover_information_set",
        "continuation_policy": "greedy_fixed_player_mapping_affine20",
        "late_cycle": "production_semantic_information_tracker_deck_le_3",
        "pairing": "identical_post_root_world_controller_seat_swap",
        "provenance": _provenance(entry, manifest),
    }
    if meta != expected_meta:
        differing = sorted(key for key in META_KEYS if meta.get(key) != expected_meta.get(key))
        raise EvidenceError(f"{raw_path}: meta/provenance mismatch in {differing}")
    seed = _uint_string(meta["seed"], f"{raw_path}:1.seed")
    pair_start = _uint_string(meta["pair_start"], f"{raw_path}:1.pair_start")
    validated = [
        _validate_pair(
            records[offset + 1], f"{raw_path}:{offset + 2}",
            pair_start + offset, seed, entry["objective"], entry["role_mapping"],
        )
        for offset in range(manifest["pair_count"])
    ]
    _validate_summary(records[-2], validated, entry["objective"], f"{raw_path}:{len(lines) - 1}")
    complete = _exact_keys(records[-1], {"record", "pairs"}, f"{raw_path}:{len(lines)}")
    if complete != {"record": "complete", "pairs": manifest["pair_count"]}:
        raise EvidenceError(f"{raw_path}: invalid completion footer")
    per_round = tuple(
        _sample_metric([
            math.fsum(row["margins"])
            for row in validated if row["round"] == round_index
        ])
        for round_index in range(3)
    )
    early = _sample_metric([
        math.fsum(row["margins"]) for row in validated if row["round"] < 2
    ])
    final_rows = [row for row in validated if row["round"] == 2]
    final_score = _sample_metric([
        math.fsum(_result_score(value) for value in row["final_margins"])
        for row in final_rows
    ])
    hybrid = _sample_metric([
        math.fsum(row["hybrid"]) for row in final_rows
    ])
    return RawMetrics(
        early_margin=early,
        round_margin=per_round,
        round2_final_match_score=final_score,
        round2_hybrid=hybrid,
        caps=sum(value for row in validated for value in row["capped"]),
        exact_moves=sum(value for row in validated for value in row["exact_moves"]),
        cap_forces=sum(value for row in validated for value in row["cap_forces"]),
        cycle_forces=sum(value for row in validated for value in row["cycle_forces"]),
    )


def _validate_cells(value: Any) -> tuple[dict[str, Any], ...]:
    if type(value) is not list or len(value) != 4:
        raise EvidenceError("manifest.cells: expected locked 2x2 cell list")
    cells = []
    combinations = set()
    names = set()
    for i, raw in enumerate(value):
        cell = _exact_keys(raw, {"cell", "cell_order", "objective", "role_mapping"}, f"manifest.cells[{i}]")
        name = _label(cell["cell"], f"manifest.cells[{i}].cell")
        order = _integer(cell["cell_order"], 0, 3, f"manifest.cells[{i}].cell_order")
        objective = _integer(cell["objective"], 0, 2, f"manifest.cells[{i}].objective")
        if objective not in (0, 2):
            raise EvidenceError("cell objective must be 0 or 2")
        role = _string(cell["role_mapping"], f"manifest.cells[{i}].role_mapping")
        if role not in ("shared", "independent"):
            raise EvidenceError("cell role mapping must be shared or independent")
        if name in names or (objective, role) in combinations:
            raise EvidenceError("duplicate cell name or 2x2 combination")
        names.add(name); combinations.add((objective, role))
        cells.append({"cell": name, "cell_order": order, "objective": objective, "role_mapping": role})
    expected = (
        {"cell": "o0-shared", "cell_order": 0,
         "objective": 0, "role_mapping": "shared"},
        {"cell": "o0-independent", "cell_order": 1,
         "objective": 0, "role_mapping": "independent"},
        {"cell": "o2-shared", "cell_order": 2,
         "objective": 2, "role_mapping": "shared"},
        {"cell": "o2-independent", "cell_order": 3,
         "objective": 2, "role_mapping": "independent"},
    )
    if tuple(cells) != expected:
        raise EvidenceError("manifest.cells: must equal canonical ordered 2x2 design")
    return tuple(cells)


def load_manifest(path: Path, artifact_root: Path) -> dict[str, Any]:
    _reject_symlink_chain(path, "manifest")
    if not path.is_file():
        raise EvidenceError("manifest: absent or non-regular")
    manifest = _load_json(path)
    if type(manifest) is not dict:
        raise EvidenceError("manifest: expected object")
    stage = manifest.get("stage")
    keys = COMMON_MANIFEST_KEYS | ({"checkpoints"} if stage == "screen-a" else {"confidence_z", "screen_a_result", "screen_a_result_id"} if stage == "screen-b" else set())
    manifest = _exact_keys(manifest, keys, "manifest")
    if manifest["schema"] != 1 or stage not in ("screen-a", "screen-b"):
        raise EvidenceError("manifest: unsupported schema or stage")
    for key in ("plan_id", "arena_id", "root_model_id", "baseline_model_id"):
        manifest[key] = _hex_id(manifest[key], f"manifest.{key}")
    for key in ("plan_path", "arena_path", "root_checkpoint", "baseline_checkpoint"):
        manifest[key] = _canonical_relative(manifest[key], f"manifest.{key}")
    manifest["seed_int"] = _uint_string(manifest["seed"], "manifest.seed")
    manifest["pair_start_int"] = _uint_string(manifest["pair_start"], "manifest.pair_start")
    manifest["pair_count"] = _integer(manifest["pair_count"], 6, 1_000_000, "manifest.pair_count")
    if manifest["pair_count"] % 3:
        raise EvidenceError("manifest.pair_count must be divisible by three")
    if manifest["target_round"] != "cycle_0_1_2":
        raise EvidenceError("manifest.target_round must be cycle_0_1_2")
    if manifest["pair_start_int"] > UINT64_MAX - manifest["pair_count"] + 1:
        raise EvidenceError("manifest pair range overflows uint64")
    manifest["cells"] = _validate_cells(manifest["cells"])
    if type(manifest["replicates"]) is not list or len(manifest["replicates"]) != 3:
        raise EvidenceError("manifest.replicates: expected exactly three")
    manifest["replicates"] = tuple(_label(value, f"manifest.replicates[{i}]") for i, value in enumerate(manifest["replicates"]))
    if len(set(manifest["replicates"])) != 3:
        raise EvidenceError("manifest.replicates: duplicates")
    _verify_artifact(artifact_root, manifest["plan_path"], manifest["plan_id"], "manifest.plan_path")
    _verify_artifact(artifact_root, manifest["arena_path"], manifest["arena_id"], "manifest.arena_path")
    _verify_artifact(artifact_root, manifest["root_checkpoint"], manifest["root_model_id"], "manifest.root_checkpoint")
    _verify_artifact(artifact_root, manifest["baseline_checkpoint"], manifest["baseline_model_id"], "manifest.baseline_checkpoint")
    cell_by_name = {cell["cell"]: cell for cell in manifest["cells"]}
    evidence = manifest["evidence"]
    if type(evidence) is not list:
        raise EvidenceError("manifest.evidence: expected list")
    normalized = []
    seen_keys = set()
    seen_raw: dict[str, tuple[str, str]] = {}
    if stage == "screen-a":
        checkpoints = manifest["checkpoints"]
        if checkpoints != list(CHECKPOINTS):
            raise EvidenceError("manifest.checkpoints: must equal locked checkpoint order")
        expected_keys = {(cell, replicate, checkpoint) for cell in cell_by_name for replicate in manifest["replicates"] for checkpoint in CHECKPOINTS}
        entry_keys = {
            "raw", "raw_id", "cell", "replicate", "checkpoint",
            "candidate_checkpoint", "candidate_artifact",
            "candidate_model_id",
        }
    else:
        if _number(manifest["confidence_z"], Z95, Z95, "manifest.confidence_z") != Z95:
            raise EvidenceError("manifest.confidence_z must be 1.645")
        manifest["screen_a_result"] = _canonical_relative(
            manifest["screen_a_result"], "manifest.screen_a_result"
        )
        manifest["screen_a_result_id"] = _hex_id(manifest["screen_a_result_id"], "manifest.screen_a_result_id")
        result_path = _regular_path(
            path.parent, manifest["screen_a_result"],
            "manifest.screen_a_result",
        )
        if _sha256(result_path) != manifest["screen_a_result_id"]:
            raise EvidenceError("manifest.screen_a_result: absent, symlinked, or hash mismatch")
        manifest["screen_a_data"] = _load_json(result_path)
        expected_keys = {(cell, variant) for cell in cell_by_name for variant in (*manifest["replicates"], "soup")}
        entry_keys = None
    for i, raw in enumerate(evidence):
        where = f"manifest.evidence[{i}]"
        if stage == "screen-a":
            entry = _exact_keys(raw, entry_keys, where)
            identity = (
                _label(entry["cell"], f"{where}.cell"),
                _label(entry["replicate"], f"{where}.replicate"),
                _label(entry["checkpoint"], f"{where}.checkpoint"),
            )
        else:
            if type(raw) is not dict:
                raise EvidenceError(f"{where}: expected object")
            variant = raw.get("variant")
            keys_for_entry = {
                "raw", "raw_id", "cell", "variant",
                "candidate_checkpoint", "candidate_artifact",
                "candidate_model_id",
            } | ({"components"} if variant == "soup" else set())
            entry = _exact_keys(raw, keys_for_entry, where)
            identity = (
                _label(entry["cell"], f"{where}.cell"),
                _label(entry["variant"], f"{where}.variant"),
            )
        if identity not in expected_keys or identity in seen_keys:
            raise EvidenceError(f"{where}: unexpected or duplicate evidence identity {identity}")
        seen_keys.add(identity)
        raw_literal = _canonical_relative(entry["raw"], f"{where}.raw")
        raw_id = _hex_id(entry["raw_id"], f"{where}.raw_id")
        if raw_literal in seen_raw:
            previous_checkpoint, previous_model = seen_raw[raw_literal]
            if not (
                stage == "screen-a"
                and identity[2] == "base"
                and previous_checkpoint == "base"
                and previous_model == entry["candidate_model_id"]
            ):
                raise EvidenceError(f"{where}: raw path reused")
        else:
            seen_raw[raw_literal] = (
                identity[2] if stage == "screen-a" else identity[1],
                entry["candidate_model_id"],
            )
        entry = dict(entry)
        entry["raw_id"] = raw_id
        entry["candidate_checkpoint"] = _canonical_relative(
            entry["candidate_checkpoint"], f"{where}.candidate_checkpoint"
        )
        entry["candidate_artifact"] = _canonical_relative(
            entry["candidate_artifact"], f"{where}.candidate_artifact"
        )
        entry["candidate_model_id"] = _hex_id(entry["candidate_model_id"], f"{where}.candidate_model_id")
        if stage == "screen-a" and identity[2] == "base" and (
            entry["candidate_checkpoint"] != manifest["root_checkpoint"]
            or entry["candidate_artifact"] != manifest["root_checkpoint"]
            or entry["candidate_model_id"] != manifest["root_model_id"]
        ):
            raise EvidenceError(
                f"{where}: base must be the exact frozen root champion"
            )
        entry["raw_path"] = _regular_path(path.parent, raw_literal, f"{where}.raw")
        if _sha256(entry["raw_path"]) != raw_id:
            raise EvidenceError(f"{where}.raw: absent, symlinked, or SHA-256 mismatch")
        # Objective and role mapping are training factors in the 2x2 cell.
        # Every selection screen itself deliberately uses the same deployed
        # evaluator: mode-2 terminal semantics and independent role mappings.
        # Otherwise cells would not be compared on common evidence units.
        entry["objective"] = 2
        entry["role_mapping"] = "independent"
        if stage == "screen-b" and entry["variant"] == "soup":
            components = entry["components"]
            if type(components) is not list or len(components) != 3:
                raise EvidenceError(f"{where}.components: expected three model IDs")
            entry["components"] = tuple(_hex_id(value, f"{where}.components[{j}]") for j, value in enumerate(components))
        _verify_artifact(
            artifact_root, entry["candidate_artifact"],
            entry["candidate_model_id"], f"{where}.candidate_artifact",
        )
        normalized.append(entry)
    if seen_keys != expected_keys:
        raise EvidenceError(f"manifest.evidence: incomplete matrix; missing {sorted(expected_keys - seen_keys)}")
    manifest["evidence"] = tuple(normalized)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for entry in manifest["evidence"]:
        metrics = validate_raw(entry["raw_path"], entry, manifest)
        identity = {"cell": entry["cell"]}
        if manifest["stage"] == "screen-a":
            identity.update({"replicate": entry["replicate"], "checkpoint": entry["checkpoint"]})
        else:
            identity["variant"] = entry["variant"]
        output.append({
            **identity,
            "candidate_checkpoint": entry["candidate_checkpoint"],
            "candidate_artifact": entry["candidate_artifact"],
            "candidate_model_id": entry["candidate_model_id"],
            "raw": entry["raw"],
            "raw_id": entry["raw_id"],
            "metrics": metrics.json(),
        })
    return output


def _extended_z(delta: float, se: float) -> float:
    if se > 0.0:
        return delta / se
    if delta > 0.0:
        return math.inf
    if delta < 0.0:
        return -math.inf
    return 0.0


def _z_json(value: float) -> float | str:
    if math.isinf(value):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return value


def _rank(metrics: dict[str, Any]) -> tuple[float, tuple[float, float, float]]:
    early = metrics["early_margin"]
    score = metrics["round2_final_match_score"]
    hybrid = metrics["round2_hybrid"]
    zs = (
        _extended_z(early["estimate"], early["pair_clustered_se"]),
        _extended_z(score["estimate"] - 0.5, score["pair_clustered_se"]),
        _extended_z(hybrid["estimate"], hybrid["pair_clustered_se"]),
    )
    return min(zs), zs


def screen_a(manifest: dict[str, Any], validated: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(item["cell"], item["replicate"], item["checkpoint"]): item for item in validated}
    selections = []
    for cell in sorted(manifest["cells"], key=lambda item: item["cell_order"]):
        for replicate_order, replicate in enumerate(manifest["replicates"]):
            ranked = []
            for checkpoint_order, checkpoint in enumerate(CHECKPOINTS):
                item = by_key[(cell["cell"], replicate, checkpoint)]
                minimum, zs = _rank(item["metrics"])
                candidate = dict(item)
                # Screen A is ranking-only.  Statistical efficacy must not
                # censor a locked checkpoint or create an extra look.  A cap
                # is an operationally incomplete tail and remains ineligible.
                candidate["eligible"] = item["metrics"]["caps"] == 0
                candidate["minimum_standardized_delta"] = _z_json(minimum)
                candidate["standardized_deltas"] = {
                    "early_margin": _z_json(zs[0]),
                    "round2_final_match_score": _z_json(zs[1]),
                    "round2_hybrid": _z_json(zs[2]),
                }
                ranked.append((minimum, -checkpoint_order, candidate))
            eligible = [value for value in ranked if value[2]["eligible"]]
            if not eligible:
                raise EvidenceError(f"screen-a: no eligible checkpoint for {cell['cell']}/{replicate}")
            def screen_a_rank(value: tuple[float, int, dict[str, Any]]) -> tuple[float, float, float, int]:
                minimum, neg_order, candidate = value
                score = candidate["metrics"]["round2_final_match_score"]
                early = candidate["metrics"]["early_margin"]
                return (
                    minimum,
                    score["estimate"] - Z95 * score["pair_clustered_se"],
                    early["estimate"] - Z95 * early["pair_clustered_se"],
                    neg_order,
                )
            winner = max(eligible, key=screen_a_rank)[2]
            selections.append({
                "cell": cell["cell"], "cell_order": cell["cell_order"],
                "objective": cell["objective"], "role_mapping": cell["role_mapping"],
                "replicate": replicate, "replicate_order": replicate_order,
                "checkpoint": winner["checkpoint"],
                "checkpoint_order": CHECKPOINTS.index(winner["checkpoint"]),
                "candidate_checkpoint": winner["candidate_checkpoint"],
                "candidate_artifact": winner["candidate_artifact"],
                "candidate_model_id": winner["candidate_model_id"],
                "raw": winner["raw"], "metrics": winner["metrics"],
                "raw_id": winner["raw_id"],
                "minimum_standardized_delta": winner["minimum_standardized_delta"],
                "standardized_deltas": winner["standardized_deltas"],
                "tie_break": "earlier_locked_checkpoint_order",
            })
    return {
        "schema": 1, "record": "continuation-v2-screen-a-selection",
        "status": "complete", "plan_id": manifest["plan_id"],
        "arena_id": manifest["arena_id"], "seed": manifest["seed"],
        "pair_start": manifest["pair_start"], "pairs_per_checkpoint": manifest["pair_count"],
        "ranking": "maximize minimum standardized delta across early margin, round-2 final match score, and round-2 hybrid; ties use final-score lower bound, early-margin lower bound, then earlier locked checkpoint",
        "selections": selections,
    }


def _strict_lower_pass(metrics: dict[str, Any]) -> bool:
    early = metrics["early_margin"]
    score = metrics["round2_final_match_score"]
    hybrid = metrics["round2_hybrid"]
    return (
        metrics["caps"] == 0
        and early["estimate"] - Z95 * early["pair_clustered_se"] > 0.0
        and score["estimate"] - Z95 * score["pair_clustered_se"] > 0.5
        and hybrid["estimate"] - Z95 * hybrid["pair_clustered_se"] > 0.0
    )


def _positive_points(metrics: dict[str, Any]) -> bool:
    return (
        metrics["caps"] == 0
        and metrics["early_margin"]["estimate"] > 0.0
        and metrics["round2_final_match_score"]["estimate"] > 0.5
        and metrics["round2_hybrid"]["estimate"] > 0.0
    )


def _nonharm_upper(metrics: dict[str, Any]) -> bool:
    early = metrics["early_margin"]
    score = metrics["round2_final_match_score"]
    hybrid = metrics["round2_hybrid"]
    return (
        metrics["caps"] == 0
        and early["estimate"] + Z95 * early["pair_clustered_se"] >= 0.0
        and score["estimate"] + Z95 * score["pair_clustered_se"] >= 0.5
        and hybrid["estimate"] + Z95 * hybrid["pair_clustered_se"] >= 0.0
    )


def _validate_screen_a_link(manifest: dict[str, Any]) -> dict[tuple[str, str], str]:
    result = manifest["screen_a_data"]
    required = {"schema", "record", "status", "plan_id", "arena_id", "seed", "pair_start", "pairs_per_checkpoint", "ranking", "selections"}
    result = _exact_keys(result, required, "screen-a result")
    if result["schema"] != 1 or result["record"] != "continuation-v2-screen-a-selection" or result["status"] != "complete" or result["plan_id"] != manifest["plan_id"] or result["arena_id"] != manifest["arena_id"]:
        raise EvidenceError("screen-a result identity/provenance mismatch")
    selections = result["selections"]
    if type(selections) is not list or len(selections) != 12:
        raise EvidenceError("screen-a result must contain twelve selections")
    linked = {}
    for i, selection in enumerate(selections):
        if type(selection) is not dict:
            raise EvidenceError(f"screen-a result selection {i}: expected object")
        cell = _label(selection.get("cell"), f"screen-a selection[{i}].cell")
        replicate = _label(selection.get("replicate"), f"screen-a selection[{i}].replicate")
        model_id = _hex_id(selection.get("candidate_model_id"), f"screen-a selection[{i}].candidate_model_id")
        key = (cell, replicate)
        if key in linked:
            raise EvidenceError("screen-a result has duplicate selections")
        linked[key] = model_id
    expected = {(cell["cell"], replicate) for cell in manifest["cells"] for replicate in manifest["replicates"]}
    if set(linked) != expected:
        raise EvidenceError("screen-a result selection matrix mismatch")
    return linked


def screen_b(manifest: dict[str, Any], validated: list[dict[str, Any]]) -> dict[str, Any]:
    linked = _validate_screen_a_link(manifest)
    by_key = {(item["cell"], item["variant"]): item for item in validated}
    cells_out = []
    eligible_ranked = []
    for cell in sorted(manifest["cells"], key=lambda item: item["cell_order"]):
        replicates = []
        selected_ids = []
        for replicate in manifest["replicates"]:
            item = by_key[(cell["cell"], replicate)]
            if item["candidate_model_id"] != linked[(cell["cell"], replicate)]:
                raise EvidenceError(f"screen-b: {cell['cell']}/{replicate} is not its Screen-A winner")
            selected_ids.append(item["candidate_model_id"])
            replicates.append({
                "replicate": replicate,
                "candidate_checkpoint": item["candidate_checkpoint"],
                "candidate_artifact": item["candidate_artifact"],
                "candidate_model_id": item["candidate_model_id"],
                "raw": item["raw"],
                "raw_id": item["raw_id"],
                "metrics": item["metrics"],
                "positive_point_estimates": _positive_points(item["metrics"]),
                "nonharm_upper_pass": _nonharm_upper(item["metrics"]),
            })
        soup = by_key[(cell["cell"], "soup")]
        source_entry = next(entry for entry in manifest["evidence"] if entry["cell"] == cell["cell"] and entry["variant"] == "soup")
        if tuple(selected_ids) != source_entry["components"]:
            raise EvidenceError(f"screen-b: {cell['cell']} soup components do not match Screen-A winners")
        replicate_passes = sum(item["positive_point_estimates"] for item in replicates)
        robustness = replicate_passes >= 2 and all(
            item["positive_point_estimates"] or item["nonharm_upper_pass"]
            for item in replicates
        )
        soup_pass = _strict_lower_pass(soup["metrics"])
        early_round_nonharm = all(
            metric["estimate"] + Z95 * metric["pair_clustered_se"] >= 0.0
            for metric in soup["metrics"]["round_margin"][:2]
        )
        eligible = soup_pass and early_round_nonharm and robustness
        minimum, zs = _rank(soup["metrics"])
        cell_result = {
            "cell": cell["cell"], "cell_order": cell["cell_order"],
            "objective": cell["objective"], "role_mapping": cell["role_mapping"],
            "replicates": replicates,
            "replicate_positive_point_passes": replicate_passes,
            "replicate_robustness_pass": robustness,
            "soup": soup,
            "soup_strict_lower_pass": soup_pass,
            "soup_each_early_round_nonharm_upper_pass": early_round_nonharm,
            "eligible": eligible,
            "minimum_standardized_delta": _z_json(minimum),
            "standardized_deltas": {
                "early_margin": _z_json(zs[0]),
                "round2_final_match_score": _z_json(zs[1]),
                "round2_hybrid": _z_json(zs[2]),
            },
        }
        cells_out.append(cell_result)
        if eligible:
            score = soup["metrics"]["round2_final_match_score"]
            early = soup["metrics"]["early_margin"]
            score_lcb = score["estimate"] - Z95 * score["pair_clustered_se"]
            early_lcb = early["estimate"] - Z95 * early["pair_clustered_se"]
            eligible_ranked.append(
                (minimum, score_lcb, early_lcb, -cell["cell_order"], cell_result)
            )
    winner = max(
        eligible_ranked, key=lambda value: (value[0], value[1], value[2], value[3])
    )[4] if eligible_ranked else None
    selected = None if winner is None else {
        "cell": winner["cell"], "cell_order": winner["cell_order"],
        "objective": winner["objective"], "role_mapping": winner["role_mapping"],
        "candidate_checkpoint": winner["soup"]["candidate_checkpoint"],
        "candidate_artifact": winner["soup"]["candidate_artifact"],
        "candidate_model_id": winner["soup"]["candidate_model_id"],
        "minimum_standardized_delta": winner["minimum_standardized_delta"],
        "tie_break": "earlier_locked_cell_order",
    }
    return {
        "schema": 1, "record": "continuation-v2-screen-b-selection",
        "status": "selected" if selected else "no-eligible-cell",
        "plan_id": manifest["plan_id"], "arena_id": manifest["arena_id"],
        "screen_a_result_id": manifest["screen_a_result_id"],
        "seed": manifest["seed"], "pair_start": manifest["pair_start"],
        "pairs_per_candidate": manifest["pair_count"],
        "confidence_z": Z95,
        "gate": "soup passes strict lower bounds for early margin, round-2 score, and hybrid; each soup early round passes a margin non-harm upper bound; at least two replicate winners have positive point estimates on all three metrics and every remaining replicate passes all non-harm upper bounds",
        "ranking": "maximize soup minimum standardized delta; ties use final-score lower bound, early-margin lower bound, then earlier locked cell order",
        "cells": cells_out, "selected": selected,
    }


def _write_output(path: Path | None, value: Any) -> None:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(text)
        return
    if path.exists() or path.is_symlink():
        raise EvidenceError(f"{path}: refusing to replace output")
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot write output: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "screen-a", "screen-b"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest, args.artifact_root)
        expected_stage = None if args.command == "validate" else args.command
        if expected_stage is not None and manifest["stage"] != expected_stage:
            raise EvidenceError(
                f"{args.command} requires a {expected_stage} manifest"
            )
        validated = validate_manifest(manifest)
        if args.command == "validate":
            result = {
                "schema": 1,
                "record": "continuation-v2-evidence-validation",
                "status": "valid",
                "stage": manifest["stage"],
                "plan_id": manifest["plan_id"],
                "arena_id": manifest["arena_id"],
                "evidence": validated,
            }
        elif args.command == "screen-a":
            result = screen_a(manifest, validated)
        else:
            result = screen_b(manifest, validated)
        _write_output(args.output, result)
        return 0
    except EvidenceError as exc:
        print(f"continuation-v2 selector: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
