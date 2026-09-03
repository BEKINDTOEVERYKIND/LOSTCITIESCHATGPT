#!/usr/bin/env python3
"""Apply a predeclared gate to one fully merged reciprocal actor panel.

The hardened arena merger is authoritative for raw-row validation and the
reciprocal estimator.  This small layer deliberately runs only after both
orientations have completed: it reopens the two merged blocks and every raw
shard through :mod:`tools.merge_arena`, requires the supplied reciprocal file
to be an exact recomputation, checks the locked identities, and then applies
either the continuation-soup safety gate or its final promotion gate.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any

if __package__:
    from tools.merge_arena import (
        EvidenceError,
        _combine_reciprocal,
        _read_json_snapshot,
        _remerge_recorded_block,
        _write_json,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from merge_arena import (  # type: ignore[no-redef]
        EvidenceError,
        _combine_reciprocal,
        _read_json_snapshot,
        _remerge_recorded_block,
        _write_json,
    )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"invalid numeric field {field}")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError(f"non-finite numeric field {field}")
    return number


def _rebuild_reciprocal(path: Path, gate_z: float) -> tuple[
        dict[str, Any], str]:
    actual, actual_digest = _read_json_snapshot(path)
    if not isinstance(actual, dict):
        raise EvidenceError("reciprocal result must be a JSON object")
    snapshots = actual.get("input_block_snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 2 or any(
            not isinstance(item, dict) or set(item) != {"path", "sha256"}
            for item in snapshots):
        raise EvidenceError("malformed reciprocal input snapshots")

    blocks: list[dict[str, Any]] = []
    verified_snapshots: list[dict[str, str]] = []
    raw_blocks: list[dict[str, Any]] = []
    for ordinal, snapshot in enumerate(snapshots):
        source = Path(snapshot["path"])
        block, digest = _read_json_snapshot(source)
        if digest != snapshot["sha256"]:
            raise EvidenceError(f"reciprocal block {ordinal} digest mismatch")
        blocks.append(block)
        verified_snapshots.append({"path": str(source), "sha256": digest})
        raw_blocks.append(_remerge_recorded_block(
            block, "first" if ordinal == 0 else "second"))

    rebuilt = _combine_reciprocal(
        blocks[0], blocks[1], verified_snapshots,
        gate_z=gate_z, require_positive_margin=True,
        raw_input_validation={
            "status": "validated",
            "method": "reopened, SHA-256 checked, and exactly remerged recorded raw inputs",
            "blocks": raw_blocks,
        })
    if rebuilt != actual:
        raise EvidenceError("reciprocal result is not the exact raw-backed recomputation")
    return actual, actual_digest


def evaluate_gate(result: dict[str, Any], mode: str,
                  gate_z: float = 1.645) -> dict[str, Any]:
    """Return the locked gate decision for an already validated result."""
    if mode not in {"safety", "final"}:
        raise EvidenceError("gate mode must be safety or final")
    if not math.isfinite(gate_z) or gate_z <= 0.0:
        raise EvidenceError("gate z must be finite and positive")
    candidate = result.get("candidate_result")
    if not isinstance(candidate, dict):
        raise EvidenceError("missing reciprocal candidate result")
    raw_validation = result.get("raw_input_validation")
    raw_valid = isinstance(raw_validation, dict) and \
        raw_validation.get("status") == "validated"
    score = _finite_number(candidate.get("match_score"), "match_score")
    score_se = _finite_number(
        candidate.get("match_score_pair_clustered_se"), "match_score_se")
    margin = _finite_number(candidate.get("margin_per_game"), "margin")
    margin_se = _finite_number(
        candidate.get("margin_pair_clustered_se"), "margin_se")
    orientations = candidate.get("orientation_match_scores")
    if not isinstance(orientations, list) or len(orientations) != 2:
        raise EvidenceError("two orientation match scores are required")
    orientation_scores = [
        _finite_number(value, f"orientation_match_scores[{index}]")
        for index, value in enumerate(orientations)
    ]
    caps = candidate.get("capped_rounds")
    if type(caps) is not int or caps < 0:
        raise EvidenceError("invalid capped-round count")
    if score_se < 0.0 or margin_se < 0.0:
        raise EvidenceError("negative standard error")

    score_lower = score - gate_z * score_se
    margin_lower = margin - gate_z * margin_se
    if mode == "safety":
        blocks = result.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != 2 or any(
                not isinstance(block, dict) or
                not isinstance(block.get("sufficient_statistics"), dict)
                for block in blocks):
            raise EvidenceError("two sufficient-statistics blocks are required")
        first = blocks[0]["sufficient_statistics"]
        second = blocks[1]["sufficient_statistics"]
        try:
            n = first["pairs"]
            second_n = second["pairs"]
            first_quarters = first["score_quarters_sum"]
            second_quarters = second["score_quarters_sum"]
            first_margin = first["margin_sum"]
            second_margin = second["margin_sum"]
        except KeyError as exc:
            raise EvidenceError("missing safety sufficient statistic") from exc
        if any(type(value) is not int for value in (
                n, second_n, first_quarters, second_quarters,
                first_margin, second_margin)) or n <= 0 or n != second_n:
            raise EvidenceError("invalid safety sufficient statistics")
        requirements = {
            "raw_inputs_validated": raw_valid,
            "zero_capped_rounds": caps == 0,
            # These four comparisons are integer exact.  For the second
            # orientation the baseline-first result is inverted to candidate.
            "combined_match_score_at_least_half":
                first_quarters >= second_quarters,
            "combined_margin_strictly_positive":
                first_margin > second_margin,
            "candidate_first_match_score_at_least_0_475":
                10 * first_quarters >= 19 * n,
            "baseline_first_inverted_match_score_at_least_0_475":
                10 * second_quarters <= 21 * n,
        }
    else:
        requirements = {
            "raw_inputs_validated": raw_valid,
            "zero_capped_rounds": caps == 0,
            "match_score_one_sided_lower_bound_above_half": score_lower > 0.5,
            "margin_one_sided_lower_bound_strictly_positive": margin_lower > 0.0,
            "each_orientation_match_score_strictly_above_half": all(
                value > 0.5 for value in orientation_scores),
        }
    return {
        "schema_version": 1,
        "artifact_kind": "locked_reciprocal_actor_gate_decision",
        "mode": mode,
        "critical_z": gate_z,
        "candidate_result": {
            "match_score": score,
            "match_score_pair_clustered_se": score_se,
            "match_score_gate_lower_bound": score_lower,
            "margin_per_game": margin,
            "margin_pair_clustered_se": margin_se,
            "margin_gate_lower_bound": margin_lower,
            "orientation_match_scores": orientation_scores,
            "capped_rounds": caps,
        },
        "requirements": requirements,
        "passed": all(requirements.values()),
    }


def _canonical_decimal(text: str, field: str) -> int:
    if not text or not text.isascii() or not text.isdigit() or \
            (len(text) > 1 and text[0] == "0"):
        raise argparse.ArgumentTypeError(f"{field} must be canonical decimal")
    return int(text)


def _append_github_output(path: Path, passed: bool) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        stream.write(f"passed={'true' if passed else 'false'}\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reciprocal", type=Path, required=True)
    parser.add_argument("--mode", choices=("safety", "final"), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--pairs-per-orientation", type=lambda value:
                        _canonical_decimal(value, "pairs per orientation"),
                        required=True)
    parser.add_argument("--candidate-first-seed", required=True)
    parser.add_argument("--baseline-first-seed", required=True)
    parser.add_argument("--gate-z", type=float, default=1.645)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        reciprocal, digest = _rebuild_reciprocal(args.reciprocal,
                                                 args.gate_z)
        if reciprocal.get("candidate") != args.candidate or \
                reciprocal.get("baseline") != args.baseline or \
                reciprocal.get("provenance") != args.provenance:
            raise EvidenceError("locked actor identity or provenance mismatch")
        blocks = reciprocal.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != 2:
            raise EvidenceError("two reciprocal blocks are required")
        expected = (
            (args.candidate, args.baseline, args.candidate_first_seed),
            (args.baseline, args.candidate, args.baseline_first_seed),
        )
        for index, (block, identity) in enumerate(zip(blocks, expected)):
            metadata = block.get("metadata") if isinstance(block, dict) else None
            if not isinstance(metadata, dict) or \
                    block.get("pair_start") != "0" or \
                    block.get("pair_count") != args.pairs_per_orientation or \
                    metadata.get("agent_a") != identity[0] or \
                    metadata.get("agent_b") != identity[1] or \
                    metadata.get("seed") != identity[2] or \
                    metadata.get("rounds") != 3 or \
                    metadata.get("provenance") != args.provenance:
                raise EvidenceError(f"locked reciprocal block {index} mismatch")
        decision = evaluate_gate(reciprocal, args.mode, args.gate_z)
        decision.update({
            "reciprocal_path": str(args.reciprocal),
            "reciprocal_sha256": digest,
            "candidate": args.candidate,
            "baseline": args.baseline,
            "provenance": args.provenance,
            "pairs_per_orientation": args.pairs_per_orientation,
            "seeds": {
                "candidate_first": args.candidate_first_seed,
                "baseline_first": args.baseline_first_seed,
            },
        })
        _write_json(args.output, decision)
        if args.github_output is not None:
            _append_github_output(args.github_output, decision["passed"])
    except (EvidenceError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
