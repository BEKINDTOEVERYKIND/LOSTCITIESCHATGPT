#!/usr/bin/env python3
"""Deterministic inference for the policy-cost calibration campaign.

This module deliberately has no dependency on the game runtime or campaign
orchestration.  It implements two disjoint operations:

* ``SELECT`` compares exactly twelve frozen policy-floor/search-onset
  configurations with source-match-clustered simultaneous inference.
* ``TEST`` evaluates only the configuration already selected by ``SELECT``.

The separation is part of the statistical contract: test evidence can never
alter the selected configuration.  All stochastic inference uses one fixed
bootstrap seed, and all returned documents carry a digest of their canonical
JSON payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


class InferenceError(ValueError):
    """Raised when evidence is incomplete, inconsistent, or non-canonical."""


SELECT_INPUT_SCHEMA = "lc-policy-cost-select-input-v1"
SELECT_RESULT_SCHEMA = "lc-policy-cost-select-result-v1"
TEST_INPUT_SCHEMA = "lc-policy-cost-test-input-v1"
TEST_RESULT_SCHEMA = "lc-policy-cost-test-result-v1"
DISCOVERY_MANIFEST_SCHEMA = "lc-policy-cost-discovery-manifest-v1"

PLY_BOUNDARIES = tuple(range(0, 44, 2)) + (44, 48, 64)
BASE_VECTOR_QUOTA = 64
MAX_ALLOCATION_SLOTS = 5


def _scheduled_base(allocation_id: int) -> tuple[int, int, bool]:
    position = allocation_id % (3 * 24 * 2)
    round_index = position % 3
    position //= 3
    frontier = position % 2
    position //= 2
    band = position % 3
    low_ply = position // 3
    return round_index, low_ply + 8 * band, bool(frontier)

POLICY_FLOORS = (0.01, 0.02)
PLY_LOS = (14, 12, 10, 8, 4, 0)
CONFIG_IDS = tuple(
    f"floor-{floor:.2f}_ply-{ply_lo:02d}"
    for ply_lo in PLY_LOS
    for floor in reversed(POLICY_FLOORS)
)
CONFIG_BY_ID = {
    f"floor-{floor:.2f}_ply-{ply_lo:02d}": {
        "id": f"floor-{floor:.2f}_ply-{ply_lo:02d}",
        "policy_floor": floor,
        "ply_lo": ply_lo,
    }
    for floor in POLICY_FLOORS
    for ply_lo in PLY_LOS
}

SELECT_ALPHA = 0.05
SELECT_BOOTSTRAP_SEED = 202611150101
SELECT_BOOTSTRAP_REPS = 20_000
SELECT_BOOTSTRAP_BATCH = 256
TEST_Z = 1.645
MIN_SOURCE_MATCH_CLUSTERS = 8
MIN_POST_STRATUM_CLUSTERS = 8
DIGEST_FIELD = "canonical_payload_sha256"

CAMPAIGN_EVIDENCE_FIELDS = frozenset((
    "raw_verified", "stage", "execution_sha256", "evaluation_sha256",
    "evaluation_header_sha256", "allocation_sha256", "calibration_sha256",
    "policy_cost_sha256", "policy_cost_content_fingerprint", "selection_sha256",
    "actor_manifest_sha256",
))


def _campaign_evidence(value: Mapping[str, Any], stage: str) -> dict[str, Any]:
    binding = _require_object(value, "campaign evidence binding")
    if set(binding) != CAMPAIGN_EVIDENCE_FIELDS or binding.get("raw_verified") is not True or \
            binding.get("stage") != stage:
        raise InferenceError("campaign evidence binding schema/stage drift")
    for field in ("execution_sha256", "evaluation_sha256", "evaluation_header_sha256",
                  "allocation_sha256", "calibration_sha256", "policy_cost_sha256"):
        digest = binding.get(field)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise InferenceError(f"campaign evidence {field} digest drift")
    fingerprint = binding.get("policy_cost_content_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{16}", fingerprint) is None:
        raise InferenceError("campaign evidence policy-cost fingerprint drift")
    if stage == "SELECT":
        if binding.get("selection_sha256") is not None or binding.get("actor_manifest_sha256") is not None:
            raise InferenceError("SELECT evidence must not claim TEST artifacts")
    else:
        for field in ("selection_sha256", "actor_manifest_sha256"):
            digest = binding.get(field)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise InferenceError(f"TEST evidence {field} digest drift")
    return dict(binding)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted JSON representation, including final newline."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InferenceError(f"value is not canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    if DIGEST_FIELD in payload:
        raise InferenceError(f"unsealed payload already contains {DIGEST_FIELD}")
    result = dict(payload)
    result[DIGEST_FIELD] = canonical_sha256(payload)
    return result


def verify_result_digest(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping):
        return False
    claimed = result.get(DIGEST_FIELD)
    if not isinstance(claimed, str) or len(claimed) != 64:
        return False
    payload = dict(result)
    del payload[DIGEST_FIELD]
    try:
        return claimed == canonical_sha256(payload)
    except InferenceError:
        return False


def write_canonical_json(path: str | os.PathLike[str], result: Mapping[str, Any]) -> None:
    """Atomically publish a sealed result without replacing evidence."""

    if not verify_result_digest(result):
        raise InferenceError("refusing to write an unsealed or stale result")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise InferenceError(
                f"refusing to replace existing evidence: {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    else:
        # The destination is a second hard link to the fully fsynced inode.
        # Keep only the canonical evidence name after successful publication.
        os.unlink(temporary)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InferenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: str | os.PathLike[str]) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_strict_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    InferenceError(f"non-finite JSON number: {value}")
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InferenceError(f"cannot read strict JSON {path}: {exc}") from exc


def _require_object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InferenceError(f"{where} must be an object")
    return value


def _require_keys(
    value: Mapping[str, Any], required: Iterable[str], where: str
) -> None:
    required_set = set(required)
    actual = set(value)
    if actual != required_set:
        missing = sorted(required_set - actual)
        extra = sorted(actual - required_set)
        raise InferenceError(
            f"{where} keys differ (missing={missing}, extra={extra})"
        )


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise InferenceError(f"{where} must be a finite number")
    return result


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise InferenceError(f"{where} must be a nonempty trimmed string")
    return value


def _nonnegative_integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InferenceError(f"{where} must be a nonnegative integer")
    return value


def _sha256_string(value: Any, where: str) -> str:
    result = _nonempty_string(value, where)
    if len(result) != 64 or any(
            character not in "0123456789abcdef" for character in result):
        raise InferenceError(f"{where} must be a lowercase SHA-256 digest")
    return result


def _canonical_post_stratum(
    round_index: Any,
    ply_stratum: Any,
    frontier_present: Any,
    allocation_slot: Any,
    where: str,
) -> tuple[int, int, bool, int, str]:
    if (isinstance(round_index, bool) or not isinstance(round_index, int)
            or round_index not in (0, 1, 2)):
        raise InferenceError(f"{where}.round must be 0, 1, or 2")
    if (isinstance(ply_stratum, bool) or not isinstance(ply_stratum, int)
            or not 0 <= ply_stratum < len(PLY_BOUNDARIES) - 1):
        raise InferenceError(
            f"{where}.ply_stratum is outside the frozen 24 strata"
        )
    if not isinstance(frontier_present, bool):
        raise InferenceError(f"{where}.frontier_present must be boolean")
    if (isinstance(allocation_slot, bool)
            or not isinstance(allocation_slot, int)
            or not 0 <= allocation_slot < MAX_ALLOCATION_SLOTS):
        raise InferenceError(
            f"{where}.allocation_slot must be in [0,4]"
        )
    name = (
        f"r{round_index}:p{ply_stratum:02d}:"
        f"f{int(frontier_present)}:j{allocation_slot}"
    )
    return (
        round_index, ply_stratum, frontier_present, allocation_slot, name
    )


def _config(config_id: Any, where: str) -> dict[str, Any]:
    if not isinstance(config_id, str) or config_id not in CONFIG_BY_ID:
        raise InferenceError(f"{where} is not one of the twelve frozen configs")
    return CONFIG_BY_ID[config_id]


def _discovery_manifest(
    raw: Mapping[str, Any], *, stage: str
) -> tuple[
    dict[str, dict[str, Any]],
    str,
    dict[tuple[str, str], dict[str, Any]],
]:
    manifest = _require_object(raw, f"{stage} discovery manifest")
    _require_keys(
        manifest,
        {
            "schema",
            "stage",
            "ply_boundaries",
            "base_vector_quota",
            "source_reservoir_sha256",
            "source_net_sha256",
            "source_exclusion_sha256",
            "eligible_state_commitment_sha256",
            "allocation_rule_sha256",
            "total_eligible_states",
            "master_width_histogram",
            "cells",
            "selected_units",
            DIGEST_FIELD,
        },
        f"{stage} discovery manifest",
    )
    if manifest["schema"] != DISCOVERY_MANIFEST_SCHEMA:
        raise InferenceError(f"{stage} discovery manifest schema mismatch")
    if manifest["stage"] != stage:
        raise InferenceError(f"{stage} discovery manifest stage mismatch")
    if manifest["ply_boundaries"] != list(PLY_BOUNDARIES):
        raise InferenceError(f"{stage} discovery ply boundaries differ")
    if manifest["base_vector_quota"] != BASE_VECTOR_QUOTA:
        raise InferenceError(f"{stage} discovery base quota differs")
    for field in (
        "source_reservoir_sha256", "source_net_sha256",
        "source_exclusion_sha256", "eligible_state_commitment_sha256",
        "allocation_rule_sha256",
    ):
        _sha256_string(manifest[field], f"{stage} discovery {field}")
    if not verify_result_digest(manifest):
        raise InferenceError(f"{stage} discovery manifest digest is invalid")
    total = _nonnegative_integer(
        manifest["total_eligible_states"],
        f"{stage} discovery total_eligible_states",
    )
    if total == 0:
        raise InferenceError(f"{stage} discovery census is empty")
    raw_width_histogram = _require_object(
        manifest["master_width_histogram"],
        f"{stage} discovery master_width_histogram",
    )
    if set(raw_width_histogram) != {"1", "2", "3", "4", "5"}:
        raise InferenceError(
            f"{stage} discovery master-width histogram keys differ"
        )
    width_histogram = {
        width: _nonnegative_integer(
            raw_width_histogram[str(width)],
            f"{stage} discovery width {width}",
        )
        for width in range(1, 6)
    }
    if sum(width_histogram.values()) != total:
        raise InferenceError(
            f"{stage} discovery master-width histogram total mismatch"
        )
    raw_cells = manifest["cells"]
    if not isinstance(raw_cells, list):
        raise InferenceError(f"{stage} discovery cells must be an array")
    cells: dict[str, dict[str, float | int]] = {}
    by_base: dict[tuple[int, int, bool], dict[int, tuple[int, int]]] = {}
    census_sum = 0
    for index, raw_cell in enumerate(raw_cells):
        where = f"{stage} discovery cell {index}"
        cell = _require_object(raw_cell, where)
        _require_keys(
            cell,
            {
                "round",
                "ply_stratum",
                "frontier_present",
                "allocation_slot",
                "post_stratum",
                "census_count",
                "allocation_quota",
                "master_width_histogram",
            },
            where,
        )
        round_index, ply_stratum, frontier, slot, name = (
            _canonical_post_stratum(
                cell["round"], cell["ply_stratum"],
                cell["frontier_present"], cell["allocation_slot"], where,
            )
        )
        if cell["post_stratum"] != name:
            raise InferenceError(f"{where}.post_stratum is not canonical")
        if name in cells:
            raise InferenceError(f"duplicate {stage} discovery cell {name}")
        census = _nonnegative_integer(
            cell["census_count"], f"{where}.census_count"
        )
        quota = _nonnegative_integer(
            cell["allocation_quota"], f"{where}.allocation_quota"
        )
        raw_cell_histogram = _require_object(
            cell["master_width_histogram"],
            f"{where}.master_width_histogram",
        )
        if set(raw_cell_histogram) != {"1", "2", "3", "4", "5"}:
            raise InferenceError(f"{where} master-width histogram keys differ")
        cell_histogram = {
            width: _nonnegative_integer(
                raw_cell_histogram[str(width)], f"{where}.width-{width}"
            )
            for width in range(1, 6)
        }
        if sum(cell_histogram.values()) != census:
            raise InferenceError(f"{where} master-width histogram total mismatch")
        if any(cell_histogram[width] != 0 for width in range(1, slot + 1)):
            raise InferenceError(
                f"{where} contains a vector narrower than allocation slot"
            )
        census_sum += census
        cells[name] = {"census_count": census, "allocation_quota": quota}
        by_base.setdefault(
            (round_index, ply_stratum, frontier), {}
        )[slot] = (census, quota)
        cells[name]["master_width_histogram"] = cell_histogram
    expected_cell_count = (
        3 * (len(PLY_BOUNDARIES) - 1) * 2 * MAX_ALLOCATION_SLOTS
    )
    if len(cells) != expected_cell_count:
        raise InferenceError(
            f"{stage} discovery must enumerate exactly "
            f"{expected_cell_count} cells"
        )
    if census_sum != total:
        raise InferenceError(f"{stage} discovery census total mismatch")
    accumulated_widths = {width: 0 for width in range(1, 6)}
    for cell in cells.values():
        histogram = cell["master_width_histogram"]
        if not isinstance(histogram, Mapping):
            raise InferenceError("internal discovery histogram is invalid")
        for width in range(1, 6):
            accumulated_widths[width] += int(histogram[width])
    if accumulated_widths != width_histogram:
        raise InferenceError(
            f"{stage} per-cell master-width histograms do not match aggregate"
        )
    for round_index in range(3):
        for ply_stratum in range(len(PLY_BOUNDARIES) - 1):
            for frontier in (False, True):
                key = (round_index, ply_stratum, frontier)
                slots = by_base.get(key)
                if slots is None or set(slots) != set(
                        range(MAX_ALLOCATION_SLOTS)):
                    raise InferenceError(
                        f"{stage} discovery base cell {key!r} is incomplete"
                    )
                active = [slot for slot in sorted(slots) if slots[slot][0] > 0]
                if not active:
                    raise InferenceError(
                        f"{stage} discovery base cell {key!r} has zero census"
                    )
                quotient, remainder = divmod(BASE_VECTOR_QUOTA, len(active))
                for position, slot in enumerate(active):
                    census, quota = slots[slot]
                    expected_quota = quotient + int(position < remainder)
                    if quota != expected_quota:
                        raise InferenceError(
                            f"{stage} discovery cell {key + (slot,)!r} "
                            f"quota {quota} != {expected_quota}"
                        )
                    if census < quota:
                        raise InferenceError(
                            f"{stage} discovery cell {key + (slot,)!r} "
                            "is too sparse for its frozen quota"
                        )
                for slot in set(range(MAX_ALLOCATION_SLOTS)) - set(active):
                    if slots[slot] != (0, 0):
                        raise InferenceError(
                            f"{stage} structural-zero cell "
                            f"{key + (slot,)!r} has a nonzero quota"
                        )
    for name, cell in cells.items():
        census = int(cell["census_count"])
        quota = int(cell["allocation_quota"])
        cell["post_stratum_mass"] = census / total
        cell["row_weight"] = (census / (quota * total)) if quota else 0.0
    raw_selected = manifest["selected_units"]
    if not isinstance(raw_selected, list):
        raise InferenceError(f"{stage} selected_units must be an array")
    selected_units: dict[tuple[str, str], dict[str, Any]] = {}
    selected_states: set[str] = set()
    selected_priorities: set[str] = set()
    selected_by_cell: dict[str, int] = {}
    previous_by_cell: dict[str, tuple[str, str, str, str]] = {}
    standalone_canonical_order: list[tuple[str, str, str, str, str]] = []
    frozen_schedule = (
        PLY_BOUNDARIES == tuple(range(0, 44, 2)) + (44, 48, 64)
        and BASE_VECTOR_QUOTA == 64
    )
    for index, raw_unit in enumerate(raw_selected):
        where = f"{stage} selected unit {index}"
        unit = _require_object(raw_unit, where)
        _require_keys(
            unit,
            {
                "source_match", "unit", "state_sha256",
                "allocation_priority_sha256", "round", "ply_stratum",
                "frontier_present", "allocation_slot", "master_width",
                "post_stratum",
            },
            where,
        )
        source = _nonempty_string(
            unit["source_match"], f"{where}.source_match"
        )
        unit_id = _nonempty_string(unit["unit"], f"{where}.unit")
        state_sha = _sha256_string(
            unit["state_sha256"], f"{where}.state_sha256"
        )
        priority_sha = _sha256_string(
            unit["allocation_priority_sha256"],
            f"{where}.allocation_priority_sha256",
        )
        round_index, ply_stratum, frontier, slot, name = (
            _canonical_post_stratum(
                unit["round"], unit["ply_stratum"],
                unit["frontier_present"], unit["allocation_slot"], where,
            )
        )
        if frozen_schedule and (
            round_index, ply_stratum, frontier
        ) != _scheduled_base(index):
            raise InferenceError(
                f"{where} violates frozen interleaved scheduling order"
            )
        if unit["post_stratum"] != name:
            raise InferenceError(f"{where}.post_stratum is not canonical")
        width = _nonnegative_integer(
            unit["master_width"], f"{where}.master_width"
        )
        if not 1 <= width <= 5 or slot >= width:
            raise InferenceError(
                f"{where}.master_width is inconsistent with allocation slot"
            )
        key = (source, unit_id)
        if key in selected_units:
            raise InferenceError(f"duplicate {stage} selected unit {key!r}")
        if state_sha in selected_states:
            raise InferenceError(f"duplicate {stage} selected state digest")
        if priority_sha in selected_priorities:
            raise InferenceError(f"duplicate {stage} allocation priority")
        selected_states.add(state_sha)
        selected_priorities.add(priority_sha)
        normalized_unit = {
            "source_match": source,
            "unit": unit_id,
            "state_sha256": state_sha,
            "allocation_priority_sha256": priority_sha,
            "round": round_index,
            "ply_stratum": ply_stratum,
            "frontier_present": frontier,
            "allocation_slot": slot,
            "master_width": width,
            "post_stratum": name,
        }
        selected_units[key] = normalized_unit
        selected_by_cell[name] = selected_by_cell.get(name, 0) + 1
        order = (priority_sha, state_sha, source, unit_id)
        previous = previous_by_cell.get(name)
        if previous is not None and order <= previous:
            raise InferenceError(
                f"{stage} selected_units have noncanonical within-cell order"
            )
        previous_by_cell[name] = order
        standalone_canonical_order.append((name, *order))
    if not frozen_schedule and \
            standalone_canonical_order != sorted(standalone_canonical_order):
        raise InferenceError(
            f"{stage} selected_units are not in canonical priority order"
        )
    for name, cell in cells.items():
        if selected_by_cell.get(name, 0) != int(cell["allocation_quota"]):
            raise InferenceError(
                f"{stage} selected-unit count for {name!r} differs from quota"
            )
    return cells, str(manifest[DIGEST_FIELD]), selected_units


def _validate_campaign_allocation_rows(
    rows: Sequence[Mapping[str, Any]],
    cells: Mapping[str, Mapping[str, float | int]],
    digest: str,
    selected_units: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    stage: str,
) -> dict[str, float]:
    units: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        where = f"{stage} allocation row {index}"
        row = _require_object(raw, where)
        source = _nonempty_string(row.get("source_match"), f"{where}.source")
        unit = _nonempty_string(row.get("unit"), f"{where}.unit")
        row_digest = _sha256_string(
            row.get("discovery_census_sha256"),
            f"{where}.discovery_census_sha256",
        )
        if row_digest != digest:
            raise InferenceError(f"{where} binds a different discovery census")
        _, _, _, _, name = _canonical_post_stratum(
            row.get("round"), row.get("ply_stratum"),
            row.get("frontier_present"), row.get("allocation_slot"), where,
        )
        if row.get("post_stratum") != name:
            raise InferenceError(f"{where}.post_stratum is not canonical")
        if name not in cells or int(cells[name]["allocation_quota"]) == 0:
            raise InferenceError(f"{where} belongs to an unallocated cell")
        expected_weight = float(cells[name]["row_weight"])
        if _finite_number(row.get("weight"), f"{where}.weight") != expected_weight:
            raise InferenceError(f"{where}.weight differs from discovery census")
        width = _nonnegative_integer(
            row.get("master_width"), f"{where}.master_width"
        )
        if not 1 <= width <= 5 or int(row["allocation_slot"]) >= width:
            raise InferenceError(
                f"{where}.master_width is inconsistent with allocation slot"
            )
        key = (source, unit)
        allocated = selected_units.get(key)
        if allocated is None:
            raise InferenceError(f"{where} is not in the sealed allocation")
        for field in (
            "state_sha256", "allocation_priority_sha256", "round",
            "ply_stratum", "frontier_present", "allocation_slot",
            "master_width", "post_stratum",
        ):
            if row.get(field) != allocated.get(field):
                raise InferenceError(
                    f"{where}.{field} differs from sealed allocation"
                )
        prior = units.setdefault(key, row)
        for field in (
            "round", "ply_stratum", "frontier_present", "allocation_slot",
            "master_width", "post_stratum", "discovery_census_sha256",
            "weight",
        ):
            if prior.get(field) != row.get(field):
                raise InferenceError(
                    f"{stage} unit {key!r} has inconsistent {field}"
                )
    counts: dict[str, int] = {}
    sources: dict[str, set[str]] = {}
    for (source, _), row in units.items():
        name = str(row["post_stratum"])
        counts[name] = counts.get(name, 0) + 1
        sources.setdefault(name, set()).add(source)
    if set(units) != set(selected_units):
        missing = sorted(set(selected_units) - set(units))[:5]
        extra = sorted(set(units) - set(selected_units))[:5]
        raise InferenceError(
            f"{stage} row identities differ from sealed allocation "
            f"(missing={missing}, extra={extra})"
        )
    for name, cell in cells.items():
        quota = int(cell["allocation_quota"])
        if counts.get(name, 0) != quota:
            raise InferenceError(
                f"{stage} cell {name!r} row count {counts.get(name, 0)} "
                f"!= frozen quota {quota}"
            )
        if quota and len(sources.get(name, set())) < MIN_POST_STRATUM_CLUSTERS:
            raise InferenceError(
                f"{stage} cell {name!r} has fewer than "
                f"{MIN_POST_STRATUM_CLUSTERS} source matches"
            )
    return {
        name: float(cell["post_stratum_mass"])
        for name, cell in cells.items()
        if int(cell["allocation_quota"]) > 0
    }


def _conservative_key(config_id: str) -> tuple[int, float, str]:
    config = CONFIG_BY_ID[config_id]
    # min() chooses later onset first, then the higher (2%) floor.
    return (-int(config["ply_lo"]), -float(config["policy_floor"]), config_id)


def _incremental_parents(config_id: str) -> tuple[str, ...]:
    config = CONFIG_BY_ID[config_id]
    floor = float(config["policy_floor"])
    ply_lo = int(config["ply_lo"])
    parents: list[str] = []
    ply_index = PLY_LOS.index(ply_lo)
    if ply_index > 0:
        later = PLY_LOS[ply_index - 1]
        parents.append(f"floor-{floor:.2f}_ply-{later:02d}")
    if floor == 0.01:
        parents.append(f"floor-0.02_ply-{ply_lo:02d}")
    return tuple(parents)


def _selection_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]
]:
    """Validate SELECT rows and return complete unit/config arrays."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise InferenceError("SELECT rows must be an array")
    cells: dict[tuple[str, str], dict[str, tuple[float, float, str]]] = {}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        where = f"SELECT row {index}"
        row = _require_object(raw, where)
        _require_keys(
            row,
            {
                "source_match",
                "unit",
                "state_sha256",
                "allocation_priority_sha256",
                "config",
                "round",
                "ply_stratum",
                "frontier_present",
                "allocation_slot",
                "master_width",
                "post_stratum",
                "discovery_census_sha256",
                "hybrid_gain",
                "weight",
                "exact_valid",
                "capped",
            },
            where,
        )
        source = _nonempty_string(row["source_match"], f"{where}.source_match")
        unit = _nonempty_string(row["unit"], f"{where}.unit")
        state_sha = _sha256_string(
            row["state_sha256"], f"{where}.state_sha256"
        )
        priority_sha = _sha256_string(
            row["allocation_priority_sha256"],
            f"{where}.allocation_priority_sha256",
        )
        round_index, ply_stratum, frontier, allocation_slot, post_stratum = (
            _canonical_post_stratum(
                row["round"], row["ply_stratum"],
                row["frontier_present"], row["allocation_slot"], where,
            )
        )
        if row["post_stratum"] != post_stratum:
            raise InferenceError(f"{where}.post_stratum is not canonical")
        discovery_digest = _sha256_string(
            row["discovery_census_sha256"],
            f"{where}.discovery_census_sha256",
        )
        master_width = _nonnegative_integer(
            row["master_width"], f"{where}.master_width"
        )
        if not 1 <= master_width <= 5 or allocation_slot >= master_width:
            raise InferenceError(
                f"{where}.master_width is inconsistent with allocation slot"
            )
        config_id = _config(row["config"], f"{where}.config")["id"]
        gain = _finite_number(row["hybrid_gain"], f"{where}.hybrid_gain")
        weight = _finite_number(row["weight"], f"{where}.weight")
        if weight <= 0.0:
            raise InferenceError(f"{where}.weight must be positive")
        if row["exact_valid"] is not True:
            raise InferenceError(f"{where} is not exactly valid")
        if _nonnegative_integer(row["capped"], f"{where}.capped") != 0:
            raise InferenceError(f"{where} contains a capped continuation")
        cell = cells.setdefault((source, unit), {})
        if config_id in cell:
            raise InferenceError(f"duplicate SELECT unit/config at {where}")
        cell[config_id] = (gain, weight, post_stratum)
        normalized.append(
            {
                "source_match": source,
                "unit": unit,
                "state_sha256": state_sha,
                "allocation_priority_sha256": priority_sha,
                "config": config_id,
                "round": round_index,
                "ply_stratum": ply_stratum,
                "frontier_present": frontier,
                "allocation_slot": allocation_slot,
                "master_width": master_width,
                "post_stratum": post_stratum,
                "discovery_census_sha256": discovery_digest,
                "hybrid_gain": gain,
                "weight": weight,
                "exact_valid": True,
                "capped": 0,
            }
        )

    if not cells:
        raise InferenceError("SELECT evidence is empty")
    ordered_units = sorted(cells)
    sources = sorted({source for source, _ in ordered_units})
    if len(sources) < MIN_SOURCE_MATCH_CLUSTERS:
        raise InferenceError(
            f"SELECT requires at least {MIN_SOURCE_MATCH_CLUSTERS} source matches"
        )
    source_index = {source: index for index, source in enumerate(sources)}
    gains = np.empty((len(ordered_units), len(CONFIG_IDS)), dtype=np.float64)
    weights = np.empty(len(ordered_units), dtype=np.float64)
    clusters = np.empty(len(ordered_units), dtype=np.int64)
    strata = np.empty(len(ordered_units), dtype=np.int64)
    stratum_names = sorted({
        value[2] for cell in cells.values() for value in cell.values()
    })
    stratum_index = {
        name: index for index, name in enumerate(stratum_names)
    }
    expected = set(CONFIG_IDS)
    unit_post_strata: list[str] = []
    for unit_index, key in enumerate(ordered_units):
        source, _ = key
        cell = cells[key]
        if set(cell) != expected:
            missing = sorted(expected - set(cell))
            extra = sorted(set(cell) - expected)
            raise InferenceError(
                f"SELECT unit {key!r} config coverage differs "
                f"(missing={missing}, extra={extra})"
            )
        unit_weights = {cell[config_id][1] for config_id in CONFIG_IDS}
        if len(unit_weights) != 1:
            raise InferenceError(
                f"SELECT unit {key!r} has config-dependent weights"
            )
        weights[unit_index] = unit_weights.pop()
        clusters[unit_index] = source_index[source]
        unit_strata = {cell[config_id][2] for config_id in CONFIG_IDS}
        if len(unit_strata) != 1:
            raise InferenceError(
                f"SELECT unit {key!r} has config-dependent post-strata"
            )
        unit_post_stratum = unit_strata.pop()
        unit_post_strata.append(unit_post_stratum)
        strata[unit_index] = stratum_index[unit_post_stratum]
        for config_index, config_id in enumerate(CONFIG_IDS):
            gains[unit_index, config_index] = cell[config_id][0]
    normalized.sort(
        key=lambda row: (row["source_match"], row["unit"], row["config"])
    )
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or abs(total_weight - 1.0) > 1.0e-12:
        raise InferenceError("SELECT discovery weights must sum to one")
    for name in stratum_names:
        indices = [
            index for index, found in enumerate(unit_post_strata)
            if found == name
        ]
        if len({float(weights[index]) for index in indices}) != 1:
            raise InferenceError(
                f"SELECT post-stratum {name!r} has unequal unit weights"
            )
        found_sources = {ordered_units[index][0] for index in indices}
        if len(found_sources) < MIN_POST_STRATUM_CLUSTERS:
            raise InferenceError(
                f"SELECT post-stratum {name!r} has only "
                f"{len(found_sources)} source matches"
            )
    return gains, weights, clusters, strata, normalized


def _cluster_influence(
    gains: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    strata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise InferenceError("SELECT total weight is invalid")
    means = np.sum(gains * weights[:, None], axis=0) / total_weight
    nclusters = int(np.max(clusters)) + 1
    influence = np.zeros((nclusters, gains.shape[1]), dtype=np.float64)
    stratum_means = np.empty((int(np.max(strata)) + 1, gains.shape[1]))
    for stratum in range(stratum_means.shape[0]):
        members = strata == stratum
        if not np.any(members):
            raise InferenceError("SELECT post-stratum is empty")
        stratum_weight = float(np.sum(weights[members]))
        if stratum_weight <= 0.0 or not math.isfinite(stratum_weight):
            raise InferenceError("SELECT post-stratum weight is invalid")
        stratum_means[stratum] = np.sum(
            gains[members] * weights[members, None], axis=0
        ) / stratum_weight
    residual = weights[:, None] * (
        gains - stratum_means[strata]
    ) / total_weight
    np.add.at(influence, clusters, residual)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(influence)):
        raise InferenceError("SELECT influence calculation is non-finite")
    return means, influence


def _pair_statistics(
    means: np.ndarray, influence: np.ndarray, fixed_strata: int
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[tuple[int, int]]]:
    pairs: list[tuple[int, int]] = []
    points: list[float] = []
    ses: list[float] = []
    nclusters = influence.shape[0]
    if fixed_strata < 1 or nclusters <= fixed_strata:
        raise InferenceError(
            "SELECT source clusters must exceed fixed post-strata"
        )
    correction = nclusters / (nclusters - fixed_strata)
    for left in range(len(CONFIG_IDS)):
        for right in range(left + 1, len(CONFIG_IDS)):
            cluster_delta = influence[:, left] - influence[:, right]
            variance = correction * float(np.dot(cluster_delta, cluster_delta))
            if variance < 0.0 or not math.isfinite(variance):
                raise InferenceError("SELECT contrast variance is invalid")
            pairs.append((left, right))
            points.append(float(means[left] - means[right]))
            ses.append(math.sqrt(variance))
    return [], np.asarray(points), np.asarray(ses), pairs


def _bootstrap_max_t(
    influence: np.ndarray,
    ses: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    fixed_strata: int,
) -> float:
    """Fixed-seed Rademacher source-cluster multiplier max-t critical value."""

    active = np.flatnonzero(ses > 0.0)
    if active.size == 0:
        return 0.0
    pair_array = np.asarray(pairs, dtype=np.int64)[active]
    active_ses = ses[active]
    nclusters = influence.shape[0]
    if nclusters <= fixed_strata:
        raise InferenceError(
            "SELECT source clusters must exceed fixed post-strata"
        )
    correction = math.sqrt(nclusters / (nclusters - fixed_strata))
    rng = np.random.Generator(np.random.PCG64(SELECT_BOOTSTRAP_SEED))
    maxima = np.empty(SELECT_BOOTSTRAP_REPS, dtype=np.float64)
    written = 0
    while written < SELECT_BOOTSTRAP_REPS:
        count = min(SELECT_BOOTSTRAP_BATCH, SELECT_BOOTSTRAP_REPS - written)
        signs = rng.integers(
            0, 2, size=(count, nclusters), dtype=np.int8
        ).astype(np.float64)
        signs *= 2.0
        signs -= 1.0
        config_draw = correction * (signs @ influence)
        pair_draw = (
            config_draw[:, pair_array[:, 0]]
            - config_draw[:, pair_array[:, 1]]
        ) / active_ses[None, :]
        # Both directions may be queried after seeing the point estimates, so
        # max absolute t protects every directed pair simultaneously.
        maxima[written:written + count] = np.max(np.abs(pair_draw), axis=1)
        written += count
    if not np.all(np.isfinite(maxima)):
        raise InferenceError("SELECT bootstrap produced a non-finite statistic")
    ordered = np.sort(maxima)
    quantile_index = math.ceil((1.0 - SELECT_ALPHA) * len(ordered)) - 1
    quantile_index = min(max(quantile_index, 0), len(ordered) - 1)
    return float(ordered[quantile_index])


def select_configuration(
    rows: Sequence[Mapping[str, Any]],
    discovery_manifest: Mapping[str, Any] | None = None,
    campaign_evidence_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one frozen configuration using SELECT evidence only."""

    campaign_binding: dict[str, Any] = {
        "required": False,
        "validated": False,
        "reason": "standalone inference regression",
    }
    if discovery_manifest is not None:
        cells, digest, selected_units = _discovery_manifest(
            discovery_manifest, stage="SELECT"
        )
        post_masses = _validate_campaign_allocation_rows(
            rows, cells, digest, selected_units, stage="SELECT"
        )
        campaign_binding = {
            "required": True,
            "validated": True,
            "discovery_manifest_sha256": digest,
            "active_post_strata": len(post_masses),
            "base_vector_quota": BASE_VECTOR_QUOTA,
            "ply_boundaries": list(PLY_BOUNDARIES),
        }
    gains, weights, clusters, strata, normalized = _selection_arrays(rows)
    means, influence = _cluster_influence(gains, weights, clusters, strata)
    fixed_strata = int(np.max(strata)) + 1
    _, points, ses, pairs = _pair_statistics(
        means, influence, fixed_strata
    )
    critical = _bootstrap_max_t(
        influence, ses, pairs, fixed_strata
    )

    pair_lookup: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    pair_rows: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(pairs):
        point = float(points[pair_index])
        se = float(ses[pair_index])
        radius = critical * se
        lcb = point - radius
        ucb = point + radius
        pair_lookup[(left, right)] = (point, se, lcb, ucb)
        pair_rows.append(
            {
                "left": CONFIG_IDS[left],
                "right": CONFIG_IDS[right],
                "point_left_minus_right": point,
                "source_match_cluster_se": se,
                "simultaneous_lcb": lcb,
                "simultaneous_ucb": ucb,
            }
        )

    index_by_id = {config_id: index for index, config_id in enumerate(CONFIG_IDS)}

    def directed(config_id: str, reference_id: str) -> tuple[float, float, float]:
        left = index_by_id[config_id]
        right = index_by_id[reference_id]
        if left == right:
            return 0.0, 0.0, 0.0
        if left < right:
            point, se, lcb, _ = pair_lookup[(left, right)]
            return point, se, lcb
        point, se, _, ucb = pair_lookup[(right, left)]
        return -point, se, -ucb

    eligible: dict[str, bool] = {}
    incremental_rows: list[dict[str, Any]] = []
    for ply_lo in PLY_LOS:
        for floor in reversed(POLICY_FLOORS):
            config_id = f"floor-{floor:.2f}_ply-{ply_lo:02d}"
            parents = _incremental_parents(config_id)
            if not parents:
                eligible[config_id] = True
                continue
            passed = True
            for parent in parents:
                point, se, lcb = directed(config_id, parent)
                edge_pass = lcb > 0.0
                incremental_rows.append(
                    {
                        "candidate": config_id,
                        "conservative_parent": parent,
                        "point_increment": point,
                        "source_match_cluster_se": se,
                        "simultaneous_lcb": lcb,
                        "passed_strict_positive_lcb": edge_pass,
                    }
                )
                passed = passed and edge_pass and eligible.get(parent, False)
            eligible[config_id] = passed

    eligible_ids = [config_id for config_id in CONFIG_IDS if eligible[config_id]]
    if not eligible_ids:
        raise InferenceError("SELECT has no eligible configuration")
    point_best = min(
        eligible_ids,
        key=lambda config_id: (
            -float(means[index_by_id[config_id]]),
            *_conservative_key(config_id),
        ),
    )
    tied = []
    for config_id in eligible_ids:
        _, _, best_minus_config_lcb = directed(point_best, config_id)
        if best_minus_config_lcb <= 0.0:
            tied.append(config_id)
    selected = min(tied, key=_conservative_key)

    config_rows = []
    for config_id in CONFIG_IDS:
        index = index_by_id[config_id]
        config_rows.append(
            {
                **CONFIG_BY_ID[config_id],
                "hybrid_gain_point": float(means[index]),
                "incrementally_eligible": eligible[config_id],
            }
        )

    payload = {
        "schema": SELECT_RESULT_SCHEMA,
        "stage": "SELECT",
        "input_sha256": canonical_sha256(normalized),
        "source_match_clusters": int(influence.shape[0]),
        "post_strata": int(np.max(strata)) + 1,
        "units": int(gains.shape[0]),
        "configurations": config_rows,
        "simultaneous_inference": {
            "method": "source-match Rademacher multiplier max-absolute-t",
            "alpha": SELECT_ALPHA,
            "bootstrap_seed": SELECT_BOOTSTRAP_SEED,
            "bootstrap_replicates": SELECT_BOOTSTRAP_REPS,
            "critical_value": critical,
            "cluster_variance_correction": (
                "G/(G-H), G=source clusters, H=fixed post-strata"
            ),
            "protected_directed_pairwise_contrasts": len(pairs) * 2,
            "pair_intervals": pair_rows,
        },
        "incremental_requirements": incremental_rows,
        "eligible_config_ids": eligible_ids,
        "point_best_config_id": point_best,
        "statistically_tied_config_ids": sorted(tied, key=_conservative_key),
        "selected": dict(CONFIG_BY_ID[selected]),
        "selection_rule": {
            "performance_first": True,
            "tie_priority": ["later ply_lo", "policy floor 0.02"],
            "aggressive_change_requires": (
                "every immediate conservative parent is eligible and the "
                "simultaneous incremental LCB is strictly positive"
            ),
            "test_evidence_used": False,
        },
        "campaign_discovery_binding": campaign_binding,
        "runtime_dependencies": {
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
            "execution_image_binding": "required_in_campaign_manifest",
        },
    }
    if campaign_evidence_binding is not None:
        payload["campaign_evidence_binding"] = dict(campaign_evidence_binding)
    return seal_result(payload)


def _test_rows(
    rows: Sequence[Mapping[str, Any]], selected_id: str
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise InferenceError("TEST rows must be an array")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    sources: set[str] = set()
    for index, raw in enumerate(rows):
        where = f"TEST row {index}"
        row = _require_object(raw, where)
        _require_keys(
            row,
            {
                "source_match",
                "unit",
                "state_sha256",
                "allocation_priority_sha256",
                "config",
                "post_stratum",
                "round",
                "ply_stratum",
                "allocation_slot",
                "master_width",
                "discovery_census_sha256",
                "weight",
                "hybrid_gain",
                "match_score_gain",
                "frontier_present",
                "exact_valid",
                "capped",
            },
            where,
        )
        source = _nonempty_string(row["source_match"], f"{where}.source_match")
        unit = _nonempty_string(row["unit"], f"{where}.unit")
        state_sha = _sha256_string(
            row["state_sha256"], f"{where}.state_sha256"
        )
        priority_sha = _sha256_string(
            row["allocation_priority_sha256"],
            f"{where}.allocation_priority_sha256",
        )
        key = (source, unit)
        if key in seen:
            raise InferenceError(f"duplicate TEST source/unit at {where}")
        seen.add(key)
        config_id = _config(row["config"], f"{where}.config")["id"]
        if config_id != selected_id:
            raise InferenceError("TEST evidence contains an unselected actor")
        round_index, ply_stratum, frontier, allocation_slot, stratum = (
            _canonical_post_stratum(
                row["round"], row["ply_stratum"],
                row["frontier_present"], row["allocation_slot"], where,
            )
        )
        if row["post_stratum"] != stratum:
            raise InferenceError(f"{where}.post_stratum is not canonical")
        discovery_digest = _sha256_string(
            row["discovery_census_sha256"],
            f"{where}.discovery_census_sha256",
        )
        master_width = _nonnegative_integer(
            row["master_width"], f"{where}.master_width"
        )
        if not 1 <= master_width <= 5 or allocation_slot >= master_width:
            raise InferenceError(
                f"{where}.master_width is inconsistent with allocation slot"
            )
        weight = _finite_number(row["weight"], f"{where}.weight")
        if weight <= 0.0:
            raise InferenceError(f"{where}.weight must be positive")
        hybrid = _finite_number(row["hybrid_gain"], f"{where}.hybrid_gain")
        score = _finite_number(
            row["match_score_gain"], f"{where}.match_score_gain"
        )
        if not -1.0 <= score <= 1.0:
            raise InferenceError(f"{where}.match_score_gain is outside [-1,1]")
        if not isinstance(row["exact_valid"], bool):
            raise InferenceError(f"{where}.exact_valid must be boolean")
        capped = _nonnegative_integer(row["capped"], f"{where}.capped")
        sources.add(source)
        normalized.append(
            {
                "source_match": source,
                "unit": unit,
                "state_sha256": state_sha,
                "allocation_priority_sha256": priority_sha,
                "config": config_id,
                "post_stratum": stratum,
                "round": int(round_index),
                "ply_stratum": ply_stratum,
                "allocation_slot": allocation_slot,
                "master_width": master_width,
                "discovery_census_sha256": discovery_digest,
                "weight": weight,
                "hybrid_gain": hybrid,
                "match_score_gain": score,
                "frontier_present": frontier,
                "exact_valid": row["exact_valid"],
                "capped": capped,
            }
        )
    if not normalized:
        raise InferenceError("TEST evidence is empty")
    if len(sources) < MIN_SOURCE_MATCH_CLUSTERS:
        raise InferenceError(
            f"TEST requires at least {MIN_SOURCE_MATCH_CLUSTERS} source matches"
        )
    normalized.sort(key=lambda row: (row["source_match"], row["unit"]))
    stratum_sources: dict[str, set[str]] = {}
    for row in normalized:
        stratum_sources.setdefault(str(row["post_stratum"]), set()).add(
            str(row["source_match"])
        )
    sparse = {
        stratum: len(found_sources)
        for stratum, found_sources in stratum_sources.items()
        if len(found_sources) < MIN_POST_STRATUM_CLUSTERS
    }
    if sparse:
        raise InferenceError(
            "TEST post-strata lack independent source-match replication: "
            f"{sparse}"
        )
    return normalized, sorted(sources)


def _weights(
    raw: Mapping[str, Any], observed: set[str]
) -> dict[str, float]:
    weights = _require_object(raw, "discovery post-stratum weights")
    if set(weights) != observed:
        raise InferenceError(
            "discovery post-stratum weight keys do not exactly match TEST strata"
        )
    normalized: dict[str, float] = {}
    for stratum in sorted(observed):
        weight = _finite_number(weights[stratum], f"weight[{stratum!r}]")
        if weight <= 0.0:
            raise InferenceError("discovery post-stratum weights must be positive")
        normalized[stratum] = weight
    total = math.fsum(normalized.values())
    if abs(total - 1.0) > 1e-12:
        raise InferenceError("discovery post-stratum weights must sum to one")
    return normalized


def _cluster_se(
    influence: Mapping[str, float], *, fixed_strata: int
) -> float:
    count = len(influence)
    if count < MIN_SOURCE_MATCH_CLUSTERS:
        raise InferenceError("too few source-match clusters for TEST inference")
    if fixed_strata < 1 or count <= fixed_strata:
        raise InferenceError(
            "TEST source clusters must exceed fixed post-strata"
        )
    variance = count / (count - fixed_strata) * math.fsum(
        value * value for value in influence.values()
    )
    if variance < 0.0 or not math.isfinite(variance):
        raise InferenceError("TEST source-match variance is invalid")
    return math.sqrt(variance)


def _poststratified_metric(
    rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    field: str,
) -> dict[str, Any]:
    by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_stratum.setdefault(str(row["post_stratum"]), []).append(row)
    means = {
        stratum: math.fsum(float(row[field]) for row in members) / len(members)
        for stratum, members in by_stratum.items()
    }
    point = math.fsum(weights[stratum] * means[stratum] for stratum in means)
    influence: dict[str, float] = {
        str(row["source_match"]): 0.0 for row in rows
    }
    for stratum, members in by_stratum.items():
        coefficient = weights[stratum] / len(members)
        center = means[stratum]
        for row in members:
            source = str(row["source_match"])
            influence[source] += coefficient * (float(row[field]) - center)
    se = _cluster_se(influence, fixed_strata=len(by_stratum))
    return {
        "point": point,
        "source_match_cluster_se": se,
        "lcb_z_1_645": point - TEST_Z * se,
        "post_stratum_points": {
            stratum: means[stratum] for stratum in sorted(means)
        },
    }


def _frontier_metric(
    rows: Sequence[Mapping[str, Any]], weights: Mapping[str, float]
) -> dict[str, Any]:
    by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_stratum.setdefault(str(row["post_stratum"]), []).append(row)
    denominator = 0.0
    numerator = 0.0
    frontier_rows = 0
    for stratum, members in by_stratum.items():
        coefficient = weights[stratum] / len(members)
        for row in members:
            if bool(row["frontier_present"]):
                frontier_rows += 1
                denominator += coefficient
                numerator += coefficient * float(row["hybrid_gain"])
    if frontier_rows == 0 or denominator <= 0.0 or not math.isfinite(denominator):
        raise InferenceError("TEST frontier is empty or has zero discovery mass")
    point = numerator / denominator

    # Linearized conditional-mean influence.  Center within each discovery
    # stratum because post-stratification fixes every stratum's contribution.
    influence: dict[str, float] = {
        str(row["source_match"]): 0.0 for row in rows
    }
    for stratum, members in by_stratum.items():
        z_values = [
            (float(row["hybrid_gain"]) - point)
            if bool(row["frontier_present"]) else 0.0
            for row in members
        ]
        z_center = math.fsum(z_values) / len(z_values)
        coefficient = weights[stratum] / len(members) / denominator
        for row, value in zip(members, z_values):
            source = str(row["source_match"])
            influence[source] += coefficient * (value - z_center)
    return {
        "point": point,
        "source_match_cluster_se": _cluster_se(
            influence, fixed_strata=len(by_stratum)
        ),
        "frontier_rows": frontier_rows,
        "estimated_discovery_mass": denominator,
    }


def _round_points(
    rows: Sequence[Mapping[str, Any]], weights: Mapping[str, float]
) -> dict[str, float]:
    stratum_round: dict[str, int] = {}
    stratum_values: dict[str, list[float]] = {}
    for row in rows:
        stratum = str(row["post_stratum"])
        round_index = int(row["round"])
        if stratum in stratum_round and stratum_round[stratum] != round_index:
            raise InferenceError("a TEST post-stratum spans multiple rounds")
        stratum_round[stratum] = round_index
        stratum_values.setdefault(stratum, []).append(float(row["hybrid_gain"]))
    result: dict[str, float] = {}
    for round_index in (0, 1, 2):
        strata = [
            stratum for stratum, found_round in stratum_round.items()
            if found_round == round_index
        ]
        if not strata:
            raise InferenceError(f"TEST has no evidence for round {round_index}")
        denominator = math.fsum(weights[stratum] for stratum in strata)
        if denominator <= 0.0:
            raise InferenceError(f"round {round_index} has zero discovery weight")
        numerator = math.fsum(
            weights[stratum]
            * (math.fsum(stratum_values[stratum]) / len(stratum_values[stratum]))
            for stratum in strata
        )
        result[str(round_index)] = numerator / denominator
    return result


def test_selected_configuration(
    rows: Sequence[Mapping[str, Any]],
    discovery_post_stratum_weights: Mapping[str, Any],
    selection_result: Mapping[str, Any],
    discovery_manifest: Mapping[str, Any] | None = None,
    campaign_evidence_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the one-actor TEST gate without any model-selection operation."""

    selection = _require_object(selection_result, "selection result")
    if selection.get("schema") != SELECT_RESULT_SCHEMA:
        raise InferenceError("TEST selection result has the wrong schema")
    if selection.get("stage") != "SELECT" or not verify_result_digest(selection):
        raise InferenceError("TEST selection result digest is invalid")
    selected = _require_object(selection.get("selected"), "selection.selected")
    selected_id = _config(selected.get("id"), "selection.selected.id")["id"]
    if selected != CONFIG_BY_ID[selected_id]:
        raise InferenceError("TEST selected configuration metadata is inconsistent")

    campaign_binding: dict[str, Any] = {
        "required": False,
        "validated": False,
        "reason": "standalone inference regression",
    }
    if discovery_manifest is not None:
        selection_binding = _require_object(
            selection.get("campaign_discovery_binding"),
            "selection.campaign_discovery_binding",
        )
        if selection_binding.get("validated") is not True:
            raise InferenceError(
                "TEST refuses a SELECT result without campaign discovery binding"
            )
        cells, digest, selected_units = _discovery_manifest(
            discovery_manifest, stage="TEST"
        )
        manifest_weights = _validate_campaign_allocation_rows(
            rows, cells, digest, selected_units, stage="TEST"
        )
        supplied_weights = {
            str(key): _finite_number(value, f"weight[{key!r}]")
            for key, value in discovery_post_stratum_weights.items()
        }
        if supplied_weights != manifest_weights:
            raise InferenceError(
                "TEST supplied post-stratum weights differ from discovery "
                "manifest"
            )
        campaign_binding = {
            "required": True,
            "validated": True,
            "discovery_manifest_sha256": digest,
            "active_post_strata": len(manifest_weights),
            "base_vector_quota": BASE_VECTOR_QUOTA,
            "ply_boundaries": list(PLY_BOUNDARIES),
        }
    normalized, sources = _test_rows(rows, selected_id)
    observed_strata = {str(row["post_stratum"]) for row in normalized}
    weights = _weights(discovery_post_stratum_weights, observed_strata)
    hybrid = _poststratified_metric(normalized, weights, "hybrid_gain")
    match_score = _poststratified_metric(normalized, weights, "match_score_gain")
    frontier = _frontier_metric(normalized, weights)
    round_points = _round_points(normalized, weights)
    exact_valid = all(bool(row["exact_valid"]) for row in normalized)
    capped = sum(int(row["capped"]) for row in normalized)

    criteria = {
        "hybrid_gain_lcb_strictly_positive": hybrid["lcb_z_1_645"] > 0.0,
        "match_score_gain_lcb_strictly_positive": (
            match_score["lcb_z_1_645"] > 0.0
        ),
        "frontier_hybrid_point_nonnegative": frontier["point"] >= 0.0,
        "each_round_hybrid_point_nonnegative": all(
            value >= 0.0 for value in round_points.values()
        ),
        "exact_validity": exact_valid,
        "zero_caps": capped == 0,
    }
    payload = {
        "schema": TEST_RESULT_SCHEMA,
        "stage": "TEST",
        "selection_payload_sha256": selection[DIGEST_FIELD],
        "selected": dict(CONFIG_BY_ID[selected_id]),
        "input_sha256": canonical_sha256(normalized),
        "discovery_post_stratum_weights": weights,
        "discovery_post_stratum_weights_sha256": canonical_sha256(weights),
        "source_match_clusters": len(sources),
        "units": len(normalized),
        "critical_z": TEST_Z,
        "hybrid_gain": hybrid,
        "match_score_gain": match_score,
        "frontier_hybrid_gain": frontier,
        "round_hybrid_points": round_points,
        "raw_validity": {
            "all_rows_exact_valid": exact_valid,
            "capped_continuations": capped,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
        "selection_performed_in_test": False,
        "campaign_discovery_binding": campaign_binding,
        "runtime_dependencies": {
            "numpy_version": np.__version__,
            "execution_image_binding": "required_in_campaign_manifest",
        },
    }
    if campaign_evidence_binding is not None:
        payload["campaign_evidence_binding"] = dict(campaign_evidence_binding)
    return seal_result(payload)


def _select_command(input_path: str, output_path: str) -> None:
    source = _require_object(strict_json(input_path), "SELECT input")
    _require_keys(
        source, {"schema", "discovery_manifest", "rows", "campaign_evidence_binding"}, "SELECT input"
    )
    if source["schema"] != SELECT_INPUT_SCHEMA:
        raise InferenceError("SELECT input schema mismatch")
    result = select_configuration(
        source["rows"],
        _require_object(
            source["discovery_manifest"], "SELECT discovery manifest"
        ),
        _campaign_evidence(source["campaign_evidence_binding"], "SELECT"),
    )
    if not result["campaign_discovery_binding"]["validated"]:
        raise InferenceError("SELECT campaign discovery binding was not validated")
    write_canonical_json(output_path, result)


def _test_command(input_path: str, selection_path: str, output_path: str) -> None:
    source = _require_object(strict_json(input_path), "TEST input")
    _require_keys(
        source,
        {
            "schema", "discovery_manifest", "rows",
            "discovery_post_stratum_weights", "campaign_evidence_binding",
        },
        "TEST input",
    )
    if source["schema"] != TEST_INPUT_SCHEMA:
        raise InferenceError("TEST input schema mismatch")
    evidence = _campaign_evidence(source["campaign_evidence_binding"], "TEST")
    if evidence["selection_sha256"] != hashlib.sha256(Path(selection_path).read_bytes()).hexdigest():
        raise InferenceError("TEST evidence SELECT digest drift")
    result = test_selected_configuration(
        source["rows"], source["discovery_post_stratum_weights"],
        _require_object(strict_json(selection_path), "selection result"),
        _require_object(
            source["discovery_manifest"], "TEST discovery manifest"
        ),
        evidence,
    )
    if not result["campaign_discovery_binding"]["validated"]:
        raise InferenceError("TEST campaign discovery binding was not validated")
    write_canonical_json(output_path, result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--input", required=True)
    select_parser.add_argument("--output", required=True)
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--input", required=True)
    test_parser.add_argument("--selection", required=True)
    test_parser.add_argument("--output", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            _select_command(args.input, args.output)
        elif args.command == "test":
            _test_command(args.input, args.selection, args.output)
        else:
            value = _require_object(strict_json(args.input), "sealed result")
            if not verify_result_digest(value):
                raise InferenceError("result digest verification failed")
    except InferenceError as exc:
        print(f"policy-cost inference: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
