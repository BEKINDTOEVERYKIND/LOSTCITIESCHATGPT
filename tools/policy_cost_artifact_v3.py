#!/usr/bin/env python3
"""Independent reader for canonical rollout5 ``.lcpc`` artifacts.

The playing runtime is implemented in C.  This deliberately small Python
reader gives evidence tooling a second implementation of the byte layout and
content hash, without importing campaign selection logic or rewriting an
artifact.  It is read-only and fails closed on trailing, truncated, malformed,
or noncanonical content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any


MAGIC = b"LCPCOS1\0"
VERSION = 3
HEADER_BYTES = 256
ANCHORS = (0, 4, 8, 12, 16, 24, 32, 40, 48, 64)
PAYLOAD_DOUBLES = 3 * len(ANCHORS)
ARTIFACT_BYTES = HEADER_BYTES + 8 * PAYLOAD_DOUBLES + 8
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
LC_MAX_PLIES = 300
SOURCE_SEED = 202701140101
STRICT_EPSILON = math.ldexp(1.0, -150)
# The locked build is GCC 13.3 on x86-64-v3.  The focused cross-language C
# writer deliberately disables fast-math, so bit zero is the sole accepted
# test-vs-production difference; every controller trajectory feature remains
# exact.  0x30d23a encodes GCC 13.3, FMA/AVX2/AVX/SSE4.2, binary64 eval.
SUPPORTED_BUILD_PROFILES = frozenset((0x0030D23A, 0x0030D23B))


class ArtifactError(ValueError):
    """The policy-cost artifact is not canonical or internally consistent."""


def _u32(raw: bytes, offset: int) -> int:
    return struct.unpack_from("<I", raw, offset)[0]


def _u64(raw: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", raw, offset)[0]


def _f32(raw: bytes, offset: int) -> float:
    return struct.unpack_from("<f", raw, offset)[0]


def _f64(raw: bytes, offset: int) -> float:
    return struct.unpack_from("<d", raw, offset)[0]


def fnv1a64(raw: bytes) -> int:
    value = FNV_OFFSET
    for byte in raw:
        value ^= byte
        value = (value * FNV_PRIME) & ((1 << 64) - 1)
    return value


def read_policy_cost(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read policy-cost artifact: {exc}") from exc
    if len(raw) != ARTIFACT_BYTES:
        raise ArtifactError(
            f"policy-cost artifact has {len(raw)} bytes, expected {ARTIFACT_BYTES}"
        )
    if raw[:8] != MAGIC:
        raise ArtifactError("invalid policy-cost magic")
    if (
        _u32(raw, 8) != VERSION
        or _u32(raw, 12) != HEADER_BYTES
        or _u32(raw, 16) != 3
        or _u32(raw, 20) != len(ANCHORS)
        or _u32(raw, 24) != PAYLOAD_DOUBLES
    ):
        raise ArtifactError("unsupported policy-cost dimensions or version")
    if (
        any(raw[28:32])
        or any(raw[148:160])
        or any(raw[184 + 4 * len(ANCHORS):HEADER_BYTES])
    ):
        raise ArtifactError("nonzero reserved policy-cost header bytes")
    expected_fingerprint = fnv1a64(raw[:-8])
    stored_fingerprint = _u64(raw, len(raw) - 8)
    if stored_fingerprint != expected_fingerprint:
        raise ArtifactError("policy-cost content fingerprint mismatch")

    anchors = tuple(_u32(raw, 184 + 4 * index) for index in range(len(ANCHORS)))
    if anchors != ANCHORS:
        raise ArtifactError("noncanonical policy-cost anchors")
    epsilon = _f64(raw, 160)
    primary_z = _f64(raw, 168)
    fresh_z = _f64(raw, 176)
    if not math.isfinite(epsilon) or epsilon != STRICT_EPSILON:
        raise ArtifactError("invalid strict probability floor")
    if _u64(raw, 32) != SOURCE_SEED:
        raise ArtifactError("invalid policy-cost calibration seed")
    if primary_z != 3.5 or fresh_z != 2.58:
        raise ArtifactError("policy-cost evidence thresholds changed")

    root_fingerprint = _u64(raw, 40)
    continuation_fingerprint = _u64(raw, 48)
    match_value_fingerprint = _u64(raw, 56)
    words = [_u32(raw, 64 + 4 * index) for index in range(18)]
    objective = words[2]
    if (
        not root_fingerprint
        or not continuation_fingerprint
        or root_fingerprint != continuation_fingerprint
    ):
        raise ArtifactError("policy-cost checkpoint binding is absent")
    if objective not in {0, 3} or (
        (objective == 3) != bool(match_value_fingerprint)
    ):
        raise ArtifactError("policy-cost objective/table binding is inconsistent")
    if words[0] != 1:
        raise ArtifactError("unsupported policy-cost controller ABI")
    if words[1] not in SUPPORTED_BUILD_PROFILES:
        raise ArtifactError("unsupported policy-cost build profile")
    if words[3] != 20:
        raise ArtifactError("unsupported root symmetry group")
    if words[4] != 20:
        raise ArtifactError("unsupported continuation symmetry group")
    if words[5] != 4 or words[6] != 1:
        raise ArtifactError("unsupported continuation sampling/pruning mode")
    if words[7] != 1 or words[8] != 1:
        raise ArtifactError(
            "policy-cost v3 requires exact-terminal uniform-belief mode"
        )
    if words[9] != 800 or words[10] != 800:
        raise ArtifactError("invalid primary/fresh world counts")
    if words[11] != 5 or words[12] != 3:
        raise ArtifactError("invalid root/core shortlist width")
    if words[13] != 1:
        raise ArtifactError("invalid minimum candidate count")
    if words[14] != 0 or words[15] != 0:
        raise ArtifactError("policy-cost v3 requires all-ply search")
    if words[16] != 1 or words[17] != 0:
        raise ArtifactError("invalid discard guard or nonzero root pruning")
    cand_floor = _f32(raw, 136)
    override_k = _f32(raw, 140)
    override_min = _f32(raw, 144)
    floor_bits = _u32(raw, 136)
    frozen_floor_bits = {
        struct.unpack("<I", struct.pack("<f", value))[0]
        for value in (0.01, 0.02)
    }
    if not math.isfinite(cand_floor) or floor_bits not in frozen_floor_bits:
        raise ArtifactError("invalid binary32 candidate floor")
    if override_k != 3.5 or _u32(raw, 144) != 0:
        raise ArtifactError("invalid legacy low-prior gate binding")

    payload_offset = HEADER_BYTES
    beta: list[float] = []
    alpha_action: list[float] = []
    alpha_draw: list[float] = []
    for index in range(len(ANCHORS)):
        beta_value = _f64(raw, payload_offset + 24 * index)
        action = _f64(raw, payload_offset + 24 * index + 8)
        draw = _f64(raw, payload_offset + 24 * index + 16)
        if not (
            math.isfinite(beta_value)
            and beta_value > 0
            and math.isfinite(action)
            and math.isfinite(draw)
            and action >= 0
            and draw >= 0
            and math.isfinite(action / beta_value)
            and math.isfinite(draw / beta_value)
        ):
            raise ArtifactError("invalid policy-cost spline coefficient")
        beta.append(beta_value)
        alpha_action.append(action)
        alpha_draw.append(draw)

    return {
        "schema": "lc-policy-cost-v3",
        "artifact_version": VERSION,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_fingerprint": f"{stored_fingerprint:016x}",
        "source_seed": _u64(raw, 32),
        "strict_probability_floor": epsilon,
        "primary_z": primary_z,
        "fresh_z": fresh_z,
        "anchors": list(anchors),
        "beta": beta,
        "alpha_action": alpha_action,
        "alpha_draw": alpha_draw,
        "controller": {
            "root_net_fingerprint": f"{root_fingerprint:016x}",
            "continuation_net_fingerprint": (
                f"{continuation_fingerprint:016x}"
            ),
            "match_value_fingerprint": f"{match_value_fingerprint:016x}",
            "controller_abi": words[0],
            "build_profile": f"{words[1]:08x}",
            "objective": objective,
            "root_symmetries": words[3],
            "playout_symmetries": words[4],
            "playout_sample": words[5],
            "playout_prune": words[6],
            "exact_terminal": words[7],
            "no_belief": words[8],
            "dets": words[9],
            "confirm_dets": words[10],
            "root_width": words[11],
            "action_core_count": words[12],
            "min_cand": words[13],
            "ply_lo": words[14],
            "ply_hi": words[15],
            "discard_guard": words[16],
            "root_prune": words[17],
            "cand_floor": cand_floor,
            "override_k": override_k,
            "override_min": override_min,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(read_policy_cost(arguments.artifact), sort_keys=True))


if __name__ == "__main__":
    main()
