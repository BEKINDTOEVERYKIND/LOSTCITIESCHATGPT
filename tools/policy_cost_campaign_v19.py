#!/usr/bin/env python3
"""Fail-closed orchestration and independent evidence conversion for policy-cost-v19.

The native collector owns policy generation and counterfactual simulation.  It
does not fit a schedule, choose a configuration, or decide a campaign gate.
This module is the deliberately separate campaign layer.  It binds the
terminal objective-3 disposition, freezes the unique execution addendum,
validates complete JSONL evidence, converts it to the calibration/SELECT/TEST
schemas, constructs the one selected actor, and emits a terminal recommendation.

No command can dispatch a workflow, overwrite evidence, read a saved probe
state, replace a sparse allocation, or make a result-dependent budget change.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import struct
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PLAN_PATH = "data/experiments/locked_policy_cost_v19_plan.json"
EXECUTION_PATH = "data/experiments/locked_policy_cost_v19_execution.json"
EXECUTION_TEMPLATE_PATH = (
    "data/experiments/locked_policy_cost_v19_execution.template.json"
)
WORKFLOW_PATH = ".github/workflows/policy-cost-v19.yml"
HELPER_PATH = "tools/policy_cost_campaign_v19.py"
DATASET_SOURCE_PATH = "tools/policy_cost_dataset_v19.c"
DATASET_SOURCE_SHA256 = (
    "728d31b8f922017c1165e4f638fc5be47f2eb861e0def4ab6eaa1431a37b092e"
)
PREREQUISITE_PATH = "data/experiments/match_value_objective3_v2_result.json"
EXCLUSION_BINDINGS = (
    ("data/experiments/policy_cost_v16_exact17_exclusions.txt",
     "10034cf8b83aadf24fa0775e4dad2712573e1d84cbf364568ce6136682ac254c"),
    ("data/experiments/policy_cost_v16_exact17_exclusions.json",
     "14469564b7d6631077f6164e7870a19394e2205c7721abc998533e8ce07ea501"),
)
PREDECESSOR_ATTEMPT_BINDINGS = (
    ("data/experiments/locked_policy_cost_v18_execution.json",
     "a7cc58bbb2763718384c1c7784be0c355017ebaf5271755173afc69cf78e2d90"),
    ("data/experiments/policy_cost_v18_run_33659346537_failure.json",
     "faa39b5f38c5f3e9249b881d29bb2e0c290160e87050c9a29ac49f0ab10227d5"),
)

PLAN_SCHEMA = "lc-policy-cost-v19-plan-v1"
EXECUTION_SCHEMA = "lc-policy-cost-v19-execution-v1"
MANIFEST_SCHEMA = "lc-policy-cost-v19-pre-efficacy-manifest-v1"
RESULT_SCHEMA = "lc-policy-cost-v19-result-v1"
CALIBRATION_GATE_SCHEMA = "lc-policy-cost-v2-calibration-gate-v1"
CALIBRATION_FAILURE_REASON = (
    "authoritative_predictive_model_adequacy_gate_failed"
)

BRANCH = "agent/correctness-and-policy-upgrade"
COMPILER = "gcc"
COMPILER_VERSION_COMMAND = "gcc -dumpfullversion -dumpversion"
COMPILER_VERSION = "13.3.0"
CFLAGS = (
    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
    "-Wall -Wextra -std=c11"
)
LDFLAGS = "-lm -pthread"
PYTHON_REQUIREMENTS_PATH = (
    "data/experiments/policy_cost_v19_python_requirements.txt"
)
PYTHON_PACKAGES = (
    (
        "numpy", "numpy", "2.3.5",
        "numpy-2.3.5-cp312-cp312-manylinux_2_27_x86_64."
        "manylinux_2_28_x86_64.whl",
        "0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28",
    ),
    (
        "PyYAML", "yaml", "6.0.3",
        "pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64."
        "manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        "ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc",
    ),
)
NUMERIC_RUNTIME_ENV = {
    "MKL_NUM_THREADS": "1",
    "NPY_DISABLE_CPU_FEATURES": "AVX512F",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
BUILD_COMPILE_COMMANDS = (
    "cp src/policy_cost_v19.c src/policy_cost.c && "
    "cp src/policy_cost_v19.h src/policy_cost.h",
    f"make -j2 CC=gcc CFLAGS={CFLAGS} LDFLAGS={LDFLAGS} "
    "bin/arena bin/build_policy_cost",
    f"gcc {CFLAGS} -o bin/policy_cost_dataset {DATASET_SOURCE_PATH} "
    "src/lc.c src/features.c src/net.c src/heuristic.c src/planner.c "
    "src/search.c src/rollout.c src/late_resolver.c src/match_value.c "
    f"src/policy_cost.c src/agent.c src/match.c src/spec.c {LDFLAGS}",
)

PLY_STRATA = tuple(
    (lower, lower + 2) for lower in range(0, 44, 2)
) + ((44, 48), (48, 64))
ANCHORS = (0, 4, 8, 12, 16, 24, 32, 40, 48, 64)
RATIO_BANDS = (
    "[1,1.25)", "[1.25,2)", "[2,4)",
    "[4,8)", "[8,32)", "[32,inf)",
)
PAIR_TYPES = ("different_core", "same_core_draw")
FLOORS = (0.01, 0.02)
PLY_LOS = (0,)
CONFIG_IDS = tuple(
    f"floor-{floor:.2f}_ply-{ply:02d}"
    for ply in PLY_LOS for floor in reversed(FLOORS)
)

TRAIN_RECORDS = 3 * len(PLY_STRATA) * len(RATIO_BANDS) * 2 * 16
HOLDOUT_RECORDS = 3 * len(PLY_STRATA) * 2 * 64
EVALUATION_SLICES = {"TRAIN": 216, "SELECT": 192, "TEST": 192}
DISCOVERY_MATCHES = {"TRAIN": 65536, "SELECT": 32768, "TEST": 32768}
DISCOVERY_SEEDS = {
    "TRAIN": "202806100101",
    "SELECT": "202806100201",
    "TEST": "202806100301",
}
PRIMARY_SEEDS = {
    "TRAIN": "202806110101",
    "SELECT": "202806110201",
    "TEST": "202806110301",
}
FRESH_SEEDS = {
    "TRAIN": "202806120101",
    "SELECT": "202806120201",
    "TEST": "202806120301",
}
TRUTH_SEEDS = {
    "TRAIN": "202806130101",
    "SELECT": "202806130201",
    "TEST": "202806130301",
}
MAINTAINED_SEEDS = {
    "TRAIN": "202806160101",
    "SELECT": "202806160201",
    "TEST": "202806160301",
}
TRUTH_WORLDS = {"TRAIN": 512, "SELECT": 1024, "TEST": 1024}

EXACT17 = (
    ("data/experiments/commented_ply_audit_v3.json",
     "be63dcae2ae1a179cf43a0c47e9971755290f9b3bfd90cc40fd4b6bd2838bbd7"),
    ("data/experiments/commented_ply_audit_v3.md",
     "a30eb93e4623e75ec2dae4c2cb73103b801d67954e0b02298e1a5b1082ebcd71"),
    ("data/experiments/commented_ply_audit_v3_result.json",
     "9897b402116b897942031ecbd46c50127b358c5a2c579e9c854720c667f55a82"),
    ("data/experiments/commented_ply_audit_v3_evidence.zip",
     "aacec0f3da9bbedd5d6512cf7bf2ef0d993232ed888d64f2e8099f5b15c03994"),
)

HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CELL_VECTOR = re.compile(r"r([0-2]):p(\d{2}):f([01]):g([0-2])\Z")
VECTOR_ALLOCATION_COLUMNS = (
    "allocation_id", "source_match_index", "source_state_index",
    "source_match_id", "unit", "round", "ply_stratum",
    "frontier_present", "allocation_slot", "post_stratum", "master_width",
    "census_count", "allocation_quota", "weight_numerator",
    "weight_denominator", "orbit_sha256", "state_sha256",
    "allocation_priority_sha256", "mask_001_sha256", "mask_002_sha256",
    "master_sha256", "state_hex", "discovery_sha256",
)
VECTOR_ALLOCATION_RULE_SHA256 = hashlib.sha256(
    b"lc-policy-cost-vector-allocation-v2|64-fixed-g0-g1-tail|priority-v1|"
    b"rank-major-three-band-base-interleave-v1"
).hexdigest()
TRAIN_ALLOCATION_RULE_SHA256 = hashlib.sha256(
    b"lc-policy-cost-train-allocation-v5|canonical-greedy-selection|"
    b"global-source-unique|quota16|rank-major-diagonal-cell-interleave-v1"
).hexdigest()


def _train_scheduled_cell(allocation_id: int) -> tuple[int, int, int, int]:
    position = allocation_id % (3 * 24 * 6 * 2)
    ply_bin = position % 24
    position //= 24
    round_index = position % 3
    position //= 3
    base_ratio = position % 6
    base_type = position // 6
    return (
        round_index,
        ply_bin,
        (base_ratio + ply_bin) % 6,
        (base_type + round_index + ply_bin) % 2,
    )


def _vector_scheduled_base(allocation_id: int) -> tuple[int, int, int]:
    position = allocation_id % (3 * 24 * 2)
    round_index = position % 3
    position //= 3
    frontier = position % 2
    position //= 2
    band = position % 3
    low_ply = position // 3
    return round_index, low_ply + 8 * band, frontier
POLICY_COST_SOURCE_SEED = "202806140101"
POLICY_COST_EPSILON = "0x1p-150"
BURNED_SOURCE_DEAL_SEEDS = (
    "1..200, maintained-800 seed 1, 202611010101, all policy-cost-v1 "
    "fixed seeds in 20261110/11/12/13/14/15/16/21/22, every 20261129 "
    "feasibility-smoke seed, all policy-cost-v2 fixed seeds in "
    "20261210/11/12/13/14/15/16/21/22, every 20261229 feasibility-smoke "
    "seed, 202612010101, all policy-cost-v3 fixed seeds in "
    "20270110/11/12/13/14/15/16/21/22, every 20270129 feasibility-smoke "
    "seed, 202701010101, all policy-cost-v4 fixed seeds in "
    "20270210/11/12/13/14/15/16/21/22, every 20270229 feasibility-smoke "
    "seed, 202702010101, all policy-cost-v5 fixed seeds in "
    "20270310/11/12/13/14/15/16/21/22, every 20270329 feasibility-smoke "
    "seed, 202703010101, all policy-cost-v6 fixed seeds in "
    "20270410/11/12/13/14/15/16/21/22, every 20270429 feasibility-smoke "
    "seed, 202704010101, all policy-cost-v7 fixed seeds in "
    "20270510/11/12/13/14/15/16/21/22, every 20270529 feasibility-smoke "
    "seed, 202705010101, all policy-cost-v8 fixed seeds in "
    "20270710/11/12/13/14/15/16/21/22, every 20270729 feasibility-smoke "
    "seed, 202707010101, all policy-cost-v9 fixed seeds in "
    "20270810/11/12/13/14/15/16/21/22, every 20270829 feasibility-smoke "
    "seed, 202708010101, all policy-cost-v10 fixed seeds in "
    "20270910/11/12/13/14/15/16/21/22, every 20270929 feasibility-smoke "
    "seed, 202709010101, all policy-cost-v11 fixed seeds in "
    "20271010/11/12/13/14/15/16/21/22, every 20271029 feasibility-smoke "
    "seed, 202710010101, all policy-cost-v12 fixed seeds in "
    "20271110/11/12/13/14/15/16/21/22, every 20271129 feasibility-smoke "
    "seed, 202711010101, all policy-cost-v14 fixed seeds in 20280110/11/12/13/14/15/16/21/22, every 20280129 feasibility-smoke seed, 202801010101, 202802010101, all policy-cost-v15 fixed seeds in 20280210/11/12/13/14/15/16/21/22, every 20280229 feasibility-smoke seed, 202803010101, all policy-cost-v16 fixed seeds in 20280310/11/12/13/14/15/16/21/22, every 20280329 feasibility-smoke seed, 202804010101, all policy-cost-v17 fixed seeds in 20280410/11/12/13/14/15/16/21/22, every 20280429 feasibility-smoke seed, 202805010101, all policy-cost-v18 fixed seeds in 20280510/11/12/13/14/15/16/21/22, every 20280529 feasibility-smoke seed, 202806010101, and every 20280629 feasibility-smoke seed"
)
POLICY_COST_CONTROLLER_TUPLE = {
    "root_symmetries": 20,
    "playout_symmetries": 20,
    "playout_sample": 4,
    "playout_prune": 1,
    "exact_terminal": 1,
    "no_belief": 1,
    "dets": 800,
    "confirm_dets": 800,
    "root_width": 5,
    "action_core_count": 3,
    "min_cand": 1,
    "ply_lo": 0,
    "ply_hi": 0,
    "discard_guard": 1,
    "root_prune": 0,
    "override_k": 3.5,
    "override_min": 0.0,
}


class EvidenceError(ValueError):
    """Evidence is incomplete, inconsistent, mutable, or out of protocol."""


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {token!r}")


def strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path}: top-level JSON must be an object")
    return value


def strict_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise EvidenceError(f"{path}: JSONL must use canonical LF and final LF")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(
                line.decode("utf-8"), object_pairs_hook=_unique,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, ValueError) as exc:
            raise EvidenceError(
                f"{path}: invalid JSONL line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceError(f"{path}: JSONL line {line_number} is not an object")
        rows.append(value)
    return rows


def canonical_json(value: Any, *, pretty: bool = True) -> bytes:
    try:
        if pretty:
            encoded = json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=True,
                allow_nan=False,
            )
        else:
            encoded = json.dumps(
                value, separators=(",", ":"), sort_keys=True,
                ensure_ascii=True, allow_nan=False,
            )
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"value is not canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_relative_path(relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise EvidenceError(f"{label} is not a canonical relative path")
    path = Path(relative)
    if path.is_absolute() or path.as_posix() != relative or \
            any(part in ("", ".", "..") for part in path.parts):
        raise EvidenceError(f"{label} is not a canonical relative path")
    return path


def binding(root: Path, relative: str) -> dict[str, Any]:
    relative_path = _canonical_relative_path(relative, "bound path")
    cursor = root
    for part in relative_path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise EvidenceError(f"bound path traverses a symlink: {relative}")
    try:
        root_resolved = root.resolve(strict=True)
        path = root / relative_path
        resolved = path.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"bound path escapes or is absent: {relative}") from exc
    if not resolved.is_file():
        raise EvidenceError(f"bound file is absent or not regular: {relative}")
    return {
        "path": relative,
        "sha256": sha256(resolved),
        "size": resolved.stat().st_size,
    }


def write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".partial",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise EvidenceError(f"refusing to replace immutable evidence {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_plan(value: Mapping[str, Any]) -> None:
    if value.get("schema") != PLAN_SCHEMA or \
            value.get("artifact_kind") != "locked_policy_cost_v19_campaign_plan" or \
            value.get("experiment") != "policy-cost-v19" or \
            value.get("status") != (
                "definition_complete_unlaunched_requires_objective3_"
                "disposition_and_unique_execution_addendum"
            ):
        raise EvidenceError("policy-cost plan identity drift")
    if value.get("artifact_schemas") != {
        "execution": EXECUTION_SCHEMA,
        "plan": PLAN_SCHEMA,
        "pre_efficacy_manifest": MANIFEST_SCHEMA,
        "result": RESULT_SCHEMA,
    }:
        raise EvidenceError("policy-cost schema registry drift")
    if value.get("predecessor_attempt") != {
        "attempt": 1,
        "bindings": [
            {"path": path, "sha256": digest}
            for path, digest in PREDECESSOR_ATTEMPT_BINDINGS
        ],
        "execution_commit": "73ccfb92712e5db86fe0a7e0514bc4ebb84a3f70",
        "fixed_roots_retired": True,
        "retry_forbidden": True,
        "run_id": 33659346537,
        "status": (
            "terminal_preflight_exact17_transport_filename_failure_"
            "before_discovery"
        ),
    }:
        raise EvidenceError("policy-cost predecessor attempt binding drift")
    if value.get("v19_hierarchical_draw_hypothesis") != {
        "dataset_source": {
            "path": DATASET_SOURCE_PATH,
            "sha256": DATASET_SOURCE_SHA256,
        },
        "exact17_canonical_payload_sha256": (
            "2a3591230311fd0e5a3937565d9cf84f58b22a6bb7612c5eecbaa081541480f9"
        ),
        "exact17_json_schema": "lc-policy-cost-exclusions-evidence-v2",
        "portable_binding": (
            "canonical exact-17 evidence binds the v19 dataset source; the "
            "platform-specific runtime ELF is sealed dynamically in build "
            "identity and transport manifests"
        ),
        "development_reason": (
            "retired v16 TRAIN evidence showed the conditional-draw term was "
            "positive while cross-action and midgame policy adjustments were "
            "harmful; v19 tests an action-core/raw-search hierarchy with only "
            "an early/late conditional-draw term on entirely fresh roots"
        ),
        "retention_repair": (
            "install and verify the already frozen NumPy/PyYAML wheels in "
            "infrastructure_retain before recomputing any retained calibration "
            "disposition; no efficacy or promotion gate changes"
        ),
        "v16_terminal_gate": (
            "authoritative model adequacy rejected the unrestricted policy-gap "
            "spline before SELECT because the frozen midgame incremental point "
            "was negative; the maintained actor was unchanged"
        ),
    }:
        raise EvidenceError("policy-cost v19 recovery binding drift")
    calibration = value.get("calibration")
    if not isinstance(calibration, dict) or \
            calibration.get("anchors") != list(ANCHORS) or \
            calibration.get("coefficient_constraints") != (
                "beta_search is a strictly positive continuous piecewise-linear "
                "round-shared spline; alpha_A is fixed exactly zero at every "
                "ply; alpha_D is a nonnegative disconnected early/late spline, "
                "fixed exactly zero for 16 <= ply < 40, with independent late "
                "values beginning at ply 40; runtime lambda_D is derived only "
                "after interpolation as alpha_D/beta"
            ) or \
            calibration.get("fit") != (
                "predict independent truth delta from beta_search(ply)*"
                "search_delta + alpha_D_phase(ply)*log conditional-draw ratio "
                "on same-action draw-source pairs using fixed-precision "
                "standardized pairwise Huber regression; alpha_A is "
                "structurally zero everywhere and alpha_D is structurally zero "
                "for different-action pairs and 16 <= ply < 40"
            ) or \
            calibration.get("huber_delta") != 1.345 or \
            calibration.get("standard_error_floor") != 0.25 or \
            calibration.get("schedule_seed") != POLICY_COST_SOURCE_SEED or \
            calibration.get("smoothness_grid") != [
                0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0
            ] or calibration.get("cross_validation") != {
                "folds": 5,
                "group": "source_match_id",
                "model_adequacy_is_nested": True,
                "seed": POLICY_COST_SOURCE_SEED,
                "stratification": "all frozen TRAIN cells",
                "untouched_outer_folds_for_model_checks": True,
            } or calibration.get("model_lack") != {
                "authoritative_pre_select_gate": True,
                "cell_saturated_support": (
                    "derive estimable columns from each outer-training "
                    "partition without consulting held-out outcomes: search "
                    "and the pair-type-defining ratio are mandatory; for "
                    "different-core pairs only, omit conditional-draw exactly "
                    "when its training-design norm is zero and record that "
                    "structural omission"
                ),
                "deployable_gap_over_beta_only_one_sided_lcb_strictly_above": 0.0,
                "nonnegative_gap_improvement_required_for_each_pair_type_and_early_mid_late_phase": True,
                "reject_if_unrestricted_full_spline_round_specific_or_cell_saturated_oof_loss_improves_by_more_than": 0.05,
            } or calibration.get("negative_evidence_contract") != {
                "authoritative_adequacy_failure": (
                    "a completed statistically valid calibration that fails "
                    "any locked model-adequacy criterion emits canonical "
                    "full-detail nondeployable evidence instead of raising "
                    "an execution error"
                ),
                "canonicalization": (
                    "sorted-key compact UTF-8 JSON, finite numbers only, one "
                    "trailing newline, with calibration_sha256 sealing the "
                    "payload before that digest field is added"
                ),
                "failed_result": {
                    "decision": {
                        "calibration_passed": False,
                        "deployment": {
                            "permitted": False,
                            "reason": (
                                "authoritative_predictive_model_adequacy_"
                                "gate_failed"
                            ),
                        },
                        "status": "failed_model_adequacy",
                    },
                    "forbidden_top_level_fields": [
                        "schedule", "derived_gap_thresholds",
                    ],
                    "full_detail_requirement": (
                        "retain all computed fit, grouped-CV, convergence, "
                        "campaign-design/input-binding, runtime-dependency, "
                        "and model-adequacy diagnostics without truncation"
                    ),
                    "required_top_level_fields": [
                        "schema", "calibration_passed", "status", "deployment",
                        "observation_input_sha256", "model", "fit",
                        "campaign_design", "model_adequacy",
                        "runtime_dependencies", "calibration_sha256",
                    ],
                },
                "hard_failure_behavior": (
                    "raise without serializing failed_model_adequacy evidence "
                    "or a false adequacy gate"
                ),
                "hard_failure_classes": [
                    "malformed, non-finite, incomplete, or binding-invalid "
                    "input/evidence",
                    "rank deficiency in any full, fold, or adequacy-comparator "
                    "design",
                    "conditioning violation in any full, fold, or adequacy-"
                    "comparator design",
                    "solver or IRLS nonconvergence or fixed iteration-cap "
                    "exhaustion",
                ],
                "sealed_gate": {
                    "campaign_input_bindings": (
                        "exact allocation_binding and evidence_binding objects "
                        "from campaign_design"
                    ),
                    "fields": [
                        "schema", "calibration_passed", "status", "reason",
                        "calibration_file_sha256", "calibration_payload_sha256",
                        "observation_input_sha256", "campaign_input_bindings",
                        "model_adequacy_sha256",
                    ],
                    "false_blocks": (
                        "LCPC materialization, SELECT evaluation/search/truth "
                        "labels, TEST evaluation, safety, final, actor "
                        "construction, promotion, and every other efficacy "
                        "consumer; the already frozen policy-only discovery "
                        "reservoirs are not efficacy"
                    ),
                    "pass_requirement": (
                        "every LCPC or SELECT-and-later consumer must verify "
                        "the sealed gate and require calibration_passed=true, "
                        "status=passed, and reason=null"
                    ),
                    "schema": "lc-policy-cost-v2-calibration-gate-v1",
                },
            }:
        raise EvidenceError("calibration lock drift")
    controller = value.get("controller")
    if not isinstance(controller, dict) or \
            controller.get("candidate_zero_bonus") != 0.0 or \
            controller.get("confidence") != {"fresh_z": 2.58, "primary_z": 3.5} or \
            controller.get("runtime_scope") != \
            "real roots only; never recursive continuation nodes" or \
            controller.get("transitivity") is not True or \
            controller.get("versioned_actor_families") != ["rolloutu5"] or \
            controller.get("artifact_binding") != {
                "source_seed": POLICY_COST_SOURCE_SEED,
                "epsilon": POLICY_COST_EPSILON,
            } or \
            controller.get("deployment_controller_tuple") != \
            POLICY_COST_CONTROLLER_TUPLE:
        raise EvidenceError("controller lock drift")
    reservoirs = value.get("reservoirs")
    if not isinstance(reservoirs, dict) or \
            reservoirs.get("burned_source_seeds") != \
            BURNED_SOURCE_DEAL_SEEDS or \
            reservoirs.get("pre_efficacy_barrier") != (
                "all TRAIN, SELECT, and TEST discovery reservoirs plus their "
                "hash-only allocations must be complete, immutable, and "
                "SHA-256-bound before any evaluator opens a search panel or "
                "requests a truth label") or \
            reservoirs.get("native_reservoir_origin_proof") != (
                "before the pre-efficacy barrier opens, the frozen source-free "
                "policy_cost_dataset binary scans every retained row in all "
                "three reservoirs; it requires the exact 174-byte native "
                "information view, source bounds, view validity, recomputed "
                "state and suit-orbit hashes, exact17 exclusion, and recomputed "
                "policy masks/cell identity, then emits a sealed proof that "
                "terminal verification independently reruns byte-for-byte"
            ) or \
            reservoirs.get("ply_strata") != [list(item) for item in PLY_STRATA] or \
            reservoirs.get("ratio_bands") != list(RATIO_BANDS) or \
            reservoirs.get("discovery", {}).get("source_matches") != \
            DISCOVERY_MATCHES or \
            reservoirs.get("discovery", {}).get("seeds") != DISCOVERY_SEEDS or \
            reservoirs.get("discovery", {}).get(
                "atomic_persistence_before_allocation"
            ) != (
                "each split uploads its native discovery census, bounded "
                "reservoir, generation log, canonical disposition, and "
                "complete source-free transport before a separate allocation "
                "job may consume those exact bytes"
            ) or \
            reservoirs.get("union_width_contract") != {
                "consumer_length": 5,
                "master_max": 5,
                "producer_length": 5,
                "required_equality": (
                    "union_width_counts equals the 1-percent master "
                    "mask_width_counts row exactly"
                ),
                "sealed_smokes": (
                    "clean definition and launch preflight compile the actual "
                    "native producer and validate TRAIN, SELECT, and TEST "
                    "census/reservoir pairs through the Python consumer under "
                    "burned 20280629 roots; a sixth bin or unequal union "
                    "histogram is rejected"
                ),
            } or \
            reservoirs.get("train", {}).get("records") != TRAIN_RECORDS or \
            reservoirs.get("train", {}).get("cells") != 864 or \
            reservoirs.get("train", {}).get("states_per_cell") != 16 or \
            reservoirs.get("select_and_test", {}).get("records_per_split") != \
            HOLDOUT_RECORDS or \
            reservoirs.get("select_and_test", {}).get("base_cells") != 144:
        raise EvidenceError("reservoir design drift")
    configs = value.get("selection", {}).get("configurations", {})
    if configs != {
        "floors": [0.01, 0.02],
        "ply_lo": [0],
        "total": 2,
    }:
        raise EvidenceError("fixed configuration lattice drift")
    if value.get("multiplicity") != {
        "arena_safety_looks": 1,
        "final_looks": 1,
        "fitted_schedules": 1,
        "select_configurations": 2,
        "test_candidates": 1,
        "test_looks": 1,
    }:
        raise EvidenceError("campaign multiplicity drift")
    selection_lock = value.get("selection", {})
    if selection_lock.get("metric") != (
            "discovery-post-stratified full-match hybrid gain over the "
            "information-view maintained actor move") or \
            selection_lock.get("rule") != (
                "the maintained Objective-3 actor already searches all plies, "
                "so onset is frozen at zero; 1% must have a strictly positive "
                "simultaneous incremental LCB over the all-ply 2% parent; on "
                "a statistical tie prefer 2%"):
        raise EvidenceError("SELECT estimand/rule drift")
    bootstrap = value.get("selection", {}).get("bootstrap")
    if bootstrap != {
        "replicates": 20000,
        "seed": "202806150101",
        "simultaneous_method": (
            "max-t over all predeclared directed configuration contrasts"
        ),
        "unit": "source-match cluster",
    }:
        raise EvidenceError("SELECT bootstrap drift")
    test = value.get("test", {})
    if test.get("critical_z") != 1.645 or \
            test.get("one_frozen_candidate_only") is not True or \
            test.get("second_test_or_top_up") is not False:
        raise EvidenceError("TEST lock drift")
    safety = value.get("safety", {})
    final = value.get("final", {})
    if safety.get("pairs_per_orientation") != 200 or \
            safety.get("pairs_per_shard") != 20 or \
            safety.get("seeds") != {
                "baseline_first": "202806210102",
                "candidate_first": "202806210101",
            } or safety.get("shard_starts") != list(range(0, 200, 20)):
        raise EvidenceError("safety schedule drift")
    if final.get("pairs_per_orientation") != 2500 or \
            final.get("pairs_per_shard") != 100 or \
            final.get("critical_z") != 1.645 or \
            final.get("seeds") != {
                "baseline_first": "202806220102",
                "candidate_first": "202806220101",
            } or final.get("shard_starts") != list(range(0, 2500, 100)):
        raise EvidenceError("final schedule drift")
    protocol = value.get("execution_protocol", {})
    if protocol.get("manual_dispatch") is not False or \
            protocol.get("no_retry") is not True or \
            protocol.get("optional_stopping") is not False or \
            protocol.get("source_free_after_preflight") is not True or \
            protocol.get("execution_addendum") != EXECUTION_PATH or \
            protocol.get("workflow") != WORKFLOW_PATH or \
            protocol.get("prerequisite_evidence_transport") != (
                "retain and terminal-reopen the canonical Objective-3 result "
                "plus every SHA-256-listed evidence member"
            ) or \
            protocol.get("numeric_runtime_environment") != \
            NUMERIC_RUNTIME_ENV or \
            protocol.get("python_dependencies") != {
                "install": (
                    "preflight and reducers only, from both sealed local "
                    "wheels with --no-index --no-deps"
                ),
                "packages": {
                    distribution: {
                        "import_name": import_name,
                        "sha256": digest,
                        "version": version,
                        "wheel": filename,
                    }
                    for distribution, import_name, version, filename, digest
                    in PYTHON_PACKAGES
                },
                "requirements": PYTHON_REQUIREMENTS_PATH,
            } or "numpy_runtime" in protocol:
        raise EvidenceError("one-shot execution protocol drift")
    firewall = value.get("probe_firewall", {})
    if firewall.get("canonical_exact17_bindings") != [
        {"path": path, "sha256": digest} for path, digest in EXACT17
    ] or firewall.get("exclusion_bindings") != [
        {"path": path, "sha256": digest}
        for path, digest in EXCLUSION_BINDINGS
    ] or \
            firewall.get("selection_use") != "forbidden" or \
            len(firewall.get("cases", [])) != 17:
        raise EvidenceError("exact-17 firewall drift")


def _int_field(fields: Sequence[str], index: int, label: str) -> int:
    try:
        value = int(fields[index])
    except (IndexError, ValueError) as exc:
        raise EvidenceError(f"maintained actor has invalid {label}") from exc
    return value


_MAINTAINED_OBJECTIVE0_TAIL = (
    "800", "5", "0.02", "0", "1", "14", "0", "0", "0", "0",
    "3.5", "2", "4", "20", "0", "0", "20", "1", "0", "800",
    "1", "0", "0", "0", "0", "0", "0", "3", "1", "0", "0",
    "0", "0", "0", "0", "1",
)
_MAINTAINED_OBJECTIVE3_PREFIX = (
    "800", "5", "0.02", "0", "1", "0", "0", "0", "3", "0",
    "3.5", "2", "4", "20", "0", "0", "20", "1", "0", "800",
    "1", "0", "0", "0", "0", "0", "0", "3", "1", "0", "0",
    "0", "0", "0", "0", "1", "0", "0", "0", "1", "0",
)


def parse_maintained_actor(spec: str) -> dict[str, Any]:
    if not isinstance(spec, str) or "\n" in spec or "\r" in spec:
        raise EvidenceError("maintained actor must be one canonical line")
    parts = spec.split(":")
    kind = parts[0] if parts else ""
    if kind == "rolloutu":
        if len(parts) < 3:
            raise EvidenceError("truncated rolloutu actor")
        root, continuation, tail = parts[1], parts[1], parts[2:]
    elif kind == "rolloutu2":
        if len(parts) < 4:
            raise EvidenceError("truncated rolloutu2 actor")
        root, continuation, tail = parts[1], parts[2], parts[3:]
    else:
        raise EvidenceError("objective prerequisite winner is not rolloutu/rolloutu2")
    if root != continuation:
        raise EvidenceError("policy-cost v5 requires identical root/continuation bytes")
    _canonical_relative_path(root, "maintained actor model path")
    if len(tail) <= 8:
        raise EvidenceError("maintained actor rollout tail length is unsupported")
    objective = _int_field(tail, 8, "objective")
    match_value_path: str | None = None
    if objective == 3:
        if kind != "rolloutu2" or len(tail) != 42 or \
                tuple(tail[:41]) != _MAINTAINED_OBJECTIVE3_PREFIX:
            raise EvidenceError(
                "objective-3 winner is not the exact all-ply verified profile"
            )
        match_value_path = tail[41]
        if not match_value_path:
            raise EvidenceError("objective-3 winner lacks one bound match-value table")
        _canonical_relative_path(
            match_value_path, "maintained actor match-value path"
        )
    elif objective == 0:
        if kind != "rolloutu" or tuple(tail) != _MAINTAINED_OBJECTIVE0_TAIL:
            raise EvidenceError(
                "objective-0 winner is not the exact verified ply-14 profile"
            )
    else:
        raise EvidenceError("policy-cost prerequisite objective must be 0 or 3")
    return {
        "spec": spec,
        "kind": kind,
        "root_path": root,
        "continuation_path": continuation,
        "objective": objective,
        "match_value_path": match_value_path,
        "truth_metric": (
            "full_match_hybrid" if objective == 3
            else "current_round_margin"
        ),
    }


def _validate_objective3_evidence_completeness(
    value: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> None:
    names = [str(row.get("path")) for row in evidence]
    required = {
        "pre-efficacy-manifest.json", "transport/BUILD_INFO.txt",
        "transport/SHA256SUMS.txt", "transport/bindings/actors.json",
        "transport/bindings/definition-lock.json",
        "transport/bindings/execution.json", "transport/bindings/plan.json",
        "transport/bindings/pre-efficacy-manifest.json",
        "transport/bindings/table-manifest.json",
        "transport/data/models/match_value_objective3_v2_raw.lcmv",
        "transport/data/models/match_value_objective3_v2_projected.lcmv",
        "development/merged/development-selection.json",
        "development/merged/RAW_ALL_PLY-reciprocal.json",
        "development/merged/PROJECTED_ALL_PLY-reciprocal.json",
    }
    challenger = value.get("challenger_actor")
    safety = value.get("safety")
    final = value.get("final")
    if challenger is None:
        required.update({"stages/safety-skipped.json",
                         "stages/final-skipped.json"})
    else:
        required.update({"safety/merged/safety-decision.json",
                         "safety/merged/reciprocal.json"})
        if final is None:
            required.add("stages/final-skipped.json")
        else:
            required.update({"final/merged/final-decision.json",
                             "final/merged/reciprocal.json"})
    missing = sorted(required - set(names))
    if missing:
        raise EvidenceError(
            "objective-3 evidence omits required files: " + ", ".join(missing)
        )
    expected_raw = _objective3_raw_triplets(
        challenger is not None, final is not None
    )
    raw_prefixes = (
        "development/downloads/", "safety/downloads/", "final/downloads/",
    )
    actual_raw = {
        name for name in names
        if name.startswith(raw_prefixes) and
        name.endswith((".jsonl", ".sha256", ".time"))
    }
    if actual_raw != expected_raw:
        missing_raw = sorted(expected_raw - actual_raw)
        extra_raw = sorted(actual_raw - expected_raw)
        raise EvidenceError(
            "objective-3 raw triplet identity drift; missing=" +
            ",".join(missing_raw[:3]) + "; extra=" +
            ",".join(extra_raw[:3])
        )
    if safety is None and challenger is not None:
        raise EvidenceError("objective-3 challenger lacks a safety decision")


def _objective3_raw_triplets(
    include_safety: bool, include_final: bool
) -> set[str]:
    """Return the exact immutable raw member names produced by the O3 matrix."""

    names: set[str] = set()
    for variant in ("RAW_ALL_PLY", "PROJECTED_ALL_PLY"):
        for orientation in ("candidate-first", "baseline-first"):
            for start in range(0, 1000, 100):
                stem = f"{orientation}-{start}"
                prefix = (
                    "development/downloads/"
                    f"match-value-objective3-v2-development-{variant}-"
                    f"{orientation}-{start}/{stem}"
                )
                names.update(prefix + suffix for suffix in (
                    ".jsonl", ".sha256", ".time"
                ))
    if include_safety:
        for orientation in ("candidate-first", "baseline-first"):
            for start in range(0, 200, 20):
                prefix = f"safety/downloads/{orientation}-{start}"
                names.update(prefix + suffix for suffix in (
                    ".jsonl", ".sha256", ".time"
                ))
    if include_final:
        for orientation in ("candidate-first", "baseline-first"):
            for start in range(0, 2500, 100):
                prefix = f"final/downloads/{orientation}-{start}"
                names.update(prefix + suffix for suffix in (
                    ".jsonl", ".sha256", ".time"
                ))
    return names


def _verify_sha256sum_tree(root: Path, manifest: Path) -> None:
    """Reopen an exact GNU sha256sum archive manifest without trusting shell."""

    if not root.is_dir() or root.is_symlink() or not manifest.is_file() or \
            manifest.is_symlink():
        raise EvidenceError("objective-3 transport archive is absent or unsafe")
    try:
        payload = manifest.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise EvidenceError("objective-3 transport checksum manifest is invalid") from exc
    if not payload.endswith("\n"):
        raise EvidenceError("objective-3 transport checksum manifest is truncated")
    rows: list[tuple[str, str]] = []
    for line in payload.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([!-~]+)", line)
        if match is None:
            raise EvidenceError("objective-3 transport checksum row is malformed")
        digest, name = match.groups()
        relative = _canonical_relative_path(
            name, "objective-3 transport checksum path"
        ).as_posix()
        rows.append((relative, digest))
    names = [name for name, _ in rows]
    if not names or names != sorted(names) or len(names) != len(set(names)):
        raise EvidenceError("objective-3 transport checksum member order drift")
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise EvidenceError("objective-3 transport contains a symlink")
        if path.is_file() and path != manifest:
            actual.add(path.relative_to(root).as_posix())
    if set(names) != actual:
        raise EvidenceError("objective-3 transport checksum member set drift")
    for name, digest in rows:
        if sha256(root / name) != digest:
            raise EvidenceError("objective-3 transport checksum content drift")


def _rebuild_reciprocal_rooted(
    path: Path, gate_z: float, base_dir: Path
) -> tuple[dict[str, Any], str]:
    """Root recorded evidence paths without changing frozen shared helpers."""

    try:
        from tools.merge_arena import (
            _combine_reciprocal, _read_json_snapshot, _uint64_string,
            _validated_block, merge_block,
        )
    except ImportError:
        from merge_arena import (  # type: ignore
            _combine_reciprocal, _read_json_snapshot, _uint64_string,
            _validated_block, merge_block,
        )
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
        recorded_block = Path(str(snapshot["path"]))
        if recorded_block.is_absolute() or ".." in recorded_block.parts:
            raise EvidenceError("reciprocal block path is unsafe")
        block, digest = _read_json_snapshot(base_dir / recorded_block)
        if digest != snapshot["sha256"]:
            raise EvidenceError(f"reciprocal block {ordinal} digest mismatch")
        _validated_block(block)
        recorded_raw = [str(source["path"]) for source in block["inputs"]]
        raw_paths: list[Path] = []
        for recorded in map(Path, recorded_raw):
            if recorded.is_absolute() or ".." in recorded.parts:
                raise EvidenceError("reciprocal raw path is unsafe")
            raw_paths.append(base_dir / recorded)
        try:
            rebuilt = merge_block(
                raw_paths, _uint64_string(block["pair_start"], "pair_start"),
                block["pair_count"], allow_caps=True,
            )
        except Exception as exc:
            raise EvidenceError(
                "recorded raw input validation failed"
            ) from exc
        for source, recorded in zip(
                rebuilt["inputs"], recorded_raw, strict=True):
            source["path"] = recorded
        if rebuilt != block:
            raise EvidenceError(
                "merged block differs from its recorded raw inputs"
            )
        blocks.append(block)
        verified_snapshots.append({
            "path": str(recorded_block), "sha256": digest,
        })
        raw_blocks.append({
            "block": "first" if ordinal == 0 else "second",
            "pair_start": block["pair_start"],
            "pair_count": block["pair_count"], "inputs": rebuilt["inputs"],
        })
    rebuilt_reciprocal = _combine_reciprocal(
        blocks[0], blocks[1], verified_snapshots, gate_z=gate_z,
        require_positive_margin=True, raw_input_validation={
            "status": "validated",
            "method": (
                "reopened, SHA-256 checked, and exactly remerged recorded "
                "raw inputs"
            ),
            "blocks": raw_blocks,
        },
    )
    if rebuilt_reciprocal != actual:
        raise EvidenceError(
            "reciprocal result is not the exact raw-backed recomputation"
        )
    return actual, actual_digest


def _verify_objective3_raw_triplets(
    root: Path, *, include_safety: bool, include_final: bool
) -> None:
    _verify_sha256sum_tree(
        root / "development", root / "development/merged/SHA256SUMS.txt"
    )
    if include_safety:
        _verify_sha256sum_tree(
            root / "safety", root / "safety/merged/SHA256SUMS.txt"
        )
    if include_final:
        _verify_sha256sum_tree(
            root / "final", root / "final/merged/SHA256SUMS.txt"
        )
    expected = _objective3_raw_triplets(include_safety, include_final)
    json_names = sorted(name for name in expected if name.endswith(".jsonl"))
    timing = re.compile(
        r"wall_s=[0-9]+(?:\.[0-9]+)? user_s=[0-9]+(?:\.[0-9]+)? "
        r"sys_s=[0-9]+(?:\.[0-9]+)? max_rss_kb=[0-9]+ exit=0\n\Z"
    )
    for raw_name in json_names:
        raw = root / raw_name
        sidecar = raw.with_suffix(".sha256")
        time_path = raw.with_suffix(".time")
        if raw.stat().st_size <= 0:
            raise EvidenceError("objective-3 raw shard is empty")
        try:
            sidecar_text = sidecar.read_text(encoding="ascii")
            time_text = time_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise EvidenceError("objective-3 raw sidecar/time is invalid") from exc
        match = re.fullmatch(
            r"([0-9a-f]{64})  raw/([^/\n]+\.jsonl)\n", sidecar_text
        )
        if match is None or match.group(1) != sha256(raw) or \
                match.group(2) != raw.name:
            raise EvidenceError("objective-3 raw shard sidecar drift")
        if timing.fullmatch(time_text) is None:
            raise EvidenceError("objective-3 raw shard timing/exit drift")
        if raw_name.startswith("development/downloads/"):
            parts = Path(raw_name).parts
            artifact_name = parts[2]
            variant = next((name for name in (
                "RAW_ALL_PLY", "PROJECTED_ALL_PLY"
            ) if f"-{name}-" in artifact_name), None)
            if variant is None:
                raise EvidenceError("objective-3 development shard variant drift")
            panel_root = root / "development/panels" / variant
            for source in (raw, sidecar, time_path):
                panel = panel_root / source.name
                if panel.read_bytes() != source.read_bytes():
                    raise EvidenceError(
                        "objective-3 development panel differs from raw shard"
                    )


def _verify_objective3_build_info(
    transport: Path, execution: Mapping[str, Any], objective3: Any
) -> None:
    try:
        lines = (transport / "BUILD_INFO.txt").read_text(
            encoding="ascii"
        ).splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError("objective-3 build identity is unreadable") from exc
    metadata: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for line in lines:
        checksum = re.fullmatch(r"([0-9a-f]{64})  (transport/[!-~]+)", line)
        if checksum is not None:
            digest, name = checksum.groups()
            if name in checksums:
                raise EvidenceError("objective-3 build checksum is duplicated")
            checksums[name] = digest
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in metadata:
            raise EvidenceError("objective-3 build identity row is malformed")
        metadata[key] = value
    required = {
        "source_parent_commit": str(execution.get("source_parent_commit")),
        "source_parent_tree": str(execution.get("source_parent_tree")),
        "definition_lock_commit": str(
            execution.get("definition_lock", {}).get("lock_commit")
        ),
        "definition_commit": str(
            execution.get("definition_lock", {}).get("definition_commit")
        ),
        "definition_tree": str(
            execution.get("definition_lock", {}).get("definition_tree")
        ),
        "plan_sha": sha256(transport / "bindings/plan.json"),
        "definition_lock_sha":
            sha256(transport / "bindings/definition-lock.json"),
        "execution_sha": sha256(transport / "bindings/execution.json"),
        "final_actor_result_sha":
            sha256(transport / "bindings/final-actor-result.json"),
        "baseline": objective3.MAINTAINED_ACTOR,
        "world_cap": str(objective3.WORLD_CAP),
        "compiler_executable": objective3.COMPILER,
        "compiler_semantic_version_command":
            objective3.COMPILER_SEMANTIC_VERSION_COMMAND,
        "compiler_semantic_version":
            objective3.REQUIRED_COMPILER_SEMANTIC_VERSION,
        "cflags": objective3.CFLAGS,
        "ldflags": objective3.LDFLAGS,
        "build_profile": objective3.BUILD_PROFILE_HEX,
        "table_seed": objective3.TABLE_SEED,
        "table_samples_per_policy_lead": str(objective3.TABLE_SAMPLES),
        "table_total_controller_simulations":
            str(objective3.TABLE_TOTAL_ROUND_SIMULATIONS),
        "table_playout_symmetries": "20",
    }
    if any(metadata.get(key) != value for key, value in required.items()) or \
            not metadata.get("launch_commit") or \
            not metadata.get("compiler_banner") or \
            metadata.get("runner_os") != "Linux" or \
            metadata.get("runner_arch") != "X64" or \
            not metadata.get("runner_image_os") or \
            not metadata.get("runner_image_version") or \
            not metadata.get("uname"):
        raise EvidenceError("objective-3 frozen build identity drift")
    expected_checksum_paths = {
        "transport/bin/arena": transport / "bin/arena",
        "transport/bin/build_match_value": transport / "bin/build_match_value",
        "transport/data/champion.bin": transport / "data/champion.bin",
        f"transport/{objective3.RAW_TABLE_PATH}":
            transport / objective3.RAW_TABLE_PATH,
        f"transport/{objective3.PROJECTED_TABLE_PATH}":
            transport / objective3.PROJECTED_TABLE_PATH,
    }
    if set(checksums) != set(expected_checksum_paths) or any(
            checksums[name] != sha256(path)
            for name, path in expected_checksum_paths.items()):
        raise EvidenceError("objective-3 frozen build asset checksum drift")
    try:
        table_time = (transport / "table-build.time").read_text(
            encoding="ascii"
        )
        build_log = (transport / "table-build.log").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError) as exc:
        raise EvidenceError("objective-3 table build evidence is unreadable") from exc
    if re.fullmatch(
            r"wall_s=[0-9]+(?:\.[0-9]+)? user_s=[0-9]+(?:\.[0-9]+)? "
            r"sys_s=[0-9]+(?:\.[0-9]+)? max_rss_kb=[0-9]+ exit=0\n\Z",
            table_time) is None or \
            build_log.count("match-value table:") != 2 or \
            build_log.count("samples=16000 seed=202610200001") != 2 or \
            build_log.count("abi=1 build=0030d23b") != 2 or \
            build_log.count("role_cycle=400 role_balance=complete") != 2 or \
            build_log.count("variant=isotonic") != 1 or \
            build_log.count("variant=raw") != 1:
        raise EvidenceError("objective-3 table build log/timing drift")


def _verify_objective3_transport(
    root: Path, objective3: Any, actor: Mapping[str, Any], asset_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebuild the retained O3 definition, execution, tables, and actors."""

    transport = root / "transport"
    _verify_sha256sum_tree(transport, transport / "SHA256SUMS.txt")
    copied_pre = root / "pre-efficacy-manifest.json"
    transport_pre = transport / "bindings/pre-efficacy-manifest.json"
    if copied_pre.read_bytes() != transport_pre.read_bytes():
        raise EvidenceError("objective-3 pre-efficacy manifest copy drift")

    execution_path = transport / "bindings/execution.json"
    execution = strict_json(execution_path)
    if execution_path.read_bytes() != canonical_json(execution):
        raise EvidenceError("objective-3 retained execution is not canonical")
    _verify_objective3_build_info(transport, execution, objective3)
    definition_map = {
        objective3.FINAL_RESULT_PATH:
            transport / "bindings/final-actor-result.json",
        objective3.AUDIT_RESULT_PATH:
            transport / "bindings/audit/commented_ply_audit_v3_result.json",
        objective3.AUDIT_JSON_PATH:
            transport / "bindings/audit/commented_ply_audit_v3.json",
        objective3.AUDIT_MARKDOWN_PATH:
            transport / "bindings/audit/commented_ply_audit_v3.md",
        objective3.AUDIT_EVIDENCE_PATH:
            transport / "bindings/audit/commented_ply_audit_v3_evidence.zip",
        objective3.MODEL_PATH: transport / "data/champion.bin",
        "data/experiments/world800_result.json":
            transport / "bindings/world800-result.json",
    }
    with tempfile.TemporaryDirectory() as directory:
        replay_root = Path(directory)
        for relative, source in definition_map.items():
            if not source.is_file() or source.is_symlink():
                raise EvidenceError(
                    "objective-3 retained definition/source member is absent"
                )
            target = replay_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        rebuilt_final_actor = objective3.authoritative_final_result(replay_root)
        rebuilt_audit = objective3.authoritative_audit_result(replay_root)

    plan_path = transport / "bindings/plan.json"
    plan = strict_json(plan_path)
    objective3.validate_plan(plan)
    lock_path = transport / "bindings/definition-lock.json"
    definition_lock = strict_json(lock_path)
    lock_record = execution.get("definition_lock")
    if not isinstance(lock_record, dict) or lock_record != {
            "path": objective3.LOCK_PATH, "sha256": sha256(lock_path),
            "size": lock_path.stat().st_size,
            "lock_commit": execution.get("source_parent_commit"),
            "definition_commit": definition_lock.get("definition", {}).get(
                "commit"
            ),
            "definition_tree": definition_lock.get("definition", {}).get(
                "tree"
            ),
        }:
        raise EvidenceError("objective-3 execution definition-lock binding drift")
    # The lock commit is the source parent of the unique execution addendum;
    # its tree remains explicitly bound even though the terminal archive is
    # intentionally source-free.
    if lock_record.get("lock_commit") != execution.get("source_parent_commit") or \
            HEX40.fullmatch(str(execution.get("source_parent_tree", ""))) is None:
        raise EvidenceError("objective-3 execution source parent drift")
    definition_rows = definition_lock.get("definition_files")
    expected_definition_names = list(
        (*objective3.DEFINITION_PATHS, *objective3.DEPENDENCY_PATHS)
    )
    locked_sources = [
        (objective3.FINAL_RESULT_PATH,
         transport / "bindings/final-actor-result.json"),
        (objective3.MODEL_PATH, transport / "data/champion.bin"),
        (objective3.AUDIT_RESULT_PATH,
         transport / "bindings/audit/commented_ply_audit_v3_result.json"),
        (objective3.AUDIT_JSON_PATH,
         transport / "bindings/audit/commented_ply_audit_v3.json"),
        (objective3.AUDIT_MARKDOWN_PATH,
         transport / "bindings/audit/commented_ply_audit_v3.md"),
        (objective3.AUDIT_EVIDENCE_PATH,
         transport / "bindings/audit/commented_ply_audit_v3_evidence.zip"),
    ]
    expected_locked = [
        {"path": name, "sha256": sha256(source),
         "size": source.stat().st_size}
        for name, source in locked_sources
    ]
    expected_lock_build = {
        "runner": "ubuntu-24.04", "compiler": objective3.COMPILER,
        "compiler_semantic_version_command":
            objective3.COMPILER_SEMANTIC_VERSION_COMMAND,
        "required_compiler_semantic_version":
            objective3.REQUIRED_COMPILER_SEMANTIC_VERSION,
        "cflags": objective3.CFLAGS, "ldflags": objective3.LDFLAGS,
    }
    predecessor = definition_lock.get("predecessor_v1")
    predecessor_rows = predecessor.get("bindings") \
        if isinstance(predecessor, dict) else None
    predecessor_ok = isinstance(predecessor_rows, list) and \
        [row.get("path") for row in predecessor_rows
         if isinstance(row, dict)] == list(objective3.PREDECESSOR_V1) and \
        all(
            isinstance(row, dict) and set(row) == {
                "path", "sha256", "size"
            } and row.get("sha256") == objective3.PREDECESSOR_V1.get(
                str(row.get("path"))
            ) and isinstance(row.get("size"), int) and row["size"] > 0
            for row in predecessor_rows
        ) and predecessor.get("status") == "inert_never_executed" and \
        predecessor.get("execution_history_count") == 0
    if lock_path.read_bytes() != canonical_json(definition_lock) or \
            definition_lock.get("schema") != objective3.LOCK_SCHEMA or \
            definition_lock.get("artifact_kind") != \
            "immutable_match_value_objective3_v2_definition_lock" or \
            definition_lock.get("status") != \
            "sealed_before_table_build_or_match_value_efficacy" or \
            definition_lock.get("authoritative_final_actor_result") != \
            rebuilt_final_actor or \
            definition_lock.get("diagnostic_exact17_audit") != rebuilt_audit or \
            not isinstance(definition_rows, list) or \
            [row.get("path") for row in definition_rows
             if isinstance(row, dict)] != expected_definition_names or \
            any(not isinstance(row, dict) or set(row) != {
                    "path", "sha256", "size", "git_mode"
                } or HEX64.fullmatch(str(row.get("sha256", ""))) is None or
                not isinstance(row.get("size"), int) or row["size"] <= 0 or
                row.get("git_mode") not in {"100644", "100755"}
                for row in definition_rows) or \
            definition_lock.get("locked_artifacts") != expected_locked or \
            not predecessor_ok or \
            definition_lock.get("build") != expected_lock_build or \
            definition_lock.get("results") is not None:
        raise EvidenceError("objective-3 retained definition lock drift")
    retained_definition = {
        objective3.PLAN_PATH: transport / "bindings/plan.json",
        objective3.WORKFLOW_PATH: transport / "bindings/workflow.yml",
        objective3.HELPER_PATH:
            transport / "tools/match_value_objective3_v2.py",
        objective3.TEST_PATH:
            transport / "bindings/test_match_value_objective3_v2.py",
        objective3.DOC_PATH:
            transport / "bindings/MATCH_VALUE_OBJECTIVE3_V2.md",
    }
    row_by_name = {row["path"]: row for row in definition_rows}
    for name, retained in retained_definition.items():
        if row_by_name[name]["sha256"] != sha256(retained) or \
                row_by_name[name]["size"] != retained.stat().st_size:
            raise EvidenceError("objective-3 retained definition byte drift")

    expected_build = {
        "runner": "ubuntu-24.04", "compiler": objective3.COMPILER,
        "compiler_semantic_version_command":
            objective3.COMPILER_SEMANTIC_VERSION_COMMAND,
        "required_compiler_semantic_version":
            objective3.REQUIRED_COMPILER_SEMANTIC_VERSION,
        "cflags": objective3.CFLAGS, "ldflags": objective3.LDFLAGS,
    }
    expected_campaign = {
        "variants": list(objective3.VARIANTS),
        "development_pairs_per_orientation": objective3.DEVELOPMENT_PAIRS,
        "safety_pairs_per_orientation": objective3.SAFETY_PAIRS,
        "final_pairs_per_orientation": objective3.FINAL_PAIRS,
        "critical_z": objective3.CRITICAL_Z,
        "manual_dispatch": False, "retry": False,
        "optional_stopping": False, "repository_write": False,
    }
    if set(execution) != {
            "schema", "artifact_kind", "status", "source_parent_commit",
            "source_parent_tree", "definition_lock", "plan", "workflow",
            "helper", "authoritative_final_actor_result",
            "diagnostic_exact17_audit", "subject", "build", "campaign",
            "results",
        } or execution.get("schema") != objective3.EXECUTION_SCHEMA or \
            execution.get("artifact_kind") != \
            "locked_match_value_objective3_v2_execution" or \
            execution.get("status") != \
            "launch_bound_before_table_build_or_efficacy" or \
            execution.get("plan") != {
                "path": objective3.PLAN_PATH, "sha256": sha256(plan_path)
            } or execution.get("workflow") != {
                "path": objective3.WORKFLOW_PATH,
                "sha256": sha256(transport / "bindings/workflow.yml"),
            } or execution.get("helper") != {
                "path": objective3.HELPER_PATH,
                "sha256": sha256(
                    transport / "tools/match_value_objective3_v2.py"
                ),
            } or execution.get("authoritative_final_actor_result") != \
            rebuilt_final_actor or execution.get("diagnostic_exact17_audit") != \
            rebuilt_audit or execution.get("subject") != {
                "baseline": rebuilt_final_actor["winner"],
                "selection_rule": (
                    "mechanically revalidate final_actor_result; exact-17 "
                    "audit is provenance only and cannot select a candidate"
                ),
            } or execution.get("build") != expected_build or \
            execution.get("campaign") != expected_campaign or \
            execution.get("results") is not None:
        raise EvidenceError("objective-3 retained execution replay drift")

    raw_table = transport / objective3.RAW_TABLE_PATH
    projected_table = transport / objective3.PROJECTED_TABLE_PATH
    tables = strict_json(transport / "bindings/table-manifest.json")
    rebuilt_tables = objective3.table_manifest(raw_table, projected_table)
    rebuilt_tables["raw"]["path"] = objective3.RAW_TABLE_PATH
    rebuilt_tables["projected"]["path"] = objective3.PROJECTED_TABLE_PATH
    if tables != rebuilt_tables or \
            (transport / "bindings/table-manifest.json").read_bytes() != \
            canonical_json(tables):
        raise EvidenceError("objective-3 table manifest replay drift")
    actors = strict_json(transport / "bindings/actors.json")
    rebuilt_actors = objective3.build_actors(
        objective3.MAINTAINED_ACTOR, objective3.WORLD_CAP,
        objective3.RAW_TABLE_PATH, objective3.PROJECTED_TABLE_PATH,
    )
    if actors != rebuilt_actors or \
            (transport / "bindings/actors.json").read_bytes() != \
            canonical_json(actors):
        raise EvidenceError("objective-3 actor manifest replay drift")

    pre = strict_json(transport_pre)
    expected_binding_sources = {
        objective3.PLAN_PATH: transport / "bindings/plan.json",
        objective3.WORKFLOW_PATH: transport / "bindings/workflow.yml",
        f"campaign/{objective3.EXECUTION_PATH}": execution_path,
        objective3.FINAL_RESULT_PATH:
            transport / "bindings/final-actor-result.json",
        objective3.AUDIT_RESULT_PATH:
            transport / "bindings/audit/commented_ply_audit_v3_result.json",
        "transport/bindings/table-manifest.json":
            transport / "bindings/table-manifest.json",
        "transport/bindings/actors.json": transport / "bindings/actors.json",
        "transport/table-build.log": transport / "table-build.log",
        "transport/BUILD_INFO.txt": transport / "BUILD_INFO.txt",
    }
    expected_bindings = {
        name: sha256(source) for name, source in expected_binding_sources.items()
    }
    if transport_pre.read_bytes() != canonical_json(pre) or \
            pre.get("schema") != objective3.MANIFEST_SCHEMA or \
            pre.get("artifact_kind") != \
            "match_value_objective3_v2_pre_efficacy_manifest" or \
            pre.get("status") != \
            "tables_and_actors_frozen_before_first_efficacy_match" or \
            pre.get("source") != {
                "commit": execution["source_parent_commit"],
                "tree": execution["source_parent_tree"],
            } or pre.get("bindings") != expected_bindings or \
            pre.get("build") != execution.get("build") or \
            pre.get("tables") != tables or pre.get("actors") != actors or \
            pre.get("diagnostic_audit_selection_use") != "forbidden" or \
            pre.get("results") is not None:
        raise EvidenceError("objective-3 pre-efficacy freeze replay drift")

    tested_model = transport / objective3.MODEL_PATH
    current_model = asset_root / str(actor["root_path"])
    if tested_model.read_bytes() != current_model.read_bytes():
        raise EvidenceError("objective-3 maintained checkpoint differs from tested bytes")
    match_path = actor.get("match_value_path")
    if match_path is not None and \
            (transport / str(match_path)).read_bytes() != \
            (asset_root / str(match_path)).read_bytes():
        raise EvidenceError("objective-3 maintained table differs from tested bytes")
    return execution, actors, tables


def _verify_objective3_prerequisite_evidence(
    root: Path, value: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]],
    actor: Mapping[str, Any], asset_root: Path,
) -> None:
    """Rebuild the O3 selection/gates/disposition from retained raw evidence."""

    _validate_objective3_evidence_completeness(value, evidence)
    try:
        from tools import match_value_objective3_v2 as objective3
    except ImportError:
        import match_value_objective3_v2 as objective3  # type: ignore
    retained_helper = root / "transport/tools/match_value_objective3_v2.py"
    loaded_helper = Path(objective3.__file__).resolve()
    if not retained_helper.is_file() or retained_helper.is_symlink() or \
            sha256(retained_helper) != sha256(loaded_helper):
        raise EvidenceError("objective-3 replay helper differs from retained authority")

    execution, actors_value, table_value = _verify_objective3_transport(
        root, objective3, actor, asset_root
    )
    _verify_objective3_raw_triplets(
        root, include_safety=value.get("challenger_actor") is not None,
        include_final=value.get("final") is not None,
    )
    actor_map = actors_value.get("actors")
    if not isinstance(actor_map, dict) or set(actor_map) != {
            "legacy", "RAW_ALL_PLY", "PROJECTED_ALL_PLY"}:
        raise EvidenceError("objective-3 retained actor family drift")
    baseline = actor_map["legacy"]
    if baseline != value.get("baseline_actor"):
        raise EvidenceError("objective-3 retained baseline actor drift")

    common_provenance = (
        f"plan={sha256(root / 'transport/bindings/plan.json')};"
        f"lock={sha256(root / 'transport/bindings/definition-lock.json')};"
        f"execution={sha256(root / 'transport/bindings/execution.json')};"
        f"post_build={sha256(root / 'transport/bindings/pre-efficacy-manifest.json')};"
        f"source={execution['source_parent_commit']};"
        f"arena={sha256(root / 'transport/bin/arena')};"
        f"model={sha256(root / 'transport/data/champion.bin')}"
    )

    def provenance(stage: str, variant: str, *,
                   development: str | None = None,
                   safety: str | None = None) -> str:
        table_key = "projected" if variant == "PROJECTED_ALL_PLY" else "raw"
        fields = [
            f"stage=match_value_objective3_v2_{stage}", f"variant={variant}",
        ]
        if development is not None:
            fields.append(f"development={development}")
        if safety is not None:
            fields.append(f"safety={safety}")
        fields.extend((
            common_provenance,
            f"table={table_value[table_key]['sha256']}",
            f"worlds={objective3.WORLD_CAP}", "threads=4",
        ))
        return ";".join(fields)

    panels: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    candidate_actors = {name: actor_map[name] for name in objective3.VARIANTS}
    for name in objective3.VARIANTS:
        relative = f"development/merged/{name}-reciprocal.json"
        panel, digest = _rebuild_reciprocal_rooted(
            root / relative, objective3.CRITICAL_Z, root / "development"
        )
        expected_provenance = provenance("development", name)
        objective3._validate_panel_identity(
            panel, candidate_actors[name], baseline, expected_provenance,
            objective3.DEVELOPMENT_PAIRS, objective3.DEVELOPMENT_SEEDS[0],
            objective3.DEVELOPMENT_SEEDS[1],
        )
        panels[name], digests[name] = panel, digest
    rebuilt_development = objective3.development_selection(
        panels, candidate_actors, digests
    )
    development_path = root / "development/merged/development-selection.json"
    if strict_json(development_path) != rebuilt_development or \
            value.get("development") != rebuilt_development:
        raise EvidenceError("objective-3 development selection replay drift")

    challenger = rebuilt_development.get("selected_actor")
    rebuilt_safety: dict[str, Any] | None = None
    rebuilt_final: dict[str, Any] | None = None
    if challenger is not None:
        selected_variant = next(
            (name for name, spec in candidate_actors.items()
             if spec == challenger), None
        )
        if selected_variant is None:
            raise EvidenceError("objective-3 selected actor has no frozen variant")
        development_sha = sha256(development_path)
        safety_path = root / "safety/merged/safety-decision.json"
        safety_value = strict_json(safety_path)
        safety_panel, safety_digest = _rebuild_reciprocal_rooted(
            root / "safety/merged/reciprocal.json", objective3.CRITICAL_Z,
            root / "safety",
        )
        safety_provenance = provenance(
            "safety", selected_variant, development=development_sha
        )
        objective3._validate_panel_identity(
            safety_panel, challenger, baseline, safety_provenance,
            objective3.SAFETY_PAIRS, objective3.SAFETY_SEEDS[0],
            objective3.SAFETY_SEEDS[1],
        )
        rebuilt_safety = objective3.safety_gate(safety_panel)
        rebuilt_safety.update({
            "candidate": challenger, "baseline": baseline,
            "provenance": safety_provenance,
            "reciprocal_path": "merged/reciprocal.json",
            "reciprocal_sha256": safety_digest,
            "seeds": {"candidate_first": objective3.SAFETY_SEEDS[0],
                      "baseline_first": objective3.SAFETY_SEEDS[1]},
        })
        if safety_value != rebuilt_safety or value.get("safety") != rebuilt_safety:
            raise EvidenceError("objective-3 safety gate replay drift")
        if rebuilt_safety.get("passed") is True:
            final_path = root / "final/merged/final-decision.json"
            final_value = strict_json(final_path)
            final_panel, final_digest = _rebuild_reciprocal_rooted(
                root / "final/merged/reciprocal.json", objective3.CRITICAL_Z,
                root / "final",
            )
            final_provenance = provenance(
                "final", selected_variant, development=development_sha,
                safety=sha256(safety_path),
            )
            objective3._validate_panel_identity(
                final_panel, challenger, baseline, final_provenance,
                objective3.FINAL_PAIRS, objective3.FINAL_SEEDS[0],
                objective3.FINAL_SEEDS[1],
            )
            rebuilt_final = objective3.final_gate(final_panel)
            rebuilt_final.update({
                "candidate": challenger, "baseline": baseline,
                "provenance": final_provenance,
                "reciprocal_path": "merged/reciprocal.json",
                "reciprocal_sha256": final_digest,
                "seeds": {"candidate_first": objective3.FINAL_SEEDS[0],
                          "baseline_first": objective3.FINAL_SEEDS[1]},
            })
            if final_value != rebuilt_final or value.get("final") != rebuilt_final:
                raise EvidenceError("objective-3 final gate replay drift")
        elif value.get("final") is not None:
            raise EvidenceError("objective-3 final exists after failed safety")
    rebuilt_result = objective3.terminal_result(
        execution, rebuilt_development, rebuilt_safety, rebuilt_final,
        [dict(row) for row in evidence],
    )
    if dict(value) != rebuilt_result:
        raise EvidenceError("objective-3 authoritative disposition replay drift")


def authoritative_prerequisite(
    root: Path, assets_root: Path | None = None
) -> dict[str, Any]:
    path = root / PREREQUISITE_PATH
    value = strict_json(path)
    if value.get("schema") != "lc-match-value-objective3-v2-result-v1" or \
            value.get("artifact_kind") != \
            "match_value_objective3_v2_authoritative_result" or \
            value.get("status") != "complete" or \
            value.get("locked_validation_relaxed") is not False or \
            value.get("diagnostic_audit_used_for_selection") is not False or \
            type(value.get("promotion_gate_passed")) is not bool:
        raise EvidenceError("objective-3 prerequisite is not authoritative and terminal")
    baseline = value.get("baseline_actor")
    winner = value.get("winner_actor")
    challenger = value.get("challenger_actor")
    passed = value["promotion_gate_passed"]
    if not isinstance(baseline, str) or not isinstance(winner, str) or \
            (passed and (not isinstance(challenger, str) or winner != challenger)) or \
            (not passed and winner != baseline):
        raise EvidenceError("objective-3 disposition/winner identity is inconsistent")
    actor = parse_maintained_actor(winner)
    if (passed and actor["objective"] != 3) or \
            (not passed and actor["objective"] != 0):
        raise EvidenceError("objective-3 gate and actor objective disagree")
    asset_base = root if assets_root is None else assets_root
    assets = [binding(asset_base, actor["root_path"])]
    if actor["match_value_path"] is not None:
        assets.append(binding(asset_base, actor["match_value_path"]))
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence or any(
        not isinstance(item, dict) or set(item) != {"path", "sha256"} or
        not isinstance(item["path"], str) or
        not isinstance(item["sha256"], str) or
        HEX64.fullmatch(item["sha256"]) is None
        for item in evidence
    ):
        raise EvidenceError("objective-3 result has no complete evidence manifest")
    evidence_paths: set[str] = set()
    evidence_bindings: list[dict[str, Any]] = []
    for item in evidence:
        relative_path = _canonical_relative_path(
            item["path"], "objective-3 evidence path"
        )
        relative = relative_path.as_posix()
        if relative in evidence_paths:
            raise EvidenceError("objective-3 evidence manifest has duplicate paths")
        evidence_paths.add(relative)
        bound = binding(root, relative)
        if bound["sha256"] != item["sha256"]:
            raise EvidenceError("objective-3 evidence SHA-256 drift")
        evidence_bindings.append(bound)
    _verify_objective3_prerequisite_evidence(
        root, value, evidence, actor, asset_base
    )
    return {
        "result": binding(root, PREREQUISITE_PATH),
        "promotion_gate_passed": passed,
        "disposition": value.get("disposition"),
        "actor": actor,
        "assets": assets,
        "evidence": evidence_bindings,
        "terminal_evidence_files": len(evidence),
    }


def _tail(*, floor: float, ply_lo: int, objective: int,
          match_value_path: str | None) -> list[str]:
    fields = [
        "800", "5", f"{floor:.2f}", "0", "1", str(ply_lo), "0", "0",
        str(objective), "0", "3.5", "0", "4", "20", "0", "0", "20",
        "1", "0", "800", "1", "0", "0", "0", "0", "0", "0", "0",
        "1", "0", "0", "0", "0", "0", "3", "1", "0", "0", "0",
        "1", "0",
    ]
    if len(fields) != 41:
        raise AssertionError("rollout tail construction bug")
    if objective == 3:
        if not match_value_path:
            raise EvidenceError("objective-3 actor needs match-value path")
        fields.append(match_value_path)
    else:
        if match_value_path is not None:
            raise EvidenceError("objective-0 actor cannot bind match-value path")
        # Field 40 is action_ranker_min and is legal only when field 41 binds
        # an action-ranker/match-value role.  Objective zero has neither, so
        # its canonical parser tail ends at field 39.
        fields.pop()
    return fields


def neutral_actor(actor: Mapping[str, Any]) -> str:
    head = f"rolloutu2:{actor['root_path']}:{actor['continuation_path']}:"
    return head + ":".join(_tail(
        floor=0.01, ply_lo=0, objective=int(actor["objective"]),
        match_value_path=actor.get("match_value_path"),
    ))


def policy_cost_actor(actor: Mapping[str, Any], *, artifact_path: str,
                      floor: float, ply_lo: int) -> str:
    if ":" in artifact_path or not artifact_path:
        raise EvidenceError("policy-cost artifact path is not actor-safe")
    head = (
        f"rolloutu5:{actor['root_path']}:{actor['continuation_path']}:"
        f"{artifact_path}:"
    )
    return head + ":".join(_tail(
        floor=floor, ply_lo=ply_lo, objective=int(actor["objective"]),
        match_value_path=actor.get("match_value_path"),
    ))


def expected_execution(root: Path, source_commit: str,
                       source_tree: str) -> dict[str, Any]:
    if HEX40.fullmatch(source_commit) is None or HEX40.fullmatch(source_tree) is None:
        raise EvidenceError("source parent commit/tree must be canonical SHA-1")
    plan = strict_json(root / PLAN_PATH)
    validate_plan(plan)
    if sha256(root / DATASET_SOURCE_PATH) != DATASET_SOURCE_SHA256:
        raise EvidenceError("portable v19 dataset source binding drift")
    for path, digest in EXACT17:
        if sha256(root / path) != digest:
            raise EvidenceError(f"exact-17 binding drift: {path}")
    for path, digest in EXCLUSION_BINDINGS:
        if sha256(root / path) != digest:
            raise EvidenceError(f"exact-17 exclusion binding drift: {path}")
    exclusion_json = strict_json(root / EXCLUSION_BINDINGS[1][0])
    if exclusion_json.get("schema") != \
            plan["v19_hierarchical_draw_hypothesis"]["exact17_json_schema"] or \
            exclusion_json.get("canonical_payload_sha256") != \
            plan["v19_hierarchical_draw_hypothesis"]["exact17_canonical_payload_sha256"]:
        raise EvidenceError("portable exact-17 semantic binding drift")
    for path, digest in PREDECESSOR_ATTEMPT_BINDINGS:
        if sha256(root / path) != digest:
            raise EvidenceError(f"predecessor attempt binding drift: {path}")
    prerequisite = authoritative_prerequisite(root)
    actor = prerequisite["actor"]
    return {
        "schema": EXECUTION_SCHEMA,
        "artifact_kind": "locked_policy_cost_v19_execution",
        "status": "launch_bound_before_discovery_or_any_search_truth_label",
        "source_parent_commit": source_commit,
        "source_parent_tree": source_tree,
        "bindings": {
            "plan": binding(root, PLAN_PATH),
            "workflow": binding(root, WORKFLOW_PATH),
            "helper": binding(root, HELPER_PATH),
            "objective3_prerequisite": prerequisite,
            "exact17": [binding(root, path) for path, _ in EXACT17],
            "exact17_exclusion_manifests": [
                binding(root, path) for path, _ in EXCLUSION_BINDINGS
            ],
            "predecessor_attempt": [
                binding(root, path)
                for path, _ in PREDECESSOR_ATTEMPT_BINDINGS
            ],
        },
        "build": {
            "runner": "ubuntu-24.04",
            "compiler": COMPILER,
            "compiler_semantic_version_command": COMPILER_VERSION_COMMAND,
            "required_compiler_semantic_version": COMPILER_VERSION,
            "cflags": CFLAGS,
            "ldflags": LDFLAGS,
            "binding": "compile exactly once in preflight; source-free SHA-256 transport thereafter",
        },
        "subject": {
            "maintained_actor": actor["spec"],
            "neutral_counterfactual_actor": neutral_actor(actor),
            "objective": actor["objective"],
            "train_truth_metric": actor["truth_metric"],
            "root_path": actor["root_path"],
            "continuation_path": actor["continuation_path"],
            "match_value_path": actor["match_value_path"],
        },
        "fixed_budgets": {
            "discovery_matches": DISCOVERY_MATCHES,
            "train_records": TRAIN_RECORDS,
            "select_records": HOLDOUT_RECORDS,
            "test_records": HOLDOUT_RECORDS,
            "configurations": list(CONFIG_IDS),
            "safety_pairs_per_orientation": 200,
            "final_pairs_per_orientation": 2500,
        },
        "fixed_seeds": {
            "discovery": DISCOVERY_SEEDS,
            "primary": PRIMARY_SEEDS,
            "fresh": FRESH_SEEDS,
            "truth": TRUTH_SEEDS,
            "maintained": MAINTAINED_SEEDS,
            "calibration_folds": POLICY_COST_SOURCE_SEED,
            "select_bootstrap": "202806150101",
            "safety": {
                "candidate_first": "202806210101",
                "baseline_first": "202806210102",
            },
            "final": {
                "candidate_first": "202806220101",
                "baseline_first": "202806220102",
            },
        },
        "results": None,
    }


def prepare_execution(root: Path, output: Path, source_commit: str,
                      source_tree: str) -> dict[str, Any]:
    expected_path = (root / EXECUTION_PATH).resolve()
    if output.resolve() != expected_path:
        raise EvidenceError("execution addendum must use its one canonical path")
    value = expected_execution(root, source_commit, source_tree)
    write_no_clobber(output, canonical_json(value))
    return value


def guard_execution(root: Path, execution: Path, source_commit: str,
                    source_tree: str) -> dict[str, Any]:
    expected = expected_execution(root, source_commit, source_tree)
    if strict_json(execution) != expected:
        raise EvidenceError("execution addendum differs from mechanical binding")
    if execution.read_bytes() != canonical_json(expected):
        raise EvidenceError("execution addendum is not canonical JSON")
    return expected


def _build_identity(transport: Path) -> dict[str, Any]:
    """Validate the preflight's canonical compiler/command/binary binding."""

    relative = "bindings/build-identity.json"
    path = transport / relative
    value = strict_json(path)
    if path.read_bytes() != canonical_json(value):
        raise EvidenceError("build identity is not canonical JSON")
    required = {
        "schema", "runner", "architecture", "compiler",
        "compiler_semantic_version_command", "compiler_semantic_version",
        "python3_version", "python_packages",
        "cflags", "ldflags", "compile_commands", "binaries",
        "numeric_runtime_environment",
    }
    if set(value) != required or value.get("schema") != \
            "lc-policy-cost-v19-build-identity-v2" or \
            value.get("runner") != "ubuntu-24.04" or \
            value.get("architecture") != "x86_64" or \
            value.get("compiler") != COMPILER or \
            value.get("compiler_semantic_version_command") != \
            COMPILER_VERSION_COMMAND or \
            value.get("compiler_semantic_version") != COMPILER_VERSION or \
            value.get("cflags") != CFLAGS or value.get("ldflags") != LDFLAGS:
        raise EvidenceError("build identity lock drift")
    if value.get("numeric_runtime_environment") != NUMERIC_RUNTIME_ENV:
        raise EvidenceError("build identity numeric runtime environment drift")
    runtime = value.get("python3_version")
    if not isinstance(runtime, str) or not runtime or not runtime.isascii() or \
            "\n" in runtime or "\r" in runtime or len(runtime) > 128 or \
            runtime != "Python 3.12.3":
        raise EvidenceError("build identity Python version is not frozen")
    packages = value.get("python_packages")
    if not isinstance(packages, list) or len(packages) != len(PYTHON_PACKAGES):
        raise EvidenceError("build identity Python package set drift")
    expected_filenames: set[str] = set()
    for item, expected in zip(packages, PYTHON_PACKAGES, strict=True):
        distribution, import_name, version, filename, digest = expected
        expected_filenames.add(filename)
        if not isinstance(item, dict) or set(item) != {
            "distribution", "import_name", "version", "wheel"
        } or item.get("distribution") != distribution or \
                item.get("import_name") != import_name or \
                item.get("version") != version:
            raise EvidenceError("build identity Python package record drift")
        wheel = item.get("wheel")
        expected_path = f"bindings/runtime/{filename}"
        if not isinstance(wheel, dict) or \
                set(wheel) != {"path", "sha256", "size"} or \
                wheel.get("path") != expected_path or \
                wheel.get("sha256") != digest or \
                isinstance(wheel.get("size"), bool) or \
                not isinstance(wheel.get("size"), int) or wheel["size"] <= 0:
            raise EvidenceError("build identity Python wheel binding drift")
        wheel_path = transport / expected_path
        if not wheel_path.is_file() or wheel_path.is_symlink() or \
                sha256(wheel_path) != digest or \
                wheel_path.stat().st_size != wheel["size"]:
            raise EvidenceError("frozen Python wheel is missing or changed")
    runtime_root = transport / "bindings/runtime"
    actual_wheels = {
        path.name for path in runtime_root.glob("*.whl")
        if path.is_file() and not path.is_symlink()
    }
    if actual_wheels != expected_filenames:
        raise EvidenceError("frozen Python wheel set contains drift")
    requirements = runtime_root / "requirements.txt"
    expected_requirements = "".join(
        f"{distribution}=={version} --hash=sha256:{digest}\n"
        for distribution, _, version, _, digest in PYTHON_PACKAGES
    )
    if not requirements.is_file() or requirements.is_symlink() or \
            requirements.read_text(encoding="ascii") != expected_requirements:
        raise EvidenceError("frozen Python requirements binding drift")
    commands = value.get("compile_commands")
    if commands != list(BUILD_COMPILE_COMMANDS):
        raise EvidenceError("build identity compile command/order drift")
    binaries = value.get("binaries")
    if not isinstance(binaries, list) or {item.get("path") for item in binaries
            if isinstance(item, dict)} != {
                "bin/arena", "bin/build_policy_cost", "bin/policy_cost_dataset"
            }:
        raise EvidenceError("build identity binary set drift")
    for item in binaries:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size", "mode"} or \
                not isinstance(item["path"], str) or \
                HEX64.fullmatch(item["sha256"]) is None or \
                isinstance(item["size"], bool) or not isinstance(item["size"], int) or \
                item["size"] <= 0 or item["mode"] != "0755":
            raise EvidenceError("build identity binary record is malformed")
        binary = transport / item["path"]
        if not binary.is_file() or binary.is_symlink() or \
                sha256(binary) != item["sha256"] or binary.stat().st_size != item["size"] or \
                binary.stat().st_mode & 0o777 != 0o755:
            raise EvidenceError("build identity binary hash/size drift")
    return binding(transport, relative)


def verify_runtime(transport: Path) -> None:
    """Verify exact Python and both sealed wheels without importing packages."""

    _build_identity(transport)
    identity = strict_json(transport / "bindings" / "build-identity.json")
    actual_python = subprocess.check_output(
        ["python3", "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    if actual_python != identity["python3_version"]:
        raise EvidenceError("Python runtime differs from frozen build identity")
    for key, expected in NUMERIC_RUNTIME_ENV.items():
        if os.environ.get(key) != expected:
            raise EvidenceError(f"numeric runtime environment drift: {key}")


def pre_efficacy_manifest(execution: Path, transport: Path) -> dict[str, Any]:
    bound = strict_json(execution)
    if bound.get("schema") != EXECUTION_SCHEMA or bound.get("results") is not None:
        raise EvidenceError("pre-efficacy manifest requires pristine execution")
    if not transport.is_dir() or transport.is_symlink():
        raise EvidenceError("transport must be one real directory")
    files: list[dict[str, Any]] = []
    forbidden_suffixes = {".c", ".h", ".o", ".state"}
    for path in sorted(transport.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise EvidenceError(f"transport contains symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(transport).as_posix()
        if path.suffix in forbidden_suffixes or relative.startswith("data/probes/"):
            raise EvidenceError(f"source/probe material entered transport: {relative}")
        files.append({
            "path": relative,
            "sha256": sha256(path),
            "size": path.stat().st_size,
            "mode": f"{path.stat().st_mode & 0o777:04o}",
        })
    build_identity = _build_identity(transport)
    build_value = strict_json(transport / "bindings/build-identity.json")
    required = {
        "bin/arena", "bin/build_policy_cost", "bin/policy_cost_dataset",
        "bindings/execution.json", "bindings/plan.json",
        "bindings/workflow.yml", "bindings/build-identity.json",
        "bindings/runtime/requirements.txt",
        "data/champion.bin", "tools/gate_actor_panel.py", "tools/merge_arena.py",
        "tools/match_value_objective3_v2.py", "tools/flagged_ply_execution.py",
        "tools/policy_cost_allocate_v19.py",
        "tools/policy_cost_artifact_v19.py", "tools/policy_cost_calibration_v19.py",
        "tools/policy_cost_campaign_v19.py", "tools/policy_cost_selection_v19.py",
    } | {
        item["wheel"]["path"] for item in build_value["python_packages"]
    } | {
        f"bindings/exact17/{Path(path).name}" for path, _ in EXACT17
    } | {
        f"bindings/exact17/{Path(path).name}"
        for path, _ in EXCLUSION_BINDINGS
    } | {
        f"bindings/predecessor/{Path(path).name}"
        for path, _ in PREDECESSOR_ATTEMPT_BINDINGS
    }
    subject = bound.get("subject")
    prerequisite = bound.get("bindings", {}).get("objective3_prerequisite")
    if not isinstance(subject, dict) or not isinstance(prerequisite, dict) or \
            not isinstance(prerequisite.get("result"), dict) or \
            not isinstance(prerequisite.get("evidence"), list):
        raise EvidenceError("execution lacks prerequisite transport bindings")
    for asset in (subject.get("root_path"), subject.get("match_value_path")):
        if asset is not None:
            required.add(_canonical_relative_path(
                asset, "pre-efficacy actor asset"
            ).as_posix())
    for record in [prerequisite["result"], *prerequisite["evidence"]]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise EvidenceError("execution prerequisite file binding drift")
        relative = _canonical_relative_path(
            record["path"], "pre-efficacy prerequisite path"
        )
        required.add(f"bindings/objective3/repo/{relative.as_posix()}")
    names = {item["path"] for item in files}
    missing = sorted(required - names)
    if missing:
        raise EvidenceError("transport lacks required files: " + ", ".join(missing))
    return {
        "schema": MANIFEST_SCHEMA,
        "artifact_kind": "policy_cost_v19_pre_efficacy_manifest",
        "status": "frozen_before_first_search_or_truth_label",
        "execution_sha256": sha256(execution),
        "source_free_after_preflight": True,
        "probe_states_absent": True,
        "build_identity": build_identity,
        "files": files,
        "results": None,
    }


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def _evaluation(path: Path, split: str) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Any]
]:
    rows = strict_jsonl(path)
    if len(rows) < 3 or rows[0].get("record_type") != "header" or \
            rows[-1].get("record_type") != "footer" or any(
                row.get("record_type") != "allocation" for row in rows[1:-1]
            ):
        raise EvidenceError("evaluation must be header/allocations/footer")
    header, allocations, footer = rows[0], rows[1:-1], rows[-1]
    if header.get("schema") != "lc-policy-cost-evaluation-v1" or \
            footer.get("schema") != "lc-policy-cost-evaluation-v1" or \
            header.get("split") != split:
        raise EvidenceError("evaluation split/schema drift")
    expected = TRAIN_RECORDS if split == "TRAIN" else HOLDOUT_RECORDS
    if len(allocations) != expected or header.get("full_manifest_records") != expected or \
            header.get("allocation_count") != expected or \
            footer.get("records") != expected:
        raise EvidenceError(f"{split} evaluation count differs from {expected}")
    caps = (
        "primary_unfinished_cap_leaves", "fresh_unfinished_cap_leaves",
        "truth_cap_hits", "maintained_unfinished_cap_leaves",
    )
    if any(footer.get(field) != 0 for field in caps) or \
            footer.get("all_exact") is not True:
        raise EvidenceError(f"{split} evaluation is capped or inexact")
    if header.get("primary") != {
        "worlds": 800, "seed": PRIMARY_SEEDS[split]
    } or header.get("fresh") != {
        "worlds": 800, "seed": FRESH_SEEDS[split]
    } or header.get("truth") != {
        "controller": "exact_policy20_full_remaining_match",
        "seed": TRUTH_SEEDS[split], "worlds": TRUTH_WORLDS[split],
    }:
        raise EvidenceError(f"{split} P/F/T seed or budget drift")
    if header.get("seed_domains_pairwise_disjoint") is not True or \
            header.get("burned_source_deal_seeds") != \
            BURNED_SOURCE_DEAL_SEEDS or \
            header.get("burned_seed_intersection") != 0:
        raise EvidenceError(f"{split} burned-seed evidence drift")
    if header.get("maintained_root_seed") != MAINTAINED_SEEDS[split]:
        raise EvidenceError(f"{split} maintained-decision seed drift")
    ids = [row.get("allocation_id") for row in allocations]
    if ids != list(range(expected)):
        raise EvidenceError(f"{split} allocation ids are not complete and ordered")
    return header, allocations, footer


def merge_evaluation_slices(paths: Sequence[Path], split: str) -> bytes:
    """Fail-closed concatenate the fixed contiguous evaluator slice set.

    No shard is inferred from its contents: the stage fixes both the complete
    count and the exact contiguous starts.  The returned JSONL carries each
    raw slice SHA-256 in its header before any calibration or inference reads
    an allocation row.
    """
    total = TRAIN_RECORDS if split == "TRAIN" else HOLDOUT_RECORDS
    slices = EVALUATION_SLICES.get(split)
    if slices is None:
        raise EvidenceError("unknown evaluation slice split")
    if len(paths) != slices or total % slices:
        raise EvidenceError("evaluation slice cardinality drift")
    if len({path.name for path in paths}) != slices:
        raise EvidenceError("evaluation slice path identity collision")
    per_slice = total // slices
    parsed: list[tuple[int, Path, list[dict[str, Any]]]] = []
    for path in paths:
        raw = strict_jsonl(path)
        if len(raw) != per_slice + 2 or raw[0].get("record_type") != "header" or \
                raw[-1].get("record_type") != "footer" or any(
                    item.get("record_type") != "allocation" for item in raw[1:-1]
                ):
            raise EvidenceError("evaluation slice record shape drift")
        start = raw[0].get("allocation_start")
        if isinstance(start, bool) or not isinstance(start, int):
            raise EvidenceError("evaluation slice start/count/id drift")
        parsed.append((start, path, raw))
    parsed.sort(key=lambda item: item[0])
    if [start for start, _, _ in parsed] != [index * per_slice
                                              for index in range(slices)]:
        raise EvidenceError("evaluation slice start/count/id drift")
    headers: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    footers: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for ordinal, (start, path, raw) in enumerate(parsed):
        header, slice_rows, footer = raw[0], raw[1:-1], raw[-1]
        if path.name != f"{ordinal}.jsonl" or \
                header.get("schema") != "lc-policy-cost-evaluation-v1" or \
                header.get("split") != split or \
                header.get("allocation_start") != start or \
                header.get("allocation_count") != per_slice or \
                header.get("full_manifest_records") != total or \
                footer.get("schema") != "lc-policy-cost-evaluation-v1" or \
                footer.get("allocation_start") != start or \
                footer.get("records") != per_slice or \
                [item.get("allocation_id") for item in slice_rows] != \
                list(range(start, start + per_slice)):
            raise EvidenceError("evaluation slice start/count/id drift")
        headers.append(header)
        rows.extend(slice_rows)
        footers.append(footer)
        bindings.append({
            "ordinal": ordinal, "path": path.name, "sha256": sha256(path),
            "allocation_start": start, "allocation_count": per_slice,
        })
    header_variable = {
        "allocation_start", "allocation_count",
    }
    header_identity = {
        key: value for key, value in headers[0].items()
        if key not in header_variable
    }
    if any(
            {key: value for key, value in header.items()
             if key not in header_variable} != header_identity
            for header in headers[1:]):
        raise EvidenceError("evaluation slice header binding drift")
    footer_additive = {
        "primary_unfinished_cap_leaves", "fresh_unfinished_cap_leaves",
        "truth_cap_hits", "maintained_unfinished_cap_leaves",
        "exact_terminal_leaves", "maintained_exact_terminal_leaves",
    }
    footer_variable = footer_additive | {"allocation_start", "records"}
    footer_identity = {
        key: value for key, value in footers[0].items()
        if key not in footer_variable
    }
    if any(
            {key: value for key, value in footer.items()
             if key not in footer_variable} != footer_identity
            for footer in footers[1:]):
        raise EvidenceError("evaluation slice footer binding drift")
    merged_header = dict(headers[0])
    merged_header.update({
        "allocation_start": 0, "allocation_count": total,
        "raw_slice_bindings": bindings,
    })
    merged_footer = dict(footers[0])
    merged_footer.update({
        "allocation_start": 0, "records": total,
        "raw_slice_bindings": bindings,
    })
    for field in (
        "primary_unfinished_cap_leaves", "fresh_unfinished_cap_leaves",
        "truth_cap_hits", "maintained_unfinished_cap_leaves",
        "exact_terminal_leaves", "maintained_exact_terminal_leaves",
    ):
        values = [footer.get(field) for footer in footers]
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in values):
            raise EvidenceError("evaluation slice footer count drift")
        merged_footer[field] = sum(values)
    if any(footer.get("all_exact") is not True for footer in footers) or any(
            merged_footer[field] != 0 for field in (
                "primary_unfinished_cap_leaves", "fresh_unfinished_cap_leaves",
                "truth_cap_hits", "maintained_unfinished_cap_leaves",
            )):
        raise EvidenceError("evaluation slice is capped or inexact")
    lines = [merged_header, *rows, merged_footer]
    return b"\n".join(canonical_json(item, pretty=False).rstrip(b"\n")
                      for item in lines) + b"\n"


def _policy_by_semantic(row: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    policy = row.get("policy")
    _verify_full_policy(policy if isinstance(policy, dict) else {})
    legal = policy.get("legal") if isinstance(policy, dict) else None
    if not isinstance(legal, list) or not legal:
        raise EvidenceError("allocation has no full legal policy")
    result: dict[int, Mapping[str, Any]] = {}
    for item in legal:
        if not isinstance(item, dict) or not isinstance(item.get("semantic_move_pack"), int):
            raise EvidenceError("legal policy row is malformed")
        key = int(item["semantic_move_pack"])
        if key in result:
            raise EvidenceError("semantic complete move is not unique after collapse")
        result[key] = item
    return result


def _semantic_card(card: int) -> int:
    rank = card % 12
    return card - rank if rank < 3 else card


def _semantic_pack(card: int, discard: int, draw: int) -> int:
    return _semantic_card(card) + 60 * discard + 120 * draw


def _semantic_core_from_pack(packed: int) -> tuple[int, int]:
    if isinstance(packed, bool) or not isinstance(packed, int) or \
            not 0 <= packed < 720:
        raise EvidenceError("semantic move pack drift")
    complete = packed % 120
    return complete % 60, complete // 60


def _ratio_band_index(ratio: float) -> int:
    if not math.isfinite(ratio) or ratio < 1.0:
        raise EvidenceError("policy ratio is outside the frozen bands")
    if ratio < 1.25:
        return 0
    if ratio < 2.0:
        return 1
    if ratio < 4.0:
        return 2
    if ratio < 8.0:
        return 3
    if ratio < 32.0:
        return 4
    return 5


def _policy_probability_f32(item: Mapping[str, Any]) -> float:
    """Recover the native policy probability from its authoritative f32 bits."""

    decimal = _require_number(
        item.get("probability"), "full legal probability"
    )
    bits = item.get("probability_bits")
    if not isinstance(bits, str) or re.fullmatch(r"[0-9a-f]{8}", bits) is None:
        raise EvidenceError("full legal probability/float-bit drift")
    packed_decimal = struct.unpack("<I", struct.pack("<f", decimal))[0]
    if bits != f"{packed_decimal:08x}":
        raise EvidenceError("full legal probability/float-bit drift")
    return float(struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0])


def _verify_full_policy(policy: Mapping[str, Any]) -> None:
    legal = policy.get("legal") if isinstance(policy, dict) else None
    if not isinstance(legal, list) or not legal or policy.get("legal_count") != len(legal) or \
            policy.get("symmetries") != 20 or policy.get("exact_group_average") is not True:
        raise EvidenceError("full legal policy schema drift")
    action: dict[tuple[int, int], float] = defaultdict(float)
    joint: dict[tuple[int, int, int], float] = defaultdict(float)
    probabilities: list[float] = []
    for index, item in enumerate(legal):
        if not isinstance(item, dict) or item.get("index") != index or \
                any(isinstance(item.get(field), bool) or not isinstance(item.get(field), int)
                    for field in ("move_pack", "semantic_move_pack", "card", "discard", "draw")):
            raise EvidenceError("full legal policy identity drift")
        card, discard, draw = item["card"], item["discard"], item["draw"]
        semantic_card = _semantic_card(card)
        if not 0 <= card < 60 or discard not in (0, 1) or not 0 <= draw <= 5 or \
                item["move_pack"] != card + 60 * discard + 120 * draw or \
                item["semantic_move_pack"] != \
                semantic_card + 60 * discard + 120 * draw:
            raise EvidenceError("full legal policy move packing drift")
        probability = _policy_probability_f32(item)
        if probability < 0.0 or probability > 1.0:
            raise EvidenceError("full legal probability/float-bit drift")
        probabilities.append(probability)
        action[(semantic_card, discard)] += probability
        joint[(semantic_card, discard, draw)] += probability
    # Mirror rollout_policy_cost_support's frozen binary32 policy contract.
    # Native policy JSON prints enough digits to recover every float bit, but
    # the sequential sum of those floats is intentionally accepted within
    # 1e-5 rather than required to satisfy the much tighter summary-algebra
    # tolerance used for double-valued moment identities below.
    if abs(sum(probabilities) - 1.0) > 1.0e-5:
        raise EvidenceError("full legal normalization drift")
    literal = max(range(len(probabilities)), key=lambda index: probabilities[index])
    if policy.get("literal_argmax_index") != literal:
        raise EvidenceError("full legal literal argmax drift")
    for item in legal:
        card, discard, draw = item["card"], item["discard"], item["draw"]
        semantic_card = _semantic_card(card)
        raw_core = action[(semantic_card, discard)]
        if raw_core < 0.0:
            raise EvidenceError("full legal semantic core has negative mass")
        # A normalized policy is serialized as binary32 complete-move
        # probabilities.  Summing those authoritative floats in double can
        # exceed one by a representational fringe.  v11's only invalid row
        # was 1.0000000298023224.  Accept and canonicalize no more than eight
        # binary32 epsilons; a material producer error still fails closed.
        if raw_core > 1.0 + 2.0**-20:
            raise EvidenceError("full legal semantic core exceeds binary32 tolerance")
        core = 1.0 if raw_core > 1.0 else raw_core
        # Exact binary32 underflow may leave a legal semantic action with no
        # mass. Such an action cannot enter a positive-mass runtime mask; the
        # native producer records its irrelevant conditional probability as 0.
        conditional = (
            joint[(semantic_card, discard, draw)] / raw_core
            if raw_core > 0.0 else 0.0
        )
        if not math.isclose(_require_number(item.get("semantic_action_probability"), "policy P_A"), core,
                            rel_tol=2e-12, abs_tol=2e-12) or \
                not math.isclose(_require_number(item.get("conditional_draw_probability"), "policy P_D"), conditional,
                                 rel_tol=2e-12, abs_tol=2e-12):
            raise EvidenceError("full legal semantic probability drift")


def _verify_runtime_masks(policy: Mapping[str, Any], expected: Mapping[str, Any], *,
                          master_is_first: bool = True) -> list[list[int]]:
    masks = policy.get("runtime_masks") if isinstance(policy, dict) else None
    legal = policy.get("legal") if isinstance(policy, dict) else None
    if not isinstance(masks, list) or len(masks) != 2 or not isinstance(legal, list):
        raise EvidenceError("runtime mask evidence is absent")
    by_index = {item.get("index"): item for item in legal if isinstance(item, dict)}
    if len(by_index) != len(legal):
        raise EvidenceError("runtime mask legal-index identity drift")

    # Independently reproduce the semantic-core half of the native shortlist.
    # Draw-source choice is state dependent, so it is checked below by its
    # parent-core and positive-mass invariants; the three core representatives
    # themselves are fully determined by the complete policy vector.
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for item in legal:
        grouped[_semantic_core_from_pack(int(item["semantic_move_pack"]))].append(item)
    core_mass = {
        core: sum(_policy_probability_f32(item) for item in items)
        for core, items in grouped.items()
    }
    core_best = {
        core: min(
            items,
            key=lambda item: (-_policy_probability_f32(item),
                              int(item["semantic_move_pack"])),
        )
        for core, items in grouped.items()
    }
    ranked_cores = sorted(
        grouped,
        key=lambda core: (-core_mass[core],
                          -_policy_probability_f32(core_best[core]),
                          int(core_best[core]["semantic_move_pack"])),
    )
    literal_index = policy.get("literal_argmax_index")
    literal_item = by_index.get(literal_index)
    if not isinstance(literal_item, dict):
        raise EvidenceError("runtime mask literal baseline identity drift")
    baseline_core = _semantic_core_from_pack(
        int(literal_item["semantic_move_pack"])
    )
    wanted = min(3, len(ranked_cores))

    float_floor = {
        floor: struct.unpack("<f", struct.pack("<f", floor))[0]
        for floor in FLOORS
    }
    nominated_cores = [
        core for core in ranked_cores[:wanted]
        if core_mass[core] >= float_floor[FLOORS[0]]
    ]
    # Candidate zero is a mandatory comparator, not one of the three policy
    # nominations when its aggregate core ranks lower.  Retain it first and
    # then every true top-three core, still within the frozen max-five vector.
    master_cores = [baseline_core] + [
        core for core in nominated_cores if core != baseline_core
    ]
    if len(master_cores) > 4:
        raise EvidenceError("runtime semantic-core support exceeds frozen width")
    master_core_indices = [int(literal_index)] + [
        int(core_best[core]["index"]) for core in master_cores[1:]
    ]
    floor_cores = {
        FLOORS[0]: master_cores,
        FLOORS[1]: [
            core for offset, core in enumerate(master_cores)
            if offset == 0 or core_mass[core] >= float_floor[FLOORS[1]]
        ],
    }
    floor_core_indices = {
        FLOORS[0]: master_core_indices,
        FLOORS[1]: [
            index for offset, (core, index) in enumerate(
                zip(master_cores, master_core_indices)
            ) if offset == 0 or core_mass[core] >= float_floor[FLOORS[1]]
        ],
    }
    expected_hashes = (expected.get("mask_001_sha256"), expected.get("mask_002_sha256"))
    packed_masks: list[list[int]] = []
    for raw, floor, digest in zip(masks, FLOORS, expected_hashes):
        indices = raw.get("legal_indices") if isinstance(raw, dict) else None
        if not isinstance(indices, list) or raw.get("floor") != floor or \
                raw.get("floor_bits") != f"{struct.unpack('<I', struct.pack('<f', floor))[0]:08x}" or \
                raw.get("count") != len(indices) or not indices or len(indices) != len(set(indices)) or \
                isinstance(raw.get("core_candidates"), bool) or \
                not isinstance(raw.get("core_candidates"), int) or \
                isinstance(raw.get("draw_candidates"), bool) or \
                not isinstance(raw.get("draw_candidates"), int) or \
                raw.get("core_candidates") + raw.get("draw_candidates") != len(indices):
            raise EvidenceError("runtime mask schema drift")
        expected_core_indices = floor_core_indices[floor]
        if raw["core_candidates"] != len(expected_core_indices) or \
                indices[:raw["core_candidates"]] != expected_core_indices:
            raise EvidenceError("runtime mask is not candidate zero plus the frozen semantic top three")
        selected_cores = floor_cores[floor]
        seen_draw_cores: set[tuple[int, int]] = set()
        for index in indices[raw["core_candidates"]:]:
            item = by_index.get(index)
            if not isinstance(item, dict) or _policy_probability_f32(item) <= 0.0:
                raise EvidenceError("runtime draw alternative is absent or has zero policy mass")
            core = _semantic_core_from_pack(int(item["semantic_move_pack"]))
            if core not in selected_cores or core in seen_draw_cores:
                raise EvidenceError("runtime draw alternative lost its unique retained parent core")
            seen_draw_cores.add(core)
        complete_mass = sum(
            _policy_probability_f32(by_index[index]) for index in indices
        )
        represented_core_mass = sum(core_mass[core] for core in selected_cores)
        _close(_require_number(raw.get("complete_move_mass"), "runtime complete mass"),
               complete_mass, "runtime complete mass")
        _close(_require_number(raw.get("semantic_core_mass"), "runtime semantic core mass"),
               represented_core_mass, "runtime semantic core mass")
        packed: list[int] = []
        for index in indices:
            item = by_index.get(index)
            if not isinstance(index, int) or not isinstance(item, dict):
                raise EvidenceError("runtime mask legal identity drift")
            packed.append(int(item["semantic_move_pack"]))
        floor_bits = struct.unpack("<I", struct.pack("<f", floor))[0]
        encoded = bytes([len(packed), (floor_bits ^ (floor_bits >> 8) ^ (floor_bits >> 16) ^ (floor_bits >> 24)) & 0xff]) + \
            b"".join(value.to_bytes(2, "little") for value in packed)
        actual = hashlib.sha256(encoded).hexdigest()
        if raw.get("sha256") != actual or digest != actual:
            raise EvidenceError("runtime mask allocation hash drift")
        packed_masks.append(packed)
    union: list[int] = []
    for mask in packed_masks:
        union.extend(value for value in mask if value not in union)
    union_digest = hashlib.sha256(
        bytes([len(union)]) + b"".join(value.to_bytes(2, "little") for value in union)
    ).hexdigest()
    expected_master = expected_hashes[0] if master_is_first else union_digest
    if expected.get("master_sha256") != expected_master:
        raise EvidenceError("runtime mask master allocation hash drift")
    return packed_masks


def _truth_pair(row: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    truth = row.get("truth")
    metrics = truth.get("metrics") if isinstance(truth, dict) else None
    selected = metrics.get(metric) if isinstance(metrics, dict) else None
    pairs = selected.get("pairs") if isinstance(selected, dict) else None
    if not isinstance(pairs, list) or len(pairs) != 1:
        raise EvidenceError("TRAIN truth metric does not contain exactly one pair")
    pair = pairs[0]
    if not isinstance(pair, dict) or pair.get("a") != 0 or pair.get("b") != 1:
        raise EvidenceError("TRAIN truth pair orientation drift")
    return pair


def _finite_support_worlds(worlds: Any, support: Any, exact: Any) -> bool:
    """Validate the requested-800 finite hidden-support exception.

    Ordinary panels either do not enumerate the intrinsic support (support=0)
    or report a support strictly larger than the 800 sampled worlds.  A panel
    may use fewer than 800 worlds only when it proves that it exhausted the
    entire intrinsic support.  In particular, exact=false/support=800 is not a
    valid representation: the late enumerator would mark that census exact.
    """

    if isinstance(worlds, bool) or not isinstance(worlds, int) or \
            isinstance(support, bool) or not isinstance(support, int) or \
            not isinstance(exact, bool) or worlds < 2 or worlds > 800 or \
            support < 0:
        return False
    if exact:
        return support == worlds
    return worlds == 800 and (support == 0 or support > 800)


def _verify_train_pair_panel(panel: Mapping[str, Any], *, seed: str, role: int,
                             semantic_moves: Sequence[int], policy: Mapping[int, Mapping[str, Any]]) -> None:
    if not isinstance(panel, dict) or panel.get("seed") != seed or \
            panel.get("requested_worlds") != 800 or panel.get("panel_role") != role or \
            panel.get("common_worlds_across_pair") is not True or \
            panel.get("unfinished_cap_leaves") != 0 or \
            re.fullmatch(r"[0-9a-f]{16}", str(
                panel.get("hidden_world_fingerprint", ""))) is None:
        raise EvidenceError("TRAIN P/F panel protocol drift")
    worlds, support = panel.get("worlds"), panel.get("hidden_support")
    if not _finite_support_worlds(
            worlds, support, panel.get("exact_hidden_support")):
        raise EvidenceError("TRAIN P/F finite-support contract drift")
    actions = panel.get("actions")
    if not isinstance(actions, list) or len(actions) != 2:
        raise EvidenceError("TRAIN P/F action coverage drift")
    means: list[float] = []
    squares: list[float] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or action.get("position") != index or \
                action.get("legal_index") not in {item.get("index") for item in policy.values()} or \
                action.get("legal_index") != next((item.get("index") for item in policy.values()
                                                    if item.get("semantic_move_pack") == semantic_moves[index]), None):
            raise EvidenceError("TRAIN P/F action identity drift")
        mean, _, square = _sample_moments(action, worlds, "TRAIN P/F action", hashes=False)
        means.append(mean); squares.append(square)
    pair = panel.get("pair")
    if not isinstance(pair, dict) or set(pair) != {"delta_a_minus_b", "paired_se", "sum_products"}:
        raise EvidenceError("TRAIN P/F pair schema drift")
    oriented = {"a": 0, "b": 1, **pair}
    _pair_moment(oriented, means, squares, worlds, "TRAIN P/F pair")


def _verify_train_truth(row: Mapping[str, Any], metric_name: str,
                        expected_support: Sequence[int]) -> None:
    truth = row.get("truth")
    support = row.get("truth_support_legal_indices")
    match, state = row.get("source_match_index"), row.get("source_state_index")
    metrics = truth.get("metrics") if isinstance(truth, dict) else None
    if not isinstance(match, int) or isinstance(match, bool) or not isinstance(state, int) or \
            isinstance(state, bool) or not isinstance(truth, dict) or \
            truth.get("controller") != "exact_policy20_full_remaining_match" or \
            truth.get("information_view_each_node") is not True or truth.get("temperature") != 0 or \
            truth.get("epsilon") != 0 or truth.get("requested_worlds") != TRUTH_WORLDS["TRAIN"] or \
            truth.get("worlds") != TRUTH_WORLDS["TRAIN"] or \
            truth.get("seed") != _domain_seed(TRUTH_SEEDS["TRAIN"], match, state, 0x5452555448574C44) or \
            truth.get("union_untruncated") is not True or truth.get("cap_hits") != 0 or \
            not isinstance(support, list) or support != list(expected_support) or \
            len(support) != 2 or len(set(support)) != 2 or \
            truth.get("union_count") != 2 or not isinstance(metrics, dict):
        raise EvidenceError("TRAIN truth protocol drift")
    if any(HEX64.fullmatch(str(truth.get(field, ""))) is None
           for field in ("hidden_worlds_sha256", "future_deals_sha256")):
        raise EvidenceError("TRAIN truth digest drift")
    for name in ("current_round_margin", "full_match_margin", "full_match_score", "full_match_hybrid"):
        metric = metrics.get(name)
        actions = metric.get("actions") if isinstance(metric, dict) else None
        pairs = metric.get("pairs") if isinstance(metric, dict) else None
        if not isinstance(actions, list) or len(actions) != 2 or not isinstance(pairs, list) or len(pairs) != 1:
            raise EvidenceError("TRAIN truth metric coverage drift")
        means: list[float] = []
        squares: list[float] = []
        for position, action in enumerate(actions):
            if not isinstance(action, dict) or action.get("position") != position:
                raise EvidenceError("TRAIN truth action identity drift")
            mean, _, square = _sample_moments(action, TRUTH_WORLDS["TRAIN"],
                                               "TRAIN truth action", hashes=True)
            means.append(mean); squares.append(square)
        if not isinstance(pairs[0], dict):
            raise EvidenceError("TRAIN truth pair drift")
        _pair_moment(pairs[0], means, squares, TRUTH_WORLDS["TRAIN"], "TRAIN truth pair")
    _truth_pair(row, metric_name)


TRAIN_ALLOCATION_COLUMNS = (
    "allocation_id", "source_match_index", "source_state_index",
    "source_match_id", "state_id", "pair_id", "cell", "round",
    "ply_bin", "ratio_bin", "pair_type", "pair_move_a", "pair_move_b",
    "orbit_sha256", "state_sha256", "pair_sha256",
    "allocation_priority_sha256", "mask_001_sha256", "mask_002_sha256",
    "master_sha256", "state_hex",
)


def train_allocation_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert the immutable TRAIN TSV into calibration's exact JSON schema.

    The emitted manifest deliberately contains no campaign-only convenience
    fields: ``policy_cost_calibration._campaign_allocation_binding`` rejects
    additions so this conversion is also a schema boundary.
    """

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read TRAIN allocation: {exc}") from exc
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise EvidenceError("TRAIN allocation must be canonical LF text")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise EvidenceError("TRAIN allocation is not ASCII") from exc
    if not lines or lines[0] != "LCPOLICYCOST-TRAIN-ALLOCATION-V5":
        raise EvidenceError("TRAIN allocation schema drift")
    cursor = 1

    def scalar(key: str) -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise EvidenceError(f"TRAIN allocation lacks {key}")
        fields = lines[cursor].split("\t")
        cursor += 1
        if len(fields) != 2 or fields[0] != key or not fields[1]:
            raise EvidenceError(f"TRAIN allocation expected {key}")
        return fields[1]

    if scalar("split") != "TRAIN" or scalar("purpose") != "campaign":
        raise EvidenceError("TRAIN allocation split/purpose drift")
    discovery_sha = scalar("discovery_sha256")
    reservoir_sha = scalar("reservoir_sha256")
    source_net_sha = scalar("source_net_sha256")
    source_exclusion_sha = scalar("source_exclusion_sha256")
    pair_commitment_sha = scalar("eligible_pair_commitment_sha256")
    rule_sha = scalar("allocation_rule_sha256")
    if any(HEX64.fullmatch(value) is None for value in (
        discovery_sha, reservoir_sha, source_net_sha, source_exclusion_sha,
        pair_commitment_sha, rule_sha,
    )) or rule_sha != TRAIN_ALLOCATION_RULE_SHA256:
        raise EvidenceError("TRAIN allocation contains a noncanonical digest")
    if int(scalar("quota_per_cell")) != 16:
        raise EvidenceError("TRAIN allocation quota drift")
    eligible_units = int(scalar("eligible_units"))
    retained_units = int(scalar("retained_reservoir_units"))
    if eligible_units < retained_units or retained_units < TRAIN_RECORDS:
        raise EvidenceError("TRAIN allocation census/reservoir drift")
    if int(scalar("probe_orbit_rejections")) < 0 or \
            int(scalar("pooled_ge64_observed")) < 0:
        raise EvidenceError("TRAIN allocation audit count drift")
    if int(scalar("records")) != TRAIN_RECORDS:
        raise EvidenceError("TRAIN allocation record count drift")
    if cursor >= len(lines) or lines[cursor].split("\t") != [
            "columns", *TRAIN_ALLOCATION_COLUMNS]:
        raise EvidenceError("TRAIN allocation columns drift")
    cursor += 1

    selected_units: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    previous_by_cell: dict[
        tuple[int, int, int, str], tuple[Any, ...]
    ] = {}
    cells: Counter[tuple[int, int, int, str]] = Counter()
    sources: set[str] = set()
    for allocation_id in range(TRAIN_RECORDS):
        if cursor >= len(lines):
            raise EvidenceError("truncated TRAIN allocation records")
        fields = lines[cursor].split("\t")
        cursor += 1
        if len(fields) != len(TRAIN_ALLOCATION_COLUMNS):
            raise EvidenceError("TRAIN allocation row width drift")
        row = dict(zip(TRAIN_ALLOCATION_COLUMNS, fields, strict=True))
        integer_fields = (
            "allocation_id", "source_match_index", "source_state_index",
            "round", "ply_bin", "ratio_bin", "pair_type", "pair_move_a",
            "pair_move_b",
        )
        try:
            numeric = {key: int(row[key]) for key in integer_fields}
        except ValueError as exc:
            raise EvidenceError("TRAIN allocation integer field drift") from exc
        if numeric["allocation_id"] != allocation_id or \
                not 0 <= numeric["source_match_index"] < 65536 or \
                not 0 <= numeric["source_state_index"] < 900 or \
                numeric["round"] not in (0, 1, 2) or \
                not 0 <= numeric["ply_bin"] < len(PLY_STRATA) or \
                not 0 <= numeric["ratio_bin"] < len(RATIO_BANDS) or \
                numeric["pair_type"] not in (0, 1) or \
                not 0 <= numeric["pair_move_a"] <= 0xffff or \
                not 0 <= numeric["pair_move_b"] <= 0xffff:
            raise EvidenceError("TRAIN allocation row identity drift")
        source = f"TRAIN-{numeric['source_match_index']:012d}"
        state_id = f"{source}:s{numeric['source_state_index']:03d}"
        pair_id = f"{numeric['pair_move_a']:05d}-{numeric['pair_move_b']:05d}"
        pair_type = PAIR_TYPES[numeric["pair_type"]]
        cell = (numeric["round"], numeric["ply_bin"], numeric["ratio_bin"], pair_type)
        scheduled = _train_scheduled_cell(allocation_id)
        if row["source_match_id"] != source or row["state_id"] != state_id or \
                row["pair_id"] != pair_id or \
                row["cell"] != f"r{cell[0]}.p{cell[1]}.g{cell[2]}.t{numeric['pair_type']}" or \
                (numeric["round"], numeric["ply_bin"], numeric["ratio_bin"],
                 numeric["pair_type"]) != scheduled:
            raise EvidenceError("TRAIN allocation source/state/pair binding drift")
        for field in (
            "orbit_sha256", "state_sha256", "pair_sha256",
            "allocation_priority_sha256", "mask_001_sha256",
            "mask_002_sha256", "master_sha256",
        ):
            if HEX64.fullmatch(row[field]) is None:
                raise EvidenceError(f"TRAIN allocation invalid {field}")
        try:
            state_bytes = bytes.fromhex(row["state_hex"])
        except ValueError as exc:
            raise EvidenceError("TRAIN allocation state encoding drift") from exc
        if len(state_bytes) != 174 or state_bytes[0] != 1 or \
                state_bytes[165] not in (0, 1) or state_bytes[166] != 0 or \
                hashlib.sha256(state_bytes).hexdigest() != row["state_sha256"] or \
                hashlib.sha256(
                    bytes.fromhex(row["state_sha256"]) +
                    numeric["pair_move_a"].to_bytes(2, "little") +
                    numeric["pair_move_b"].to_bytes(2, "little")
                ).hexdigest() != row["pair_sha256"]:
            raise EvidenceError("TRAIN allocation state/pair hash drift")
        if source in sources:
            raise EvidenceError("TRAIN allocation reuses a source match")
        sources.add(source)
        order = (row["allocation_priority_sha256"], row["state_sha256"],
                 source, state_id, pair_id)
        previous = previous_by_cell.get(cell)
        if previous is not None and order <= previous:
            raise EvidenceError("TRAIN within-cell allocation order drift")
        previous_by_cell[cell] = order
        cells[cell] += 1
        selected_units.append({
            "source_match_id": source,
            "state_id": state_id,
            "pair_id": pair_id,
            "state_sha256": row["state_sha256"],
            "pair_sha256": row["pair_sha256"],
            "allocation_priority_sha256": row["allocation_priority_sha256"],
            "round": cell[0], "ply_bin": cell[1], "ratio_bin": cell[2],
            "pair_type": pair_type,
        })
        bindings.append({
            "allocation_id": allocation_id,
            "source_match_index": numeric["source_match_index"],
            "source_state_index": numeric["source_state_index"],
            "orbit_sha256": row["orbit_sha256"],
            "mask_001_sha256": row["mask_001_sha256"],
            "mask_002_sha256": row["mask_002_sha256"],
            "master_sha256": row["master_sha256"],
            "state_hex": row["state_hex"],
            **selected_units[-1],
        })
    if cursor != len(lines):
        raise EvidenceError("trailing TRAIN allocation content")
    expected_cells = {
        (round_index, ply_bin, ratio_bin, pair_type)
        for round_index in range(3) for ply_bin in range(24)
        for ratio_bin in range(6) for pair_type in PAIR_TYPES
    }
    if set(cells) != expected_cells or any(value != 16 for value in cells.values()):
        raise EvidenceError("TRAIN allocation cell quotas differ")
    # The native allocation is intentionally serialized in allocation-id
    # schedule order so every evaluator slice can consume a contiguous range.
    # Calibration has an independent, stricter manifest boundary: selected
    # units must be in canonical cell/priority order.  Sort only the manifest
    # projection here; keep ``bindings`` in allocation-id order.
    selected_units.sort(key=lambda row: (
        row["round"], row["ply_bin"], row["ratio_bin"], row["pair_type"],
        row["allocation_priority_sha256"], row["state_sha256"],
        row["source_match_id"], row["state_id"], row["pair_id"],
    ))
    payload = {
        "schema": "lc-policy-cost-train-allocation-v1",
        "source_reservoir_sha256": reservoir_sha,
        "eligible_pair_commitment_sha256": pair_commitment_sha,
        "allocation_rule_sha256": rule_sha,
        "ply_bins": [list(item) for item in PLY_STRATA],
        "ratio_bins": list(RATIO_BANDS),
        "pair_types": list(PAIR_TYPES),
        "cell_quota": 16,
        "selected_units": selected_units,
    }
    manifest = dict(payload)
    manifest["canonical_payload_sha256"] = _canonical_payload_digest(payload)
    # Keep the raw discovery digest strictly out of calibration's exact
    # manifest schema.  It is used only while binding the evaluator below.
    for binding_row in bindings:
        binding_row["discovery_sha256"] = discovery_sha
        binding_row["reservoir_sha256"] = reservoir_sha
    return manifest, bindings


def train_input(evaluation: Path, execution: Path, allocation: Path) -> bytes:
    bound = strict_json(execution)
    if bound.get("schema") != EXECUTION_SCHEMA:
        raise EvidenceError("invalid execution binding")
    truth_metric = bound.get("subject", {}).get("train_truth_metric")
    if truth_metric not in {"current_round_margin", "full_match_hybrid"}:
        raise EvidenceError("execution has invalid TRAIN truth metric")
    header, rows, _ = _evaluation(evaluation, "TRAIN")
    subject = bound.get("subject")
    if not isinstance(subject, dict) or \
            header.get("actor_spec") != subject.get("neutral_counterfactual_actor") or \
            header.get("maintained_actor_spec") is not None or \
            header.get("evaluation_support") != "one_pair_per_state" or \
            header.get("seed_domains_pairwise_disjoint") is not True or \
            header.get("arbitrary_top_five_truncation") is not False or \
            any(re.fullmatch(r"[0-9a-f]{16}", str(header.get(field, ""))) is None
                for field in ("root_net_fingerprint", "continuation_net_fingerprint",
                              "candidate_match_value_fingerprint")):
        raise EvidenceError("TRAIN execution/evaluator header binding drift")
    _, allocation_bindings = train_allocation_manifest(allocation)
    if header.get("manifest_sha256") != sha256(allocation) or \
            header.get("discovery_sha256") != allocation_bindings[0]["discovery_sha256"] or \
            header.get("reservoir_sha256") != allocation_bindings[0]["reservoir_sha256"]:
        raise EvidenceError("TRAIN evaluation is not bound to its allocation")
    expected_by_id = {
        int(item["allocation_id"]): item for item in allocation_bindings
    }
    cells: Counter[tuple[int, int, int, str]] = Counter()
    sources: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        source = row.get("source_match_id")
        state_sha = row.get("state_sha256")
        pair_packs = row.get("pair_semantic_moves")
        pair_type_code = row.get("pair_type")
        ratio_bin = row.get("ratio_bin")
        expected = expected_by_id.get(row.get("allocation_id"))
        if expected is None or any(row.get(key) != expected.get(key) for key in (
                "source_match_id", "source_match_index", "source_state_index",
                "state_sha256", "orbit_sha256", "round", "ratio_bin",
            )) or row.get("ply_stratum") != expected.get("ply_bin") or \
                row.get("cell") != (
                    f"r{expected.get('round')}.p{expected.get('ply_bin')}."
                    f"g{expected.get('ratio_bin')}."
                    f"t{PAIR_TYPES.index(expected.get('pair_type'))}"
                ) or \
                row.get("pair_type") != PAIR_TYPES.index(expected["pair_type"]) or \
                row.get("pair_semantic_moves") != [
                    int(expected["pair_id"][:5]), int(expected["pair_id"][6:])
                ]:
            raise EvidenceError("TRAIN evaluation is not bound to allocation row")
        if "maintained_baseline" not in row or \
                row.get("maintained_baseline") is not None or \
                "production_decisions" in row or "config_decisions" in row:
            raise EvidenceError("TRAIN row contains holdout-only actor decisions")
        if not isinstance(source, str) or not source or source in sources or \
                not isinstance(state_sha, str) or HEX64.fullmatch(state_sha) is None or \
                not isinstance(pair_packs, list) or len(pair_packs) != 2 or \
                any(isinstance(item, bool) or not isinstance(item, int)
                    for item in pair_packs) or \
                isinstance(pair_type_code, bool) or pair_type_code not in (0, 1) or \
                isinstance(ratio_bin, bool) or not isinstance(ratio_bin, int) or \
                not 0 <= ratio_bin < len(RATIO_BANDS):
            raise EvidenceError("TRAIN source/state/pair identity drift")
        sources.add(source)
        pair_type = PAIR_TYPES[int(pair_type_code)]
        _verify_full_policy(row.get("policy") if isinstance(row.get("policy"), dict) else {})
        policy = _policy_by_semantic(row)
        packed_masks = _verify_runtime_masks(
            row.get("policy") if isinstance(row.get("policy"), dict) else {}, expected,
            master_is_first=False,
        )
        master_union = list(packed_masks[0])
        master_union.extend(
            item for item in packed_masks[1] if item not in master_union
        )
        if any(item not in master_union for item in pair_packs):
            raise EvidenceError("TRAIN pair left the exact master union")
        try:
            left = policy[int(pair_packs[0])]
            right = policy[int(pair_packs[1])]
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceError("TRAIN pair left the full legal policy") from exc
        left_core = _require_number(left.get("semantic_action_probability"), "left P_A")
        right_core = _require_number(right.get("semantic_action_probability"), "right P_A")
        left_draw = _require_number(left.get("conditional_draw_probability"), "left P_D")
        right_draw = _require_number(right.get("conditional_draw_probability"), "right P_D")
        if min(left_core, right_core, left_draw, right_draw) <= 0.0 or \
                left_core > 1.0 or right_core > 1.0 or \
                left_draw > 1.0 or right_draw > 1.0:
            raise EvidenceError("TRAIN pair has invalid probability")
        same_core = _semantic_core_from_pack(int(pair_packs[0])) == \
            _semantic_core_from_pack(int(pair_packs[1]))
        if same_core != (pair_type == "same_core_draw"):
            raise EvidenceError("TRAIN pair semantic type drift")
        high = left_draw if same_core else left_core
        low = right_draw if same_core else right_core
        if high < low or (high == low and pair_packs[0] > pair_packs[1]) or \
                (same_core and left_core != right_core) or \
                _ratio_band_index(high / low) != int(ratio_bin):
            raise EvidenceError("TRAIN pair is not canonically prior-oriented")
        _verify_train_truth(
            row, str(truth_metric), [int(left["index"]), int(right["index"])]
        )
        search = row.get("search")
        if not isinstance(search, dict) or set(search) != {"primary", "fresh"}:
            raise EvidenceError("TRAIN search evidence absent")
        match_index = row.get("source_match_index")
        state_index = row.get("source_state_index")
        if not isinstance(match_index, int) or isinstance(match_index, bool) or \
                not isinstance(state_index, int) or isinstance(state_index, bool):
            raise EvidenceError("TRAIN source seed identity drift")
        _verify_train_pair_panel(
            search.get("primary") if isinstance(search.get("primary"), dict) else {},
            seed=_domain_seed(PRIMARY_SEEDS["TRAIN"], match_index, state_index,
                              0x5052494D41525932), role=0,
            semantic_moves=pair_packs, policy=policy,
        )
        _verify_train_pair_panel(
            search.get("fresh") if isinstance(search.get("fresh"), dict) else {},
            seed=_domain_seed(FRESH_SEEDS["TRAIN"], match_index, state_index,
                              0x46524553485F5032), role=1,
            semantic_moves=pair_packs, policy=policy,
        )
        primary_fingerprint = search["primary"].get("hidden_world_fingerprint")
        fresh_fingerprint = search["fresh"].get("hidden_world_fingerprint")
        equal_complete_census = (
            search["primary"].get("exact_hidden_support") is True and
            search["fresh"].get("exact_hidden_support") is True and
            search["primary"].get("worlds") == search["fresh"].get("worlds") and
            search["primary"].get("hidden_support") == search["primary"].get("worlds") and
            search["fresh"].get("hidden_support") == search["fresh"].get("worlds") and
            search["primary"].get("worlds") < 800
        )
        if primary_fingerprint == fresh_fingerprint and not equal_complete_census:
            raise EvidenceError("TRAIN P/F panels reuse hidden worlds")
        round_index = row.get("round")
        ply = row.get("nply")
        ply_bin = row.get("ply_stratum")
        ratio_bin = row.get("ratio_bin")
        if round_index not in (0, 1, 2) or \
                not isinstance(ply, int) or isinstance(ply, bool) or \
                not isinstance(ply_bin, int) or not 0 <= ply_bin < 24 or \
                not isinstance(ratio_bin, int) or not 0 <= ratio_bin < 6 or \
                not PLY_STRATA[ply_bin][0] <= ply < PLY_STRATA[ply_bin][1]:
            raise EvidenceError("TRAIN stratum identity drift")
        cell = (int(round_index), ply_bin, ratio_bin, pair_type)
        cells[cell] += 1
        truth_pair = _truth_pair(row, str(truth_metric))
        truth_delta = _require_number(
            truth_pair.get("delta_a_minus_b"), "truth delta"
        )
        truth_se = _require_number(truth_pair.get("paired_se"), "truth SE")
        for panel_name in ("primary", "fresh"):
            panel = search.get(panel_name)
            pair = panel.get("pair") if isinstance(panel, dict) else None
            if not isinstance(pair, dict) or \
                    panel.get("unfinished_cap_leaves") != 0:
                raise EvidenceError(f"TRAIN {panel_name} pair is invalid")
            output.append({
                "source_match_id": source,
                "state_id": expected["state_id"],
                "pair_id": expected["pair_id"],
                "round": int(round_index),
                "ply": ply,
                "pair_type": pair_type,
                "search_delta": _require_number(
                    pair.get("delta_a_minus_b"), f"{panel_name} delta"
                ),
                "truth_delta": truth_delta,
                "log_core_ratio": math.log(left_core / right_core),
                "log_draw_ratio": math.log(left_draw / right_draw),
                "search_se": _require_number(
                    pair.get("paired_se"), f"{panel_name} SE"
                ),
                "truth_se": truth_se,
                "search_panel_id": panel_name,
                "truth_panel_id": f"truth-{TRUTH_SEEDS['TRAIN']}",
                "orientation": "canonical-left-minus-right",
                "state_weight": 1.0,
            })
    expected_cells = {
        (round_index, ply_bin, ratio_bin, pair_type)
        for round_index in range(3)
        for ply_bin in range(24)
        for ratio_bin in range(6)
        for pair_type in PAIR_TYPES
    }
    if set(cells) != expected_cells or any(count != 16 for count in cells.values()):
        raise EvidenceError("TRAIN is not exactly 16 states in every frozen cell")
    lines = [canonical_json(item, pretty=False).rstrip(b"\n") for item in output]
    return b"\n".join(lines) + b"\n"


def train_evidence_binding(evaluation: Path, execution: Path, allocation: Path,
                           train_payload: bytes) -> dict[str, Any]:
    """Seal the raw TRAIN conversion into the calibration result chain."""

    header, _, _ = _evaluation(evaluation, "TRAIN")
    if strict_json(execution).get("schema") != EXECUTION_SCHEMA:
        raise EvidenceError("TRAIN evidence execution schema drift")
    if not train_payload.endswith(b"\n") or not train_payload:
        raise EvidenceError("TRAIN evidence input is not canonical JSONL")
    return {
        "schema": "lc-policy-cost-v19-train-evidence-binding-v1",
        "stage": "TRAIN",
        "raw_verified": True,
        "execution_sha256": sha256(execution),
        "evaluation_sha256": sha256(evaluation),
        "evaluation_header_sha256": _canonical_payload_digest(header),
        "allocation_sha256": sha256(allocation),
        "train_input_sha256": hashlib.sha256(train_payload).hexdigest(),
    }


def verify_reservoir_freeze(path: Path, root: Path, split: str) -> None:
    value = strict_json(path)
    if path.read_bytes() != canonical_json(value) or split not in {"TRAIN", "SELECT", "TEST"} or \
            value.get("schema") != "lc-policy-cost-v19-reservoir-freeze-v1" or \
            value.get("status") != "all_discovery_reservoirs_and_allocations_frozen_before_first_efficacy":
        raise EvidenceError("reservoir-freeze manifest identity drift")
    expected_by_split = {
        "TRAIN": ("train-discovery.jsonl", "train-reservoir.tsv",
                  "train-allocation.tsv", "train-reservoir-proof.json"),
        "SELECT": ("select-discovery.jsonl", "select-reservoir.tsv",
                   "select-allocation.tsv", "select-reservoir-proof.json"),
        "TEST": ("test-discovery.jsonl", "test-reservoir.tsv",
                 "test-allocation.tsv", "test-reservoir-proof.json"),
    }
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 12:
        raise EvidenceError("reservoir-freeze full coverage drift")
    seen: set[tuple[str, str]] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"split", "path", "sha256", "size"} or HEX64.fullmatch(str(item.get("sha256", ""))) is None:
            raise EvidenceError("reservoir-freeze record schema drift")
        key = (item.get("split"), item.get("path"))
        if key in seen or key[0] not in expected_by_split or key[1] not in expected_by_split[key[0]] or \
                isinstance(item.get("size"), bool) or not isinstance(item.get("size"), int) or item["size"] <= 0:
            raise EvidenceError("reservoir-freeze record coverage drift")
        seen.add(key)
    if seen != {(stage, name) for stage, names in expected_by_split.items() for name in names}:
        raise EvidenceError("reservoir-freeze full coverage drift")
    records = [item for item in files if item["split"] == split]
    for item in records:
        candidate = root / str(item["path"])
        if not candidate.is_file() or candidate.is_symlink() or candidate.stat().st_size != item.get("size") or \
                sha256(candidate) != item["sha256"]:
            raise EvidenceError("reservoir-freeze local hash drift")


def _canonical_stratum(round_index: int, ply: int, frontier: int,
                       slot: int) -> str:
    return f"r{round_index}:p{ply:02d}:f{frontier}:g{min(slot, 2)}"


def _canonical_payload_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value, pretty=False)).hexdigest()


def vector_allocation_manifest(path: Path, split: str) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, float]
]:
    """Reopen the complete outcome-blind vector allocation and seal its census."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read vector allocation: {exc}") from exc
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise EvidenceError("vector allocation must be canonical LF text")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise EvidenceError("vector allocation is not ASCII") from exc
    if not lines or lines[0] != "LCPOLICYCOST-VECTOR-ALLOCATION-V2":
        raise EvidenceError("vector allocation schema drift")
    cursor = 1

    def scalar(key: str) -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise EvidenceError(f"vector allocation lacks {key}")
        fields = lines[cursor].split("\t")
        cursor += 1
        if len(fields) != 2 or fields[0] != key or not fields[1]:
            raise EvidenceError(f"vector allocation expected {key}")
        return fields[1]

    found_split = scalar("split")
    if found_split != split or split not in {"SELECT", "TEST"}:
        raise EvidenceError("vector allocation split drift")
    if scalar("purpose") != "campaign":
        raise EvidenceError("smoke allocation cannot enter holdout inference")
    discovery_sha = scalar("discovery_sha256")
    reservoir_sha = scalar("reservoir_sha256")
    source_net_sha = scalar("source_net_sha256")
    source_exclusion_sha = scalar("source_exclusion_sha256")
    chain_sha = scalar("eligible_state_commitment_sha256")
    rule_sha = scalar("allocation_rule_sha256")
    if any(HEX64.fullmatch(item) is None for item in (
        discovery_sha, reservoir_sha, source_net_sha, source_exclusion_sha,
        chain_sha, rule_sha
    )) or rule_sha != VECTOR_ALLOCATION_RULE_SHA256:
        raise EvidenceError("vector allocation contains a noncanonical digest")
    if int(scalar("quota_per_base_cell")) != 64 or \
            int(scalar("source_minimum_per_poststratum")) != 8:
        raise EvidenceError("vector allocation quota/source minimum drift")
    total = int(scalar("total_census"))
    retained = int(scalar("retained_reservoir_vectors"))
    if total <= 0 or retained < HOLDOUT_RECORDS:
        raise EvidenceError("vector allocation has an invalid census/reservoir")
    if int(scalar("poststratum_cells")) != 432:
        raise EvidenceError("vector allocation must enumerate 432 hJ groups")
    histogram_values = scalar("aggregate_master_width_histogram").split(",")
    if len(histogram_values) != 5:
        raise EvidenceError("aggregate master-width histogram drift")
    histogram = {str(index + 1): int(value)
                 for index, value in enumerate(histogram_values)}
    if any(value < 0 for value in histogram.values()) or \
            sum(histogram.values()) != total:
        raise EvidenceError("aggregate master-width histogram is inconsistent")
    probe_rejections = int(scalar("probe_orbit_rejections"))
    pooled = int(scalar("pooled_ge64_observed"))
    if probe_rejections < 0 or pooled < 0:
        raise EvidenceError("vector allocation firewall/tail count is invalid")
    if int(scalar("records")) != HOLDOUT_RECORDS:
        raise EvidenceError("vector allocation record count drift")
    cells: list[dict[str, Any]] = []
    raw_cell_names: list[str] = []
    cell_values: dict[str, tuple[int, int, dict[str, int]]] = {}
    for _ in range(432):
        if cursor >= len(lines):
            raise EvidenceError("truncated vector post-stratum census")
        fields = lines[cursor].split("\t")
        cursor += 1
        if len(fields) != 7 or fields[0] != "poststratum":
            raise EvidenceError("malformed vector post-stratum census")
        match = CELL_VECTOR.fullmatch(fields[1])
        if match is None:
            raise EvidenceError("noncanonical vector post-stratum name")
        rd, pb, frontier, group = map(int, match.groups())
        if not 0 <= pb < 24:
            raise EvidenceError("vector post-stratum ply is outside design")
        census, quota, numerator, denominator = map(int, fields[2:6])
        widths_raw = fields[6].split(",")
        if len(widths_raw) != 5:
            raise EvidenceError("per-cell width histogram drift")
        widths = {str(index + 1): int(value)
                  for index, value in enumerate(widths_raw)}
        if census < 0 or quota < 0 or numerator != census or \
                denominator != total or sum(widths.values()) != census or \
                any(widths[str(width)] != 0 for width in range(1, group + 1)) or \
                quota != (22, 21, 21)[group] or census < quota:
            raise EvidenceError("per-cell census/quota/width binding drift")
        canonical = _canonical_stratum(rd, pb, frontier, group)
        if canonical in cell_values:
            raise EvidenceError("duplicate vector post-stratum")
        raw_cell_names.append(fields[1])
        cell_values[canonical] = (census, quota, widths)
        cells.append({
            "round": rd,
            "ply_stratum": pb,
            "frontier_present": bool(frontier),
            "allocation_slot": group,
            "post_stratum": canonical,
            "census_count": census,
            "allocation_quota": quota,
            "master_width_histogram": widths,
        })
    expected_cells = {
        _canonical_stratum(rd, pb, frontier, slot)
        for rd in range(3) for pb in range(24)
        for frontier in range(2) for slot in range(3)
    }
    if set(cell_values) != expected_cells or \
            sum(value[0] for value in cell_values.values()) != total or \
            sum(value[1] for value in cell_values.values()) != HOLDOUT_RECORDS:
        raise EvidenceError("vector post-stratum census is not complete")
    aggregate = {
        str(width): sum(values[2][str(width)] for values in cell_values.values())
        for width in range(1, 6)
    }
    if aggregate != histogram:
        raise EvidenceError("vector aggregate width histogram drift")
    for rd in range(3):
        for pb in range(24):
            for frontier in range(2):
                base = [_canonical_stratum(rd, pb, frontier, group)
                        for group in range(3)]
                for group, name in enumerate(base):
                    if cell_values[name][1] != (22, 21, 21)[group]:
                        raise EvidenceError("vector group quota is not fixed")
                if sum(cell_values[name][1] for name in base) != 64:
                    raise EvidenceError("vector base quota differs from 64")
    if cursor >= len(lines):
        raise EvidenceError("vector allocation lacks columns")
    columns = lines[cursor].split("\t")
    cursor += 1
    if columns != ["columns", *VECTOR_ALLOCATION_COLUMNS]:
        raise EvidenceError("vector allocation columns drift")
    allocations: list[dict[str, Any]] = []
    # The discovery manifest is consumed by policy_cost_selection_v19.py, whose
    # selected-unit contract is deliberately just these ten identities.  Keep
    # audit-only allocation metadata in ``allocations`` below: it binds raw
    # evaluator rows without widening the cross-module manifest schema.
    selected_units: list[dict[str, Any]] = []
    previous_by_cell: dict[
        str, tuple[str, str, str, str]
    ] = {}
    for allocation_id in range(HOLDOUT_RECORDS):
        if cursor >= len(lines):
            raise EvidenceError("truncated vector allocation records")
        fields = lines[cursor].split("\t")
        cursor += 1
        if len(fields) != len(VECTOR_ALLOCATION_COLUMNS):
            raise EvidenceError("vector allocation row width drift")
        raw_row = dict(zip(VECTOR_ALLOCATION_COLUMNS, fields, strict=True))
        integer_fields = (
            "allocation_id", "source_match_index", "source_state_index",
            "round", "ply_stratum", "frontier_present", "allocation_slot",
            "master_width", "census_count", "allocation_quota",
            "weight_numerator", "weight_denominator",
        )
        row: dict[str, Any] = dict(raw_row)
        try:
            for field in integer_fields:
                row[field] = int(raw_row[field])
        except ValueError as exc:
            raise EvidenceError("vector allocation integer field drift") from exc
        if row["allocation_id"] != allocation_id or \
                row["discovery_sha256"] != discovery_sha or \
                not 0 <= row["source_match_index"] < 32768 or \
                not 0 <= row["source_state_index"] < 900:
            raise EvidenceError("vector allocation id/discovery binding drift")
        for field in (
            "orbit_sha256", "state_sha256", "allocation_priority_sha256",
            "mask_001_sha256", "mask_002_sha256", "master_sha256",
        ):
            if HEX64.fullmatch(str(row[field])) is None:
                raise EvidenceError(f"vector allocation invalid {field}")
        try:
            state_bytes = bytes.fromhex(str(row["state_hex"]))
        except ValueError as exc:
            raise EvidenceError("vector allocation state encoding drift") from exc
        if len(state_bytes) != 174 or state_bytes[0] != 1 or \
                state_bytes[165] not in (0, 1) or state_bytes[166] != 0 or \
                hashlib.sha256(state_bytes).hexdigest() != row["state_sha256"]:
            raise EvidenceError("vector allocation state bytes/hash drift")
        rd = int(row["round"])
        pb = int(row["ply_stratum"])
        frontier = int(row["frontier_present"])
        slot = int(row["allocation_slot"])
        canonical = _canonical_stratum(rd, pb, frontier, slot)
        if (rd, pb, frontier) != _vector_scheduled_base(allocation_id):
            raise EvidenceError("vector allocation scheduling order drift")
        if row["post_stratum"] != canonical or \
                not 1 <= int(row["master_width"]) <= 5 or \
                slot >= int(row["master_width"]):
            raise EvidenceError("vector allocation stratum/width drift")
        census, quota, _ = cell_values[canonical]
        if row["census_count"] != census or row["allocation_quota"] != quota or \
                row["weight_numerator"] != census or \
                row["weight_denominator"] != quota * total:
            raise EvidenceError("vector allocation row weight drift")
        normalized = {
            "allocation_id": allocation_id,
            "source_match": row["source_match_id"],
            "unit": row["unit"],
            "state_sha256": row["state_sha256"],
            "allocation_priority_sha256": row["allocation_priority_sha256"],
            "round": rd,
            "ply_stratum": pb,
            "frontier_present": bool(frontier),
            "allocation_slot": slot,
            "master_width": int(row["master_width"]),
            "post_stratum": canonical,
            "discovery_census_sha256": "",
            "weight": census / (quota * total),
        }
        order = (
            str(row["allocation_priority_sha256"]),
            str(row["state_sha256"]), str(row["source_match_id"]),
            str(row["unit"]),
        )
        previous = previous_by_cell.get(canonical)
        if previous is not None and order <= previous:
            raise EvidenceError("vector within-cell allocation order drift")
        previous_by_cell[canonical] = order
        selected_units.append({
            "source_match": normalized["source_match"],
            "unit": normalized["unit"],
            **{key: normalized[key] for key in (
                "state_sha256", "allocation_priority_sha256", "round",
                "ply_stratum", "frontier_present", "allocation_slot",
                "master_width", "post_stratum",
            )},
        })
        allocations.append({
            **normalized,
            "source_match_index": row["source_match_index"],
            "source_state_index": row["source_state_index"],
            "source_match_id": normalized["source_match"],
            "orbit_sha256": row["orbit_sha256"],
            "mask_001_sha256": row["mask_001_sha256"],
            "mask_002_sha256": row["mask_002_sha256"],
            "master_sha256": row["master_sha256"],
            "census_count": row["census_count"],
            "allocation_quota": row["allocation_quota"],
            "weight_numerator": row["weight_numerator"],
            "weight_denominator": row["weight_denominator"],
            # Audit-only complete information-view encoding.  This never
            # enters SELECT/TEST inference, but lets the independent converter
            # reproduce the public dead-discard guard instead of trusting the
            # native composed-decision label.
            "state_hex": row["state_hex"],
            # This is the SHA-256 of the raw discovery JSONL, not the
            # synthesized selection manifest payload below.
            "discovery_sha256": discovery_sha,
        })
    if cursor != len(lines):
        raise EvidenceError("trailing vector allocation content")
    manifest_payload = {
        "schema": "lc-policy-cost-discovery-manifest-v1",
        "stage": split,
        "ply_boundaries": list(range(0, 44, 2)) + [44, 48, 64],
        "base_vector_quota": 64,
        "source_reservoir_sha256": reservoir_sha,
        "source_net_sha256": source_net_sha,
        "source_exclusion_sha256": source_exclusion_sha,
        "eligible_state_commitment_sha256": chain_sha,
        "allocation_rule_sha256": rule_sha,
        "total_eligible_states": total,
        "master_width_histogram": histogram,
        "cells": cells,
        "selected_units": selected_units,
    }
    digest = _canonical_payload_digest(manifest_payload)
    manifest = dict(manifest_payload)
    manifest["canonical_payload_sha256"] = digest
    for row in allocations:
        row["discovery_census_sha256"] = digest
    masses = {
        name: census / total
        for name, (census, quota, _) in sorted(cell_values.items()) if quota > 0
    }
    return manifest, allocations, masses


def _truth_action_mean(row: Mapping[str, Any], metric: str, position: int) -> float:
    truth = row.get("truth")
    metrics = truth.get("metrics") if isinstance(truth, dict) else None
    selected = metrics.get(metric) if isinstance(metrics, dict) else None
    actions = selected.get("actions") if isinstance(selected, dict) else None
    if not isinstance(actions, list) or not 0 <= position < len(actions):
        raise EvidenceError(f"truth metric {metric} lacks selected position")
    action = actions[position]
    if not isinstance(action, dict) or action.get("position") != position:
        raise EvidenceError("truth action position drift")
    return _require_number(action.get("mean"), f"truth {metric} mean")


def _decision(row: Mapping[str, Any], floor: float, ply_lo: int) -> int:
    policy = row.get("policy")
    if not isinstance(policy, dict) or not isinstance(policy.get("literal_argmax_index"), int):
        raise EvidenceError("holdout literal policy argmax is absent")
    if int(row.get("nply", -1)) < ply_lo:
        return int(policy["literal_argmax_index"])
    decisions = row.get("production_decisions")
    key = f"floor-{floor:.2f}"
    selected = decisions.get(key) if isinstance(decisions, dict) else None
    if not isinstance(selected, dict) or selected.get("exact_valid") is not True or \
            selected.get("capped") != 0 or \
            not isinstance(selected.get("selected_legal_index"), int):
        raise EvidenceError(
            f"holdout lacks exact composed P/F/discard-guard decision for {key}"
        )
    return int(selected["selected_legal_index"])


def _truth_position(row: Mapping[str, Any], legal_index: int) -> int:
    support = row.get("truth_support_legal_indices")
    if not isinstance(support, list) or any(
        not isinstance(item, int) for item in support
    ) or len(support) != len(set(support)):
        raise EvidenceError("truth support is malformed")
    try:
        return support.index(legal_index)
    except ValueError as exc:
        raise EvidenceError("selected move is absent from untruncated truth support") from exc


def _selected_config(selection: Mapping[str, Any]) -> tuple[str, float, int]:
    selected = selection.get("selected")
    if not isinstance(selected, dict) or set(selected) != {
        "id", "policy_floor", "ply_lo"
    }:
        raise EvidenceError("selection has no frozen selected configuration")
    config_id = selected["id"]
    floor = selected["policy_floor"]
    ply_lo = selected["ply_lo"]
    if config_id not in CONFIG_IDS or floor not in FLOORS or ply_lo not in PLY_LOS or \
            config_id != f"floor-{floor:.2f}_ply-{ply_lo:02d}":
        raise EvidenceError("selected configuration metadata drift")
    return str(config_id), float(floor), int(ply_lo)


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _domain_seed(root: str, match: int, state: int, tag: int) -> str:
    value = int(root) ^ _mix64(match) ^ _mix64(state) ^ \
        _mix64(0x100000001B3) ^ tag
    return str(_mix64(value))


def _close(left: float, right: float, label: str) -> None:
    """Require the deterministic summary algebra, allowing JSON roundoff only."""

    if not math.isfinite(left) or not math.isfinite(right) or \
            abs(left - right) > 2.0e-9 * max(1.0, abs(left), abs(right)):
        raise EvidenceError(f"{label} algebra drift")


def _sample_moments(action: Mapping[str, Any], worlds: int, label: str,
                    *, hashes: bool) -> tuple[float, float, float]:
    mean = _require_number(action.get("mean"), f"{label} mean")
    se = _require_number(action.get("se"), f"{label} SE")
    total = _require_number(action.get("sum"), f"{label} sum")
    sum_squares = _require_number(action.get("sum_squares"), f"{label} sum_squares")
    if se < 0.0:
        raise EvidenceError(f"{label} has negative SE")
    if hashes and HEX64.fullmatch(str(action.get("samples_sha256", ""))) is None:
        raise EvidenceError(f"{label} samples digest drift")
    _close(total, worlds * mean, label + " sum")
    _close(sum_squares, se * se * worlds * (worlds - 1) + worlds * mean * mean,
           label + " sum_squares")
    return mean, se, sum_squares


def _pair_moment(pair: Mapping[str, Any], means: Sequence[float],
                 sum_squares: Sequence[float], worlds: int, label: str) -> None:
    if not isinstance(pair.get("a"), int) or not isinstance(pair.get("b"), int) or \
            not 0 <= pair["a"] < pair["b"] < len(means):
        raise EvidenceError(f"{label} identity drift")
    delta = _require_number(pair.get("delta_a_minus_b"), label + " delta")
    paired_se = _require_number(pair.get("paired_se"), label + " paired SE")
    products = _require_number(pair.get("sum_products"), label + " sum_products")
    if paired_se < 0.0:
        raise EvidenceError(f"{label} negative paired SE")
    left, right = pair["a"], pair["b"]
    _close(delta, means[left] - means[right], label + " delta")
    diff_squares = paired_se * paired_se * worlds * (worlds - 1) + worlds * delta * delta
    _close(products, 0.5 * (sum_squares[left] + sum_squares[right] - diff_squares),
           label + " sum_products")


def _finite_support_panel(panel: Mapping[str, Any], *, seed: str, z: float,
                          masks: Sequence[Sequence[int]], role: int,
                          mask_indices: Sequence[int] | None = None) -> list[Mapping[str, Any]]:
    if panel.get("seed") != seed or panel.get("requested_worlds") != 800 or \
            panel.get("z") != z or panel.get("common_worlds_across_actions") is not True or \
            panel.get("overlap_bit_exact") is not True:
        raise EvidenceError("holdout P/F panel protocol drift")
    raw_masks = panel.get("masks")
    if not isinstance(raw_masks, list) or len(raw_masks) != len(masks):
        raise EvidenceError("holdout P/F mask coverage drift")
    result: list[Mapping[str, Any]] = []
    for index, (raw, expected) in enumerate(zip(raw_masks, masks)):
        expected_index = mask_indices[index] if mask_indices is not None else index
        if not isinstance(raw, dict) or raw.get("mask_index") != expected_index or \
                raw.get("panel_role") != role or raw.get("unfinished_cap_leaves") != 0 or \
                re.fullmatch(r"[0-9a-f]{16}", str(raw.get("hidden_world_fingerprint", ""))) is None:
            raise EvidenceError("holdout P/F panel identity/cap drift")
        worlds, support = raw.get("worlds"), raw.get("hidden_support")
        if not _finite_support_worlds(
                worlds, support, raw.get("exact_hidden_support")):
            raise EvidenceError("holdout P/F finite-support contract drift")
        actions = raw.get("actions")
        if not isinstance(actions, list) or len(actions) != len(expected):
            raise EvidenceError("holdout P/F action coverage drift")
        means: list[float] = []
        sum_squares: list[float] = []
        for position, action in enumerate(actions):
            if not isinstance(action, dict) or action.get("position") != position or \
                    action.get("legal_index") != expected[position]:
                raise EvidenceError("holdout P/F action identity drift")
            mean, _, sum_square = _sample_moments(
                action, worlds, "holdout P/F action", hashes=False
            )
            means.append(mean)
            sum_squares.append(sum_square)
        pairs = raw.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != len(actions) * (len(actions) - 1) // 2:
            raise EvidenceError("holdout P/F pair coverage drift")
        pair_keys: set[tuple[int, int]] = set()
        for pair in pairs:
            if not isinstance(pair, dict):
                raise EvidenceError("holdout P/F pair algebra drift")
            _pair_moment(pair, means, sum_squares, worlds, "holdout P/F pair")
            pair_keys.add((pair["a"], pair["b"]))
        if pair_keys != {(left, right) for left in range(len(means))
                         for right in range(left + 1, len(means))}:
            raise EvidenceError("holdout P/F pair keys are not complete and unique")
        result.append(raw)
    return result


def _overlap_exact(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]],
                   label: str) -> None:
    """Check that independently serialized masks are exact projections of one panel."""

    by_index: dict[int, Mapping[str, Any]] = {}
    for raw in left:
        for action in raw["actions"]:
            legal = action["legal_index"]
            if legal in by_index:
                raise EvidenceError(f"{label} left mask legal duplicate")
            by_index[legal] = action
    other: dict[int, Mapping[str, Any]] = {}
    for raw in right:
        for action in raw["actions"]:
            legal = action["legal_index"]
            if legal in other:
                raise EvidenceError(f"{label} right mask legal duplicate")
            other[legal] = action
    for legal in set(by_index).intersection(other):
        # These values are claimed bit-exact by the native evaluator; JSON's
        # float representation round-trips them exactly.
        for field in ("mean", "se", "sum", "sum_squares"):
            if by_index[legal].get(field) != other[legal].get(field):
                raise EvidenceError(f"{label} action overlap is not bit exact")
    # Pair summaries also must agree after remapping local positions to legal moves.
    def pairs(raws: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
        result: dict[tuple[int, int], Mapping[str, Any]] = {}
        for raw in raws:
            legal = [item["legal_index"] for item in raw["actions"]]
            for pair in raw["pairs"]:
                key = tuple(sorted((legal[pair["a"]], legal[pair["b"]])))
                if key in result:
                    raise EvidenceError(f"{label} pair overlap duplicate")
                result[key] = pair
        return result
    left_pairs, right_pairs = pairs(left), pairs(right)
    for key in set(left_pairs).intersection(right_pairs):
        for field in ("delta_a_minus_b", "paired_se", "sum_products"):
            if left_pairs[key].get(field) != right_pairs[key].get(field):
                raise EvidenceError(f"{label} pair overlap is not bit exact")


def _lambda_at(values: Sequence[Any], ply: int) -> tuple[int, float]:
    if not isinstance(ply, int) or isinstance(ply, bool) or not 0 <= ply <= ANCHORS[-1] or \
            not isinstance(values, list) or len(values) != len(ANCHORS):
        raise EvidenceError("policy-cost schedule/ply drift")
    checked = [_require_number(value, "policy-cost lambda") for value in values]
    for index, anchor in enumerate(ANCHORS):
        if ply == anchor:
            return index, checked[index]
        if ply < anchor:
            lower = index - 1
            fraction = (ply - ANCHORS[lower]) / (anchor - ANCHORS[lower])
            # Match the native C interpolation expression exactly; the
            # algebraically equivalent convex form can differ by one binary64
            # ulp under the frozen fast-math/FMA build.
            return lower, checked[lower] + fraction * (checked[index] - checked[lower])
    return len(ANCHORS) - 1, checked[-1]


def _verify_policy_cost_decision(raw: Mapping[str, Any], mask: Mapping[str, Any],
                                 policy: Mapping[str, Any], nply: int,
                                 z: float, beta_search: Sequence[Any],
                                 alpha_action: Sequence[Any],
                                 alpha_draw: Sequence[Any]) -> tuple[int, bool]:
    """Second implementation of ``policy_cost_decide_summary`` over summaries."""

    if not isinstance(raw, dict):
        raise EvidenceError("policy-cost decision is absent")
    actions = raw.get("actions")
    search_actions = mask.get("actions")
    legal_indices = [item.get("legal_index") for item in search_actions] \
        if isinstance(search_actions, list) else []
    policy_legal = policy.get("legal")
    if not isinstance(actions, list) or len(actions) != len(legal_indices) or \
            not isinstance(policy_legal, list):
        raise EvidenceError("policy-cost decision action coverage drift")
    legal_by_index = {item.get("index"): item for item in policy_legal if isinstance(item, dict)}
    interval, beta = _lambda_at(beta_search, nply)
    action_interval, action_alpha = _lambda_at(alpha_action, nply)
    draw_interval, draw_alpha = _lambda_at(alpha_draw, nply)
    if beta <= 0.0:
        raise EvidenceError("policy-cost interpolated beta is not positive")
    action_lambda = action_alpha / beta
    draw_lambda = draw_alpha / beta
    if interval != action_interval or interval != draw_interval or \
            raw.get("anchor_interval") != interval or \
            not math.isclose(_require_number(raw.get("beta_search"), "policy beta"),
                             beta, rel_tol=2e-15, abs_tol=2e-15) or \
            not math.isclose(_require_number(raw.get("alpha_action"), "policy alpha action"),
                             action_alpha, rel_tol=2e-15, abs_tol=2e-15) or \
            not math.isclose(_require_number(raw.get("alpha_draw"), "policy alpha draw"),
                             draw_alpha, rel_tol=2e-15, abs_tol=2e-15) or \
            not math.isclose(_require_number(raw.get("lambda_action"), "policy lambda action"),
                               action_lambda, rel_tol=2e-15, abs_tol=2e-15) or \
            not math.isclose(_require_number(raw.get("lambda_draw"), "policy lambda draw"),
                               draw_lambda, rel_tol=2e-15, abs_tol=2e-15):
        raise EvidenceError("policy-cost decision schedule drift")
    adjusted: list[float] = []
    core: list[float] = []
    draw: list[float] = []
    semantic_core: list[tuple[int, int]] = []
    for position, item in enumerate(actions):
        expected = legal_by_index.get(legal_indices[position])
        if not isinstance(item, dict) or item.get("position") != position or \
                not isinstance(expected, dict):
            raise EvidenceError("policy-cost decision legal identity drift")
        prior = _require_number(expected.get("semantic_action_probability"), "policy core prior")
        conditional = _require_number(expected.get("conditional_draw_probability"), "policy draw prior")
        expected_cost = -action_lambda * math.log(prior) - draw_lambda * math.log(conditional)
        if prior <= 0.0 or prior > 1.0 or conditional <= 0.0 or conditional > 1.0 or \
                item.get("semantic_prior") != prior or \
                item.get("conditional_draw_prior") != conditional or \
                not math.isclose(_require_number(item.get("cost"), "policy cost"), expected_cost,
                                   rel_tol=2e-9, abs_tol=2e-9):
            raise EvidenceError("policy-cost decision terms drift")
        q = _require_number(search_actions[position].get("mean"), "policy search Q")
        value = q - expected_cost
        if not math.isclose(_require_number(item.get("adjusted_q"), "policy adjusted Q"), value,
                            rel_tol=2e-9, abs_tol=2e-9):
            raise EvidenceError("policy-cost adjusted score drift")
        adjusted.append(value); core.append(prior); draw.append(conditional)
        semantic_core.append((
            _semantic_card(int(expected["card"])), int(expected["discard"])
        ))
    q = [_require_number(item.get("mean"), "policy search Q")
         for item in search_actions]
    hierarchical_draw_only = action_alpha == 0.0
    if hierarchical_draw_only:
        raw_leader = max(range(len(q)), key=lambda index: q[index])
        leader = max(
            (index for index in range(len(adjusted))
             if semantic_core[index] == semantic_core[raw_leader]),
            key=lambda index: adjusted[index],
        )
    else:
        leader = max(range(len(adjusted)), key=lambda index: adjusted[index])
    if raw.get("leader_position") != leader:
        raise EvidenceError("policy-cost leader drift")
    if leader == 0:
        if raw.get("all_pair_passed") is not True or raw.get("selected_position") != 0 or \
                raw.get("prior_protected_rivals") != 0:
            raise EvidenceError("policy-cost baseline decision drift")
        return leader, True
    pair_by_key = {
        (pair["a"], pair["b"]): pair for pair in mask.get("pairs", [])
        if isinstance(pair, dict) and isinstance(pair.get("a"), int) and isinstance(pair.get("b"), int)
    }
    passed, protected_count = True, 0
    for rival in range(len(adjusted)):
        if rival == leader:
            continue
        pair = pair_by_key.get((rival, leader) if rival < leader else (leader, rival))
        if pair is None:
            raise EvidenceError("policy-cost pair protection coverage drift")
        se = _require_number(pair.get("paired_se"), "policy paired SE")
        same_core = semantic_core[rival] == semantic_core[leader]
        evidence_delta = (
            q[leader] - q[rival]
            if hierarchical_draw_only and not same_core
            else adjusted[leader] - adjusted[rival]
        )
        protected = (
            same_core and draw[rival] > draw[leader]
            if hierarchical_draw_only
            else rival == 0 or core[rival] > core[leader] or
            core[rival] * draw[rival] > core[leader] * draw[leader] or
            (same_core and draw[rival] > draw[leader])
        )
        protected_count += int(protected)
        if not (evidence_delta > z * se) or \
                (protected and not (q[leader] - q[rival] > 0.0)):
            passed = False
    if raw.get("all_pair_passed") is not passed or \
            raw.get("selected_position") != (leader if passed else 0) or \
            raw.get("prior_protected_rivals") != protected_count:
        raise EvidenceError("policy-cost all-pair decision drift")
    return leader, passed


def _discard_dominated_from_view(state_hex: str, policy: Mapping[str, Any],
                                 legal_index: int) -> bool:
    """Independently reproduce ``lc_discard_dominated`` from sealed view bytes.

    The allocation's v1 encoding is fixed-width and deliberately excludes the
    hidden deck/opponent hand.  The guard itself needs only the mover hand,
    public expedition tops, and the complete move, so this check remains
    hidden-information-safe while preventing a native summary from choosing
    between ``selected`` and ``discard_guard`` after the fact.
    """

    if not isinstance(state_hex, str):
        raise EvidenceError("discard guard state encoding is absent")
    try:
        state = bytes.fromhex(state_hex)
    except ValueError as exc:
        raise EvidenceError("discard guard state encoding drift") from exc
    # encode_view(): version/deck/hand counts, two u64 hands, two played
    # masks, discarded, two known masks, four 2x5 expedition arrays, five
    # fixed 12-card pile arrays, then turn/over/nply/round/cumulative scores.
    if len(state) != 174 or state[0] != 1 or state[165] not in (0, 1) or \
            state[166] != 0:
        raise EvidenceError("discard guard state encoding drift")
    turn = state[165]
    hand_offset = 4 + 8 * turn
    hand = int.from_bytes(state[hand_offset:hand_offset + 8], "little")
    tops = state[70:80]
    if len(tops) != 10 or any(value > 10 for value in tops):
        raise EvidenceError("discard guard expedition tops drift")

    legal = policy.get("legal")
    if not isinstance(legal, list):
        raise EvidenceError("discard guard full legal policy is absent")
    matches = [item for item in legal if isinstance(item, dict) and
               item.get("index") == legal_index]
    if len(matches) != 1:
        raise EvidenceError("discard guard legal move identity drift")
    move = matches[0]
    card, discard, draw = move.get("card"), move.get("discard"), move.get("draw")
    if isinstance(card, bool) or not isinstance(card, int) or not 0 <= card < 60 or \
            discard not in (0, 1) or draw not in range(6):
        raise EvidenceError("discard guard move encoding drift")
    if discard == 0:
        return False

    dead = 0
    for suit in range(5):
        top0, top1 = tops[suit], tops[5 + suit]
        base = 12 * suit
        if top0 > 0 and top1 > 0:
            dead |= 0b111 << base
        for value in range(2, min(top0, top1) + 1):
            dead |= 1 << (base + value + 1)
    if (dead >> card) & 1:
        return False
    alternatives = hand & dead
    while alternatives:
        low = alternatives & -alternatives
        candidate = low.bit_length() - 1
        alternatives ^= low
        if candidate == card:
            continue
        if draw != 0 and draw - 1 == candidate // 12:
            continue
        return True
    return False


def _verify_composed_decision(value: Mapping[str, Any], primary: tuple[int, bool],
                              fresh: tuple[int, bool], legal: Sequence[int],
                              *, active: bool,
                              discard_rejected: bool = False) -> None:
    if not isinstance(value, dict) or value.get("exact_valid") is not True or value.get("capped") != 0:
        raise EvidenceError("composed policy-cost decision validity drift")
    expected_position = 0
    expected_reasons: set[str]
    if not active:
        expected_reasons = {"before_onset_policy_baseline"}
    elif primary[0] == 0:
        expected_reasons = {"adjusted_baseline"}
    elif not primary[1]:
        expected_reasons = {"primary_evidence"}
    elif fresh[0] != primary[0]:
        expected_reasons = {"fresh_leader_mismatch"}
    elif not fresh[1]:
        expected_reasons = {"fresh_evidence"}
    else:
        expected_position = 0 if discard_rejected else primary[0]
        expected_reasons = {
            "discard_guard" if discard_rejected else "selected"
        }
    reason = value.get("gate_reason")
    if reason not in expected_reasons:
        raise EvidenceError("composed policy-cost gate drift")
    if value.get("selected_position") != expected_position or \
            value.get("selected_legal_index") != legal[expected_position]:
        raise EvidenceError("composed policy-cost selected action drift")


def _verify_holdout_raw(row: Mapping[str, Any], *, stage: str,
                        selected_floor: float | None, selected_ply: int | None,
                        state_hex: str,
                        beta_search: Sequence[Any] | None = None,
                        alpha_action: Sequence[Any] | None = None,
                        alpha_draw: Sequence[Any] | None = None) -> None:
    if stage == "TEST" and (
        "config_decisions" not in row or row.get("config_decisions") is not None
    ):
        raise EvidenceError("TEST configuration decision map must be JSON null")
    policy = row.get("policy")
    _verify_full_policy(policy if isinstance(policy, dict) else {})
    masks_raw = policy.get("runtime_masks") if isinstance(policy, dict) else None
    if not isinstance(masks_raw, list) or len(masks_raw) != 2:
        raise EvidenceError("holdout lacks runtime masks")
    masks = [item.get("legal_indices") for item in masks_raw if isinstance(item, dict)]
    if len(masks) != 2 or any(not isinstance(item, list) for item in masks):
        raise EvidenceError("holdout runtime mask structure drift")
    active = stage == "SELECT" or int(row.get("nply", -1)) >= int(selected_ply)
    search = row.get("search")
    if not isinstance(search, dict):
        raise EvidenceError("holdout raw search panels are absent")
    match, state = row.get("source_match_index"), row.get("source_state_index")
    if isinstance(match, bool) or not isinstance(match, int) or \
            isinstance(state, bool) or not isinstance(state, int):
        raise EvidenceError("holdout source seed identity drift")
    selected_mask = 0 if selected_floor == 0.01 else 1
    primary = _finite_support_panel(
        search.get("primary") if isinstance(search.get("primary"), dict) else {},
        seed=_domain_seed(PRIMARY_SEEDS[stage], match, state, 0x5052494D41525932),
        z=3.5, masks=masks, role=0, mask_indices=(0, 1),
    )
    fresh = _finite_support_panel(
        search.get("fresh") if isinstance(search.get("fresh"), dict) else {},
        seed=_domain_seed(FRESH_SEEDS[stage], match, state, 0x46524553485F5032),
        z=2.58, masks=masks, role=1, mask_indices=(0, 1),
    )
    if len(primary) == 2:
        if search["primary"].get("same_seed_across_exact_masks") is not True or \
                search["fresh"].get("same_seed_across_exact_masks") is not True or \
                primary[0].get("hidden_world_fingerprint") != primary[1].get("hidden_world_fingerprint") or \
                fresh[0].get("hidden_world_fingerprint") != fresh[1].get("hidden_world_fingerprint"):
            raise EvidenceError("exact mask seed/fingerprint binding drift")
        _overlap_exact(primary[:1], primary[1:], "primary")
        _overlap_exact(fresh[:1], fresh[1:], "fresh")
    for left, right in zip(primary, fresh):
        equal_complete_census = (
            left.get("exact_hidden_support") is True and
            right.get("exact_hidden_support") is True and
            left.get("worlds") == right.get("worlds") and
            left.get("hidden_support") == left.get("worlds") and
            right.get("hidden_support") == right.get("worlds") and
            left.get("worlds") < 800
        )
        if left.get("hidden_world_fingerprint") == right.get("hidden_world_fingerprint") and \
                not equal_complete_census:
            raise EvidenceError("P/F panels reuse hidden worlds")
    if beta_search is None or alpha_action is None or alpha_draw is None:
        raise EvidenceError("holdout raw decision lacks LCPC binding")
    computed: list[tuple[tuple[int, bool], tuple[int, bool], list[int]]] = []
    for primary_mask, fresh_mask in zip(primary, fresh):
        primary_decision = _verify_policy_cost_decision(
            primary_mask.get("policy_cost_decision"), primary_mask, policy,
            int(row.get("nply", -1)), 3.5,
            beta_search, alpha_action, alpha_draw,
        )
        fresh_decision = _verify_policy_cost_decision(
            fresh_mask.get("policy_cost_decision"), fresh_mask, policy,
            int(row.get("nply", -1)), 2.58,
            beta_search, alpha_action, alpha_draw,
        )
        computed.append((primary_decision, fresh_decision,
                         [action["legal_index"] for action in primary_mask["actions"]]))
    decisions = row.get("production_decisions")
    if not isinstance(decisions, dict):
        raise EvidenceError("holdout production decision map is absent")

    def discard_rejected(
        evidence: tuple[tuple[int, bool], tuple[int, bool], list[int]]
    ) -> bool:
        primary_decision, fresh_decision, legal = evidence
        proposed = primary_decision[0]
        if proposed == 0 or not primary_decision[1] or \
                fresh_decision[0] != proposed or not fresh_decision[1]:
            return False
        return (
            not _discard_dominated_from_view(state_hex, policy, legal[0])
            and _discard_dominated_from_view(
                state_hex, policy, legal[proposed]
            )
        )

    if stage == "TEST":
        if selected_floor is None:
            raise EvidenceError("TEST selected floor is absent")
        if set(decisions) != {f"floor-{selected_floor:.2f}"}:
            raise EvidenceError("TEST production decision map drift")
        evidence = computed[selected_mask]
        _verify_composed_decision(
            decisions.get(f"floor-{selected_floor:.2f}"), evidence[0], evidence[1],
            evidence[2], active=active,
            discard_rejected=active and discard_rejected(evidence),
        )
    else:
        if set(decisions) != {"floor-0.01", "floor-0.02"}:
            raise EvidenceError("SELECT production decision map drift")
        configs = row.get("config_decisions")
        if not isinstance(configs, dict) or set(configs) != set(CONFIG_IDS):
            raise EvidenceError("SELECT configuration decision map drift")
        for mask_index, floor in enumerate(FLOORS):
            _verify_composed_decision(
                decisions.get(f"floor-{floor:.2f}"), computed[mask_index][0],
                computed[mask_index][1], computed[mask_index][2], active=True,
                discard_rejected=discard_rejected(computed[mask_index]),
            )
            for ply_lo in PLY_LOS:
                config_active = int(row["nply"]) >= ply_lo
                _verify_composed_decision(
                    configs.get(f"floor-{floor:.2f}_ply-{ply_lo:02d}"),
                    computed[mask_index][0], computed[mask_index][1],
                    computed[mask_index][2], active=config_active,
                    discard_rejected=(
                        config_active and discard_rejected(computed[mask_index])
                    ),
                )
    truth = row.get("truth")
    metrics = truth.get("metrics") if isinstance(truth, dict) else None
    support = row.get("truth_support_legal_indices")
    match = row.get("source_match_index")
    state = row.get("source_state_index")
    if not isinstance(match, int) or isinstance(match, bool) or \
            not isinstance(state, int) or isinstance(state, bool) or \
            not isinstance(truth, dict) or truth.get("controller") != \
            "exact_policy20_full_remaining_match" or \
            truth.get("information_view_each_node") is not True or \
            truth.get("temperature") != 0 or truth.get("epsilon") != 0 or \
            truth.get("requested_worlds") != TRUTH_WORLDS[stage] or \
            truth.get("worlds") != TRUTH_WORLDS[stage] or \
            truth.get("seed") != _domain_seed(
                TRUTH_SEEDS[stage], match, state, 0x5452555448574C44
            ) or \
            truth.get("cap_hits") != 0 or truth.get("union_untruncated") is not True or \
            not isinstance(support, list) or len(support) != len(set(support)) or \
            truth.get("union_count") != len(support) or not isinstance(metrics, dict):
        raise EvidenceError("holdout truth protocol drift")
    maintained = row.get("maintained_baseline")
    legal = policy.get("legal") if isinstance(policy, dict) else None
    master = masks[0]
    if not isinstance(maintained, dict) or not isinstance(legal, list) or \
            maintained.get("actor_selected") is not True or \
            maintained.get("information_view") is not True or \
            maintained.get("unfinished_cap_leaves") != 0 or \
            maintained.get("root_seed") != _domain_seed(
                MAINTAINED_SEEDS[stage], match, state, 0x4D41494E5441494E
            ) or not isinstance(maintained.get("semantic_move_pack"), int) or \
            not isinstance(maintained.get("truth_position"), int) or \
            type(maintained.get("appended_outside_new_union")) is not bool:
        raise EvidenceError("holdout maintained truth binding drift")
    matching = [item.get("index") for item in legal if isinstance(item, dict) and
                item.get("semantic_move_pack") == maintained["semantic_move_pack"]]
    if len(matching) != 1:
        raise EvidenceError("maintained semantic move is not uniquely legal")
    maintained_legal = matching[0]
    expected_support = list(master)
    appended = maintained_legal not in expected_support
    if appended:
        expected_support.append(maintained_legal)
    if support != expected_support or maintained.get("truth_position") != expected_support.index(maintained_legal) or \
            maintained.get("appended_outside_new_union") is not appended:
        raise EvidenceError("holdout truth support/maintained action drift")
    if any(HEX64.fullmatch(str(truth.get(field, ""))) is None
           for field in ("hidden_worlds_sha256", "future_deals_sha256")):
        raise EvidenceError("holdout truth digest drift")
    for name in ("current_round_margin", "full_match_margin", "full_match_score", "full_match_hybrid"):
        metric = metrics.get(name)
        actions = metric.get("actions") if isinstance(metric, dict) else None
        pairs = metric.get("pairs") if isinstance(metric, dict) else None
        if not isinstance(actions, list) or len(actions) != len(support) or any(
                not isinstance(action, dict) or action.get("position") != index
                for index, action in enumerate(actions)):
            raise EvidenceError("holdout truth action support drift")
        means: list[float] = []
        sum_squares: list[float] = []
        for position, action in enumerate(actions):
            if not isinstance(action, dict) or action.get("position") != position:
                raise EvidenceError("holdout truth action support drift")
            mean, _, sum_square = _sample_moments(
                action, int(truth["worlds"]), "truth action", hashes=True
            )
            means.append(mean)
            sum_squares.append(sum_square)
        if not isinstance(pairs, list) or len(pairs) != len(actions) * (len(actions) - 1) // 2:
            raise EvidenceError("holdout truth pair coverage drift")
        keys: set[tuple[int, int]] = set()
        for pair in pairs:
            if not isinstance(pair, dict):
                raise EvidenceError("holdout truth pair algebra drift")
            _pair_moment(pair, means, sum_squares, int(truth["worlds"]), "truth pair")
            keys.add((pair["a"], pair["b"]))
        if keys != {(left, right) for left in range(len(means))
                    for right in range(left + 1, len(means))}:
            raise EvidenceError("holdout truth pair keys drift")


def _verify_holdout_artifact(header: Mapping[str, Any], *, calibration: Path,
                             artifact: Path, stage: str, selection: Mapping[str, Any] | None,
                             actors: Path | None, execution: Mapping[str, Any]) -> dict[str, Any]:
    """Bind each evaluator header to the independently parsed LCPC schedule."""

    schedule = strict_json(calibration)
    try:
        from tools.policy_cost_artifact_v19 import read_policy_cost
    except ImportError:
        from policy_cost_artifact_v19 import read_policy_cost  # type: ignore
    parsed = read_policy_cost(artifact)
    subject = execution.get("subject") if isinstance(execution, dict) else None
    if not isinstance(subject, dict) or \
            header.get("actor_spec") != subject.get("neutral_counterfactual_actor") or \
            header.get("maintained_actor_spec") != subject.get("maintained_actor") or \
            header.get("truth", {}).get("controller") != "exact_policy20_full_remaining_match" or \
            header.get("seed_domains_pairwise_disjoint") is not True or \
            header.get("arbitrary_top_five_truncation") is not False:
        raise EvidenceError("holdout execution/evaluator identity binding drift")
    if schedule.get("schema") != "lc-policy-cost-calibration-v2" or \
            schedule.get("calibration_passed") is not True or \
            schedule.get("status") != "passed" or \
            schedule.get("deployment") != {
                "permitted": True, "reason": None,
            } or \
            schedule.get("schedule", {}).get("ply_anchors") != list(ANCHORS) or \
            parsed.get("beta") != schedule.get("schedule", {}).get("beta_search") or \
            parsed.get("alpha_action") != schedule.get("schedule", {}).get("alpha_core") or \
            parsed.get("alpha_draw") != schedule.get("schedule", {}).get("alpha_draw") or \
            header.get("policy_cost_sha256") != sha256(artifact) or \
            header.get("policy_cost_payload_fingerprint") != parsed.get("content_fingerprint"):
        raise EvidenceError("holdout calibration/LCPC/header binding drift")
    controller = parsed["controller"]
    if header.get("root_net_fingerprint") != controller["root_net_fingerprint"] or \
            header.get("continuation_net_fingerprint") != controller["continuation_net_fingerprint"]:
        raise EvidenceError("holdout LCPC checkpoint binding drift")
    if stage == "TEST":
        if selection is None or actors is None:
            raise EvidenceError("TEST requires actor-manifest binding")
        actor = strict_json(actors)
        if actor.get("selection_sha256") != sha256(selection) or \
                actor.get("calibration_sha256") != sha256(calibration) or \
                actor.get("policy_cost_artifact", {}).get("sha256") != sha256(artifact) or \
                actor.get("policy_cost_artifact", {}).get("content_fingerprint") != \
                parsed.get("content_fingerprint"):
            raise EvidenceError("TEST actor-manifest/LCPC binding drift")
    return {
        "calibration_sha256": sha256(calibration),
        "policy_cost_sha256": sha256(artifact),
        "policy_cost_content_fingerprint": parsed["content_fingerprint"],
        "evaluation_header_sha256": _canonical_payload_digest(header),
        "raw_verified": True,
        "_beta_search": parsed["beta"],
        "_alpha_action": parsed["alpha_action"],
        "_alpha_draw": parsed["alpha_draw"],
    }


def _sealed_campaign_selection(selection: Mapping[str, Any], *, require_evidence: bool = False) -> tuple[str, float, int]:
    """Validate the immutable SELECT result before any TEST/actor use."""

    if selection.get("schema") != "lc-policy-cost-select-result-v1" or \
            selection.get("stage") != "SELECT":
        raise EvidenceError("SELECT result schema drift")
    try:
        from tools.policy_cost_selection_v19 import verify_result_digest
    except ImportError:
        from policy_cost_selection_v19 import verify_result_digest  # type: ignore
    if not verify_result_digest(selection):
        raise EvidenceError("SELECT result digest is invalid")
    discovery_binding = selection.get("campaign_discovery_binding")
    if not isinstance(discovery_binding, dict) or \
            discovery_binding.get("required") is not True or \
            discovery_binding.get("validated") is not True:
        raise EvidenceError("SELECT result lacks campaign discovery binding")
    selection_rule = selection.get("selection_rule")
    if not isinstance(selection_rule, dict) or \
            selection_rule.get("test_evidence_used") is not False:
        raise EvidenceError("SELECT result has invalid evidence use")
    if require_evidence:
        _validated_holdout_evidence(selection.get("campaign_evidence_binding"), "SELECT")
    return _selected_config(selection)


def _validated_holdout_evidence(value: Any, stage: str) -> Mapping[str, Any]:
    fields = {
        "raw_verified", "stage", "execution_sha256", "evaluation_sha256",
        "evaluation_header_sha256", "allocation_sha256", "calibration_sha256",
        "policy_cost_sha256", "policy_cost_content_fingerprint", "selection_sha256",
        "actor_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields or value.get("raw_verified") is not True or \
            value.get("stage") != stage:
        raise EvidenceError(f"{stage} campaign evidence schema drift")
    for field in ("execution_sha256", "evaluation_sha256", "evaluation_header_sha256",
                  "allocation_sha256", "calibration_sha256", "policy_cost_sha256"):
        if HEX64.fullmatch(str(value.get(field, ""))) is None:
            raise EvidenceError(f"{stage} campaign evidence digest drift")
    if re.fullmatch(r"[0-9a-f]{16}", str(value.get("policy_cost_content_fingerprint", ""))) is None:
        raise EvidenceError(f"{stage} campaign evidence LCPC fingerprint drift")
    if stage == "SELECT":
        if value.get("selection_sha256") is not None or value.get("actor_manifest_sha256") is not None:
            raise EvidenceError("SELECT campaign evidence claims TEST artifacts")
    else:
        for field in ("selection_sha256", "actor_manifest_sha256"):
            if HEX64.fullmatch(str(value.get(field, ""))) is None:
                raise EvidenceError(f"TEST campaign evidence {field} drift")
    return value


def holdout_input(evaluation: Path, allocation: Path, stage: str,
                  selection: Path | None = None, *, calibration: Path | None = None,
                  policy_cost: Path | None = None, actors: Path | None = None,
                  execution: Path | None = None,
                  verify_raw: bool = False) -> dict[str, Any]:
    if stage not in {"SELECT", "TEST"}:
        raise EvidenceError("holdout stage must be SELECT or TEST")
    header, allocations, _ = _evaluation(evaluation, stage)
    manifest, allocation_bindings, weights = vector_allocation_manifest(
        allocation, stage
    )
    if not allocation_bindings or \
            header.get("manifest_sha256") != sha256(allocation) or \
            header.get("discovery_sha256") != \
            allocation_bindings[0].get("discovery_sha256") or \
            header.get("reservoir_sha256") != \
            manifest.get("source_reservoir_sha256"):
        raise EvidenceError(f"{stage} evaluator header lineage drift")
    expected_by_id = {
        int(row["allocation_id"]): row for row in allocation_bindings
    }
    if set(expected_by_id) != set(range(HOLDOUT_RECORDS)):
        raise EvidenceError("vector allocation ids are incomplete")
    selected_config: tuple[str, float, int] | None = None
    selection_value: dict[str, Any] | None = None
    if stage == "TEST":
        if selection is None:
            raise EvidenceError("TEST conversion requires frozen SELECT result")
        selection_value = strict_json(selection)
        selected_config = _sealed_campaign_selection(selection_value, require_evidence=verify_raw)
    evidence_binding: dict[str, Any] | None = None
    if verify_raw:
        if calibration is None or policy_cost is None or execution is None:
            raise EvidenceError("holdout raw verification requires execution, calibration and LCPC")
        execution_value = strict_json(execution)
        if execution_value.get("schema") != EXECUTION_SCHEMA:
            raise EvidenceError("holdout execution binding drift")
        evidence_binding = _verify_holdout_artifact(
            header, calibration=calibration, artifact=policy_cost, stage=stage,
            selection=selection_value, actors=actors, execution=execution_value,
        )
        beta_search = evidence_binding.pop("_beta_search")
        alpha_action = evidence_binding.pop("_alpha_action")
        alpha_draw = evidence_binding.pop("_alpha_draw")
        evidence_binding.update({
            "stage": stage,
            "evaluation_sha256": sha256(evaluation),
            "allocation_sha256": sha256(allocation),
            "selection_sha256": sha256(selection) if selection else None,
            "actor_manifest_sha256": sha256(actors) if actors else None,
            "execution_sha256": sha256(execution),
        })
    post_counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for row in allocations:
        source = row.get("source_match_id")
        unit = row.get("state_sha256")
        allocation_id = row.get("allocation_id")
        rd, pb = row.get("round"), row.get("ply_stratum")
        policy = row.get("policy")
        if stage == "TEST" and (
            "config_decisions" not in row or row.get("config_decisions") is not None
        ):
            raise EvidenceError("TEST configuration decision map must be JSON null")
        masks = policy.get("runtime_masks") if isinstance(policy, dict) else None
        if not isinstance(source, str) or not source or not isinstance(unit, str) or \
                HEX64.fullmatch(unit) is None or rd not in (0, 1, 2) or \
                not isinstance(pb, int) or not 0 <= pb < 24 or \
                not isinstance(allocation_id, int) or \
                not isinstance(masks, list) or len(masks) != 2:
            raise EvidenceError(f"{stage} allocation identity/masks drift")
        expected = expected_by_id.get(allocation_id)
        if expected is None or any(
            row.get(key) != expected.get(key) for key in (
                "source_match_id", "state_sha256", "allocation_priority_sha256",
                "source_match_index", "source_state_index", "orbit_sha256",
                "round", "ply_stratum", "frontier_present", "allocation_slot",
                "master_width", "post_stratum", "census_count", "allocation_quota",
                "weight_numerator", "weight_denominator", "discovery_sha256",
            )
        ) or row.get("unit") != expected["unit"]:
            raise EvidenceError(f"{stage} evaluation is not bound to allocation")
        if verify_raw:
            _verify_runtime_masks(policy if isinstance(policy, dict) else {}, expected)
            _verify_holdout_raw(
                row, stage=stage,
                selected_floor=selected_config[1] if selected_config else None,
                selected_ply=selected_config[2] if selected_config else None,
                state_hex=str(expected["state_hex"]),
                beta_search=beta_search if verify_raw else None,
                alpha_action=alpha_action if verify_raw else None,
                alpha_draw=alpha_draw if verify_raw else None,
            )
        first = masks[0].get("legal_indices") if isinstance(masks[0], dict) else None
        second = masks[1].get("legal_indices") if isinstance(masks[1], dict) else None
        if not isinstance(first, list) or not isinstance(second, list) or \
                any(isinstance(item, bool) or not isinstance(item, int)
                    for item in first + second) or \
                len(first) != len(set(first)) or len(second) != len(set(second)):
            raise EvidenceError(f"{stage} shortlist masks are malformed")
        cursor = iter(first)
        if any(not any(candidate == item for candidate in cursor)
               for item in second):
            raise EvidenceError(
                f"{stage} 2% shortlist is not an ordered no-refill subset"
            )
        frontier = int(bool(expected["frontier_present"]))
        stratum = str(expected["post_stratum"])
        post_counts[stratum] += 1
        maintained = row.get("maintained_baseline")
        if not isinstance(maintained, dict) or \
                maintained.get("actor_selected") is not True or \
                maintained.get("information_view") is not True or \
                maintained.get("unfinished_cap_leaves") != 0 or \
                not isinstance(maintained.get("truth_position"), int):
            raise EvidenceError(f"{stage} maintained action is invalid")
        maintained_pos = int(maintained["truth_position"])
        configs: Iterable[tuple[str, float, int]] = (
            ((config_id, floor, ply_lo)
             for ply_lo in PLY_LOS for floor in reversed(FLOORS)
             for config_id in [f"floor-{floor:.2f}_ply-{ply_lo:02d}"])
            if stage == "SELECT" else (selected_config,)  # type: ignore[arg-type]
        )
        for config_id, floor, ply_lo in configs:
            selected_legal = _decision(row, floor, ply_lo)
            selected_pos = _truth_position(row, selected_legal)
            hybrid_gain = (
                _truth_action_mean(row, "full_match_hybrid", selected_pos)
                - _truth_action_mean(row, "full_match_hybrid", maintained_pos)
            )
            common = {
                "source_match": source,
                "unit": expected["unit"],
                "state_sha256": unit,
                "allocation_priority_sha256": expected["allocation_priority_sha256"],
                "config": config_id,
                "post_stratum": stratum,
                "round": rd,
                "ply_stratum": pb,
                "frontier_present": bool(frontier),
                "allocation_slot": expected["allocation_slot"],
                "master_width": expected["master_width"],
                "discovery_census_sha256": manifest["canonical_payload_sha256"],
                "hybrid_gain": hybrid_gain,
                "weight": expected["weight"],
                "exact_valid": True,
                "capped": 0,
            }
            if stage != "SELECT":
                common.update({
                    "match_score_gain": (
                        _truth_action_mean(row, "full_match_score", selected_pos)
                        - _truth_action_mean(row, "full_match_score", maintained_pos)
                    ),
                })
            output.append(common)
    expected_counts = Counter(
        str(row["post_stratum"]) for row in allocation_bindings
    )
    if post_counts != expected_counts or set(weights) != {
        key for key, value in expected_counts.items() if value > 0
    }:
        raise EvidenceError(f"{stage} evaluation/allocation post-strata drift")
    if stage == "SELECT":
        value = {
            "schema": "lc-policy-cost-select-input-v1",
            "discovery_manifest": manifest,
            "rows": output,
        }
        if evidence_binding is not None:
            value["campaign_evidence_binding"] = evidence_binding
        return value
    value = {
        "schema": "lc-policy-cost-test-input-v1",
        "rows": output,
        "discovery_manifest": manifest,
        "discovery_post_stratum_weights": weights,
    }
    if evidence_binding is not None:
        value["campaign_evidence_binding"] = evidence_binding
    return value


def calibration_gate_result(
    calibration: Path, execution: Path,
) -> dict[str, Any]:
    """Seal the only decision allowed between TRAIN and SELECT.

    A structurally valid authoritative model-adequacy rejection is evidence,
    not an infrastructure exception.  This reducer never refits the model; it
    validates and binds the already-computed canonical calibration result.
    """

    value = strict_json(calibration)
    strict_json(execution)
    try:
        from tools import policy_cost_calibration_v19 as calibration_tool
    except ImportError:
        import policy_cost_calibration_v19 as calibration_tool  # type: ignore
    if value.get("schema") != calibration_tool.SCHEMA:
        raise EvidenceError("calibration result schema drift")
    if calibration.read_bytes() != calibration_tool._canonical_json_bytes(value):
        raise EvidenceError("calibration result is not canonical JSON")
    claimed_digest = value.get("calibration_sha256")
    digest_payload = dict(value)
    digest_payload.pop("calibration_sha256", None)
    if not isinstance(claimed_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", claimed_digest) or claimed_digest != hashlib.sha256(
                calibration_tool._canonical_json_bytes(digest_payload)
            ).hexdigest():
        raise EvidenceError("calibration result digest is invalid")

    passed = value.get("calibration_passed")
    if not isinstance(passed, bool):
        raise EvidenceError("calibration result lacks an explicit pass decision")
    expected_status = "passed" if passed else "failed_model_adequacy"
    expected_reason = None if passed else CALIBRATION_FAILURE_REASON
    if value.get("status") != expected_status or value.get("deployment") != {
        "permitted": passed,
        "reason": expected_reason,
    }:
        raise EvidenceError("calibration deployment disposition is inconsistent")

    observation_digest = value.get("observation_input_sha256")
    if not isinstance(observation_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", observation_digest) is None:
        raise EvidenceError("calibration observation-input binding is invalid")
    design = value.get("campaign_design")
    allocation = design.get("allocation_binding") \
        if isinstance(design, dict) else None
    train_evidence = design.get("evidence_binding") \
        if isinstance(design, dict) else None
    expected_train_fields = {
        "required", "validated", "schema", "stage", "raw_verified",
        "execution_sha256", "evaluation_sha256",
        "evaluation_header_sha256", "allocation_sha256",
        "train_input_sha256",
    }
    expected_allocation_fields = {
        "required", "validated", "allocation_manifest_sha256",
        "source_reservoir_sha256", "eligible_pair_commitment_sha256",
        "allocation_rule_sha256", "selected_units",
    }
    if not isinstance(allocation, dict) or \
            set(allocation) != expected_allocation_fields or \
            allocation.get("required") is not True or \
            allocation.get("validated") is not True or \
            allocation.get("selected_units") != TRAIN_RECORDS or \
            not isinstance(train_evidence, dict) or \
            set(train_evidence) != expected_train_fields or \
            train_evidence.get("required") is not True or \
            train_evidence.get("validated") is not True or \
            train_evidence.get("schema") != \
            "lc-policy-cost-v19-train-evidence-binding-v1" or \
            train_evidence.get("stage") != "TRAIN" or \
            train_evidence.get("raw_verified") is not True or \
            train_evidence.get("execution_sha256") != sha256(execution):
        raise EvidenceError("calibration campaign input binding is invalid")
    for field in (
        "allocation_manifest_sha256", "source_reservoir_sha256",
        "eligible_pair_commitment_sha256", "allocation_rule_sha256",
    ):
        if not isinstance(allocation.get(field), str) or re.fullmatch(
                r"[0-9a-f]{64}", allocation[field]) is None:
            raise EvidenceError(f"calibration allocation {field} is invalid")
    for field in (
        "execution_sha256", "evaluation_sha256",
        "evaluation_header_sha256", "allocation_sha256",
        "train_input_sha256",
    ):
        if not isinstance(train_evidence.get(field), str) or re.fullmatch(
                r"[0-9a-f]{64}", train_evidence[field]) is None:
            raise EvidenceError(f"calibration TRAIN {field} is invalid")

    adequacy = value.get("model_adequacy")
    required_adequacy_fields = {
        "required", "evaluated", "authoritative_pre_select_gate", "loss",
        "hyperparameter_tuning", "nested_outer_folds", "deployable_model",
        "comparators", "challengers", "mean_group_losses",
        "gap_over_beta_only_comparison",
        "pooled_beta_only_minus_gap_loss_reduction",
        "relative_improvement_over_deployable",
        "maximum_allowed_relative_improvement", "rule",
        "richer_model_check_passed", "passed",
    }
    if not isinstance(adequacy, dict) or \
            set(adequacy) != required_adequacy_fields or \
            adequacy.get("required") is not True or \
            adequacy.get("evaluated") is not True or \
            adequacy.get("authoritative_pre_select_gate") is not True or \
            adequacy.get("passed") is not passed:
        raise EvidenceError("calibration model-adequacy evidence is incomplete")
    nested_folds = adequacy.get("nested_outer_folds")
    if not isinstance(nested_folds, list) or len(nested_folds) != 5 or \
            {item.get("outer_fold") for item in nested_folds
             if isinstance(item, dict)} != set(range(5)):
        raise EvidenceError("calibration nested adequacy folds are incomplete")
    gap_check = adequacy.get("gap_over_beta_only_comparison")
    pooled_check = adequacy.get(
        "pooled_beta_only_minus_gap_loss_reduction"
    )
    if not isinstance(gap_check, dict) or \
            not isinstance(gap_check.get("passed"), bool) or \
            not isinstance(pooled_check, dict) or \
            not isinstance(pooled_check.get("passed"), bool) or \
            not isinstance(adequacy.get("richer_model_check_passed"), bool):
        raise EvidenceError("calibration adequacy sub-decisions are incomplete")
    gap_lcb = _require_number(
        gap_check.get("one_sided_lcb_z_1_645"),
        "calibration gap-over-beta LCB",
    )
    points = pooled_check.get("points")
    improvements = adequacy.get("relative_improvement_over_deployable")
    maximum_improvement = _require_number(
        adequacy.get("maximum_allowed_relative_improvement"),
        "calibration richer-model limit",
    )
    if not isinstance(points, dict) or set(points) != {
            "pair:different_core", "pair:same_core_draw", "ply:early_0_15",
            "ply:mid_16_39", "ply:late_40_63",
            } or not isinstance(improvements, dict) or set(improvements) != {
                "full_gap", "round_specific", "cell_saturated",
            } or not isinstance(adequacy.get("mean_group_losses"), dict) or \
            set(adequacy["mean_group_losses"]) != {
                "identity_search", "beta_only", "draw_only_gap",
                "full_gap", "round_specific", "cell_saturated",
            }:
        raise EvidenceError("calibration adequacy decision details are absent")
    for name, item in adequacy["mean_group_losses"].items():
        _require_number(item, f"calibration mean group loss {name}")
    point_values = [
        _require_number(item, f"calibration adequacy point {name}")
        for name, item in points.items()
    ]
    improvement_values = [
        _require_number(item, f"calibration richer improvement {name}")
        for name, item in improvements.items()
    ]
    expected_gap_pass = gap_lcb > 0.0
    expected_pooled_pass = all(item >= 0.0 for item in point_values)
    expected_richer_pass = all(
        item <= maximum_improvement + 1.0e-15
        for item in improvement_values
    )
    expected_adequacy_pass = (
        expected_gap_pass and expected_pooled_pass and expected_richer_pass
    )
    if gap_check["passed"] is not expected_gap_pass or \
            pooled_check["passed"] is not expected_pooled_pass or \
            adequacy["richer_model_check_passed"] is not expected_richer_pass or \
            passed is not expected_adequacy_pass:
        raise EvidenceError("calibration adequacy decision is not reproducible")

    has_schedule = "schedule" in value
    has_thresholds = "derived_gap_thresholds" in value
    if passed:
        schedule = value.get("schedule")
        if not isinstance(schedule, dict) or \
                schedule.get("ply_anchors") != list(ANCHORS) or \
                not has_thresholds:
            raise EvidenceError("passed calibration lacks deployable outputs")
        for field in ("beta_search", "alpha_core", "alpha_draw"):
            values = schedule.get(field)
            if not isinstance(values, list) or len(values) != len(ANCHORS) or \
                    any(isinstance(item, bool) or not isinstance(item, (int, float))
                        or not math.isfinite(float(item)) for item in values):
                raise EvidenceError(
                    f"passed calibration {field} schedule is invalid"
                )
        try:
            validated_schedule = calibration_tool.PolicyCostSchedule(
                anchors=tuple(schedule["ply_anchors"]),
                beta_search=tuple(schedule["beta_search"]),
                alpha_core=tuple(schedule["alpha_core"]),
                alpha_draw=tuple(schedule["alpha_draw"]),
            ).validated()
        except (KeyError, TypeError, ValueError, calibration_tool.CalibrationError) \
                as exc:
            raise EvidenceError(
                f"passed calibration schedule violates constraints: {exc}"
            ) from exc
        if any(item != 0.0 for item in schedule["alpha_core"]):
            raise EvidenceError(
                "passed calibration violates the exact-zero action-core cost"
            )
        zero_indices = [
            index for index, anchor in enumerate(ANCHORS)
            if 16 <= anchor < 40
        ]
        if any(
            schedule[field][index] != 0.0
            for field in ("alpha_draw",)
            for index in zero_indices
        ):
            raise EvidenceError(
                "passed calibration violates the exact-zero midgame policy phase"
            )
        if value.get("derived_gap_thresholds") != \
                calibration_tool.derived_gap_threshold_table(
                    validated_schedule
                ):
            raise EvidenceError(
                "passed calibration derived thresholds differ from schedule"
            )
    elif has_schedule or has_thresholds:
        raise EvidenceError(
            "failed calibration exposes a forbidden deployable output"
        )

    return {
        "schema": CALIBRATION_GATE_SCHEMA,
        "calibration_passed": passed,
        "status": expected_status,
        "reason": expected_reason,
        "calibration_file_sha256": sha256(calibration),
        "calibration_payload_sha256": claimed_digest,
        "observation_input_sha256": observation_digest,
        "campaign_input_bindings": {
            "allocation_binding": allocation,
            "evidence_binding": train_evidence,
        },
        "model_adequacy_sha256": _canonical_payload_digest(adequacy),
    }


def actor_manifest(execution: Path, calibration: Path, selection: Path,
                   artifact: Path, preselect_artifact: Path, actor_artifact_path: str) -> dict[str, Any]:
    bound = strict_json(execution)
    schedule = strict_json(calibration)
    selected_value = strict_json(selection)
    config_id, floor, ply_lo = _sealed_campaign_selection(selected_value, require_evidence=True)
    selection_evidence = selected_value.get("campaign_evidence_binding")
    if _validated_holdout_evidence(selection_evidence, "SELECT").get("calibration_sha256") != sha256(calibration) or \
            selection_evidence.get("policy_cost_sha256") != sha256(preselect_artifact):
        raise EvidenceError("actor manifest SELECT evidence binding drift")
    try:
        from tools import policy_cost_calibration_v19 as calibration_tool
    except ImportError:
        import policy_cost_calibration_v19 as calibration_tool  # type: ignore
    claimed_calibration_sha = schedule.get("calibration_sha256")
    calibration_payload = dict(schedule)
    calibration_payload.pop("calibration_sha256", None)
    if not isinstance(claimed_calibration_sha, str) or \
            claimed_calibration_sha != hashlib.sha256(
                calibration_tool._canonical_json_bytes(calibration_payload)
            ).hexdigest():
        raise EvidenceError("calibration result digest is invalid")
    allocation_binding = schedule.get("campaign_design", {}).get(
        "allocation_binding"
    ) if isinstance(schedule.get("campaign_design"), dict) else None
    train_evidence = schedule.get("campaign_design", {}).get(
        "evidence_binding"
    ) if isinstance(schedule.get("campaign_design"), dict) else None
    model_adequacy = schedule.get("model_adequacy")
    if schedule.get("calibration_passed") is not True or \
            schedule.get("status") != "passed" or \
            schedule.get("deployment") != {
                "permitted": True, "reason": None,
            } or \
            not isinstance(allocation_binding, dict) or \
            allocation_binding.get("required") is not True or \
            allocation_binding.get("validated") is not True or \
            not isinstance(train_evidence, dict) or train_evidence.get("required") is not True or \
            train_evidence.get("validated") is not True or train_evidence.get("stage") != "TRAIN" or \
            train_evidence.get("raw_verified") is not True or \
            train_evidence.get("execution_sha256") != sha256(execution) or \
            train_evidence.get("train_input_sha256") is None or \
            not isinstance(model_adequacy, dict) or \
            model_adequacy.get("required") is not True or \
            model_adequacy.get("evaluated") is not True or \
            model_adequacy.get("authoritative_pre_select_gate") is not True or \
            model_adequacy.get("passed") is not True:
        raise EvidenceError("calibration did not pass its authoritative pre-SELECT gate")
    try:
        from tools.policy_cost_artifact_v19 import read_policy_cost
    except ImportError:
        from policy_cost_artifact_v19 import read_policy_cost  # type: ignore
    parsed = read_policy_cost(artifact)
    preselect = read_policy_cost(preselect_artifact)
    controller = parsed["controller"]
    subject = bound.get("subject")
    if not isinstance(subject, dict) or \
            parsed.get("source_seed") != int(POLICY_COST_SOURCE_SEED) or \
            parsed.get("strict_probability_floor") != \
            float.fromhex(POLICY_COST_EPSILON) or \
            parsed.get("primary_z") != 3.5 or parsed.get("fresh_z") != 2.58 or \
            controller["objective"] != subject.get("objective") or \
            controller["dets"] != 800 or controller["confirm_dets"] != 800 or \
            controller["root_width"] != 5 or controller["action_core_count"] != 3 or \
            controller["min_cand"] != 1 or not math.isclose(
                _require_number(controller["cand_floor"], "selected controller floor"), floor,
                rel_tol=0.0, abs_tol=1e-7
            ) or \
            controller["ply_lo"] != ply_lo or controller["ply_hi"] != 0 or \
            controller["override_k"] != 3.5 or controller["override_min"] != 0.0:
        raise EvidenceError("selected .lcpc controller binding drift")
    if schedule.get("schema") != "lc-policy-cost-calibration-v2" or \
            schedule.get("schedule", {}).get("ply_anchors") != list(ANCHORS):
        raise EvidenceError("calibration result schema/anchors drift")
    calibrated_beta = schedule.get("schedule", {}).get("beta_search")
    calibrated_action = schedule.get("schedule", {}).get("alpha_core")
    calibrated_draw = schedule.get("schedule", {}).get("alpha_draw")
    if parsed.get("beta") != calibrated_beta or \
            parsed.get("alpha_action") != calibrated_action or \
            parsed.get("alpha_draw") != calibrated_draw:
        raise EvidenceError("policy-cost artifact schedule differs from calibration")
    # SELECT is bound to a fixed preselect .01/0 LCPC.  The final actor may
    # change only the selected runtime floor/onset; every model/schedule byte
    # that can affect the raw evidence must stay identical.
    if preselect.get("beta") != parsed.get("beta") or \
            preselect.get("alpha_action") != parsed.get("alpha_action") or \
            preselect.get("alpha_draw") != parsed.get("alpha_draw") or \
            preselect.get("source_seed") != parsed.get("source_seed") or \
            preselect.get("strict_probability_floor") != parsed.get("strict_probability_floor") or \
            preselect.get("primary_z") != parsed.get("primary_z") or \
            preselect.get("fresh_z") != parsed.get("fresh_z"):
        raise EvidenceError("preselect/final LCPC schedule drift")
    pre_controller = dict(preselect["controller"])
    final_controller = dict(parsed["controller"])
    for field in ("cand_floor",):
        pre_controller.pop(field, None); final_controller.pop(field, None)
    if pre_controller != final_controller or not math.isclose(
            _require_number(preselect["controller"].get("cand_floor"), "preselect floor"),
                             0.01, rel_tol=0.0, abs_tol=1e-7) or \
            preselect["controller"].get("ply_lo") != 0:
        raise EvidenceError("preselect/final LCPC controller drift")
    maintained = subject.get("maintained_actor")
    parsed_maintained = parse_maintained_actor(maintained)
    candidate = policy_cost_actor(
        parsed_maintained, artifact_path=actor_artifact_path,
        floor=floor, ply_lo=ply_lo,
    )
    return {
        "schema": "lc-policy-cost-v19-actor-manifest-v1",
        "selected_configuration": {
            "id": config_id, "policy_floor": floor, "ply_lo": ply_lo,
        },
        "maintained_actor": maintained,
        "candidate_actor": candidate,
        "policy_cost_artifact": {
            "path": actor_artifact_path,
            "sha256": sha256(artifact),
            "size": artifact.stat().st_size,
            "content_fingerprint": parsed["content_fingerprint"],
        },
        "preselect_policy_cost_artifact": {
            "sha256": sha256(preselect_artifact),
            "content_fingerprint": preselect["content_fingerprint"],
        },
        "calibration_sha256": sha256(calibration),
        "selection_sha256": sha256(selection),
        "campaign_evidence_binding": dict(selection_evidence),
        "train_evidence_binding": dict(train_evidence),
        "legacy_validation_relaxed": False,
        "results": None,
    }


def _panel_gate(panel: Mapping[str, Any], *, stage: str, candidate: str,
                baseline: str, evidence_root: Path) -> tuple[bool, dict[str, Any]]:
    """Reopen and exactly rebuild the raw-backed reciprocal gate decision."""

    expected = {
        "safety": (200, "202806210101", "202806210102", "policy-cost-v19-safety"),
        "final": (2500, "202806220101", "202806220102", "policy-cost-v19-final"),
    }.get(stage)
    if expected is None:
        raise EvidenceError("unknown arena stage")
    pairs, candidate_seed, baseline_seed, provenance = expected
    raw_bindings = _validate_raw_arena_manifest(evidence_root, stage)
    reciprocal_path = panel.get("reciprocal_path")
    if not isinstance(reciprocal_path, str) or not reciprocal_path or \
            Path(reciprocal_path).is_absolute() or ".." in Path(reciprocal_path).parts:
        raise EvidenceError(f"{stage} reciprocal path is unsafe")
    try:
        from tools.gate_actor_panel import evaluate_gate
    except ImportError:
        from gate_actor_panel import evaluate_gate  # type: ignore
    reciprocal_file = evidence_root / reciprocal_path
    reciprocal_snapshot = strict_json(reciprocal_file)
    expected_block_paths = [
        f"{stage}-candidate-first.json", f"{stage}-baseline-first.json"
    ]
    snapshots = reciprocal_snapshot.get("input_block_snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 2 or \
            [item.get("path") if isinstance(item, dict) else None
             for item in snapshots] != expected_block_paths:
        raise EvidenceError(f"{stage} reciprocal block snapshot path drift")
    for relative in expected_block_paths:
        binding(evidence_root, relative)
    try:
        reciprocal, reciprocal_sha = _rebuild_reciprocal_rooted(
            reciprocal_file, 1.645, evidence_root
        )
    except Exception as exc:
        raise EvidenceError(f"{stage} reciprocal reopen/remerge failed: {exc}") from exc
    if reciprocal.get("candidate") != candidate or reciprocal.get("baseline") != baseline or \
            reciprocal.get("provenance") != provenance:
        raise EvidenceError(f"{stage} reciprocal identity/provenance drift")
    blocks = reciprocal.get("blocks")
    expected_blocks = (
        (candidate, baseline, candidate_seed),
        (baseline, candidate, baseline_seed),
    )
    if not isinstance(blocks, list) or len(blocks) != 2:
        raise EvidenceError(f"{stage} reciprocal blocks drift")
    for index, (block, identity) in enumerate(zip(blocks, expected_blocks)):
        metadata = block.get("metadata") if isinstance(block, dict) else None
        if not isinstance(metadata, dict) or block.get("pair_start") != "0" or \
                block.get("pair_count") != pairs or \
                metadata.get("agent_a") != identity[0] or \
                metadata.get("agent_b") != identity[1] or \
                metadata.get("seed") != identity[2] or \
                metadata.get("rounds") != 3 or \
                metadata.get("provenance") != provenance:
            raise EvidenceError(f"{stage} reciprocal block {index} drift")
    sources = [source for block in blocks for source in block["inputs"]]
    expected_paths = {
        f"evidence/{stage}/raw/{name}" for name in raw_bindings
    }
    if {source.get("path") for source in sources} != expected_paths or any(
            source.get("sha256") != raw_bindings.get(
                Path(str(source.get("path"))).name
            ) for source in sources):
        raise EvidenceError(f"{stage} reciprocal raw shard binding drift")
    rebuilt = evaluate_gate(reciprocal, stage, 1.645)
    rebuilt.update({
        "reciprocal_path": reciprocal_path,
        "reciprocal_sha256": reciprocal_sha,
        "candidate": candidate,
        "baseline": baseline,
        "provenance": provenance,
        "pairs_per_orientation": pairs,
        "seeds": {
            "candidate_first": candidate_seed,
            "baseline_first": baseline_seed,
        },
    })
    if dict(panel) != rebuilt:
        raise EvidenceError(f"{stage} gate decision differs from raw-backed recomputation")
    result = rebuilt["candidate_result"]
    score = _require_number(result["match_score"], f"{stage} score")
    margin = _require_number(result["margin_per_game"], f"{stage} margin")
    orientations = result["orientation_match_scores"]
    if stage == "safety":
        criteria = {
            "combined_score": score >= 0.5,
            "combined_margin": margin > 0.0,
            "orientations": all(item >= 0.475 for item in orientations),
        }
    else:
        criteria = {
            "combined_score_lcb": score - 1.645 * result["match_score_pair_clustered_se"] > 0.5,
            "combined_margin_lcb": margin - 1.645 * result["margin_pair_clustered_se"] > 0.0,
            "orientations": all(item > 0.5 for item in orientations),
        }
    criteria_passed = all(criteria.values())
    if rebuilt.get("passed") is not criteria_passed:
        raise EvidenceError(f"{stage} gate/local criterion disagreement")
    return bool(rebuilt.get("passed")) and criteria_passed, criteria


def _validate_raw_arena_manifest(evidence_root: Path, stage: str) -> dict[str, str]:
    """Bind every immutable matrix shard retained with terminal evidence."""

    if stage == "safety":
        starts, per_shard = tuple(range(0, 200, 20)), 20
    elif stage == "final":
        starts, per_shard = tuple(range(0, 2500, 100)), 100
    else:
        raise EvidenceError("unknown raw arena stage")
    path = evidence_root / "evidence" / stage / "raw-shards.json"
    value = strict_json(path)
    if path.read_bytes() != canonical_json(value):
        raise EvidenceError(f"{stage} raw shard manifest is not canonical JSON")
    expected_names = {
        f"raw-{orientation}-{start}.json"
        for orientation in ("candidate-first", "baseline-first")
        for start in starts
    }
    if set(value) != {"schema", "stage", "pairs_per_shard", "files"} or \
            value.get("schema") != "lc-policy-cost-v19-raw-shard-manifest-v1" or \
            value.get("stage") != stage or value.get("pairs_per_shard") != per_shard:
        raise EvidenceError(f"{stage} raw shard manifest schema drift")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(expected_names):
        raise EvidenceError(f"{stage} raw shard manifest count drift")
    seen: set[str] = set()
    bindings: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"} or \
                not isinstance(item.get("path"), str) or \
                Path(item["path"]).name != item["path"] or \
                HEX64.fullmatch(item.get("sha256", "")) is None or \
                isinstance(item.get("size"), bool) or not isinstance(item.get("size"), int) or \
                item["size"] <= 0 or item["path"] in seen:
            raise EvidenceError(f"{stage} raw shard manifest record drift")
        seen.add(item["path"])
        bindings[item["path"]] = item["sha256"]
        raw = evidence_root / "evidence" / stage / "raw" / item["path"]
        sidecar = raw.with_name(raw.name + ".sha256")
        if not raw.is_file() or raw.is_symlink() or raw.stat().st_size != item["size"] or \
                sha256(raw) != item["sha256"] or not sidecar.is_file() or \
                sidecar.is_symlink() or sidecar.read_bytes() != \
                (item["sha256"] + "\n").encode("ascii"):
            raise EvidenceError(f"{stage} raw shard hash binding drift")
    if seen != expected_names:
        raise EvidenceError(f"{stage} raw shard set drift")
    raw_root = evidence_root / "evidence" / stage / "raw"
    if {path.name for path in raw_root.iterdir() if path.is_file()} != \
            expected_names | {f"{name}.sha256" for name in expected_names}:
        raise EvidenceError(f"{stage} raw shard sidecar set drift")
    return bindings


def _validate_raw_evaluation_manifest(
    evidence_root: Path, stage: str
) -> list[Path]:
    """Reopen every immutable TRAIN/SELECT/TEST evaluator slice and sidecar."""

    expected_count = {"TRAIN": 216, "SELECT": 192, "TEST": 192}.get(stage)
    if expected_count is None:
        raise EvidenceError("unknown efficacy evidence stage")
    lower = stage.lower()
    manifest_path = evidence_root / "evidence" / lower / "raw-shards.json"
    manifest = strict_json(manifest_path)
    if manifest_path.read_bytes() != canonical_json(manifest) or \
            set(manifest) != {"schema", "stage", "files"} or \
            manifest.get("schema") != \
            "lc-policy-cost-v19-raw-evaluation-manifest-v1" or \
            manifest.get("stage") != stage:
        raise EvidenceError(f"{stage} raw evaluation manifest drift")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != expected_count:
        raise EvidenceError(f"{stage} raw evaluation shard count drift")
    raw_root = evidence_root / "evidence" / lower / "raw"
    expected_names = {f"{index}.jsonl" for index in range(expected_count)}
    seen: set[str] = set()
    paths: list[Path] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"} or \
                not isinstance(item.get("path"), str) or \
                Path(item["path"]).name != item["path"] or \
                HEX64.fullmatch(str(item.get("sha256", ""))) is None or \
                isinstance(item.get("size"), bool) or \
                not isinstance(item.get("size"), int) or item["size"] <= 0 or \
                item["path"] in seen:
            raise EvidenceError(f"{stage} raw evaluation manifest record drift")
        seen.add(item["path"])
        raw = raw_root / item["path"]
        sidecar = raw.with_name(raw.name + ".sha256")
        if not raw.is_file() or raw.is_symlink() or \
                raw.stat().st_size != item["size"] or \
                sha256(raw) != item["sha256"] or \
                not sidecar.is_file() or sidecar.is_symlink() or \
                sidecar.read_bytes() != (item["sha256"] + "\n").encode("ascii"):
            raise EvidenceError(f"{stage} raw evaluation shard binding drift")
        paths.append(raw)
    if seen != expected_names or {
        path.name for path in raw_root.iterdir() if path.is_file()
    } != expected_names | {f"{name}.sha256" for name in expected_names}:
        raise EvidenceError(f"{stage} raw evaluation shard set drift")
    return sorted(paths, key=lambda path: int(path.stem))


def _validate_source_free_execution_snapshot(
    execution: Path, evidence_root: Path
) -> dict[str, Any]:
    """Revalidate every execution field whose authority survives transport."""

    value = strict_json(execution)
    if execution.read_bytes() != canonical_json(value) or set(value) != {
        "schema", "artifact_kind", "status", "source_parent_commit",
        "source_parent_tree", "bindings", "build", "subject",
        "fixed_budgets", "fixed_seeds", "results",
    } or value.get("schema") != EXECUTION_SCHEMA or \
            value.get("artifact_kind") != "locked_policy_cost_v19_execution" or \
            value.get("status") != \
            "launch_bound_before_discovery_or_any_search_truth_label" or \
            value.get("results") is not None or \
            HEX40.fullmatch(str(value.get("source_parent_commit", ""))) is None or \
            HEX40.fullmatch(str(value.get("source_parent_tree", ""))) is None:
        raise EvidenceError("source-free execution snapshot identity drift")
    if value.get("build") != {
        "runner": "ubuntu-24.04", "compiler": COMPILER,
        "compiler_semantic_version_command": COMPILER_VERSION_COMMAND,
        "required_compiler_semantic_version": COMPILER_VERSION,
        "cflags": CFLAGS, "ldflags": LDFLAGS,
        "binding": (
            "compile exactly once in preflight; source-free SHA-256 transport "
            "thereafter"
        ),
    } or value.get("fixed_budgets") != {
        "discovery_matches": DISCOVERY_MATCHES,
        "train_records": TRAIN_RECORDS, "select_records": HOLDOUT_RECORDS,
        "test_records": HOLDOUT_RECORDS, "configurations": list(CONFIG_IDS),
        "safety_pairs_per_orientation": 200,
        "final_pairs_per_orientation": 2500,
    } or value.get("fixed_seeds") != {
        "discovery": DISCOVERY_SEEDS, "primary": PRIMARY_SEEDS,
        "fresh": FRESH_SEEDS, "truth": TRUTH_SEEDS,
        "maintained": MAINTAINED_SEEDS,
        "calibration_folds": POLICY_COST_SOURCE_SEED,
        "select_bootstrap": "202806150101",
        "safety": {"candidate_first": "202806210101",
                   "baseline_first": "202806210102"},
        "final": {"candidate_first": "202806220101",
                  "baseline_first": "202806220102"},
    }:
        raise EvidenceError("source-free execution frozen protocol drift")
    subject = value.get("subject")
    if not isinstance(subject, dict) or set(subject) != {
        "maintained_actor", "neutral_counterfactual_actor", "objective",
        "train_truth_metric", "root_path", "continuation_path",
        "match_value_path",
    }:
        raise EvidenceError("source-free execution subject schema drift")
    actor = parse_maintained_actor(subject.get("maintained_actor"))
    if subject != {
        "maintained_actor": actor["spec"],
        "neutral_counterfactual_actor": neutral_actor(actor),
        "objective": actor["objective"],
        "train_truth_metric": actor["truth_metric"],
        "root_path": actor["root_path"],
        "continuation_path": actor["continuation_path"],
        "match_value_path": actor["match_value_path"],
    }:
        raise EvidenceError("source-free execution maintained actor drift")

    bindings_value = value.get("bindings")
    if not isinstance(bindings_value, dict) or set(bindings_value) != {
        "plan", "workflow", "helper", "objective3_prerequisite",
        "exact17", "exact17_exclusion_manifests", "predecessor_attempt",
    }:
        raise EvidenceError("source-free execution binding schema drift")

    def require_record(
        record: Any, original: str, retained: str,
        expected_digest: str | None = None,
    ) -> None:
        if not isinstance(record, dict) or set(record) != {
            "path", "sha256", "size"
        } or record.get("path") != original:
            raise EvidenceError("source-free execution file record drift")
        actual = binding(evidence_root, retained)
        if record.get("sha256") != actual["sha256"] or \
                (expected_digest is not None and
                 record.get("sha256") != expected_digest) or \
                record.get("size") != actual["size"]:
            raise EvidenceError("source-free execution file binding drift")

    require_record(bindings_value["plan"], PLAN_PATH, "bindings/plan.json")
    validate_plan(strict_json(evidence_root / "bindings/plan.json"))
    require_record(
        bindings_value["workflow"], WORKFLOW_PATH, "bindings/workflow.yml"
    )
    require_record(bindings_value["helper"], HELPER_PATH, HELPER_PATH)
    exact17_records = bindings_value.get("exact17")
    exclusion_records = bindings_value.get("exact17_exclusion_manifests")
    if not isinstance(exact17_records, list) or \
            len(exact17_records) != len(EXACT17) or \
            not isinstance(exclusion_records, list) or \
            len(exclusion_records) != len(EXCLUSION_BINDINGS):
        raise EvidenceError("source-free execution exact17 binding count drift")
    for record, (original, digest) in zip(exact17_records, EXACT17):
        require_record(
            record, original, f"bindings/exact17/{Path(original).name}", digest
        )
    for record, (original, digest) in zip(
            exclusion_records, EXCLUSION_BINDINGS):
        require_record(
            record, original, f"bindings/exact17/{Path(original).name}", digest
        )
    predecessor_records = bindings_value.get("predecessor_attempt")
    if not isinstance(predecessor_records, list) or \
            len(predecessor_records) != len(PREDECESSOR_ATTEMPT_BINDINGS):
        raise EvidenceError("source-free predecessor binding count drift")
    for record, (original, digest) in zip(
            predecessor_records, PREDECESSOR_ATTEMPT_BINDINGS):
        require_record(
            record, original,
            f"bindings/predecessor/{Path(original).name}", digest,
        )

    prerequisite = bindings_value.get("objective3_prerequisite")
    if not isinstance(prerequisite, dict) or set(prerequisite) != {
        "result", "promotion_gate_passed", "disposition", "actor", "assets",
        "evidence", "terminal_evidence_files",
    } or type(prerequisite.get("promotion_gate_passed")) is not bool or \
            not isinstance(prerequisite.get("terminal_evidence_files"), int) or \
            prerequisite["terminal_evidence_files"] <= 0 or \
            prerequisite.get("actor") != actor:
        raise EvidenceError("source-free objective3 prerequisite snapshot drift")
    result_record = prerequisite.get("result")
    if not isinstance(result_record, dict) or set(result_record) != {
        "path", "sha256", "size"
    } or result_record.get("path") != PREREQUISITE_PATH or \
            HEX64.fullmatch(str(result_record.get("sha256", ""))) is None or \
            not isinstance(result_record.get("size"), int) or \
            result_record["size"] <= 0:
        raise EvidenceError("source-free objective3 result binding drift")
    passed = prerequisite["promotion_gate_passed"]
    if (passed and actor["objective"] != 3) or \
            (not passed and actor["objective"] != 0):
        raise EvidenceError("source-free objective3 disposition/actor drift")
    asset_records = prerequisite.get("assets")
    asset_paths = [actor["root_path"]] + (
        [actor["match_value_path"]] if actor["match_value_path"] else []
    )
    if not isinstance(asset_records, list) or len(asset_records) != len(asset_paths):
        raise EvidenceError("source-free objective3 asset count drift")
    for record, relative in zip(asset_records, asset_paths):
        require_record(record, relative, relative)
    retained_prerequisite = authoritative_prerequisite(
        evidence_root / "bindings/objective3/repo", assets_root=evidence_root
    )
    if retained_prerequisite != prerequisite:
        raise EvidenceError("source-free objective3 evidence/disposition drift")
    return value


def _verify_pre_efficacy_and_allocation_chain(
    evidence_root: Path, execution: Path
) -> dict[str, list[dict[str, Any]]]:
    """Reopen the source-free freeze and reproduce all hash-only allocations."""

    execution_value = _validate_source_free_execution_snapshot(
        execution, evidence_root
    )
    pre_path = evidence_root / "bindings/pre-efficacy.json"
    pre = strict_json(pre_path)
    if pre_path.read_bytes() != canonical_json(pre) or \
            set(pre) != {
                "schema", "artifact_kind", "status", "execution_sha256",
                "source_free_after_preflight", "probe_states_absent",
                "build_identity", "files", "results",
            } or pre.get("schema") != MANIFEST_SCHEMA or \
            pre.get("artifact_kind") != "policy_cost_v19_pre_efficacy_manifest" or \
            pre.get("status") != "frozen_before_first_search_or_truth_label" or \
            pre.get("execution_sha256") != sha256(execution) or \
            pre.get("source_free_after_preflight") is not True or \
            pre.get("probe_states_absent") is not True or \
            pre.get("results") is not None or \
            pre.get("build_identity") != _build_identity(evidence_root):
        raise EvidenceError("terminal pre-efficacy manifest drift")
    files = pre.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceError("terminal pre-efficacy file manifest is absent")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "path", "sha256", "size", "mode"
        } or not isinstance(item.get("path"), str) or \
                HEX64.fullmatch(str(item.get("sha256", ""))) is None or \
                isinstance(item.get("size"), bool) or \
                not isinstance(item.get("size"), int) or item["size"] <= 0 or \
                not isinstance(item.get("mode"), str) or \
                re.fullmatch(r"0[0-7]{3}", item["mode"]) is None or \
                item["path"] in seen:
            raise EvidenceError("terminal pre-efficacy file record drift")
        seen.add(item["path"])
        bound = binding(evidence_root, item["path"])
        path = evidence_root / item["path"]
        if bound["sha256"] != item["sha256"] or \
                bound["size"] != item["size"] or \
                f"{path.stat().st_mode & 0o777:04o}" != item["mode"]:
            raise EvidenceError("terminal pre-efficacy file binding drift")
    build_value = strict_json(evidence_root / "bindings/build-identity.json")
    required = {
        "bin/arena", "bin/build_policy_cost", "bin/policy_cost_dataset",
        "bindings/execution.json", "bindings/plan.json",
        "bindings/workflow.yml", "bindings/build-identity.json",
        "bindings/runtime/requirements.txt",
        "data/champion.bin", "tools/policy_cost_allocate_v19.py",
        "tools/policy_cost_artifact_v19.py", "tools/policy_cost_calibration_v19.py",
        "tools/policy_cost_campaign_v19.py", "tools/policy_cost_selection_v19.py",
        "tools/gate_actor_panel.py", "tools/merge_arena.py",
        "tools/match_value_objective3_v2.py", "tools/flagged_ply_execution.py",
    } | {
        item["wheel"]["path"] for item in build_value["python_packages"]
    } | {
        f"bindings/exact17/{Path(path).name}" for path, _ in EXACT17
    } | {
        f"bindings/exact17/{Path(path).name}"
        for path, _ in EXCLUSION_BINDINGS
    } | {
        f"bindings/predecessor/{Path(path).name}"
        for path, _ in PREDECESSOR_ATTEMPT_BINDINGS
    }
    subject = execution_value["subject"]
    prerequisite = execution_value["bindings"]["objective3_prerequisite"]
    for asset in (subject.get("root_path"), subject.get("match_value_path")):
        if asset is not None:
            required.add(_canonical_relative_path(
                asset, "terminal actor asset"
            ).as_posix())
    for record in [prerequisite["result"], *prerequisite["evidence"]]:
        relative = _canonical_relative_path(
            record["path"], "terminal prerequisite path"
        )
        required.add(f"bindings/objective3/repo/{relative.as_posix()}")
    if not required <= seen:
        raise EvidenceError("terminal pre-efficacy manifest lacks runtime members")

    freeze = evidence_root / "bindings/reservoir-freeze.json"
    for split in ("TRAIN", "SELECT", "TEST"):
        verify_reservoir_freeze(freeze, evidence_root, split)
    try:
        from tools import policy_cost_allocate_v19 as allocation_tool
    except ImportError:
        import policy_cost_allocate_v19 as allocation_tool  # type: ignore
    exclusion_path = evidence_root / \
        "bindings/exact17/policy_cost_v16_exact17_exclusions.txt"
    exclusion_lines = exclusion_path.read_text(encoding="ascii").splitlines()
    if not exclusion_lines or exclusion_lines[0] != \
            "lc-policy-cost-exclusions-v1" or len(exclusion_lines) != 18 or \
            any(HEX64.fullmatch(item) is None for item in exclusion_lines[1:]):
        raise EvidenceError("terminal exact-17 exclusion text drift")
    excluded = set(exclusion_lines[1:])
    champion_sha = sha256(evidence_root / "data/champion.bin")
    exclusion_sha = sha256(exclusion_path)
    bindings_by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage in ("TRAIN", "SELECT", "TEST"):
        lower = stage.lower()
        discovery = evidence_root / f"{lower}-discovery.jsonl"
        reservoir = evidence_root / f"{lower}-reservoir.tsv"
        allocation = evidence_root / f"{lower}-allocation.tsv"
        discovery_sha, reservoir_sha = sha256(discovery), sha256(reservoir)
        rebuilt = allocation_tool.build_manifest(
            discovery, discovery_sha, reservoir, reservoir_sha
        )
        if allocation.read_bytes() != rebuilt:
            raise EvidenceError(f"{stage} allocation differs from hash-only replay")
        discovery_header, _, _ = allocation_tool.discovery(
            discovery, discovery_sha
        )
        _, retained_rows, _ = allocation_tool.reservoir(
            reservoir, reservoir_sha
        )
        if discovery_header.get("net_sha256") != champion_sha or \
                discovery_header.get("exclusion_manifest_sha256") != exclusion_sha:
            raise EvidenceError(f"{stage} discovery model/exclusion binding drift")
        if any(row.get("orbit_sha256") in excluded for row in retained_rows):
            raise EvidenceError(f"{stage} retained reservoir contains exact-17 orbit")
        _verify_native_reservoir_proof(
            evidence_root, stage, allocation, reservoir
        )
        if stage == "TRAIN":
            _, rows = train_allocation_manifest(allocation)
        else:
            _, rows, _ = vector_allocation_manifest(allocation, stage)
        if any(row.get("orbit_sha256") in excluded for row in rows):
            raise EvidenceError(f"{stage} allocation contains an exact-17 orbit")
        bindings_by_stage[stage] = rows
    return bindings_by_stage


def _verify_native_reservoir_proof(
    evidence_root: Path, stage: str, allocation: Path, reservoir: Path
) -> None:
    """Rerun the frozen native full-reservoir origin/firewall verifier."""

    lower = stage.lower()
    proof = evidence_root / f"{lower}-reservoir-proof.json"
    value = strict_json(proof)
    expected_keys = {
        "schema", "split", "allocation_sha256", "reservoir_sha256",
        "source_net_sha256", "exclusion_sha256", "retained_rows",
        "eligible_units", "rejected_by_bound", "state_bytes",
        "excluded_orbits_found", "all_views_native_valid",
        "all_state_hashes_exact", "all_orbits_recomputed",
        "all_policy_masks_exact", "verified_chain_sha256",
    }
    if set(value) != expected_keys or \
            value.get("schema") != "lc-policy-cost-verified-reservoir-v1" or \
            value.get("split") != stage or \
            value.get("allocation_sha256") != sha256(allocation) or \
            value.get("reservoir_sha256") != sha256(reservoir) or \
            value.get("source_net_sha256") != sha256(
                evidence_root / "data/champion.bin"
            ) or value.get("exclusion_sha256") != sha256(
                evidence_root /
                "bindings/exact17/policy_cost_v16_exact17_exclusions.txt"
            ) or value.get("state_bytes") != 174 or \
            value.get("excluded_orbits_found") != 0 or \
            value.get("all_views_native_valid") is not True or \
            value.get("all_state_hashes_exact") is not True or \
            value.get("all_orbits_recomputed") is not True or \
            value.get("all_policy_masks_exact") is not True or \
            HEX64.fullmatch(str(value.get("verified_chain_sha256", ""))) is None:
        raise EvidenceError(f"{stage} native reservoir proof identity drift")
    numeric = ("retained_rows", "eligible_units", "rejected_by_bound")
    if any(isinstance(value.get(key), bool) or not isinstance(value.get(key), int)
           or value[key] < 0 for key in numeric) or \
            value["retained_rows"] <= 0 or \
            value["eligible_units"] != \
            value["retained_rows"] + value["rejected_by_bound"]:
        raise EvidenceError(f"{stage} native reservoir proof census drift")
    if proof.read_bytes() != canonical_json(value, pretty=False):
        raise EvidenceError(f"{stage} native reservoir proof is noncanonical")
    with tempfile.TemporaryDirectory(
            prefix=f"policy-cost-{lower}-reservoir-replay-") as raw:
        rebuilt = Path(raw) / "proof.json"
        command = [
            str(evidence_root / "bin/policy_cost_dataset"),
            "verify-reservoir", "--out", str(rebuilt),
            "--manifest", str(allocation), "--manifest-sha256",
            sha256(allocation), "--reservoir", str(reservoir),
            "--reservoir-sha256", sha256(reservoir), "--net",
            str(evidence_root / "data/champion.bin"), "--exclusions",
            str(evidence_root /
                "bindings/exact17/policy_cost_v16_exact17_exclusions.txt"),
            "--exclusions-sha256", value["exclusion_sha256"],
        ]
        try:
            completed = subprocess.run(
                command, cwd=evidence_root, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False, timeout=7200,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise EvidenceError(
                f"{stage} native reservoir replay failed: {exc}"
            ) from exc
        if completed.returncode != 0 or not rebuilt.is_file() or \
                rebuilt.read_bytes() != proof.read_bytes():
            raise EvidenceError(
                f"{stage} native reservoir proof differs on replay"
            )


def _rebuild_policy_cost_artifact(
    *, evidence_root: Path, calibration: Path, output: Path,
    floor: float, ply_lo: int, execution_value: Mapping[str, Any],
) -> None:
    schedule = strict_json(calibration).get("schedule", {})
    beta_search = schedule.get("beta_search")
    alpha_action = schedule.get("alpha_core")
    alpha_draw = schedule.get("alpha_draw")
    subject = execution_value.get("subject")
    if not isinstance(subject, dict) or not isinstance(beta_search, list) or \
            not isinstance(alpha_action, list) or not isinstance(alpha_draw, list):
        raise EvidenceError("artifact replay lacks execution/calibration schedule")
    with tempfile.TemporaryDirectory(prefix="policy-cost-terminal-rebuild-") as raw:
        rebuilt = Path(raw) / "rebuilt.lcpc"
        command = [
            str(evidence_root / "bin/build_policy_cost"),
            "--root-model", str(subject.get("root_path")),
            "--continuation-model", str(subject.get("continuation_path")),
            "--out", str(rebuilt), "--source-seed", str(POLICY_COST_SOURCE_SEED),
            "--epsilon", POLICY_COST_EPSILON, "--objective",
            str(subject.get("objective")),
        ]
        match_value = subject.get("match_value_path")
        if match_value is not None:
            command.extend(("--match-value", str(match_value)))
        command.extend((
            "--root-symmetries", "20", "--playout-symmetries", "20",
            "--playout-sample", "4", "--playout-prune", "1",
            "--exact-terminal", "1", "--no-belief", "1",
            "--dets", "800", "--confirm-dets", "800",
            "--root-width", "5", "--action-core-count", "3",
            "--min-cand", "1", "--ply-lo", str(ply_lo), "--ply-hi", "0",
            "--discard-guard", "1", "--root-prune", "0",
            "--cand-floor", str(floor), "--override-k", "3.5",
            "--override-min", "0", "--beta",
            ",".join(map(str, beta_search)), "--alpha-action",
            ",".join(map(str, alpha_action)), "--alpha-draw",
            ",".join(map(str, alpha_draw)),
        ))
        try:
            completed = subprocess.run(
                command, cwd=evidence_root, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False, timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise EvidenceError(f"selected artifact replay failed: {exc}") from exc
        if completed.returncode != 0 or not rebuilt.is_file() or \
                rebuilt.read_bytes() != output.read_bytes():
            raise EvidenceError("selected artifact differs from native deterministic replay")


def _verify_terminal_efficacy_evidence(
    *, evidence_root: Path, execution: Path, selection: Path, test: Path,
    actors: Path, selected: Mapping[str, Any], test_result: Mapping[str, Any],
    actor_value: Mapping[str, Any], train_evidence: Mapping[str, Any],
    selected_evidence: Mapping[str, Any], test_evidence: Mapping[str, Any],
) -> None:
    """Deterministically rebuild all three efficacy reductions from raw shards."""

    execution_value = strict_json(execution)
    allocation_bindings = _verify_pre_efficacy_and_allocation_chain(
        evidence_root, execution
    )
    if any(len(allocation_bindings[stage]) != (
        TRAIN_RECORDS if stage == "TRAIN" else HOLDOUT_RECORDS
    ) for stage in ("TRAIN", "SELECT", "TEST")):
        raise EvidenceError("terminal allocation replay record count drift")
    calibration = evidence_root / "calibration.json"
    preselect_artifact = evidence_root / "preselect-policy-cost.lcpc"
    final_artifact = evidence_root / str(
        actor_value.get("policy_cost_artifact", {}).get("path", "")
    )
    if not calibration.is_file() or calibration.is_symlink() or \
            sha256(calibration) != selected_evidence.get("calibration_sha256") or \
            sha256(calibration) != test_evidence.get("calibration_sha256") or \
            sha256(calibration) != actor_value.get("calibration_sha256"):
        raise EvidenceError("terminal calibration file binding drift")
    if not preselect_artifact.is_file() or preselect_artifact.is_symlink() or \
            sha256(preselect_artifact) != selected_evidence.get("policy_cost_sha256"):
        raise EvidenceError("terminal preselect LCPC file binding drift")
    if not final_artifact.is_file() or final_artifact.is_symlink():
        raise EvidenceError("terminal selected LCPC artifact is absent")
    try:
        from tools.policy_cost_artifact_v19 import read_policy_cost
    except ImportError:
        from policy_cost_artifact_v19 import read_policy_cost  # type: ignore
    preselect_parsed = read_policy_cost(preselect_artifact)
    final_parsed = read_policy_cost(final_artifact)
    subject = execution_value.get("subject")
    if not isinstance(subject, dict) or \
            subject.get("root_path") != "data/champion.bin" or \
            subject.get("continuation_path") != "data/champion.bin":
        raise EvidenceError("terminal execution model path is not frozen champion")
    champion_sha = sha256(evidence_root / "data/champion.bin")

    merged_paths: dict[str, Path] = {}
    allocation_paths: dict[str, Path] = {}
    for stage in ("TRAIN", "SELECT", "TEST"):
        lower = stage.lower()
        raw_paths = _validate_raw_evaluation_manifest(evidence_root, stage)
        merged = evidence_root / f"{lower}-evaluation.jsonl"
        allocation = evidence_root / f"{lower}-allocation.tsv"
        if not merged.is_file() or merged.is_symlink() or \
                merged.read_bytes() != merge_evaluation_slices(raw_paths, stage):
            raise EvidenceError(f"{stage} merged evaluation differs from raw shards")
        binding_value = train_evidence if stage == "TRAIN" else (
            selected_evidence if stage == "SELECT" else test_evidence
        )
        if sha256(merged) != binding_value.get("evaluation_sha256") or \
                not allocation.is_file() or allocation.is_symlink() or \
                sha256(allocation) != binding_value.get("allocation_sha256"):
            raise EvidenceError(f"{stage} merged/allocation evidence binding drift")
        header, _, _ = _evaluation(merged, stage)
        artifact_value = (
            None if stage == "TRAIN" else
            preselect_parsed if stage == "SELECT" else final_parsed
        )
        fingerprint_source = preselect_parsed if stage != "TEST" else final_parsed
        controller = fingerprint_source["controller"]
        expected_policy_sha = "none" if artifact_value is None else artifact_value["sha256"]
        expected_policy_fingerprint = (
            "0000000000000000" if artifact_value is None else
            artifact_value["content_fingerprint"]
        )
        expected_match_value = controller["match_value_fingerprint"]
        expected_maintained_match_value = (
            "0000000000000000" if stage == "TRAIN" else expected_match_value
        )
        if header.get("actor_spec") != subject.get("neutral_counterfactual_actor") or \
                header.get("maintained_actor_spec") != (
                    None if stage == "TRAIN" else subject.get("maintained_actor")
                ) or header.get("truth_net_sha256") != champion_sha or \
                header.get("root_net_fingerprint") != \
                controller["root_net_fingerprint"] or \
                header.get("continuation_net_fingerprint") != \
                controller["continuation_net_fingerprint"] or \
                header.get("candidate_match_value_fingerprint") != \
                expected_match_value or \
                header.get("maintained_match_value_fingerprint") != \
                expected_maintained_match_value or \
                header.get("policy_cost_sha256") != expected_policy_sha or \
                header.get("policy_cost_payload_fingerprint") != \
                expected_policy_fingerprint:
            raise EvidenceError(f"{stage} evaluator model/actor/table binding drift")
        merged_paths[stage] = merged
        allocation_paths[stage] = allocation

    train_payload = train_input(
        merged_paths["TRAIN"], execution, allocation_paths["TRAIN"]
    )
    train_input_path = evidence_root / "train-input.jsonl"
    if not train_input_path.is_file() or train_input_path.is_symlink() or \
            train_input_path.read_bytes() != train_payload or \
            sha256(train_input_path) != train_evidence.get("train_input_sha256"):
        raise EvidenceError("TRAIN converted input evidence binding drift")
    rebuilt_train_binding = train_evidence_binding(
        merged_paths["TRAIN"], execution, allocation_paths["TRAIN"], train_payload
    )
    train_binding_path = evidence_root / "train-evidence.json"
    allocation_manifest_path = evidence_root / "train-allocation.json"
    train_manifest, _ = train_allocation_manifest(allocation_paths["TRAIN"])
    if not train_binding_path.is_file() or \
            train_binding_path.read_bytes() != canonical_json(
                rebuilt_train_binding, pretty=False
            ) or dict(train_evidence) != rebuilt_train_binding or \
            not allocation_manifest_path.is_file() or \
            allocation_manifest_path.read_bytes() != canonical_json(
                train_manifest, pretty=False
            ):
        raise EvidenceError("TRAIN derived binding/manifest drift")
    try:
        from tools import policy_cost_calibration_v19 as calibration_tool
    except ImportError:
        import policy_cost_calibration_v19 as calibration_tool  # type: ignore
    rebuilt_calibration = calibration_tool.calibrate_policy_cost(
        calibration_tool.read_observation_jsonl(train_input_path),
        calibration_tool.FitConfig(require_campaign_design=True),
        train_manifest, rebuilt_train_binding,
    )
    if calibration.read_bytes() != rebuilt_calibration.canonical_json():
        raise EvidenceError("calibration differs from deterministic TRAIN reduction")
    _, selected_floor, selected_ply = _sealed_campaign_selection(
        selected, require_evidence=True
    )
    _rebuild_policy_cost_artifact(
        evidence_root=evidence_root, calibration=calibration,
        output=preselect_artifact, floor=0.01, ply_lo=0,
        execution_value=execution_value,
    )
    _rebuild_policy_cost_artifact(
        evidence_root=evidence_root, calibration=calibration,
        output=final_artifact, floor=selected_floor, ply_lo=selected_ply,
        execution_value=execution_value,
    )

    rebuilt_select_input = holdout_input(
        merged_paths["SELECT"], allocation_paths["SELECT"], "SELECT",
        calibration=calibration, policy_cost=preselect_artifact,
        execution=execution, verify_raw=True,
    )
    select_input_path = evidence_root / "select-input.json"
    select_input_bytes = canonical_json(rebuilt_select_input, pretty=False)
    if not select_input_path.is_file() or select_input_path.is_symlink() or \
            select_input_path.read_bytes() != select_input_bytes:
        raise EvidenceError("SELECT converted input differs from raw evidence")

    try:
        from tools import policy_cost_selection_v19 as inference
    except ImportError:
        import policy_cost_selection_v19 as inference  # type: ignore
    rebuilt_selection = inference.select_configuration(
        rebuilt_select_input["rows"], rebuilt_select_input["discovery_manifest"],
        rebuilt_select_input["campaign_evidence_binding"],
    )
    if rebuilt_selection != dict(selected) or \
            selection.read_bytes() != \
            inference.canonical_json_bytes(rebuilt_selection):
        raise EvidenceError("SELECT result differs from deterministic raw reduction")
    rebuilt_actor = actor_manifest(
        execution, calibration, selection, final_artifact, preselect_artifact,
        str(actor_value["policy_cost_artifact"]["path"]),
    )
    if rebuilt_actor != dict(actor_value) or \
            actors.read_bytes() != canonical_json(rebuilt_actor):
        raise EvidenceError("actor manifest differs from rebuilt SELECT artifacts")

    rebuilt_test_input = holdout_input(
        merged_paths["TEST"], allocation_paths["TEST"], "TEST", selection,
        calibration=calibration, policy_cost=final_artifact, actors=actors,
        execution=execution, verify_raw=True,
    )
    test_input_path = evidence_root / "test-input.json"
    test_input_bytes = canonical_json(rebuilt_test_input, pretty=False)
    if not test_input_path.is_file() or test_input_path.is_symlink() or \
            test_input_path.read_bytes() != test_input_bytes:
        raise EvidenceError("TEST converted input differs from raw evidence")
    rebuilt_test_result = inference.test_selected_configuration(
        rebuilt_test_input["rows"],
        rebuilt_test_input["discovery_post_stratum_weights"],
        dict(selected), rebuilt_test_input["discovery_manifest"],
        rebuilt_test_input["campaign_evidence_binding"],
    )
    if rebuilt_test_result != dict(test_result) or \
            test.read_bytes() != \
            inference.canonical_json_bytes(rebuilt_test_result) or \
            strict_json(test) != rebuilt_test_result:
        raise EvidenceError("TEST result differs from deterministic raw reduction")


def evidence_manifest(root: Path, excluded: Path | None = None) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise EvidenceError("terminal evidence root must be a real directory")
    excluded_resolved = excluded.resolve() if excluded is not None else None
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise EvidenceError(f"terminal evidence contains symlink: {path}")
        if not path.is_file() or (
            excluded_resolved is not None and path.resolve() == excluded_resolved
        ) or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(root).as_posix()
        rows.append({"path": relative, "sha256": sha256(path), "size": path.stat().st_size})
    if not rows:
        raise EvidenceError("terminal evidence tree is empty")
    return rows


def terminal_result(execution: Path, selection: Path, test: Path,
                    actors: Path, safety: Path | None, final: Path | None,
                    evidence_root: Path, output: Path,
                    retained_baseline_reason: str | None = None) -> dict[str, Any]:
    if not evidence_root.is_dir() or evidence_root.is_symlink():
        raise EvidenceError("terminal evidence root must be a real directory")
    required_members = (
        (execution, "bindings/execution.json"),
        (selection, "select-result.json"),
        (test, "test-result.json"),
        (actors, "actor-manifest.json"),
    )
    optional_members = (
        (safety, "safety-gate.json"), (final, "final-gate.json")
    )
    for supplied, relative in required_members + tuple(
            item for item in optional_members if item[0] is not None):
        if supplied != evidence_root / relative:
            raise EvidenceError("terminal argument is not its canonical evidence member")
        binding(evidence_root, relative)
    if output != evidence_root / "policy-cost-v19-result.json":
        raise EvidenceError("terminal result output path is not canonical")
    bound = strict_json(execution)
    selected = strict_json(selection)
    test_result = strict_json(test)
    actor_value = strict_json(actors)
    config_id, floor, ply_lo = _sealed_campaign_selection(selected, require_evidence=True)
    selected_evidence = _validated_holdout_evidence(
        selected.get("campaign_evidence_binding"), "SELECT"
    )
    if selected_evidence.get("execution_sha256") != sha256(execution):
        raise EvidenceError("SELECT execution evidence binding drift")
    try:
        from tools.policy_cost_selection_v19 import verify_result_digest
    except ImportError:
        from policy_cost_selection_v19 import verify_result_digest  # type: ignore
    if test_result.get("schema") != "lc-policy-cost-test-result-v1" or \
            test_result.get("stage") != "TEST" or \
            not verify_result_digest(test_result) or \
            test_result.get("selection_payload_sha256") != \
            selected.get("canonical_payload_sha256") or \
            test_result.get("selected") != selected.get("selected") or \
            test_result.get("campaign_discovery_binding", {}).get("required") is not True or \
            test_result.get("campaign_discovery_binding", {}).get("validated") is not True or \
            type(test_result.get("passed")) is not bool:
        raise EvidenceError("TEST result is not bound to sole SELECT winner")
    test_evidence = test_result.get("campaign_evidence_binding")
    if _validated_holdout_evidence(test_evidence, "TEST").get("selection_sha256") != sha256(selection) or \
            test_evidence.get("actor_manifest_sha256") != sha256(actors):
        raise EvidenceError("TEST evidence binding drift")
    for field in ("execution_sha256", "calibration_sha256"):
        if test_evidence.get(field) != selected_evidence.get(field):
            raise EvidenceError("SELECT/TEST evidence substitution drift")
    if actor_value.get("schema") != "lc-policy-cost-v19-actor-manifest-v1" or \
            actor_value.get("legacy_validation_relaxed") is not False or \
            actor_value.get("results") is not None or \
            actor_value.get("selected_configuration") != selected.get("selected") or \
            actor_value.get("selection_sha256") != sha256(selection) or \
            actor_value.get("campaign_evidence_binding") != selected.get("campaign_evidence_binding"):
        raise EvidenceError("actor manifest SELECT binding drift")
    train_evidence = actor_value.get("train_evidence_binding")
    if not isinstance(train_evidence, dict) or train_evidence.get("schema") != \
            "lc-policy-cost-v19-train-evidence-binding-v1" or \
            train_evidence.get("stage") != "TRAIN" or \
            train_evidence.get("raw_verified") is not True or \
            train_evidence.get("execution_sha256") != sha256(execution) or any(
                HEX64.fullmatch(str(train_evidence.get(field, ""))) is None
                for field in ("evaluation_sha256", "evaluation_header_sha256",
                              "allocation_sha256", "train_input_sha256")):
        raise EvidenceError("actor manifest TRAIN evidence binding drift")
    candidate = actor_value.get("candidate_actor")
    baseline = actor_value.get("maintained_actor")
    if not isinstance(candidate, str) or not isinstance(baseline, str) or \
            baseline != bound.get("subject", {}).get("maintained_actor"):
        raise EvidenceError("terminal actor manifest identity drift")
    artifact = actor_value.get("policy_cost_artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) or \
            artifact["path"] not in candidate or Path(artifact["path"]).is_absolute() or \
            ".." in Path(artifact["path"]).parts:
        raise EvidenceError("terminal policy-cost artifact identity drift")
    artifact_path = evidence_root / artifact["path"]
    if not artifact_path.is_file() or artifact.get("sha256") != sha256(artifact_path) or \
            artifact.get("size") != artifact_path.stat().st_size:
        raise EvidenceError("terminal policy-cost artifact binding drift")
    preselect_artifact = actor_value.get("preselect_policy_cost_artifact")
    if not isinstance(preselect_artifact, dict) or \
            selected_evidence.get("policy_cost_sha256") != preselect_artifact.get("sha256") or \
            selected_evidence.get("policy_cost_content_fingerprint") != preselect_artifact.get("content_fingerprint") or \
            test_evidence.get("policy_cost_sha256") != artifact.get("sha256") or \
            test_evidence.get("policy_cost_content_fingerprint") != artifact.get("content_fingerprint"):
        raise EvidenceError("terminal SELECT/TEST LCPC artifact binding drift")
    parsed_maintained = parse_maintained_actor(baseline)
    if candidate != policy_cost_actor(
            parsed_maintained, artifact_path=artifact["path"], floor=floor, ply_lo=ply_lo):
        raise EvidenceError("terminal candidate actor/configuration drift")
    _verify_terminal_efficacy_evidence(
        evidence_root=evidence_root, execution=execution, selection=selection,
        test=test, actors=actors, selected=selected, test_result=test_result,
        actor_value=actor_value, train_evidence=train_evidence,
        selected_evidence=selected_evidence, test_evidence=test_evidence,
    )
    if retained_baseline_reason is not None and (
            not retained_baseline_reason or not retained_baseline_reason.isascii() or
            "\n" in retained_baseline_reason or "\r" in retained_baseline_reason):
        raise EvidenceError("retained-baseline reason is malformed")
    test_passed = bool(test_result["passed"])
    safety_passed = False
    final_passed = False
    safety_criteria: dict[str, Any] | None = None
    final_criteria: dict[str, Any] | None = None
    used_retained_reason = False
    if test_passed:
        if safety is None:
            if retained_baseline_reason is None:
                raise EvidenceError("passing TEST lacks safety evidence")
            safety_criteria = {"incomplete": retained_baseline_reason}
            used_retained_reason = True
        else:
            safety_value = strict_json(safety)
            safety_passed, safety_criteria = _panel_gate(
                safety_value, stage="safety", candidate=candidate, baseline=baseline,
                evidence_root=evidence_root,
            )
            if safety_passed:
                if final is None:
                    if retained_baseline_reason is None:
                        raise EvidenceError("passing safety lacks final evidence")
                    final_criteria = {"incomplete": retained_baseline_reason}
                    used_retained_reason = True
                else:
                    final_value = strict_json(final)
                    final_passed, final_criteria = _panel_gate(
                        final_value, stage="final", candidate=candidate, baseline=baseline,
                        evidence_root=evidence_root,
                    )
            elif final is not None:
                raise EvidenceError("final evidence exists after failed safety")
    elif safety is not None or final is not None:
        raise EvidenceError("arena evidence exists after failed TEST")
    elif retained_baseline_reason is not None:
        # A negative TEST verdict is itself a complete, canonical campaign
        # disposition.  The workflow records why the downstream arena gates
        # were not opened; accepting that reason must not be confused with an
        # incomplete positive TEST path.
        used_retained_reason = True
    if retained_baseline_reason is not None and not used_retained_reason:
        raise EvidenceError("retained-baseline reason does not match incomplete gate")
    promotion = test_passed and safety_passed and final_passed
    result = {
        "schema": RESULT_SCHEMA,
        "artifact_kind": "policy_cost_v19_authoritative_result_recommendation",
        "status": "complete",
        "selected_configuration": selected["selected"],
        "test_passed": test_passed,
        "safety_passed": safety_passed,
        "final_passed": final_passed,
        "promotion_gate_passed": promotion,
        "maintained_actor": candidate if promotion else baseline,
        "candidate_actor": candidate,
        "baseline_actor": baseline,
        "locked_validation_relaxed": False,
        "retained_baseline_reason": retained_baseline_reason,
        "commented_probe_used_for_training_or_selection": False,
        "criteria": {
            "test": test_result.get("criteria"),
            "safety": safety_criteria,
            "final": final_criteria,
        },
        "bindings": {
            "execution_sha256": sha256(execution),
            "selection_sha256": sha256(selection),
            "test_sha256": sha256(test),
            "actor_manifest_sha256": sha256(actors),
        },
        "evidence": evidence_manifest(evidence_root, excluded=output),
    }
    return result


def infrastructure_retain_result(
    execution: Path,
    statuses: Sequence[str],
    evidence_root: Path | None = None,
    calibration: Path | None = None,
    calibration_gate: Path | None = None,
) -> dict[str, Any]:
    """Record a terminal non-promotion when a locked stage cannot finish."""

    bound = strict_json(execution)
    baseline = bound.get("subject", {}).get("maintained_actor")
    failed = sorted(set(statuses))
    if not isinstance(baseline, str) or not baseline or not failed or any(
            re.fullmatch(r"[a-z_]+:(failure|cancelled|skipped)", item) is None
            for item in failed):
        raise EvidenceError("infrastructure retain binding drift")
    if (calibration is None) != (calibration_gate is None):
        raise EvidenceError(
            "calibration and calibration gate must be retained together"
        )
    calibration_disposition: dict[str, Any] | None = None
    if calibration is not None and calibration_gate is not None:
        calibration_disposition = calibration_gate_result(
            calibration, execution
        )
        sealed_gate = strict_json(calibration_gate)
        if calibration_gate.read_bytes() != canonical_json(sealed_gate) or \
                sealed_gate != calibration_disposition:
            raise EvidenceError("retained calibration gate is not canonical")
        if calibration_disposition["calibration_passed"] is False:
            if "select_evaluate:skipped" not in failed or any(
                    item.startswith("select_evaluate:") and
                    item != "select_evaluate:skipped" for item in failed):
                raise EvidenceError(
                    "negative calibration did not mechanically skip SELECT"
                )
    available: list[dict[str, Any]] = []
    if evidence_root is not None:
        if not evidence_root.is_dir() or evidence_root.is_symlink():
            raise EvidenceError("infrastructure evidence root is unsafe")
        if any(True for _ in evidence_root.rglob("*")):
            available = evidence_manifest(evidence_root)
    statistical_rejection = (
        calibration_disposition is not None
        and calibration_disposition["calibration_passed"] is False
    )
    result: dict[str, Any] = {
        "schema": "lc-policy-cost-v19-infrastructure-retain-v1",
        "artifact_kind": (
            "policy_cost_v19_locked_attempt_statistical_disposition"
            if statistical_rejection else
            "policy_cost_v19_locked_attempt_infrastructure_disposition"
        ),
        "status": (
            "complete_terminal_statistical_rejection"
            if statistical_rejection else
            "complete_terminal_non_promotable"
        ),
        "promotion_gate_passed": False,
        "maintained_actor": baseline,
        "failed_or_skipped_stages": failed,
        "execution_sha256": sha256(execution),
        "partial_evidence_retained": bool(available),
        "available_evidence": available,
    }
    if calibration_disposition is not None:
        result["calibration_disposition"] = calibration_disposition
    return result


def handoff_manifest(stage: str, paths: Sequence[Path]) -> dict[str, Any]:
    if stage not in {"calibration_to_select", "selection_to_test"} or not paths:
        raise EvidenceError("handoff manifest stage/path drift")
    rows = []
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.name != str(path):
            raise EvidenceError("handoff manifest unsafe input")
        rows.append({"path": path.name, "sha256": sha256(path), "size": path.stat().st_size})
    if len({row["path"] for row in rows}) != len(rows):
        raise EvidenceError("handoff manifest duplicate input")
    return {"schema": "lc-policy-cost-v19-handoff-v1", "stage": stage,
            "files": sorted(rows, key=lambda row: row["path"])}


def _github_output(path: Path | None, values: Mapping[str, str]) -> None:
    if path is None:
        return
    if any("\n" in value or "\r" in value for value in values.values()):
        raise EvidenceError("multiline GitHub output value is forbidden")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("--root", type=Path, default=Path("."))
    validate_jsonl = commands.add_parser("validate-jsonl")
    validate_jsonl.add_argument("--input", type=Path, required=True)
    for name in ("prepare-execution", "guard-execution"):
        item = commands.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--execution", type=Path, required=True)
        item.add_argument("--source-parent-commit", required=True)
        item.add_argument("--source-parent-tree", required=True)
        if name == "guard-execution":
            item.add_argument("--github-output", type=Path)
    manifest = commands.add_parser("pre-efficacy-manifest")
    manifest.add_argument("--execution", type=Path, required=True)
    manifest.add_argument("--transport", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    runtime = commands.add_parser("verify-runtime")
    runtime.add_argument("--transport", type=Path, required=True)
    freeze = commands.add_parser("verify-reservoir-freeze")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--split", choices=("TRAIN", "SELECT", "TEST"), required=True)
    train = commands.add_parser("train-input")
    train.add_argument("--evaluation", type=Path, required=True)
    train.add_argument("--execution", type=Path, required=True)
    train.add_argument("--allocation", type=Path, required=True)
    train.add_argument(
        "--allocation-manifest-output", type=Path, required=True,
        help="sealed calibration allocation JSON emitted from the TRAIN TSV",
    )
    train.add_argument("--evidence-binding-output", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    holdout = commands.add_parser("holdout-input")
    holdout.add_argument("--evaluation", type=Path, required=True)
    holdout.add_argument("--allocation", type=Path, required=True)
    holdout.add_argument("--stage", choices=("SELECT", "TEST"), required=True)
    holdout.add_argument("--selection", type=Path)
    holdout.add_argument("--calibration", type=Path, required=True)
    holdout.add_argument("--policy-cost", type=Path, required=True)
    holdout.add_argument("--actor-manifest", type=Path)
    holdout.add_argument("--execution", type=Path, required=True)
    holdout.add_argument("--output", type=Path, required=True)
    merge_evaluation = commands.add_parser("merge-evaluation-slices")
    merge_evaluation.add_argument("--stage", choices=("TRAIN", "SELECT", "TEST"), required=True)
    merge_evaluation.add_argument("--input", type=Path, action="append", required=True)
    merge_evaluation.add_argument("--output", type=Path, required=True)
    actors = commands.add_parser("actor-manifest")
    actors.add_argument("--execution", type=Path, required=True)
    actors.add_argument("--calibration", type=Path, required=True)
    actors.add_argument("--selection", type=Path, required=True)
    actors.add_argument("--artifact", type=Path, required=True)
    actors.add_argument("--preselect-artifact", type=Path, required=True)
    actors.add_argument("--actor-artifact-path", required=True)
    actors.add_argument("--output", type=Path, required=True)
    terminal = commands.add_parser("terminal-result")
    terminal.add_argument("--execution", type=Path, required=True)
    terminal.add_argument("--selection", type=Path, required=True)
    terminal.add_argument("--test", type=Path, required=True)
    terminal.add_argument("--actors", type=Path, required=True)
    terminal.add_argument("--safety", type=Path)
    terminal.add_argument("--final", type=Path)
    terminal.add_argument("--retained-baseline-reason")
    terminal.add_argument("--evidence-root", type=Path, required=True)
    terminal.add_argument("--output", type=Path, required=True)
    terminal.add_argument("--github-output", type=Path)
    calibration_gate = commands.add_parser("calibration-gate")
    calibration_gate.add_argument("--calibration", type=Path, required=True)
    calibration_gate.add_argument("--execution", type=Path, required=True)
    calibration_gate.add_argument("--output", type=Path, required=True)
    calibration_gate.add_argument("--github-output", type=Path)
    infrastructure = commands.add_parser("infrastructure-retain-result")
    infrastructure.add_argument("--execution", type=Path, required=True)
    infrastructure.add_argument("--stage-status", action="append", required=True)
    infrastructure.add_argument("--evidence-root", type=Path)
    infrastructure.add_argument("--calibration", type=Path)
    infrastructure.add_argument("--calibration-gate", type=Path)
    infrastructure.add_argument("--output", type=Path, required=True)
    handoff = commands.add_parser("handoff-manifest")
    handoff.add_argument("--stage", required=True)
    handoff.add_argument("--input", type=Path, action="append", required=True)
    handoff.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    try:
        if args.command == "validate-plan":
            validate_plan(strict_json(args.root / PLAN_PATH))
        elif args.command == "validate-jsonl":
            if not strict_jsonl(args.input):
                raise EvidenceError("JSONL evidence is empty")
        elif args.command == "prepare-execution":
            prepare_execution(
                args.root, args.execution, args.source_parent_commit,
                args.source_parent_tree,
            )
        elif args.command == "guard-execution":
            value = guard_execution(
                args.root, args.execution, args.source_parent_commit,
                args.source_parent_tree,
            )
            _github_output(args.github_output, {
                "execution_sha": sha256(args.execution),
                "maintained_actor": value["subject"]["maintained_actor"],
                "neutral_actor": value["subject"]["neutral_counterfactual_actor"],
                "root_path": value["subject"]["root_path"],
                "objective": str(value["subject"]["objective"]),
                "truth_metric": value["subject"]["train_truth_metric"],
            })
        elif args.command == "pre-efficacy-manifest":
            value = pre_efficacy_manifest(args.execution, args.transport)
            write_no_clobber(args.output, canonical_json(value))
        elif args.command == "verify-runtime":
            verify_runtime(args.transport)
        elif args.command == "verify-reservoir-freeze":
            verify_reservoir_freeze(args.manifest, args.root, args.split)
        elif args.command == "train-input":
            payload = train_input(args.evaluation, args.execution, args.allocation)
            write_no_clobber(args.output, payload)
            manifest, _ = train_allocation_manifest(args.allocation)
            write_no_clobber(
                args.allocation_manifest_output, canonical_json(manifest, pretty=False)
            )
            write_no_clobber(
                args.evidence_binding_output,
                canonical_json(train_evidence_binding(
                    args.evaluation, args.execution, args.allocation, payload
                ), pretty=False),
            )
        elif args.command == "holdout-input":
            value = holdout_input(
                args.evaluation, args.allocation, args.stage, args.selection,
                calibration=args.calibration, policy_cost=args.policy_cost,
                actors=args.actor_manifest, execution=args.execution,
                verify_raw=True,
            )
            write_no_clobber(args.output, canonical_json(value, pretty=False))
        elif args.command == "merge-evaluation-slices":
            write_no_clobber(
                args.output, merge_evaluation_slices(args.input, args.stage)
            )
        elif args.command == "actor-manifest":
            value = actor_manifest(
                args.execution, args.calibration, args.selection,
                args.artifact, args.preselect_artifact, args.actor_artifact_path,
            )
            write_no_clobber(args.output, canonical_json(value))
        elif args.command == "terminal-result":
            value = terminal_result(
                execution=args.execution, selection=args.selection,
                test=args.test, actors=args.actors, safety=args.safety,
                final=args.final, evidence_root=args.evidence_root,
                output=args.output,
                retained_baseline_reason=args.retained_baseline_reason,
            )
            write_no_clobber(args.output, canonical_json(value))
            _github_output(args.github_output, {
                "promotion_gate_passed": str(value["promotion_gate_passed"]).lower(),
                "maintained_actor": value["maintained_actor"],
            })
        elif args.command == "calibration-gate":
            value = calibration_gate_result(
                args.calibration, args.execution
            )
            write_no_clobber(args.output, canonical_json(value))
            _github_output(args.github_output, {
                "calibration_passed": str(
                    value["calibration_passed"]
                ).lower(),
            })
        elif args.command == "infrastructure-retain-result":
            write_no_clobber(args.output, canonical_json(
                infrastructure_retain_result(
                    args.execution, args.stage_status, args.evidence_root,
                    args.calibration, args.calibration_gate,
                )
            ))
        elif args.command == "handoff-manifest":
            write_no_clobber(args.output, canonical_json(
                handoff_manifest(args.stage, args.input)
            ))
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"policy-cost campaign: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
