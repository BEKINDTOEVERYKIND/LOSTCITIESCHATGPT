#!/usr/bin/env python3
"""Deterministically reduce native belief-history accuracy evidence.

The input is the native ``lc-history-belief-match-v1`` JSONL emitted once per
source-match trajectory, including any valid capped prefix. All contrasts are
paired on that match. Frozen TEST gates use a 20,000-draw SplitMix64
source-match cluster bootstrap and a simultaneous one-sided 99% bound across
all nine direct replacement components. The tool can certify only a
belief-accuracy artifact; it can never promote an actor or claim playing
strength.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


class ReductionError(ValueError):
    """Evidence is incomplete, malformed, non-finite, or mis-bound."""


ROW_SCHEMA = "lc-history-belief-match-v1"
IDENTITY_SCHEMA = "lc-belief-history-evaluation-identity-v1"
RESULT_SCHEMA = "lc-belief-history-accuracy-verdict-v1"
DIGEST_FIELD = "canonical_payload_sha256"
VERIFIED_IDENTITY_SHA256_FIELD = "_verified_identity_sha256"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BATCH = 64
MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FINGERPRINT_RE = re.compile(r"[0-9a-f]{16}")
EXACT17_TEXT_SHA256 = \
    "10034cf8b83aadf24fa0775e4dad2712573e1d84cbf364568ce6136682ac254c"
CAMPAIGN_ID = "belief-history-v1"
PLAN_SCHEMA = "lc-belief-history-v1-plan-v1"
PLAN_ARTIFACT_KIND = "locked_belief_history_v1_accuracy_campaign_plan"
FROZEN_TEST_MATCHES = 65_536
FROZEN_TEST_SHARDS = 16
FROZEN_TEST_ROOT = 202_706_100_403
FROZEN_BOOTSTRAP_SEED = 202_706_150_101
FROZEN_ACTOR_SHA256 = \
    "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"
FROZEN_MAX_SCORED_PLY = 300
FROZEN_SYMMETRIES = 20
FROZEN_TEMPERATURE = 0.03
FROZEN_BASE_ALPHA = 1.0
FROZEN_INCUMBENT_ALPHA = 1.15
FROZEN_PRIMARY_RELATIVE = 0.0
FROZEN_HISTORY_RELATIVE = 0.0
FROZEN_POINT_GAIN = 0.0
FROZEN_CONFIDENCE = 0.99
FROZEN_SIMULTANEOUS_COMPONENTS = 9
FROZEN_SIMULTANEOUS_METHOD = "single_step_max_standardized_error"
FROZEN_SIMULTANEOUS_STUDENTIZATION = \
    "fixed_original_source_match_cluster_se"
FROZEN_COVERAGE_CLAIM = "nominal_asymptotic"
FROZEN_ZERO_SE_POLICY = (
    "report null simultaneous LCB, mark inferentially ineligible, and fail "
    "the affected replacement bundle"
)
FROZEN_BOOTSTRAP_METHOD = (
    "paired source-match cluster bootstrap with deterministic SplitMix64 "
    "resampling, marginal percentile bounds, and single-step simultaneous "
    "max-standardized-error one-sided bounds"
)
FROZEN_COMPARISON_BUNDLE = (
    "For each ordered replacement comparison, require the frozen all-state "
    "joint-NLL, post-opponent-action joint-NLL, and all-state per-card-Brier "
    "point/simultaneous-one-sided-nominal-99%-LCB gates directly. One "
    "max-standardized-error critical value controls the nine-component "
    "selection family asymptotically; exact finite-sample coverage is not "
    "claimed and marginal percentile bounds are report-only. A zero original "
    "cluster SE is inferentially ineligible and fails its bundle. Never infer "
    "a pair by transitivity."
)

ROW_KEY_ORDER = (
    "actor_fingerprint", "base_alpha", "base_net_fingerprint",
    "matched_base_alpha", "incumbent_alpha", "incumbent_net_fingerprint",
    "matched_base_net_fingerprint", "capped_rounds",
    "excluded_state_count", "exclusion_manifest_count",
    "exclusion_manifest_sha256",
    "history_model_fingerprint", "max_scored_ply", "metrics",
    "rounds_completed", "reviewed_ply_inputs_used", "schema", "seed_root",
    "source_match_id", "structural_contract", "symmetries", "temperature",
)
ROW_KEYS = frozenset(ROW_KEY_ORDER)
GROUP_KEY_ORDER = ("all_states", "post_opponent_action")
GROUP_KEYS = frozenset(GROUP_KEY_ORDER)
MODEL_KEY_ORDER = (
    "base_262k_head", "matched_head_control", "incumbent_head", "history",
    "uniform_exact_k",
)
MODEL_KEYS = frozenset(MODEL_KEY_ORDER)
METRIC_KEY_ORDER = (
    "brier_sum", "nll_sum", "positive_count", "state_count",
    "top_hits_sum", "uncertain_card_count",
)
METRIC_KEYS = frozenset(METRIC_KEY_ORDER)
ROW_CONTRACT_KEYS = frozenset((
    "actor_fingerprint", "base_alpha", "base_net_fingerprint",
    "history_model_fingerprint", "incumbent_alpha",
    "incumbent_net_fingerprint", "matched_base_alpha",
    "matched_base_net_fingerprint", "max_scored_ply",
    "seed_root", "source_match_count", "source_match_start", "symmetries",
    "temperature",
))
STRUCTURAL_KEY_ORDER = (
    "action_history_public_only", "current_view_truth_scrubbed",
    "opening_history_uniform", "playing_actor_changed",
    "public_transcript_complete",
    "residual_features_opponent_action_anchored",
    "reviewed_ply_orbit_exclusion_enabled",
    "suit_equivariant_features", "truth_read_after_prediction",
    "wager_identity_collapsed",
)
STRUCTURAL_KEYS = frozenset(STRUCTURAL_KEY_ORDER)
REQUIRED_BINDING_DIGESTS = frozenset((
    "actor_sha256", "base_262k_head_sha256",
    "exact17_exclusions_sha256", "execution_sha256", "history_model_sha256",
    "matched_head_control_sha256", "native_structural_test_sha256",
    "shared_train_source_manifest_sha256", "test_generator_manifest_sha256",
    "transport_sha256",
))


def canonical_bytes(value: Any) -> bytes:
    """Return the sole verdict/identity encoding, including its final LF."""
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReductionError(f"value is not canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if DIGEST_FIELD in payload:
        raise ReductionError("payload is already sealed")
    result = dict(payload)
    result[DIGEST_FIELD] = canonical_sha256(payload)
    return result


def verify_seal(value: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    claimed = value.get(DIGEST_FIELD)
    if not isinstance(claimed, str) or SHA256_RE.fullmatch(claimed) is None:
        return False
    payload = dict(value)
    del payload[DIGEST_FIELD]
    try:
        return claimed == canonical_sha256(payload)
    except ReductionError:
        return False


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReductionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads(data: bytes, where: str) -> Any:
    try:
        return json.loads(
            data.decode("ascii"), object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReductionError(f"non-finite JSON token {token} in {where}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReductionError(f"invalid JSON in {where}: {exc}") from exc


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReductionError(f"{where} must be an object")
    return value


def _require_key_order(value: Mapping[str, Any], expected: Sequence[str],
                       where: str) -> None:
    if tuple(value) != tuple(expected):
        raise ReductionError(f"{where} native field order drifted")


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReductionError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReductionError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ReductionError(f"{where} must be finite and >= {minimum}")
    return result


def _probability(value: Any, where: str) -> float:
    result = _number(value, where)
    if result > 1.0:
        raise ReductionError(f"{where} must be <= 1")
    return result


def _binary32(value: float) -> float:
    """Round a JSON number exactly as the native ``float`` provenance does."""
    return struct.unpack("=f", struct.pack("=f", value))[0]


def frozen_test_generator_manifest(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Build the pre-efficacy TEST generator/range manifest.

    This manifest deliberately excludes raw TEST shard hashes, which do not
    exist when the identity is frozen. Those hashes are added only to the
    terminal verdict after all shards are complete.
    """
    bindings = _object(identity.get("bindings"), "identity.bindings")
    contract = _object(identity.get("row_contract"), "identity.row_contract")
    required = {
        "actor_sha256", "base_262k_head_sha256",
        "exact17_exclusions_sha256", "history_model_sha256",
        "matched_head_control_sha256",
    }
    if not required.issubset(bindings):
        raise ReductionError("generator manifest identity bindings are incomplete")
    per_shard = 65_536 // 16
    return {
        "schema": "lc-belief-history-test-generator-manifest-v1",
        "campaign_id": CAMPAIGN_ID,
        "stage": "TEST",
        "source_match_root": str(202_706_100_403),
        "source_match_start": 0,
        "source_match_count": 65_536,
        "shards": [
            {
                "shard_id": shard,
                "source_match_start": shard * per_shard,
                "source_match_count": per_shard,
            }
            for shard in range(16)
        ],
        "rounds": 3,
        "max_scored_ply": 300,
        "symmetries": 20,
        "temperature_binary32": _binary32(_number(
            contract.get("temperature"), "generator temperature")),
        "base_alpha_binary32": _binary32(_number(
            contract.get("base_alpha"), "generator base-alpha")),
        "matched_base_alpha_binary32": _binary32(_number(
            contract.get("matched_base_alpha"),
            "generator matched base-alpha")),
        "incumbent_alpha_binary32": _binary32(_number(
            contract.get("incumbent_alpha"),
            "generator incumbent alpha")),
        "actor_sha256": bindings["actor_sha256"],
        "base_262k_head_sha256": bindings["base_262k_head_sha256"],
        "matched_head_control_sha256":
            bindings["matched_head_control_sha256"],
        "history_model_sha256": bindings["history_model_sha256"],
        "row_fingerprints": {
            "actor": _fingerprint(
                contract.get("actor_fingerprint"),
                "generator actor fingerprint"),
            "base_262k_head": _fingerprint(
                contract.get("base_net_fingerprint"),
                "generator base-head fingerprint"),
            "matched_head_control": _fingerprint(
                contract.get("matched_base_net_fingerprint"),
                "generator matched-head fingerprint"),
            "history_model": _fingerprint(
                contract.get("history_model_fingerprint"),
                "generator history-model fingerprint"),
            "incumbent_head": _fingerprint(
                contract.get("incumbent_net_fingerprint"),
                "generator incumbent fingerprint"),
        },
        "exact17_exclusions_count": 17,
        "exact17_exclusions_sha256":
            bindings["exact17_exclusions_sha256"],
        "native_row_schema": ROW_SCHEMA,
        "reviewed_ply_inputs_used": False,
        "playing_actor_changed": False,
    }


def _read_json(path: Path, *, require_canonical: bool) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReductionError(f"cannot read {path}: {exc}") from exc
    value = _loads(raw, str(path))
    if not isinstance(value, dict):
        raise ReductionError(f"{path} must contain an object")
    if require_canonical and raw != canonical_bytes(value):
        raise ReductionError(f"{path} is not canonical JSON")
    return value, raw


def _bindings(value: Any) -> dict[str, Any]:
    obj = _object(value, "identity.bindings")
    if set(obj) != REQUIRED_BINDING_DIGESTS:
        raise ReductionError("identity binding fields drift")
    result = dict(obj)
    for key, item in result.items():
        if not isinstance(key, str) or not key:
            raise ReductionError("identity contains an invalid binding name")
        if key.endswith("_sha256") and (
                not isinstance(item, str) or SHA256_RE.fullmatch(item) is None):
            raise ReductionError(f"identity binding {key} is not lowercase SHA-256")
        if isinstance(item, (dict, list)):
            raise ReductionError(f"identity binding {key} is not scalar")
        canonical_bytes(item)
    return result


def _fingerprint(value: Any, where: str) -> str:
    if not isinstance(value, str) or FINGERPRINT_RE.fullmatch(value) is None:
        raise ReductionError(f"{where} must be a lowercase 16-hex fingerprint")
    return value


def load_identity(path: Path, stage: str) -> tuple[dict[str, Any], str]:
    value, raw = _read_json(path, require_canonical=True)
    if not verify_seal(value):
        raise ReductionError("evaluation identity seal is missing or stale")
    if set(value) != {
        "schema", "campaign_id", "stage", "bindings", "row_contract",
        DIGEST_FIELD,
    }:
        raise ReductionError("evaluation identity fields drift")
    if value["schema"] != IDENTITY_SCHEMA or value["stage"] != stage:
        raise ReductionError("evaluation identity schema/stage drift")
    if not isinstance(value["campaign_id"], str) or not value["campaign_id"]:
        raise ReductionError("evaluation identity campaign_id is invalid")
    bindings = _bindings(value["bindings"])
    if bindings["exact17_exclusions_sha256"] != EXACT17_TEXT_SHA256:
        raise ReductionError("identity does not bind the canonical exact17 text")
    contract = _object(value["row_contract"], "identity.row_contract")
    if set(contract) != ROW_CONTRACT_KEYS:
        raise ReductionError("identity row-contract fields drift")
    normalized = {
        "actor_fingerprint": _fingerprint(
            contract["actor_fingerprint"], "row_contract.actor_fingerprint"),
        "base_net_fingerprint": _fingerprint(
            contract["base_net_fingerprint"],
            "row_contract.base_net_fingerprint"),
        "history_model_fingerprint": _fingerprint(
            contract["history_model_fingerprint"],
            "row_contract.history_model_fingerprint"),
        "incumbent_net_fingerprint": _fingerprint(
            contract["incumbent_net_fingerprint"],
            "row_contract.incumbent_net_fingerprint"),
        "matched_base_net_fingerprint": _fingerprint(
            contract["matched_base_net_fingerprint"],
            "row_contract.matched_base_net_fingerprint"),
        "base_alpha": _binary32(_number(
            contract["base_alpha"], "row_contract.base_alpha")),
        "incumbent_alpha": _binary32(_number(
            contract["incumbent_alpha"],
            "row_contract.incumbent_alpha")),
        "matched_base_alpha": _binary32(_number(
            contract["matched_base_alpha"],
            "row_contract.matched_base_alpha")),
        "max_scored_ply": _integer(contract["max_scored_ply"],
                                   "row_contract.max_scored_ply"),
        "seed_root": _integer(contract["seed_root"], "row_contract.seed_root"),
        "source_match_count": _integer(contract["source_match_count"],
                                       "row_contract.source_match_count", 2),
        "source_match_start": _integer(contract["source_match_start"],
                                       "row_contract.source_match_start"),
        "symmetries": _integer(contract["symmetries"],
                               "row_contract.symmetries", 1),
        "temperature": _binary32(_number(
            contract["temperature"], "row_contract.temperature")),
    }
    return {
        "campaign_id": value["campaign_id"],
        "bindings": bindings,
        "row_contract": normalized,
        VERIFIED_IDENTITY_SHA256_FIELD: hashlib.sha256(raw).hexdigest(),
    }, hashlib.sha256(raw).hexdigest()


def _validate_plan(plan: Mapping[str, Any], identity: Mapping[str, Any],
                   stage: str) -> dict[str, Any]:
    if stage != "TEST":
        raise ReductionError("belief-history-v1 reducer accepts TEST only")
    if plan.get("schema") != PLAN_SCHEMA or \
            plan.get("artifact_kind") != PLAN_ARTIFACT_KIND:
        raise ReductionError("belief-history-v1 plan schema/kind drifted")
    schemas = _object(plan.get("artifact_schemas"), "plan.artifact_schemas")
    if schemas.get("evaluation_identity") != IDENTITY_SCHEMA or \
            schemas.get("raw_match_metrics") != ROW_SCHEMA or \
            schemas.get("result") != RESULT_SCHEMA:
        raise ReductionError("belief-history-v1 artifact schemas drifted")
    campaign_id = plan.get("experiment", plan.get("campaign_id"))
    if campaign_id != CAMPAIGN_ID or identity["campaign_id"] != CAMPAIGN_ID:
        raise ReductionError("plan and identity campaign IDs differ")
    evaluation = _object(plan.get("evaluation"), "plan.evaluation")
    stages = _object(evaluation.get("stages"), "plan.evaluation.stages")
    if set(stages) != {"TRAIN", "TEST"}:
        raise ReductionError("belief-history-v1 stage set drifted")
    stage_row = _object(stages.get(stage), f"evaluation.stages.{stage}")
    if stage_row.get("one_frozen_candidate") is not True or \
            stage_row.get("one_look") is not True or \
            stage_row.get("second_test_or_top_up") is not False or \
            stage_row.get("test_gate_applies") is not True:
        raise ReductionError("untouched one-look TEST contract drifted")
    train_stage = _object(stages.get("TRAIN"), "evaluation.stages.TRAIN")
    if train_stage.get("base_control_matches") != 262_144 or \
            train_stage.get("history_matches") != 65_536 or \
            train_stage.get("matched_control_additional_matches") != 65_536 or \
            train_stage.get("test_gate_applies") is not False:
        raise ReductionError("frozen TRAIN stage contract drifted")
    matches = _integer(stage_row.get("matches"), f"{stage}.matches", 2)
    data = _object(plan.get("data"), "plan.data")
    split = _object(_object(data.get("splits"), "plan.data.splits").get(stage),
                    f"plan.data.splits.{stage}")
    if matches != FROZEN_TEST_MATCHES or \
            split.get("matches") != FROZEN_TEST_MATCHES:
        raise ReductionError(f"{stage} data/evaluation counts differ")
    shards = _integer(split.get("shards"), f"{stage}.shards", 1)
    if shards != FROZEN_TEST_SHARDS or matches % shards:
        raise ReductionError(f"{stage} shard design drifted")
    train_split = _object(
        _object(data.get("splits"), "plan.data.splits").get("TRAIN"),
        "plan.data.splits.TRAIN")
    if train_split.get("base_control_matches") != 262_144 or \
            train_split.get("base_control_root") != "202706110102" or \
            train_split.get("base_control_shards") != 4 or \
            train_split.get("history_matches") != 65_536 or \
            train_split.get("history_root") != "202706100101" or \
            train_split.get("history_shards") != 1 or \
            train_split.get("matched_control_additional_matches") != 65_536 or \
            train_split.get("matched_control_root") != "202706100101" or \
            train_split.get("matched_control_shards") != 1:
        raise ReductionError("belief-history-v1 TRAIN design drifted")
    contract = identity["row_contract"]
    if contract["source_match_count"] != matches:
        raise ReductionError("identity source-match count differs from plan")
    if contract["source_match_start"] != 0:
        raise ReductionError("TEST source-match range must start at zero")
    root = split.get("root")
    if root != str(FROZEN_TEST_ROOT) or \
            contract["seed_root"] != FROZEN_TEST_ROOT:
        raise ReductionError("identity seed root differs from plan")
    source_actor = _object(data.get("source_actor"), "plan.data.source_actor")
    actor_net = _object(source_actor.get("actor_net"),
                        "plan.data.source_actor.actor_net")
    if actor_net.get("sha256") != FROZEN_ACTOR_SHA256 or \
            identity["bindings"]["actor_sha256"] != FROZEN_ACTOR_SHA256:
        raise ReductionError("identity actor SHA differs from frozen plan")
    plan_temperature = _binary32(_number(
        source_actor.get("temperature"), "plan source-actor temperature"))
    if source_actor.get("trajectory_symmetries") != FROZEN_SYMMETRIES or \
            plan_temperature != _binary32(FROZEN_TEMPERATURE) or \
            contract["symmetries"] != FROZEN_SYMMETRIES or \
            contract["temperature"] != _binary32(FROZEN_TEMPERATURE):
        raise ReductionError("identity trajectory settings differ from frozen plan")
    models = _object(plan.get("models"), "plan.models")
    candidate = _object(models.get("candidate"), "plan.models.candidate")
    matched = _object(models.get("matched_head_only_control"),
                      "plan.models.matched_head_only_control")
    incumbent = _object(models.get("incumbent_head"),
                        "plan.models.incumbent_head")
    if _binary32(_number(candidate.get("base_alpha"),
                         "candidate base-alpha")) != \
            _binary32(FROZEN_BASE_ALPHA) or \
            _binary32(_number(matched.get("base_alpha"),
                              "matched-control base-alpha")) != \
            _binary32(FROZEN_BASE_ALPHA) or \
            contract["max_scored_ply"] != FROZEN_MAX_SCORED_PLY or \
            contract["base_alpha"] != _binary32(FROZEN_BASE_ALPHA) or \
            contract["matched_base_alpha"] != \
            _binary32(FROZEN_BASE_ALPHA) or \
            contract["incumbent_alpha"] != \
            _binary32(FROZEN_INCUMBENT_ALPHA) or \
            contract["incumbent_net_fingerprint"] != \
            contract["actor_fingerprint"]:
        raise ReductionError("identity max-ply/base-alpha differs from belief-history-v1")
    if matched.get("primary_test_comparator") is not True or \
            incumbent.get("alpha") != FROZEN_INCUMBENT_ALPHA or \
            incumbent.get("sha256") != FROZEN_ACTOR_SHA256 or \
            incumbent.get("replacement_fallback") is not True:
        raise ReductionError("matched/incumbent comparator contract drifted")
    selection = _object(evaluation.get("terminal_artifact_selection"),
                        "evaluation.terminal_artifact_selection")
    if set(selection) != {
            "comparison_bundle", "rule",
            "selected_artifact_is_playing_actor"} or \
            selection.get("comparison_bundle") != FROZEN_COMPARISON_BUNDLE or \
            selection.get("rule") != [
            "retain history only if history passes directly against both matched_head_control and incumbent_head",
            "otherwise retain matched_head_control only if it passes directly against incumbent_head",
            "otherwise retain incumbent_head",
            ] or selection.get("selected_artifact_is_playing_actor") is not False:
        raise ReductionError("terminal artifact-selection contract drifted")

    bootstrap = _object(evaluation.get("bootstrap"), "evaluation.bootstrap")
    if bootstrap.get("replicates") != BOOTSTRAP_REPLICATES or \
            bootstrap.get("unit") != "source_match" or \
            bootstrap.get("method") != FROZEN_BOOTSTRAP_METHOD:
        raise ReductionError("locked 20k SplitMix64 bootstrap changed")
    simultaneous = _object(
        bootstrap.get("simultaneous_familywise"),
        "evaluation.bootstrap.simultaneous_familywise")
    if set(simultaneous) != {
            "components", "confidence", "coverage_claim",
            "exact_finite_sample_coverage_claimed", "method",
            "studentization", "zero_standard_error_policy"} or \
            simultaneous.get("components") != \
            FROZEN_SIMULTANEOUS_COMPONENTS or \
            _probability(simultaneous.get("confidence"),
                         "simultaneous familywise confidence") != \
            FROZEN_CONFIDENCE or \
            simultaneous.get("method") != FROZEN_SIMULTANEOUS_METHOD or \
            simultaneous.get("studentization") != \
            FROZEN_SIMULTANEOUS_STUDENTIZATION or \
            simultaneous.get("coverage_claim") != FROZEN_COVERAGE_CLAIM or \
            simultaneous.get("exact_finite_sample_coverage_claimed") is not \
            False or simultaneous.get("zero_standard_error_policy") != \
            FROZEN_ZERO_SE_POLICY:
        raise ReductionError("simultaneous 99% familywise bootstrap drifted")
    seed = bootstrap.get("seed")
    if seed != str(FROZEN_BOOTSTRAP_SEED):
        raise ReductionError("bootstrap seed differs from frozen TEST design")
    primary = _object(evaluation.get("primary_gate"), "primary_gate")
    history = _object(evaluation.get("history_gate"), "history_gate")
    brier = _object(evaluation.get("brier_gate"), "brier_gate")
    gates = {
        "primary_relative": _probability(
            primary.get("relative_nll_improvement_at_least"),
            "primary relative threshold"),
        "primary_point": _number(primary.get("point_gain_strictly_above"),
                                  "primary point-gain threshold"),
        "primary_lcb": _number(primary.get("nll_lcb_strictly_above"),
                               "primary LCB threshold"),
        "primary_confidence": _probability(primary.get("confidence"),
                                           "primary confidence"),
        "history_relative": _probability(
            history.get("relative_nll_improvement_at_least"),
            "history relative threshold"),
        "history_point": _number(history.get("point_gain_strictly_above"),
                                  "history point-gain threshold"),
        "history_lcb": _number(history.get("nll_lcb_strictly_above"),
                               "history LCB threshold"),
        "history_confidence": _probability(history.get("confidence"),
                                           "history confidence"),
        "min_opponent_actions": _integer(history.get("min_opponent_actions"),
                                         "minimum opponent actions", 1),
        "brier_lcb": _number(brier.get("lcb_strictly_above"),
                             "Brier LCB threshold"),
        "brier_point": _number(brier.get("point_gain_strictly_above"),
                                "Brier point-gain threshold"),
        "brier_confidence": _probability(brier.get("confidence"),
                                         "Brier confidence"),
    }
    for name in ("primary_confidence", "history_confidence", "brier_confidence"):
        if gates[name] != FROZEN_CONFIDENCE:
            raise ReductionError(f"{name} differs from frozen 99% confidence")
    if gates["primary_relative"] != FROZEN_PRIMARY_RELATIVE or \
            gates["primary_point"] != FROZEN_POINT_GAIN or \
            gates["primary_lcb"] != 0.0 or \
            gates["history_relative"] != FROZEN_HISTORY_RELATIVE or \
            gates["history_point"] != FROZEN_POINT_GAIN or \
            gates["history_lcb"] != 0.0 or \
            gates["brier_point"] != FROZEN_POINT_GAIN or \
            gates["brier_lcb"] != 0.0 or \
            gates["min_opponent_actions"] != 1:
        raise ReductionError("belief-history-v1 TEST thresholds drifted")
    generator_manifest = frozen_test_generator_manifest(identity)
    if identity["bindings"]["test_generator_manifest_sha256"] != \
            canonical_sha256(generator_manifest):
        raise ReductionError("pre-efficacy TEST generator manifest drifted")
    return {
        "matches": matches, "shards": shards, "bootstrap_seed": int(seed),
        "gates": gates, "generator_manifest": generator_manifest,
    }


def _metric(value: Any, where: str) -> dict[str, Any]:
    obj = _object(value, where)
    if set(obj) != METRIC_KEYS:
        raise ReductionError(f"{where} metric fields drift")
    _require_key_order(obj, METRIC_KEY_ORDER, where)
    states = _integer(obj["state_count"], f"{where}.state_count")
    cards = _integer(obj["uncertain_card_count"],
                     f"{where}.uncertain_card_count")
    positives = _integer(obj["positive_count"], f"{where}.positive_count")
    nll = _number(obj["nll_sum"], f"{where}.nll_sum")
    brier = _number(obj["brier_sum"], f"{where}.brier_sum")
    top_hits = _number(obj["top_hits_sum"], f"{where}.top_hits_sum")
    if (states == 0) != (cards == 0) or (states and cards < states):
        raise ReductionError(f"{where} state/card counts are inconsistent")
    if states and (cards > 60 * states or positives < states or
                   positives > cards - states):
        raise ReductionError(f"{where} violates per-state exact-K support bounds")
    if positives > cards or top_hits > positives + 1e-12 or brier > cards + 1e-12:
        raise ReductionError(f"{where} count-bounded metric is invalid")
    if not states and (positives or nll or brier or top_hits):
        raise ReductionError(f"{where} has metrics without scored states")
    return {
        "state_count": states, "uncertain_card_count": cards,
        "positive_count": positives, "nll_sum": nll, "brier_sum": brier,
        "top_hits_sum": top_hits,
    }


def _metric_group(value: Any, where: str) -> dict[str, Any]:
    obj = _object(value, where)
    if set(obj) != MODEL_KEYS:
        raise ReductionError(f"{where} model fields drift")
    _require_key_order(obj, MODEL_KEY_ORDER, where)
    result = {name: _metric(obj[name], f"{where}.{name}") for name in MODEL_KEYS}
    counts = {
        (row["state_count"], row["uncertain_card_count"], row["positive_count"])
        for row in result.values()
    }
    if len(counts) != 1:
        raise ReductionError(f"{where} paired model counts differ")
    if result["history"]["state_count"] == 0:
        raise ReductionError(f"{where} must contain at least one scored state")
    return result


def _validate_row(value: Any, where: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(value, where)
    if set(row) != ROW_KEYS or row.get("schema") != ROW_SCHEMA:
        raise ReductionError(f"{where} row schema/fields drift")
    _require_key_order(row, ROW_KEY_ORDER, where)
    contract = identity["row_contract"]
    for key in (
        "actor_fingerprint", "base_net_fingerprint",
        "history_model_fingerprint", "incumbent_net_fingerprint",
        "matched_base_net_fingerprint", "max_scored_ply", "seed_root",
        "symmetries",
    ):
        if row[key] != contract[key]:
            raise ReductionError(f"{where} provenance field {key} drifted")
    if _binary32(_number(row["base_alpha"], f"{where}.base_alpha")) != \
            contract["base_alpha"] or \
            _binary32(_number(
                row["matched_base_alpha"],
                f"{where}.matched_base_alpha")) != \
            contract["matched_base_alpha"] or \
            _binary32(_number(
                row["incumbent_alpha"],
                f"{where}.incumbent_alpha")) != \
            contract["incumbent_alpha"] or \
            _binary32(_number(row["temperature"], f"{where}.temperature")) != \
            contract["temperature"]:
        raise ReductionError(f"{where} binary32 provenance drifted")
    _fingerprint(row["actor_fingerprint"], f"{where}.actor_fingerprint")
    _fingerprint(row["base_net_fingerprint"],
                 f"{where}.base_net_fingerprint")
    _fingerprint(row["history_model_fingerprint"],
                 f"{where}.history_model_fingerprint")
    _fingerprint(row["matched_base_net_fingerprint"],
                 f"{where}.matched_base_net_fingerprint")
    _fingerprint(row["incumbent_net_fingerprint"],
                 f"{where}.incumbent_net_fingerprint")
    if row["incumbent_net_fingerprint"] != row["actor_fingerprint"]:
        raise ReductionError(f"{where} incumbent/actor fingerprint drifted")
    source_id = _integer(row["source_match_id"], f"{where}.source_match_id")
    capped_rounds = _integer(row["capped_rounds"], f"{where}.capped_rounds")
    excluded_states = _integer(
        row["excluded_state_count"], f"{where}.excluded_state_count")
    if row["exclusion_manifest_count"] != 17 or \
            row["exclusion_manifest_sha256"] != EXACT17_TEXT_SHA256 or \
            row["exclusion_manifest_sha256"] != \
            identity["bindings"]["exact17_exclusions_sha256"]:
        raise ReductionError(f"{where} exact17 exclusion binding drifted")
    rounds_completed = _integer(
        row["rounds_completed"], f"{where}.rounds_completed")
    if rounds_completed > 3:
        raise ReductionError(f"{where}.rounds_completed exceeds three")
    if row["reviewed_ply_inputs_used"] is not False:
        raise ReductionError(f"{where} used reviewed-ply inputs")
    structural = _object(row["structural_contract"],
                         f"{where}.structural_contract")
    _require_key_order(structural, STRUCTURAL_KEY_ORDER,
                       f"{where}.structural_contract")
    if set(structural) != STRUCTURAL_KEYS or \
            structural["playing_actor_changed"] is not False or \
            any(structural[key] is not True
                for key in STRUCTURAL_KEYS - {"playing_actor_changed"}):
        raise ReductionError(f"{where} structural contract failed or drifted")
    metrics = _object(row["metrics"], f"{where}.metrics")
    if set(metrics) != GROUP_KEYS:
        raise ReductionError(f"{where} metric-group fields drift")
    _require_key_order(metrics, GROUP_KEY_ORDER, f"{where}.metrics")
    groups = {name: _metric_group(metrics[name], f"{where}.metrics.{name}")
              for name in GROUP_KEYS}
    for model in MODEL_KEYS:
        all_metrics = groups["all_states"][model]
        post_metrics = groups["post_opponent_action"][model]
        for key in METRIC_KEYS:
            if post_metrics[key] > all_metrics[key] + 1e-12:
                raise ReductionError(
                    f"{where} post-action {model}.{key} exceeds all states")
    history_all = groups["all_states"]["history"]
    history_post = groups["post_opponent_action"]["history"]
    uniform_all = groups["all_states"]["uniform_exact_k"]
    uniform_post = groups["post_opponent_action"]["uniform_exact_k"]
    for key in METRIC_KEYS:
        history_opening = history_all[key] - history_post[key]
        uniform_opening = uniform_all[key] - uniform_post[key]
        if key in {"state_count", "uncertain_card_count", "positive_count"}:
            equal = history_opening == uniform_opening
        else:
            equal = math.isclose(
                history_opening, uniform_opening,
                rel_tol=1e-12, abs_tol=1e-10)
        if not equal:
            raise ReductionError(
                f"{where} opening history differs from exact-K uniform for {key}")
    if capped_rounds not in (0, 1) or \
            (capped_rounds == 0 and rounds_completed != 3) or \
            (capped_rounds == 1 and rounds_completed > 2):
        raise ReductionError(f"{where} capped/completed round semantics drifted")
    return {
        "source_match_id": source_id, "metrics": groups,
        "capped_rounds": capped_rounds,
        "excluded_state_count": excluded_states,
        "rounds_completed": rounds_completed,
    }


def load_rows(paths: Sequence[Path], identity: Mapping[str, Any],
              expected: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(paths) != expected["shards"]:
        raise ReductionError(
            f"expected {expected['shards']} shards, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for shard_id, path in enumerate(paths):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ReductionError(f"cannot read {path}: {exc}") from exc
        if not raw or not raw.endswith(b"\n"):
            raise ReductionError(f"{path} is empty or lacks final LF")
        if b"\r" in raw or b"\t" in raw or b" " in raw:
            raise ReductionError(f"{path} is not native fixed-order LF JSONL")
        start = len(rows)
        for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
            if line in (b"\n", b"\r\n"):
                raise ReductionError(f"blank line in {path}:{line_number}")
            if not line.endswith(b"\n"):
                raise ReductionError(f"line lacks LF in {path}:{line_number}")
            rows.append(_validate_row(
                _loads(line, f"{path}:{line_number}"),
                f"{path}:{line_number}", identity))
        count = len(rows) - start
        if count == 0:
            raise ReductionError(f"{path} has no records")
        manifests.append({
            "shard_id": shard_id, "records": count,
            "first_source_match_id": rows[start]["source_match_id"],
            "last_source_match_id": rows[-1]["source_match_id"],
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    if len(rows) != expected["matches"]:
        raise ReductionError(
            f"expected {expected['matches']} matches, found {len(rows)}")
    contract = identity["row_contract"]
    first = contract["source_match_start"]
    ids = [row["source_match_id"] for row in rows]
    if ids != list(range(first, first + contract["source_match_count"])):
        raise ReductionError(
            "source_match_id is not unique, contiguous, and globally ordered")
    expected_per_shard = expected["matches"] // expected["shards"]
    if any(item["records"] != expected_per_shard for item in manifests):
        raise ReductionError("shard match counts are not exact")
    return rows, manifests


def _arrays(rows: Sequence[Mapping[str, Any]], group: str, model: str,
            metric: str, denominator: str | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    default_denominator = {
        "nll_sum": "state_count",
        "brier_sum": "uncertain_card_count",
        "top_hits_sum": "positive_count",
    }.get(metric)
    if denominator is None:
        denominator = default_denominator
    if default_denominator is None or denominator not in {
            "state_count", "uncertain_card_count", "positive_count"}:
        raise ReductionError(f"unsupported aggregate metric {metric}")
    return (
        np.asarray([row["metrics"][group][model][metric] for row in rows],
                   dtype=np.float64),
        np.asarray([row["metrics"][group][model][denominator] for row in rows],
                   dtype=np.float64),
    )


def _statistic(name: str, numerator: np.ndarray, denominator: np.ndarray,
               *, observation_count: int | None = None) -> dict[str, Any]:
    if numerator.ndim != 1 or numerator.shape != denominator.shape or \
            not np.all(np.isfinite(numerator)) or \
            not np.all(np.isfinite(denominator)):
        raise ReductionError(f"{name} arrays are invalid")
    total_denominator = float(np.sum(denominator))
    contributing = int(np.count_nonzero(denominator > 0.0))
    if total_denominator <= 0.0 or contributing < 2:
        raise ReductionError(f"{name} requires two contributing source matches")
    estimate = float(np.sum(numerator) / total_denominator)
    residual = numerator - estimate * denominator
    n = len(numerator)
    se = math.sqrt(n / (n - 1) * float(np.dot(residual, residual))) \
        / total_denominator
    if not math.isfinite(estimate) or not math.isfinite(se):
        raise ReductionError(f"{name} statistic is non-finite")
    observations = int(total_denominator) if observation_count is None \
        else _integer(observation_count, f"{name}.observation_count", 1)
    return {
        "name": name, "numerator": numerator, "denominator": denominator,
        "estimate": estimate, "source_match_cluster_se": se,
        "clusters": n, "contributing_clusters": contributing,
        "observations": observations,
    }


def _paired_models(rows: Sequence[Mapping[str, Any]], group: str, metric: str,
                   prefix: str, candidate_model: str,
                   baseline_model: str, *,
                   include_relative: bool = True) -> dict[str, Any]:
    candidate, denominator = _arrays(rows, group, candidate_model, metric)
    baseline, baseline_denominator = _arrays(
        rows, group, baseline_model, metric)
    if not np.array_equal(denominator, baseline_denominator):
        raise ReductionError(f"{prefix} paired denominators differ")
    result = {
        "candidate": _statistic(f"{prefix}.candidate", candidate, denominator),
        "baseline": _statistic(f"{prefix}.baseline", baseline, denominator),
        "absolute": _statistic(f"{prefix}.absolute", baseline - candidate,
                               denominator),
    }
    result["relative"] = (
        _statistic(f"{prefix}.relative", baseline - candidate, baseline,
                   observation_count=int(np.sum(denominator)))
        if include_relative else None)
    return result


def _paired(rows: Sequence[Mapping[str, Any]], group: str, metric: str,
            prefix: str) -> dict[str, Any]:
    return _paired_models(
        rows, group, metric, prefix, "history", "matched_head_control")


def _splitmix_indices(seed: int, offset: int, count: int, n: int) -> np.ndarray:
    size = count * n
    sequence = np.arange(offset + 1, offset + size + 1, dtype=np.uint64)
    z = np.uint64(seed & MASK64) + SPLITMIX_GAMMA * sequence
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    uniform = (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return np.floor(uniform * n).astype(np.int64).reshape(count, n)


def _bootstrap(statistics: Sequence[dict[str, Any]], seed: int) -> dict[str, np.ndarray]:
    n = len(statistics[0]["numerator"])
    numerator = np.column_stack([item["numerator"] for item in statistics])
    denominator = np.column_stack([item["denominator"] for item in statistics])
    draws = np.empty((BOOTSTRAP_REPLICATES, len(statistics)), dtype=np.float64)
    written = offset = 0
    while written < BOOTSTRAP_REPLICATES:
        count = min(BOOTSTRAP_BATCH, BOOTSTRAP_REPLICATES - written)
        indices = _splitmix_indices(seed, offset, count, n)
        offset += count * n
        row_offset = np.arange(count, dtype=np.int64)[:, None] * n
        counts = np.bincount(
            (indices + row_offset).ravel(), minlength=count * n,
        ).reshape(count, n).astype(np.float64)
        num = counts @ numerator
        den = counts @ denominator
        if np.any(den <= 0.0):
            raise ReductionError("bootstrap encountered an empty denominator")
        draws[written:written + count] = num / den
        written += count
    if not np.all(np.isfinite(draws)):
        raise ReductionError("bootstrap produced non-finite estimates")
    return {item["name"]: draws[:, index]
            for index, item in enumerate(statistics)}


def _empirical_quantile_index(length: int, probability: Fraction) -> int:
    """Return the zero-based inverse-ECDF rank, ``ceil(p*n)-1``."""
    if length < 1 or probability < 0 or probability > 1:
        raise ReductionError("invalid empirical-quantile request")
    scaled_numerator = probability.numerator * length
    rank = (scaled_numerator + probability.denominator - 1) // \
        probability.denominator - 1
    return min(max(rank, 0), length - 1)


def _lower(values: np.ndarray, confidence: float) -> float:
    ordered = np.sort(values)
    probability = Fraction(1, 1) - Fraction(str(confidence))
    return float(ordered[_empirical_quantile_index(
        len(ordered), probability)])


def _upper(values: np.ndarray, confidence: float) -> float:
    ordered = np.sort(values)
    return float(ordered[_empirical_quantile_index(
        len(ordered), Fraction(str(confidence)))])


def _public(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "estimate": item["estimate"],
        "source_match_cluster_se": item["source_match_cluster_se"],
        "clusters": item["clusters"],
        "contributing_clusters": item["contributing_clusters"],
        "observations": item["observations"],
    }


def _bounded(item: Mapping[str, Any], draws: np.ndarray,
             confidence: float) -> dict[str, Any]:
    result = _public(item)
    result.update({
        "one_sided_confidence": confidence,
        "percentile_lcb": _lower(draws, confidence),
        "percentile_ucb": _upper(draws, confidence),
    })
    return result


def _simultaneous_lower_bounds(
        statistics: Sequence[Mapping[str, Any]],
        draws: Mapping[str, np.ndarray], confidence: float) -> dict[str, Any]:
    """Return single-step familywise lower bounds for paired cluster gains.

    Bootstrap errors are standardized by each component's original
    source-match cluster standard error, then maximized across the frozen
    family before taking the one-sided critical quantile. A zero standard
    error is accepted only for last-bit numerical variation around a
    mathematically degenerate empirical distribution. Such a component is
    ineligible for inference and receives no LCB, so its replacement gate
    fails closed.
    """
    if len(statistics) != FROZEN_SIMULTANEOUS_COMPONENTS:
        raise ReductionError("simultaneous bootstrap family size drifted")
    names = [str(item["name"]) for item in statistics]
    if len(set(names)) != len(names) or set(draws) != set(names):
        raise ReductionError("simultaneous bootstrap component names drifted")
    replicate_counts = {len(draws[name]) for name in names}
    if replicate_counts != {BOOTSTRAP_REPLICATES}:
        raise ReductionError("simultaneous bootstrap replicate count drifted")

    standardized_errors = np.empty(
        (BOOTSTRAP_REPLICATES, len(statistics)), dtype=np.float64)
    zero_se_components: set[str] = set()
    for index, item in enumerate(statistics):
        estimate = float(item["estimate"])
        standard_error = float(item["source_match_cluster_se"])
        values = np.asarray(draws[item["name"]], dtype=np.float64)
        if values.shape != (BOOTSTRAP_REPLICATES,):
            raise ReductionError("simultaneous bootstrap draw shape drifted")
        if not math.isfinite(estimate) or not math.isfinite(standard_error) or \
                standard_error < 0.0 or not np.all(np.isfinite(values)):
            raise ReductionError("simultaneous bootstrap input is non-finite")
        if standard_error == 0.0:
            tolerance = 64.0 * np.finfo(np.float64).eps * \
                max(1.0, abs(estimate))
            if np.max(np.abs(values - estimate)) > tolerance:
                raise ReductionError(
                    "zero-SE simultaneous component has variable draws")
            standardized_errors[:, index] = 0.0
            zero_se_components.add(item["name"])
        else:
            standardized_errors[:, index] = \
                (values - estimate) / standard_error
    if not np.all(np.isfinite(standardized_errors)):
        raise ReductionError("simultaneous bootstrap errors are non-finite")

    maximum_errors = np.max(standardized_errors, axis=1)
    # This floor is conservative if the finite bootstrap's 99th percentile of
    # the maximum centered error is negative, and ensures LCB <= estimate.
    critical_value = max(0.0, _upper(maximum_errors, confidence))
    lower_bounds = {
        item["name"]: (
            None if item["name"] in zero_se_components else
            float(item["estimate"] - critical_value *
                  item["source_match_cluster_se"]))
        for item in statistics
    }
    if not all(value is None or math.isfinite(value)
               for value in lower_bounds.values()):
        raise ReductionError("simultaneous bootstrap LCB is non-finite")
    return {
        "method": FROZEN_SIMULTANEOUS_METHOD,
        "studentization": FROZEN_SIMULTANEOUS_STUDENTIZATION,
        "one_sided_familywise_confidence": confidence,
        "family_size": len(statistics),
        "component_names": names,
        "critical_value": critical_value,
        "critical_value_floor": 0.0,
        "empirical_quantile_rank": _empirical_quantile_index(
            BOOTSTRAP_REPLICATES, Fraction(str(confidence))),
        "lower_bounds": lower_bounds,
        "zero_se_components_failed_closed": sorted(zero_se_components),
    }


def _select_belief_artifact(pairwise_passes: Mapping[str, bool]) -> str:
    expected = {
        "history_vs_matched_head_control",
        "history_vs_incumbent_head",
        "matched_head_control_vs_incumbent_head",
    }
    if set(pairwise_passes) != expected or \
            any(type(value) is not bool for value in pairwise_passes.values()):
        raise ReductionError("pairwise replacement verdict fields drifted")
    if pairwise_passes["history_vs_matched_head_control"] and \
            pairwise_passes["history_vs_incumbent_head"]:
        return "history"
    if pairwise_passes["matched_head_control_vs_incumbent_head"]:
        return "matched_head_control"
    return "incumbent_head"


def reduce_evidence(plan: Mapping[str, Any], plan_sha256: str,
                    identity: Mapping[str, Any], identity_sha256: str,
                    stage: str, inputs: Sequence[Path]) -> dict[str, Any]:
    if plan_sha256 != canonical_sha256(plan):
        raise ReductionError("supplied plan SHA-256 is stale")
    if not isinstance(identity_sha256, str) or \
            SHA256_RE.fullmatch(identity_sha256) is None or \
            identity.get(VERIFIED_IDENTITY_SHA256_FIELD) != identity_sha256:
        raise ReductionError("supplied identity SHA-256 is stale or unverified")
    expected = _validate_plan(plan, identity, stage)
    rows, shards = load_rows(inputs, identity, expected)
    gates = expected["gates"]
    pair_models = {
        "history_vs_matched_head_control":
            ("history", "matched_head_control"),
        "history_vs_incumbent_head": ("history", "incumbent_head"),
        "matched_head_control_vs_incumbent_head":
            ("matched_head_control", "incumbent_head"),
    }
    pairs: dict[str, dict[str, Any]] = {}
    for name, (candidate_model, baseline_model) in pair_models.items():
        pairs[name] = {
            "candidate_model": candidate_model,
            "baseline_model": baseline_model,
            "all_states_joint_nll": _paired_models(
                rows, "all_states", "nll_sum", f"{name}.all_states_nll",
                candidate_model, baseline_model),
            "post_opponent_action_joint_nll": _paired_models(
                rows, "post_opponent_action", "nll_sum",
                f"{name}.post_action_nll", candidate_model, baseline_model),
            "all_states_brier": _paired_models(
                rows, "all_states", "brier_sum", f"{name}.all_states_brier",
                candidate_model, baseline_model, include_relative=False),
        }
    base_diagnostics = {
        "all_states_joint_nll": _paired_models(
            rows, "all_states", "nll_sum", "base_diagnostic_all_nll",
            "history", "base_262k_head"),
        "post_opponent_action_joint_nll": _paired_models(
            rows, "post_opponent_action", "nll_sum",
            "base_diagnostic_post_nll", "history", "base_262k_head"),
        "all_states_brier": _paired_models(
            rows, "all_states", "brier_sum", "base_diagnostic_all_brier",
            "history", "base_262k_head", include_relative=False),
    }
    scope_reports: dict[str, dict[str, dict[str, Any]]] = {}
    for group in ("all_states", "post_opponent_action"):
        scope_reports[group] = {}
        for model in MODEL_KEY_ORDER:
            nll, states = _arrays(rows, group, model, "nll_sum")
            nll_again, cards = _arrays(
                rows, group, model, "nll_sum", "uncertain_card_count")
            if not np.array_equal(nll, nll_again):
                raise ReductionError("internal NLL aggregation drift")
            brier_sum, brier_cards = _arrays(
                rows, group, model, "brier_sum")
            top_hits, positives = _arrays(
                rows, group, model, "top_hits_sum")
            scope_reports[group][model] = {
                "joint_nll_per_state": _statistic(
                    f"{group}.{model}.nll_per_state", nll, states),
                "joint_nll_per_uncertain_card": _statistic(
                    f"{group}.{model}.nll_per_card", nll, cards),
                "brier_per_uncertain_card": _statistic(
                    f"{group}.{model}.brier_per_card",
                    brier_sum, brier_cards),
                "top_k_recall": _statistic(
                    f"{group}.{model}.top_k_recall", top_hits, positives),
            }
    simultaneous_statistics = [
        pair[metric]["absolute"]
        for pair in pairs.values()
        for metric in (
            "all_states_joint_nll", "post_opponent_action_joint_nll",
            "all_states_brier",
        )
    ]
    statistics = []
    for pair in pairs.values():
        for metric in (
                "all_states_joint_nll", "post_opponent_action_joint_nll",
                "all_states_brier"):
            statistics.append(pair[metric]["absolute"])
            if pair[metric]["relative"] is not None:
                statistics.append(pair[metric]["relative"])
    draws = _bootstrap(statistics, expected["bootstrap_seed"])
    simultaneous = _simultaneous_lower_bounds(
        simultaneous_statistics,
        {item["name"]: draws[item["name"]]
         for item in simultaneous_statistics},
        FROZEN_CONFIDENCE)
    bounded_pairs: dict[str, dict[str, Any]] = {}
    for name, pair in pairs.items():
        bounded: dict[str, Any] = {
            "candidate_model": pair["candidate_model"],
            "baseline_model": pair["baseline_model"],
        }
        for metric, confidence in (
            ("all_states_joint_nll", gates["primary_confidence"]),
            ("post_opponent_action_joint_nll", gates["history_confidence"]),
            ("all_states_brier", gates["brier_confidence"]),
        ):
            values = pair[metric]
            bounded[metric] = {
                "candidate": _public(values["candidate"]),
                "baseline": _public(values["baseline"]),
                "absolute_improvement": _bounded(
                    values["absolute"], draws[values["absolute"]["name"]],
                    confidence),
                "relative_improvement": (
                    None if values["relative"] is None else
                    _bounded(values["relative"],
                             draws[values["relative"]["name"]],
                             confidence)),
            }
            bounded[metric]["absolute_improvement"].update({
                "simultaneous_familywise_lcb": simultaneous["lower_bounds"]
                    [values["absolute"]["name"]],
                "simultaneous_inferentially_eligible":
                    simultaneous["lower_bounds"]
                    [values["absolute"]["name"]] is not None,
                "simultaneous_familywise_confidence":
                    simultaneous["one_sided_familywise_confidence"],
                "simultaneous_family_size": simultaneous["family_size"],
            })
        all_nll = bounded["all_states_joint_nll"]
        post_nll = bounded["post_opponent_action_joint_nll"]
        all_brier = bounded["all_states_brier"]
        all_nll_lcb = all_nll["absolute_improvement"] \
            ["simultaneous_familywise_lcb"]
        post_nll_lcb = post_nll["absolute_improvement"] \
            ["simultaneous_familywise_lcb"]
        all_brier_lcb = all_brier["absolute_improvement"] \
            ["simultaneous_familywise_lcb"]
        bounded["metric_gates"] = {
            "all_states_joint_nll": (
                all_nll["absolute_improvement"]["estimate"] >
                gates["primary_point"] and
                all_nll["relative_improvement"]["estimate"] >=
                gates["primary_relative"] and
                all_nll_lcb is not None and
                all_nll_lcb >
                gates["primary_lcb"]),
            "post_opponent_action_joint_nll": (
                post_nll["absolute_improvement"]["estimate"] >
                gates["history_point"] and
                post_nll["relative_improvement"]["estimate"] >=
                gates["history_relative"] and
                post_nll_lcb is not None and
                post_nll_lcb >
                gates["history_lcb"]),
            "all_states_brier": (
                all_brier["absolute_improvement"]["estimate"] >
                gates["brier_point"] and
                all_brier_lcb is not None and
                all_brier_lcb >
                gates["brier_lcb"]),
        }
        bounded["bundle_passed"] = all(bounded["metric_gates"].values())
        bounded_pairs[name] = bounded

    pairwise_passes = {
        name: result["bundle_passed"]
        for name, result in bounded_pairs.items()
    }
    history_passed = (
        pairwise_passes["history_vs_matched_head_control"] and
        pairwise_passes["history_vs_incumbent_head"])
    matched_passed = pairwise_passes[
        "matched_head_control_vs_incumbent_head"]
    selected_artifact = _select_belief_artifact(pairwise_passes)
    selected_binding = {
        "history": {
            "sha256": identity["bindings"]["history_model_sha256"],
            "alpha": FROZEN_BASE_ALPHA,
            "fingerprint": identity["row_contract"]
                ["history_model_fingerprint"],
            "base_262k_head_sha256":
                identity["bindings"]["base_262k_head_sha256"],
            "base_net_fingerprint": identity["row_contract"]
                ["base_net_fingerprint"],
            "base_alpha": identity["row_contract"]["base_alpha"],
        },
        "matched_head_control": {
            "sha256": identity["bindings"]["matched_head_control_sha256"],
            "alpha": FROZEN_BASE_ALPHA,
            "fingerprint": identity["row_contract"]
                ["matched_base_net_fingerprint"],
        },
        "incumbent_head": {
            "sha256": identity["bindings"]["actor_sha256"],
            "alpha": FROZEN_INCUMBENT_ALPHA,
            "fingerprint": identity["row_contract"]
                ["incumbent_net_fingerprint"],
        },
    }[selected_artifact]
    stage_passed = selected_artifact != "incumbent_head"
    accuracy_passed = stage_passed

    payload = {
        "schema": RESULT_SCHEMA,
        "campaign_id": identity["campaign_id"],
        "stage": stage,
        "plan_sha256": plan_sha256,
        "identity_sha256": identity_sha256,
        "bindings": identity["bindings"],
        "evidence": {
            "source_matches": len(rows),
            "first_source_match_id": rows[0]["source_match_id"],
            "last_source_match_id": rows[-1]["source_match_id"],
            "shards": shards,
            "exact_counts": True,
            "exact_provenance_against_sealed_identity": True,
            "external_binding_bytes_reverified_by_reducer": False,
            "finite_metric_sums": True,
            "reviewed_ply_inputs_used": False,
            "native_row_structural_contract": {
                "action_history_public_only": True,
                "current_view_truth_scrubbed": True,
                "opening_history_uniform": True,
                "playing_actor_changed": False,
                "public_transcript_complete": True,
                "residual_features_opponent_action_anchored": True,
                "reviewed_ply_orbit_exclusion_enabled": True,
                "suit_equivariant_features": True,
                "truth_read_after_prediction": True,
                "wager_identity_collapsed": True,
            },
            "unique_contiguous_globally_ordered_source_match_ids": True,
            "rounds_completed": sum(row["rounds_completed"] for row in rows),
            "capped_rounds": sum(row["capped_rounds"] for row in rows),
            "excluded_state_count": sum(
                row["excluded_state_count"] for row in rows),
            "capped_prefix_metrics_retained": True,
            "exact17_manifest_count": 17,
            "exact17_manifest_sha256": EXACT17_TEXT_SHA256,
            "native_structural_contract_bound_not_recomputed": {
                "exact17_exclusions_sha256":
                    identity["bindings"]["exact17_exclusions_sha256"],
                "native_structural_test_sha256":
                    identity["bindings"]["native_structural_test_sha256"],
            },
            "pre_efficacy_test_generator_manifest":
                expected["generator_manifest"],
            "sample_counts": {
                group: {
                    key: sum(
                        row["metrics"][group]["history"][key]
                        for row in rows)
                    for key in (
                        "state_count", "uncertain_card_count", "positive_count")
                }
                for group in ("all_states", "post_opponent_action")
            },
        },
        "inference": {
            "method": "paired-source-match-cluster-bootstrap",
            "prng": "SplitMix64-counter-upper53",
            "marginal_bounds_report_only": "one-sided-percentile",
            "replacement_gate_bounds":
                "single-step-simultaneous-max-standardized-error",
            "familywise_coverage_claimed": "nominal_asymptotic",
            "exact_finite_sample_coverage_claimed": False,
            "one_sided_familywise_confidence": FROZEN_CONFIDENCE,
            "simultaneous_family_size": FROZEN_SIMULTANEOUS_COMPONENTS,
            "simultaneous_components": simultaneous["component_names"],
            "studentization": FROZEN_SIMULTANEOUS_STUDENTIZATION,
            "simultaneous_critical_value": simultaneous["critical_value"],
            "simultaneous_critical_value_floor":
                simultaneous["critical_value_floor"],
            "empirical_quantile_rank": "ceil(probability*replicates)-1",
            "simultaneous_critical_quantile_rank":
                simultaneous["empirical_quantile_rank"],
            "joint_bootstrap_replicates_at_or_below_critical_at_least":
                simultaneous["empirical_quantile_rank"] + 1,
            "zero_se_components_failed_closed":
                simultaneous["zero_se_components_failed_closed"],
            "coverage_assumptions": [
                "exchangeable independent source-match clusters",
                "regular nondegenerate ratio-of-sums estimators",
                "finite moments",
                "cluster-bootstrap consistency",
            ],
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": str(expected["bootstrap_seed"]),
        },
        "metrics": {
            "pairwise_replacement_comparisons": bounded_pairs,
            "top_k_recall_not_a_gate": {
                group: {
                    model: _public(
                        scope_reports[group][model]["top_k_recall"])
                    for model in MODEL_KEY_ORDER
                }
                for group in ("all_states", "post_opponent_action")
            },
            "per_model_scope_reports_not_additional_gates": {
                group: {
                    model: {
                        metric: _public(value)
                        for metric, value in scope_reports[group][model].items()
                    }
                    for model in MODEL_KEY_ORDER
                }
                for group in ("all_states", "post_opponent_action")
            },
            "base_262k_head_candidate_deltas_diagnostic_only": {
                name: {
                    "candidate": _public(values["candidate"]),
                    "base_262k_head": _public(values["baseline"]),
                    "absolute_improvement": _public(values["absolute"]),
                    "relative_improvement": (
                        None if values["relative"] is None else
                        _public(values["relative"])),
                }
                for name, values in base_diagnostics.items()
            },
        },
        "gates": {
            "terminal_test_gate_applied": True,
            "metric_thresholds_for_every_ordered_replacement": {
                "all_states_joint_nll": {
                    "point_gain_strictly_above": gates["primary_point"],
                    "relative_improvement_at_least":
                        gates["primary_relative"],
                    "simultaneous_absolute_improvement_lcb_strictly_above":
                        gates["primary_lcb"],
                },
                "post_opponent_action_joint_nll": {
                    "min_opponent_actions": gates["min_opponent_actions"],
                    "point_gain_strictly_above": gates["history_point"],
                    "relative_improvement_at_least":
                        gates["history_relative"],
                    "simultaneous_absolute_improvement_lcb_strictly_above":
                        gates["history_lcb"],
                },
                "all_states_brier": {
                    "point_gain_strictly_above": gates["brier_point"],
                    "simultaneous_absolute_improvement_lcb_strictly_above":
                        gates["brier_lcb"],
                },
            },
            "pairwise_bundle_passed": {
                name: passed for name, passed in pairwise_passes.items()
            },
            "selection_rule": [
                "history only if it passes directly against matched and incumbent",
                "otherwise matched only if it passes directly against incumbent",
                "otherwise incumbent",
            ],
            "history_replacement_passed": history_passed,
            "matched_replacement_passed": matched_passed,
        },
        "not_evaluated_metrics": {
            "ece": "native per-match rows do not carry calibration-bin counts; no proxy is inferred",
        },
        "verdict": {
            "stage_passed": stage_passed,
            "accuracy_artifact_passed": accuracy_passed,
            "selected_belief_artifact": selected_artifact,
            "selected_belief_artifact_binding": selected_binding,
            "selected_artifact_is_playing_actor": False,
            "base_262k_head_eligible_for_selection": False,
            "actor_promotion_authorized": False,
            "playing_strength_claimed": False,
            "reason": (
                f"selected {selected_artifact} by the frozen direct-comparison hierarchy"
            ),
        },
    }
    return seal(payload)


def write_no_clobber(path: Path, result: Mapping[str, Any]) -> None:
    if not verify_seal(result):
        raise ReductionError("refusing to write an unsealed verdict")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ReductionError(f"refusing to replace {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("TEST",))
    parser.add_argument("--input", required=True, action="append", type=Path,
                        help="repeat in ascending shard/range order")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        plan, _ = _read_json(args.plan, require_canonical=False)
        identity, identity_sha = load_identity(args.identity, args.stage)
        result = reduce_evidence(
            plan, canonical_sha256(plan), identity, identity_sha,
            args.stage, args.input)
        write_no_clobber(args.output, result)
    except ReductionError as exc:
        print(f"belief-history reducer: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
