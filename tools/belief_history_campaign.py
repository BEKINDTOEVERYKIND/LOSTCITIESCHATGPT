#!/usr/bin/env python3
"""Fail-closed definition and launch bindings for belief-history-v1.

This helper deliberately has no training or evaluation command.  It can only
validate the inert accuracy-campaign definition, materialize its unique
execution binding, and verify that binding at launch.  Model efficacy remains
inside the separately frozen workflow and reducer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any


PLAN_PATH = Path("data/experiments/locked_belief_history_v1_plan.json")
EXCLUSIONS_PATH = Path("data/experiments/belief_history_v1_exclusions.json")
TEMPLATE_PATH = Path(
    "data/experiments/locked_belief_history_v1_execution.template.json"
)
EXECUTION_PATH = Path(
    "data/experiments/locked_belief_history_v1_execution.json"
)
WORKFLOW_PATH = Path(".github/workflows/belief-history-v1.yml")
DEFINITION_WORKFLOW_PATH = Path(
    ".github/workflows/belief-history-v1-definition.yml"
)
DEFINITION_REQUIREMENTS_PATH = Path(
    "data/experiments/belief_history_v1_definition_python_requirements.txt"
)
NATIVE_STRUCTURAL_TEST_PATH = Path(
    "tests/test_history_belief_exclusion.c"
)
NATIVE_STRUCTURAL_SMOKE_ROOT = "202706290103"

INTEGRATION_PATHS = (
    Path("README.md"),
    Path("tools/belief_eval.c"),
    Path("tools/rl.c"),
    Path("tests/test_action_core_campaign.py"),
    Path("tests/test_belief_eval.c"),
    Path("tests/test_belief_eval.py"),
    Path("tests/test_rl_population.py"),
)

# Exact current expansion of history_belief_train's $(CORE) and $(HDRS)
# prerequisites in Makefile.  Keep these explicit in the execution binding;
# the definition trigger also carries a src/*.h guard so a newly added header
# cannot enter $(HDRS) without forcing a new exact-parent definition run.
HISTORY_BELIEF_TRANSITIVE_PATHS = (
    Path("src/lc.c"),
    Path("src/features.c"),
    Path("src/net.c"),
    Path("src/heuristic.c"),
    Path("src/planner.c"),
    Path("src/search.c"),
    Path("src/rollout.c"),
    Path("src/late_resolver.c"),
    Path("src/match_value.c"),
    Path("src/policy_cost.c"),
    Path("src/agent.c"),
    Path("src/match.c"),
    Path("src/spec.c"),
    Path("src/agent.h"),
    Path("src/features.h"),
    Path("src/heuristic.h"),
    Path("src/history_belief_exclusion.h"),
    Path("src/history_belief_model.h"),
    Path("src/late_resolver.h"),
    Path("src/lc.h"),
    Path("src/match.h"),
    Path("src/match_value.h"),
    Path("src/net.h"),
    Path("src/planner.h"),
    Path("src/policy_cost.h"),
    Path("src/policy_cost_v3.h"),
    Path("src/policy_cost_v4.h"),
    Path("src/policy_cost_v5.h"),
    Path("src/policy_cost_v6.h"),
    Path("src/policy_cost_v7.h"),
    Path("src/search.h"),
    Path("src/spec.h"),
)

PRIOR_POLICY_COST_PLAN_PATHS = tuple(
    Path(f"data/experiments/locked_policy_cost_v{version}_plan.json")
    for version in range(1, 8)
)

# These are trigger guards, not literal files to bind.  They force definition
# validation if the Makefile's wildcard header set or the dynamically checked
# prior-plan inventory grows after this definition is frozen.
DEFINITION_TRIGGER_GUARD_PATTERNS = (
    "src/*.h",
    "data/experiments/locked_policy_cost_v*_plan.json",
)

BOUND_PATHS = (
    Path("BELIEF_HISTORY_V1.md"),
    *INTEGRATION_PATHS,
    Path("Makefile"),
    PLAN_PATH,
    EXCLUSIONS_PATH,
    TEMPLATE_PATH,
    WORKFLOW_PATH,
    DEFINITION_WORKFLOW_PATH,
    Path("data/champion.bin"),
    Path("data/experiments/belief_accuracy_dev_20260829.json"),
    Path("data/experiments/belief_history_v1_python_requirements.txt"),
    DEFINITION_REQUIREMENTS_PATH,
    Path("src/history_belief_model.c"),
    Path("src/history_belief_exclusion.c"),
    *HISTORY_BELIEF_TRANSITIVE_PATHS,
    Path("tools/history_belief_train.c"),
    Path("tools/belief_history_campaign.py"),
    Path("tools/belief_history_reduce.py"),
    Path("tests/test_history_belief_model.c"),
    NATIVE_STRUCTURAL_TEST_PATH,
    Path("tests/test_history_belief_train.py"),
    Path("tests/test_belief_history_campaign.py"),
    Path("tests/test_belief_history_reduce.py"),
    Path("data/experiments/commented_ply_audit_v3.json"),
    Path("data/experiments/commented_ply_audit_v3.md"),
    Path("data/experiments/commented_ply_audit_v3_result.json"),
    Path("data/experiments/commented_ply_audit_v3_evidence.zip"),
    Path("data/experiments/locked_commented_ply_audit_definition_lock_v3.json"),
    Path("data/experiments/policy_cost_v7_exact17_exclusions.json"),
    Path("data/experiments/policy_cost_v7_exact17_exclusions.txt"),
    *PRIOR_POLICY_COST_PLAN_PATHS,
)


class DefinitionError(ValueError):
    """The frozen campaign definition or execution is invalid."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def strict_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DefinitionError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                DefinitionError(f"non-finite JSON token {token} in {path}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DefinitionError(f"invalid canonical JSON in {path}: {exc}") from exc
    _finite(value, str(path))
    return value


def exact17_fixture_paths(root: Path) -> tuple[Path, ...]:
    """Discover test-only fixtures from the bound canonical definition."""
    exact17 = strict_json(
        root / "data/experiments/policy_cost_v7_exact17_exclusions.json")
    cases = exact17.get("cases")
    _require(isinstance(cases, list) and len(cases) == 17,
             "canonical exact17 case inventory changed")
    paths = tuple(Path(item.get("state_path", "")) for item in cases)
    _require(all(path.as_posix() and not path.is_absolute() and
                 ".." not in path.parts for path in paths),
             "canonical exact17 fixture path is invalid")
    return paths


def _finite(value: Any, where: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DefinitionError(f"non-finite number in {where}")
        return
    if isinstance(value, list):
        for item in value:
            _finite(item, where)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise DefinitionError(f"non-string JSON key in {where}")
        for item in value.values():
            _finite(item, where)
        return
    raise DefinitionError(f"unsupported JSON value in {where}")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError as exc:
        raise DefinitionError(f"cannot hash {path}: {exc}") from exc
    return h.hexdigest()


def binding(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise DefinitionError(f"required bound path is absent: {relative}")
    return {
        "path": relative.as_posix(),
        "sha256": digest(path),
        "size": path.stat().st_size,
    }


def verify_head_control(champion_path: Path, control_path: Path) -> dict[str, Any]:
    """Prove that a current-format Net changed only wbel/bbel bytes."""
    before = champion_path.read_bytes()
    after = control_path.read_bytes()
    _require(len(before) == len(after) and len(before) >= 24,
             "head-control checkpoint size changed")
    header = struct.unpack("=6I", before[:24])
    _require(after[:24] == before[:24], "head-control header changed")
    _, feature_dim, hidden1, hidden2, nplay, _ = header
    _require((hidden1, hidden2, nplay) == (512, 256, 120),
             "unexpected Net geometry")
    cards, draws, header_size = 60, 6, 24
    prefix_floats = (
        feature_dim * hidden1 + hidden1 + hidden1 * hidden2 + hidden2
        + hidden2 + 1 + nplay * hidden2 + nplay + draws * hidden2 + draws
    )
    belief_start = header_size + prefix_floats * 4
    belief_weight_bytes = cards * hidden2 * 4
    belief_end = belief_start + belief_weight_bytes + cards * 4
    expected_size = belief_end + (nplay * draws * hidden2 + nplay * draws) * 4
    _require(expected_size == len(before), "invalid Net payload size")
    _require(before[:belief_start] == after[:belief_start],
             "head-control changed trunk/policy/value bytes")
    _require(before[belief_end:] == after[belief_end:],
             "head-control changed combination bytes")
    _require(before[belief_start:belief_end] != after[belief_start:belief_end],
             "head-control did not change any belief byte")
    for offset in range(belief_start, belief_end, 4):
        _require(math.isfinite(struct.unpack_from("=f", after, offset)[0]),
                 "head-control contains a non-finite belief parameter")
    weights = after[belief_start:belief_start + belief_weight_bytes]
    biases = after[belief_start + belief_weight_bytes:belief_end]
    row_size = hidden2 * 4
    for suit in range(5):
        card = suit * 12
        row0 = weights[card * row_size:(card + 1) * row_size]
        bias0 = biases[card * 4:(card + 1) * 4]
        for physical in (1, 2):
            other = card + physical
            _require(row0 == weights[other * row_size:(other + 1) * row_size]
                     and bias0 == biases[other * 4:(other + 1) * 4],
                     "head-control broke physical-wager tying")
    return {
        "artifact_kind": "belief_history_v1_head_only_control_proof",
        "belief_byte_end": belief_end,
        "belief_byte_start": belief_start,
        "champion_sha256": digest(champion_path),
        "control_sha256": digest(control_path),
        "nonbelief_bytes_identical": True,
        "schema": "lc-belief-history-head-control-proof-v1",
        "wager_rows_tied": True,
    }


def seed_values(value: Any) -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for item in value.values():
            result.update(seed_values(item))
        return result
    if isinstance(value, list):
        result = set()
        for item in value:
            result.update(seed_values(item))
        return result
    if isinstance(value, str) and re.fullmatch(r"20\d{10}", value):
        return {value}
    return set()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DefinitionError(message)


def _verify_external_bindings(root: Path, exclusions: dict[str, Any]) -> None:
    bindings = exclusions.get("exact17", {}).get("canonical_bindings")
    _require(isinstance(bindings, list) and len(bindings) == 7,
             "exact17 must bind exactly seven canonical files")
    for item in bindings:
        relative = Path(item.get("path", ""))
        _require(relative.as_posix() and not relative.is_absolute(),
                 "invalid exact17 binding path")
        _require(digest(root / relative) == item.get("sha256"),
                 f"exact17 binding mismatch: {relative}")
    prior = exclusions["burned_seed_contract"]["policy_cost_v7_plan"]
    prior_path = root / prior["path"]
    _require(digest(prior_path) == prior["sha256"],
             "policy-cost-v7 seed-ledger binding mismatch")
    exact17 = strict_json(
        root / "data/experiments/policy_cost_v7_exact17_exclusions.json")
    cases = exact17.get("cases")
    case_paths = exact17_fixture_paths(root)
    for item, relative in zip(cases, case_paths):
        _require(digest(root / relative) == item.get("state_file_sha256"),
                 f"exact17 fixture binding mismatch: {relative}")
    development = exclusions["burned_seed_contract"][
        "belief_accuracy_development"]
    development_path = root / development["path"]
    _require(digest(development_path) == development["sha256"],
             "belief-accuracy development seed-ledger binding mismatch")
    development_value = strict_json(development_path)
    expected_roots = set(development["burned_roots"])
    _require(expected_roots == set(development_value["seed_disposition"]),
             "belief-accuracy development seed ledger is incomplete")
    _require(exclusions["burned_seed_contract"]
             ["reserved_namespace_prefixes"] == ["20260829"],
             "reserved belief-development namespace changed")
    _require(all(root.startswith("20260829") for root in
                 exclusions["burned_seed_contract"]
                 ["known_belief_development_roots"]),
             "known belief-development root escaped its burned namespace")
    _require(set(exclusions["burned_seed_contract"]["ad_hoc_roots"]) == {
                 "101", "202", "71001", "72002", "880001", "880002",
             }, "ad-hoc burned-root ledger changed")
    _require(exclusions["burned_seed_contract"]
             ["retired_prelaunch_roots"] == {
                 "202706100401":
                 "burned when a local compiled-row structural test accidentally evaluated match 0 before definition freeze; no campaign or accuracy gate ran",
             }, "prelaunch retired-root ledger changed")


def _verify_definition_input_inventory(root: Path) -> None:
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    core = re.search(r"(?ms)^CORE\s*:=\s*(.*?)(?=\n\n)", makefile)
    _require(core is not None,
             "cannot recover history trainer CORE prerequisites")
    core_paths = tuple(
        Path("src") / name
        for name in re.findall(r"\$\(SRC\)/([A-Za-z0-9_]+\.c)",
                               core.group(1))
    )
    header_paths = tuple(
        path.relative_to(root)
        for path in sorted((root / "src").glob("*.h"))
    )
    _require(core_paths + header_paths == HISTORY_BELIEF_TRANSITIVE_PATHS,
             "history trainer transitive source inventory changed")
    discovered_prior_plans = tuple(
        path.relative_to(root)
        for path in sorted((root / "data/experiments").glob(
            "locked_policy_cost_v*_plan.json"))
    )
    _require(discovered_prior_plans == PRIOR_POLICY_COST_PLAN_PATHS,
             "prior policy-cost plan inventory changed")
    _require(len(BOUND_PATHS) == len(set(BOUND_PATHS)) and
             all((root / path).is_file() for path in BOUND_PATHS),
             "definition binding inventory is incomplete or duplicated")
    definition_workflow = (root / DEFINITION_WORKFLOW_PATH).read_text(
        encoding="utf-8")
    trigger = definition_workflow.split("\npermissions:\n", 1)[0]
    _require("  push:\n" in trigger and
             "pull_request:" not in trigger and
             "workflow_dispatch" not in trigger and
             "branches: [agent/correctness-and-policy-upgrade]" in trigger,
             "definition workflow is not exact-branch push-only")
    push_paths = re.findall(r"(?m)^      - (.+)$", trigger)
    expected_paths = ({path.as_posix() for path in BOUND_PATHS} |
                      {path.as_posix() for path in exact17_fixture_paths(root)} |
                      set(DEFINITION_TRIGGER_GUARD_PATTERNS))
    _require(len(push_paths) == len(set(push_paths)) and
             set(push_paths) == expected_paths and
             EXECUTION_PATH.as_posix() not in push_paths,
             "definition workflow push filter does not match its bound inputs")


def validate_plan(root: Path, *, require_inert: bool = True) -> dict[str, Any]:
    plan = strict_json(root / PLAN_PATH)
    exclusions = strict_json(root / EXCLUSIONS_PATH)
    _verify_definition_input_inventory(root)
    _require(plan.get("schema") == "lc-belief-history-v1-plan-v1" and
             plan.get("artifact_schemas", {}).get("execution") ==
             "lc-belief-history-v1-execution-v1",
             "invalid belief-history-v1 plan schema")
    _require(plan.get("experiment") == "belief-history-v1",
             "wrong experiment")
    _require(plan.get("status") ==
             "definition_complete_inert_execution_addendum_absent",
             "definition is not inert")
    _require(exclusions.get("schema") ==
             "lc-belief-history-v1-exclusions-v1",
             "invalid exclusions schema")
    _require(exclusions.get("exact17", {}).get("case_count") == 17,
             "exact17 case count changed")
    _require(exclusions["exact17"]["training_use"] == "forbidden" and
             exclusions["exact17"]["natural_state_orbit_rejection_claimed"]
             is True,
             "active exact17 training firewall changed")
    manifest = next(
        item for item in exclusions["exact17"]["canonical_bindings"]
        if item["path"].endswith("policy_cost_v7_exact17_exclusions.txt")
    )
    _require(manifest["sha256"] ==
             "10034cf8b83aadf24fa0775e4dad2712573e1d84cbf364568ce6136682ac254c",
             "canonical exact17 manifest digest changed")
    _verify_external_bindings(root, exclusions)

    expected_matches = {
        "TEST": (65536, 16),
    }
    splits = plan.get("data", {}).get("splits", {})
    for stage, (matches, shards) in expected_matches.items():
        row = splits.get(stage, {})
        _require(row.get("matches") == matches, f"{stage} match count changed")
        _require(row.get("shards") == shards, f"{stage} shard count changed")
        _require(matches % shards == 0 and matches // shards == 4096,
                 f"{stage} shards are not fixed 4096-match blocks")
    train = splits.get("TRAIN", {})
    _require(train.get("history_matches") == 65536 and
             train.get("matched_control_additional_matches") == 65536 and
             train.get("base_control_matches") == 262144 and
             train.get("history_shards") == 1 and
             train.get("matched_control_shards") == 1 and
             train.get("base_control_shards") == 4 and
             train.get("history_root") == train.get("matched_control_root"),
             "TRAIN sample-size contract changed")
    _require("fixed states-per-match" in
             plan["data"]["state_count_contract"],
             "data-dependent state-count contract is absent")

    active = plan.get("seeds", {}).get("active_production_roots", [])
    smoke = plan.get("seeds", {}).get("definition_smoke_roots", [])
    _require(len(active) == 4 and len(set(active)) == 4,
             "production root count or uniqueness changed")
    _require(len(smoke) == 3 and len(set(smoke)) == 3,
             "smoke root count or uniqueness changed")
    _require(smoke == ["202706290101", "202706290102",
                       NATIVE_STRUCTURAL_SMOKE_ROOT],
             "definition smoke roots changed")
    _require(not set(active) & set(smoke), "production and smoke roots overlap")
    _require(all(re.fullmatch(r"202706\d{6}", value)
                 for value in [*active, *smoke]),
             "a root is outside the fresh 202706 namespace")
    previous: set[str] = set()
    for relative in PRIOR_POLICY_COST_PLAN_PATHS:
        previous.update(seed_values(strict_json(root / relative)))
    _require(not set(active) & previous and not set(smoke) & previous,
             "belief-history root overlaps a prior policy-cost plan")
    burned_development = set(exclusions["burned_seed_contract"]
                             ["belief_accuracy_development"]
                             ["burned_roots"])
    _require(not (set(active) | set(smoke)) & burned_development,
             "belief-history root overlaps burned belief development")
    _require(not any(value.startswith("20260829")
                     for value in [*active, *smoke]),
             "belief-history root overlaps reserved development namespace")
    _require(not (set(active) | set(smoke)) &
             set(exclusions["burned_seed_contract"]["ad_hoc_roots"]),
             "belief-history root overlaps an ad-hoc burned root")
    _require(not (set(active) | set(smoke)) &
             set(exclusions["burned_seed_contract"]
                 ["retired_prelaunch_roots"]),
             "belief-history root overlaps a retired prelaunch root")
    structural_source = (root / NATIVE_STRUCTURAL_TEST_PATH).read_text(
        encoding="utf-8")
    structural_seed_lines = [
        line.strip() for line in structural_source.splitlines()
        if "rng_seed(&rng" in line
    ]
    _require(structural_seed_lines == [
        f"rng_seed(&rng, UINT64_C({NATIVE_STRUCTURAL_SMOKE_ROOT}));"
    ], "native structural test seed drifted from its declared smoke root")

    gates = plan["evaluation"]
    _require(gates["bootstrap"] == {
        "method": "paired source-match cluster bootstrap with deterministic SplitMix64 resampling, marginal percentile bounds, and single-step simultaneous max-standardized-error one-sided bounds",
        "replicates": 20000,
        "seed": "202706150101",
        "simultaneous_familywise": {
            "components": 9,
            "confidence": 0.99,
            "coverage_claim": "nominal_asymptotic",
            "exact_finite_sample_coverage_claimed": False,
            "method": "single_step_max_standardized_error",
            "studentization": "fixed_original_source_match_cluster_se",
            "zero_standard_error_policy": "report null simultaneous LCB, mark inferentially ineligible, and fail the affected replacement bundle",
        },
        "unit": "source_match",
    }, "bootstrap contract changed")
    _require(gates["primary_gate"]["confidence"] == 0.99 and
             gates["primary_gate"]["relative_nll_improvement_at_least"] == 0.0 and
             gates["primary_gate"]["point_gain_strictly_above"] == 0.0 and
             gates["primary_gate"]["nll_lcb_strictly_above"] == 0.0,
             "primary accuracy gate changed")
    _require(gates["history_gate"]["confidence"] == 0.99 and
             gates["history_gate"]["relative_nll_improvement_at_least"] == 0.0 and
             gates["history_gate"]["point_gain_strictly_above"] == 0.0 and
             gates["history_gate"]["min_opponent_actions"] == 1,
             "history-specific gate changed")
    _require(gates["brier_gate"]["confidence"] == 0.99 and
             gates["brier_gate"]["point_gain_strictly_above"] == 0.0 and
             gates["brier_gate"]["lcb_strictly_above"] == 0.0,
             "Brier gate changed")
    _require(gates["stages"]["TEST"]["one_look"] is True and
             gates["stages"]["TEST"]["second_test_or_top_up"] is False,
             "untouched TEST contract changed")
    _require(gates["terminal_artifact_selection"] == {
        "comparison_bundle": "For each ordered replacement comparison, require the frozen all-state joint-NLL, post-opponent-action joint-NLL, and all-state per-card-Brier point/simultaneous-one-sided-nominal-99%-LCB gates directly. One max-standardized-error critical value controls the nine-component selection family asymptotically; exact finite-sample coverage is not claimed and marginal percentile bounds are report-only. A zero original cluster SE is inferentially ineligible and fails its bundle. Never infer a pair by transitivity.",
        "rule": [
            "retain history only if history passes directly against both matched_head_control and incumbent_head",
            "otherwise retain matched_head_control only if it passes directly against incumbent_head",
            "otherwise retain incumbent_head",
        ],
        "selected_artifact_is_playing_actor": False,
    }, "terminal artifact comparison bundle changed")
    _require(isinstance(gates.get("structural_gates"), dict) and
             set(gates["structural_gates"]) == {
                 "artifact_provenance", "exact_k", "head_only_control",
                 "hidden_information", "opening_uniform", "probe_firewall",
                 "suit_equivariance", "wager_tying",
             }, "structural gate set changed")

    _require(plan["models"]["candidate"]["playing_actor_bytes_changed"] is False,
             "history model is not accuracy-only")
    _require(plan["models"]["candidate"]["base_alpha"] == 1.0,
             "history candidate base alpha changed")
    _require(plan["models"]["candidate"].get("learning_rate") == 0.01 and
             plan["models"]["candidate"].get("l2") == 0.0000001 and
             "--lr 0.01 --l2 0.0000001" in
             plan["models"]["candidate"]["train_command_contract"],
             "history candidate optimizer drifted from development selection")
    _require(plan["models"]["candidate"]
             ["base_reconstruction_marginal_tolerance"] == 0.000002,
             "history base reconstruction tolerance changed")
    _require(plan["models"]["base_head_only_control"]
             ["playing_actor_bytes_changed"] is False,
             "base head control is not byte-isolated")
    _require(plan["models"]["matched_head_only_control"]
             ["playing_actor_bytes_changed"] is False and
             plan["models"]["matched_head_only_control"]
             ["primary_test_comparator"] is True,
             "matched head control is not the accuracy-only comparator")
    _require(plan["models"]["base_head_only_control"]["base_alpha"] == 1.0 and
             plan["models"]["matched_head_only_control"]["base_alpha"] == 1.0,
             "a control base alpha changed")
    _require(plan["models"]["base_head_only_control"]["shard_starts"] ==
             [0, 65536, 131072, 196608],
             "ordered base-control ranges changed")
    base_control = plan["models"]["base_head_only_control"]
    _require(base_control["control_batch_states"] == 256 and
             base_control.get("learning_rate") == 0.0001 and
             base_control.get("l2") == 0.0000001 and
             "--lr 0.0001 --l2 0.0000001" in
             base_control["shard_command_contract"] and
             base_control["first_shard_omits_control_state_in"] is True and
             base_control["final_shard_adds_control_finalize"] is True and
             base_control["intermediate_shards_forbid_control_finalize"] is True,
             "base-control resumable/finalization contract changed")
    matched_control = plan["models"]["matched_head_only_control"]
    matched_command = matched_control["train_command_contract"]
    expected_matched_command = (
        "bin/history_belief_train train-control --out "
        "matched-control-327680.bin --control-state-out "
        "matched-control-327680.state --actor-net data/champion.bin "
        "--base-net base-control-262144.bin --matches 65536 --rounds 3 "
        "--seed 202706100101 --match-start 0 --max-ply 300 "
        "--symmetries 20 --temperature 0.03 --base-alpha 1.0 --epochs 1 "
        "--lr 0.00015 --l2 0.0000001 --control-batch-states 256 "
        "--control-finalize --exclusions "
        "data/experiments/policy_cost_v7_exact17_exclusions.txt "
        "--exclusions-sha256 "
        "10034cf8b83aadf24fa0775e4dad2712573e1d84cbf364568ce6136682ac254c"
    )
    _require(matched_control.get("learning_rate") == 0.00015 and
             matched_control.get("l2") == 0.0000001 and
             matched_command == expected_matched_command,
             "matched-control command drifted from development selection")
    _require(plan["models"]["incumbent_head"]["alpha"] == 1.15 and
             plan["models"]["incumbent_head"]["sha256"] ==
             plan["data"]["source_actor"]["actor_net"]["sha256"],
             "maintained incumbent-head contract changed")
    for model in ("candidate", "base_head_only_control",
                  "matched_head_only_control"):
        command = " ".join(str(value) for key, value in
                           plan["models"][model].items()
                           if key.endswith("command_contract"))
        _require("--base-alpha 1.0" in command,
                 f"{model} does not explicitly bind base alpha 1.0")
        _require("bin/rl" not in command and
                 "--exclusions-sha256" in command,
                 f"{model} bypasses the exclusion-aware native trainer")
    _require("match playing-strength evaluation" in plan["non_goals"],
             "playing-strength non-goal is absent")
    _require(plan["probe_firewall"]["selection_use"] ==
             "not_applicable_no_select_split",
             "exact17 selection firewall changed")
    _require(plan["probe_firewall"]
             ["natural_state_orbit_rejection_claimed"] is True,
             "active exact17 orbit rejection is absent")
    _require(plan["execution_protocol"]["manual_dispatch"] is False and
             plan["execution_protocol"]["no_retry"] is True and
             plan["execution_protocol"]["no_seed_reuse"] is True,
             "one-shot execution contract changed")
    throughput = plan["execution_protocol"]["pre_efficacy_throughput_guard"]
    _require(throughput == {
        "candidate_job_timeout_minutes": 360,
        "control_matches": 100,
        "history_matches": 100,
        "minimum_control_matches_per_second": 3.5,
        "minimum_history_matches_per_second": 3.5,
        "purpose": "Fail before any production root is touched unless source-free history and head-control smokes demonstrate that every fixed 65,536-match training or evaluation job fits the hosted-job budget with reserve.",
        "smoke_root": "202706290101",
    }, "pre-efficacy throughput guard changed")

    actor = plan["data"]["source_actor"]["actor_net"]
    _require(digest(root / actor["path"]) == actor["sha256"],
             "source actor checkpoint binding mismatch")
    workflow = (root / WORKFLOW_PATH).read_text(encoding="utf-8")
    _require("workflow_dispatch" not in workflow and "pull_request:" not in workflow,
             "campaign workflow is not push-only")
    _require("locked_belief_history_v1_execution.json" in workflow,
             "workflow does not bind the unique execution path")
    _require("-fno-fast-math -ffp-contract=off" in workflow,
             "native numeric build flags are not frozen")
    _require("Python 3.12.3" not in workflow and "= 13.3.0" not in workflow and
             "sys.version_info[:2] == (3, 12)" in workflow and
             "gcc -dumpversion | cut -d. -f1" in workflow,
             "preflight pins mutable runner patch versions")
    _require("--no-deps --target python-runtime \"$NUMPY_WHEEL\"" in workflow and
             "export PYTHONNOUSERSITE=1 PYTHONPATH=\"$PY_RUNTIME\"" in workflow and
             "actual.is_relative_to(expected)" in workflow and
             "python-dependency-proof.json" in workflow and
             "python_dependency_proof_sha256" in workflow and
             "'numpy_wheel_sha256':'0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28'" in workflow,
             "NumPy-dependent preflight is not isolated to the pinned wheel")
    heredocs = list(re.finditer(
        r"(?ms)^( +)python3 - <<'PY'\n(.*?)^\1PY$", workflow))
    _require(len(heredocs) == 5,
             "workflow Python heredoc inventory drifted")
    for index, match in enumerate(heredocs):
        indent = match.group(1)
        lines = match.group(2).splitlines()
        _require(all(not line or line.startswith(indent) for line in lines),
                 f"workflow Python heredoc {index} indentation drifted")
        source = "\n".join(
            line[len(indent):] if line else "" for line in lines
        ) + "\n"
        try:
            compile(source, f"{WORKFLOW_PATH}:heredoc-{index}", "exec")
        except SyntaxError as exc:
            raise DefinitionError(
                f"workflow Python heredoc {index} does not compile: {exc}"
            ) from exc
    job_names = (
        "base_control_1", "base_control_2", "base_control_3",
        "base_control_4", "history_train", "matched_control_train",
        "test_evaluate",
    )
    sections = {}
    for name in job_names:
        match = re.search(
            rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_]+:\n|\Z)",
            workflow,
        )
        _require(match is not None, f"workflow job {name} is missing")
        sections[name] = match.group(1)
    _require(all("--control-finalize" not in sections[name]
                 for name in ("base_control_1", "base_control_2",
                              "base_control_3")) and
             "--control-finalize" in sections["base_control_4"] and
             "--control-batch-states 256" in sections["base_control_4"] and
             all("--lr 0.0001 --l2 0.0000001" in sections[name]
                 for name in ("base_control_1", "base_control_2",
                              "base_control_3", "base_control_4")),
             "base-control finalization is not last-shard-only")
    matched_section = sections["matched_control_train"]
    matched_command_fragments = (
        "train-control --out out/matched-control.bin",
        "--control-state-out out/matched-control.state",
        "--actor-net runtime/data/champion.bin --base-net base/control.bin",
        "--matches 65536 --rounds 3 --seed 202706100101 --match-start 0",
        "--max-ply 300 --symmetries 20 --temperature 0.03 --base-alpha 1.0",
        "--epochs 1 --lr 0.00015 --l2 0.0000001",
        "--control-batch-states 256 --control-finalize",
        "--exclusions runtime/bindings/exact17/exclusions.txt",
        "--exclusions-sha256 \"$EXCLUSIONS_SHA256\"",
    )
    _require(all(fragment in matched_section
                 for fragment in matched_command_fragments) and
             "--control-state-in" not in matched_section,
             "matched control does not start a fresh finalized optimizer")
    _require("--control-batch-states" not in sections["history_train"] and
             "--control-finalize" not in sections["history_train"] and
             "--lr 0.01 --l2 0.0000001" in sections["history_train"],
             "residual training received control-only flags")
    _require("--incumbent-alpha 1.15" in sections["test_evaluate"],
             "TEST does not bind the maintained incumbent alpha")
    _require(workflow.count("--incumbent-alpha 1.15") >= 3,
             "smoke and TEST do not all bind incumbent alpha 1.15")
    _require("'incumbent_alpha':1.15" in workflow and
             "'incumbent_net_fingerprint':h['actor_fingerprint']" in workflow,
             "sealed TEST identity omits incumbent provenance")
    _require("control-chain-manifest.json" in workflow and
             "cp -a runtime complete/runtime" in workflow and
             "cp -a shards complete/test-shards" in workflow,
             "terminal artifact is not independently replayable")
    _require("summary['trained_state_count'] - previous_trained == summary['source_state_count']" in workflow and
             "summary['optimizer_steps'] - previous_steps == expected_step_delta" in workflow and
             "m['trained_state_count'] == m['source_state_count']" in workflow,
             "control chain does not prove exactly-once label consumption")
    _require("h['model_train_states'] == h['source_state_count']" in workflow and
             "h['output_sha256'] == sha('history/history-model.bin')" in workflow and
             "f32(h['training_learning_rate']) == f32(0.01)" in workflow and
             "h['training_l2']" in workflow and
             "f32(m['lr']) == f32(0.00015)" in workflow and
             "f32(m['l2']) == f32(0.0000001)" in workflow,
             "history training receipt is not fully frozen")
    matched_receipt_fragments = (
        "m['schema'] == 'lc-history-belief-control-run-v1'",
        "m['mode'] == 'train-control'",
        "m['next_match_start'] == 65536",
        "h['rounds'] == m['rounds'] == 3",
        "h['max_scored_ply'] == m['max_scored_ply'] == 300",
        "h['symmetries'] == m['symmetries'] == 20",
        "f32(h['temperature']) == f32(m['temperature']) == f32(0.03)",
        "m['control_state_source_manifest_scope'] == 'current_invocation'",
        "m['playing_actor_changed'] is False",
        "m['control_changed_only_belief_head'] is True",
        "m['training_augmentation'] == 'one deterministic scheduled member per state from the declared suit group'",
        "summary['input_checkpoint_sha256'] == sha('runtime/data/champion.bin')",
        "summary['input_checkpoint_sha256'] == previous_output_sha256",
        "summary['control_state_checkpoint_sha256'] == summary['output_sha256']",
        "m['input_checkpoint_sha256'] == sha('base/control.bin')",
        "m['control_state_checkpoint_sha256'] == m['output_sha256']",
    )
    _require(all(fragment in workflow for fragment in matched_receipt_fragments),
             "matched-control receipt is not fully frozen")
    for forbidden in ("policy-cost-v7.yml", "locked_policy_cost_v7_execution.json"):
        _require(forbidden not in workflow,
                 "belief workflow reaches the live policy-cost campaign")
    if require_inert:
        _require(not (root / EXECUTION_PATH).exists(),
                 "execution addendum exists in an inert definition")
    return plan


def _execution_payload(root: Path, source_parent_commit: str,
                       source_parent_tree: str) -> dict[str, Any]:
    plan = validate_plan(root, require_inert=False)
    _require(re.fullmatch(r"[0-9a-f]{40}", source_parent_commit) is not None,
             "invalid source parent commit")
    _require(re.fullmatch(r"[0-9a-f]{40}", source_parent_tree) is not None,
             "invalid source parent tree")
    return {
        "artifact_kind": "locked_belief_history_v1_execution",
        "bindings": [binding(root, path) for path in BOUND_PATHS],
        "definition_validation_prerequisite": {
            "conclusion": "success",
            "enforcement": (
                "Externally verify the exact-parent GitHub Actions result "
                "before committing this addendum; Git history alone cannot "
                "attest an Actions conclusion."
            ),
            "event": "push",
            "head_sha": source_parent_commit,
            "head_tree": source_parent_tree,
            "required_attempt": 1,
            "workflow": binding(root, DEFINITION_WORKFLOW_PATH),
        },
        "experiment": "belief-history-v1",
        "fixed_roots": plan["seeds"]["active_production_roots"],
        "github": {
            "branch": "agent/correctness-and-policy-upgrade",
            "event": "push",
            "required_attempt": 1,
        },
        "results": None,
        "schema": "lc-belief-history-v1-execution-v1",
        "source_parent_commit": source_parent_commit,
        "source_parent_tree": source_parent_tree,
        "status": "authorized_unstarted_one_shot_accuracy_campaign",
    }


def prepare_execution(root: Path, output: Path, source_parent_commit: str,
                      source_parent_tree: str) -> None:
    _require(not output.exists(), "execution output already exists")
    value = _execution_payload(root, source_parent_commit, source_parent_tree)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(value))


def guard_execution(root: Path, execution_path: Path,
                    source_parent_commit: str,
                    source_parent_tree: str,
                    check_git_child: bool) -> dict[str, Any]:
    value = strict_json(execution_path)
    expected = _execution_payload(root, source_parent_commit, source_parent_tree)
    _require(value == expected, "execution binding is not canonical or current")
    _require(execution_path.read_bytes() == canonical_bytes(value),
             "execution file is not canonical JSON")
    if check_git_child:
        parent = subprocess.check_output(
            ["git", "rev-parse", "HEAD^"], cwd=root, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^^{tree}"], cwd=root, text=True
        ).strip()
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        _require(parent == source_parent_commit, "launch is not a direct child")
        _require(tree == source_parent_tree, "source parent tree changed")
        _require(changed == f"A\t{EXECUTION_PATH.as_posix()}",
                 "launch commit is not execution-addendum-only")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-plan")
    validate.add_argument("--root", type=Path, default=Path("."))
    prepare = sub.add_parser("prepare-execution")
    prepare.add_argument("--root", type=Path, default=Path("."))
    prepare.add_argument("--output", type=Path, default=EXECUTION_PATH)
    prepare.add_argument("--source-parent-commit", required=True)
    prepare.add_argument("--source-parent-tree", required=True)
    guard = sub.add_parser("guard-execution")
    guard.add_argument("--root", type=Path, default=Path("."))
    guard.add_argument("--execution", type=Path, default=EXECUTION_PATH)
    guard.add_argument("--source-parent-commit", required=True)
    guard.add_argument("--source-parent-tree", required=True)
    guard.add_argument("--check-git-child", action="store_true")
    head = sub.add_parser("verify-head-control")
    head.add_argument("--champion", type=Path, required=True)
    head.add_argument("--control", type=Path, required=True)
    head.add_argument("--proof", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate-plan":
            validate_plan(args.root)
        elif args.command == "prepare-execution":
            prepare_execution(args.root, args.output,
                              args.source_parent_commit,
                              args.source_parent_tree)
        elif args.command == "guard-execution":
            guard_execution(args.root, args.execution,
                            args.source_parent_commit,
                            args.source_parent_tree,
                            args.check_git_child)
        else:
            proof = verify_head_control(args.champion, args.control)
            payload = canonical_bytes(proof)
            if args.proof:
                args.proof.write_bytes(payload)
            else:
                sys.stdout.buffer.write(payload)
    except (DefinitionError, OSError, subprocess.CalledProcessError) as exc:
        print(f"belief-history-v1: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
