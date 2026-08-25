#!/usr/bin/env python3
"""Fail-closed orchestration for the locked objective-3 match-value v2 campaign.

This campaign is deliberately separate from the historical match-value
factorial.  It derives its baseline mechanically from the authoritative final
actor result, treats the completed exact-17 audit as provenance only, builds a
single raw/projected table pair, and evaluates exactly two all-ply actors.

Every result-dependent operation used by the workflow lives here.  The module
never dispatches a workflow, mutates a branch, widens a candidate set, or
permits partial evidence to select an actor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.flagged_ply_execution import (  # noqa: E402
    ExecutionError as EvidenceError,
    authoritative_final_result,
    sha256,
    strict_json,
)
from tools.gate_actor_panel import _rebuild_reciprocal  # noqa: E402
from tools.merge_arena import _write_json  # noqa: E402


PLAN_SCHEMA = "lc-match-value-objective3-v2-plan-v1"
LOCK_SCHEMA = "lc-match-value-objective3-v2-definition-lock-v1"
EXECUTION_SCHEMA = "lc-match-value-objective3-v2-execution-v1"
TABLE_SCHEMA = "lc-controller-bound-match-value-table-pair-v2"
MANIFEST_SCHEMA = "lc-match-value-objective3-v2-pre-efficacy-manifest-v1"
DEVELOPMENT_SCHEMA = "lc-match-value-objective3-v2-development-selection-v1"
RESULT_SCHEMA = "lc-match-value-objective3-v2-result-v1"

PLAN_PATH = "data/experiments/locked_match_value_objective3_v2_plan.json"
LOCK_PATH = (
    "data/experiments/locked_match_value_objective3_v2_definition_lock.json"
)
EXECUTION_PATH = (
    "data/experiments/locked_match_value_objective3_v2_execution.json"
)
WORKFLOW_PATH = ".github/workflows/match-value-objective3-v2.yml"
HELPER_PATH = "tools/match_value_objective3_v2.py"
TEST_PATH = "tests/test_match_value_objective3_v2.py"
DOC_PATH = "MATCH_VALUE_OBJECTIVE3_V2.md"
FINAL_RESULT_PATH = "data/experiments/final_actor_result.json"
AUDIT_RESULT_PATH = "data/experiments/commented_ply_audit_v3_result.json"
AUDIT_JSON_PATH = "data/experiments/commented_ply_audit_v3.json"
AUDIT_MARKDOWN_PATH = "data/experiments/commented_ply_audit_v3.md"
AUDIT_EVIDENCE_PATH = "data/experiments/commented_ply_audit_v3_evidence.zip"
MODEL_PATH = "data/champion.bin"
RAW_TABLE_PATH = "data/models/match_value_objective3_v2_raw.lcmv"
PROJECTED_TABLE_PATH = (
    "data/models/match_value_objective3_v2_projected.lcmv"
)
RESULT_PATH = "data/experiments/match_value_objective3_v2_result.json"
DEFINITION_PARENT_COMMIT = "42c89f554a92269ce6051a2808d4fb495530c37e"
DEFINITION_PARENT_TREE = "e594c2d8609b7f28e75b8cc3a5ae661448b9385e"

MODEL_SHA256 = (
    "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"
)
FINAL_RESULT_SHA256 = (
    "5ad3566c5fb66aa45efa0c3b45eff6bde83eacece2eefd13a6794eb57530e476"
)
AUDIT_SHA256S = {
    AUDIT_RESULT_PATH:
        "9897b402116b897942031ecbd46c50127b358c5a2c579e9c854720c667f55a82",
    AUDIT_JSON_PATH:
        "be63dcae2ae1a179cf43a0c47e9971755290f9b3bfd90cc40fd4b6bd2838bbd7",
    AUDIT_MARKDOWN_PATH:
        "a30eb93e4623e75ec2dae4c2cb73103b801d67954e0b02298e1a5b1082ebcd71",
    AUDIT_EVIDENCE_PATH:
        "aacec0f3da9bbedd5d6512cf7bf2ef0d993232ed888d64f2e8099f5b15c03994",
}
MAINTAINED_ACTOR = (
    "rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
WORLD_CAP = 800

VARIANTS = ("RAW_ALL_PLY", "PROJECTED_ALL_PLY")
TIE_PRIORITY = ("PROJECTED_ALL_PLY", "RAW_ALL_PLY")

TABLE_SAMPLES = 16000
TABLE_TOTAL_ROUND_SIMULATIONS = 9_632_000
TABLE_THREADS = 8
TABLE_SEED = "202610200001"
TABLE_ROLE_CYCLE = 400
TABLE_CONTROLLER_ABI = 1
TABLE_CONTROLLER_WORDS = [0, 20, 4, 1, 1, 0, 0, 0, 0, 0, 300]
BUILD_PROFILE_HEX = "0030d23b"

DEVELOPMENT_PAIRS = 1000
DEVELOPMENT_SEEDS = ("202610200101", "202610200102")
SAFETY_PAIRS = 200
SAFETY_SEEDS = ("202610210101", "202610210102")
FINAL_PAIRS = 2500
FINAL_SEEDS = ("202610220101", "202610220102")
SMOKE_NAMESPACE = "20261029"
CRITICAL_Z = 1.645

COMPILER = "gcc"
COMPILER_SEMANTIC_VERSION_COMMAND = "gcc -dumpfullversion -dumpversion"
REQUIRED_COMPILER_SEMANTIC_VERSION = "13.3.0"
CFLAGS = (
    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
    "-Wall -Wextra -std=c11"
)
LDFLAGS = "-lm -pthread"

PREDECESSOR_V1 = {
    ".github/workflows/match-value-variant.yml":
        "6e2fc483671912b9c5f7ddfdcfb27399f0e44f90f79bda0b0128a6dd709e9683",
    "data/experiments/match_value_variant_plan.json":
        "30b1f5ee0ef8565bdb6b5182b86ad7236fbd1774eeed5097560f322e7314a001",
    "data/experiments/locked_match_value_variant_execution.template.json":
        "fc1ed86c1a55ebbc73309e8c729516cc6b1a101b218f11cec90ba9b6187b084d",
    "tools/match_value_campaign.py":
        "96f0783374f8e26f2f79008e5976d9db1130157feed092ea519c5c5911ce4690",
}
PREDECESSOR_EXECUTION_PATH = (
    "data/experiments/locked_match_value_variant_execution.json"
)

DEFINITION_PATHS = (
    WORKFLOW_PATH,
    DOC_PATH,
    PLAN_PATH,
    TEST_PATH,
    HELPER_PATH,
)
DEPENDENCY_PATHS = (
    "Makefile",
    "src/agent.c",
    "src/agent.h",
    "src/features.c",
    "src/features.h",
    "src/heuristic.c",
    "src/heuristic.h",
    "src/late_resolver.c",
    "src/late_resolver.h",
    "src/lc.c",
    "src/lc.h",
    "src/match.c",
    "src/match.h",
    "src/match_value.c",
    "src/match_value.h",
    "src/net.c",
    "src/net.h",
    "src/planner.c",
    "src/planner.h",
    "src/rollout.c",
    "src/search.c",
    "src/search.h",
    "src/spec.c",
    "src/spec.h",
    "tools/arena.c",
    "tools/build_match_value.c",
    "tools/flagged_ply_execution.py",
    "tools/gate_actor_panel.py",
    "tools/match_value_campaign.py",
    "tools/merge_arena.py",
    "tools/validate_actor_shards.py",
)
AUDIT_PATHS = (
    AUDIT_RESULT_PATH,
    AUDIT_JSON_PATH,
    AUDIT_MARKDOWN_PATH,
    AUDIT_EVIDENCE_PATH,
)

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_binding(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"required regular file is absent: {name}")
    return {"path": name, "sha256": sha256(path), "size": path.stat().st_size}


def _git_blob_binding(root: Path, commit: str, name: str) -> dict[str, Any]:
    try:
        row = subprocess.check_output(
            ["git", "ls-tree", commit, "--", name], cwd=root, text=True,
            stderr=subprocess.STDOUT,
        ).rstrip("\n")
        metadata, found = row.split("\t", 1)
        mode, kind, blob = metadata.split()
        if found != name or kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("not a regular blob")
        payload = subprocess.check_output(
            ["git", "cat-file", "blob", blob], cwd=root,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise EvidenceError(f"cannot bind {name} at {commit}: {exc}") from exc
    return {
        "path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "git_mode": mode,
    }


def _commit_tree(root: Path, commit: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"{commit}^{{tree}}"], cwd=root,
            text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError(f"cannot resolve commit/tree {commit}") from exc


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
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
        raise EvidenceError(f"refusing to replace immutable file {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _expect_schedule(section: Any, *, pairs: int, seeds: tuple[str, str],
                     per_shard: int, total_shards: int) -> None:
    if not isinstance(section, dict):
        raise EvidenceError("campaign stage is not an object")
    starts = list(range(0, pairs, per_shard))
    stage_seeds = section.get("seeds")
    if section.get("pairs_per_orientation") != pairs or \
            stage_seeds != {
                "candidate_first": seeds[0], "baseline_first": seeds[1]} or \
            section.get("pairs_per_shard") != per_shard or \
            section.get("shard_starts") != starts or \
            section.get("total_raw_shards") != total_shards:
        raise EvidenceError("campaign stage schedule drift")


def validate_plan(value: dict[str, Any]) -> None:
    """Validate the immutable scientific and one-shot protocol."""
    if value.get("schema") != PLAN_SCHEMA or \
            value.get("experiment") != "match-value-objective3-v2" or \
            value.get("status") != "definition_source_pending_unique_seal" or \
            value.get("scope") != \
            "development_selection_then_safety_then_reserved_final":
        raise EvidenceError("objective-3 v2 plan identity drift")
    if value.get("artifact_schemas") != {
        "definition_lock": LOCK_SCHEMA,
        "development_selection": DEVELOPMENT_SCHEMA,
        "execution": EXECUTION_SCHEMA,
        "plan": PLAN_SCHEMA,
        "pre_efficacy_manifest": MANIFEST_SCHEMA,
        "result": RESULT_SCHEMA,
        "table_pair": TABLE_SCHEMA,
    }:
        raise EvidenceError("objective-3 v2 artifact schema registry drift")
    if value.get("definition_lock", {}).get("path") != LOCK_PATH:
        raise EvidenceError("plan does not name the canonical definition lock")
    seal = value["definition_lock"]
    if seal.get("definition_parent_commit") != DEFINITION_PARENT_COMMIT or \
            seal.get("definition_parent_tree") != DEFINITION_PARENT_TREE or \
            seal.get("definition_files") != list(DEFINITION_PATHS):
        raise EvidenceError("definition parent or exact five-file set drift")

    actor = value.get("authoritative_actor")
    if not isinstance(actor, dict) or actor.get("path") != FINAL_RESULT_PATH or \
            actor.get("sha256") != FINAL_RESULT_SHA256 or \
            actor.get("actor") != MAINTAINED_ACTOR or \
            actor.get("checkpoint_path") != MODEL_PATH or \
            actor.get("checkpoint_sha256") != MODEL_SHA256 or \
            "mechanically validate" not in str(actor.get("derivation")):
        raise EvidenceError("authoritative actor binding drift")
    audit = value.get("diagnostic_audit")
    if not isinstance(audit, dict) or \
            audit.get("result") != {
                "path": AUDIT_RESULT_PATH,
                "sha256": AUDIT_SHA256S[AUDIT_RESULT_PATH],
            } or audit.get("json") != {
                "path": AUDIT_JSON_PATH,
                "sha256": AUDIT_SHA256S[AUDIT_JSON_PATH],
            } or audit.get("markdown") != {
                "path": AUDIT_MARKDOWN_PATH,
                "sha256": AUDIT_SHA256S[AUDIT_MARKDOWN_PATH],
            } or audit.get("evidence") != {
                "path": AUDIT_EVIDENCE_PATH,
                "sha256": AUDIT_SHA256S[AUDIT_EVIDENCE_PATH],
            } or \
            audit.get("selection_use") != "forbidden":
        raise EvidenceError("diagnostic audit isolation drift")

    predecessor = value.get("predecessor_v1")
    if not isinstance(predecessor, dict) or \
            predecessor.get("execution_path") != PREDECESSOR_EXECUTION_PATH or \
            predecessor.get("execution_history_count") != 0:
        raise EvidenceError("predecessor disposition drift")
    if predecessor.get("execution_path_present") is not False or \
            predecessor.get("status") != \
            "immutable_inert_never_launched_and_never_to_be_backfilled" or \
            predecessor.get("bindings") != PREDECESSOR_V1:
        raise EvidenceError("predecessor immutable hashes drift")

    variants = value.get("variants")
    constructed = build_actors(
        MAINTAINED_ACTOR, WORLD_CAP, RAW_TABLE_PATH, PROJECTED_TABLE_PATH)
    expected_variants = {}
    for name, projected, table in (
            ("RAW_ALL_PLY", False, RAW_TABLE_PATH),
            ("PROJECTED_ALL_PLY", True, PROJECTED_TABLE_PATH)):
        expected_variants[name] = {
            "actor": constructed["actors"][name],
            "objective": 3,
            "phase": {"ply_lo": 0, "ply_hi": 0},
            "isotonic_projected": projected,
            "table_path": table,
        }
    if variants != expected_variants:
        raise EvidenceError("candidate set is not the exact two all-ply actors")

    build = value.get("artifact_build")
    if not isinstance(build, dict) or \
            build.get("samples_per_policy_lead") != TABLE_SAMPLES or \
            build.get("total_round_simulations") != \
            TABLE_TOTAL_ROUND_SIMULATIONS or \
            build.get("threads") != TABLE_THREADS or \
            build.get("seed") != TABLE_SEED or \
            build.get("playout_symmetries") != 20 or \
            build.get("role_cycle_size") != TABLE_ROLE_CYCLE or \
            build.get("role_balance") != "complete_20x20_product_cycles" or \
            build.get("single_invocation_emits_both_tables") is not True or \
            build.get("variants_share_identical_transition_histograms") is not True or \
            build.get("raw_path") != RAW_TABLE_PATH or \
            build.get("projected_path") != PROJECTED_TABLE_PATH or \
            build.get("model") != {"path": MODEL_PATH, "sha256": MODEL_SHA256} or \
            build.get("controller_abi") != TABLE_CONTROLLER_ABI or \
            build.get("controller_words") != TABLE_CONTROLLER_WORDS or \
            build.get("build_profile_hex") != BUILD_PROFILE_HEX or \
            build.get("cflags") != CFLAGS or build.get("ldflags") != LDFLAGS or \
            build.get("compiler") != {
                "executable": COMPILER,
                "required_semantic_version": REQUIRED_COMPILER_SEMANTIC_VERSION,
                "semantic_version_command": COMPILER_SEMANTIC_VERSION_COMMAND,
            } or build.get("build_command") != (
                "./bin/build_match_value --model data/champion.bin --out "
                f"{PROJECTED_TABLE_PATH} --raw-out {RAW_TABLE_PATH} --samples "
                "16000 --threads 8 --seed 202610200001 "
                "--playout-symmetries 20"):
        raise EvidenceError("table build contract drift")

    _expect_schedule(value.get("development"), pairs=DEVELOPMENT_PAIRS,
                     seeds=DEVELOPMENT_SEEDS, per_shard=100,
                     total_shards=40)
    _expect_schedule(value.get("safety"), pairs=SAFETY_PAIRS,
                     seeds=SAFETY_SEEDS, per_shard=20,
                     total_shards=20)
    _expect_schedule(value.get("final"), pairs=FINAL_PAIRS,
                     seeds=FINAL_SEEDS, per_shard=100,
                     total_shards=50)
    if value["development"].get("variants") != list(VARIANTS) or \
            value["final"].get("confidence", {}).get("critical_z") != CRITICAL_Z:
        raise EvidenceError("stage cardinality or critical value drift")
    ordinary_gate = {
        "combined_equal_weight_margin_strictly_above_zero": True,
        "combined_equal_weight_match_score_at_least": 0.5,
        "each_orientation_match_score_at_least": 0.475,
        "exact_validity_required": True,
        "zero_capped_rounds_required": True,
    }
    if value["development"].get("eligibility_gate") != ordinary_gate or \
            value["safety"].get("gate") != ordinary_gate:
        raise EvidenceError("development/safety criterion drift")
    if value["final"].get("promotion_gate") != {
        "combined_margin_pair_clustered_lcb_strictly_above": 0.0,
        "combined_match_score_pair_clustered_lcb_strictly_above": 0.5,
        "each_orientation_match_score_strictly_above": 0.5,
        "exact_validity_required": True,
        "zero_capped_rounds_required": True,
    } or value["final"].get("confidence") != {
        "critical_z": CRITICAL_Z,
        "estimator": (
            "mirrored-pair clusters within orientation; combine the two "
            "independent reciprocal orientations with equal weight"),
        "meaning": (
            "95% one-sided lower confidence bound, equivalently the lower "
            "endpoint of a 90% two-sided interval"),
    }:
        raise EvidenceError("reserved final criterion drift")
    if value["development"].get("selection_rule") != [
        "discard every ineligible variant",
        "maximize exact combined match-score numerator",
        "then maximize exact combined margin numerator",
        "then use the fixed tie priority PROJECTED_ALL_PLY before RAW_ALL_PLY",
    ] or value["development"].get(
            "identical_mirrored_pair_indices_across_variants") is not True:
        raise EvidenceError("development selection rule drift")

    firewall = value.get("seed_firewall")
    production = set((*DEVELOPMENT_SEEDS, *SAFETY_SEEDS, *FINAL_SEEDS,
                      TABLE_SEED))
    if not isinstance(firewall, dict) or \
            firewall.get("production_namespaces") != [
                "20261020", "20261021", "20261022"] or \
            firewall.get("smoke_namespace") != SMOKE_NAMESPACE or \
            firewall.get("smoke_seeds_are_never_production_seeds") is not True or \
            firewall.get("all_other_seed_namespaces_forbidden") is not True or \
            any(seed.startswith(SMOKE_NAMESPACE) for seed in production) or \
            len(production) != 7:
        raise EvidenceError("production/smoke seed firewall drift")

    protocol = value.get("execution_protocol")
    if not isinstance(protocol, dict) or \
            protocol.get("manual_dispatch") is not False or \
            protocol.get("no_retry") is not True or \
            protocol.get("no_seed_reuse") is not True or \
            protocol.get("optional_stopping") is not False or \
            protocol.get("no_partial_result_inspection") is not True or \
            protocol.get("no_repository_writes_from_workflow") is not True or \
            protocol.get("result_dependent_table_rebuild") is not False:
        raise EvidenceError("one-shot execution protocol drift")


def authoritative_audit_result(root: Path) -> dict[str, Any]:
    """Bind the completed exact-17 audit without allowing it to select."""
    result_path = root / AUDIT_RESULT_PATH
    for name, expected in AUDIT_SHA256S.items():
        if sha256(root / name) != expected:
            raise EvidenceError(f"authoritative exact-17 binding drift: {name}")
    value = strict_json(result_path)
    if result_path.read_text(encoding="utf-8") != \
            json.dumps(value, indent=2, sort_keys=True) + "\n":
        raise EvidenceError("audit result is not canonical JSON")
    run = value.get("run")
    audit = value.get("audit")
    artifacts = value.get("artifacts")
    persisted = value.get("persisted_evidence")
    bindings = value.get("bindings")
    jobs = value.get("jobs")
    if value.get("schema") != "lc-commented-ply-audit-v3-result-v1" or \
            value.get("artifact_kind") != \
            "locked_exact_17_commented_ply_audit_authoritative_result" or \
            value.get("status") != "complete_verified_diagnostic_only" or \
            value.get("attempt_id") != "v3" or \
            not isinstance(run, dict) or run.get("attempt") != 1 or \
            run.get("status") != "completed" or run.get("conclusion") != "success" or \
            run.get("job_count") != 19 or \
            not isinstance(audit, dict) or audit.get("cases_completed") != 17 or \
            audit.get("raw_shards") != 17 or \
            audit.get("total_paired_worlds") != 18432 or \
            audit.get("counterfactual_cap_hits") != 0 or \
            audit.get("exact_world_budget") is not True or \
            audit.get("policy20_full_remaining_match") is not True or \
            audit.get("fixed_k_belief_valid") is not True or \
            audit.get("diagnostic_only") is not True or \
            audit.get("promotion_use") != "forbidden" or \
            value.get("maintained_actor") != MAINTAINED_ACTOR or \
            not isinstance(bindings, dict) or \
            bindings.get("final_actor_result_sha256") != \
            sha256(root / FINAL_RESULT_PATH) or \
            run.get("head_sha") != \
            "330488423fad6f7a83fc1b1ace2bb8fae75449b1":
        raise EvidenceError("exact-17 audit is incomplete or selectable")
    if not isinstance(artifacts, list) or len(artifacts) != 19 or any(
            not isinstance(row, dict) or row.get("expired") is not False or
            row.get("digest") != "sha256:" + str(row.get("local_zip_sha256")) or
            _HEX64.fullmatch(str(row.get("local_zip_sha256"))) is None
            for row in artifacts):
        raise EvidenceError("exact-17 artifact inventory drift")
    artifact_names = [str(row.get("name")) for row in artifacts]
    if len(set(artifact_names)) != 19 or \
            sum("-case-" in name for name in artifact_names) != 17 or \
            "commented-ply-audit-v3-evaluator" not in artifact_names or \
            "commented-ply-audit-v3-complete-evidence" not in artifact_names:
        raise EvidenceError("exact-17 artifact names/cardinality drift")
    if not isinstance(jobs, list) or len(jobs) != 19 or any(
            not isinstance(row, dict) or row.get("status") != "completed" or
            row.get("conclusion") != "success" for row in jobs) or \
            sum(str(row.get("name", "")).startswith("exact commented ply ")
                for row in jobs) != 17:
        raise EvidenceError("exact-17 job completion inventory drift")
    expected_persisted = {
        path: sha256(root / path)
        for path in (AUDIT_JSON_PATH, AUDIT_MARKDOWN_PATH, AUDIT_EVIDENCE_PATH)
    }
    if not isinstance(persisted, list) or len(persisted) != 3 or {
        str(row.get("path")): str(row.get("sha256"))
        for row in persisted if isinstance(row, dict)
    } != expected_persisted:
        raise EvidenceError("persisted exact-17 evidence hash drift")
    if value.get("packaging_recovery", {}).get("efficacy_rerun") is not False or \
            value.get("packaging_recovery", {}).get(
                "reconstructed_top_manifest_check") is not True:
        raise EvidenceError("exact-17 packaging recovery is not fail-closed")
    return {
        "selection_use": "forbidden",
        "result": _canonical_binding(root, AUDIT_RESULT_PATH),
        "canonical_json": _canonical_binding(root, AUDIT_JSON_PATH),
        "canonical_markdown": _canonical_binding(root, AUDIT_MARKDOWN_PATH),
        "evidence": _canonical_binding(root, AUDIT_EVIDENCE_PATH),
        "run": {
            "id": run.get("id"), "attempt": 1,
            "head_sha": run.get("head_sha"), "conclusion": "success",
        },
        "cases": 17,
        "raw_shards": 17,
        "total_paired_worlds": 18432,
        "results": None,
    }


def _verify_predecessor(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path, expected in PREDECESSOR_V1.items():
        row = _canonical_binding(root, path)
        if row["sha256"] != expected:
            raise EvidenceError(f"immutable predecessor v1 drifted: {path}")
        rows.append(row)
    if (root / PREDECESSOR_EXECUTION_PATH).exists():
        raise EvidenceError("inert predecessor v1 unexpectedly has an execution")
    if (root / ".git").exists():
        try:
            history = subprocess.check_output(
                ["git", "rev-list", "--all", "--count", "--",
                 PREDECESSOR_EXECUTION_PATH], cwd=root, text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EvidenceError("cannot verify predecessor execution history") from exc
        if history != "0":
            raise EvidenceError("predecessor v1 execution history is not empty")
    return rows


def expected_definition_lock(root: Path, definition_commit: str,
                             definition_tree: str) -> dict[str, Any]:
    if _HEX40.fullmatch(definition_commit) is None or \
            _HEX40.fullmatch(definition_tree) is None:
        raise EvidenceError("definition commit/tree must be canonical SHA-1")
    if (root / ".git").exists() and _commit_tree(root, definition_commit) != \
            definition_tree:
        raise EvidenceError("definition commit/tree mismatch")
    plan_path = root / PLAN_PATH
    plan = strict_json(plan_path)
    if plan_path.read_text(encoding="utf-8") != \
            json.dumps(plan, indent=2, sort_keys=True) + "\n":
        raise EvidenceError("objective-3 v2 plan is not canonical JSON")
    validate_plan(plan)
    seal = plan["definition_lock"]
    if seal.get("definition_files") != list(DEFINITION_PATHS):
        raise EvidenceError("plan definition-file set/order drift")
    if (root / ".git").exists():
        try:
            parents = subprocess.check_output(
                ["git", "rev-list", "--parents", "-n", "1",
                 definition_commit], cwd=root, text=True,
                stderr=subprocess.STDOUT,
            ).split()
            changes = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--name-status", "-r",
                 definition_commit], cwd=root, text=True,
                stderr=subprocess.STDOUT,
            ).splitlines()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EvidenceError("cannot verify definition topology") from exc
        parent = str(seal.get("definition_parent_commit"))
        parent_tree = str(seal.get("definition_parent_tree"))
        expected_changes = sorted(f"A\t{name}" for name in DEFINITION_PATHS)
        if parents != [definition_commit, parent] or \
                _commit_tree(root, parent) != parent_tree or \
                sorted(changes) != expected_changes:
            raise EvidenceError(
                "definition must add exactly five files to its locked parent")
    final = authoritative_final_result(root)
    if final.get("sha256") != FINAL_RESULT_SHA256 or \
            final["winner"]["spec"] != MAINTAINED_ACTOR or \
            final["winner"].get("checkpoints") != [{
                "path": MODEL_PATH, "role": 0,
                "sha256": MODEL_SHA256, "size": (root / MODEL_PATH).stat().st_size,
            }]:
        raise EvidenceError("authoritative maintained actor family drift")
    audit = authoritative_audit_result(root)
    predecessor = _verify_predecessor(root)
    names = (*DEFINITION_PATHS, *DEPENDENCY_PATHS)
    if (root / ".git").exists():
        definition_files = [
            _git_blob_binding(root, definition_commit, name) for name in names
        ]
        for row in definition_files:
            current = _canonical_binding(root, row["path"])
            if current["sha256"] != row["sha256"] or \
                    current["size"] != row["size"]:
                raise EvidenceError(f"definition file drift: {row['path']}")
    else:
        definition_files = [
            {**_canonical_binding(root, name), "git_mode": "100755" if
             os.access(root / name, os.X_OK) else "100644"}
            for name in names
        ]
    locked = [_canonical_binding(root, FINAL_RESULT_PATH),
              _canonical_binding(root, MODEL_PATH)] + [
                  _canonical_binding(root, name) for name in AUDIT_PATHS
              ]
    return {
        "schema": LOCK_SCHEMA,
        "artifact_kind": "immutable_match_value_objective3_v2_definition_lock",
        "status": "sealed_before_table_build_or_match_value_efficacy",
        "definition": {"commit": definition_commit, "tree": definition_tree},
        "definition_files": definition_files,
        "locked_artifacts": locked,
        "authoritative_final_actor_result": final,
        "diagnostic_exact17_audit": audit,
        "predecessor_v1": {
            "status": "inert_never_executed",
            "execution_history_count": 0,
            "bindings": predecessor,
        },
        "build": {
            "runner": "ubuntu-24.04",
            "compiler": COMPILER,
            "compiler_semantic_version_command":
                COMPILER_SEMANTIC_VERSION_COMMAND,
            "required_compiler_semantic_version":
                REQUIRED_COMPILER_SEMANTIC_VERSION,
            "cflags": CFLAGS,
            "ldflags": LDFLAGS,
        },
        "results": None,
    }


def prepare_definition_lock(root: Path, output: Path, definition_commit: str,
                            definition_tree: str) -> dict[str, Any]:
    if output.resolve() != (root / LOCK_PATH).resolve():
        raise EvidenceError("definition lock must use the canonical path")
    if not (root / ".git").exists():
        raise EvidenceError("definition lock must be created in a git checkout")
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError("cannot resolve definition HEAD") from exc
    if head != definition_commit:
        raise EvidenceError("definition lock must seal checked-out HEAD")
    value = expected_definition_lock(root, definition_commit, definition_tree)
    _atomic_create(output, value)
    return value


def validate_definition_lock(root: Path, source_commit: str,
                             source_tree: str,
                             lock_commit_hint: str | None = None) -> tuple[
                                 dict[str, Any], dict[str, Any]]:
    path = root / LOCK_PATH
    lock = strict_json(path)
    if path.read_text(encoding="utf-8") != \
            json.dumps(lock, indent=2, sort_keys=True) + "\n":
        raise EvidenceError("definition lock is not canonical JSON")
    definition = lock.get("definition")
    if not isinstance(definition, dict):
        raise EvidenceError("definition lock has no source identity")
    expected = expected_definition_lock(
        root, str(definition.get("commit")), str(definition.get("tree")))
    if lock != expected:
        raise EvidenceError("definition lock differs from exact recomputation")
    if _HEX40.fullmatch(source_commit) is None or \
            _HEX40.fullmatch(source_tree) is None:
        raise EvidenceError("source parent commit/tree must be canonical SHA-1")
    if (root / ".git").exists():
        if _commit_tree(root, source_commit) != source_tree:
            raise EvidenceError("source parent commit/tree mismatch")
        try:
            history = subprocess.check_output(
                ["git", "log", "--all", "--format=%H", "--", LOCK_PATH],
                cwd=root, text=True, stderr=subprocess.STDOUT,
            ).splitlines()
            if len(history) != 1:
                raise EvidenceError("definition lock must have one history commit")
            lock_commit = history[0]
            parents = subprocess.check_output(
                ["git", "rev-list", "--parents", "-n", "1", lock_commit],
                cwd=root, text=True, stderr=subprocess.STDOUT,
            ).split()
            changes = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--name-status", "-r",
                 lock_commit], cwd=root, text=True,
                stderr=subprocess.STDOUT,
            ).splitlines()
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", lock_commit,
                 source_commit], cwd=root, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EvidenceError("cannot verify definition-lock topology") from exc
        if parents != [lock_commit, definition["commit"]] or \
                changes != [f"A\t{LOCK_PATH}"] or not ancestor:
            raise EvidenceError("definition lock is not the unique add-only child")
        if lock_commit_hint is not None and lock_commit_hint != lock_commit:
            raise EvidenceError("definition lock commit hint drift")
    else:
        if lock_commit_hint is None or _HEX40.fullmatch(lock_commit_hint) is None:
            raise EvidenceError("archive transport lacks lock commit hint")
        lock_commit = lock_commit_hint
    binding = {
        "path": LOCK_PATH,
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "lock_commit": lock_commit,
        "definition_commit": definition["commit"],
        "definition_tree": definition["tree"],
    }
    return binding, lock


def expected_execution(root: Path, source_commit: str,
                       source_tree: str,
                       definition_lock_binding: tuple[
                           dict[str, Any], dict[str, Any]] | None = None,
                       lock_commit_hint: str | None = None) -> dict[str, Any]:
    binding, lock = definition_lock_binding or validate_definition_lock(
        root, source_commit, source_tree, lock_commit_hint)
    plan = strict_json(root / PLAN_PATH)
    validate_plan(plan)
    final = authoritative_final_result(root)
    audit = authoritative_audit_result(root)
    if final != lock["authoritative_final_actor_result"] or \
            audit != lock["diagnostic_exact17_audit"]:
        raise EvidenceError("actor or diagnostic audit drifted after definition seal")
    return {
        "schema": EXECUTION_SCHEMA,
        "artifact_kind": "locked_match_value_objective3_v2_execution",
        "status": "launch_bound_before_table_build_or_efficacy",
        "source_parent_commit": source_commit,
        "source_parent_tree": source_tree,
        "definition_lock": binding,
        "plan": {"path": PLAN_PATH, "sha256": sha256(root / PLAN_PATH)},
        "workflow": {
            "path": WORKFLOW_PATH, "sha256": sha256(root / WORKFLOW_PATH),
        },
        "helper": {"path": HELPER_PATH, "sha256": sha256(root / HELPER_PATH)},
        "authoritative_final_actor_result": final,
        "diagnostic_exact17_audit": audit,
        "subject": {
            "baseline": final["winner"],
            "selection_rule": (
                "mechanically revalidate final_actor_result; exact-17 audit is "
                "provenance only and cannot select a candidate"
            ),
        },
        "build": lock["build"],
        "campaign": {
            "variants": list(VARIANTS),
            "development_pairs_per_orientation": DEVELOPMENT_PAIRS,
            "safety_pairs_per_orientation": SAFETY_PAIRS,
            "final_pairs_per_orientation": FINAL_PAIRS,
            "critical_z": CRITICAL_Z,
            "manual_dispatch": False,
            "retry": False,
            "optional_stopping": False,
            "repository_write": False,
        },
        "results": None,
    }


def prepare_execution(root: Path, output: Path, source_commit: str,
                      source_tree: str) -> dict[str, Any]:
    if output.resolve() != (root / EXECUTION_PATH).resolve():
        raise EvidenceError("execution addendum must use the canonical path")
    value = expected_execution(root, source_commit, source_tree)
    _atomic_create(output, value)
    return value


def guard_execution(root: Path, execution: Path, source_commit: str,
                    source_tree: str) -> dict[str, Any]:
    supplied = strict_json(execution)
    hint = supplied.get("definition_lock", {}).get("lock_commit") \
        if isinstance(supplied.get("definition_lock"), dict) else None
    expected = expected_execution(
        root, source_commit, source_tree, lock_commit_hint=hint)
    if supplied != expected:
        raise EvidenceError("execution addendum differs from locked inputs")
    return expected


def verify_one_shot_add(root: Path, before: str, after: str) -> None:
    if _HEX40.fullmatch(before) is None or _HEX40.fullmatch(after) is None or \
            set(before) == {"0"}:
        raise EvidenceError("one-shot launch requires canonical commits")
    try:
        parents = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", after], cwd=root,
            text=True, stderr=subprocess.STDOUT,
        ).split()
        changes = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r",
             after], cwd=root, text=True, stderr=subprocess.STDOUT,
        ).splitlines()
        existed = subprocess.run(
            ["git", "cat-file", "-e", f"{before}:{EXECUTION_PATH}"], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        history = subprocess.check_output(
            ["git", "rev-list", "--all", "--count", "--", EXECUTION_PATH],
            cwd=root, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError("cannot verify execution topology") from exc
    if parents != [after, before] or changes != [f"A\t{EXECUTION_PATH}"] or \
            existed or history != "1":
        raise EvidenceError("execution must be a unique direct-parent add-only commit")


def _fnv1a(snapshot: bytes) -> int:
    value = 1469598103934665603
    for byte in snapshot:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def inspect_table(path: Path, projected: bool,
                  expected_seed: str = TABLE_SEED) -> dict[str, Any]:
    try:
        snapshot = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read match-value table {path}") from exc
    if len(snapshot) < 136 or snapshot[:8] != b"LCMVAL1\0":
        raise EvidenceError("invalid match-value table header")
    u32 = lambda offset: struct.unpack_from("<I", snapshot, offset)[0]
    u64 = lambda offset: struct.unpack_from("<Q", snapshot, offset)[0]
    f64 = lambda offset: struct.unpack_from("<d", snapshot, offset)[0]
    r1_count, r2_count = u32(20), u32(24)
    expected_size = 128 + 16 * (r1_count + r2_count) + 8
    if u32(8) != 1 or u32(12) != 128 or r1_count != 2361 or \
            r2_count != 4721 or u32(28) != 150 or len(snapshot) != expected_size:
        raise EvidenceError("unsupported match-value table dimensions")
    fingerprint = _fnv1a(snapshot[:-8])
    if u64(len(snapshot) - 8) != fingerprint:
        raise EvidenceError("match-value table fingerprint mismatch")
    words = [u32(40 + 4 * index) for index in range(11)]
    adjustments = [f64(92), f64(100)]
    if not isinstance(expected_seed, str) or not expected_seed.isdecimal():
        raise EvidenceError("expected table seed must be a decimal string")
    if u32(16) != TABLE_SAMPLES or u64(84) != int(expected_seed) or \
            u32(108) != TABLE_ROLE_CYCLE or u32(112) != 1 or \
            u32(116) != int(projected) or u32(120) != TABLE_CONTROLLER_ABI or \
            f"{u32(124):08x}" != BUILD_PROFILE_HEX or \
            words != TABLE_CONTROLLER_WORDS or any(
                not math.isfinite(value) or value < 0.0 for value in adjustments):
        raise EvidenceError("frozen match-value table metadata drift")
    count = 2 * (r1_count + r2_count)
    values = struct.unpack_from(f"<{count}d", snapshot, 128)
    if any(not math.isfinite(value) or abs(value) > 227.0 for value in values):
        raise EvidenceError("invalid match-value table value")
    offset = 0
    for length in (r1_count, r2_count):
        not_start = values[offset:offset + length]
        starts = values[offset + length:offset + 2 * length]
        if any(a != -b for a, b in zip(starts, reversed(not_start))):
            raise EvidenceError("table violates player-swap zero sum")
        if projected and (any(a > b for a, b in zip(not_start, not_start[1:])) or
                          any(a > b for a, b in zip(starts, starts[1:]))):
            raise EvidenceError("projected match-value table is nonmonotone")
        offset += 2 * length
    return {
        "path": str(path), "sha256": hashlib.sha256(snapshot).hexdigest(),
        "size": len(snapshot), "version": 1,
        "samples_per_policy_lead": u32(16),
        "total_round_simulations": TABLE_TOTAL_ROUND_SIMULATIONS,
        "source_seed": str(u64(84)), "role_cycle_size": u32(108),
        "role_balance_complete": bool(u32(112)),
        "isotonic_projected": projected,
        "max_isotonic_adjustment": adjustments,
        "payload_fingerprint": f"{fingerprint:016x}",
        "controller": {
            "net_fingerprint": f"{u64(32):016x}",
            "controller_words": words, "controller_abi": u32(120),
            "build_profile_hex": f"{u32(124):08x}",
        },
    }


def table_manifest(raw_path: Path, projected_path: Path,
                   expected_seed: str = TABLE_SEED) -> dict[str, Any]:
    raw = inspect_table(raw_path, False, expected_seed)
    projected = inspect_table(projected_path, True, expected_seed)
    shared = (
        "samples_per_policy_lead", "total_round_simulations", "source_seed",
        "role_cycle_size", "role_balance_complete", "max_isotonic_adjustment",
        "controller",
    )
    if any(raw[field] != projected[field] for field in shared):
        raise EvidenceError("raw/projected tables do not share one corpus")
    if raw["sha256"] == projected["sha256"] or \
            raw["payload_fingerprint"] == projected["payload_fingerprint"]:
        raise EvidenceError("raw/projected table identities collapsed")
    return {
        "schema": TABLE_SCHEMA,
        "artifact_kind": "controller_bound_match_value_table_pair",
        "status": "complete_valid_single_transition_generation",
        "single_builder_invocation": True,
        "variants_share_identical_transition_histograms": True,
        "raw": raw,
        "projected": projected,
    }


def build_actors(baseline: str, world_cap: int, raw_path: str,
                 projected_path: str) -> dict[str, Any]:
    if baseline != MAINTAINED_ACTOR or world_cap != WORLD_CAP:
        raise EvidenceError("baseline is not the authoritative maintained actor")
    if raw_path != RAW_TABLE_PATH or projected_path != PROJECTED_TABLE_PATH:
        raise EvidenceError("candidate table path drift")
    fields = baseline.split(":")
    if len(fields) != 38 or fields[:3] != ["rolloutu", MODEL_PATH, "800"]:
        raise EvidenceError("maintained actor family drift")
    tail = fields[2:]
    if len(tail) != 36 or tail[19] != "800" or tail[5] != "14" or \
            tail[8] != "0" or tail[12] != "4" or tail[13] != "20" or \
            tail[16] != "20" or tail[20] != "1" or tail[27] != "3" or \
            tail[35] != "1":
        raise EvidenceError("maintained controller fields drift")
    tail.extend(["0", "0", "0", "1", "0"])

    def candidate(table: str) -> str:
        item = list(tail)
        item[5] = "0"
        item[8] = "3"
        item.append(table)
        return "rolloutu2:" + ":".join([MODEL_PATH, MODEL_PATH, *item])

    actors = {
        "legacy": baseline,
        "RAW_ALL_PLY": candidate(raw_path),
        "PROJECTED_ALL_PLY": candidate(projected_path),
    }
    if len(set(actors.values())) != 3:
        raise EvidenceError("objective-3 candidate identities collapsed")
    return {
        "schema": "lc-match-value-objective3-v2-actors-v1",
        "artifact_kind": "locked_match_value_objective3_v2_actors",
        "world_cap": world_cap,
        "model": {"path": MODEL_PATH, "sha256": MODEL_SHA256},
        "actors": actors,
    }


def _panel_exact_statistics(value: dict[str, Any]) -> dict[str, int]:
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 2:
        raise EvidenceError("reciprocal panel lacks exactly two blocks")
    sufficient = []
    for block in blocks:
        item = block.get("sufficient_statistics") if isinstance(block, dict) \
            else None
        if not isinstance(item, dict):
            raise EvidenceError("reciprocal block lacks sufficient statistics")
        sufficient.append(item)
    fields = ("pairs", "score_quarters_sum", "margin_sum", "capped_rounds")
    if any(type(item.get(field)) is not int
           for item in sufficient for field in fields):
        raise EvidenceError("noninteger reciprocal sufficient statistics")
    first, second = sufficient
    if first["pairs"] <= 0 or first["pairs"] != second["pairs"]:
        raise EvidenceError("unbalanced reciprocal panel")
    n = first["pairs"]
    return {
        "pairs_per_orientation": n,
        "combined_score_numerator": first["score_quarters_sum"] +
            (4 * n - second["score_quarters_sum"]),
        "combined_score_denominator": 8 * n,
        "combined_margin_numerator": first["margin_sum"] -
            second["margin_sum"],
        "combined_margin_denominator": 4 * n,
        "candidate_first_score_quarters": first["score_quarters_sum"],
        "baseline_first_score_quarters": second["score_quarters_sum"],
        "capped_rounds": first["capped_rounds"] + second["capped_rounds"],
    }


def _numeric_result(value: dict[str, Any]) -> dict[str, Any]:
    result = value.get("candidate_result")
    numeric = (
        "match_score", "match_score_pair_clustered_se", "margin_per_game",
        "margin_pair_clustered_se",
    )
    if not isinstance(result, dict) or any(
            isinstance(result.get(field), bool) or
            not isinstance(result.get(field), (int, float)) or
            not math.isfinite(float(result[field])) or
            (field.endswith("_se") and float(result[field]) < 0.0)
            for field in numeric):
        raise EvidenceError("panel has invalid numeric result")
    orientations = result.get("orientation_match_scores")
    if not isinstance(orientations, list) or len(orientations) != 2 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or
            not math.isfinite(float(item)) for item in orientations):
        raise EvidenceError("panel has invalid orientation estimates")
    return result


def common_gate(value: dict[str, Any], pairs: int) -> tuple[
        dict[str, int], dict[str, bool]]:
    exact = _panel_exact_statistics(value)
    result = _numeric_result(value)
    if exact["pairs_per_orientation"] != pairs:
        raise EvidenceError(f"panel does not contain {pairs} pairs/orientation")
    requirements = {
        "complete_equal_reciprocal_blocks": True,
        "raw_inputs_validated":
            value.get("raw_input_validation", {}).get("status") == "validated",
        "zero_capped_rounds": exact["capped_rounds"] == 0 and
            result.get("capped_rounds") == 0,
        "combined_match_score_at_least_half":
            exact["combined_score_numerator"] >=
            exact["combined_score_denominator"] // 2,
        "combined_margin_strictly_positive":
            exact["combined_margin_numerator"] > 0,
        "each_orientation_match_score_at_least_0_475":
            all(float(item) >= 0.475
                for item in result["orientation_match_scores"]),
    }
    return exact, requirements


def development_selection(panels: dict[str, dict[str, Any]],
                          actors: dict[str, str],
                          digests: dict[str, str] | None = None) -> dict[str, Any]:
    canonical = build_actors(
        MAINTAINED_ACTOR, WORLD_CAP, RAW_TABLE_PATH, PROJECTED_TABLE_PATH)[
            "actors"]
    expected_actors = {name: canonical[name] for name in VARIANTS}
    if set(panels) != set(VARIANTS) or set(actors) != set(VARIANTS) or \
            actors != expected_actors:
        raise EvidenceError("development requires exactly two distinct variants")
    digest_map = digests or {name: "0" * 64 for name in VARIANTS}
    if set(digest_map) != set(VARIANTS) or any(
            _HEX64.fullmatch(value) is None for value in digest_map.values()):
        raise EvidenceError("development reciprocal digest set is invalid")
    evidence: dict[str, Any] = {}
    eligible: list[str] = []
    for name in VARIANTS:
        exact, requirements = common_gate(panels[name], DEVELOPMENT_PAIRS)
        result = _numeric_result(panels[name])
        passed = all(requirements.values())
        if passed:
            eligible.append(name)
        evidence[name] = {
            "actor": actors[name],
            "reciprocal_sha256": digest_map[name],
            "exact_statistics": exact,
            "requirements": requirements,
            "eligible": passed,
            "match_score": float(result["match_score"]),
            "margin_per_game": float(result["margin_per_game"]),
        }
    priority = {name: len(TIE_PRIORITY) - TIE_PRIORITY.index(name)
                for name in TIE_PRIORITY}
    selected = max(eligible, key=lambda name: (
        evidence[name]["exact_statistics"]["combined_score_numerator"],
        evidence[name]["exact_statistics"]["combined_margin_numerator"],
        priority[name],
    )) if eligible else None
    return {
        "schema": DEVELOPMENT_SCHEMA,
        "artifact_kind": "match_value_objective3_v2_development_selection",
        "status": "complete_two_variant_reciprocal_development",
        "selection_rule": (
            "eligible only after exact development gate; maximize exact "
            "combined score numerator, then exact margin numerator, then "
            "PROJECTED_ALL_PLY"
        ),
        "evidence": evidence,
        "eligible_variants": eligible,
        "selected_variant": selected,
        "selected_actor": None if selected is None else actors[selected],
        "eligible": selected is not None,
        "challenge_exists": selected is not None,
        "promotion_claim": False,
    }


def safety_gate(value: dict[str, Any]) -> dict[str, Any]:
    exact, requirements = common_gate(value, SAFETY_PAIRS)
    result = _numeric_result(value)
    return {
        "schema": "lc-match-value-objective3-v2-safety-gate-v1",
        "artifact_kind": "match_value_objective3_v2_safety_gate",
        "status": "complete_locked_safety_panel",
        "exact_statistics": exact,
        "candidate_result": {
            "match_score": float(result["match_score"]),
            "margin_per_game": float(result["margin_per_game"]),
            "orientation_match_scores": [
                float(item) for item in result["orientation_match_scores"]],
        },
        "requirements": requirements,
        "passed": all(requirements.values()),
        "promotion_claim": False,
    }


def final_gate(value: dict[str, Any], critical_z: float = CRITICAL_Z) -> dict[str, Any]:
    if not math.isfinite(critical_z) or critical_z != CRITICAL_Z:
        raise EvidenceError("final critical value must remain exactly 1.645")
    exact = _panel_exact_statistics(value)
    if exact["pairs_per_orientation"] != FINAL_PAIRS:
        raise EvidenceError("final panel does not contain 2500 pairs/orientation")
    result = _numeric_result(value)
    score = float(result["match_score"])
    score_se = float(result["match_score_pair_clustered_se"])
    margin = float(result["margin_per_game"])
    margin_se = float(result["margin_pair_clustered_se"])
    score_lcb = score - critical_z * score_se
    margin_lcb = margin - critical_z * margin_se
    requirements = {
        "complete_equal_reciprocal_blocks": True,
        "raw_inputs_validated":
            value.get("raw_input_validation", {}).get("status") == "validated",
        "zero_capped_rounds": exact["capped_rounds"] == 0 and
            result.get("capped_rounds") == 0,
        "pair_clustered_orientation_stratified_score_lcb_above_half":
            score_lcb > 0.5,
        "pair_clustered_orientation_stratified_margin_lcb_above_zero":
            margin_lcb > 0.0,
        "each_reciprocal_orientation_strictly_above_half":
            all(float(item) > 0.5 for item in result["orientation_match_scores"]),
    }
    return {
        "schema": "lc-match-value-objective3-v2-final-gate-v1",
        "artifact_kind": "match_value_objective3_v2_reserved_final_gate",
        "status": "complete_reserved_final_test",
        "critical_z": critical_z,
        "confidence_bound": "95% one-sided lower confidence bounds",
        "estimator": (
            "mirrored-pair clusters within orientation; independent reciprocal "
            "orientations combined with equal weight"
        ),
        "exact_statistics": exact,
        "candidate_result": {
            "match_score": score,
            "match_score_pair_clustered_se": score_se,
            "match_score_lower_bound": score_lcb,
            "margin_per_game": margin,
            "margin_pair_clustered_se": margin_se,
            "margin_lower_bound": margin_lcb,
            "orientation_match_scores": [
                float(item) for item in result["orientation_match_scores"]],
        },
        "requirements": requirements,
        "promotion_gate_passed": all(requirements.values()),
        "repository_promotion_performed": False,
    }


def _validate_panel_identity(value: dict[str, Any], candidate: str,
                             baseline: str, provenance: str, pairs: int,
                             first_seed: str, second_seed: str) -> None:
    canonical = build_actors(
        MAINTAINED_ACTOR, WORLD_CAP, RAW_TABLE_PATH, PROJECTED_TABLE_PATH)[
            "actors"]
    if baseline != MAINTAINED_ACTOR or candidate not in {
            canonical[name] for name in VARIANTS}:
        raise EvidenceError("reciprocal panel actor is outside the locked family")
    if value.get("artifact_kind") != "locked_reciprocal_arena_result" or \
            value.get("candidate") != candidate or \
            value.get("baseline") != baseline or \
            value.get("provenance") != provenance or \
            value.get("raw_input_validation", {}).get("status") != "validated":
        raise EvidenceError("locked reciprocal panel identity drift")
    blocks = value.get("blocks")
    expected = ((candidate, baseline, first_seed),
                (baseline, candidate, second_seed))
    if not isinstance(blocks, list) or len(blocks) != 2:
        raise EvidenceError("locked reciprocal panel lacks orientations")
    for block, identity in zip(blocks, expected):
        metadata = block.get("metadata") if isinstance(block, dict) else None
        if not isinstance(metadata, dict) or block.get("pair_start") != "0" or \
                block.get("pair_count") != pairs or \
                metadata.get("agent_a") != identity[0] or \
                metadata.get("agent_b") != identity[1] or \
                metadata.get("seed") != identity[2] or \
                metadata.get("rounds") != 3 or \
                metadata.get("provenance") != provenance:
            raise EvidenceError("locked reciprocal block identity drift")
    exact = _panel_exact_statistics(value)
    if exact["pairs_per_orientation"] != pairs or exact["capped_rounds"] != 0:
        raise EvidenceError("locked reciprocal panel size/cap drift")


def load_verified_panel(path: Path, candidate: str, baseline: str,
                        provenance: str, pairs: int, first_seed: str,
                        second_seed: str) -> tuple[dict[str, Any], str]:
    value, digest = _rebuild_reciprocal(path, CRITICAL_Z)
    _validate_panel_identity(value, candidate, baseline, provenance, pairs,
                             first_seed, second_seed)
    return value, digest


def post_build_manifest(root: Path, execution_path: Path, table_path: Path,
                        actors_path: Path, builder_log: Path, build_info: Path,
                        source_commit: str, source_tree: str) -> dict[str, Any]:
    execution = strict_json(execution_path)
    tables = strict_json(table_path)
    actors = strict_json(actors_path)
    baseline = execution.get("subject", {}).get("baseline", {}).get("spec")
    if tables.get("schema") != TABLE_SCHEMA or \
            tables.get("status") != "complete_valid_single_transition_generation" or \
            actors.get("artifact_kind") != \
            "locked_match_value_objective3_v2_actors" or \
            actors.get("actors", {}).get("legacy") != baseline:
        raise EvidenceError("post-build inputs are incomplete or actor-bound wrong")
    return {
        "schema": MANIFEST_SCHEMA,
        "artifact_kind": "match_value_objective3_v2_pre_efficacy_manifest",
        "status": "tables_and_actors_frozen_before_first_efficacy_match",
        "source": {"commit": source_commit, "tree": source_tree},
        "bindings": {
            PLAN_PATH: sha256(root / PLAN_PATH),
            WORKFLOW_PATH: sha256(root / WORKFLOW_PATH),
            str(execution_path): sha256(execution_path),
            FINAL_RESULT_PATH: sha256(root / FINAL_RESULT_PATH),
            AUDIT_RESULT_PATH: sha256(root / AUDIT_RESULT_PATH),
            str(table_path): sha256(table_path),
            str(actors_path): sha256(actors_path),
            str(builder_log): sha256(builder_log),
            str(build_info): sha256(build_info),
        },
        "build": execution["build"],
        "tables": tables,
        "actors": actors,
        "diagnostic_audit_selection_use": "forbidden",
        "results": None,
    }


def terminal_result(execution: dict[str, Any], development: dict[str, Any],
                    safety: dict[str, Any] | None,
                    final: dict[str, Any] | None,
                    evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Create the canonical terminal disposition without inventing evidence."""
    baseline = execution.get("subject", {}).get("baseline", {}).get("spec")
    if baseline != MAINTAINED_ACTOR or development.get("schema") != \
            DEVELOPMENT_SCHEMA:
        raise EvidenceError("terminal inputs are not execution/development bound")
    challenger = development.get("selected_actor")
    canonical_candidates = build_actors(
        MAINTAINED_ACTOR, WORLD_CAP, RAW_TABLE_PATH, PROJECTED_TABLE_PATH)[
            "actors"]
    allowed_challengers = {canonical_candidates[name] for name in VARIANTS}
    if challenger is not None and challenger not in allowed_challengers:
        raise EvidenceError("terminal challenger is outside the locked actor family")
    if development.get("eligible") is not (challenger is not None) or \
            development.get("challenge_exists") is not (challenger is not None):
        raise EvidenceError("development eligibility/selection is inconsistent")
    if challenger is None:
        if safety is not None or final is not None:
            raise EvidenceError("no-challenge result cannot contain later efficacy")
        passed = False
        stage = "development_no_eligible_candidate"
    elif safety is None or safety.get("passed") is not True:
        if final is not None:
            raise EvidenceError("final evidence exists after failed/absent safety")
        passed = False
        stage = "safety_failed"
    elif final is None:
        raise EvidenceError("passed safety requires a complete final decision")
    else:
        passed = final.get("promotion_gate_passed") is True
        stage = "final_passed" if passed else "final_failed"
    if safety is not None and (
            safety.get("schema") !=
            "lc-match-value-objective3-v2-safety-gate-v1" or
            safety.get("candidate") != challenger or
            safety.get("baseline") != baseline):
        raise EvidenceError("terminal safety decision identity drift")
    if final is not None and (
            final.get("schema") !=
            "lc-match-value-objective3-v2-final-gate-v1" or
            final.get("candidate") != challenger or
            final.get("baseline") != baseline):
        raise EvidenceError("terminal final decision identity drift")
    winner = challenger if passed else baseline
    if not isinstance(evidence, list) or any(
            not isinstance(row, dict) or set(row) != {"path", "sha256"} or
            _HEX64.fullmatch(str(row.get("sha256"))) is None
            for row in evidence):
        raise EvidenceError("terminal evidence manifest is malformed")
    names = [str(row["path"]) for row in evidence]
    if len(names) != len(set(names)):
        raise EvidenceError("terminal evidence manifest has duplicate paths")
    required = {
        "pre-efficacy-manifest.json",
        "transport/BUILD_INFO.txt",
        "transport/SHA256SUMS.txt",
        "transport/bindings/actors.json",
        "transport/bindings/definition-lock.json",
        "transport/bindings/execution.json",
        "transport/bindings/plan.json",
        "transport/bindings/pre-efficacy-manifest.json",
        "transport/bindings/table-manifest.json",
        f"transport/{RAW_TABLE_PATH}",
        f"transport/{PROJECTED_TABLE_PATH}",
        "development/merged/development-selection.json",
        "development/merged/RAW_ALL_PLY-reciprocal.json",
        "development/merged/PROJECTED_ALL_PLY-reciprocal.json",
    }
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
            "terminal evidence omits required files: " + ", ".join(missing))
    expected_counts = {"development/downloads/": 40}
    if challenger is not None:
        expected_counts["safety/downloads/"] = 20
    if final is not None:
        expected_counts["final/downloads/"] = 50
    for prefix, count in expected_counts.items():
        for suffix in (".jsonl", ".sha256", ".time"):
            actual = sum(name.startswith(prefix) and name.endswith(suffix)
                         for name in names)
            if actual != count:
                raise EvidenceError(
                    f"terminal evidence {prefix} {suffix} count {actual} != {count}")
    return {
        "schema": RESULT_SCHEMA,
        "artifact_kind": "match_value_objective3_v2_authoritative_result",
        "status": "complete",
        "disposition": stage,
        "baseline_actor": baseline,
        "challenger_actor": challenger,
        "winner_actor": winner,
        "promotion_gate_passed": passed,
        "locked_validation_relaxed": False,
        "diagnostic_audit_used_for_selection": False,
        "development": development,
        "safety": safety,
        "final": final,
        "evidence": evidence,
    }


def evidence_manifest(root: Path, excluded: Path | None = None) -> list[dict[str, Any]]:
    """Hash a complete terminal evidence tree in canonical path order."""
    if not root.is_dir() or root.is_symlink():
        raise EvidenceError("terminal evidence root must be a real directory")
    excluded_resolved = excluded.resolve() if excluded is not None else None
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise EvidenceError(f"terminal evidence contains symlink: {path}")
        if not path.is_file() or (excluded_resolved is not None and
                                  path.resolve() == excluded_resolved):
            continue
        relative = path.relative_to(root).as_posix()
        if not relative or any(part in {"", ".", ".."}
                               for part in Path(relative).parts):
            raise EvidenceError("terminal evidence path is noncanonical")
        rows.append({"path": relative, "sha256": sha256(path)})
    if not rows:
        raise EvidenceError("terminal evidence tree is empty")
    return rows


def _assignment(values: list[str], allowed: tuple[str, ...],
                label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or name not in allowed or not item or name in result:
            raise EvidenceError(f"invalid {label} assignment {value!r}")
        result[name] = item
    if set(result) != set(allowed):
        raise EvidenceError(f"{label} assignments are incomplete")
    return result


def _github_output(path: Path | None, values: dict[str, str]) -> None:
    if path is None:
        return
    if any("\n" in value or "\r" in value for value in values.values()):
        raise EvidenceError("multiline GitHub output is forbidden")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)

    validate = command.add_parser("validate-plan")
    validate.add_argument("--root", type=Path, default=Path("."))

    lock = command.add_parser("prepare-definition-lock")
    lock.add_argument("--root", type=Path, required=True)
    lock.add_argument("--definition-commit", required=True)
    lock.add_argument("--definition-tree", required=True)
    lock.add_argument("--lock", type=Path, required=True)

    for name in ("prepare-execution", "guard-execution"):
        item = command.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--source-parent-commit", required=True)
        item.add_argument("--source-parent-tree", required=True)
        item.add_argument("--execution", type=Path, required=True)
        if name == "guard-execution":
            item.add_argument("--github-output", type=Path)

    tables = command.add_parser("table-manifest")
    tables.add_argument("--raw", type=Path, required=True)
    tables.add_argument("--projected", type=Path, required=True)
    tables.add_argument("--output", type=Path, required=True)

    actors = command.add_parser("actors")
    actors.add_argument("--baseline", required=True)
    actors.add_argument("--world-cap", type=int, required=True)
    actors.add_argument("--raw-path", required=True)
    actors.add_argument("--projected-path", required=True)
    actors.add_argument("--output", type=Path, required=True)

    post = command.add_parser("post-build-manifest")
    post.add_argument("--root", type=Path, required=True)
    post.add_argument("--execution", type=Path, required=True)
    post.add_argument("--tables", type=Path, required=True)
    post.add_argument("--actors", type=Path, required=True)
    post.add_argument("--builder-log", type=Path, required=True)
    post.add_argument("--build-info", type=Path, required=True)
    post.add_argument("--source-commit", required=True)
    post.add_argument("--source-tree", required=True)
    post.add_argument("--output", type=Path, required=True)

    select = command.add_parser("select-development")
    select.add_argument("--panel", action="append", default=[])
    select.add_argument("--actor", action="append", default=[])
    select.add_argument("--provenance", action="append", default=[])
    select.add_argument("--baseline", required=True)
    select.add_argument("--candidate-first-seed", required=True)
    select.add_argument("--baseline-first-seed", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--github-output", type=Path)

    for name in ("gate-safety", "gate-final"):
        gate = command.add_parser(name)
        gate.add_argument("--reciprocal", type=Path, required=True)
        gate.add_argument("--candidate", required=True)
        gate.add_argument("--baseline", required=True)
        gate.add_argument("--provenance", required=True)
        gate.add_argument("--candidate-first-seed", required=True)
        gate.add_argument("--baseline-first-seed", required=True)
        gate.add_argument("--output", type=Path, required=True)
        gate.add_argument("--github-output", type=Path)

    terminal = command.add_parser("terminal-result")
    terminal.add_argument("--execution", type=Path, required=True)
    terminal.add_argument("--development", type=Path, required=True)
    terminal.add_argument("--safety", type=Path)
    terminal.add_argument("--final", type=Path)
    terminal.add_argument("--evidence-root", type=Path, required=True)
    terminal.add_argument("--output", type=Path, required=True)
    terminal.add_argument("--github-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    try:
        if args.command == "validate-plan":
            validate_plan(strict_json(args.root / PLAN_PATH))
        elif args.command == "prepare-definition-lock":
            prepare_definition_lock(args.root, args.lock,
                                    args.definition_commit,
                                    args.definition_tree)
        elif args.command == "prepare-execution":
            prepare_execution(args.root, args.execution,
                              args.source_parent_commit,
                              args.source_parent_tree)
        elif args.command == "guard-execution":
            value = guard_execution(args.root, args.execution,
                                    args.source_parent_commit,
                                    args.source_parent_tree)
            _github_output(args.github_output, {
                "winner_actor": value["subject"]["baseline"]["spec"],
                "world_cap": str(WORLD_CAP),
                "execution_sha": sha256(args.execution),
            })
        elif args.command == "table-manifest":
            _write_json(args.output, table_manifest(args.raw, args.projected))
        elif args.command == "actors":
            _write_json(args.output, build_actors(
                args.baseline, args.world_cap, args.raw_path,
                args.projected_path))
        elif args.command == "post-build-manifest":
            _write_json(args.output, post_build_manifest(
                args.root, args.execution, args.tables, args.actors,
                args.builder_log, args.build_info, args.source_commit,
                args.source_tree))
        elif args.command == "select-development":
            paths = _assignment(args.panel, VARIANTS, "panel")
            actors = _assignment(args.actor, VARIANTS, "actor")
            provenances = _assignment(args.provenance, VARIANTS, "provenance")
            panels: dict[str, dict[str, Any]] = {}
            digests: dict[str, str] = {}
            for name in VARIANTS:
                panels[name], digests[name] = load_verified_panel(
                    Path(paths[name]), actors[name], args.baseline,
                    provenances[name], DEVELOPMENT_PAIRS,
                    args.candidate_first_seed, args.baseline_first_seed)
            value = development_selection(panels, actors, digests)
            _write_json(args.output, value)
            _github_output(args.github_output, {
                "selected_variant": value["selected_variant"] or "NONE",
                "selected_actor": value["selected_actor"] or "",
                "eligible": "true" if value["challenge_exists"] else "false",
                "decision_sha": sha256(args.output),
            })
        elif args.command in {"gate-safety", "gate-final"}:
            pairs = SAFETY_PAIRS if args.command == "gate-safety" else FINAL_PAIRS
            panel, digest = load_verified_panel(
                args.reciprocal, args.candidate, args.baseline,
                args.provenance, pairs, args.candidate_first_seed,
                args.baseline_first_seed)
            value = safety_gate(panel) if args.command == "gate-safety" \
                else final_gate(panel)
            value.update({
                "candidate": args.candidate, "baseline": args.baseline,
                "provenance": args.provenance,
                "reciprocal_path": str(args.reciprocal),
                "reciprocal_sha256": digest,
                "seeds": {
                    "candidate_first": args.candidate_first_seed,
                    "baseline_first": args.baseline_first_seed,
                },
            })
            _write_json(args.output, value)
            passed = value.get("passed", value.get("promotion_gate_passed"))
            _github_output(args.github_output, {
                "passed": "true" if passed else "false",
                "decision_sha": sha256(args.output),
            })
        else:
            execution = strict_json(args.execution)
            development = strict_json(args.development)
            safety = strict_json(args.safety) if args.safety is not None else None
            final = strict_json(args.final) if args.final is not None else None
            evidence = evidence_manifest(args.evidence_root, args.output)
            value = terminal_result(
                execution, development, safety, final, evidence)
            _write_json(args.output, value)
            _github_output(args.github_output, {
                "promotion_gate_passed": "true" if
                    value["promotion_gate_passed"] else "false",
                "winner_actor": value["winner_actor"],
                "result_sha": sha256(args.output),
            })
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"match-value objective3 v2: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
