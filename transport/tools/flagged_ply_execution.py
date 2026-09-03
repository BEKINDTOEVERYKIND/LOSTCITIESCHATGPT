#!/usr/bin/env python3
"""Prepare and verify the one-shot, post-selection flagged-ply audit.

The audit actors are never typed into a workflow input or copied into an
execution addendum. A completed repository-owned final actor result binds a
distinct challenger/reference comparison and a raw-validated final gate. Its
boolean gate mechanically selects the actor audited as the final winner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lc-flagged-ply-audit-execution-v2"
FINAL_SCHEMA = "lc-authoritative-final-actor-result-v2"
EXECUTION_PATH = "data/flagged_ply_audit_execution.json"
FINAL_RESULT_PATH = "data/experiments/final_actor_result.json"
PLAN_PATH = "data/flagged_ply_audit_plan.json"
WORKFLOW_PATH = ".github/workflows/flagged-ply-audit.yml"
MANIFEST_PATH = "data/user_reviewed_plies.json"
BRANCH = "agent/correctness-and-policy-upgrade"
DECISION_WORLDS = 16384
HISTORY_WORLDS = 20000
BELIEF_ALPHA = 1.15
BASE_SEED = 202608231701
SHARD_COUNT = 12
COMPILER = "gcc"
COMPILER_SEMANTIC_VERSION_COMMAND = "gcc -dumpfullversion -dumpversion"
REQUIRED_COMPILER_SEMANTIC_VERSION = "13.3.0"
CFLAGS = (
    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
    "-Wall -Wextra -std=c11"
)
LDFLAGS = "-lm -pthread"
TOOL_PATHS = (
    "tools/flagged_ply_execution.py",
    "tools/flagged_ply_audit.py",
    "tools/flagged_ply_probe.c",
    "tools/history_belief.py",
    "tools/history_belief.c",
    "tools/merge_flagged_ply_audit.py",
    "tools/render_flagged_ply_audit.py",
    "tools/gate_actor_panel.py",
    "tools/match_value_campaign.py",
    "tools/merge_arena.py",
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ROLLOUT_KINDS = {
    "rollout": 1, "rolloutu": 1,
    "rollout2": 2, "rolloutu2": 2,
    "rollout3": 3, "rolloutu3": 3,
    "rollout4": 3, "rolloutu4": 3,
}


class ExecutionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {token}")
            ),
        )
        _finite(value)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"cannot load strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutionError(f"{path}: top-level JSON must be an object")
    return value


def _repo_file(root: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExecutionError(f"{label} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or relative.as_posix() != value or \
            any(part in {"", ".", ".."} for part in relative.parts):
        raise ExecutionError(f"{label} is not a canonical repository path")
    path = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ExecutionError(f"{label} crosses a symbolic link")
    if not path.is_file():
        raise ExecutionError(f"{label} is absent: {value}")
    return path, value


def _actor_provenance(root: Path, spec: Any, label: str) -> dict[str, Any]:
    if not isinstance(spec, str) or any(
            ord(char) < 0x20 or char == "%" for char in spec):
        raise ExecutionError(f"{label} is not a safe actor specification")
    fields = spec.split(":")
    checkpoint_count = _ROLLOUT_KINDS.get(fields[0], 0) if fields else 0
    if not checkpoint_count or len(fields) <= checkpoint_count:
        raise ExecutionError(f"{label} must be an explicit rollout actor")
    checkpoints = []
    for role, value in enumerate(fields[1:checkpoint_count + 1]):
        path, relative = _repo_file(root, value, f"{label} checkpoint {role}")
        checkpoints.append({
            "role": role,
            "path": relative,
            "sha256": sha256(path),
            "size": path.stat().st_size,
        })
    result: dict[str, Any] = {"spec": spec, "checkpoints": checkpoints}
    # Tail field 41 is the optional controller-bound match-value table.
    table_index = 1 + checkpoint_count + 41
    if len(fields) > table_index:
        table_path, table_relative = _repo_file(
            root, fields[table_index], f"{label} match-value table")
        result["match_value_table"] = {
            "path": table_relative,
            "sha256": sha256(table_path),
            "size": table_path.stat().st_size,
        }
    return result


def _actor_assets(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    assets = [
        {
            "kind": "checkpoint",
            "role": item["role"],
            "path": item["path"],
            "sha256": item["sha256"],
            "size": item["size"],
        }
        for item in provenance["checkpoints"]
    ]
    if "match_value_table" in provenance:
        assets.append({
            "kind": "match_value_table",
            "role": "controller",
            **provenance["match_value_table"],
        })
    return assets


def _manifest_binding(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path, _ = _repo_file(root, MANIFEST_PATH, "frozen corpus manifest")
    manifest = strict_json(path)
    if manifest.get("schema") != "lc-user-reviewed-ply-corpus-v1":
        raise ExecutionError("unsupported frozen corpus schema")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 36:
        raise ExecutionError("frozen corpus must contain exactly 36 cases")
    kinds = Counter(case.get("kind") for case in cases if isinstance(case, dict))
    if kinds != Counter({"decision": 34, "belief": 2}):
        raise ExecutionError("frozen corpus must contain 34 decisions and 2 beliefs")
    seen: set[str] = set()
    artifacts: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ExecutionError("frozen corpus has duplicate/invalid case ids")
        seen.add(case_id)
        state, state_name = _repo_file(root, case.get("state"), f"{case_id} state")
        if sha256(state) != case.get("state_sha256"):
            raise ExecutionError(f"{case_id}: state hash mismatch")
        artifacts[state_name] = {
            "path": state_name, "sha256": sha256(state), "size": state.stat().st_size,
        }
        if case.get("kind") == "belief":
            view, view_name = _repo_file(root, case.get("view"), f"{case_id} view")
            if sha256(view) != case.get("view_sha256"):
                raise ExecutionError(f"{case_id}: view hash mismatch")
            artifacts[view_name] = {
                "path": view_name, "sha256": sha256(view), "size": view.stat().st_size,
            }
            inference = case.get("history_inference")
            if not isinstance(inference, dict):
                raise ExecutionError(f"{case_id}: history inference is absent")
            checkpoint, name = _repo_file(
                root, inference.get("checkpoint"), f"{case_id} history checkpoint")
            artifacts[name] = {
                "path": name, "sha256": sha256(checkpoint),
                "size": checkpoint.stat().st_size,
            }
    return ({
        "path": MANIFEST_PATH,
        "sha256": sha256(path),
        "cases": 36,
        "decision_cases": 34,
        "belief_cases": 2,
    }, [artifacts[key] for key in sorted(artifacts)])


def _git_source_binding(root: Path, commit: str, tree: str) -> None:
    # Repository launches must bind a real reachable commit/tree. Source
    # archives deliberately have no .git directory and are revalidated by all
    # the content hashes carried in the execution addendum.
    if not (root / ".git").exists():
        return
    try:
        actual_tree = subprocess.check_output(
            ["git", "rev-parse", f"{commit}^{{tree}}"],
            cwd=root, text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionError("authoritative source commit is not reachable") from exc
    if actual_tree != tree:
        raise ExecutionError("authoritative source commit/tree mismatch")


def _gate_boolean(decision: dict[str, Any], field: str,
                  expected_requirements: set[str]) -> bool:
    passed = decision.get(field)
    requirements = decision.get("requirements")
    if type(passed) is not bool or not isinstance(requirements, dict) or \
            set(requirements) != expected_requirements or \
            any(type(value) is not bool for value in requirements.values()) or \
            passed != all(requirements.values()):
        raise ExecutionError("decisive gate boolean is not mechanically reproducible")
    if requirements.get("raw_inputs_validated") is not True or \
            requirements.get("zero_capped_rounds") is not True:
        raise ExecutionError("decisive gate lacks valid raw input or zero-cap evidence")
    return passed


def _revalidate_standard_gate(root: Path, decision: dict[str, Any],
                              reciprocal_path: Path,
                              reciprocal_sha: str) -> bool:
    try:
        from tools.gate_actor_panel import _rebuild_reciprocal, evaluate_gate
        reciprocal, rebuilt_sha = _rebuild_reciprocal(reciprocal_path, 1.645)
        expected = evaluate_gate(reciprocal, "final", 1.645)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise ExecutionError(f"cannot rebuild standard final gate: {exc}") from exc
    added = {
        "reciprocal_path", "reciprocal_sha256", "candidate", "baseline",
        "provenance", "pairs_per_orientation", "seeds",
    }
    if rebuilt_sha != reciprocal_sha or set(decision) != set(expected) | added or \
            any(decision.get(key) != value for key, value in expected.items()) or \
            decision.get("pairs_per_orientation") != 2500:
        raise ExecutionError("standard final gate differs from exact recomputation")
    candidate = decision["candidate"]
    baseline = decision["baseline"]
    provenance = decision["provenance"]
    seeds = decision.get("seeds")
    blocks = reciprocal.get("blocks")
    if reciprocal.get("candidate") != candidate or \
            reciprocal.get("baseline") != baseline or \
            reciprocal.get("provenance") != provenance or \
            not isinstance(seeds, dict) or set(seeds) != {
                "candidate_first", "baseline_first"} or \
            not isinstance(blocks, list) or len(blocks) != 2:
        raise ExecutionError("standard final reciprocal identity drift")
    expected_blocks = (
        (candidate, baseline, seeds["candidate_first"]),
        (baseline, candidate, seeds["baseline_first"]),
    )
    for block, identity in zip(blocks, expected_blocks):
        metadata = block.get("metadata") if isinstance(block, dict) else None
        if not isinstance(metadata, dict) or block.get("pair_start") != "0" or \
                block.get("pair_count") != 2500 or \
                metadata.get("agent_a") != identity[0] or \
                metadata.get("agent_b") != identity[1] or \
                metadata.get("seed") != identity[2] or \
                metadata.get("rounds") != 3 or \
                metadata.get("provenance") != provenance:
            raise ExecutionError("standard final reciprocal schedule drift")
    return bool(expected["passed"])


def _revalidate_match_value_gate(root: Path, decision: dict[str, Any],
                                 reciprocal_path: Path,
                                 reciprocal_sha: str) -> bool:
    seeds = decision.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != {
            "candidate_first", "baseline_first"}:
        raise ExecutionError("match-value final seeds are absent")
    try:
        from tools.match_value_campaign import final_gate, load_verified_panel
        reciprocal, rebuilt_sha = load_verified_panel(
            reciprocal_path, decision.get("candidate"),
            decision.get("baseline"), decision.get("provenance"), 2500,
            seeds["candidate_first"], seeds["baseline_first"],
        )
        expected = final_gate(reciprocal)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise ExecutionError(f"cannot rebuild match-value final gate: {exc}") from exc
    added = {
        "candidate", "baseline", "provenance", "reciprocal_path",
        "reciprocal_sha256", "seeds",
    }
    if rebuilt_sha != reciprocal_sha or set(decision) != set(expected) | added or \
            any(decision.get(key) != value for key, value in expected.items()):
        raise ExecutionError("match-value final gate differs from exact recomputation")
    return bool(expected["promotion_gate_passed"])


def authoritative_final_result(root: Path) -> dict[str, Any]:
    final_path, _ = _repo_file(root, FINAL_RESULT_PATH, "authoritative final result")
    result = strict_json(final_path)
    required = {
        "schema", "status", "source_commit", "source_tree",
        "selection_mode", "reference_actor", "challenger_actor",
        "winner_actor", "actor_assets", "decisive_result",
        "authoritative_results",
    }
    if set(result) != required or result.get("schema") != FINAL_SCHEMA or \
            result.get("status") != "complete":
        raise ExecutionError("authoritative final result has the wrong contract")
    if _HEX40.fullmatch(str(result.get("source_commit"))) is None or \
            _HEX40.fullmatch(str(result.get("source_tree"))) is None:
        raise ExecutionError("authoritative result source commit/tree are invalid")
    source_commit = str(result["source_commit"])
    source_tree = str(result["source_tree"])
    _git_source_binding(root, source_commit, source_tree)
    mode = result.get("selection_mode")
    if mode not in {"no_challenge", "component_final", "composition_final"}:
        raise ExecutionError("unknown authoritative final selection mode")
    reference = _actor_provenance(root, result.get("reference_actor"), "reference actor")
    challenger_value = result.get("challenger_actor")
    challenger = None if challenger_value is None else _actor_provenance(
        root, challenger_value, "challenger actor")
    bindings = result.get("authoritative_results")
    if not isinstance(bindings, list) or not bindings:
        raise ExecutionError("authoritative_results must be a nonempty list")
    bound: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(bindings):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "role"}:
            raise ExecutionError(f"authoritative result binding {index} is malformed")
        path, name = _repo_file(root, item["path"], f"authoritative result {index}")
        if name in bound or _HEX64.fullmatch(str(item["sha256"])) is None or \
                sha256(path) != item["sha256"]:
            raise ExecutionError(f"authoritative result binding {index} hash/path mismatch")
        if not isinstance(item["role"], str) or not item["role"] or any(
                existing["role"] == item["role"] for existing in bound.values()):
            raise ExecutionError(f"authoritative result binding {index} has no role")
        bound[name] = dict(item)
    roles = {item["role"]: (name, item) for name, item in bound.items()}
    decision_binding = result.get("decisive_result")
    passed = False
    if mode == "no_challenge":
        if challenger is not None or decision_binding is not None or \
                "world_winner" not in roles:
            raise ExecutionError("no_challenge must bind only an archived world winner")
        world_name, world_binding = roles["world_winner"]
        try:
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from tools.match_value_campaign import (  # type: ignore
                WORLD800_SOURCE_COMMIT, WORLD800_SOURCE_TREE, _world_result,
            )
            _, _, world_winner, _ = _world_result(root / world_name)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise ExecutionError(f"cannot revalidate archived world winner: {exc}") from exc
        if world_binding["sha256"] != sha256(root / world_name) or \
                source_commit != WORLD800_SOURCE_COMMIT or \
                source_tree != WORLD800_SOURCE_TREE or \
                reference["spec"] != world_winner:
            raise ExecutionError("no_challenge does not carry forward the world winner")
        winner = reference
    else:
        if challenger is None or challenger["spec"] == reference["spec"]:
            raise ExecutionError("a final comparison requires distinct actors")
        if not isinstance(decision_binding, dict) or \
                set(decision_binding) != {"path", "sha256"}:
            raise ExecutionError("decisive_result binding is malformed")
        decision_name = decision_binding.get("path")
        if decision_name not in bound or \
                bound[decision_name]["role"] != "final_decision" or \
                bound[decision_name]["sha256"] != decision_binding.get("sha256"):
            raise ExecutionError("decisive result is not the bound final decision")
        decision_path, _ = _repo_file(root, decision_name, "final gate decision")
        decision = strict_json(decision_path)
        if decision.get("candidate") != challenger["spec"] or \
                decision.get("baseline") != reference["spec"]:
            raise ExecutionError("final decision does not bind the declared comparison")
        reciprocal_sha = decision.get("reciprocal_sha256")
        if _HEX64.fullmatch(str(reciprocal_sha)) is None or \
                "final_reciprocal" not in roles or \
                roles["final_reciprocal"][1]["sha256"] != reciprocal_sha:
            raise ExecutionError("final decision lacks its bound reciprocal panel")
        reciprocal_name, _ = roles["final_reciprocal"]
        reciprocal_path, _ = _repo_file(
            root, reciprocal_name, "final reciprocal panel")
        decision_kind = decision.get("artifact_kind")
        if decision_kind == "locked_reciprocal_actor_gate_decision":
            required = {
                "raw_inputs_validated", "zero_capped_rounds",
                "match_score_one_sided_lower_bound_above_half",
                "margin_one_sided_lower_bound_strictly_positive",
                "each_orientation_match_score_strictly_above_half",
            }
            if decision.get("artifact_kind") != "locked_reciprocal_actor_gate_decision" or \
                    decision.get("mode") != "final":
                raise ExecutionError("standard final decision schema mismatch")
            passed = _gate_boolean(decision, "passed", required)
            if _revalidate_standard_gate(
                    root, decision, reciprocal_path, str(reciprocal_sha)) != passed:
                raise ExecutionError("standard final gate selection drift")
            provenance = decision.get("provenance")
            if not isinstance(provenance, str) or \
                    f"source={source_commit}" not in provenance.split(";") or \
                    f"tree={source_tree}" not in provenance.split(";"):
                raise ExecutionError("standard decision source provenance mismatch")
        elif decision_kind == "match_value_reserved_final_gate":
            required = {
                "complete_equal_reciprocal_blocks", "raw_inputs_validated",
                "zero_capped_rounds",
                "pair_clustered_orientation_stratified_score_lcb_above_half",
                "combined_match_score_point_estimate_above_half",
                "each_reciprocal_orientation_strictly_above_half",
                "combined_margin_strictly_positive",
            }
            if decision.get("artifact_kind") != "match_value_reserved_final_gate" or \
                    decision.get("status") != "complete_reserved_final_test":
                raise ExecutionError("match-value final decision schema mismatch")
            passed = _gate_boolean(decision, "promotion_gate_passed", required)
            if _revalidate_match_value_gate(
                    root, decision, reciprocal_path, str(reciprocal_sha)) != passed:
                raise ExecutionError("match-value final gate selection drift")
            if mode != "component_final" or "match_value_source_binding" not in roles:
                raise ExecutionError("match-value component lacks its source/actor binding")
            source_name, _ = roles["match_value_source_binding"]
            source_binding = strict_json(root / source_name)
            actors = source_binding.get("actors", {}).get("actors", {}) \
                if isinstance(source_binding.get("actors"), dict) else {}
            if source_binding.get("artifact_kind") != \
                    "match_value_pre_efficacy_build_manifest" or \
                    source_binding.get("source") != {
                        "commit": source_commit, "tree": source_tree} or \
                    not isinstance(actors, dict) or \
                    actors.get("legacy") != reference["spec"] or \
                    challenger["spec"] not in actors.values():
                raise ExecutionError("match-value source/actor binding mismatch")
        else:
            raise ExecutionError("unrecognized decisive final-gate schema")
        if mode == "composition_final":
            if decision_kind != "locked_reciprocal_actor_gate_decision" or \
                    "composition_source_binding" not in roles:
                raise ExecutionError(
                    "composition final requires the standard gate and its frozen binding")
            composition_name, _ = roles["composition_source_binding"]
            try:
                from tools.composition_campaign import (  # type: ignore
                    validate_frozen_composition_manifest,
                )
                composition = validate_frozen_composition_manifest(
                    root, root / composition_name)
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                raise ExecutionError(
                    f"cannot revalidate frozen composition manifest: {exc}") from exc
            if composition.get("artifact_kind") != \
                    "locked_composition_pre_efficacy_manifest" or \
                    composition.get("status") != \
                    "frozen_before_composition_efficacy" or \
                    composition.get("source") != {
                        "commit": source_commit, "tree": source_tree} or \
                    composition.get("actors", {}).get("reference", {}).get("spec") != \
                    reference["spec"] or not any(
                        isinstance(item, dict) and item.get("spec") == challenger["spec"]
                        for item in composition.get("actors", {}).get("challengers", [])
                    ):
                raise ExecutionError("composition source/actor binding mismatch")
        winner = challenger if passed else reference
    if result.get("winner_actor") != winner["spec"]:
        raise ExecutionError("explicit winner_actor differs from mechanical selection")
    expected_assets = {
        "reference": _actor_assets(reference),
        "challenger": [] if challenger is None else _actor_assets(challenger),
        "winner": _actor_assets(winner),
    }
    if result.get("actor_assets") != expected_assets:
        raise ExecutionError("actor asset hash manifest differs from actor specifications")
    return {
        "path": FINAL_RESULT_PATH,
        "sha256": sha256(final_path),
        "selection_mode": mode,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "decisive_result": decision_binding,
        "authoritative_results": bindings,
        "promotion_gate_passed": passed,
        "reference": reference,
        "challenger": challenger,
        "winner": winner,
        "actor_assets": expected_assets,
        "no_change": winner["spec"] == reference["spec"],
    }


def expected_execution(root: Path, source_commit: str,
                       source_tree: str) -> dict[str, Any]:
    if _HEX40.fullmatch(source_commit) is None or _HEX40.fullmatch(source_tree) is None:
        raise ExecutionError("source parent commit/tree must be canonical SHA-1 values")
    manifest, corpus_artifacts = _manifest_binding(root)
    final = authoritative_final_result(root)
    plan, _ = _repo_file(root, PLAN_PATH, "locked audit plan")
    workflow, _ = _repo_file(root, WORKFLOW_PATH, "locked audit workflow")
    tool_paths = list(TOOL_PATHS)
    if final["selection_mode"] == "composition_final":
        tool_paths.append("tools/composition_campaign.py")
    tools = []
    for name in tool_paths:
        path, _ = _repo_file(root, name, f"audit tool {name}")
        tools.append({"path": name, "sha256": sha256(path), "size": path.stat().st_size})
    return {
        "schema": SCHEMA,
        "artifact_kind": "locked_flagged_ply_audit_execution",
        "status": "launch_bound_after_authoritative_final_actor_selection",
        "source_parent_commit": source_commit,
        "source_parent_tree": source_tree,
        "branch": BRANCH,
        "plan": {"path": PLAN_PATH, "sha256": sha256(plan)},
        "workflow": {"path": WORKFLOW_PATH, "sha256": sha256(workflow)},
        "manifest": manifest,
        "tools": tools,
        "corpus_artifacts": corpus_artifacts,
        "authoritative_final_actor_result": final,
        "actors": {
            "reference": final["reference"],
            "winner": final["winner"],
            "winner_is_reference": final["no_change"],
            "selection_rule": (
                "revalidate final_actor_result selection_mode and every bound "
                "result; derive rather than accept its explicit winner_actor"
            ),
        },
        "audit": {
            "decision_worlds_per_actor_per_case": DECISION_WORLDS,
            "history_worlds": HISTORY_WORLDS,
            "belief_alpha": BELIEF_ALPHA,
            "base_seed": BASE_SEED,
            "shard_count": SHARD_COUNT,
            "candidate_rule": (
                "top three complete semantic policy moves per actor; "
                "deterministic union capped at five; never all legal moves"
            ),
        },
        "build": {
            "runner": "ubuntu-24.04",
            "compiler": COMPILER,
            "compiler_semantic_version_command": COMPILER_SEMANTIC_VERSION_COMMAND,
            "required_compiler_semantic_version": REQUIRED_COMPILER_SEMANTIC_VERSION,
            "cflags": CFLAGS,
            "ldflags": LDFLAGS,
            "binding": "compile once in preflight; SHA-256 transport everywhere else",
        },
        "results": None,
    }


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False) as stream:
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
        raise ExecutionError(f"{path} already exists") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_execution(root: Path, output: Path, source_commit: str,
                      source_tree: str) -> dict[str, Any]:
    if output.resolve() != (root / EXECUTION_PATH).resolve():
        raise ExecutionError("execution output must use the canonical path")
    value = expected_execution(root, source_commit, source_tree)
    _atomic_create(output, value)
    return value


def guard_execution(root: Path, execution: Path, source_commit: str,
                    source_tree: str) -> dict[str, Any]:
    expected = expected_execution(root, source_commit, source_tree)
    if strict_json(execution) != expected:
        raise ExecutionError(
            "execution addendum does not exactly match the locked audit and "
            "authoritative final actor result"
        )
    return expected


def verify_one_shot_add(root: Path, before: str, after: str) -> None:
    if _HEX40.fullmatch(before) is None or set(before) == {"0"} or \
            _HEX40.fullmatch(after) is None:
        raise ExecutionError("one-shot launch requires canonical commits")
    try:
        parents = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", after],
            cwd=root, text=True, stderr=subprocess.STDOUT).split()
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", after],
            cwd=root, text=True, stderr=subprocess.STDOUT).splitlines()
        existed = subprocess.run(
            ["git", "cat-file", "-e", f"{before}:{EXECUTION_PATH}"],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        history = subprocess.check_output(
            ["git", "rev-list", "--all", "--count", "--", EXECUTION_PATH],
            cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionError(f"cannot verify one-shot launch topology: {exc}") from exc
    if parents != [after, before] or changed != [f"A\t{EXECUTION_PATH}"] or \
            existed or history != "1":
        raise ExecutionError("launch must be the unique direct-parent addendum-only commit")


def emit_github_output(path: Path, value: dict[str, Any]) -> None:
    final = value["authoritative_final_actor_result"]
    outputs = {
        "reference_actor": value["actors"]["reference"]["spec"],
        "winner_actor": value["actors"]["winner"]["spec"],
        "worlds": str(DECISION_WORLDS),
        "belief_alpha": str(BELIEF_ALPHA),
        "history_worlds": str(HISTORY_WORLDS),
        "base_seed": str(BASE_SEED),
        "shard_count": str(SHARD_COUNT),
        "final_result_sha": final["sha256"],
        "selection_mode": final["selection_mode"],
        "final_gate_passed": "true" if final["promotion_gate_passed"] else "false",
        "plan_sha": value["plan"]["sha256"],
        "workflow_sha": value["workflow"]["sha256"],
        "manifest_sha": value["manifest"]["sha256"],
    }
    if any("\n" in item or "\r" in item for item in outputs.values()):
        raise ExecutionError("multiline GitHub output is forbidden")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as stream:
        for key, item in outputs.items():
            stream.write(f"{key}={item}\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-execution", "guard-execution"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--source-parent-commit", required=True)
        command.add_argument("--source-parent-tree", required=True)
        command.add_argument("--execution", type=Path, required=True)
        if name == "guard-execution":
            command.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare-execution":
            prepare_execution(args.root, args.execution,
                              args.source_parent_commit, args.source_parent_tree)
        else:
            value = guard_execution(args.root, args.execution,
                                    args.source_parent_commit, args.source_parent_tree)
            if args.github_output:
                emit_github_output(args.github_output, value)
            else:
                print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except ExecutionError as exc:
        print(f"flagged_ply_execution.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
