#!/usr/bin/env python3
"""Freeze and guard the one-shot exact-17 commented-ply audit.

The audit is diagnostic and occurs only after ``final_actor_result.json`` has
mechanically selected the maintained actor.  This helper deliberately reuses
the final-result verifier from :mod:`tools.flagged_ply_execution`, but it binds
the much narrower literal 17-case corpus and never launches the older 36-case
flagged-ply workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.flagged_ply_execution import (  # noqa: E402
    ExecutionError,
    authoritative_final_result,
    sha256,
    strict_json,
)


SCHEMA = "lc-commented-ply-audit-execution-v1"
LOCK_SCHEMA = "lc-commented-ply-audit-definition-lock-v1"
ATTEMPT_ID = "v3"
PLAN_PATH = "data/experiments/locked_commented_ply_audit_v3_plan.json"
LOCK_PATH = (
    "data/experiments/locked_commented_ply_audit_definition_lock_v3.json"
)
EXECUTION_PATH = (
    "data/experiments/locked_commented_ply_audit_v3_execution.json"
)
V2_PLAN_PATH = "data/experiments/locked_commented_ply_audit_plan.json"
V2_LOCK_PATH = (
    "data/experiments/locked_commented_ply_audit_definition_lock_v2.json"
)
V2_EXECUTION_PATH = (
    "data/experiments/locked_commented_ply_audit_execution.json"
)
V2_FAILURE_PATH = (
    "data/experiments/commented_ply_audit_v2_preflight_failure.json"
)
WORKFLOW_PATH = ".github/workflows/commented-ply-audit-v3.yml"
FINAL_RESULT_PATH = "data/experiments/final_actor_result.json"
DRIVER_PATH = "tools/audit_commented_plies.py"
HELPER_SOURCE_PATH = "tools/commented_ply_eval.c"
BRANCH = "agent/correctness-and-policy-upgrade"
TEACHER_SPEC = "policy:data/champion.bin:0:20"
TEACHER_PATH = "data/champion.bin"
DEFAULT_WORLDS = 1024
P10_WORLDS = 2048
SYMMETRIES = 20
BELIEF_ALPHA = 1.15
CASE_COUNT = 17
PRODUCTION_SEED_NAMESPACE = "20261001"
SMOKE_SEED_NAMESPACE = "20261009"
PRODUCTION_SEEDS = (
    "202610010103", "202610010104", "202610010108", "202610010110",
    "202610010112", "202610010113", "202610010116", "202610010120",
    "202610010214", "202610010215", "202610010217", "202610010232",
    "202610010321", "202610010322", "202610010323", "202610010325",
    "202610010444",
)
V2_PRODUCTION_SEEDS = (
    "202608230103", "202608230104", "202608230108", "202608230110",
    "202608230112", "202608230113", "202608230116", "202608230120",
    "202608230214", "202608230215", "202608230217", "202608230232",
    "202608230321", "202608230322", "202608230323", "202608230325",
    "202608230444",
)
V2_CONSUMED_PREFLIGHT_SEEDS = ("202608230113", "202608230444")
SMOKE_SEEDS = (
    "202610090001", "202610090002", "202610090013", "202610090044",
    "202610090099",
)
RECOVERY_ARTIFACTS = (
    (V2_PLAN_PATH,
     "ac1aab28cb2507c3d37b6692ae8623dc7f7783aaacc9600210f298f705eb0ab0"),
    (V2_LOCK_PATH,
     "99ef765235f820711d3a8e0853c5c51e87fc7afe8a867c5acec029d0ed2cfd2f"),
    (V2_EXECUTION_PATH,
     "de385af4ec2e7ac004f72b1a928bc86a461e39b91c2ca05c28b452921eb6951b"),
    (V2_FAILURE_PATH,
     "fef8abdc6348e7ac538a604773c2b0da93cc8e05e0d6f2129fce117acbb9eb9e"),
)
COMPILER = "gcc"
COMPILER_SEMANTIC_VERSION_COMMAND = "gcc -dumpfullversion -dumpversion"
REQUIRED_COMPILER_SEMANTIC_VERSION = "13.3.0"
CFLAGS = (
    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
    "-Wall -Wextra -std=c11"
)
LDFLAGS = "-lm -pthread"
CASE_IDS = (
    "ui-221-p3", "ui-221-p4", "ui-221-p8", "ui-221-p10",
    "ui-221-p12", "ui-221-p13", "ui-221-p16", "ui-221-p20",
    "showcase-572-p14", "showcase-572-p15", "showcase-572-p17",
    "showcase-572-p32", "ui-725-p21", "ui-725-p22", "ui-725-p23",
    "ui-725-p25", "ui-956-p44",
)
TOOL_PATHS = (
    "tools/commented_ply_execution.py",
    DRIVER_PATH,
    HELPER_SOURCE_PATH,
    "tools/flagged_ply_execution.py",
    "tools/gate_actor_panel.py",
    "tools/match_value_campaign.py",
    "tools/policy_cost_artifact.py",
    "tools/merge_arena.py",
)
TEST_PATHS = (
    "tests/test_commented_ply_audit.py",
    "tests/test_commented_ply_execution.py",
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

# The completed v3 lock predates rollout5.  Its source closure is an immutable
# historical fact, not today's src/ glob: using the live glob made adding any
# later source file retroactively invalidate the already sealed campaign.
# Future definition commits deliberately use the expanded live closure below.
V3_DEFINITION_COMMIT = "7a8e7ee0598e461cf3bad03d3817a76671341327"
V3_DEFINITION_PATHS = (
    ".github/workflows/commented-ply-audit-v3.yml",
    "Makefile",
    "data/experiments/locked_commented_ply_audit_v3_plan.json",
    "src/agent.c", "src/agent.h", "src/features.c", "src/features.h",
    "src/heuristic.c", "src/heuristic.h", "src/late_resolver.c",
    "src/late_resolver.h", "src/lc.c", "src/lc.h", "src/match.c",
    "src/match.h", "src/match_value.c", "src/match_value.h",
    "src/net.c", "src/net.h", "src/planner.c", "src/planner.h",
    "src/rollout.c", "src/search.c", "src/search.h", "src/spec.c",
    "src/spec.h", "tests/test_commented_ply_audit.py",
    "tests/test_commented_ply_execution.py",
    "tools/audit_commented_plies.py", "tools/commented_ply_eval.c",
    "tools/commented_ply_execution.py", "tools/flagged_ply_execution.py",
    "tools/gate_actor_panel.py", "tools/match_value_campaign.py",
    "tools/merge_arena.py",
)


def _repo_file(root: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExecutionError(f"{label} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or relative.as_posix() != value or any(
            part in {"", ".", ".."} for part in relative.parts):
        raise ExecutionError(f"{label} is not a canonical repository path")
    path = root / relative
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ExecutionError(f"{label} crosses a symbolic link")
    if not path.is_file():
        raise ExecutionError(f"{label} is absent: {value}")
    return path, value


def _case_row(case: Any, root: Path) -> dict[str, Any]:
    required_attributes = (
        "case_id", "source_seed", "ply", "state", "candidates",
        "reviewed_moves", "review", "audit_seed", "min_worlds",
        "belief_card",
    )
    if any(not hasattr(case, name) for name in required_attributes):
        raise ExecutionError("commented-ply CASES entry has the wrong shape")
    path, state = _repo_file(root, case.state, f"{case.case_id} state")
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "source_seed": str(case.source_seed),
        "ply": case.ply,
        "state": state,
        "state_sha256": sha256(path),
        "candidates": list(case.candidates),
        "reviewed_moves": list(case.reviewed_moves),
        "review": case.review,
        "audit_seed": str(case.audit_seed),
        "min_worlds": case.min_worlds,
        "belief_card": case.belief_card,
    }
    if case.case_id == "ui-221-p13":
        view_name = "data/probes/ui_seed2214615196_p13.view.json"
        view, _ = _repo_file(root, view_name, "ui-221-p13 view")
        row["view"] = view_name
        row["view_sha256"] = sha256(view)
    return row


def _without_audit_seed(rows: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ExecutionError(f"{label} case rows are malformed")
    stripped = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("audit_seed"), str):
            raise ExecutionError(f"{label} case row lacks its audit seed")
        copy = dict(row)
        del copy["audit_seed"]
        stripped.append(copy)
    return stripped


def _recovery_binding(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("attempt_id") != ATTEMPT_ID:
        raise ExecutionError("locked plan attempt identity drifted")
    recovery = plan.get("recovery")
    firewall = plan.get("seed_firewall")
    if not isinstance(recovery, dict) or not isinstance(firewall, dict):
        raise ExecutionError("locked plan lacks v3 recovery/seed firewall")
    expected_paths = {
        "previous_plan": V2_PLAN_PATH,
        "previous_definition_lock": V2_LOCK_PATH,
        "previous_execution": V2_EXECUTION_PATH,
        "previous_failure": V2_FAILURE_PATH,
    }
    expected_hashes = dict(RECOVERY_ARTIFACTS)
    for key, name in expected_paths.items():
        row = recovery.get(key)
        if row != {"path": name, "sha256": expected_hashes[name]}:
            raise ExecutionError(f"v3 recovery {key} binding drifted")
    if recovery.get("previous_attempt") != "v2" or \
            recovery.get("rerun_previous_attempt") != "forbidden":
        raise ExecutionError("v3 recovery does not burn the failed attempt")

    artifacts = []
    for name, expected in RECOVERY_ARTIFACTS:
        path, _ = _repo_file(root, name, f"v3 recovery artifact {name}")
        if sha256(path) != expected:
            raise ExecutionError(f"v3 recovery artifact drifted: {name}")
        artifacts.append(_current_binding(
            root, name, f"v3 recovery artifact {name}"))

    previous_plan = strict_json(root / V2_PLAN_PATH)
    previous_rows = previous_plan.get("cases")
    current_rows = plan.get("cases")
    if _without_audit_seed(current_rows, "v3") != \
            _without_audit_seed(previous_rows, "v2"):
        raise ExecutionError(
            "v3 case rows must differ from immutable v2 only by audit_seed")
    old_seeds = tuple(row["audit_seed"] for row in previous_rows)
    new_seeds = tuple(row["audit_seed"] for row in current_rows)
    if old_seeds != V2_PRODUCTION_SEEDS or \
            new_seeds != PRODUCTION_SEEDS or \
            len(set(new_seeds)) != CASE_COUNT or \
            set(new_seeds) & set(old_seeds) or any(
                not seed.startswith(PRODUCTION_SEED_NAMESPACE)
                for seed in new_seeds):
        raise ExecutionError("v3 production audit seed firewall drifted")
    if firewall != {
        "burned_v2_production_namespace": "20260823",
        "retired_v2_production_seeds": list(V2_PRODUCTION_SEEDS),
        "consumed_v2_preflight_seeds": list(V2_CONSUMED_PREFLIGHT_SEEDS),
        "production_namespace": PRODUCTION_SEED_NAMESPACE,
        "smoke_namespace": SMOKE_SEED_NAMESPACE,
        "namespace_rule": (
            "Production and smoke namespaces are disjoint. Smoke "
            "computations cannot consume a production seed, and no v2 "
            "audit seed may be reused."
        ),
        "declared_smoke_seeds": list(SMOKE_SEEDS),
    } or set(new_seeds) & set(SMOKE_SEEDS) or any(
            not seed.startswith(SMOKE_SEED_NAMESPACE)
            for seed in SMOKE_SEEDS):
        raise ExecutionError("v3 production/smoke seed namespaces drifted")

    failure = strict_json(root / V2_FAILURE_PATH)
    computations = failure.get("audit_efficacy", {}).get(
        "preflight_test_computations")
    consumed = tuple(str(row.get("audit_seed")) for row in computations) \
        if isinstance(computations, list) else ()
    if failure.get("status") != \
            "complete_fail_closed_before_any_canonical_case_job" or \
            failure.get("run", {}).get("run_id") != 32714751993 or \
            failure.get("run", {}).get("attempt") != 1 or \
            failure.get("disposition", {}).get("rerun_forbidden") is not True or \
            failure.get("audit_efficacy", {}).get(
                "case_evaluation_jobs_started") != 0 or \
            failure.get("audit_efficacy", {}).get("ply_evidence_produced") \
            is not False or consumed != V2_CONSUMED_PREFLIGHT_SEEDS or any(
                row.get("worlds") != 2 or
                row.get("persistence") != "ephemeral_test_output_only"
                for row in computations):
        raise ExecutionError("v2 failed-attempt disposition drifted")
    return {
        "attempt_id": ATTEMPT_ID,
        "previous_attempt": "v2",
        "rerun_previous_attempt": "forbidden",
        "production_seed_namespace": PRODUCTION_SEED_NAMESPACE,
        "smoke_seed_namespace": SMOKE_SEED_NAMESPACE,
        "artifacts": artifacts,
    }


def _case_binding(root: Path, plan: dict[str, Any]) -> tuple[
        dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _recovery_binding(root, plan)
    try:
        from tools.audit_commented_plies import CASES, definition_sha256
    except (ImportError, OSError, ValueError) as exc:
        raise ExecutionError(f"cannot import commented-ply definition: {exc}") from exc
    rows = [_case_row(case, root) for case in CASES]
    ids = tuple(row["case_id"] for row in rows)
    if ids != CASE_IDS or len(rows) != CASE_COUNT:
        raise ExecutionError("commented-ply CASES is not the exact ordered 17-set")
    if [row["min_worlds"] for row in rows] != [
            P10_WORLDS if case_id == "ui-221-p10" else DEFAULT_WORLDS
            for case_id in CASE_IDS]:
        raise ExecutionError("commented-ply world budgets drifted")
    p13 = rows[5]
    if p13["source_seed"] != "2214615196" or p13["ply"] != 13 or \
            p13["belief_card"] != "Y9" or p13["candidates"]:
        raise ExecutionError("fixed-K belief case drifted")
    plan_cases = plan.get("cases")
    if rows != plan_cases:
        raise ExecutionError("driver CASES/state hashes differ from locked plan")
    if any(plan.get(key) != value for key, value in {
            "case_count": 17,
            "action_panel_cases": 17,
            "nominated_action_cases": 16,
            "fixed_k_belief_cases": 1,
    }.items()):
        raise ExecutionError("locked plan case counts drifted")
    definition = definition_sha256(CASES)
    if plan.get("case_definition_sha256") != definition:
        raise ExecutionError("commented-ply definition hash differs from plan")
    artifacts: dict[str, dict[str, Any]] = {}
    for row in rows:
        for prefix in ("state", "view"):
            name = row.get(prefix)
            if name is None:
                continue
            path = root / name
            artifacts[name] = {
                "path": name,
                "sha256": row[f"{prefix}_sha256"],
                "size": path.stat().st_size,
            }
    return ({
        "case_count": CASE_COUNT,
        "case_ids": list(CASE_IDS),
        "definition_sha256": definition,
        "action_panel_cases": 17,
        "nominated_action_cases": 16,
        "fixed_k_belief_cases": 1,
    }, rows, [artifacts[name] for name in sorted(artifacts)])


def _tool_bindings(root: Path) -> list[dict[str, Any]]:
    names = list(TOOL_PATHS)
    result = []
    for name in names:
        path, _ = _repo_file(root, name, f"audit tool {name}")
        result.append({
            "path": name, "sha256": sha256(path), "size": path.stat().st_size,
        })
    return result


def _definition_paths(
    root: Path, definition_commit: str | None = None,
) -> tuple[str, ...]:
    """Return the complete evaluator/verifier source closure frozen by the lock."""
    if definition_commit == V3_DEFINITION_COMMIT:
        return V3_DEFINITION_PATHS
    names = {
        "Makefile", PLAN_PATH, WORKFLOW_PATH, DRIVER_PATH,
        HELPER_SOURCE_PATH, *TOOL_PATHS, *TEST_PATHS,
    }
    src = root / "src"
    if not src.is_dir():
        raise ExecutionError("evaluator src directory is absent")
    for pattern in ("*.c", "*.h"):
        for path in src.glob(pattern):
            if not path.is_file() or path.is_symlink():
                raise ExecutionError("evaluator source closure contains a non-file")
            names.add(path.relative_to(root).as_posix())
    return tuple(sorted(names))


def _current_binding(root: Path, name: str, label: str,
                     git_mode: str = "100644") -> dict[str, Any]:
    path, relative = _repo_file(root, name, label)
    return {
        "path": relative,
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "git_mode": git_mode,
    }


def _git_blob_binding(root: Path, commit: str, name: str,
                      label: str) -> dict[str, Any]:
    try:
        row = subprocess.check_output(
            ["git", "ls-tree", commit, "--", name], cwd=root, text=True,
            stderr=subprocess.STDOUT,
        ).rstrip("\n")
        if not row:
            raise ExecutionError(f"{label} is absent from definition commit")
        metadata, found = row.split("\t", 1)
        mode, kind, blob = metadata.split()
        if found != name or kind != "blob" or mode not in {"100644", "100755"}:
            raise ExecutionError(f"{label} is not a regular committed blob")
        payload = subprocess.check_output(
            ["git", "cat-file", "blob", blob], cwd=root,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise ExecutionError(f"cannot bind {label} from definition commit: {exc}") \
            from exc
    return {
        "path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "git_mode": mode,
    }


def _has_git(root: Path) -> bool:
    return (root / ".git").exists()


def _commit_tree(root: Path, commit: str, label: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"{commit}^{{tree}}"], cwd=root, text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionError(f"cannot resolve {label} commit/tree") from exc


def expected_definition_lock(root: Path, definition_commit: str,
                             definition_tree: str) -> dict[str, Any]:
    if _HEX40.fullmatch(definition_commit) is None or \
            _HEX40.fullmatch(definition_tree) is None:
        raise ExecutionError("definition commit/tree must be canonical SHA-1")
    if _has_git(root) and _commit_tree(
            root, definition_commit, "definition") != definition_tree:
        raise ExecutionError("definition commit/tree mismatch")

    plan_path, _ = _repo_file(root, PLAN_PATH, "locked exact-17 plan")
    plan = strict_json(plan_path)
    if plan.get("schema") != "lc-commented-ply-audit-plan-v1" or \
            plan.get("status") != "definition_source_pending_unique_seal":
        raise ExecutionError("locked exact-17 plan has the wrong seal contract")
    if plan.get("definition_lock", {}).get("path") != LOCK_PATH:
        raise ExecutionError("plan does not name the canonical definition lock")
    cases, rows, corpus = _case_binding(root, plan)
    recovery = _recovery_binding(root, plan)

    names = _definition_paths(root, definition_commit)
    if _has_git(root):
        definition_files = [
            _git_blob_binding(root, definition_commit, name,
                              f"definition file {name}")
            for name in names
        ]
        # Revalidation of the completed v3 campaign reads its evaluator from
        # the recorded commit.  Later source additions must not retroactively
        # require the live checkout to equal that old tree.  New definitions
        # retain the strict live-vs-commit drift check.
        if definition_commit != V3_DEFINITION_COMMIT:
            for committed in definition_files:
                current = _current_binding(
                    root, committed["path"],
                    f"current definition file {committed['path']}",
                    committed["git_mode"],
                )
                if current != committed:
                    raise ExecutionError(
                        f"definition file drifted from definition commit: "
                        f"{committed['path']}")
    else:
        definition_files = [
            _current_binding(root, name, f"definition file {name}")
            for name in names
        ]

    teacher = _current_binding(
        root, TEACHER_PATH, "policy-20 teacher checkpoint")
    locked_artifacts = [
        _current_binding(root, row["path"],
                         f"definition artifact {row['path']}")
        for row in corpus
    ] + [teacher] + recovery["artifacts"]
    if _has_git(root):
        committed_artifacts = [
            _git_blob_binding(root, definition_commit, row["path"],
                              f"definition artifact {row['path']}")
            for row in locked_artifacts
        ]
        if definition_commit != V3_DEFINITION_COMMIT:
            current_artifacts = [
                _current_binding(root, row["path"],
                                 f"current definition artifact {row['path']}",
                                 row["git_mode"])
                for row in committed_artifacts
            ]
            if current_artifacts != committed_artifacts:
                raise ExecutionError(
                    "corpus/checkpoint drifted from definition commit")
        locked_artifacts = committed_artifacts

    return {
        "schema": LOCK_SCHEMA,
        "artifact_kind": "immutable_exact_17_audit_definition_lock",
        "attempt_id": ATTEMPT_ID,
        "status": (
            "sealed_after_actor_selection_before_fresh_diagnostic_execution"
        ),
        "branch": BRANCH,
        "definition": {
            "commit": definition_commit,
            "tree": definition_tree,
        },
        "definition_files": definition_files,
        "cases": cases,
        "case_rows": rows,
        "recovery": recovery,
        "locked_artifacts": locked_artifacts,
        "teacher_checkpoint": teacher,
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


def validate_definition_lock(
    root: Path, source_commit: str, source_tree: str,
    lock_commit_hint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path, _ = _repo_file(root, LOCK_PATH, "exact-17 definition lock")
    lock = strict_json(lock_path)
    if lock_path.read_text(encoding="utf-8") != \
            json.dumps(lock, indent=2, sort_keys=True) + "\n":
        raise ExecutionError("definition lock is not canonical JSON")
    definition = lock.get("definition") if isinstance(lock, dict) else None
    if not isinstance(definition, dict):
        raise ExecutionError("definition lock lacks its source identity")
    definition_commit = definition.get("commit")
    definition_tree = definition.get("tree")
    if not isinstance(definition_commit, str) or \
            not isinstance(definition_tree, str):
        raise ExecutionError("definition lock source identity is malformed")
    expected = expected_definition_lock(
        root, definition_commit, definition_tree)
    if lock != expected:
        raise ExecutionError("definition lock differs from committed definition")

    if _has_git(root):
        if _HEX40.fullmatch(source_commit) is None or \
                _HEX40.fullmatch(source_tree) is None or \
                _commit_tree(root, source_commit, "source parent") != source_tree:
            raise ExecutionError("source parent commit/tree mismatch")
        try:
            history = subprocess.check_output(
                ["git", "log", "--all", "--format=%H", "--", LOCK_PATH],
                cwd=root, text=True, stderr=subprocess.STDOUT,
            ).splitlines()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ExecutionError("cannot inspect definition-lock history") from exc
        if len(history) != 1 or _HEX40.fullmatch(history[0]) is None:
            raise ExecutionError("definition lock must have exactly one history commit")
        lock_commit = history[0]
        try:
            parents = subprocess.check_output(
                ["git", "rev-list", "--parents", "-n", "1", lock_commit],
                cwd=root, text=True, stderr=subprocess.STDOUT,
            ).split()
            changes = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--name-status", "-r",
                 lock_commit], cwd=root, text=True,
                stderr=subprocess.STDOUT,
            ).splitlines()
            existed = subprocess.run(
                ["git", "cat-file", "-e", f"{definition_commit}:{LOCK_PATH}"],
                cwd=root, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", lock_commit,
                 source_commit], cwd=root, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            first_parent = subprocess.check_output(
                ["git", "rev-list", "--first-parent", source_commit],
                cwd=root, text=True, stderr=subprocess.STDOUT,
            ).splitlines()
            pinned_paths = [
                row["path"] for row in lock["definition_files"]
            ] + [row["path"] for row in lock["locked_artifacts"]]
            later_touches = subprocess.check_output(
                ["git", "log", "--format=%H",
                 f"{definition_commit}..{source_commit}", "--", *pinned_paths],
                cwd=root, text=True, stderr=subprocess.STDOUT,
            ).splitlines()
            committed_lock = _git_blob_binding(
                root, lock_commit, LOCK_PATH, "committed definition lock")
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ExecutionError("cannot verify definition-lock topology") from exc
        current_lock = _current_binding(
            root, LOCK_PATH, "current definition lock",
            committed_lock["git_mode"])
        if parents != [lock_commit, definition_commit] or \
                changes != [f"A\t{LOCK_PATH}"] or existed or not ancestor or \
                committed_lock["git_mode"] != "100644" or \
                current_lock != committed_lock or lock_commit not in first_parent or \
                later_touches:
            raise ExecutionError(
                "definition lock must be the unique add-only child of its "
                "definition and an ancestor of the source parent")
        if lock_commit_hint is not None and lock_commit_hint != lock_commit:
            raise ExecutionError("definition-lock commit hint drift")
    else:
        if lock_commit_hint is None or _HEX40.fullmatch(lock_commit_hint) is None:
            raise ExecutionError("sealed transport lacks definition-lock commit")
        lock_commit = lock_commit_hint

    binding = {
        "path": LOCK_PATH,
        "sha256": sha256(lock_path),
        "size": lock_path.stat().st_size,
        "lock_commit": lock_commit,
        "definition_commit": definition_commit,
        "definition_tree": definition_tree,
    }
    return binding, lock


def expected_execution(root: Path, source_commit: str,
                       source_tree: str,
                       final_binding: dict[str, Any] | None = None,
                       definition_lock_binding: tuple[
                           dict[str, Any], dict[str, Any]] | None = None,
                       lock_commit_hint: str | None = None,
                       ) -> dict[str, Any]:
    if _HEX40.fullmatch(source_commit) is None or \
            _HEX40.fullmatch(source_tree) is None:
        raise ExecutionError("source parent commit/tree must be canonical SHA-1")
    lock_binding, lock = definition_lock_binding \
        if definition_lock_binding is not None else validate_definition_lock(
            root, source_commit, source_tree, lock_commit_hint)
    plan_path, _ = _repo_file(root, PLAN_PATH, "locked exact-17 plan")
    workflow_path, _ = _repo_file(root, WORKFLOW_PATH, "locked exact-17 workflow")
    plan = strict_json(plan_path)
    if plan.get("schema") != "lc-commented-ply-audit-plan-v1" or \
            plan.get("status") != "definition_source_pending_unique_seal":
        raise ExecutionError("locked exact-17 plan has the wrong contract")
    cases, rows, corpus = _case_binding(root, plan)
    recovery = _recovery_binding(root, plan)
    if lock.get("attempt_id") != ATTEMPT_ID or \
            lock.get("cases") != cases or lock.get("case_rows") != rows or \
            lock.get("recovery") != recovery:
        raise ExecutionError("sealed exact-17 cases drifted")
    final = final_binding if final_binding is not None \
        else authoritative_final_result(root)
    required_final = {
        "path", "sha256", "selection_mode", "source_commit", "source_tree",
        "decisive_result", "authoritative_results", "promotion_gate_passed",
        "reference", "challenger", "winner", "actor_assets", "no_change",
    }
    if not isinstance(final, dict) or set(final) != required_final or \
            final.get("path") != FINAL_RESULT_PATH or \
            not isinstance(final.get("winner"), dict) or \
            not isinstance(final["winner"].get("spec"), str) or \
            _HEX64.fullmatch(str(final.get("sha256"))) is None:
        raise ExecutionError("authoritative final actor binding is malformed")
    if final["selection_mode"] == "composition_final":
        raise ExecutionError("sealed audit definition does not support a later composition")
    teacher_path, _ = _repo_file(root, TEACHER_PATH, "policy-20 teacher")
    if plan.get("continuation", {}).get("actor") != TEACHER_SPEC:
        raise ExecutionError("plan does not lock the policy-20 teacher")
    return {
        "schema": SCHEMA,
        "artifact_kind": "locked_exact_17_commented_ply_audit_execution",
        "attempt_id": ATTEMPT_ID,
        "status": "launch_bound_after_authoritative_final_actor_selection",
        "source_parent_commit": source_commit,
        "source_parent_tree": source_tree,
        "branch": BRANCH,
        "definition_lock": lock_binding,
        "plan": {"path": PLAN_PATH, "sha256": sha256(plan_path)},
        "workflow": {"path": WORKFLOW_PATH, "sha256": sha256(workflow_path)},
        "tools": _tool_bindings(root),
        "cases": cases,
        "corpus_artifacts": corpus,
        "recovery": recovery,
        "authoritative_final_actor_result": final,
        "subject": {
            "actor": final["winner"],
            "selection_rule": (
                "revalidate final_actor_result and mechanically derive its "
                "winner; never accept a manually supplied audit actor"
            ),
        },
        "continuation": {
            "actor": TEACHER_SPEC,
            "checkpoint": {
                "path": TEACHER_PATH,
                "sha256": sha256(teacher_path),
                "size": teacher_path.stat().st_size,
            },
            "symmetries": SYMMETRIES,
            "scope": "full_remaining_three_round_match",
            "pairing": "same hidden world, future deal, and branch-neutral RNG domain",
        },
        "audit": {
            "diagnostic_only": True,
            "training_use": "forbidden",
            "promotion_use": "forbidden",
            "default_paired_worlds": DEFAULT_WORLDS,
            "ui_221_p10_paired_worlds": P10_WORLDS,
            "belief": "hidden-information-safe exact fixed-K at ui-221-p13",
            "actor_selected_action": "append semantically when outside nominated support",
            "shard_count": CASE_COUNT,
            "assignment": "one immutable ordered CASES entry per shard",
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
            "binding": "compile once in preflight; SHA-256 transport everywhere else",
        },
        "results": None,
    }


def _atomic_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
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


def prepare_definition_lock(root: Path, output: Path, definition_commit: str,
                            definition_tree: str) -> dict[str, Any]:
    if output.resolve() != (root / LOCK_PATH).resolve():
        raise ExecutionError("definition lock output must use the canonical path")
    if not _has_git(root):
        raise ExecutionError("definition lock must be prepared in the source repository")
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionError("cannot resolve definition HEAD") from exc
    if head != definition_commit:
        raise ExecutionError("definition lock must seal the checked-out HEAD")
    value = expected_definition_lock(root, definition_commit, definition_tree)
    _atomic_create(output, value)
    return value


def prepare_execution(root: Path, output: Path, source_commit: str,
                      source_tree: str) -> dict[str, Any]:
    if output.resolve() != (root / EXECUTION_PATH).resolve():
        raise ExecutionError("execution output must use the canonical path")
    value = expected_execution(root, source_commit, source_tree)
    _atomic_create(output, value)
    return value


def guard_execution(root: Path, execution: Path, source_commit: str,
                    source_tree: str) -> dict[str, Any]:
    supplied = strict_json(execution)
    hint = supplied.get("definition_lock", {}).get("lock_commit") \
        if isinstance(supplied.get("definition_lock"), dict) else None
    value = expected_execution(
        root, source_commit, source_tree, lock_commit_hint=hint)
    if supplied != value:
        raise ExecutionError(
            "execution addendum differs from exact-17 plan or mechanical winner")
    return value


def verify_one_shot_add(root: Path, before: str, after: str) -> None:
    if _HEX40.fullmatch(before) is None or set(before) == {"0"} or \
            _HEX40.fullmatch(after) is None:
        raise ExecutionError("one-shot launch requires canonical commits")
    try:
        parents = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", after], cwd=root,
            text=True, stderr=subprocess.STDOUT).split()
        changes = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", after],
            cwd=root, text=True, stderr=subprocess.STDOUT).splitlines()
        existed = subprocess.run(
            ["git", "cat-file", "-e", f"{before}:{EXECUTION_PATH}"], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        history = subprocess.check_output(
            ["git", "rev-list", "--all", "--count", "--", EXECUTION_PATH],
            cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionError(f"cannot verify one-shot launch topology: {exc}") from exc
    if parents != [after, before] or changes != [f"A\t{EXECUTION_PATH}"] or \
            existed or history != "1":
        raise ExecutionError("launch must be the unique direct-parent addendum-only commit")


def emit_github_output(path: Path, value: dict[str, Any]) -> None:
    final = value["authoritative_final_actor_result"]
    outputs = {
        "attempt_id": value["attempt_id"],
        "winner_actor": value["subject"]["actor"]["spec"],
        "final_result_sha": final["sha256"],
        "selection_mode": final["selection_mode"],
        "final_gate_passed": "true" if final["promotion_gate_passed"] else "false",
        "plan_sha": value["plan"]["sha256"],
        "workflow_sha": value["workflow"]["sha256"],
        "definition_sha": value["cases"]["definition_sha256"],
        "teacher_sha": value["continuation"]["checkpoint"]["sha256"],
        "definition_lock_sha": value["definition_lock"]["sha256"],
        "definition_lock_commit": value["definition_lock"]["lock_commit"],
        "definition_commit": value["definition_lock"]["definition_commit"],
        "definition_tree": value["definition_lock"]["definition_tree"],
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
    lock_command = commands.add_parser("prepare-definition-lock")
    lock_command.add_argument("--root", type=Path, required=True)
    lock_command.add_argument("--definition-commit", required=True)
    lock_command.add_argument("--definition-tree", required=True)
    lock_command.add_argument("--lock", type=Path, required=True)
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
        if args.command == "prepare-definition-lock":
            prepare_definition_lock(
                args.root, args.lock, args.definition_commit,
                args.definition_tree)
        elif args.command == "prepare-execution":
            prepare_execution(args.root, args.execution,
                              args.source_parent_commit, args.source_parent_tree)
        else:
            value = guard_execution(args.root, args.execution,
                                    args.source_parent_commit, args.source_parent_tree)
            if args.github_output:
                emit_github_output(args.github_output, value)
            else:
                print(json.dumps(value, indent=2, sort_keys=True))
    except (ExecutionError, OSError, ValueError) as exc:
        print(f"commented_ply_execution.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
