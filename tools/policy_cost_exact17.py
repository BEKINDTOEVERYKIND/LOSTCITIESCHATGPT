#!/usr/bin/env python3
"""Freeze the exact-17 information-view suit-orbit exclusion firewall.

This utility is deliberately diagnostic-only.  It reopens the already
authoritative v3 audit and its locked plan, invokes the native dataset tool's
``hash-probe`` mode for exactly those 17 saved states, and writes a minimal
runtime text manifest plus complete canonical provenance.  It has no code
path to policy discovery, allocation, search, fitting, or promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping


PLAN = "data/experiments/locked_commented_ply_audit_v3_plan.json"
AUDIT_FILES = (
    ("data/experiments/commented_ply_audit_v3.json",
     "be63dcae2ae1a179cf43a0c47e9971755290f9b3bfd90cc40fd4b6bd2838bbd7"),
    ("data/experiments/commented_ply_audit_v3.md",
     "a30eb93e4623e75ec2dae4c2cb73103b801d67954e0b02298e1a5b1082ebcd71"),
    ("data/experiments/commented_ply_audit_v3_result.json",
     "9897b402116b897942031ecbd46c50127b358c5a2c579e9c854720c667f55a82"),
    ("data/experiments/commented_ply_audit_v3_evidence.zip",
     "aacec0f3da9bbedd5d6512cf7bf2ef0d993232ed888d64f2e8099f5b15c03994"),
)
P13_VIEW = "data/probes/ui_seed2214615196_p13.view.json"
P13_VIEW_SHA256 = "a9ef8595235d5b1de3e168c13cbe57fe4943fd703cce665d1acee68b46944725"
DATASET_SOURCE = "tools/policy_cost_dataset.c"
TEXT_SCHEMA = "lc-policy-cost-exclusions-v1"
JSON_SCHEMA = "lc-policy-cost-exclusions-evidence-v1"
PROBE_SCHEMA = "lc-policy-cost-probe-orbit-v1"
EXPECTED_CASES = (
    ("2214615196", 3), ("2214615196", 4), ("2214615196", 8),
    ("2214615196", 10), ("2214615196", 12), ("2214615196", 13),
    ("2214615196", 16), ("2214615196", 20),
    ("5726968372613385", 14), ("5726968372613385", 15),
    ("5726968372613385", 17), ("5726968372613385", 32),
    ("725402798", 21), ("725402798", 22),
    ("725402798", 23), ("725402798", 25),
    ("95647345759839", 44),
)


class ExclusionError(ValueError):
    """An authoritative input or native orbit result is inconsistent."""


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExclusionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ExclusionError(f"non-finite JSON constant {token!r}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExclusionError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExclusionError(f"{path} is not a JSON object")
    return value


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ExclusionError(f"cannot hash {path}: {exc}") from exc


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ExclusionError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _canonical(value: Any) -> bytes:
    try:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExclusionError(f"noncanonical evidence: {exc}") from exc
    return (text + "\n").encode("ascii")


def _require_file(root: Path, relative: str, digest: str | None = None) -> str:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ExclusionError(f"bound input is absent/nonregular: {relative}")
    actual = _sha(path)
    if digest is not None and actual != digest:
        raise ExclusionError(f"bound input hash changed: {relative}")
    return actual


def _probe(root: Path, binary: Path, state: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(binary), "hash-probe", "--state", state],
            cwd=root, check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        value = json.loads(completed.stdout, object_pairs_hook=_unique)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ExclusionError(f"native hash-probe failed for {state}: {exc}") from exc
    required = {
        "schema", "state_path", "state_file_sha256", "state_sha256",
        "information_view_sha256", "suit_orbit_information_view_sha256",
    }
    if not isinstance(value, dict) or set(value) != required or \
            value.get("schema") != PROBE_SCHEMA or value.get("state_path") != state:
        raise ExclusionError(f"native hash-probe schema/path changed for {state}")
    for key in required - {"schema", "state_path"}:
        _hex64(value[key], f"{state}.{key}")
    return value


def build_outputs(root: Path, hash_probe: Path) -> tuple[bytes, dict[str, Any]]:
    """Return runtime text bytes and complete companion evidence."""

    root = root.resolve()
    binary = hash_probe if hash_probe.is_absolute() else root / hash_probe
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        raise ExclusionError("hash-probe binary is absent, symlinked, or nonexecutable")
    plan_sha = _require_file(root, PLAN)
    bound_files = []
    for relative, digest in AUDIT_FILES:
        _require_file(root, relative, digest)
        bound_files.append({"path": relative, "sha256": digest})
    _require_file(root, P13_VIEW, P13_VIEW_SHA256)
    dataset_source_sha = _require_file(root, DATASET_SOURCE)

    plan = _json(root / PLAN)
    audit = _json(root / AUDIT_FILES[0][0])
    result = _json(root / AUDIT_FILES[2][0])
    cases = plan.get("cases")
    if plan.get("schema") != "lc-commented-ply-audit-plan-v1" or \
            plan.get("attempt_id") != "v3" or plan.get("case_count") != 17 or \
            not isinstance(cases, list) or len(cases) != 17:
        raise ExclusionError("locked v3 plan identity/case count changed")
    expected = tuple((row.get("source_seed"), row.get("ply")) for row in cases)
    if expected != EXPECTED_CASES:
        raise ExclusionError("locked v3 plan is not the exact requested 17 plies")
    if result.get("schema") != "lc-commented-ply-audit-v3-result-v1" or \
            result.get("attempt_id") != "v3" or \
            result.get("status") != "complete_verified_diagnostic_only" or \
            result.get("bindings", {}).get("plan_sha256") != plan_sha:
        raise ExclusionError("authoritative v3 result/plan binding changed")
    result_audit = result.get("audit", {})
    if result_audit.get("cases_completed") != 17 or \
            result_audit.get("raw_shards") != 17 or \
            result_audit.get("counterfactual_cap_hits") != 0 or \
            result_audit.get("fixed_k_belief_valid") is not True or \
            result_audit.get("promotion_use") != "forbidden":
        raise ExclusionError("authoritative v3 result is incomplete/non-diagnostic")
    verification = result.get("independent_verification", {})
    if verification.get("exact_case_set_and_order_verified") is not True or \
            verification.get("exact_state_hashes_verified") is not True or \
            verification.get("zero_caps_verified") is not True:
        raise ExclusionError("v3 independent verification is incomplete")

    audit_cases = audit.get("cases")
    case_ids = [row.get("case_id") for row in cases]
    if audit.get("schema") != "lc-commented-ply-audit-v2" or \
            audit.get("attempt_id") != "v3" or \
            audit.get("selection", {}).get("case_ids") != case_ids or \
            not isinstance(audit_cases, list) or \
            [row.get("case_id") for row in audit_cases] != case_ids:
        raise ExclusionError("canonical audit case set/order changed")
    contract = audit.get("contract", {})
    if contract.get("case_count") != 17 or \
            contract.get("training_use") != "forbidden_diagnostic_only" or \
            contract.get("promotion_gate") is not False or \
            contract.get("decision_input") != "fresh_agent_information_view_at_every_node":
        raise ExclusionError("canonical audit diagnostic/information contract changed")

    evidence_cases: list[dict[str, Any]] = []
    orbit_seen: set[str] = set()
    for definition, archived in zip(cases, audit_cases):
        state = definition.get("state")
        state_sha = _hex64(definition.get("state_sha256"), "plan state hash")
        if not isinstance(state, str) or _require_file(root, state, state_sha) != state_sha:
            raise ExclusionError("plan state path/hash changed")
        source = archived.get("source", {})
        evaluation = archived.get("evaluation", {})
        counterfactual = evaluation.get("counterfactual", {})
        review = archived.get("review_context", {})
        if source.get("game_seed") != definition.get("source_seed") or \
                source.get("displayed_ply") != definition.get("ply") or \
                source.get("state_path") != state or \
                source.get("state_sha256") != state_sha or \
                evaluation.get("actor", {}).get("information_view") is not True or \
                counterfactual.get("root_information_view") is not True or \
                counterfactual.get("cap_hits") != 0 or \
                counterfactual.get("completed_worlds") != definition.get("min_worlds") or \
                counterfactual.get("requested_worlds") != definition.get("min_worlds") or \
                counterfactual.get("continuation", {}).get("symmetries") != 20 or \
                counterfactual.get("continuation", {}).get("scope") != \
                    "full_remaining_three_round_match" or \
                review.get("used_as_training_label") is not False or \
                review.get("used_as_promotion_gate") is not False:
            raise ExclusionError(f"archived case contract changed: {definition.get('case_id')}")
        probe = _probe(root, binary, state)
        if probe["state_file_sha256"] != state_sha:
            raise ExclusionError(f"native probe file hash changed: {state}")
        orbit = probe["suit_orbit_information_view_sha256"]
        if orbit in orbit_seen:
            raise ExclusionError("exact-17 information-view suit orbits are not unique")
        orbit_seen.add(orbit)
        evidence_cases.append({
            "case_id": definition["case_id"],
            "source_seed": definition["source_seed"],
            "displayed_ply": definition["ply"],
            "state_path": state,
            "state_file_sha256": state_sha,
            "state_sha256": probe["state_sha256"],
            "information_view_sha256": probe["information_view_sha256"],
            "suit_orbit_information_view_sha256": orbit,
        })
    if len(orbit_seen) != 17:
        raise ExclusionError("exact-17 orbit cardinality is not 17")

    p13 = next(row for row in audit_cases if row["case_id"] == "ui-221-p13")
    belief = p13.get("evaluation", {}).get("belief", {})
    if belief.get("kind") != "fixed_k" or \
            belief.get("information_view") is not True or \
            belief.get("complete_state_used_only_as_truth_label") is not True or \
            belief.get("valid") is not True or belief.get("need") != 8 or \
            abs(float(belief.get("marginal_sum", -1.0)) - 8.0) > 1e-6:
        raise ExclusionError("p13 fixed-K information boundary changed")

    text = (TEXT_SCHEMA + "\n" + "".join(
        row["suit_orbit_information_view_sha256"] + "\n"
        for row in evidence_cases
    )).encode("ascii")
    text_sha = hashlib.sha256(text).hexdigest()
    companion = {
        "schema": JSON_SCHEMA,
        "case_count": 17,
        "orbit_count": 17,
        "order": "locked exact-17 v3 plan order",
        "orbit_key": (
            "suit-symmetrized mover information view: public state/history, "
            "mover hand, round, nply, cumulative score, and deck count; no "
            "opponent private hand or hidden deck order"
        ),
        "runtime_exclusion_text": {
            "schema": TEXT_SCHEMA,
            "sha256": text_sha,
            "line_count": 18,
        },
        "bindings": {
            "locked_plan": {"path": PLAN, "sha256": plan_sha},
            "canonical_exact17": bound_files,
            "p13_information_view": {
                "path": P13_VIEW, "sha256": P13_VIEW_SHA256,
            },
            "native_hash_probe": {
                "binary_sha256": _sha(binary),
                "source_path": DATASET_SOURCE,
                "source_sha256": dataset_source_sha,
                "mode": "hash-probe",
                "discovery_or_efficacy_reachable": False,
            },
        },
        "p13_fixed_k": {
            "kind": "fixed_k",
            "information_view": True,
            "complete_state_used_only_as_truth_label": True,
            "valid": True,
            "need": 8,
            "marginal_sum": belief["marginal_sum"],
            "target": belief.get("target"),
        },
        "cases": evidence_cases,
        "training_use": "forbidden",
        "selection_use": "forbidden",
        "promotion_use": "forbidden",
    }
    companion["canonical_payload_sha256"] = hashlib.sha256(
        _canonical(companion)
    ).hexdigest()
    return text, companion


def _publish_pair(text_path: Path, text: bytes,
                  json_path: Path, companion: Mapping[str, Any]) -> None:
    if text_path.exists() or json_path.exists():
        raise ExclusionError("refusing to replace immutable exclusion evidence")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    linked: list[Path] = []
    try:
        for destination, payload in (
            (text_path, text), (json_path, _canonical(companion))
        ):
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".partial",
                dir=destination.parent, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                staged.append(temporary)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
            linked.append(destination)
        for directory in {text_path.parent, json_path.parent}:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for path in linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--root", type=Path, required=True)
    export.add_argument("--hash-probe", type=Path, required=True)
    export.add_argument("--text-out", type=Path, required=True)
    export.add_argument("--json-out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "export":
        text, companion = build_outputs(arguments.root, arguments.hash_probe)
        _publish_pair(
            arguments.text_out, text, arguments.json_out, companion
        )


if __name__ == "__main__":
    main()
