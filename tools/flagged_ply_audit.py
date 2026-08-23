#!/usr/bin/env python3
"""Run the frozen human-commented ply corpus without scanning every move.

For decision cases the C worker evaluates only the deterministic union of the
reference and candidate actors' top-three complete semantic policy moves,
capped at five.  Draw sources remain distinct.  Both actors receive the same uniform hidden-world panel
and production exact-late continuation rules.  Belief-only cases report the
fixed-cardinality network marginals from the same frozen information state;
they do not waste rollout worlds on an unrelated action claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/user_reviewed_plies.json"
DEFAULT_PROBE = ROOT / "bin/flagged_ply_probe"
DEFAULT_HISTORY_WORKER = ROOT / "bin/history_belief"

_HISTORY_PATH = ROOT / "tools/history_belief.py"
_HISTORY_SPEC = importlib.util.spec_from_file_location(
    "flagged_ply_history_belief", _HISTORY_PATH
)
if not _HISTORY_SPEC or not _HISTORY_SPEC.loader:
    raise RuntimeError("cannot load history_belief.py")
history_belief = importlib.util.module_from_spec(_HISTORY_SPEC)
_HISTORY_SPEC.loader.exec_module(history_belief)


class AuditError(RuntimeError):
    """A frozen input or diagnostic worker violated its contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_seed(base: int, case_id: str) -> int:
    token = hashlib.sha256(case_id.encode("utf-8")).digest()
    return (base ^ int.from_bytes(token[:8], "big")) & ((1 << 64) - 1)


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load manifest {path}: {exc}") from exc
    if manifest.get("schema") != "lc-user-reviewed-ply-corpus-v1":
        raise AuditError("unsupported corpus schema")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 36:
        raise AuditError("the literal user corpus must contain exactly 36 cases")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise AuditError(f"duplicate or invalid case id: {case_id!r}")
        seen.add(case_id)
        if case.get("kind") not in {"decision", "belief"}:
            raise AuditError(f"{case_id}: invalid case kind")
        relative = case.get("state")
        if not isinstance(relative, str):
            raise AuditError(f"{case_id}: state path is absent")
        state = (ROOT / relative).resolve()
        try:
            state.relative_to(ROOT)
        except ValueError as exc:
            raise AuditError(f"{case_id}: state escapes repository") from exc
        if not state.is_file():
            raise AuditError(f"{case_id}: missing state {relative}")
        actual = sha256(state)
        if actual != case.get("state_sha256"):
            raise AuditError(
                f"{case_id}: state hash changed ({actual}, expected "
                f"{case.get('state_sha256')})"
            )
        if case["kind"] == "belief":
            view = case.get("view")
            if not isinstance(view, str) or not (ROOT / view).is_file():
                raise AuditError(f"{case_id}: frozen perspective view is absent")
            actual_view = sha256(ROOT / view)
            if actual_view != case.get("view_sha256"):
                raise AuditError(f"{case_id}: perspective-view hash changed")
            inference = case.get("history_inference")
            if not isinstance(inference, dict):
                raise AuditError(f"{case_id}: history inference is absent")
            if not isinstance(inference.get("source_actor"), str):
                raise AuditError(f"{case_id}: history source actor is absent")
    return manifest, hashlib.sha256(raw).hexdigest()


def actor_layout(spec: str) -> tuple[list[str], int]:
    fields = spec.split(":")
    if not fields or fields[0] not in {
        "rollout", "rolloutu", "rollout2", "rolloutu2",
        "rollout3", "rolloutu3", "rollout4", "rolloutu4",
    }:
        raise AuditError("both audit actors must be rollout specifications")
    count = 3 if fields[0] in {
        "rollout3", "rolloutu3", "rollout4", "rolloutu4",
    } else (
        2 if fields[0] in {"rollout2", "rolloutu2"} else 1
    )
    if len(fields) <= count:
        raise AuditError(f"actor spec has missing checkpoint: {spec!r}")
    return fields, count


def actor_checkpoint_paths(spec: str) -> list[Path]:
    fields, count = actor_layout(spec)
    paths: list[Path] = []
    for item in fields[1 : count + 1]:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise AuditError(f"actor checkpoint is absent: {path}")
        paths.append(path)
    return paths


def actor_provenance(spec: str) -> dict[str, Any]:
    fields, checkpoint_count = actor_layout(spec)
    paths = actor_checkpoint_paths(spec)
    provenance = {
        "spec": spec,
        "checkpoints": [
            {
                "path": str(path.relative_to(ROOT))
                if path.is_relative_to(ROOT) else str(path),
                "sha256": sha256(path),
            }
            for path in paths
        ],
    }
    # Optional rollout-tail field 41 is a controller-bound match-value table.
    # It is just as strength-defining as a network checkpoint and therefore
    # must be hashed whenever a win_q=3 actor supplies it.
    match_value_field = 1 + checkpoint_count + 41
    if len(fields) > match_value_field:
        table = Path(fields[match_value_field])
        if not table.is_absolute():
            table = ROOT / table
        table = table.resolve()
        if not table.is_file():
            raise AuditError(f"actor match-value table is absent: {table}")
        provenance["match_value_table"] = {
            "path": str(table.relative_to(ROOT))
            if table.is_relative_to(ROOT) else str(table),
            "sha256": sha256(table),
        }
    return provenance


def classify_move(case: dict[str, Any], selected: str,
                  admitted: set[str], stage: str = "rollout_panel") -> str:
    preferred = set(case.get("preferred", []))
    criticized = set(case.get("criticized", []))
    if preferred and selected in preferred:
        return "review_aligned"
    if preferred and not (preferred & admitted):
        return "preferred_move_missing_from_top_policy_union"
    if selected in criticized:
        return f"flagged_move_selected_by_{stage}"
    if preferred:
        return f"preferred_move_admitted_but_not_selected_by_{stage}"
    if criticized & admitted:
        return f"flagged_move_admitted_but_avoided_by_{stage}"
    if criticized:
        return "flagged_move_outside_top_policy_union"
    return "descriptive_only"


def classify_policy(case: dict[str, Any], selected: str,
                    admitted: set[str]) -> str:
    preferred = set(case.get("preferred", []))
    criticized = set(case.get("criticized", []))
    if preferred and selected in preferred:
        return "review_aligned"
    if selected in criticized:
        return "flagged_move_is_policy_argmax"
    if preferred and not (preferred & admitted):
        return "preferred_move_missing_from_top_policy_union"
    if preferred:
        return "policy_prefers_another_admitted_move"
    if criticized & admitted:
        return "flagged_move_is_in_top_policy_union_not_argmax"
    if criticized:
        return "flagged_move_outside_top_policy_union"
    return "descriptive_only"


def focus_belief(case: dict[str, Any], actor: dict[str, Any]) -> list[dict[str, Any]]:
    belief = actor.get("belief")
    if not isinstance(belief, dict):
        raise AuditError(f"{case['id']}: worker omitted belief diagnostics")
    focus = set(case.get("focus_cards", []))
    return [card for card in belief.get("cards", []) if card.get("card") in focus]


def run_history_belief(
    case: dict[str, Any], args: argparse.Namespace, seed: int
) -> dict[str, Any]:
    inference = case["history_inference"]
    view_path = (ROOT / case["view"]).resolve()
    view = json.loads(view_path.read_text(encoding="utf-8"))
    checkpoint = Path(inference["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise AuditError(f"{case['id']}: history checkpoint is absent")
    if not args.history_worker.is_file():
        raise AuditError(f"history worker is absent: {args.history_worker}")
    try:
        source_actor = history_belief.validate_actor_prefix(
            inference["source_actor"], int(view["target_round_ply"]),
            int(inference["symmetries"]), checkpoint,
        )
        wire = history_belief.worker_wire(view)
    except (KeyError, TypeError, ValueError, history_belief.ViewError) as exc:
        raise AuditError(f"{case['id']}: invalid history view: {exc}") from exc
    history_seed = case_seed(seed, f"{case['id']}:history")
    command = [
        str(args.history_worker), "-n", str(checkpoint),
        "-w", str(args.history_worlds), "-s", str(history_seed),
        "-y", str(inference["symmetries"]),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, input=wire, text=True, capture_output=True
    )
    provenance = {
        "view": case["view"],
        "view_sha256": case["view_sha256"],
        "information_contract": (
            "observer initial hand + public actions + observer deck draws; "
            "opponent hidden draws, actual opponent hand, future actions, "
            "and truth labels are excluded"
        ),
        "source_actor": source_actor,
        "checkpoint": str(checkpoint.relative_to(ROOT))
        if checkpoint.is_relative_to(ROOT) else str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "symmetries": int(inference["symmetries"]),
        "seed": history_seed,
        "worlds": args.history_worlds,
        "applies_to": (
            "the frozen historical source actor; this is separate from each "
            "comparison actor's snapshot-only learned belief head"
        ),
    }
    if completed.returncode:
        detail = completed.stderr.strip()
        if "no sampled world reproduced the public prefix" in detail:
            return {
                "schema": "lc-history-belief-audit-v1",
                "status": "insufficient_accepted_support",
                "accepted": 0,
                "cards": [],
                "note": (
                    "The snapshot-only network marginals are not a "
                    "history-conditioned rebuttal; this actor-aware rejection "
                    "panel accepted no worlds at the frozen budget."
                ),
                "provenance": provenance,
            }
        raise AuditError(f"{case['id']}: history worker failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{case['id']}: history worker emitted invalid JSON") from exc
    for card in result.get("cards", []):
        card["card"] = history_belief.card_name(
            int(card["suit"]), int(card["value"])
        )
    result["status"] = "ok"
    result["provenance"] = provenance
    return result


def run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    seed = case_seed(args.seed, case["id"])
    worlds = 2 if case["kind"] == "belief" else args.worlds
    command = [
        str(args.probe),
        "-S", str(ROOT / case["state"]),
        "-a", args.reference,
        "-b", args.candidate,
        "-w", str(worlds),
        "-s", str(seed),
        "-B", str(args.belief_alpha),
    ]
    if case["kind"] == "belief":
        command.append("--belief-only")
    for move in case.get("preferred", []) + case.get("criticized", []):
        command.extend(["--assert-legal", move])
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True
    )
    if completed.returncode:
        detail = completed.stderr.strip() or "worker failed without diagnostics"
        raise AuditError(f"{case['id']}: {detail}")
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{case['id']}: worker emitted invalid JSON") from exc
    if probe.get("schema") != "lc-flagged-ply-probe-v1":
        raise AuditError(f"{case['id']}: worker schema mismatch")
    if int(probe.get("evaluated_moves", 0)) > 5:
        raise AuditError(f"{case['id']}: worker evaluated more than five moves")
    if int(probe.get("evaluated_moves", 0)) >= int(probe.get("legal_moves", 0)):
        # A genuinely forced/small position is allowed.  The corpus positions
        # are not, so equality would expose an accidental exhaustive audit.
        if int(probe.get("legal_moves", 0)) > 5:
            raise AuditError(f"{case['id']}: worker scanned all legal moves")
    admitted = {row["move"] for row in probe.get("candidates", [])}
    actors = probe.get("actors")
    if not isinstance(actors, list) or len(actors) != 2:
        raise AuditError(f"{case['id']}: worker omitted an actor")
    classifications: dict[str, Any] = {}
    for actor in actors:
        label = actor.get("label")
        if label not in {"reference", "candidate"}:
            raise AuditError(f"{case['id']}: unknown actor label")
        if actor.get("unfinished_cap_leaves"):
            panel_class = "invalid_unfinished_cap_leaf"
        elif case["kind"] == "belief":
            panel_class = "belief_diagnostic_only"
        else:
            panel_class = classify_move(
                case, actor["panel_selected"], admitted, "rollout_panel"
            )
        classifications[label] = {
            "policy": "belief_diagnostic_only" if case["kind"] == "belief"
            else classify_policy(case, actor["root_policy_selected"], admitted),
            "panel": panel_class,
            "deployed": "belief_diagnostic_only" if case["kind"] == "belief"
            else (
                "invalid_unfinished_cap_leaf"
                if actor.get("deployed_unfinished_cap_leaves")
                else classify_move(
                    case, actor["deployed_selected"], admitted,
                    "deployed_actor",
                )
            ),
        }
        if case["kind"] == "belief":
            classifications[label]["focus_cards"] = focus_belief(case, actor)
    history = None
    if case["kind"] == "belief":
        history = run_history_belief(case, args, seed)
        focus = set(case.get("focus_cards", []))
        classifications["history_aware"] = {
            "status": history["status"],
            "focus_cards": [
                row for row in history.get("cards", [])
                if row.get("card") in focus
            ],
        }
    return {
        "id": case["id"],
        "seed": case["seed"],
        "ply": case["ply"],
        "kind": case["kind"],
        "comment": case["comment"],
        "certainty": case["certainty"],
        "preferred": case.get("preferred", []),
        "criticized": case.get("criticized", []),
        "focus_cards": case.get("focus_cards", []),
        "state": case["state"],
        "state_sha256": case["state_sha256"],
        "panel_seed": seed,
        "classifications": classifications,
        "history_aware_belief": history,
        "probe": probe,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument(
        "--history-worker", type=Path, default=DEFAULT_HISTORY_WORKER
    )
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--history-worlds", type=int, default=20000)
    parser.add_argument("--belief-alpha", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=202608231701)
    parser.add_argument("--execution-sha256")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.worlds < 2:
            raise AuditError("--worlds must be at least 2")
        if not 1 <= args.history_worlds <= 10000000:
            raise AuditError("--history-worlds must be in [1,10000000]")
        if not 0.0 <= args.belief_alpha <= 5.0:
            raise AuditError("--belief-alpha must be in [0,5]")
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise AuditError("invalid shard index/count")
        if not 0 <= args.seed < 1 << 64:
            raise AuditError("--seed must fit unsigned 64-bit")
        if args.execution_sha256 is not None and (
            len(args.execution_sha256) != 64
            or any(char not in "0123456789abcdef" for char in args.execution_sha256)
        ):
            raise AuditError("--execution-sha256 must be lowercase SHA-256")
        if not args.probe.is_file():
            raise AuditError(f"probe binary is absent: {args.probe}")
        manifest, manifest_sha = load_manifest(args.manifest)
        requested = set(args.case)
        known = {case["id"] for case in manifest["cases"]}
        if requested - known:
            raise AuditError(
                "unknown --case values: " + ", ".join(sorted(requested - known))
            )
        selected = [
            case for index, case in enumerate(manifest["cases"])
            if index % args.shard_count == args.shard_index
            and (not requested or case["id"] in requested)
        ]
        if not selected:
            raise AuditError("this selection contains no cases")
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for case in selected:
            try:
                results.append(run_case(case, args))
            except AuditError as exc:
                if not args.keep_going:
                    raise
                errors.append({"id": case["id"], "error": str(exc)})
        counts: dict[str, Any] = {}
        for label in ("reference", "candidate"):
            counts[label] = {
                "policy": dict(Counter(
                    result["classifications"][label]["policy"]
                    for result in results
                )),
                "panel": dict(Counter(
                    result["classifications"][label]["panel"]
                    for result in results
                )),
                "deployed": dict(Counter(
                    result["classifications"][label]["deployed"]
                    for result in results
                )),
            }
        try:
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = None
        output = {
            "schema": "lc-flagged-ply-audit-v1",
            "provenance": {
                "source_commit": source_commit,
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest_sha,
                "reference": actor_provenance(args.reference),
                "candidate": actor_provenance(args.candidate),
                "decision_worlds": args.worlds,
                "belief_alpha": args.belief_alpha,
                "history_worlds": args.history_worlds,
                "base_seed": args.seed,
                "execution_sha256": args.execution_sha256,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "candidate_rule": (
                    "union of each actor's top-three complete semantic policy "
                    "moves, capped at five; physical wager IDs are deduped "
                    "but draw sources remain distinct"
                ),
                "world_model": "uniform common hidden worlds",
                "selection": "all selected frozen cases; no result filtering",
            },
            "requested_cases": len(selected),
            "completed_cases": len(results),
            "errors": errors,
            "classification_counts": counts,
            "cases": results,
        }
        rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists() and not args.force:
                raise AuditError(
                    f"output already exists (use --force): {args.output}"
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(f".{args.output.name}.tmp")
            temporary.write_text(rendered, encoding="utf-8")
            temporary.replace(args.output)
        else:
            sys.stdout.write(rendered)
        return 0 if not errors else 1
    except AuditError as exc:
        print(f"flagged_ply_audit.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
