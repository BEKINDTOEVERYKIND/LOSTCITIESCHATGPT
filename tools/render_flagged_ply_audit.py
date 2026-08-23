#!/usr/bin/env python3
"""Render completed flagged-ply JSON evidence as deterministic Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


class RenderError(RuntimeError):
    pass


def esc(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.2f}%"


def actor_by_label(case: dict[str, Any], label: str) -> dict[str, Any]:
    for actor in case["probe"]["actors"]:
        if actor.get("label") == label:
            return actor
    raise RenderError(f"{case.get('id')}: missing {label} actor")


def render_belief(case: dict[str, Any], lines: list[str]) -> None:
    if case.get("kind") != "belief":
        return
    lines.extend(["", "### Belief evidence", ""])
    lines.append(
        "The comparison-actor rows are snapshot-only learned marginals; the "
        "history row is a separate likelihood panel over the frozen public prefix."
    )
    lines.extend([
        "",
        "| Source | Card | Head / posterior | Card-count prior | Head − prior | Rank / interval |",
        "|---|---|---:|---:|---:|---|",
    ])
    classification = case["classifications"]
    for label in ("reference", "candidate"):
        for row in classification[label].get("focus_cards", []):
            expected_count = row.get("metric") == "expected_count"
            estimate = number(row.get("estimate")) if expected_count else percent(
                row.get("probability")
            )
            prior = number(row.get("prior")) if expected_count else percent(
                row.get("prior")
            )
            delta = number(row.get("head_minus_prior")) if expected_count else percent(
                row.get("head_minus_prior")
            )
            lines.append(
                f"| {label} snapshot | {esc(row['card'])} | "
                f"{estimate} | {prior} | {delta} | "
                f"rank {row.get('rank', '—')} |"
            )
    history = case.get("history_aware_belief") or {}
    history_focus = classification.get("history_aware", {}).get("focus_cards", [])
    if history.get("status") == "ok":
        for row in history_focus:
            interval = row.get("ci95", [None, None])
            lines.append(
                f"| historical actor-aware | {esc(row['card'])} | "
                f"{number(row.get('expected_count'))} expected copies | — | — | "
                f"95% CI {number(interval[0])}–{number(interval[1])} |"
            )
    else:
        lines.append(
            f"| historical actor-aware | focus set | — | — | — | "
            f"{esc(history.get('status', 'missing'))}: "
            f"{esc(history.get('note', 'no valid result'))} |"
        )
    provenance = history.get("provenance", {})
    if provenance:
        lines.extend([
            "",
            f"History panel: `{esc(provenance.get('view', '—'))}` "
            f"(`{esc(provenance.get('view_sha256', '—'))}`), "
            f"{history.get('accepted', 0)}/{provenance.get('worlds', '—')} "
            "worlds accepted.",
        ])


def render_case(case: dict[str, Any], lines: list[str]) -> None:
    probe = case["probe"]
    lines.extend([
        "",
        f"## {esc(case['id'])} — seed {case['seed']}, ply {case['ply']}",
        "",
        f"> {esc(case['comment'])}",
        "",
        f"State: `{esc(case['state'])}` (`{esc(case['state_sha256'])}`); "
        f"round {probe['round']}, round ply {probe['round_ply']}, "
        f"deck {probe['deck_left']}.",
    ])
    if case.get("kind") == "belief":
        lines.extend([
            "",
            "No action or rollout-Q panel was run for this belief-only case; "
            "the evidence below is exclusively about the hand-read claim.",
        ])
        render_belief(case, lines)
        return

    lines.extend([
        "",
        f"Evaluated {probe['evaluated_moves']} of {probe['legal_moves']} legal "
        "moves.",
        "",
        "| Actor | Policy argmax | Policy p | Panel selection | Deployed selection | Policy class | Panel class | Deployed class |",
        "|---|---|---:|---|---|---|---|---|",
    ])
    for label in ("reference", "candidate"):
        actor = actor_by_label(case, label)
        classification = case["classifications"][label]
        lines.append(
            f"| {label} | {esc(actor['root_policy_selected'])} | "
            f"{percent(actor.get('root_policy_probability'))} | "
            f"{esc(actor['panel_selected'])} | {esc(actor['deployed_selected'])} | "
            f"{esc(classification['policy'])} | {esc(classification['panel'])} | "
            f"{esc(classification['deployed'])} |"
        )

    reference = actor_by_label(case, "reference")
    candidate = actor_by_label(case, "candidate")
    if len(reference["rows"]) != len(probe["candidates"]) or len(
        candidate["rows"]
    ) != len(probe["candidates"]):
        raise RenderError(f"{case['id']}: candidate/actor row mismatch")
    lines.extend([
        "",
        "Panel objectives: reference "
        f"`{esc(reference.get('objective_label', '—'))}` in "
        f"`{esc(reference.get('objective_units', '—'))}`; candidate "
        f"`{esc(candidate.get('objective_label', '—'))}` in "
        f"`{esc(candidate.get('objective_units', '—'))}`. "
        "Q values are comparable across actors only when both labels and "
        "units match.",
        "",
        "### Admitted policy moves and common-world values",
        "",
        "| Move | Ref top-3 rank | Cand top-3 rank | Ref complete rank / p; action-core rank / mass | Cand complete rank / p; action-core rank / mass | Ref Q ± SE | Ref Δ ± SE | Cand Q ± SE | Cand Δ ± SE |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ])
    for index, admitted in enumerate(probe["candidates"]):
        admission = admitted["admission"]
        rp = admitted["reference_policy"]
        cp = admitted["candidate_policy"]
        rr = reference["rows"][index]
        cr = candidate["rows"][index]
        lines.append(
            f"| {esc(admitted['move'])} | "
            f"{admission.get('reference_top_move_rank', 0) or '—'} | "
            f"{admission.get('candidate_top_move_rank', 0) or '—'} | "
            f"#{rp.get('complete_rank', 0) or '—'} / {percent(rp.get('probability'))}; "
            f"core #{rp.get('core_rank', 0) or '—'} / {percent(rp.get('core_mass'))} | "
            f"#{cp.get('complete_rank', 0) or '—'} / {percent(cp.get('probability'))}; "
            f"core #{cp.get('core_rank', 0) or '—'} / {percent(cp.get('core_mass'))} | "
            f"{number(rr.get('q'))} ± {number(rr.get('se'))} | "
            f"{number(rr.get('delta_vs_policy_baseline'))} ± {number(rr.get('delta_se'))} | "
            f"{number(cr.get('q'))} ± {number(cr.get('se'))} | "
            f"{number(cr.get('delta_vs_policy_baseline'))} ± {number(cr.get('delta_se'))} |"
        )
    lines.extend([
        "",
        f"Panel support: reference {reference['worlds']} worlds "
        f"(exact hidden support: {str(reference['exact_hidden_support']).lower()}), "
        f"candidate {candidate['worlds']} worlds "
        f"(exact hidden support: {str(candidate['exact_hidden_support']).lower()}).",
    ])


def render(result: dict[str, Any]) -> str:
    if result.get("schema") not in {
        "lc-flagged-ply-audit-v1", "lc-flagged-ply-audit-merged-v1"
    }:
        raise RenderError("unsupported audit schema")
    cases = result.get("cases")
    if not isinstance(cases, list):
        raise RenderError("audit has no case list")
    provenance = result.get("provenance", {})
    reference = provenance.get("reference", {})
    candidate = provenance.get("candidate", {})
    def artifacts(actor: dict[str, Any]) -> str:
        rows = list(actor.get("checkpoints", []))
        if actor.get("match_value_table"):
            rows.append(actor["match_value_table"])
        return ", ".join(
            f"`{esc(row.get('path', '—'))}` `{esc(row.get('sha256', '—'))}`"
            for row in rows
        ) or "—"
    lines = [
        "# Frozen user-reviewed ply audit",
        "",
        "This report is a deterministic rendering of completed evidence; it "
        "does not add model-selection conclusions beyond the recorded classifications.",
        "",
        f"Cases: {len(cases)}. Decision worlds: "
        f"{provenance.get('decision_worlds', '—')}. Base seed: "
        f"{provenance.get('base_seed', '—')}. Manifest: "
        f"`{esc(provenance.get('manifest_sha256', '—'))}`.",
        "",
        f"Candidate rule: {esc(provenance.get('candidate_rule', '—'))}.",
        "",
        f"Source commit: `{esc(provenance.get('source_commit', '—'))}`; "
        f"tree: `{esc(provenance.get('source_tree', '—'))}`; execution addendum: "
        f"`{esc(provenance.get('execution_sha256') or 'local')}`.",
        "",
        f"Evaluator manifest: "
        f"`{esc(provenance.get('evaluator_manifest_sha256') or 'local')}`; "
        f"authoritative actor result: "
        f"`{esc(provenance.get('authoritative_result_sha256') or 'local')}`; "
        f"launch mode: `{esc(provenance.get('launch_mode', 'local_unbound'))}`.",
        "",
        f"Reference actor: `{esc(reference.get('spec', '—'))}`. Artifacts: "
        f"{artifacts(reference)}.",
        "",
        f"Candidate actor: `{esc(candidate.get('spec', '—'))}`. Artifacts: "
        f"{artifacts(candidate)}.",
    ]
    for case in cases:
        render_case(case, lines)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        value = json.loads(args.input.read_text(encoding="utf-8"))
        rendered = render(value)
        if args.output.exists() and not args.force:
            raise RenderError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, RenderError) as exc:
        print(f"render_flagged_ply_audit.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
