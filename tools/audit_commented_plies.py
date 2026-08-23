#!/usr/bin/env python3
"""Audit the exact set of plies explicitly reviewed by the user.

This is diagnostic evidence, never a training corpus or a promotion gate.  At
each saved information state it records the supplied actor's actual seeded
choice, then grades the human-nominated alternatives on common hidden worlds
with an exact suit-ensemble policy continuation.  The search actor is not
recursively invoked inside those counterfactual branches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MIN_PAIRED_WORLDS = 1024
DEFAULT_SYMMETRIES = 20
DEFAULT_BELIEF_ALPHA = 1.15


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    source_seed: int
    ply: int
    state: str
    candidates: tuple[str, ...]
    reviewed_moves: tuple[str, ...]
    review: str
    audit_seed: int
    min_worlds: int = MIN_PAIRED_WORLDS
    belief_card: str | None = None


# This tuple is deliberately narrow.  Broader engineering probes in the
# manifest are useful, but they were not all explicit comments from the user
# and therefore do not belong in this audit population.
CASES: tuple[AuditCase, ...] = (
    AuditCase("ui-221-p3", 2214615196, 3,
              "data/probes/ui_seed2214615196_p3.state",
              ("Bx p deck", "Bx p W"), ("Bx p deck",),
              "Low W2 pickup should not rival the deck.", 202608230103),
    AuditCase("ui-221-p4", 2214615196, 4,
              "data/probes/ui_seed2214615196_p4.state",
              ("Bx p deck", "Bx p W"), ("Bx p deck",),
              "Low W2 pickup should not rival the deck.", 202608230104),
    AuditCase("ui-221-p8", 2214615196, 8,
              "data/probes/ui_seed2214615196_p8.state",
              ("B3 p deck", "B3 p W"), ("B3 p deck",),
              "Low W2 pickup should not rival the deck.", 202608230108),
    AuditCase("ui-221-p10", 2214615196, 10,
              "data/probes/ui_seed2214615196_p10.state",
              ("Wx p deck", "Wx p W"), ("Wx p deck",),
              "The W2 pickup was called overrated; prior evidence was "
              "inconclusive and must be reported honestly.", 202608230110,
              min_worlds=2048),
    AuditCase("ui-221-p12", 2214615196, 12,
              "data/probes/ui_seed2214615196_p12.state",
              ("W4 p deck", "W4 p R"), ("W4 p deck",),
              "Prefer the deck to the low R2 pickup.", 202608230112),
    AuditCase("ui-221-p13", 2214615196, 13,
              "data/probes/ui_seed2214615196_p13.state", (), (),
              "Audit the exact-cardinality belief rather than an "
              "independent-card approximation.", 202608230113,
              belief_card="Y9"),
    AuditCase("ui-221-p16", 2214615196, 16,
              "data/probes/ui_seed2214615196_p16.state",
              ("Y2 d deck", "W7 p deck", "Yx d deck"),
              ("Y2 d deck",),
              "Preserve White options instead of committing W7 early.",
              202608230116),
    AuditCase("ui-221-p20", 2214615196, 20,
              "data/probes/ui_seed2214615196_p20.state",
              ("W3 d deck", "W7 p deck", "Wx d deck"),
              ("W3 d deck", "Wx d deck"),
              "W7 was the concern; W3 and the wager discard were both "
              "reviewed, with no forced ordering between close discards.",
              202608230120),
    AuditCase("showcase-572-p14", 5726968372613385, 14,
              "data/probes/showcase_5726968372613385_p14.state",
              ("R4 d deck", "G7 p deck", "B3 d deck"),
              ("G7 p deck", "B3 d deck"),
              "Evaluate both suggested alternatives against the recorded "
              "R4 discard; do not force-rank G7 versus B3.", 202608230214),
    AuditCase("showcase-572-p15", 5726968372613385, 15,
              "data/probes/showcase_5726968372613385_p15.state",
              ("B5 d deck", "W4 d deck"), ("B5 d deck",),
              "Preserve W4 and discard B5.", 202608230215),
    AuditCase("showcase-572-p17", 5726968372613385, 17,
              "data/probes/showcase_5726968372613385_p17.state",
              ("B5 d R", "R8 p R"), ("B5 d R",),
              "Discard B5 and take R4 rather than prematurely play R8.",
              202608230217),
    AuditCase("showcase-572-p32", 5726968372613385, 32,
              "data/probes/showcase_5726968372613385_p32.state",
              ("W10 p deck", "Bx p deck", "R10 p deck"),
              ("W10 p deck", "R10 p deck"),
              "Compare the third Blue wager with safe ten plays.",
              202608230232),
    AuditCase("ui-725-p21", 725402798, 21,
              "data/probes/ui_seed725402798_p21.state",
              ("Bx d deck", "Bx d G", "G5 p deck"),
              ("Bx d deck", "Bx d G"),
              "Avoid overvaluing G5; audit the clean Blue-wager discard "
              "with both relevant draw sources.", 202608230321),
    AuditCase("ui-725-p22", 725402798, 22,
              "data/probes/ui_seed725402798_p22.state",
              ("R2 d deck", "R7 p deck"), ("R2 d deck",),
              "Discard R2 rather than commit R7 early.", 202608230322),
    AuditCase("ui-725-p23", 725402798, 23,
              "data/probes/ui_seed725402798_p23.state",
              ("Wx d deck", "G5 p deck"), ("Wx d deck",),
              "Avoid overvaluing G5 when the White wager can be discarded.",
              202608230323),
    AuditCase("ui-725-p25", 725402798, 25,
              "data/probes/ui_seed725402798_p25.state",
              ("Y4 p B", "Y4 p deck"), ("Y4 p B",),
              "The face-up Blue wager pickup must be evaluated even when it "
              "falls outside the policy prefix.", 202608230325),
    AuditCase("ui-956-p44", 95647345759839, 44,
              "data/probes/ui_seed95647345759839_p44.state",
              ("W10 p deck", "W10 p G"), ("W10 p deck",),
              "End the round through the last deck card instead of gifting "
              "another turn for a Green-wager pickup.", 202608230444),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def definition_sha256(cases: Iterable[AuditCase] = CASES) -> str:
    rows = [
        {
            "case_id": case.case_id,
            "source_seed": str(case.source_seed),
            "ply": case.ply,
            "state": case.state,
            "candidates": list(case.candidates),
            "reviewed_moves": list(case.reviewed_moves),
            "review": case.review,
            "audit_seed": str(case.audit_seed),
            "min_worlds": case.min_worlds,
            "belief_card": case.belief_card,
        }
        for case in cases
    ]
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def descriptive_signal(delta: float, se: float) -> str:
    """Classify a 95% interval descriptively; this is not a pass/fail gate."""
    if not math.isfinite(delta) or not math.isfinite(se) or se < 0:
        return "invalid"
    low, high = delta - 1.96 * se, delta + 1.96 * se
    if low > 0:
        return "alternative_ahead"
    if high < 0:
        return "reference_ahead"
    return "inconclusive"


def _repo_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def _checkpoint_hashes(actor_spec: str, net_path: Path) -> list[dict[str, str]]:
    found: list[tuple[str, Path]] = [(str(net_path), net_path)]
    for token in actor_spec.split(":")[1:]:
        candidate = _repo_path(token)
        if token and candidate.is_file():
            found.append((token, candidate))
    result: list[dict[str, str]] = []
    seen: set[Path] = set()
    for label, path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append({"path": label, "sha256": sha256_file(resolved)})
    return result


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_case(
    case: AuditCase,
    *,
    helper: Path,
    actor_spec: str,
    net_path: Path,
    requested_worlds: int,
    symmetries: int,
    belief_alpha: float,
) -> dict[str, Any]:
    worlds = max(requested_worlds, case.min_worlds)
    state_path = ROOT / case.state
    command = [
        str(helper),
        "--state", case.state,
        "--actor", actor_spec,
        "--net", str(net_path),
        "--seed", str(case.audit_seed),
        "--worlds", str(worlds),
        "--symmetries", str(symmetries),
    ]
    for candidate in case.candidates:
        command.extend(("--candidate", candidate))
    if case.belief_card is not None:
        command.extend((
            "--belief", "--belief-alpha", str(belief_alpha),
            "--belief-card", case.belief_card,
        ))
    result = subprocess.run(
        command, cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    evaluation = json.loads(result.stdout)
    if evaluation.get("schema") != "lc-commented-ply-eval-v1":
        raise RuntimeError(f"{case.case_id}: unexpected helper schema")
    counterfactual = evaluation["counterfactual"]
    if counterfactual["worlds"] < case.min_worlds and case.candidates:
        raise RuntimeError(f"{case.case_id}: insufficient paired worlds")
    continuation = counterfactual["continuation"]
    if (
        continuation.get("kind") != "argmax_policy"
        or continuation.get("symmetries") != symmetries
        or not continuation.get("exact_group_average")
        or continuation.get("recursive_actor") is not False
    ):
        raise RuntimeError(f"{case.case_id}: invalid continuation contract")
    selected = evaluation["actor"]["selected"]
    expected_candidates = list(case.candidates)
    selected_is_extra = bool(case.candidates) and selected not in expected_candidates
    if selected_is_extra:
        expected_candidates.append(selected)
    actual_candidates = [row["move"] for row in counterfactual["candidates"]]
    if actual_candidates != expected_candidates:
        raise RuntimeError(
            f"{case.case_id}: candidate drift {actual_candidates!r}"
        )
    if case.candidates and not counterfactual.get("actor_selected_included"):
        raise RuntimeError(f"{case.case_id}: actor selection was not graded")
    for index, row in enumerate(counterfactual["candidates"]):
        row["role"] = (
            "reference" if index == 0 else
            "actor_selected_extra"
            if selected_is_extra and row["move"] == selected else
            "alternative"
        )
        row["descriptive_signal"] = (
            "reference" if index == 0 else descriptive_signal(
                row["delta_vs_reference"], row["delta_se"]
            )
        )
    if case.belief_card is not None:
        belief = evaluation.get("belief")
        if not belief or not belief.get("valid") or belief.get("kind") != "fixed_k":
            raise RuntimeError(f"{case.case_id}: invalid fixed-K belief")
        if abs(belief["marginal_sum"] - belief["need"]) > 2e-4:
            raise RuntimeError(f"{case.case_id}: belief marginals violate fixed K")

    return {
        "case_id": case.case_id,
        "source": {
            "game_seed": str(case.source_seed),
            "displayed_ply": case.ply,
            "state_path": case.state,
            "state_sha256": sha256_file(state_path),
        },
        "review_context": {
            "note": case.review,
            "reviewed_moves": list(case.reviewed_moves),
            "used_as_training_label": False,
            "used_as_promotion_gate": False,
        },
        "evaluation": evaluation,
    }


def build_audit(
    *,
    helper: Path,
    actor_spec: str,
    net_path: Path,
    worlds: int = MIN_PAIRED_WORLDS,
    symmetries: int = DEFAULT_SYMMETRIES,
    belief_alpha: float = DEFAULT_BELIEF_ALPHA,
) -> dict[str, Any]:
    if worlds < MIN_PAIRED_WORLDS:
        raise ValueError(f"worlds must be at least {MIN_PAIRED_WORLDS}")
    if symmetries not in (1, 5, 10, 20, 120):
        raise ValueError("invalid exact suit group")
    if not helper.is_file():
        raise FileNotFoundError(helper)
    if not net_path.is_file():
        raise FileNotFoundError(net_path)
    cases = [
        run_case(
            case, helper=helper, actor_spec=actor_spec,
            net_path=net_path, requested_worlds=worlds,
            symmetries=symmetries, belief_alpha=belief_alpha,
        )
        for case in CASES
    ]
    action_cases = [case for case in cases if case["evaluation"]["counterfactual"]["candidates"]]
    candidate_rows = [
        row
        for case in action_cases
        for row in case["evaluation"]["counterfactual"]["candidates"]
        if row["role"] != "actor_selected_extra"
    ]
    alternative_rows = [
        row
        for case in action_cases
        for row in case["evaluation"]["counterfactual"]["candidates"][1:]
        if row["role"] == "alternative"
    ]
    actor_extra_rows = [
        row
        for case in action_cases
        for row in case["evaluation"]["counterfactual"]["candidates"][1:]
        if row["role"] == "actor_selected_extra"
    ]
    actor_aligned = sum(
        case["evaluation"]["actor"]["selected"]
        in case["review_context"]["reviewed_moves"]
        for case in cases if case["review_context"]["reviewed_moves"]
    )
    cap_hits = sum(
        row["cap_hits"]
        for case in action_cases
        for row in case["evaluation"]["counterfactual"]["candidates"]
    )
    return {
        "schema": "lc-commented-ply-audit-v1",
        "audit_definition_sha256": definition_sha256(),
        "repository_head": _git_head(),
        "subject": {
            "actor_spec": actor_spec,
            "actor_spec_sha256": hashlib.sha256(actor_spec.encode()).hexdigest(),
            "evaluation_net_path": str(net_path),
            "checkpoints": _checkpoint_hashes(actor_spec, net_path),
        },
        "implementation": {
            "driver_path": "tools/audit_commented_plies.py",
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "helper_path": str(helper),
            "helper_sha256": sha256_file(helper),
        },
        "contract": {
            "population": "17 explicit user-commented displayed plies only",
            "case_count": 17,
            "paired_worlds_minimum": MIN_PAIRED_WORLDS,
            "requested_worlds": worlds,
            "exact_policy_symmetries": symmetries,
            "world_model": "uniform_exact_card_count",
            "decision_input": "agent_information_view",
            "complete_hidden_state_use": "p13 offline truth label only",
            "recursive_rollout_actor_in_counterfactuals": False,
            "actor_selected_action_graded_for_action_cases": True,
            "training_use": "forbidden_diagnostic_only",
            "promotion_gate": False,
            "locked_validation_criteria_changed": False,
        },
        "summary": {
            "action_cases": len(action_cases),
            "belief_cases": len(cases) - len(action_cases),
            "actor_selected_reviewed_move": actor_aligned,
            "actor_reviewed_move_denominator": sum(
                bool(case["review_context"]["reviewed_moves"])
                for case in cases
            ),
            "alternative_signals": {
                signal: sum(row["descriptive_signal"] == signal
                            for row in alternative_rows)
                for signal in (
                    "reference_ahead", "alternative_ahead", "inconclusive",
                    "invalid",
                )
            },
            "actor_selected_extra_signals": {
                signal: sum(row["descriptive_signal"] == signal
                            for row in actor_extra_rows)
                for signal in (
                    "reference_ahead", "alternative_ahead", "inconclusive",
                    "invalid",
                )
            },
            "reviewed_candidates_below_two_percent_policy_prior": sum(
                row["policy_prior"] < 0.02 for row in candidate_rows
            ),
            "counterfactual_cap_hits": cap_hits,
        },
        "cases": cases,
    }


def _fmt_delta(row: dict[str, Any], reference: str) -> str:
    low = row["delta_vs_reference"] - 1.96 * row["delta_se"]
    high = row["delta_vs_reference"] + 1.96 * row["delta_se"]
    return (
        f"{row['move']} {row['delta_vs_reference']:+.2f}±"
        f"{row['delta_se']:.2f} vs {reference} "
        f"(95% [{low:+.2f}, {high:+.2f}]; "
        f"{row['descriptive_signal'].replace('_', ' ')})"
    )


def render_markdown(audit: dict[str, Any]) -> str:
    subject = audit["subject"]
    contract = audit["contract"]
    summary = audit["summary"]
    lines = [
        "# Explicit commented-ply audit",
        "",
        f"Actor: `{subject['actor_spec']}`  ",
        f"Evaluation network: `{subject['evaluation_net_path']}`  ",
        f"Evidence: {contract['case_count']} explicit displayed plies; at "
        f"least {contract['paired_worlds_minimum']} paired hidden worlds per "
        f"disputed action; exact {contract['exact_policy_symmetries']}-way "
        "suit-ensemble policy continuations.",
        "",
        "> This is diagnostic evidence only. The reviewed moves are excluded "
        "from training and are neither safety gates nor promotion gates. The "
        "locked validation criteria are unchanged.",
        "",
        "## Per-ply evidence",
        "",
        "| Ply | Actor's actual move | Reviewed context | Paired evidence |",
        "|---|---|---|---|",
    ]
    for case in audit["cases"]:
        evaluation = case["evaluation"]
        actor_move = evaluation["actor"]["selected"]
        source = case["source"]
        label = f"{source['game_seed']} / {source['displayed_ply']}"
        note = case["review_context"]["note"].replace("|", "\\|")
        candidates = evaluation["counterfactual"]["candidates"]
        if candidates:
            evidence = "; ".join(
                _fmt_delta(row, candidates[0]["move"])
                for row in candidates[1:]
            )
        else:
            belief = evaluation["belief"]
            target = belief["target"]
            top = belief["cards"][0]
            evidence = (
                f"fixed-K {target['card']}={100*target['marginal']:.2f}% "
                f"vs {100*belief['uniform_marginal']:.2f}% prior; "
                f"top {top['card']}={100*top['marginal']:.2f}% "
                f"({'held' if top['held'] else 'not held'}); "
                f"marginal sum={belief['marginal_sum']:.6f} for K={belief['need']}"
            )
        lines.append(f"| {label} | `{actor_move}` | {note} | {evidence} |")

    signals = summary["alternative_signals"]
    actor_extra_signals = summary["actor_selected_extra_signals"]
    lines.extend([
        "",
        "## Aggregate diagnostic signals",
        "",
        f"- The actor selected a reviewed move on "
        f"{summary['actor_selected_reviewed_move']}/"
        f"{summary['actor_reviewed_move_denominator']} action-review plies. "
        "This is descriptive alignment, not an accuracy score.",
        "",
        f"- Among reference-relative alternatives, the 95% intervals show "
        f"{signals['reference_ahead']} reference-ahead, "
        f"{signals['alternative_ahead']} alternative-ahead, and "
        f"{signals['inconclusive']} inconclusive comparisons.",
        "",
        f"- Actor-selected moves outside the nominated support were also "
        f"graded: {actor_extra_signals['reference_ahead']} reference-ahead, "
        f"{actor_extra_signals['alternative_ahead']} actor-move-ahead, and "
        f"{actor_extra_signals['inconclusive']} inconclusive comparisons.",
        "",
        f"- {summary['reviewed_candidates_below_two_percent_policy_prior']} "
        "reviewed candidates had policy prior below 2%; they were still evaluated "
        "because this audit is not restricted to the deployed shortlist.",
        "",
        f"- Counterfactual continuations hit the artificial ply cap "
        f"{summary['counterfactual_cap_hits']} times.",
        "",
        "## General-improvement implications",
        "",
        "- Learn signed paired action advantages from independent natural "
        "states, retaining wins, losses, and ties. Keep these reviewed plies "
        "as a held-out diagnostic so improvements must generalize.",
        "",
        "- Model card/action and draw-source value jointly. The repeated low-"
        "pile, face-up-wager, and final-deck cases are one interaction family, "
        "not a list of moves to hard-code.",
        "",
        "- Improve option-value and commitment timing across suits (early "
        "sevens, G5, the third wager) through broader counterfactual data, not "
        "per-position patches.",
        "",
        "- Keep action quality and belief calibration separate. The p13 "
        "posterior is an exact fixed-cardinality diagnostic whose marginals "
        "must sum to the opponent's unknown hand size.",
        "",
        "## Provenance",
        "",
        f"- Audit definition SHA-256: `{audit['audit_definition_sha256']}`",
        f"- Actor spec SHA-256: `{subject['actor_spec_sha256']}`",
        f"- Repository HEAD: `{audit.get('repository_head')}`",
    ])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", help="agent spec to ask for the actual move")
    parser.add_argument("--net", default="data/champion.bin",
                        help="network for exact policy continuations/belief")
    parser.add_argument("--helper", default="bin/commented_ply_eval")
    parser.add_argument("--worlds", type=int, default=MIN_PAIRED_WORLDS)
    parser.add_argument("--symmetries", type=int, default=DEFAULT_SYMMETRIES)
    parser.add_argument("--belief-alpha", type=float,
                        default=DEFAULT_BELIEF_ALPHA)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--from-json", type=Path,
                        help="only regenerate Markdown from canonical JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.from_json:
        audit = json.loads(args.from_json.read_text())
    else:
        if not args.actor:
            raise SystemExit("--actor is required unless --from-json is used")
        audit = build_audit(
            helper=_repo_path(args.helper), actor_spec=args.actor,
            net_path=_repo_path(args.net), worlds=args.worlds,
            symmetries=args.symmetries, belief_alpha=args.belief_alpha,
        )
    canonical = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    markdown = render_markdown(audit)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(canonical)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown)
    if not args.output_json and not args.output_md:
        sys.stdout.write(canonical)
    elif args.output_md and not args.output_json:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
