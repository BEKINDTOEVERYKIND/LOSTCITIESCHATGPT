#!/usr/bin/env python3
"""Audit the exact set of plies explicitly reviewed by the user.

This is diagnostic evidence, never a training corpus or a promotion gate.  At
each saved information state it records the supplied actor's actual seeded
choice, then grades the human-nominated alternatives on common hidden worlds
and future deals with an exact policy-20 continuation through the complete
remaining three-round match.  The search actor is not recursively invoked
inside those counterfactual branches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MIN_PAIRED_WORLDS = 1024
DEFAULT_SYMMETRIES = 20
DEFAULT_BELIEF_ALPHA = 1.15
EVAL_SCHEMA = "lc-commented-ply-eval-v2"
AUDIT_SCHEMA = "lc-commented-ply-audit-v2"
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")


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
            "state_sha256": sha256_file(ROOT / case.state),
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


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _checkpoint_hashes(actor_spec: str, net_path: Path) -> list[dict[str, str]]:
    found: list[tuple[str, Path]] = [(_display_path(net_path), net_path)]
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


def _locked_worlds(case: AuditCase) -> int:
    return case.min_worlds


def _validate_locked_options(
    worlds: int, symmetries: int, belief_alpha: float
) -> None:
    if worlds != MIN_PAIRED_WORLDS:
        raise ValueError(
            f"worlds must equal the locked base budget {MIN_PAIRED_WORLDS}"
        )
    if symmetries != DEFAULT_SYMMETRIES:
        raise ValueError(
            f"symmetries must equal locked policy-{DEFAULT_SYMMETRIES}"
        )
    if not math.isfinite(belief_alpha) or not math.isclose(
        belief_alpha, DEFAULT_BELIEF_ALPHA, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(
            f"belief alpha must equal locked value {DEFAULT_BELIEF_ALPHA}"
        )


def _validate_evaluation(
    case: AuditCase,
    evaluation: dict[str, Any],
    *,
    actor_spec: str,
    net_label: str,
    worlds: int,
    symmetries: int,
    belief_alpha: float,
) -> None:
    if evaluation.get("schema") != EVAL_SCHEMA:
        raise RuntimeError(f"{case.case_id}: unexpected helper schema")
    state = evaluation.get("state")
    if not isinstance(state, dict) or (
        state.get("path") != case.state
        or state.get("nply") != case.ply - 1
        or state.get("input_deck_entries") != 0
    ):
        raise RuntimeError(f"{case.case_id}: saved-state binding drift")
    actor = evaluation.get("actor")
    if not isinstance(actor, dict) or (
        actor.get("spec") != actor_spec
        or actor.get("information_view") is not True
        or not isinstance(actor.get("selected"), str)
    ):
        raise RuntimeError(f"{case.case_id}: actor binding drift")

    counterfactual = evaluation.get("counterfactual")
    if not isinstance(counterfactual, dict) or (
        counterfactual.get("seed") != str(case.audit_seed)
        or counterfactual.get("requested_worlds") != worlds
        or counterfactual.get("completed_worlds") != worlds
        or counterfactual.get("cap_hits") != 0
        or counterfactual.get("world_model")
        != "uniform_exact_card_count_plus_future_deals"
        or counterfactual.get("hash_algorithm") != "fnv1a64"
        or counterfactual.get("policy_probability_aggregation")
        != "sum_by_semantic_move"
    ):
        raise RuntimeError(f"{case.case_id}: incomplete paired worlds")
    for key in (
        "shared_current_hidden_worlds", "shared_future_deals",
        "branch_neutral_rng_domains", "root_information_view",
        "actor_selected_included",
    ):
        if counterfactual.get(key) is not True:
            raise RuntimeError(f"{case.case_id}: false {key} contract")
    for key in (
        "hidden_world_set_hash", "future_deal_set_hash",
        "branch_rng_domain_hash",
    ):
        if not isinstance(counterfactual.get(key), str) or not _HEX16.fullmatch(
            counterfactual[key]
        ):
            raise RuntimeError(f"{case.case_id}: invalid {key}")
    continuation = counterfactual.get("continuation")
    if not isinstance(continuation, dict) or (
        continuation.get("kind") != "exact_policy_argmax"
        or continuation.get("scope") != "full_remaining_three_round_match"
        or continuation.get("checkpoint") != net_label
        or continuation.get("temperature") != 0
        or continuation.get("epsilon") != 0
        or continuation.get("symmetries") != symmetries
        or continuation.get("exact_group_average") is not True
        or continuation.get("fresh_information_view_each_node") is not True
        or continuation.get("recursive_actor") is not False
    ):
        raise RuntimeError(f"{case.case_id}: invalid continuation contract")

    selected = actor["selected"]
    rows = counterfactual.get("candidates")
    if not isinstance(rows, list) or len(rows) < 2:
        raise RuntimeError(f"{case.case_id}: action panel is not paired")
    actual_candidates = [row.get("move") for row in rows]
    semantic_keys = [row.get("semantic_key") for row in rows]
    if (
        any(not isinstance(move, str) for move in actual_candidates)
        or any(not isinstance(key, int) for key in semantic_keys)
        or len(set(actual_candidates)) != len(actual_candidates)
        or len(set(semantic_keys)) != len(semantic_keys)
    ):
        raise RuntimeError(f"{case.case_id}: duplicate semantic candidate")
    selected_is_extra: bool
    if case.candidates:
        if actual_candidates[:len(case.candidates)] != list(case.candidates):
            raise RuntimeError(
                f"{case.case_id}: nominated candidate drift "
                f"{actual_candidates!r}"
            )
        expected_tail = [] if selected in case.candidates else [selected]
        if actual_candidates[len(case.candidates):] != expected_tail:
            raise RuntimeError(f"{case.case_id}: actor-selected tail drift")
        if counterfactual.get("nominated_candidates") != len(case.candidates):
            raise RuntimeError(f"{case.case_id}: nominated count drift")
        if counterfactual.get("policy_reference_candidates") != 0:
            raise RuntimeError(f"{case.case_id}: unexpected policy references")
        selected_is_extra = bool(expected_tail)
    else:
        if (
            counterfactual.get("nominated_candidates") != 0
            or counterfactual.get("policy_reference_candidates") != 2
        ):
            raise RuntimeError(f"{case.case_id}: neutral pair drift")
        if len(actual_candidates) not in (2, 3) or selected not in actual_candidates:
            raise RuntimeError(f"{case.case_id}: actor absent from neutral pair")
        if len(actual_candidates) == 3 and actual_candidates[-1] != selected:
            raise RuntimeError(f"{case.case_id}: neutral actor tail drift")
        selected_is_extra = len(actual_candidates) == 3

    if counterfactual.get("actor_selected_appended") is not selected_is_extra:
        raise RuntimeError(f"{case.case_id}: actor append accounting drift")
    for index, row in enumerate(rows):
        if (
            row.get("completed_worlds") != worlds
            or row.get("cap_hits") != 0
            or not isinstance(row.get("policy_prior"), (int, float))
            or not math.isfinite(row["policy_prior"])
            or not 0 <= row["policy_prior"] <= 1
        ):
            raise RuntimeError(f"{case.case_id}: candidate world drift")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != {
            "match_score", "final_margin", "hybrid"
        }:
            raise RuntimeError(f"{case.case_id}: incomplete full-match metrics")
        for metric in metrics.values():
            if (
                not isinstance(metric, dict)
                or set(metric) != {
                    "mean", "se", "delta_vs_reference", "delta_se",
                    "samples_fnv1a64",
                }
                or any(
                    not isinstance(metric[key], (int, float))
                    or not math.isfinite(metric[key])
                    for key in (
                        "mean", "se", "delta_vs_reference", "delta_se"
                    )
                )
                or metric["se"] < 0
                or metric["delta_se"] < 0
                or not isinstance(metric["samples_fnv1a64"], str)
                or not _HEX16.fullmatch(metric["samples_fnv1a64"])
            ):
                raise RuntimeError(f"{case.case_id}: invalid metric")
        score = metrics["match_score"]
        margin = metrics["final_margin"]
        hybrid = metrics["hybrid"]
        if not -1e-12 <= score["mean"] <= 1 + 1e-12:
            raise RuntimeError(f"{case.case_id}: invalid match score")
        if not math.isclose(
            hybrid["mean"],
            0.05 * margin["mean"] + 100.0 * (score["mean"] - 0.5),
            rel_tol=1e-10,
            abs_tol=1e-9,
        ) or not math.isclose(
            hybrid["delta_vs_reference"],
            0.05 * margin["delta_vs_reference"]
            + 100.0 * score["delta_vs_reference"],
            rel_tol=1e-10,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"{case.case_id}: inconsistent hybrid metric")
        if index == 0 and any(
            metrics[name][key] != 0
            for name in metrics
            for key in ("delta_vs_reference", "delta_se")
        ):
            raise RuntimeError(f"{case.case_id}: reference delta is nonzero")
        if index == 0:
            role = "reference"
        elif selected_is_extra and row["move"] == selected:
            role = "actor_selected_extra"
        elif case.candidates:
            role = "alternative"
        else:
            role = "neutral_policy_alternative"
        row["role"] = role
        row["descriptive_signal"] = (
            "reference" if index == 0 else descriptive_signal(
                hybrid["delta_vs_reference"], hybrid["delta_se"]
            )
        )

    belief = evaluation.get("belief")
    if case.belief_card is None:
        if belief is not None:
            raise RuntimeError(f"{case.case_id}: unexpected belief overlay")
        return
    if not isinstance(belief, dict) or (
        belief.get("valid") is not True
        or belief.get("kind") != "fixed_k"
        or belief.get("information_view") is not True
        or belief.get("complete_state_used_only_as_truth_label") is not True
        or belief.get("symmetries") != symmetries
        or not math.isclose(
            belief.get("alpha", math.nan), belief_alpha,
            rel_tol=0, abs_tol=1e-6,
        )
        or not isinstance(belief.get("n"), int)
        or not isinstance(belief.get("need"), int)
        or belief["n"] <= 0
        or not 0 <= belief["need"] <= belief["n"]
        or not math.isclose(
            belief.get("marginal_sum", math.nan), belief["need"],
            rel_tol=0, abs_tol=2e-4,
        )
    ):
        raise RuntimeError(f"{case.case_id}: invalid fixed-K belief")
    target = belief.get("target")
    cards = belief.get("cards")
    if (
        not isinstance(target, dict)
        or target.get("card") != case.belief_card
        or not isinstance(target.get("marginal"), (int, float))
        or not 0 <= target["marginal"] <= 1
        or not isinstance(target.get("held"), bool)
        or not isinstance(cards, list)
        or len(cards) != belief["n"]
    ):
        raise RuntimeError(f"{case.case_id}: invalid belief target")


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
    net_label = _display_path(net_path)
    command = [
        str(helper),
        "--state", case.state,
        "--actor", actor_spec,
        "--net", net_label,
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
    _validate_evaluation(
        case, evaluation, actor_spec=actor_spec, net_label=net_label,
        worlds=worlds, symmetries=symmetries, belief_alpha=belief_alpha,
    )

    return {
        "case_id": case.case_id,
        "source": {
            "game_seed": str(case.source_seed),
            "displayed_ply": case.ply,
            "state_path": case.state,
            "state_sha256": sha256_file(state_path),
            "audit_seed": str(case.audit_seed),
            "paired_worlds": worlds,
        },
        "review_context": {
            "note": case.review,
            "reviewed_moves": list(case.reviewed_moves),
            "used_as_training_label": False,
            "used_as_promotion_gate": False,
            "nominated_candidates": list(case.candidates),
        },
        "evaluation": evaluation,
    }


def _contract() -> dict[str, Any]:
    return {
        "population": "exactly 17 explicit user-commented displayed plies",
        "case_count": len(CASES),
        "base_paired_worlds": MIN_PAIRED_WORLDS,
        "paired_worlds_by_case": {
            case.case_id: _locked_worlds(case) for case in CASES
        },
        "exact_policy_teacher": "policy:data/champion.bin:0:20",
        "exact_policy_symmetries": DEFAULT_SYMMETRIES,
        "continuation_scope": "full_remaining_three_round_match",
        "world_model": "uniform_exact_card_count_plus_future_deals",
        "current_hidden_worlds_shared_across_actions": True,
        "future_deals_shared_across_actions": True,
        "branch_rng_domains_neutral": True,
        "decision_input": "fresh_agent_information_view_at_every_node",
        "complete_hidden_state_use": "p13 offline truth label only",
        "p13_action_panel": "exact_policy20_top_two_plus_actor_if_absent",
        "p13_belief_overlay": "fixed_k_alpha_1.15_target_Y9",
        "semantic_wager_deduplication": True,
        "policy_probability_aggregation": "sum_by_semantic_move",
        "actor_selected_action_graded_on_all_17": True,
        "counterfactual_caps": "zero_fail_closed",
        "hash_algorithm": "fnv1a64",
        "training_use": "forbidden_diagnostic_only",
        "promotion_gate": False,
        "locked_validation_criteria_changed": False,
    }


def _signal_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    materialized = list(rows)
    return {
        signal: sum(row["descriptive_signal"] == signal
                    for row in materialized)
        for signal in (
            "reference_ahead", "alternative_ahead", "inconclusive", "invalid"
        )
    }


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    alternative_rows: list[dict[str, Any]] = []
    actor_extra_rows: list[dict[str, Any]] = []
    neutral_rows: list[dict[str, Any]] = []
    nominated_rows: list[dict[str, Any]] = []
    cap_hits = 0
    actor_aligned = 0
    actor_denominator = 0
    belief_overlays = 0
    for record in cases:
        definition = next(
            case for case in CASES if case.case_id == record["case_id"]
        )
        rows = record["evaluation"]["counterfactual"]["candidates"]
        nominated_rows.extend(rows[:len(definition.candidates)])
        alternative_rows.extend(
            row for row in rows if row["role"] == "alternative"
        )
        actor_extra_rows.extend(
            row for row in rows if row["role"] == "actor_selected_extra"
        )
        neutral_rows.extend(
            row for row in rows if row["role"] == "neutral_policy_alternative"
        )
        cap_hits += record["evaluation"]["counterfactual"]["cap_hits"]
        cap_hits += sum(row["cap_hits"] for row in rows)
        reviewed = record["review_context"]["reviewed_moves"]
        if reviewed:
            actor_denominator += 1
            actor_aligned += record["evaluation"]["actor"]["selected"] in reviewed
        belief_overlays += record["evaluation"].get("belief") is not None
    return {
        "completed_cases": len(cases),
        "action_panels": len(cases),
        "belief_overlays": belief_overlays,
        "total_paired_worlds": sum(
            record["source"]["paired_worlds"] for record in cases
        ),
        "actor_selected_reviewed_move": actor_aligned,
        "actor_reviewed_move_denominator": actor_denominator,
        "alternative_signals": _signal_counts(alternative_rows),
        "actor_selected_extra_signals": _signal_counts(actor_extra_rows),
        "p13_neutral_alternative_signals": _signal_counts(neutral_rows),
        "reviewed_candidates_below_two_percent_policy_prior": sum(
            row["policy_prior"] < 0.02 for row in nominated_rows
        ),
        "counterfactual_cap_hits": cap_hits,
    }


def _envelope(
    *, helper: Path, actor_spec: str, net_path: Path,
    repository_head: str, cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": AUDIT_SCHEMA,
        "audit_definition_sha256": definition_sha256(),
        "repository_head": repository_head,
        "subject": {
            "actor_spec": actor_spec,
            "actor_spec_sha256": hashlib.sha256(actor_spec.encode()).hexdigest(),
            "evaluation_net_path": _display_path(net_path),
            "checkpoints": _checkpoint_hashes(actor_spec, net_path),
        },
        "implementation": {
            "driver_path": "tools/audit_commented_plies.py",
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "helper_path": _display_path(helper),
            "helper_sha256": sha256_file(helper),
        },
        "contract": _contract(),
        "selection": {"case_ids": [case["case_id"] for case in cases]},
        "summary": _summarize(cases),
        "cases": cases,
    }


def build_audit(
    *,
    helper: Path,
    actor_spec: str,
    net_path: Path,
    worlds: int = MIN_PAIRED_WORLDS,
    symmetries: int = DEFAULT_SYMMETRIES,
    belief_alpha: float = DEFAULT_BELIEF_ALPHA,
    selected_cases: Iterable[AuditCase] = CASES,
    source_commit: str | None = None,
) -> dict[str, Any]:
    _validate_locked_options(worlds, symmetries, belief_alpha)
    if not helper.is_file():
        raise FileNotFoundError(helper)
    if not net_path.is_file():
        raise FileNotFoundError(net_path)
    if _display_path(net_path) != "data/champion.bin":
        raise ValueError("locked continuation net must be data/champion.bin")
    chosen = tuple(selected_cases)
    if not chosen:
        raise ValueError("at least one exact audit case is required")
    by_id = {case.case_id: case for case in CASES}
    if len({case.case_id for case in chosen}) != len(chosen) or any(
        by_id.get(case.case_id) != case for case in chosen
    ):
        raise ValueError("selected cases must be unique frozen audit definitions")
    chosen = tuple(case for case in CASES if case.case_id in {
        selected.case_id for selected in chosen
    })
    head = _git_head()
    repository_head = source_commit or head
    if not isinstance(repository_head, str) or not re.fullmatch(
        r"[0-9a-f]{40}", repository_head
    ):
        raise ValueError("source commit must be a full lowercase Git SHA")
    # The locked workflow executes immutable, SHA-256-sealed source archives
    # without a .git directory.  Compare when repository metadata is present;
    # otherwise retain the workflow-supplied source-parent binding verbatim.
    if source_commit is not None and head is not None and head != source_commit:
        raise ValueError("source commit does not match checked-out HEAD")
    records = [
        run_case(
            case, helper=helper, actor_spec=actor_spec,
            net_path=net_path, requested_worlds=worlds,
            symmetries=symmetries, belief_alpha=belief_alpha,
        )
        for case in chosen
    ]
    return _envelope(
        helper=helper, actor_spec=actor_spec, net_path=net_path,
        repository_head=repository_head, cases=records,
    )


def _validate_case_record(
    record: dict[str, Any], *, actor_spec: str, net_label: str
) -> AuditCase:
    case_id = record.get("case_id")
    definition = next((case for case in CASES if case.case_id == case_id), None)
    if definition is None:
        raise RuntimeError(f"unknown audit case {case_id!r}")
    worlds = _locked_worlds(definition)
    expected_source = {
        "game_seed": str(definition.source_seed),
        "displayed_ply": definition.ply,
        "state_path": definition.state,
        "state_sha256": sha256_file(ROOT / definition.state),
        "audit_seed": str(definition.audit_seed),
        "paired_worlds": worlds,
    }
    expected_review = {
        "note": definition.review,
        "reviewed_moves": list(definition.reviewed_moves),
        "used_as_training_label": False,
        "used_as_promotion_gate": False,
        "nominated_candidates": list(definition.candidates),
    }
    if record.get("source") != expected_source:
        raise RuntimeError(f"{case_id}: source/state binding drift")
    if record.get("review_context") != expected_review:
        raise RuntimeError(f"{case_id}: frozen review-context drift")
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError(f"{case_id}: missing evaluation")
    _validate_evaluation(
        definition, evaluation, actor_spec=actor_spec, net_label=net_label,
        worlds=worlds, symmetries=DEFAULT_SYMMETRIES,
        belief_alpha=DEFAULT_BELIEF_ALPHA,
    )
    return definition


def validate_audit_document(
    audit: dict[str, Any], *, require_full: bool
) -> None:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise RuntimeError("unexpected audit schema")
    if audit.get("audit_definition_sha256") != definition_sha256():
        raise RuntimeError("audit definition/state hash drift")
    if audit.get("contract") != _contract():
        raise RuntimeError("locked audit contract drift")
    head = audit.get("repository_head")
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise RuntimeError("invalid repository-head binding")
    subject = audit.get("subject")
    implementation = audit.get("implementation")
    if not isinstance(subject, dict) or not isinstance(implementation, dict):
        raise RuntimeError("missing provenance")
    actor_spec = subject.get("actor_spec")
    if not isinstance(actor_spec, str) or subject != {
        "actor_spec": actor_spec,
        "actor_spec_sha256": hashlib.sha256(actor_spec.encode()).hexdigest(),
        "evaluation_net_path": "data/champion.bin",
        "checkpoints": _checkpoint_hashes(actor_spec, ROOT / "data/champion.bin"),
    }:
        raise RuntimeError("actor/checkpoint binding drift")
    helper = ROOT / "bin/commented_ply_eval"
    if implementation != {
        "driver_path": "tools/audit_commented_plies.py",
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "helper_path": "bin/commented_ply_eval",
        "helper_sha256": sha256_file(helper),
    }:
        raise RuntimeError("audit implementation binding drift")
    records = audit.get("cases")
    if not isinstance(records, list) or not records:
        raise RuntimeError("audit contains no cases")
    definitions = [
        _validate_case_record(
            record, actor_spec=actor_spec, net_label="data/champion.bin"
        )
        for record in records
        if isinstance(record, dict)
    ]
    if len(definitions) != len(records):
        raise RuntimeError("invalid case record")
    case_ids = [case.case_id for case in definitions]
    expected_order = [case.case_id for case in CASES if case.case_id in case_ids]
    if len(set(case_ids)) != len(case_ids) or case_ids != expected_order:
        raise RuntimeError("audit cases are duplicated or out of frozen order")
    if audit.get("selection") != {"case_ids": case_ids}:
        raise RuntimeError("audit selection drift")
    if audit.get("summary") != _summarize(records):
        raise RuntimeError("audit summary drift")
    if require_full and case_ids != [case.case_id for case in CASES]:
        raise RuntimeError("merged audit does not contain exactly all 17 cases")


def merge_audits(fragments: Iterable[dict[str, Any]]) -> dict[str, Any]:
    documents = list(fragments)
    if not documents:
        raise ValueError("at least one audit fragment is required")
    for document in documents:
        validate_audit_document(document, require_full=False)
    first = documents[0]
    static_keys = (
        "schema", "audit_definition_sha256", "repository_head", "subject",
        "implementation", "contract",
    )
    for document in documents[1:]:
        for key in static_keys:
            if document[key] != first[key]:
                raise RuntimeError(f"audit fragments disagree on {key}")
    records_by_id: dict[str, dict[str, Any]] = {}
    for document in documents:
        for record in document["cases"]:
            case_id = record["case_id"]
            if case_id in records_by_id:
                raise RuntimeError(f"duplicate audit case {case_id}")
            records_by_id[case_id] = record
    expected_ids = [case.case_id for case in CASES]
    if set(records_by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(records_by_id))
        extra = sorted(set(records_by_id) - set(expected_ids))
        raise RuntimeError(
            f"merge requires exact-17 population; missing={missing}, extra={extra}"
        )
    records = [records_by_id[case_id] for case_id in expected_ids]
    merged = {
        key: first[key] for key in static_keys
    }
    merged.update({
        "selection": {"case_ids": expected_ids},
        "summary": _summarize(records),
        "cases": records,
    })
    validate_audit_document(merged, require_full=True)
    return merged


def _fmt_delta(row: dict[str, Any], reference: str) -> str:
    score = row["metrics"]["match_score"]
    margin = row["metrics"]["final_margin"]
    hybrid = row["metrics"]["hybrid"]
    low = 100 * (score["delta_vs_reference"] - 1.96 * score["delta_se"])
    high = 100 * (score["delta_vs_reference"] + 1.96 * score["delta_se"])
    return (
        f"{row['move']} vs {reference}: match-score "
        f"{100*score['delta_vs_reference']:+.2f}±"
        f"{100*score['delta_se']:.2f} pp (95% [{low:+.2f}, {high:+.2f}]); "
        f"final margin {margin['delta_vs_reference']:+.2f}±"
        f"{margin['delta_se']:.2f}; hybrid "
        f"{hybrid['delta_vs_reference']:+.2f}±{hybrid['delta_se']:.2f}; "
        f"{row['descriptive_signal'].replace('_', ' ')}"
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
        f"Evidence: {summary['completed_cases']}/{contract['case_count']} "
        f"explicit displayed plies; exactly {contract['base_paired_worlds']} "
        "paired current/future worlds per case, except 2214615196 / 10 at "
        "2,048; exact policy-20 continuation through the full remaining match.",
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
        evidence = "; ".join(
            _fmt_delta(row, candidates[0]["move"])
            for row in candidates[1:]
        )
        if evaluation.get("belief") is not None:
            belief = evaluation["belief"]
            target = belief["target"]
            top = belief["cards"][0]
            belief_text = (
                f"fixed-K {target['card']}={100*target['marginal']:.2f}% "
                f"vs {100*belief['uniform_marginal']:.2f}% prior; "
                f"top {top['card']}={100*top['marginal']:.2f}% "
                f"({'held' if top['held'] else 'not held'}); "
                f"marginal sum={belief['marginal_sum']:.6f} for K={belief['need']}"
            )
            evidence += f"; belief overlay: {belief_text}"
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
        f"- The p13 policy-neutral action panel has "
        f"{summary['p13_neutral_alternative_signals']['reference_ahead']} "
        "reference-ahead, "
        f"{summary['p13_neutral_alternative_signals']['alternative_ahead']} "
        "alternative-ahead, and "
        f"{summary['p13_neutral_alternative_signals']['inconclusive']} "
        "inconclusive alternatives.",
        "",
        f"- Counterfactual cap hits: {summary['counterfactual_cap_hits']}. "
        "The evaluator aborts the whole case on the first cap rather than "
        "reporting a truncated value.",
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
    parser.add_argument("--case", action="append", dest="case_ids",
                        help="run one frozen case; repeat for multiple cases")
    parser.add_argument("--source-commit",
                        help="full checked-out Git SHA recorded in evidence")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-json", type=Path,
        help="validate canonical exact-17 JSON and regenerate Markdown",
    )
    source.add_argument(
        "--merge", type=Path, nargs="+", metavar="SHARD_JSON",
        help="losslessly merge validated fragments into exact-17 evidence",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.from_json:
        audit = json.loads(args.from_json.read_text())
        validate_audit_document(audit, require_full=True)
    elif args.merge:
        if args.actor or args.case_ids or args.source_commit:
            raise SystemExit(
                "--actor/--case/--source-commit cannot be combined with --merge"
            )
        audit = merge_audits(
            json.loads(path.read_text()) for path in args.merge
        )
    else:
        if not args.actor:
            raise SystemExit(
                "--actor is required unless --from-json or --merge is used"
            )
        by_id = {case.case_id: case for case in CASES}
        requested_ids = args.case_ids or [case.case_id for case in CASES]
        if len(set(requested_ids)) != len(requested_ids):
            raise SystemExit("--case values must be unique")
        unknown = [case_id for case_id in requested_ids if case_id not in by_id]
        if unknown:
            raise SystemExit(f"unknown --case values: {unknown}")
        audit = build_audit(
            helper=_repo_path(args.helper), actor_spec=args.actor,
            net_path=_repo_path(args.net), worlds=args.worlds,
            symmetries=args.symmetries, belief_alpha=args.belief_alpha,
            selected_cases=(by_id[case_id] for case_id in requested_ids),
            source_commit=args.source_commit,
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
