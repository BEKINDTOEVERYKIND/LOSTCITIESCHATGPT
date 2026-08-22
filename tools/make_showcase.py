#!/usr/bin/env python3
"""Generate one unscreened random-match artifact for the web viewer.

The seed must be chosen before this command is run.  This script never retries,
scores, or filters a match; it validates the recorded trajectory and then adds
explicit actor and selection provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
MAX_VIEWER_NUMBER = 1e9
LATE_SERIALIZATION_TOLERANCE = 1e-7
DEFAULT_ACTOR = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1"
)
DEFAULT_EVALUATOR = (
    "rolloutu:data/champion.bin:2048:5:0.01:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:2048:1:0:0:0:0:0:0:3:1:0:0:"
    "2:1:0:3:1:0:0:1"
)
GAME_MARKER = '<script type="application/json" id="game-data">'
ROLLOUT_KINDS = {"rollout", "rolloutu", "rollout2", "rolloutu2"}
TWO_NETWORK_ROLLOUT_KINDS = {"rollout2", "rolloutu2"}


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def actor_model_paths(spec: str) -> tuple[Path, Path | None]:
    """Return the actor's root and optional continuation checkpoints."""
    fields = spec.split(":")
    if len(fields) < 2 or fields[0] not in {"policy", *ROLLOUT_KINDS}:
        raise RuntimeError("showcase actor must be a policy or rollout network spec")
    if not fields[1]:
        raise RuntimeError("showcase actor spec has no checkpoint path")
    continuation = None
    if fields[0] in TWO_NETWORK_ROLLOUT_KINDS:
        if len(fields) < 3 or not fields[2]:
            raise RuntimeError(
                "two-network showcase actor spec has no continuation checkpoint path"
            )
        continuation = repo_path(Path(fields[2]))
    return repo_path(Path(fields[1])), continuation


def actor_model_path(spec: str) -> Path:
    """Backward-compatible accessor for the actor's root checkpoint."""
    return actor_model_paths(spec)[0]


def rollout_tail_start(fields: list[str]) -> int:
    """Index of the unchanged rollout tail after its checkpoint field(s)."""
    if not fields or fields[0] not in ROLLOUT_KINDS:
        raise RuntimeError("actor is not a rollout network spec")
    return 3 if fields[0] in TWO_NETWORK_ROLLOUT_KINDS else 2


def same_semantic_action(first: dict, second: dict) -> bool:
    """Whether two rendered moves differ only in their draw source."""
    return (
        first.get("card") == second.get("card")
        and first.get("act") == second.get("act")
    )


def paths_alias(first: Path, second: Path) -> bool:
    """Return whether two existing or prospective paths name one artifact."""
    try:
        if first.resolve(strict=False) == second.resolve(strict=False):
            return True
    except OSError:
        pass
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, OSError):
        return False


def viewer_template(path: Path) -> tuple[str, int, int]:
    """Read and validate the viewer insertion point without changing it."""
    if path.is_symlink():
        raise RuntimeError(f"refusing a symlinked viewer path: {path}")
    if not path.is_file():
        raise RuntimeError(f"viewer is not a regular file: {path}")
    source = path.read_text(encoding="utf-8")
    if source.count(GAME_MARKER) != 1:
        raise RuntimeError(f"{path} must contain exactly one game-data script")
    start = source.index(GAME_MARKER) + len(GAME_MARKER)
    end = source.find("</script>", start)
    if end < 0:
        raise RuntimeError(f"{path} has an unterminated game-data script")
    return source, start, end


def check_destination(path: Path, label: str) -> None:
    """Preflight a replace destination before starting an expensive match."""
    if path.is_symlink():
        raise RuntimeError(f"refusing a symlinked {label} path: {path}")
    if path.exists() and not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Actually creating a sibling catches a missing/unwritable staging
    # directory before the analyzer spends hours producing the match.
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.preflight.",
        delete=True,
    ):
        pass


def destination_mode(path: Path) -> int:
    """Preserve an existing artifact's mode; use a normal data-file default."""
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return 0o644


def stage_text(path: Path, payload: str) -> Path:
    """Write, flush, and permission a sibling temporary file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as fp:
        fp.write(payload)
        fp.flush()
        os.fsync(fp.fileno())
        temporary = Path(fp.name)
    temporary.chmod(destination_mode(path))
    return temporary


def backup_regular_file(path: Path) -> Path | None:
    """Make a same-directory rollback copy for a two-artifact install."""
    if not path.exists():
        return None
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.rollback.",
        suffix=".tmp",
        delete=False,
    ) as fp:
        backup = Path(fp.name)
    shutil.copy2(path, backup)
    with backup.open("rb") as fp:
        os.fsync(fp.fileno())
    return backup


def install_staged(
    output_temporary: Path,
    output: Path,
    viewer_temporary: Path | None,
    viewer: Path | None,
) -> None:
    """Install validated artifacts, restoring output if viewer install fails."""
    backup = backup_regular_file(output) if viewer_temporary else None
    output_existed = output.exists()
    try:
        output_temporary.replace(output)
        if viewer_temporary is not None:
            assert viewer is not None
            try:
                viewer_temporary.replace(viewer)
            except BaseException:
                # The viewer is still untouched if its atomic replace failed.
                # Put the standalone artifact back as well so the tracked pair
                # cannot silently describe two different matches.
                try:
                    if backup is not None:
                        backup.replace(output)
                        backup = None
                    elif not output_existed:
                        output.unlink(missing_ok=True)
                except OSError as rollback_error:
                    raise RuntimeError(
                        "viewer install failed and standalone rollback also failed"
                    ) from rollback_error
                raise
    finally:
        if backup is not None:
            backup.unlink(missing_ok=True)


def same_move(a: dict, b: dict) -> bool:
    return all(a.get(key) == b.get(key) for key in ("card", "act", "draw"))


def validate_late_resolver(late: object, ply: object, panel_name: str) -> None:
    """Validate every bounded-panel payload before either artifact installs."""
    location = f"ply {ply} {panel_name}"
    if not isinstance(late, dict):
        raise RuntimeError(f"{location} omits late-resolver provenance")
    boolean_fields = (
        "enabled", "attempted", "completed", "used_to_select", "stable",
        "retained_policy", "override_authorized", "practical_gate_passed",
    )
    if any(not isinstance(late.get(field), bool) for field in boolean_fields):
        raise RuntimeError(f"{location} has invalid late-resolver lifecycle")
    practical_threshold = late.get("practical_threshold")
    if (
        not isinstance(practical_threshold, (int, float))
        or isinstance(practical_threshold, bool)
        or not math.isfinite(practical_threshold)
        or not 0 <= practical_threshold <= MAX_VIEWER_NUMBER
    ):
        raise RuntimeError(
            f"{location} has invalid late-resolver practical threshold"
        )
    reason = late.get("selection_reason")
    valid_reasons = {
        "not_attempted", "unavailable", "baseline_best",
        "below_practical_gain", "horizon_disagreement",
        "challenger_override",
    }
    if reason not in valid_reasons:
        raise RuntimeError(
            f"{location} has invalid late-resolver selection reason"
        )
    support = late.get("support")
    count = late.get("candidate_count")
    candidates = late.get("candidates")
    if (
        not isinstance(support, int) or isinstance(support, bool)
        or not 0 <= support <= 990
        or not isinstance(count, int) or isinstance(count, bool)
        or not 0 <= count <= 6
        or not isinstance(candidates, list) or len(candidates) != count
    ):
        raise RuntimeError(f"{location} has invalid late-resolver support")
    horizons = []
    for name in ("horizon2", "horizon4"):
        horizon = late.get(name)
        if not isinstance(horizon, dict):
            raise RuntimeError(f"{location} omits {name} diagnostics")
        best = horizon.get("best_index")
        if not isinstance(best, int) or isinstance(best, bool) or not -1 <= best < 6:
            raise RuntimeError(f"{location} has invalid {name} best index")
        for field in ("value", "delta_vs_policy"):
            value = horizon.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool) or not math.isfinite(value)
                or not -MAX_VIEWER_NUMBER <= value <= MAX_VIEWER_NUMBER
            ):
                raise RuntimeError(f"{location} has invalid {name}.{field}")
        for field in (
            "nodes", "improved_root_nodes", "frozen_opponent_nodes",
            "transitions", "deviation_evaluations", "exact_terminal_leaves",
        ):
            value = horizon.get(field)
            if (
                not isinstance(value, int) or isinstance(value, bool)
                or not 0 <= value <= MAX_SAFE_JSON_INTEGER
            ):
                raise RuntimeError(f"{location} has invalid {name}.{field}")
        horizons.append(horizon)
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or not isinstance(candidate.get("card"), str)
            or re.fullmatch(r"[YBWGR](?:x|[2-9]|10)", candidate["card"])
                is None
            or candidate.get("act") not in {"play", "discard"}
            or candidate.get("draw") not in {"deck", "Y", "B", "W", "G", "R"}
        ):
            raise RuntimeError(f"{location} has an invalid bounded candidate")
        for field in ("policy_prob", "horizon2_q", "horizon4_q"):
            value = candidate.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool) or not math.isfinite(value)
                or not -MAX_VIEWER_NUMBER <= value <= MAX_VIEWER_NUMBER
            ):
                raise RuntimeError(
                    f"{location} has an invalid bounded candidate {field}"
                )
        if not 0.0 <= candidate["policy_prob"] <= 1.0:
            raise RuntimeError(
                f"{location} has an invalid bounded candidate policy_prob"
            )
        if any(
            not isinstance(candidate.get(field), bool)
            for field in ("policy_baseline", "horizon2_best", "horizon4_best")
        ):
            raise RuntimeError(f"{location} has invalid bounded candidate flags")
        for field in (
            "q", "q_se", "se", "delta_vs_baseline", "delta_se",
            "delta_vs_reference", "delta_reference_se", "prob", "visits",
            "confirmation_delta", "confirmation_se", "coherent_q",
            "coherent_q_se", "coherent_delta_vs_baseline",
            "coherent_delta_se", "qw", "bounded_h2_q", "bounded_h4_q",
        ):
            if field in candidate:
                value = candidate[field]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool) or not math.isfinite(value)
                    or not -MAX_VIEWER_NUMBER <= value <= MAX_VIEWER_NUMBER
                ):
                    raise RuntimeError(
                        f"{location} has an invalid bounded candidate {field}"
                    )
        for field in (
            "played", "baseline", "policy_top", "retained", "trusted_prefix",
            "prefix_proposed", "selection_reference", "highest_mean",
            "confirmed_best", "primary_pass", "coherent_evaluated",
            "coherent_numerical_agreement", "coherent_gate_pass",
            "confirmation_pass", "guard_rejected", "chosen",
        ):
            if field in candidate and not isinstance(candidate[field], bool):
                raise RuntimeError(
                    f"{location} has an invalid bounded candidate {field}"
                )
    move_keys = [
        (candidate["card"], candidate["act"], candidate["draw"])
        for candidate in candidates
    ]
    if len(set(move_keys)) != len(move_keys):
        raise RuntimeError(f"{location} has duplicate bounded candidates")
    if late["completed"] and (
        not late["attempted"] or support == 0 or count == 0
        or any(not 0 <= horizon["best_index"] < count for horizon in horizons)
    ):
        raise RuntimeError(f"{location} has an incomplete completed late panel")
    retained = late["retained_policy"]
    override = late["override_authorized"]
    if retained and override:
        raise RuntimeError(f"{location} has conflicting late outcomes")
    if late["practical_gate_passed"] != override:
        raise RuntimeError(f"{location} mislabels the challenger gain gate")
    best2 = horizons[0]["best_index"]
    best4 = horizons[1]["best_index"]
    if late["completed"]:
        for index, candidate in enumerate(candidates):
            if (
                candidate["policy_baseline"] != (index == 0)
                or candidate["horizon2_best"] != (index == best2)
                or candidate["horizon4_best"] != (index == best4)
            ):
                raise RuntimeError(f"{location} has inconsistent candidate flags")
        if late["stable"] != (best2 == best4):
            raise RuntimeError(f"{location} mislabels bounded horizon stability")
        for horizon, best, q_field in (
            (horizons[0], best2, "horizon2_q"),
            (horizons[1], best4, "horizon4_q"),
        ):
            expected_value = candidates[best][q_field]
            expected_delta = expected_value - candidates[0][q_field]
            if not math.isclose(
                horizon["value"], expected_value,
                rel_tol=0.0, abs_tol=LATE_SERIALIZATION_TOLERANCE,
            ) or not math.isclose(
                horizon["delta_vs_policy"], expected_delta,
                rel_tol=0.0, abs_tol=LATE_SERIALIZATION_TOLERANCE,
            ):
                raise RuntimeError(
                    f"{location} has inconsistent bounded value/delta diagnostics"
                )
    if override and (best2 <= 0 or best2 != best4):
        raise RuntimeError(f"{location} mislabels a bounded challenger")
    used = late["used_to_select"]
    if used != late["completed"]:
        raise RuntimeError(
            f"{location} does not make every completed bounded panel authoritative"
        )
    if used and retained == override:
        raise RuntimeError(f"{location} lacks one authoritative late outcome")
    if not used and (retained or override or late["practical_gate_passed"]):
        raise RuntimeError(f"{location} has an outcome from an unused late panel")
    if late["attempted"] and not late["enabled"]:
        raise RuntimeError(f"{location} ran a disabled bounded late panel")
    if override and (
        not late["stable"] or best2 <= 0 or best2 != best4
    ):
        raise RuntimeError(f"{location} mislabels a bounded challenger")
    delta2 = horizons[0]["delta_vs_policy"]
    delta4 = horizons[1]["delta_vs_policy"]
    definitely_below_gate = (
        delta2 <= practical_threshold - LATE_SERIALIZATION_TOLERANCE
        or delta4 <= practical_threshold - LATE_SERIALIZATION_TOLERANCE
    )
    definitely_clears_gate = (
        delta2 > practical_threshold + LATE_SERIALIZATION_TOLERANCE
        and delta4 > practical_threshold + LATE_SERIALIZATION_TOLERANCE
    )
    if override and definitely_below_gate:
        raise RuntimeError(
            f"{location} authorizes a challenger below the practical threshold"
        )
    if reason == "below_practical_gain" and definitely_clears_gate:
        raise RuntimeError(
            f"{location} retains a challenger that cleared the practical threshold"
        )

    if not late["attempted"]:
        expected_reason = "not_attempted"
    elif not late["completed"]:
        expected_reason = "unavailable"
    elif override:
        expected_reason = "challenger_override"
    elif best2 == 0 and best4 == 0:
        expected_reason = "baseline_best"
    elif not late["stable"] or best2 != best4:
        expected_reason = "horizon_disagreement"
    else:
        expected_reason = "below_practical_gain"
    if reason != expected_reason:
        raise RuntimeError(
            f"{location} mislabels the authoritative selection reason"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--evaluator", default=DEFAULT_EVALUATOR)
    parser.add_argument(
        "--model",
        type=Path,
        help=(
            "optional checkpoint assertion; its SHA-256 must match the model "
            "path embedded in --actor"
        ),
    )
    parser.add_argument(
        "--embed-viewer",
        type=Path,
        help=(
            "atomically replace the single game-data script in this viewer "
            "after the standalone artifact validates"
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.seed <= MAX_SAFE_JSON_INTEGER:
        parser.error(
            f"--seed must be between 0 and {MAX_SAFE_JSON_INTEGER} so the "
            "provenance value remains exact in JavaScript"
        )

    initial_actor_fields = args.actor.split(":")
    if initial_actor_fields[0] in TWO_NETWORK_ROLLOUT_KINDS:
        actor_model, continuation_model = actor_model_paths(args.actor)
    else:
        # Keep this legacy entry point as the single-network lookup used by
        # existing callers and tests.
        actor_model = actor_model_path(args.actor)
        continuation_model = None
    actor_checkpoints = [("root", actor_model)]
    if continuation_model is not None:
        actor_checkpoints.append(("continuation", continuation_model))
    for role, checkpoint in actor_checkpoints:
        if paths_alias(args.output, checkpoint):
            checkpoint_label = (
                "actor checkpoint"
                if continuation_model is None
                else f"actor {role} checkpoint"
            )
            raise RuntimeError(
                f"--output must not replace the {checkpoint_label}"
            )
    viewer_source = None
    viewer_start = viewer_end = None
    if args.embed_viewer:
        if paths_alias(args.output, args.embed_viewer):
            raise RuntimeError("--embed-viewer must differ from --output")
        for role, checkpoint in actor_checkpoints:
            if paths_alias(args.embed_viewer, checkpoint):
                checkpoint_label = (
                    "actor checkpoint"
                    if continuation_model is None
                    else f"actor {role} checkpoint"
                )
                raise RuntimeError(
                    "--embed-viewer must not replace the "
                    f"{checkpoint_label}"
                )
        viewer_source, viewer_start, viewer_end = viewer_template(args.embed_viewer)
    check_destination(args.output, "output")
    if args.embed_viewer:
        # viewer_template already established that this is a regular readable
        # file; this also verifies that a sibling staging file can be created.
        check_destination(args.embed_viewer, "viewer")

    checkpoint_hashes = {
        role: hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for role, checkpoint in actor_checkpoints
    }
    model_hash = checkpoint_hashes["root"]
    if args.model:
        asserted_hash = hashlib.sha256(repo_path(args.model).read_bytes()).hexdigest()
        if asserted_hash != model_hash:
            raise RuntimeError(
                "--model does not match the checkpoint named by --actor"
            )

    command = [
        str(ROOT / "bin/analyze"),
        "-r", "3",
        "-s", str(args.seed),
        "-a", args.actor,
    ]
    command.extend(("-e", args.evaluator))
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    final_checkpoint_hashes = {
        role: hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for role, checkpoint in actor_checkpoints
    }
    for role, initial_hash in checkpoint_hashes.items():
        if final_checkpoint_hashes[role] != initial_hash:
            checkpoint_label = (
                "actor checkpoint"
                if continuation_model is None
                else f"actor {role} checkpoint"
            )
            raise RuntimeError(
                f"{checkpoint_label} changed while the showcase was being "
                "analyzed"
            )
    final_model_hash = final_checkpoint_hashes["root"]
    if args.model:
        final_asserted_hash = hashlib.sha256(
            repo_path(args.model).read_bytes()
        ).hexdigest()
        if final_asserted_hash != final_model_hash:
            raise RuntimeError(
                "asserted checkpoint changed or no longer matches the actor"
            )
    game = json.loads(result.stdout)
    meta = game.get("meta") if isinstance(game, dict) else None
    plies = game.get("plies") if isinstance(game, dict) else None
    expected_meta = {
        "actor": args.actor,
        "evaluator": args.evaluator,
        "seed": args.seed,
        "rounds": 3,
        "generated": "analyze",
    }
    if not isinstance(meta, dict) or not isinstance(plies, list):
        raise RuntimeError("analyzer output omits match provenance")
    for field, expected in expected_meta.items():
        if meta.get(field) != expected:
            raise RuntimeError(
                f"analyzer did not attest requested {field}: "
                f"expected {expected!r}, got {meta.get(field)!r}"
            )
    if meta.get("plies") != len(plies):
        raise RuntimeError("analyzer ply attestation does not match its payload")

    for ply in game.get("plies", []):
        for panel_name in ("actor_decision", "analysis"):
            panel = ply.get(panel_name, {})
            unfinished = panel.get("deck2_replan", {}).get(
                "unfinished_continuation_leaves", 0
            )
            if unfinished:
                raise RuntimeError(
                    f"ply {ply.get('n')} {panel_name} contains "
                    f"{unfinished} unfinished continuation leaf/leaves"
                )
            validate_late_resolver(
                panel.get("late_resolver"), ply.get("n"), panel_name
            )

    actor_fields = args.actor.split(":")
    actor_kind = actor_fields[0]
    rollout_actor = actor_kind in ROLLOUT_KINDS
    dual_network_actor = actor_kind in TWO_NETWORK_ROLLOUT_KINDS
    policy_actor = actor_kind == "policy"
    tail_start = rollout_tail_start(actor_fields) if rollout_actor else 0
    actor_draw_root_deck_max = 0
    actor_draw_playout_deck_max = 0
    actor_terminal_mode = 1 if rollout_actor else 0
    actor_deck2_replan_worlds = 0
    actor_deck2_replan_cores = 0
    actor_bounded_late_root = False
    actor_bounded_late_min = 1.0
    try:
        if rollout_actor and len(actor_fields) > tail_start + 29:
            actor_draw_root_deck_max = int(actor_fields[tail_start + 29])
            if len(actor_fields) > tail_start + 30:
                actor_draw_playout_deck_max = int(
                    actor_fields[tail_start + 30]
                )
            if len(actor_fields) > tail_start + 35:
                actor_terminal_mode = int(actor_fields[tail_start + 35])
            if len(actor_fields) > tail_start + 36:
                actor_deck2_replan_worlds = int(
                    actor_fields[tail_start + 36]
                )
            if len(actor_fields) > tail_start + 37:
                actor_deck2_replan_cores = int(
                    actor_fields[tail_start + 37]
                )
            if len(actor_fields) > tail_start + 38:
                actor_bounded_late_root = bool(
                    int(actor_fields[tail_start + 38])
                )
            if len(actor_fields) > tail_start + 39:
                actor_bounded_late_min = float(
                    actor_fields[tail_start + 39]
                )
        elif policy_actor and len(actor_fields) > 6:
            actor_draw_root_deck_max = int(actor_fields[6])
    except ValueError as exc:
        raise RuntimeError("actor rollout tail is invalid") from exc
    for ply in game["plies"]:
        if not ply["policy"]:
            raise RuntimeError(f"ply {ply['n']} has no policy diagnostics")
        if not rollout_actor and not same_move(ply["policy"][0], ply["move"]):
            actor_decision = ply.get("actor_decision", {})
            planned_draw = (
                actor_draw_root_deck_max > 0
                and actor_decision.get("baseline_source")
                    == "draw_source_planner"
                and same_semantic_action(ply["policy"][0], ply["move"])
            )
            if not planned_draw:
                raise RuntimeError(
                    f"ply {ply['n']} is not the deterministic champion argmax"
                )
        actor_candidates = ply.get("actor_decision", {}).get("candidates", [])
        if rollout_actor and not (
            any(same_move(candidate, ply["move"]) for candidate in ply["policy"])
            or any(
                same_move(candidate, ply["move"])
                for candidate in actor_candidates
            )
        ):
            raise RuntimeError(
                f"ply {ply['n']} rollout move is absent from both the policy "
                "and targeted actor diagnostics"
            )
        # Forced and confidence-gated positions did not run a comparison.
        # Keep their sole diagnostic row descriptive, never "confirmed".
        if not ply["analysis"]["searched"]:
            for candidate in ply["search"]:
                candidate["confirmed_best"] = False

    actor_label = "Champion policy · exact 20-way suit-symmetry ensemble"
    actor_method = "policy_argmax"
    actor_search_from_round_ply = None
    actor_worlds = None
    actor_confirmation_worlds = None
    actor_root_width = None
    if rollout_actor:
        try:
            actor_worlds = int(actor_fields[tail_start])
            actor_root_width = int(actor_fields[tail_start + 1])
            actor_search_from_round_ply = int(actor_fields[tail_start + 5])
            actor_confirmation_worlds = int(actor_fields[tail_start + 19])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("rollout actor spec is incomplete") from exc
        if dual_network_actor:
            root_name = Path(actor_fields[1]).name
            continuation_name = Path(actor_fields[2]).name
            actor_method = "dual_network_late_round_rollout_consensus"
            actor_label = (
                f"Root {root_name} (policy, value, belief, shortlist) + "
                f"continuation {continuation_name} (post-candidate play) · "
                "validated coherent rollout consensus "
                f"({actor_worlds}+{actor_confirmation_worlds} worlds, "
                f"top {actor_root_width})"
            )
        else:
            actor_method = "late_round_rollout_consensus"
            actor_label = (
                "Champion + validated coherent rollout consensus "
                f"({actor_worlds}+{actor_confirmation_worlds} worlds, "
                f"top {actor_root_width})"
            )
    if actor_draw_root_deck_max > 0:
        actor_method += "_with_information_set_draw_repair"
        actor_label += (
            f" + root draw repair at deck ≤ {actor_draw_root_deck_max}"
        )
    actor_exact_terminal = actor_terminal_mode != 0
    actor_exact_terminal_continuations = actor_terminal_mode == 1
    if actor_exact_terminal_continuations:
        actor_method += "_with_exact_terminal_tail"
        actor_label += " + exact terminal tail"
    elif actor_terminal_mode == 3:
        actor_method += "_with_policy_action_terminal_control"
        actor_label += " + policy-action terminal control"
    elif actor_exact_terminal:
        actor_method += "_with_root_only_terminal_solver"
        actor_label += " + root-only terminal solver"
    if actor_deck2_replan_worlds > 0:
        actor_method += "_with_recursive_late_information_set_replan"
        actor_label += (
            " + recursive deck-2/3 replan "
            f"({actor_deck2_replan_worlds} worlds, "
            f"top {actor_deck2_replan_cores} cores)"
        )
    if actor_bounded_late_root:
        actor_method += "_with_authoritative_bounded_late_root"
        actor_label += (
            " + authoritative bounded deck-2/3 root gate "
            f"(>{actor_bounded_late_min:g} point gain)"
        )
    match_id = f"{args.seed}-{model_hash[:12]}"
    dual_model_provenance = {}
    if dual_network_actor:
        assert continuation_model is not None
        continuation_hash = checkpoint_hashes["continuation"]
        # Include both complete content identities.  Root-only IDs would make
        # two distinct continuation policies appear to be the same match.
        match_id = f"{args.seed}-{model_hash}-{continuation_hash}"
        dual_model_provenance = {
            "root_model_path": actor_fields[1],
            "root_model_sha256": model_hash,
            "root_model_role": "policy, value, belief, and root shortlist",
            "continuation_model_path": actor_fields[2],
            "continuation_model_sha256": continuation_hash,
            "continuation_model_role": (
                "policy decisions after each evaluated root candidate"
            ),
        }
    game["meta"].update(
        actor_label=actor_label,
        actor_method=actor_method,
        actor_search_from_round_ply=actor_search_from_round_ply,
        actor_worlds=actor_worlds,
        actor_confirmation_worlds=actor_confirmation_worlds,
        actor_root_width=actor_root_width,
        actor_draw_root_deck_max=actor_draw_root_deck_max,
        actor_draw_playout_deck_max=actor_draw_playout_deck_max,
        actor_exact_terminal=actor_exact_terminal,
        actor_exact_terminal_continuations=(
            actor_exact_terminal_continuations
        ),
        actor_terminal_mode=actor_terminal_mode,
        actor_deck2_replan_worlds=actor_deck2_replan_worlds,
        actor_deck2_replan_cores=actor_deck2_replan_cores,
        actor_bounded_late_root=actor_bounded_late_root,
        actor_bounded_late_min=actor_bounded_late_min,
        model_sha256=model_hash,
        model_path=str(Path(args.actor.split(":")[1])),
        match_id=match_id,
        selection="random_unfiltered",
        selection_note=(
            "Seed generated randomly once before simulation; the match result "
            "and decisions were not screened, retried, or selected."
        ),
        **dual_model_provenance,
    )

    output_payload = json.dumps(game, separators=(",", ":")) + "\n"
    temporary = stage_text(args.output, output_payload)
    viewer_temporary = None
    try:
        if json.loads(temporary.read_text(encoding="utf-8")) != game:
            raise RuntimeError("standalone showcase encoding did not round-trip")

        if args.embed_viewer:
            # Analysis can take a long time.  Re-read the template immediately
            # before staging so a concurrent UI fix is incorporated rather
            # than silently overwritten by the pre-run snapshot.
            viewer_source, viewer_start, viewer_end = viewer_template(
                args.embed_viewer
            )
            embedded = json.dumps(game, separators=(",", ":"))
            embedded = (
                embedded.replace("&", "\\u0026")
                .replace("<", "\\u003c")
                .replace(">", "\\u003e")
            )
            if json.loads(embedded) != game:
                raise RuntimeError("script-safe showcase encoding did not round-trip")
            rendered = (
                viewer_source[:viewer_start]
                + embedded
                + viewer_source[viewer_end:]
            )
            viewer_temporary = stage_text(args.embed_viewer, rendered)
            staged_source, staged_start, staged_end = viewer_template(viewer_temporary)
            if json.loads(staged_source[staged_start:staged_end]) != game:
                raise RuntimeError("staged viewer showcase did not round-trip")

        install_staged(
            temporary,
            args.output,
            viewer_temporary,
            args.embed_viewer,
        )
        temporary = None
        viewer_temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if viewer_temporary is not None:
            viewer_temporary.unlink(missing_ok=True)

    print(
        f"wrote {args.output}: seed {args.seed}, "
        f"{game['meta']['plies']} plies, final {game['meta']['final']}"
    )


if __name__ == "__main__":
    main()
