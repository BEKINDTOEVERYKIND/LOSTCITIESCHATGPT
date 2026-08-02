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
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
DEFAULT_ACTOR = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:2:20:0:0:20:1:0:512:1:0:0:0:0:0:0:2"
)
GAME_MARKER = '<script type="application/json" id="game-data">'


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def actor_model_path(spec: str) -> Path:
    fields = spec.split(":")
    if len(fields) < 2 or fields[0] not in {"policy", "rollout", "rolloutu"}:
        raise RuntimeError("showcase actor must be a policy or rollout network spec")
    if not fields[1]:
        raise RuntimeError("showcase actor spec has no checkpoint path")
    return repo_path(Path(fields[1]))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--evaluator")
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

    actor_model = actor_model_path(args.actor)
    if paths_alias(args.output, actor_model):
        raise RuntimeError("--output must not replace the actor checkpoint")
    viewer_source = None
    viewer_start = viewer_end = None
    if args.embed_viewer:
        if paths_alias(args.output, args.embed_viewer):
            raise RuntimeError("--embed-viewer must differ from --output")
        if paths_alias(args.embed_viewer, actor_model):
            raise RuntimeError("--embed-viewer must not replace the actor checkpoint")
        viewer_source, viewer_start, viewer_end = viewer_template(args.embed_viewer)
    check_destination(args.output, "output")
    if args.embed_viewer:
        # viewer_template already established that this is a regular readable
        # file; this also verifies that a sibling staging file can be created.
        check_destination(args.embed_viewer, "viewer")

    model_hash = hashlib.sha256(actor_model.read_bytes()).hexdigest()
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
    if args.evaluator:
        command.extend(("-e", args.evaluator))
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    game = json.loads(result.stdout)
    if game.get("meta", {}).get("actor") != args.actor:
        raise RuntimeError("analyzer did not attest the requested actor spec")

    actor_fields = args.actor.split(":")
    rollout_actor = args.actor.startswith(("rollout:", "rolloutu:"))
    policy_actor = args.actor.startswith("policy:")
    actor_draw_root_deck_max = 0
    actor_draw_playout_deck_max = 0
    try:
        if rollout_actor and len(actor_fields) > 31:
            actor_draw_root_deck_max = int(actor_fields[31])
            if len(actor_fields) > 32:
                actor_draw_playout_deck_max = int(actor_fields[32])
        elif policy_actor and len(actor_fields) > 6:
            actor_draw_root_deck_max = int(actor_fields[6])
    except ValueError as exc:
        raise RuntimeError("actor draw-planner threshold is invalid") from exc
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
            actor_worlds = int(actor_fields[2])
            actor_root_width = int(actor_fields[3])
            actor_search_from_round_ply = int(actor_fields[7])
            actor_confirmation_worlds = int(actor_fields[21])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("rollout actor spec is incomplete") from exc
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
    game["meta"].update(
        actor_label=actor_label,
        actor_method=actor_method,
        actor_search_from_round_ply=actor_search_from_round_ply,
        actor_worlds=actor_worlds,
        actor_confirmation_worlds=actor_confirmation_worlds,
        actor_root_width=actor_root_width,
        actor_draw_root_deck_max=actor_draw_root_deck_max,
        actor_draw_playout_deck_max=actor_draw_playout_deck_max,
        model_sha256=model_hash,
        model_path=str(Path(args.actor.split(":")[1])),
        selection="random_unfiltered",
        selection_note=(
            "Seed generated randomly once before simulation; the match result "
            "and decisions were not screened, retried, or selected."
        ),
    )

    output_payload = json.dumps(game, separators=(",", ":")) + "\n"
    temporary = stage_text(args.output, output_payload)
    viewer_temporary = None
    try:
        if json.loads(temporary.read_text(encoding="utf-8")) != game:
            raise RuntimeError("standalone showcase encoding did not round-trip")

        if args.embed_viewer:
            assert viewer_source is not None
            assert viewer_start is not None and viewer_end is not None
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
