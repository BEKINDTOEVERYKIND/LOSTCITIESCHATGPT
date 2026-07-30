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
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
DEFAULT_ACTOR = (
    "rolloutu:data/champion.bin:512:4:0.02:0:1:20:0:0:0:0:"
    "3.5:2:2:20:0:0:20:1:0:512:1"
)


def same_move(a: dict, b: dict) -> bool:
    return all(a.get(key) == b.get(key) for key in ("card", "act", "draw"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--evaluator")
    parser.add_argument("--model", type=Path, default=ROOT / "data/champion.bin")
    args = parser.parse_args()
    if not 0 <= args.seed <= MAX_SAFE_JSON_INTEGER:
        parser.error(
            f"--seed must be between 0 and {MAX_SAFE_JSON_INTEGER} so the "
            "provenance value remains exact in JavaScript"
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

    rollout_actor = args.actor.startswith(("rollout:", "rolloutu:"))
    for ply in game["plies"]:
        if not ply["policy"]:
            raise RuntimeError(f"ply {ply['n']} has no policy diagnostics")
        if not rollout_actor and not same_move(ply["policy"][0], ply["move"]):
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

    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    actor_label = "Champion policy · exact 20-way suit-symmetry ensemble"
    actor_method = "policy_argmax"
    actor_search_from_round_ply = None
    actor_worlds = None
    actor_confirmation_worlds = None
    actor_root_width = None
    if rollout_actor:
        fields = args.actor.split(":")
        try:
            actor_worlds = int(fields[2])
            actor_root_width = int(fields[3])
            actor_search_from_round_ply = int(fields[7])
            actor_confirmation_worlds = int(fields[21])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("rollout actor spec is incomplete") from exc
        actor_method = "late_round_rollout"
        actor_label = (
            "Champion + validated late-round rollout "
            f"({actor_worlds}+{actor_confirmation_worlds} worlds, "
            f"top {actor_root_width})"
        )
    game["meta"].update(
        actor_label=actor_label,
        actor_method=actor_method,
        actor_search_from_round_ply=actor_search_from_round_ply,
        actor_worlds=actor_worlds,
        actor_confirmation_worlds=actor_confirmation_worlds,
        actor_root_width=actor_root_width,
        model_sha256=model_hash,
        selection="random_unfiltered",
        selection_note=(
            "Seed generated randomly once before simulation; the match result "
            "and decisions were not screened, retried, or selected."
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as fp:
        json.dump(game, fp, separators=(",", ":"))
        fp.write("\n")
        temporary = Path(fp.name)
    temporary.replace(args.output)

    print(
        f"wrote {args.output}: seed {args.seed}, "
        f"{game['meta']['plies']} plies, final {game['meta']['final']}"
    )


if __name__ == "__main__":
    main()
