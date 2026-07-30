#!/usr/bin/env python3
"""Generate one unscreened random-match artifact for the web viewer.

The seed must be chosen before this command is run.  This script never retries,
scores, or filters a match; it validates that the recorded trajectory is the
deterministic champion policy and then adds provenance metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTOR = "policy:data/champion.bin:0:20"


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

    for ply in game["plies"]:
        if not ply["policy"] or not same_move(ply["policy"][0], ply["move"]):
            raise RuntimeError(
                f"ply {ply['n']} is not the deterministic champion argmax"
            )
        # Forced and confidence-gated positions did not run a comparison.
        # Keep their sole diagnostic row descriptive, never "confirmed".
        if not ply["analysis"]["searched"]:
            for candidate in ply["search"]:
                candidate["confirmed_best"] = False

    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    game["meta"].update(
        actor_label="Champion policy · exact 20-way suit-symmetry ensemble",
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
