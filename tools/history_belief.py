#!/usr/bin/env python3
"""Offline actor-aware hand inference from a perspective-scrubbed history.

The input may be an omniscient ``analyze``/showcase JSON file, but inference
never receives that object.  ``build_view`` first reduces it to information
the selected observer legitimately has: their initial hand, public actions,
and their own deck draws before the target position.  Opponent deck-draw
identities, opponent hands, future actions, and truth labels are absent from
the wire format accepted by ``bin/history_belief``.

Version 1's transport format is limited to positions at round ply 0..20, but
the actor-specific check is stricter: every preceding action must come from
the deterministic exact-symmetry policy prefix.  The current maintained actor
starts search at ply 14, so its safe targets end at ply 14.  Later rollout
decisions contain private search randomness and require a sequential
likelihood model rather than pretending they are deterministic policy actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUITS = "YBWGR"
DEFAULT_SEED = 0xA7710B311EF


class ViewError(ValueError):
    """The source cannot be reduced to a valid perspective view."""


def parse_card(name: str) -> tuple[int, int]:
    if not isinstance(name, str) or len(name) < 2 or name[0] not in SUITS:
        raise ViewError(f"invalid card name: {name!r}")
    suffix = name[1:]
    if suffix == "x":
        value = 0
    else:
        try:
            value = int(suffix)
        except ValueError as exc:
            raise ViewError(f"invalid card name: {name!r}") from exc
        if not 2 <= value <= 10:
            raise ViewError(f"invalid card value: {name!r}")
    return SUITS.index(name[0]), value


def card_name(suit: int, value: int) -> str:
    return f"{SUITS[suit]}{'x' if value == 0 else value}"


def draw_source(name: str) -> int:
    if name == "deck":
        return 0
    if isinstance(name, str) and len(name) == 1 and name in SUITS:
        return SUITS.index(name) + 1
    raise ViewError(f"invalid draw source: {name!r}")


def canonical_view_bytes(view: dict[str, Any]) -> bytes:
    return json.dumps(
        view, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def build_view(
    game: dict[str, Any], target_ply: int, observer: int | None = None
) -> dict[str, Any]:
    """Return the complete, and only, information available to inference."""
    if observer is not None and observer not in (0, 1):
        raise ViewError("observer must be 0 or 1")
    plies = game.get("plies")
    if not isinstance(plies, list):
        raise ViewError("analysis JSON has no ply list")
    try:
        target = next(p for p in plies if int(p["n"]) == target_ply)
    except (StopIteration, KeyError, TypeError, ValueError) as exc:
        raise ViewError(f"target ply {target_ply} is absent") from exc

    round_index = int(target["round"])
    round_ply = int(target["round_ply"])
    if not 0 <= round_ply <= 20:
        raise ViewError(
            "v1 supports only target positions through round ply 20; "
            "later actions may have used stochastic rollout search"
        )
    if observer is None:
        observer = int(target["player"])

    round_records = sorted(
        (p for p in plies if int(p["round"]) == round_index),
        key=lambda p: int(p["round_ply"]),
    )
    if not round_records or int(round_records[0]["round_ply"]) != 0:
        raise ViewError("round does not start at round ply zero")
    if round_ply > len(round_records):
        raise ViewError("round-ply index exceeds recorded round")

    # Reading only hands[observer] is deliberate.  The other hand remains
    # outside the scrubbed view even though the source artifact contains it.
    first = round_records[0]
    hands = first.get("hands")
    if (
        not isinstance(hands, list)
        or len(hands) != 2
        or not isinstance(hands[observer], list)
        or len(hands[observer]) != 8
    ):
        raise ViewError("observer initial hand is malformed")
    own_initial = sorted(
        (parse_card(name) for name in hands[observer]),
        key=lambda item: (item[0], item[1]),
    )

    prefix = [p for p in round_records if int(p["round_ply"]) < round_ply]
    if [int(record["round_ply"]) for record in prefix] != list(range(round_ply)):
        raise ViewError("round history has a gap before target")

    events: list[dict[str, Any]] = []
    expected_actor = int(first["player"])
    for record in prefix:
        actor = int(record["player"])
        if actor != expected_actor:
            raise ViewError("recorded actors do not alternate")
        expected_actor ^= 1
        move = record.get("move")
        if not isinstance(move, dict):
            raise ViewError("recorded move is malformed")
        suit, value = parse_card(move.get("card"))
        action = move.get("act")
        if action not in ("play", "discard"):
            raise ViewError(f"invalid action: {action!r}")
        draw = draw_source(move.get("draw"))
        event: dict[str, Any] = {
            "actor": actor,
            "card": [suit, value],
            "discard": int(action == "discard"),
            "draw": draw,
        }
        # This is the sole private-history field.  Never even access "drawn"
        # for an opponent deck draw.
        if actor == observer and draw == 0:
            event["own_draw"] = list(parse_card(move.get("drawn")))
        events.append(event)

    cum = first.get("cum")
    if (
        not isinstance(cum, list)
        or len(cum) != 2
        or not all(isinstance(x, int) for x in cum)
    ):
        raise ViewError("round cumulative score is malformed")

    return {
        "schema": "lc-perspective-history-v1",
        "observer": observer,
        "round": round_index,
        "start_turn": int(first["player"]),
        "cum": [int(cum[0]), int(cum[1])],
        "target_global_ply": int(target["n"]),
        "target_round_ply": round_ply,
        "own_initial": [list(card) for card in own_initial],
        "events": events,
    }


def worker_wire(view: dict[str, Any]) -> str:
    """Serialize only the scrubbed schema understood by the C worker."""
    lines = ["LCBH1"]
    lines.append(
        f"{view['observer']} {view['round']} {view['start_turn']} "
        f"{view['cum'][0]} {view['cum'][1]} {len(view['events'])}"
    )
    own = view["own_initial"]
    lines.append(
        str(len(own))
        + "".join(f" {int(suit)} {int(value)}" for suit, value in own)
    )
    for event in view["events"]:
        known = event.get("own_draw", [-1, -1])
        lines.append(
            f"{event['actor']} {event['card'][0]} {event['card'][1]} "
            f"{event['discard']} {event['draw']} {known[0]} {known[1]}"
        )
    return "\n".join(lines) + "\n"


def validate_actor_prefix(
    actor_spec: Any,
    target_round_ply: int,
    symmetries: int,
    net_path: Path,
) -> str:
    """Prove that all observed opponent actions used the policy directly."""
    if not isinstance(actor_spec, str):
        raise ViewError("analysis metadata does not identify its actor")
    fields = actor_spec.split(":")
    kind = fields[0]
    if len(fields) < 2 or not fields[1]:
        raise ViewError("source actor spec does not identify its checkpoint")
    source_net_path = Path(fields[1])
    if not source_net_path.is_absolute():
        source_net_path = ROOT / source_net_path
    try:
        source_hash = hashlib.sha256(source_net_path.read_bytes()).digest()
        inference_hash = hashlib.sha256(net_path.read_bytes()).digest()
    except OSError as exc:
        raise ViewError(f"cannot verify source actor checkpoint: {exc}") from exc
    if source_hash != inference_hash:
        raise ViewError(
            "inference checkpoint does not match the source actor checkpoint"
        )

    def numeric(index: int, default: float, label: str) -> float:
        if index >= len(fields):
            return default
        try:
            return float(fields[index])
        except ValueError as exc:
            raise ViewError(
                f"source actor has malformed {label}: {fields[index]!r}"
            ) from exc

    try:
        if kind == "policy":
            temperature = numeric(2, 0.0, "policy temperature")
            actor_symmetries = int(numeric(3, 1.0, "policy symmetries"))
            plan_deck_max = int(numeric(4, 0.0, "planner deck threshold"))
            plan_block_gap = int(numeric(5, 0.0, "planner block gap"))
            if target_round_ply > 0 and temperature != 0.0:
                raise ViewError(
                    "target prefix used stochastic policy sampling; v1 "
                    "requires deterministic argmax actions"
                )
            if (
                target_round_ply > 0
                and plan_deck_max > 0
                and plan_block_gap > 0
            ):
                raise ViewError(
                    "target prefix comes from a planner-enabled policy; v1 "
                    "models only unmodified argmax actions"
                )
        elif kind in ("rollout", "rolloutu"):
            search_from = int(numeric(7, 0.0, "rollout ply threshold"))
            actor_symmetries = int(
                numeric(15, 1.0, "rollout policy symmetries")
            )
            plan_deck_max = int(
                numeric(23, 0.0, "rollout planner deck threshold")
            )
            plan_block_gap = int(
                numeric(24, 0.0, "rollout planner block gap")
            )
            semantic_candidates = int(
                numeric(25, 0.0, "semantic-candidate flag")
            )
            if target_round_ply > search_from:
                raise ViewError(
                    "target prefix includes a rollout-search action; v1 can "
                    "infer only the deterministic policy prefix"
                )
            if (
                target_round_ply > 0
                and plan_deck_max > 0
                and plan_block_gap > 0
            ):
                raise ViewError(
                    "target prefix comes from a planner-enabled rollout actor; "
                    "v1 models only unmodified argmax actions"
                )
            if target_round_ply > 0 and semantic_candidates:
                raise ViewError(
                    "target prefix comes from a semantic-search-enabled "
                    "rollout actor; v1 cannot integrate over that search"
                )
        else:
            raise ViewError(f"unsupported source actor: {kind!r}")
    except ViewError:
        raise
    except (OverflowError, ValueError) as exc:
        raise ViewError("source actor spec has malformed numeric fields") from exc
    if actor_symmetries != symmetries:
        raise ViewError(
            f"source actor used {actor_symmetries} policy symmetries, "
            f"but inference requested {symmetries}"
        )
    return actor_spec


def annotate(
    game: dict[str, Any],
    *,
    target_ply: int,
    observer: int | None,
    worlds: int,
    seed: int,
    symmetries: int,
    net_path: Path,
    worker_path: Path,
) -> dict[str, Any]:
    if not 0 <= seed <= (1 << 64) - 1:
        raise ViewError("annotation seed must fit an unsigned 64-bit integer")
    view = build_view(game, target_ply, observer)
    view_bytes = canonical_view_bytes(view)
    if not worker_path.is_file():
        raise ViewError(
            f"worker not found at {worker_path}; run `make bin/history_belief`"
        )
    if not net_path.is_file():
        raise ViewError(f"network not found at {net_path}")

    meta = game.get("meta") if isinstance(game.get("meta"), dict) else {}
    source_actor = validate_actor_prefix(
        meta.get("actor"), view["target_round_ply"], symmetries, net_path
    )

    command = [
        str(worker_path),
        "-n",
        str(net_path),
        "-w",
        str(worlds),
        "-s",
        str(seed),
        "-y",
        str(symmetries),
    ]
    completed = subprocess.run(
        command,
        input=worker_wire(view),
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "worker failed without diagnostics"
        raise ViewError(detail)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ViewError("worker emitted invalid JSON") from exc

    for card in result.get("cards", []):
        card["card"] = card_name(int(card["suit"]), int(card["value"]))

    result["provenance"] = {
        "information_contract": (
            "observer initial hand + public prefix + observer deck draws; "
            "opponent deck draws, opponent hands, truth labels, and future "
            "actions excluded"
        ),
        "target_global_ply": view["target_global_ply"],
        "target_round_ply": view["target_round_ply"],
        "observer": view["observer"],
        "round": view["round"],
        "source_actor_spec": source_actor,
        "inference_actor": (
            f"policy:{net_path}:0:{symmetries} "
            "(source actor's pre-search behavior)"
        ),
        "model_sha256": hashlib.sha256(net_path.read_bytes()).hexdigest(),
        "view_sha256": hashlib.sha256(view_bytes).hexdigest(),
        "annotation_seed": seed,
        "selection": "all sampled worlds; no posterior screening or retries",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--ply", required=True, type=int)
    parser.add_argument("--observer", type=int, choices=(0, 1))
    parser.add_argument("--worlds", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--symmetries", type=int, default=20)
    parser.add_argument("--net", type=Path, default=ROOT / "data/champion.bin")
    parser.add_argument(
        "--worker", type=Path, default=ROOT / "bin/history_belief"
    )
    parser.add_argument(
        "--dump-view",
        action="store_true",
        help="emit only the perspective-scrubbed view; do not run inference",
    )
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        game = json.loads(args.analysis.read_text(encoding="utf-8"))
        if args.dump_view:
            output: Any = build_view(game, args.ply, args.observer)
        else:
            output = annotate(
                game,
                target_ply=args.ply,
                observer=args.observer,
                worlds=args.worlds,
                seed=args.seed,
                symmetries=args.symmetries,
                net_path=args.net.resolve(),
                worker_path=args.worker.resolve(),
            )
    except (OSError, json.JSONDecodeError, ViewError) as exc:
        print(f"history_belief.py: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
