#!/usr/bin/env python3
"""Mine independently useful review states from a completed duel corpus.

The opponent is used only as a *state proposer*: a state is retained when its
searched action convincingly differs from its own static-policy action.  The
opponent's score is deliberately not exported as a training label.  Run
``robust_distill --generate`` on the resulting states with our frozen champion
to create labels under our evaluator and our uncertainty gates.

Every match is replayed from its recorded deck before any state is emitted.
This catches corrupt logs, illegal actions, broken cumulative scores, and
partially written tournaments.  Output is no-clobber and directory-atomic.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable

from referee import NSUIT, State, card_is_wager, card_suit, card_value


SUIT_NAMES = "YBWGR"
COMPETITION_ROUNDS = 3
SAFE_MATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
AT_FDCWD = -100
RENAME_NOREPLACE = 1


def checked_match_id(value: Any) -> str:
    """Accept only one portable filename component from an untrusted log."""
    if not isinstance(value, str):
        raise ValueError("match id must be a string")
    match_id = value
    if not SAFE_MATCH_ID.fullmatch(match_id):
        raise ValueError(
            "match id must be 1-128 portable filename characters and start "
            "with a letter or digit"
        )
    return match_id


def checked_rounds(value: Any) -> list[Any]:
    """Reject a self-consistent but incomplete competition transcript."""
    if not isinstance(value, list) or len(value) != COMPETITION_ROUNDS:
        raise ValueError(
            f"competition match must contain exactly {COMPETITION_ROUNDS} rounds"
        )
    return value


def install_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without ever replacing an existing path.

    Linux's renameat2 is used deliberately: os.replace() and ordinary POSIX
    rename() may replace an empty destination directory after the initial
    existence check.  Failing closed on platforms without RENAME_NOREPLACE is
    safer than weakening the tool's advertised no-clobber contract.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-clobber directory install unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    rc = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if rc != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def card_name(card: int) -> str:
    return f"{SUIT_NAMES[card_suit(card)]}{'x' if card_is_wager(card) else card_value(card)}"


def cards(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def semantic_move(raw: Iterable[Any]) -> tuple[int, int, int]:
    values = tuple(int(x) for x in tuple(raw)[:3])
    if len(values) != 3:
        raise ValueError("move does not contain three fields")
    card, discard, draw = values
    if not 0 <= card < 60 or discard not in (0, 1) or not 0 <= draw <= 5:
        raise ValueError(f"invalid move {values}")
    if card_is_wager(card):
        card = card_suit(card) * 12
    return card, discard, draw


def find_candidate(candidates: list[list[Any]], move: tuple[int, int, int]) -> list[Any] | None:
    for candidate in candidates:
        if len(candidate) >= 5 and semantic_move(candidate) == move:
            return candidate
    return None


def state_text(st: State) -> str:
    lines = [
        f"turn {st.turn}",
        f"round {st.round}",
        f"nply {st.nply}",
        f"deck_left {st.deck_left}",
        f"cum {st.cum[0]} {st.cum[1]}",
    ]
    for player in (0, 1):
        lines.append(f"hand{player} " + " ".join(card_name(c) for c in cards(st.hand[player])))
        lines.append(f"known{player} " + " ".join(card_name(c) for c in cards(st.known[player])))
        for suit in range(NSUIT):
            expedition = st.played[player] & (((1 << 12) - 1) << (12 * suit))
            lines.append(
                f"exp {player} {suit} "
                + " ".join(card_name(c) for c in cards(expedition))
            )
    for suit in range(NSUIT):
        lines.append(f"pile {suit} " + " ".join(card_name(c) for c in st.pile[suit]))
    return "\n".join(line.rstrip() for line in lines) + "\n"


def finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def winner(total: list[int]) -> int:
    return 0 if total[0] > total[1] else 1 if total[1] > total[0] else -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_dir", type=Path, help="directory of duel match JSON files")
    parser.add_argument("output_dir", type=Path, help="new directory for .state files and manifest")
    parser.add_argument("--teacher", default="claude", help="engine name used only to propose states")
    parser.add_argument(
        "--teacher-policy-key",
        default="el",
        help="per-ply JSON field containing the teacher's static policy (default: el)",
    )
    parser.add_argument("--min-gap", type=float, default=2.0, help="minimum teacher mean gap")
    parser.add_argument("--z", type=float, default=3.5, help="paired standard-error multiplier")
    parser.add_argument("--min-ply", type=int, default=14, help="minimum within-round ply")
    parser.add_argument("--expect-matches", type=int, default=0, help="fail unless this many logs validate")
    parser.add_argument("--expect-selected", type=int, default=0, help="fail unless this many states qualify")
    args = parser.parse_args()

    if not math.isfinite(args.min_gap) or args.min_gap < 0:
        parser.error("--min-gap must be finite and nonnegative")
    if not math.isfinite(args.z) or args.z < 0:
        parser.error("--z must be finite and nonnegative")
    if not 0 <= args.min_ply < 300:
        parser.error("--min-ply must be between 0 and 299")
    if os.path.lexists(args.output_dir):
        parser.error(f"output already exists (refusing to replace it): {args.output_dir}")

    paths = sorted(args.match_dir.glob("*.json"))
    if not paths:
        parser.error(f"no JSON matches found in {args.match_dir}")
    if args.expect_matches and len(paths) != args.expect_matches:
        parser.error(f"expected {args.expect_matches} matches, found {len(paths)}")

    specs: dict[str, set[str]] = {}
    selected: list[tuple[str, str, dict[str, Any]]] = []
    total_plies = searched = overrides = candidate_pairs = buried = 0
    corpus_digest = hashlib.sha256()
    seen_ids: set[str] = set()

    for path in paths:
        raw_hash = sha256_file(path)
        corpus_digest.update(path.name.encode("utf-8") + b"\0" + raw_hash.encode("ascii") + b"\n")
        with path.open(encoding="utf-8") as source:
            match = json.load(source)
        if match.get("format") != 1 or match.get("ok") is not True:
            raise ValueError(f"{path}: incomplete or unsupported match")
        match_id = checked_match_id(match.get("id", ""))
        if match_id in seen_ids:
            raise ValueError(f"{path}: missing or duplicate match id {match_id!r}")
        seen_ids.add(match_id)
        seats = [match.get("seat0"), match.get("seat1")]
        if args.teacher not in seats:
            raise ValueError(f"{path}: teacher {args.teacher!r} is not seated")
        for engine in seats:
            spec = match.get(f"spec_{engine}")
            if not isinstance(spec, str) or not spec:
                raise ValueError(f"{path}: missing spec for {engine}")
            specs.setdefault(str(engine), set()).add(spec)

        cumulative = [0, 0]
        replayed_plies = 0
        rounds = checked_rounds(match.get("rounds"))
        for round_index, round_log in enumerate(rounds):
            if round_log.get("round") != round_index:
                raise ValueError(f"{path}: nonsequential round index")
            deck = round_log.get("deck")
            if not isinstance(deck, list) or sorted(deck) != list(range(60)):
                raise ValueError(f"{path}: round {round_index} has invalid deck")
            if round_log.get("cum") != cumulative:
                raise ValueError(f"{path}: round {round_index} cumulative score mismatch")
            starter = round_log.get("starter")
            if starter not in (0, 1):
                raise ValueError(f"{path}: round {round_index} invalid starter")

            st = State(deck)
            st.round = round_index
            st.cum = list(cumulative)
            st.turn = starter
            for ply_index, ply in enumerate(round_log.get("plies", [])):
                total_plies += 1
                replayed_plies += 1
                if ply.get("p") != st.turn or ply.get("dl") != st.deck_left:
                    raise ValueError(f"{path}: round {round_index} ply {ply_index} state mismatch")
                exact = tuple(int(x) for x in ply.get("m", ()))
                if len(exact) != 3 or exact not in st.moves():
                    raise ValueError(f"{path}: round {round_index} ply {ply_index} illegal move {exact}")

                engine = seats[st.turn]
                search = ply.get("s", {})
                candidates = search.get("cand", [])
                searched_here = (
                    engine == args.teacher
                    and search.get("sr") == 1
                    and isinstance(candidates, list)
                    and len(candidates) > 1
                )
                if searched_here:
                    searched += 1
                    actual = semantic_move(exact)
                    local = ply.get(args.teacher_policy_key, {}).get("t", [])
                    if not local:
                        raise ValueError(f"{path}: searched teacher ply lacks local policy")
                    baseline = semantic_move(local[0])
                    if actual != baseline:
                        overrides += 1
                        actual_row = find_candidate(candidates, actual)
                        baseline_row = find_candidate(candidates, baseline)
                        if actual_row is not None and baseline_row is not None:
                            candidate_pairs += 1
                            actual_mean = finite_number(actual_row[3], "actual mean")
                            baseline_mean = finite_number(baseline_row[3], "baseline mean")
                            actual_se = finite_number(actual_row[4], "actual SE")
                            baseline_se = finite_number(baseline_row[4], "baseline SE")
                            if actual_se < 0 or baseline_se < 0:
                                raise ValueError(f"{path}: negative candidate SE")
                            gap = actual_mean - baseline_mean
                            combined_se = math.hypot(actual_se, baseline_se)
                            if (
                                st.nply >= args.min_ply
                                and gap >= args.min_gap
                                and gap > args.z * combined_se
                            ):
                                name = f"{match_id}-r{round_index}-p{st.nply:03d}.state"
                                has_buried = any(len(pile) > 1 for pile in st.pile)
                                buried += int(has_buried)
                                selected.append((name, state_text(st), {
                                    "match": path.name,
                                    "match_sha256": raw_hash,
                                    "match_id": match_id,
                                    "round": round_index,
                                    "nply": st.nply,
                                    "player": st.turn,
                                    "teacher_action": list(exact),
                                    "teacher_static_action": list(local[0][:3]),
                                    "teacher_gap": gap,
                                    "teacher_combined_se": combined_se,
                                    "teacher_z": gap / combined_se if combined_se else None,
                                    "buried_pile": has_buried,
                                }))
                st.apply(exact)

            round_score = [st.score(0), st.score(1)]
            if not st.over or round_score != round_log.get("score"):
                raise ValueError(f"{path}: round {round_index} terminal score mismatch")
            cumulative = [cumulative[p] + round_score[p] for p in (0, 1)]

        if replayed_plies != match.get("plies") or cumulative != match.get("total"):
            raise ValueError(f"{path}: match totals mismatch")
        expected_winner = winner(cumulative)
        if match.get("winner") != expected_winner:
            raise ValueError(f"{path}: winner mismatch")
        expected_engine = "draw" if expected_winner < 0 else seats[expected_winner]
        if match.get("winner_engine") != expected_engine:
            raise ValueError(f"{path}: winner engine mismatch")

    if any(len(values) != 1 for values in specs.values()):
        detail = {engine: sorted(values) for engine, values in specs.items()}
        raise ValueError(f"engine specifications changed within corpus: {detail}")
    if args.expect_selected and len(selected) != args.expect_selected:
        parser.error(f"expected {args.expect_selected} selected states, found {len(selected)}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        entries = []
        for name, saved_state, metadata in selected:
            (temp_dir / name).write_text(saved_state, encoding="utf-8")
            entries.append({"file": name, **metadata})
        manifest = {
            "format": 1,
            "purpose": "opponent-proposed states; no opponent evaluation is a training label",
            "source_dir": str(args.match_dir),
            "source_corpus_sha256": corpus_digest.hexdigest(),
            "teacher": args.teacher,
            "teacher_policy_key": args.teacher_policy_key,
            "criteria": {
                "min_ply": args.min_ply,
                "min_gap": args.min_gap,
                "z": args.z,
                "requires_teacher_action_and_static_action_in_search_candidates": True,
            },
            "specs": {engine: next(iter(values)) for engine, values in sorted(specs.items())},
            "counts": {
                "matches": len(paths),
                "plies": total_plies,
                "teacher_searched": searched,
                "teacher_overrides": overrides,
                "comparable_overrides": candidate_pairs,
                "selected": len(selected),
                "selected_with_buried_pile": buried,
            },
            "states": entries,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        install_directory_noreplace(temp_dir, args.output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(
        f"validated {len(paths)} matches / {total_plies} plies; "
        f"selected {len(selected)} states ({buried} with buried piles)"
    )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
