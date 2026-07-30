#!/usr/bin/env python3
"""Slow semantic regressions for the human-reviewed rollout positions.

These checks intentionally use fixed state files rather than replaying the
current actor. The original random showcase trajectory is not regenerated.
They are opt-in because exact per-decision suit ensembles and the two urgent
16,384-world semantic confirmations make them substantially more expensive
than the normal runtime suite:

    make audit-test
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    ply: int
    worlds: int
    symmetries: int
    candidates: tuple[str, ...]
    expected_worse: tuple[int, ...] = ()
    must_not_confidently_beat_first: tuple[int, ...] = ()


CASES = (
    Case("ply 3: deck over W2", 3, 1000, 5,
         ("Bx p deck", "Bx p W"), (1,)),
    Case("ply 4: deck over W2", 4, 1000, 5,
         ("Bx p deck", "Bx p W"), (1,)),
    Case("ply 8: deck over W2", 8, 1000, 5,
         ("B3 p deck", "B3 p W"), (1,)),
    # The wager has some continuation value at ply 10.  The regression is the
    # honest requirement: it must not be declared a confident improvement.
    Case("ply 10: W2 remains inconclusive", 10, 2000, 20,
         ("Wx p deck", "Wx p W"), (), (1,)),
    Case("ply 12: deck over R2", 12, 1024, 20,
         ("W4 p deck", "W4 p R"), (1,)),
    Case("ply 16: discard Y2 over W7", 16, 1024, 20,
         ("Y2 d deck", "W7 p deck"), (1,)),
    # Do not assert W3 over Wx: the high-compute audit says those are close.
    Case("ply 20: discard W3 over W7", 20, 1024, 20,
         ("W3 d deck", "W7 p deck", "Wx d deck"), (1,), (2,)),
)

EVAL_SPEC = (
    "rolloutu:data/champion.bin:1000:4:0.01:0.995:2:20:0:0:"
    "2:0:1.96:1:2:20:0.995:250:20:1:0:1000:1:16:12:1"
)


@dataclass(frozen=True)
class EvaluatorCase:
    name: str
    state: str
    seed: int
    selected: str
    absent: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    not_blocked: tuple[str, ...] = ()
    status_contains: tuple[str, ...] = ()
    exact_candidates: tuple[str, ...] = ()
    primary_cap: int = 0
    min_primary_worlds: int = 0
    confirmation_worlds: int = 0


# These exercise rollout_move itself, including the top-policy cap, cheap
# random-symmetry/greedy world model, independent confirmation and structural
# guard.  They are positions from the second human-reviewed UI match.
EVALUATOR_CASES = (
    EvaluatorCase(
        "ply 29: admit one planner play, not the G5 discard",
        "ui_seed725402798_p29.state",
        990029,
        "G5 p deck",
        absent=("G5 d deck",),
    ),
    EvaluatorCase(
        "ply 36: visible-hand planner chooses B10 over Y10",
        "ui_seed725402798_p36.state",
        990036,
        "B10 p deck",
    ),
    EvaluatorCase(
        "ply 40: block unsafe dead-discard replacements",
        "ui_seed725402798_p40.state",
        991588,
        "W10 p deck",
        blocked=("W2 d deck",),
        not_blocked=("R2 d deck",),
    ),
    EvaluatorCase(
        "ply 31: do not admit the low-prior G5 discard",
        "ui_seed725402798_p31.state",
        991442,
        "Y9 p deck",
        absent=("G5 d deck",),
    ),
    EvaluatorCase(
        "showcase ply 59: one-sided B wager plus useful R pickup",
        "showcase_5726968372613385_p59.state",
        990059,
        "Bx d R",
        not_blocked=("Bx d R",),
        status_contains=("confirmation: 16384 worlds, passed",),
        exact_candidates=("Y2 p deck", "Bx d R"),
        primary_cap=16384,
        min_primary_worlds=1000,
        confirmation_worlds=16384,
    ),
    EvaluatorCase(
        "showcase ply 61: one-sided B wager correction confirms",
        "showcase_5726968372613385_p61.state",
        990061,
        "Bx d R",
        status_contains=("confirmation: 16384 worlds, passed",),
        primary_cap=16384,
        min_primary_worlds=1000,
        confirmation_worlds=16384,
    ),
    EvaluatorCase(
        "showcase ply 96: finish through the last deck card",
        "showcase_5726968372613385_p96.state",
        990096,
        "G9 p deck",
        status_contains=("worlds: 0/1000",),
    ),
)


def parse_rows(output: str, candidates: tuple[str, ...]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index, candidate in enumerate(candidates):
        line = next(
            (line for line in output.splitlines() if line.startswith(candidate)),
            None,
        )
        if line is None:
            raise AssertionError(f"missing qpair row for {candidate!r}")
        fields = line[len(candidate):].split()
        row = {"q": float(fields[0]), "q_se": float(fields[1])}
        if index:
            row["delta"] = float(fields[2])
            row["delta_se"] = float(fields[4])
        else:
            row["delta"] = 0.0
            row["delta_se"] = 0.0
        rows.append(row)
    return rows


def run_case(case: Case) -> None:
    command = [
        str(ROOT / "bin/qpair"),
        "-n", str(ROOT / "data/champion.bin"),
        "-S", str(ROOT / f"data/probes/ui_seed2214615196_p{case.ply}.state"),
        "-s", str(9000 + case.ply),
        "-w", str(case.worlds),
        "-U",
        "-y", str(case.symmetries),
    ]
    for candidate in case.candidates:
        command.extend(("-c", candidate))
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"\n[{case.name}]\n{result.stdout}", end="")
    rows = parse_rows(result.stdout, case.candidates)
    for index in case.expected_worse:
        if not rows[index]["delta"] < 0.0:
            raise AssertionError(
                f"{case.name}: {case.candidates[index]} did not rank below "
                f"{case.candidates[0]} ({rows[index]})"
            )
    for index in case.must_not_confidently_beat_first:
        row = rows[index]
        if row["delta"] > 3.5 * row["delta_se"]:
            raise AssertionError(
                f"{case.name}: {case.candidates[index]} was falsely declared "
                f"a clear improvement ({row})"
            )


def run_evaluator_case(case: EvaluatorCase) -> None:
    command = [
        str(ROOT / "bin/qpair"),
        "-n", str(ROOT / "data/champion.bin"),
        "-S", str(ROOT / "data/probes" / case.state),
        "-s", str(case.seed),
        "-E", EVAL_SPEC,
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"\n[{case.name}]\n{result.stdout}", end="")
    rows = [
        line for line in result.stdout.splitlines()
        if line and not line.startswith(("position:", "hand:", "rollout ",
                                         "worlds:", "candidate"))
    ]
    selected = [
        line for line in rows
        if line.split() and line.split()[-1] == "yes"
    ]
    if len(selected) != 1 or not selected[0].startswith(case.selected):
        raise AssertionError(
            f"{case.name}: expected only {case.selected!r} selected, got "
            f"{selected!r}"
        )
    if case.exact_candidates:
        actual = tuple(" ".join(line.split()[:3]) for line in rows)
        if actual != case.exact_candidates:
            raise AssertionError(
                f"{case.name}: expected exactly {case.exact_candidates!r}, "
                f"got {actual!r}"
            )
    status_match = re.search(
        r"worlds: (\d+)/(\d+).*confirmation: (\d+) worlds",
        result.stdout,
    )
    if (
        case.primary_cap
        or case.min_primary_worlds
        or case.confirmation_worlds
    ):
        if status_match is None:
            raise AssertionError(f"{case.name}: missing world-count status")
        primary, cap, confirmation = map(int, status_match.groups())
        if case.primary_cap and cap != case.primary_cap:
            raise AssertionError(
                f"{case.name}: expected primary cap {case.primary_cap}, "
                f"got {cap}"
            )
        if case.min_primary_worlds and primary < case.min_primary_worlds:
            raise AssertionError(
                f"{case.name}: primary stopped at {primary}, below required "
                f"{case.min_primary_worlds}"
            )
        if (
            case.confirmation_worlds
            and confirmation != case.confirmation_worlds
        ):
            raise AssertionError(
                f"{case.name}: expected {case.confirmation_worlds} "
                f"confirmation worlds, got {confirmation}"
            )
    for status in case.status_contains:
        if status not in result.stdout:
            raise AssertionError(
                f"{case.name}: missing evaluator status {status!r}"
            )
    for move in case.absent:
        if any(line.startswith(move) for line in rows):
            raise AssertionError(
                f"{case.name}: low-prior candidate {move!r} entered audit"
            )
    for move in case.blocked:
        line = next((row for row in rows if row.startswith(move)), None)
        if line is None or " discard " not in f" {line} ":
            raise AssertionError(
                f"{case.name}: expected {move!r} to be guard-blocked"
            )
    for move in case.not_blocked:
        line = next((row for row in rows if row.startswith(move)), None)
        if line is None or " discard " in f" {line} ":
            raise AssertionError(
                f"{case.name}: expected {move!r} to remain guard-clear"
            )


def main() -> None:
    for case in CASES:
        run_case(case)
    for case in EVALUATOR_CASES:
        run_evaluator_case(case)
    print("\nall semantic rollout audit regressions passed")


if __name__ == "__main__":
    main()
