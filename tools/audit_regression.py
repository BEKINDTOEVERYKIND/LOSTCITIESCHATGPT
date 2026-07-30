#!/usr/bin/env python3
"""Slow semantic regressions for the human-reviewed rollout positions.

These checks intentionally use fixed state files rather than replaying the
current actor.  They are opt-in because exact per-decision suit ensembles make
them substantially more expensive than the normal runtime suite:

    make audit-test
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def main() -> None:
    for case in CASES:
        run_case(case)
    print("\nall semantic rollout audit regressions passed")


if __name__ == "__main__":
    main()
