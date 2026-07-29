#!/usr/bin/env python3
"""Count elementary mistakes in self-play match transcripts.

Runs bin/showgame over a range of seeds and reports, per side:

  doomed opens    expeditions that finished with <= 3 cards and a clearly
                  negative score (< -8): opening a suit you cannot fund is the
                  most visible beginner error in Lost Cities
  wasted wagers   expeditions whose wagers multiplied a negative sum (score
                  below -20 with at least one wager): a wager on a failing
                  expedition doubles the damage
  instant gifts   a discard the opponent picked up on their very next turn and
                  later played: handing the opponent a card they used

These are heuristics for "elementary", not exhaustive; the same numbers run on
different checkpoints make progress (or regress) visible.
"""
import re
import subprocess
import sys
from collections import defaultdict

SUITS = "YBWGR"
SUIT_BY_NAME = {"Yellow": "Y", "Blue": "B", "White": "W", "Green": "G", "Red": "R"}


def one_match(spec, seed):
    out = subprocess.run(["./bin/showgame", "-a", spec, "-s", str(seed), "-r", "3"],
                         capture_output=True, text=True, timeout=300).stdout
    stats = defaultdict(int)
    # split rounds
    rounds = re.split(r"=+ ROUND \d+.*?=+\n", out)[0:]
    # final expedition blocks: "  Player N" then suit lines "(sum-20) x m = +s"
    for m in re.finditer(
            r"Player (\d)\n((?:\s+\w+\s+[^\n]*\n)+?)\s+total", out):
        p = int(m.group(1))
        for line in m.group(2).splitlines():
            lm = re.match(r"\s+(Yellow|Blue|White|Green|Red)\s+(.*?)\s+\((\d+)-20\) x (\d)"
                          r"( \+ 20 bonus)? = ([+-]\d+)", line)
            if not lm:
                continue
            cards = lm.group(2).split()
            mult = int(lm.group(4))
            score = int(lm.group(6))
            stats[f"p{p}_expeditions"] += 1
            if len(cards) <= 3 and score < -8:
                stats[f"p{p}_doomed_opens"] += 1
            if mult > 1 and score < -20:
                stats[f"p{p}_wasted_wagers"] += 1

    # instant gifts: parse move tables per round
    for chunk in re.split(r"=+ ROUND \d+", out)[1:]:
        moves = re.findall(r"\d+\s+P(\d)\s+(play|discard)\s+(\S+)\s+(\S+) \((\S+)\)", chunk)
        # Count a gift when P discards card X, the opponent immediately takes
        # X from that suit's pile, and later plays X.
        for i, (p, act, card, src, drawn) in enumerate(moves):
            if act != "discard":
                continue
            p = int(p)
            o = 3 - p
            # opponent's next move is moves[i+1]
            if i + 1 < len(moves):
                np_, nact, ncard, nsrc, ndrawn = moves[i + 1]
                if int(np_) == o and nsrc != "deck" and ndrawn == card \
                        and SUIT_BY_NAME.get(nsrc, nsrc) == card[0]:
                    # did the opponent eventually play it?
                    for q, (pp, aact, ccard, ssrc, ddrawn) in enumerate(moves[i + 2:]):
                        if int(pp) == o and aact == "play" and ccard == card:
                            stats[f"p{p}_instant_gifts"] += 1
                            break
    stats["matches"] += 1
    return stats


def main(spec, n):
    total = defaultdict(int)
    for seed in range(1, n + 1):
        for k, v in one_match(spec, seed).items():
            total[k] += v
    m = total["matches"]
    exp = total["p1_expeditions"] + total["p2_expeditions"]
    doomed = total["p1_doomed_opens"] + total["p2_doomed_opens"]
    wasted = total["p1_wasted_wagers"] + total["p2_wasted_wagers"]
    gifts = total["p1_instant_gifts"] + total["p2_instant_gifts"]
    print(f"{spec}: {m} matches ({exp} expeditions)")
    print(f"  doomed opens   {doomed:4d}  ({doomed/m:.2f}/match)  <=3 cards, score < -8")
    print(f"  wasted wagers  {wasted:4d}  ({wasted/m:.2f}/match)  wager on score < -20")
    print(f"  instant gifts  {gifts:4d}  ({gifts/m:.2f}/match)  discard taken next turn, played")
    return 0


if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else "policy:data/c8.bin"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    sys.exit(main(spec, n))
