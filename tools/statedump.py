#!/usr/bin/env python3
"""Export one ply of an analysis JSON as a qpair state file (-S).

The moves-file replay only reconstructs round 0 (later deals depend on RNG
the search consumed during generation), so rounds 1-2 are probed by dumping
the recorded state directly:

    python3 tools/statedump.py data/analysis.json 72 > /tmp/ply72.state

Everything the rollout evaluation needs is public-plus-mover-hand; the
opponent hand in the record is included for completeness but is resampled
by the belief determinizer anyway.
"""
import json
import sys


def main() -> None:
    path, n = sys.argv[1], int(sys.argv[2])
    d = json.load(open(path))
    p = next(q for q in d["plies"] if q["n"] == n)
    nply = sum(1 for q in d["plies"] if q["round"] == p["round"] and q["n"] < n)
    out = []
    out.append(f"turn {p['player']}")
    out.append(f"round {p['round']}")
    out.append(f"nply {nply}")
    out.append(f"deck_left {p['deck_left']}")
    out.append(f"cum {p['cum'][0]} {p['cum'][1]}")
    for pl in (0, 1):
        out.append(f"hand{pl} " + " ".join(p["hands"][pl]))
        out.append(f"known{pl} " + " ".join(p["known"][pl]))
        for s in range(5):
            out.append(f"exp {pl} {s} " + " ".join(p["exps"][pl][s]))
    for s in range(5):
        out.append(f"pile {s} " + " ".join(p["piles"][s]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
