#!/bin/sh
# Final evaluation.  The ladder covers the cheap agents; the search agents are
# measured separately because a rollout move costs thousands of forward passes
# and a full round robin with them would take hours.
set -e
cd "$(dirname "$0")/.."
N="${1:-300}"
BEST="${2:-data/champion.bin}"

echo "=== ladder (n=$N paired deals per pairing) ==="
./bin/ladder -n "$N" -t 4 \
  random \
  heur \
  policy:data/best.bin \
  "policy:$BEST:0:20"

echo
echo "=== classical baseline: heuristic + perfect-information Monte Carlo ==="
./bin/arena -a hrollout:24:4 -b heur -n 60 -t 4 -r 3
./bin/arena -a "policy:$BEST:0:20" -b hrollout:24:4 -n 60 -t 4 -r 3

echo
echo "=== exact visible-hand scheduler, isolated from rollout ==="
# Locked independent result (seed 703101, 2,000 mirrored pairs):
# +2.38 +- 0.33 points/match, 51.9% +- 0.4% match score.
./bin/planarena "$BEST" 2000 703101 16 12

echo
echo "=== maintained late-round wrapper on top of the trained policy ==="
# Planner and semantic tails are measured separately; their combined screen
# did not beat this locked actor, so they are not silently included here.
./bin/arena \
  -a "rolloutu:$BEST:512:4:0.02:0:1:20:0:0:0:0:3.5:2:2:20:0:0:20:1:0:512:1" \
  -b "policy:$BEST:0:20" -n 200 -t 4 -r 3
