#!/bin/sh
# Final evaluation.  The ladder covers the cheap agents; the search agents are
# measured separately because a rollout move costs thousands of forward passes
# and a full round robin with them would take hours.
set -e
cd "$(dirname "$0")/.."
N="${1:-300}"
BEST="${2:-data/c8.bin}"

echo "=== ladder (n=$N paired deals per pairing) ==="
./bin/ladder -n "$N" -t 4 \
  random \
  heur \
  policy:data/best.bin \
  "policy:$BEST"

echo
echo "=== classical baseline: heuristic + perfect-information Monte Carlo ==="
./bin/arena -a hrollout:24:4 -b heur -n 60 -t 4 -r 3
./bin/arena -a "policy:$BEST" -b hrollout:24:4 -n 60 -t 4 -r 3

echo
echo "=== search on top of the trained policy ==="
./bin/arena -a "rollout:$BEST:128:4" -b "policy:$BEST" -n 120 -t 4 -r 3
