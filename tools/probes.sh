#!/bin/sh
# Regression probes: human-verified position judgments, scored against a net.
#   tools/probes.sh [NET] [WORLDS]
# Each probe replays a recorded game to a ply and reports the paired rollout
# Q difference of the known-better move over the known-worse one (positive =
# net agrees with the human).  The last column is the policy prior the net
# puts on the better move -- the leak is fixed when that rises.
NET=${1:-data/champion.bin}
W=${2:-4000}
DIR=$(dirname "$0")/../data/probes
printf "%-16s %10s %12s\n" probe "diff+-se" "prior(better)"
grep -v '^#' "$DIR/manifest.tsv" | while IFS="	" read -r name file seed ply better worse note; do
    [ -z "$name" ] && continue
    case "$file" in
      *.state) src="-S $DIR/$file" ;;
      *)       src="-f $DIR/$file -p $ply" ;;
    esac
    out=$(./bin/qpair -n "$NET" -s "$seed" $src -w "$W" -c "$better" -c "$worse" 2>/dev/null)
    diff=$(echo "$out" | tail -1 | sed 's/.*  *\([+-][0-9.]* +- [0-9.]*\)$/\1/')
    prior=$(echo "$out" | grep -o "\[$better\] [0-9.]*" | awk '{print $NF}')
    # qpair reports worse-vs-better, so flip the sign for "better over worse"
    flipped=$(echo "$diff" | awk '{s=$1; sub(/^\+/,"",s); printf "%+.2f +- %s", -s, $3}')
    printf "%-16s %10s %12s   %s\n" "$name" "$flipped" "$prior" "$note"
done
