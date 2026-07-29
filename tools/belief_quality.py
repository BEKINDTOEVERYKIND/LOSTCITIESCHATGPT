#!/usr/bin/env python3
"""Evaluate opponent-hand beliefs from an analyze.c JSON dump.

New dumps contain every uncertain card and the information-set card-count
prior.  Older top-14-only dumps remain readable, but their calibration and
baseline comparisons are explicitly labelled as selection-biased.
"""
import bisect
import json
import math
import sys


def auc(rows):
    pos = sorted(p for p, held in rows if held)
    neg = sorted(p for p, held in rows if not held)
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def brier(rows):
    return sum((p - held) ** 2 for p, held in rows) / len(rows)


def logloss(rows):
    eps = 1e-9
    return -sum(
        held * math.log(min(1.0 - eps, max(eps, p)))
        + (1 - held) * math.log(min(1.0 - eps, max(eps, 1.0 - p)))
        for p, held in rows
    ) / len(rows)


def calibration(rows):
    buckets = [[] for _ in range(5)]
    for p, held in rows:
        # Integer binning avoids float-edge overlaps such as 0.6 appearing in
        # both [0.4, 0.6000000000000001) and [0.6, 0.8).
        k = min(4, max(0, int(p * 5.0)))
        buckets[k].append((p, held))
    ece = 0.0
    for k, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        observed = sum(held for _, held in bucket) / len(bucket)
        ece += len(bucket) / len(rows) * abs(mean_p - observed)
        print(
            f"  {k / 5:.1f}-{(k + 1) / 5:.1f}: n={len(bucket):5d}  "
            f"mean predicted {mean_p:.3f}  observed {observed:.3f}"
        )
    return ece


def main(path):
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)

    rows = []
    prior_rows = []
    state_aucs = []
    recalls = []
    complete = True
    states = 0

    for ply in data["plies"]:
        belief = ply.get("belief")
        if not belief:
            continue
        cards = belief.get("all_cards")
        if cards is None:
            cards = belief.get("cards", [])
            complete = False
        if not cards:
            continue

        state = [(float(card["p"]), 1 if card["held"] else 0)
                 for card in cards]
        rows.extend(state)
        states += 1
        state_auc = auc(state)
        if math.isfinite(state_auc):
            state_aucs.append(state_auc)

        prior = belief.get("prior")
        if prior is not None:
            prior_rows.extend((float(prior), held) for _, held in state)

        need = belief.get("unknown_hand")
        if need is not None and complete and need > 0:
            top = sorted(state, reverse=True)[:int(need)]
            held_total = sum(held for _, held in state)
            if held_total:
                recalls.append(sum(held for _, held in top) / held_total)

    if not rows:
        print("no belief records in", path)
        return 1

    scope = "all uncertain cards" if complete else "top-ranked cards only (selection-biased)"
    print(
        f"{len(rows)} belief records over {states} states; "
        f"{sum(held for _, held in rows)} held; {scope}"
    )
    print(f"pooled AUC: {auc(rows):.3f}")
    if state_aucs:
        print(f"mean within-state AUC: {sum(state_aucs) / len(state_aucs):.3f}")
    print(f"Brier score: {brier(rows):.4f}")
    print(f"log loss: {logloss(rows):.4f}")
    if recalls:
        print(f"top-hand-size recall: {sum(recalls) / len(recalls):.3f}")

    if prior_rows:
        print(
            f"card-count prior: AUC {auc(prior_rows):.3f}, "
            f"Brier {brier(prior_rows):.4f}, log loss {logloss(prior_rows):.4f}"
        )
        print(
            f"lift over prior: Brier {brier(prior_rows) - brier(rows):+.4f}, "
            f"log loss {logloss(prior_rows) - logloss(rows):+.4f}"
        )

    print("calibration (predicted -> observed frequency):")
    ece = calibration(rows)
    print(f"ECE (5 bins): {ece:.4f}")
    if not complete:
        print("warning: regenerate the analysis JSON to obtain unbiased all-card metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/analysis.json"))
