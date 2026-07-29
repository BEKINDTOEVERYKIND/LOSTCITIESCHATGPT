#!/usr/bin/env python3
"""Replay a showgame transcript against an independent implementation of the rules.

Nothing here shares code with the C engine: the point is that a transcript the
engine produced still checks out when the rules are written a second time.

Handles both single-round transcripts (the whole file is one deal, P1 starts,
ends with a "final score:" line) and multi-round match transcripts produced by
showgame -r N (rounds separated by "================ ROUND k (...; Pj starts)"
lines, each ending with a "round score:" line, and a closing
"================ MATCH RESULT: ..." line).  Each round is verified
independently with its actual starting player, then the round scores must sum
to the MATCH RESULT totals.
"""
import re
import sys
from collections import Counter

SUITS = "YBWGR"
SUIT_BY_NAME = {"Yellow": "Y", "Blue": "B", "White": "W", "Green": "G", "Red": "R"}

LC_MAX_PLIES = 300   # engine's anti-stall cap: a round also ends at this many plies

ROUND_SEP = re.compile(
    r"^=+ ROUND (\d+) \(totals so far P1 ([+-]\d+), P2 ([+-]\d+); P(\d) starts\) =+")
MATCH_RESULT = re.compile(
    r"^=+ MATCH RESULT: Player 1 ([+-]\d+), Player 2 ([+-]\d+) -- (.+?) =+")


def parse_round(lines):
    """Extract hands, moves, expedition listing and printed score from one round."""
    hands, moves, expeditions, final = {}, [], {}, {}
    player = None
    for line in lines:
        m = re.match(r"Player (\d) hand:\s+(.*)", line)
        if m:
            hands[int(m.group(1))] = m.group(2).split()
            continue
        m = re.match(r"\s*(\d+)\s+P(\d)\s+(play|discard)\s+(\S+)\s+(\S+) \((\S+)\)", line)
        if m:
            moves.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(4),
                          m.group(5), m.group(6)))
            continue
        m = re.match(r"\s*Player (\d)\s*$", line)
        if m:
            player = int(m.group(1))
            continue
        m = re.match(r"\s+(Yellow|Blue|White|Green|Red)\s+(-|[YBWGRx0-9 ]+?)\s+(?:\(|-$)", line)
        if m and player:
            cards = [] if m.group(2).strip() == "-" else m.group(2).split()
            expeditions[(player, SUIT_BY_NAME[m.group(1)])] = cards
            continue
        m = re.match(r"(?:final|round) score: Player 1 ([+-]\d+), Player 2 ([+-]\d+)", line)
        if m:
            final = {1: int(m.group(1)), 2: int(m.group(2))}
    return hands, moves, expeditions, final


def value(card):
    return 0 if card[1] == "x" else int(card[1:])


def verify_round(lines, start_player, tag, errs):
    """Verify one deal.  Returns the independently recomputed scores {1: .., 2: ..}.

    start_player is 1 or 2: who moves first in this deal.
    tag prefixes every reported problem so match rounds stay distinguishable.
    """
    hands, moves, expeditions, final = parse_round(lines)

    if sorted(hands) != [1, 2]:
        errs.append(f"{tag}missing dealt-hand lines")
        return None
    if not moves:
        errs.append(f"{tag}no moves found")
        return None
    if not final:
        errs.append(f"{tag}no score line found")

    # --- the deck must be exactly the 60 real cards -------------------------
    full = [s + "x" for s in SUITS for _ in range(3)] + \
           [s + str(v) for s in SUITS for v in range(2, 11)]
    assert len(full) == 60

    hand = {1: list(hands[1]), 2: list(hands[2])}
    piles = {s: [] for s in SUITS}
    played = {(p, s): [] for p in (1, 2) for s in SUITS}
    seen = list(hand[1]) + list(hand[2])          # cards revealed so far
    deck_draws = 0
    expect_turn = start_player
    expect_ply = 1

    for ply, p, act, card, src, drawn in moves:
        if ply != expect_ply:
            errs.append(f"{tag}ply {ply}: numbering broken (expected {expect_ply})")
        expect_ply = ply + 1
        if p != expect_turn:
            errs.append(f"{tag}ply {ply}: turn order broken (expected P{expect_turn})")
        expect_turn = 3 - p
        suit = card[0]

        if card not in hand[p]:
            errs.append(f"{tag}ply {ply}: P{p} used {card}, not in hand {sorted(hand[p])}")
        else:
            hand[p].remove(card)

        if act == "play":
            nums = [c for c in played[(p, suit)] if c[1] != "x"]
            if card[1] == "x":
                if nums:
                    errs.append(f"{tag}ply {ply}: wager {card} after number cards {nums}")
            elif nums and value(card) <= max(value(c) for c in nums):
                errs.append(f"{tag}ply {ply}: {card} not ascending over {nums}")
            played[(p, suit)].append(card)
        else:
            piles[suit].append(card)

        if src == "deck":
            deck_draws += 1
            seen.append(drawn)
        else:
            ps = SUIT_BY_NAME[src]
            if act == "discard" and ps == suit:
                errs.append(f"{tag}ply {ply}: took back the card just discarded to {src}")
            if not piles[ps]:
                errs.append(f"{tag}ply {ply}: drew from empty {src} pile")
            elif piles[ps][-1] != drawn:
                errs.append(f"{tag}ply {ply}: drew {drawn} from {src}, top was {piles[ps][-1]}")
                piles[ps].pop()
            else:
                piles[ps].pop()
        hand[p].append(drawn)

        if len(hand[p]) != 8:
            errs.append(f"{tag}ply {ply}: P{p} hand size {len(hand[p])}")

    # --- deck accounting ---------------------------------------------------
    # A round normally ends when the 44th deck card is drawn.  The engine also
    # cuts a stalled round off at LC_MAX_PLIES plies, in which case the deck
    # may legitimately still hold cards -- but only in that case.
    capped = len(moves) == LC_MAX_PLIES
    if not capped:
        if deck_draws != 44:
            errs.append(f"{tag}deck draws {deck_draws}, expected 44")
        if moves[-1][4] != "deck":
            errs.append(f"{tag}last ply did not draw from the deck, "
                        "so the round could not have ended")
    elif deck_draws > 44:
        errs.append(f"{tag}deck draws {deck_draws}, more than the deck holds")

    # --- every card exists exactly as often as the deck allows -------------
    # The three wagers of a suit share a name, so this is a multiset test: the
    # 16 dealt cards plus the deck draws must fit inside the 60 card deck, and
    # exhaust it exactly unless the round was cut off at the ply cap.
    got, want = Counter(seen), Counter(full)
    for card in sorted(set(full) | set(seen)):
        if got[card] > want[card] or (not capped and got[card] < want[card]):
            errs.append(f"{tag}{card}: revealed {got[card]} times, "
                        f"deck holds {want[card]}")
    if capped and sum(want.values()) - sum((want & got).values()) != 44 - deck_draws:
        errs.append(f"{tag}unrevealed cards do not match the "
                    f"{44 - deck_draws} left in the deck")

    # --- transcript's expedition listing must match the replay -------------
    for key, cards in played.items():
        listed = expeditions.get(key, [])
        if sorted(cards) != sorted(listed):
            errs.append(f"{tag}P{key[0]} {key[1]}: replay {sorted(cards)} "
                        f"vs listed {sorted(listed)}")

    # --- scoring -----------------------------------------------------------
    totals = {}
    for p in (1, 2):
        tot = 0
        for s in SUITS:
            cards = played[(p, s)]
            if not cards:
                continue
            wag = sum(1 for c in cards if c[1] == "x")
            tot += (sum(value(c) for c in cards) - 20) * (1 + wag) + (20 if len(cards) >= 8 else 0)
        totals[p] = tot
    if final and totals != final:
        errs.append(f"{tag}recomputed score {totals} vs printed {final}")

    left = sum(len(v) for v in played.values()) + sum(len(v) for v in piles.values()) + 16
    if left != 16 + deck_draws:
        errs.append(f"{tag}cards accounted for: {left}, expected {16 + deck_draws}")

    print(f"{tag or 'game: '}replayed {len(moves)} plies (P{start_player} started), "
          f"{deck_draws} deck draws, {len(moves) - deck_draws} pile draws; "
          f"independent score P1 {totals[1]:+d}, P2 {totals[2]:+d}"
          + (f" (transcript says P1 {final[1]:+d}, P2 {final[2]:+d})" if final else ""))
    return totals


def main(path):
    lines = open(path).read().splitlines()
    errs = []

    # split the file on ROUND separators; a file with none is a single round
    rounds = []          # (round_no, printed_cum, start_player, body_lines)
    match_result = None
    cur = None
    preamble = []
    for line in lines:
        m = ROUND_SEP.match(line)
        if m:
            cur = []
            rounds.append((int(m.group(1)),
                           {1: int(m.group(2)), 2: int(m.group(3))},
                           int(m.group(4)), cur))
            continue
        m = MATCH_RESULT.match(line)
        if m:
            match_result = {1: int(m.group(1)), 2: int(m.group(2))}
            cur = None
            continue
        (cur if cur is not None else preamble).append(line)

    if not rounds:
        # ---- single-round transcript: P1 starts, ends with its score line
        totals = verify_round(preamble, 1, "", errs)
        if totals is None:
            errs.append("could not parse transcript")
    else:
        # ---- match transcript
        cum = {1: 0, 2: 0}
        for round_no, printed_cum, start_player, body in rounds:
            tag = f"round {round_no}: "
            expected_start = 1 if round_no % 2 == 1 else 2   # P(r%2+1), r 0-based
            if start_player != expected_start:
                errs.append(f"{tag}separator says P{start_player} starts, "
                            f"rules say P{expected_start}")
            if printed_cum != cum:
                errs.append(f"{tag}separator totals {printed_cum} but round scores "
                            f"so far sum to {cum}")
            totals = verify_round(body, start_player, tag, errs)
            if totals:
                cum = {p: cum[p] + totals[p] for p in (1, 2)}
        if match_result is None:
            errs.append("no MATCH RESULT line found")
        elif cum != match_result:
            errs.append(f"round scores sum to {cum} but MATCH RESULT says {match_result}")
        else:
            print(f"match: {len(rounds)} rounds, totals P1 {cum[1]:+d}, P2 {cum[2]:+d} "
                  f"match the MATCH RESULT line")

    if errs:
        print(f"\n{len(errs)} PROBLEMS:")
        for e in errs:
            print("  " + e)
        return 1
    print("no rule violations, no accounting errors, scoring matches")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/game.txt"))
