#!/usr/bin/env python3
"""referee.py -- architecture-independent Lost Cities referee.

Pure-python + numpy port of the C engine (src/lc.c), feature encoder
(src/features.c), network forward pass (src/net.c) and argmax policy agent
(src/agent.c policy_probs / AG_POLICY).  Loads any saved net file, reading the
dimensions from the header, so nets of different architectures can be played
head-to-head even though each C binary compiles one fixed size.

Usage:
  referee.py --selftest NETPATH SEED [--dumpfeat BIN]   parity check vs C dump
  referee.py --replay   NETPATH SEED [--dumpfeat BIN]   full-game determinism
  referee.py match NETA NETB --pairs N --seed S [--rounds R]  paired-deal match
                                                (R-round matches, default 3)
"""
import argparse
import subprocess
import sys

import numpy as np

# ---- constants (lc.h) -----------------------------------------------------
NSUIT = 5
NRANK = 12
NCARD = 60
HAND_SIZE = 8
WAGERS_PER_SUIT = 3
LC_MAX_PLIES = 300

FEAT_PLANES = 7
FEAT_BIN = FEAT_PLANES * NCARD          # 420
SUIT_FEATS = 24
GLOBAL_FEATS = 16
FEAT_LEGACY_DENSE = NSUIT * SUIT_FEATS + GLOBAL_FEATS  # 136
FEAT_LEGACY_DIM = FEAT_BIN + FEAT_LEGACY_DENSE        # 556
FEAT_PILE_BURIED_DEPTHS = NRANK - 1
FEAT_PILE_CARD_FEATS = 2
FEAT_PILE_ORDER = NSUIT * FEAT_PILE_BURIED_DEPTHS * FEAT_PILE_CARD_FEATS
FEAT_DENSE = FEAT_LEGACY_DENSE + FEAT_PILE_ORDER       # 246
FEAT_DIM = FEAT_BIN + FEAT_DENSE                       # 666
NDRAW = NSUIT + 1
NCOMB = NCARD * 2 * NDRAW
VAL_SCALE = np.float32(50.0)
NET_MAGIC = 0x4C435651
MATCH_ROUNDS = 3


def card_suit(c):
    return c // NRANK


def card_rank(c):
    return c % NRANK


def card_is_wager(c):
    return card_rank(c) < WAGERS_PER_SUIT


def card_value(c):
    return 0 if card_is_wager(c) else card_rank(c) - 1


# ---- state (lc.c) ---------------------------------------------------------
class State:
    __slots__ = ("deck", "deck_pos", "deck_left", "hand", "hand_n", "played",
                 "discarded", "known", "exp_wager", "exp_top", "exp_n",
                 "exp_sum", "pile", "pile_n", "turn", "over", "nply",
                 "round", "cum")

    def __init__(self, deck):
        """lc_deal_from_deck: deck is a sequence of the 60 card ids."""
        self.deck = list(deck)
        self.deck_pos = 0
        self.hand = [0, 0]
        for p in range(2):
            for _ in range(HAND_SIZE):
                c = self.deck[self.deck_pos]
                self.deck_pos += 1
                self.hand[p] |= 1 << c
        self.hand_n = [HAND_SIZE, HAND_SIZE]
        self.deck_left = NCARD - 2 * HAND_SIZE
        self.played = [0, 0]
        self.discarded = 0
        self.known = [0, 0]  # cards each player is publicly known to hold
        self.round = 0       # match context: 0-based round index
        self.cum = [0, 0]    # match context: cumulative score from earlier rounds
        self.exp_wager = [[0] * NSUIT, [0] * NSUIT]
        self.exp_top = [[0] * NSUIT, [0] * NSUIT]
        self.exp_n = [[0] * NSUIT, [0] * NSUIT]
        self.exp_sum = [[0] * NSUIT, [0] * NSUIT]
        self.pile = [[] for _ in range(NSUIT)]
        self.pile_n = [0] * NSUIT  # kept implicitly = len(pile[s]); mirrored for clarity
        self.turn = 0
        self.over = False
        self.nply = 0

    def hand_cards(self, p):
        """ascending card ids, matching the ctz loop in lc_hand_cards"""
        h = self.hand[p]
        out = []
        while h:
            c = (h & -h).bit_length() - 1
            h &= h - 1
            out.append(c)
        return out

    def moves(self):
        """lc_moves: list of (card, discard, draw)"""
        if self.over:
            return []
        p = self.turn
        cards = self.hand_cards(p)
        src = []
        if self.deck_left > 0:
            src.append(0)
        for s in range(NSUIT):
            if len(self.pile[s]) > 0:
                src.append(s + 1)
        out = []
        wager_seen = [False] * NSUIT
        for c in cards:
            suit = card_suit(c)
            if card_is_wager(c):
                if wager_seen[suit]:
                    continue
                wager_seen[suit] = True
            val = card_value(c)
            if card_is_wager(c):
                playable = self.exp_top[p][suit] == 0
            else:
                playable = val > self.exp_top[p][suit]
            if playable:
                for k in src:
                    out.append((c, 0, k))
            for k in src:
                if k == suit + 1:
                    continue  # discard to suit forbids drawing that pile
                out.append((c, 1, k))
        return out

    def apply(self, m):
        """lc_apply = lc_apply_play + lc_apply_draw(card=-1)"""
        card, disc, draw = m
        p = self.turn
        suit = card_suit(card)
        self.hand[p] &= ~(1 << card)
        self.hand_n[p] -= 1
        if card_is_wager(card):
            wager_mask = ((1 << WAGERS_PER_SUIT) - 1) << (suit * NRANK)
            known_wagers = self.known[p] & wager_mask
            if known_wagers:
                clear = ((1 << card) if known_wagers & (1 << card)
                         else known_wagers & -known_wagers)
                self.known[p] &= ~clear
        else:
            self.known[p] &= ~(1 << card)
        if disc:
            self.pile[suit].append(card)
            self.discarded |= 1 << card
        else:
            self.played[p] |= 1 << card
            self.exp_n[p][suit] += 1
            if card_is_wager(card):
                self.exp_wager[p][suit] += 1
            else:
                self.exp_top[p][suit] = card_value(card)
                self.exp_sum[p][suit] += card_value(card)
        if draw == 0:
            c = self.deck[self.deck_pos]
            self.deck_pos += 1
            self.deck_left -= 1
            self.hand[p] |= 1 << c
        else:
            s = draw - 1
            c = self.pile[s].pop()
            self.hand[p] |= 1 << c
            self.discarded &= ~(1 << c)
            self.known[p] |= 1 << c    # taken face up: everyone saw it
        self.hand_n[p] += 1
        self.nply += 1
        if self.deck_left == 0 or self.nply >= LC_MAX_PLIES:
            self.over = True
        self.turn ^= 1

    def unseen_mask(self, p):
        """lc_unseen as a bitmask: cards p cannot locate (excludes cards the
        opponent is publicly known to hold)."""
        return (~(self.hand[p] | self.played[0] | self.played[1]
                  | self.discarded | self.known[p ^ 1])
                & ((1 << NCARD) - 1))

    def exp_score(self, p, suit):
        if self.exp_n[p][suit] == 0:
            return 0
        s = (self.exp_sum[p][suit] - 20) * (1 + self.exp_wager[p][suit])
        if self.exp_n[p][suit] >= 8:
            s += 20
        return s

    def score(self, p):
        return sum(self.exp_score(p, s) for s in range(NSUIT))


# ---- features (features.c) ------------------------------------------------
def feat_extract(st, p):
    """Returns (idx_list, dense float32[FEAT_DENSE]) matching feat_extract."""
    o = p ^ 1
    idx = []
    for plane, mask in enumerate((st.hand[p], st.played[p], st.played[o], st.discarded)):
        m = mask
        while m:
            c = (m & -m).bit_length() - 1
            m &= m - 1
            idx.append(plane * NCARD + c)
    for s in range(NSUIT):
        if len(st.pile[s]) > 0:
            idx.append(4 * NCARD + st.pile[s][-1])
    for plane, mask in ((5, st.known[o]), (6, st.known[p])):
        m = mask
        while m:
            c = (m & -m).bit_length() - 1
            m &= m - 1
            idx.append(plane * NCARD + c)

    d = np.zeros(FEAT_DENSE, dtype=np.float32)
    unseen = (~(st.hand[p] | st.played[0] | st.played[1] | st.discarded | st.known[o])
              & ((1 << NCARD) - 1))

    my_started = op_started = my_score = op_score = 0
    for s in range(NSUIT):
        b = s * SUIT_FEATS
        mytop = st.exp_top[p][s]
        optop = st.exp_top[o][s]
        msc = st.exp_score(p, s)
        osc = st.exp_score(o, s)
        my_score += msc
        op_score += osc
        if st.exp_n[p][s]:
            my_started += 1
        if st.exp_n[o][s]:
            op_started += 1

        d[b + 0] = 1.0 if st.exp_n[p][s] else 0.0
        d[b + 1] = st.exp_wager[p][s] * (1.0 / 3.0)
        d[b + 2] = st.exp_n[p][s] * (1.0 / 12.0)
        d[b + 3] = st.exp_sum[p][s] * (1.0 / 54.0)
        d[b + 4] = mytop * (1.0 / 10.0)
        d[b + 5] = msc * (1.0 / 50.0)
        d[b + 6] = 1.0 if st.exp_n[o][s] else 0.0
        d[b + 7] = st.exp_wager[o][s] * (1.0 / 3.0)
        d[b + 8] = st.exp_n[o][s] * (1.0 / 12.0)
        d[b + 9] = st.exp_sum[o][s] * (1.0 / 54.0)
        d[b + 10] = optop * (1.0 / 10.0)
        d[b + 11] = osc * (1.0 / 50.0)
        d[b + 12] = len(st.pile[s]) * (1.0 / 12.0)
        if len(st.pile[s]) > 0:
            tc = st.pile[s][-1]
            d[b + 13] = card_value(tc) * (1.0 / 10.0)
            d[b + 14] = 1.0 if card_is_wager(tc) else 0.0

        hand_cnt = play_cnt = play_sum = hand_wag = 0
        for c in range(s * NRANK, (s + 1) * NRANK):
            if (st.hand[p] >> c) & 1:
                hand_cnt += 1
                if card_is_wager(c):
                    hand_wag += 1
                    if mytop == 0:
                        play_cnt += 1
                elif card_value(c) > mytop:
                    play_cnt += 1
                    play_sum += card_value(c)
        d[b + 15] = hand_cnt * (1.0 / 12.0)
        d[b + 16] = play_cnt * (1.0 / 12.0)
        d[b + 17] = play_sum * (1.0 / 54.0)
        d[b + 18] = hand_wag * (1.0 / 3.0)

        uns_cnt = uns_mine = uns_opp = 0
        for c in range(s * NRANK, (s + 1) * NRANK):
            if (unseen >> c) & 1:
                uns_cnt += 1
                val = card_value(c)
                if not card_is_wager(c):
                    if val > mytop:
                        uns_mine += val
                    if val > optop:
                        uns_opp += val
        d[b + 19] = uns_cnt * (1.0 / 12.0)
        d[b + 20] = uns_mine * (1.0 / 54.0)
        d[b + 21] = uns_opp * (1.0 / 54.0)
        d[b + 22] = ((8 - st.exp_n[p][s]) * 0.125 if 8 - st.exp_n[p][s] > 0 else 0.0) if st.exp_n[p][s] else 0.0
        d[b + 23] = ((8 - st.exp_n[o][s]) * 0.125 if 8 - st.exp_n[o][s] > 0 else 0.0) if st.exp_n[o][s] else 0.0

    g = NSUIT * SUIT_FEATS
    d[g + 0] = st.deck_left * (1.0 / 44.0)
    d[g + 1] = 1.0 if st.turn == p else 0.0
    d[g + 2] = my_score * (1.0 / 50.0)
    d[g + 3] = op_score * (1.0 / 50.0)
    d[g + 4] = my_started * 0.2
    d[g + 5] = op_started * 0.2
    d[g + 6] = st.nply * 0.01
    d[g + 7] = st.hand_n[p] * 0.125
    d[g + 8] = st.hand_n[o] * 0.125
    d[g + 9] = 1.0
    d[g + 10] = 1.0 if st.deck_left <= 5 else 0.0
    d[g + 11] = 1.0 if st.deck_left <= 12 else 0.0
    d[g + 12] = 1.0 if st.round == 0 else 0.0
    d[g + 13] = 1.0 if st.round == 1 else 0.0
    d[g + 14] = 1.0 if st.round >= 2 else 0.0
    cm = np.float32(st.cum[p] - st.cum[o]) * np.float32(0.01)
    d[g + 15] = min(max(cm, np.float32(-1.5)), np.float32(1.5))

    po = FEAT_LEGACY_DENSE
    for s in range(NSUIT):
        for depth in range(2, NRANK + 1):
            pos = len(st.pile[s]) - depth
            if pos < 0:
                continue
            card = st.pile[s][pos]
            k = ((s * FEAT_PILE_BURIED_DEPTHS) + (depth - 2)) * 2
            d[po + k] = card_value(card) * (1.0 / 10.0)
            d[po + k + 1] = 1.0 if card_is_wager(card) else 0.0
    return idx, d


# ---- net (net.c) ----------------------------------------------------------
class Net:
    def __init__(self, path):
        with open(path, "rb") as fp:
            hdr = np.fromfile(fp, dtype=np.uint32, count=6)
            if len(hdr) != 6 or hdr[0] != NET_MAGIC:
                raise ValueError(f"{path}: bad magic")
            feat_dim, h1, h2, nplay = int(hdr[1]), int(hdr[2]), int(hdr[3]), int(hdr[4])
            version = int(hdr[5])
            legacy = version in (3, 4, 5) and feat_dim == FEAT_LEGACY_DIM
            current = version == 6 and feat_dim == FEAT_DIM
            if (not legacy and not current) or nplay != NCARD * 2:
                raise ValueError(f"{path}: unexpected FEAT_DIM/NPLAY {feat_dim}/{nplay}")
            self.h1, self.h2 = h1, h2
            n_base = (feat_dim * h1 + h1 + h1 * h2 + h2 + h2 + 1
                      + nplay * h2 + nplay + NDRAW * h2 + NDRAW)
            n_bel = NCARD * h2 + NCARD if version >= 4 else 0
            n_comb = NCOMB * h2 + NCOMB if version >= 5 else 0
            n_floats = n_base + n_bel + n_comb
            w = np.fromfile(fp, dtype=np.float32)
            if len(w) != n_floats:
                raise ValueError(
                    f"{path}: expected {n_floats} floats for v{version}, got {len(w)}"
                )
        pos = 0

        def take(*shape):
            nonlocal pos
            n = int(np.prod(shape))
            a = w[pos:pos + n].reshape(shape).copy()
            pos += n
            return a

        file_w1 = take(feat_dim, h1)
        self.w1 = np.zeros((FEAT_DIM, h1), dtype=np.float32)
        self.w1[:feat_dim] = file_w1
        self.b1 = take(h1)
        self.w2 = take(h1, h2)
        self.b2 = take(h2)
        self.w3 = take(h2)
        self.b3 = np.float32(take(1)[0])
        self.wplay = take(nplay, h2)
        self.bplay = take(nplay)
        self.wdraw = take(NDRAW, h2)
        self.bdraw = take(NDRAW)
        if version >= 4:
            self.wbel = take(NCARD, h2)
            self.bbel = take(NCARD)
        else:
            self.wbel = np.zeros((NCARD, h2), dtype=np.float32)
            self.bbel = np.zeros(NCARD, dtype=np.float32)
        if version >= 5:
            self.wcomb = take(NCOMB, h2)
            self.bcomb = take(NCOMB)
        else:
            self.wcomb = np.zeros((NCOMB, h2), dtype=np.float32)
            self.bcomb = np.zeros(NCOMB, dtype=np.float32)
        assert pos == n_floats
        self.w1_dense = self.w1[FEAT_BIN:]

    def trunk(self, idx, dense):
        h1 = self.b1 + self.w1[idx].sum(axis=0, dtype=np.float32) + dense @ self.w1_dense
        a1 = np.maximum(h1, np.float32(0.0))
        h2 = self.b2 + a1 @ self.w2
        return np.maximum(h2, np.float32(0.0))

    def value(self, a2):
        return np.float32(self.b3 + a2 @ self.w3)

    def policy_probs(self, st):
        """agent.c policy_probs: (moves, probs float32[n], value_in_points)"""
        mv = st.moves()
        if not mv:
            return [], None, None
        idx, dense = feat_extract(st, st.turn)
        a2 = self.trunk(idx, dense)
        value = self.value(a2) * VAL_SCALE
        ip = np.array([c * 2 + d for c, d, _ in mv])
        dr = np.array([s for _, _, s in mv])
        ic = ip * NDRAW + dr
        lg = (self.bplay[ip] + self.wplay[ip] @ a2
              + self.bdraw[dr] + self.wdraw[dr] @ a2
              + self.bcomb[ic] + self.wcomb[ic] @ a2).astype(np.float32)
        e = np.exp(lg - lg.max(), dtype=np.float32)
        prob = e / e.sum(dtype=np.float32)
        return mv, prob, float(value)

    def argmax_move(self, st):
        """AG_POLICY, temp 0: first index of the maximal probability."""
        mv, prob, _ = self.policy_probs(st)
        return mv[int(np.argmax(prob))]


# ---- verification against dumpfeat ---------------------------------------
def run_dump(binpath, netpath, seed, game=False):
    cmd = [binpath, "-n", netpath, "-s", str(seed)] + (["-g"] if game else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"dumpfeat failed: {cmd}\n{out.stderr}")
    return out.stdout.splitlines()


def selftest(netpath, seed, binpath):
    lines = run_dump(binpath, netpath, seed)
    net = Net(netpath)
    deck = None
    st = None
    max_feat = max_val = max_prob = 0.0
    nstates = 0
    nknown = 0  # dumped states where some card is publicly known to be held
    i = 0
    while i < len(lines):
        t = lines[i].split()
        if t[0] == "DEAL":
            deck = [int(x) for x in t[1:]]
            st = State(deck)
        elif t[0] == "HANDS":
            assert st.hand[0] == int(t[1], 16) and st.hand[1] == int(t[2], 16), "hand mismatch"
        elif t[0] == "STATE":
            persp = int(t[3])
            assert persp == st.turn, f"perspective mismatch at state {t[1]}"
            if st.known[0] | st.known[1]:
                nknown += 1
            cfeat = np.array(lines[i + 1].split()[1:], dtype=np.float32)
            idx, dense = feat_extract(st, persp)
            pfeat = np.zeros(FEAT_DIM, dtype=np.float32)
            pfeat[idx] = 1.0
            pfeat[FEAT_BIN:] = dense
            dfeat = float(np.abs(pfeat - cfeat).max())
            max_feat = max(max_feat, dfeat)
            assert dfeat <= 1e-4, f"feature mismatch {dfeat} at state {t[1]}"
            cval = float(lines[i + 2].split()[1])
            nmv = int(lines[i + 3].split()[1])
            cmv, cprob = [], []
            for k in range(nmv):
                q = lines[i + 4 + k].split()
                cmv.append((int(q[1]), int(q[2]), int(q[3])))
                cprob.append(float(q[4]))
            mv, prob, val = net.policy_probs(st)
            assert mv == cmv, f"legal move list mismatch at state {t[1]}"
            dval = abs(val - cval)
            dprob = float(np.abs(prob - np.array(cprob, dtype=np.float32)).max())
            max_val = max(max_val, dval)
            max_prob = max(max_prob, dprob)
            assert dval <= 1e-3, f"value mismatch {dval} at state {t[1]}"
            assert dprob <= 1e-4, f"prob mismatch {dprob} at state {t[1]}"
            nstates += 1
            i += 4 + nmv
            continue
        elif t[0] == "CHOSEN":
            m = (int(t[1]), int(t[2]), int(t[3]))
            assert m in st.moves(), "chosen move not legal in python"
            st.apply(m)
        i += 1
    print(f"selftest {netpath} seed {seed}: {nstates} states OK "
          f"({nknown} with nonzero known mask)  "
          f"max|dfeat|={max_feat:.2e} max|dvalue|={max_val:.2e} max|dprob|={max_prob:.2e}")
    return max_feat, max_val, max_prob, nknown


def replay(netpath, seed, binpath):
    lines = run_dump(binpath, netpath, seed, game=True)
    deck = None
    cmoves = []
    cscore = None
    for ln in lines:
        t = ln.split()
        if t[0] == "DEAL":
            deck = [int(x) for x in t[1:]]
        elif t[0] == "MOVE":
            cmoves.append((int(t[1]), int(t[2]), int(t[3])))
        elif t[0] == "SCORE":
            cscore = (int(t[1]), int(t[2]))
    net = Net(netpath)
    st = State(deck)
    pmoves = []
    while not st.over:
        m = net.argmax_move(st)
        pmoves.append(m)
        st.apply(m)
    identical = pmoves == cmoves
    pscore = (st.score(0), st.score(1))
    if identical and pscore == cscore:
        print(f"replay {netpath} seed {seed}: IDENTICAL "
              f"({len(pmoves)} moves, score {pscore[0]:+d}/{pscore[1]:+d})")
    else:
        ndiff = sum(1 for a, b in zip(pmoves, cmoves) if a != b)
        first = next((k for k, (a, b) in enumerate(zip(pmoves, cmoves)) if a != b),
                     min(len(pmoves), len(cmoves)))
        print(f"replay {netpath} seed {seed}: MISMATCH  first diff at ply {first}, "
              f"{ndiff} differing, C {len(cmoves)} vs py {len(pmoves)} moves, "
              f"score C {cscore} py {pscore}")
    return identical and pscore == cscore


# ---- match ----------------------------------------------------------------
def play_match(net_first, net_second, decks):
    """src/match.c play_one: one full match of len(decks) rounds with
    cumulative context; round r starts with player r & 1.  net_first sits in
    seat 0, net_second in seat 1.  Returns the two match totals."""
    nets = (net_first, net_second)
    cum = [0, 0]
    for r, deck in enumerate(decks):
        st = State(deck)
        st.round = r
        # Mirror src/match.c exactly.  The feature encoder may normalize the
        # difference, but changing the totals independently changes the lead.
        st.cum = [cum[0], cum[1]]
        st.turn = r & 1
        while not st.over:
            st.apply(nets[st.turn].argmax_move(st))
        cum[0] += st.score(0)
        cum[1] += st.score(1)
    return cum[0], cum[1]


def match(patha, pathb, pairs, seed, rounds=MATCH_ROUNDS):
    neta, netb = Net(patha), Net(pathb)
    rng = np.random.default_rng(seed)
    psum = psumsq = wpsum = wpsumsq = 0.0
    wins = losses = draws = 0
    for g in range(pairs):
        decks = [rng.permutation(NCARD).tolist() for _ in range(rounds)]
        s0, s1 = play_match(neta, netb, decks)    # A in seat 0
        pair = s0 - s1
        pair_win = 0.0
        if s0 > s1:
            wins += 1
            pair_win += 1.0
        elif s0 < s1:
            losses += 1
        else:
            draws += 1
            pair_win += 0.5
        s0, s1 = play_match(netb, neta, decks)    # seats swapped, same decks
        pair += s1 - s0
        if s1 > s0:
            wins += 1
            pair_win += 1.0
        elif s1 < s0:
            losses += 1
        else:
            draws += 1
            pair_win += 0.5
        psum += pair
        psumsq += pair * pair
        wpsum += pair_win
        wpsumsq += pair_win * pair_win
        if (g + 1) % 100 == 0:
            done = g + 1
            mean_pair = psum / done
            var = ((psumsq - psum * psum / done) / (done - 1)
                   if done > 1 else 0.0)
            print(f"  {done}/{pairs} pairs  margin {mean_pair / 2.0:+.2f} "
                  f"+- {np.sqrt(max(var, 0.0) / done) / 2.0:.2f}", flush=True)
    ngames = 2.0 * pairs
    mean_pair = psum / pairs
    var = ((psumsq - psum * psum / pairs) / (pairs - 1)
           if pairs > 1 else 0.0)
    wvar = ((wpsumsq - wpsum * wpsum / pairs) / (pairs - 1)
            if pairs > 1 else 0.0)
    margin = mean_pair / 2.0
    se = float(np.sqrt(max(var, 0.0) / pairs)) / 2.0
    winrate = wpsum / pairs / 2.0
    wse = float(np.sqrt(max(wvar, 0.0) / pairs)) / 2.0
    print(f"match {patha} vs {pathb}: {pairs} pairs "
          f"({int(ngames)} matches of {rounds} round(s))")
    print(f"  margin/match {margin:+.2f} +- {se:.2f}   "
          f"winrate {winrate * 100.0:.1f}% +- {wse * 100.0:.1f}% "
          f"(pair-clustered SE)   "
          f"W-L-D {wins}-{losses}-{draws}")
    return margin, se, winrate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", nargs=2, metavar=("NET", "SEED"))
    ap.add_argument("--replay", nargs=2, metavar=("NET", "SEED"))
    ap.add_argument("--dumpfeat", default="bin/dumpfeat",
                    help="dumpfeat binary compiled with matching dims")
    ap.add_argument("cmd", nargs="*", help="match NETA NETB")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=MATCH_ROUNDS,
                    help="rounds per match in match mode (default 3)")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.selftest[0], int(args.selftest[1]), args.dumpfeat)
    if args.replay:
        if not replay(args.replay[0], int(args.replay[1]), args.dumpfeat):
            return 1
    if args.cmd:
        if args.cmd[0] != "match" or len(args.cmd) != 3:
            ap.error("positional usage: match NETA NETB")
        match(args.cmd[1], args.cmd[2], args.pairs, args.seed, args.rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
