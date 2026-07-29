#include "lc.h"
#include <stdio.h>

void lc_deal_from_deck(State *st, const uint8_t deck[NCARD])
{
    memset(st, 0, sizeof(*st));
    memcpy(st->deck, deck, NCARD);
    for (int p = 0; p < 2; p++) {
        for (int i = 0; i < HAND_SIZE; i++) {
            uint8_t c = st->deck[st->deck_pos++];
            st->hand[p] |= 1ULL << c;
        }
        st->hand_n[p] = HAND_SIZE;
    }
    st->deck_left = NCARD - 2 * HAND_SIZE;
    st->turn = 0;
    st->over = 0;
    st->nply = 0;
}

void lc_deal(State *st, Rng *rng)
{
    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    for (int i = NCARD - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = deck[i]; deck[i] = deck[j]; deck[j] = t;
    }
    lc_deal_from_deck(st, deck);
}

int lc_hand_cards(const State *st, int p, uint8_t *out)
{
    int n = 0;
    uint64_t h = st->hand[p];
    while (h) {
        int c = __builtin_ctzll(h);
        h &= h - 1;
        out[n++] = (uint8_t)c;
    }
    return n;
}

int lc_moves(const State *st, Move *out)
{
    if (st->over) return 0;
    const int p = st->turn;
    uint8_t cards[HAND_SIZE];
    int nc = lc_hand_cards(st, p, cards);

    /* draw sources available when nothing was discarded this turn */
    uint8_t src[6];
    int nsrc = 0;
    if (st->deck_left > 0) src[nsrc++] = 0;
    for (int s = 0; s < NSUIT; s++)
        if (st->pile_n[s] > 0) src[nsrc++] = (uint8_t)(s + 1);

    int n = 0;
    uint8_t wager_seen[NSUIT] = { 0 };
    for (int i = 0; i < nc; i++) {
        uint8_t c = cards[i];
        int suit = CARD_SUIT(c);
        /* The three wagers of a suit are one rules-level card type.  Offering
         * three copies of the same turn wastes policy/search capacity and
         * makes sampling depend on how many arbitrary physical IDs happen to
         * be held.  The lowest held ID is a canonical representative. */
        if (CARD_IS_WAGER(c)) {
            if (wager_seen[suit]) continue;
            wager_seen[suit] = 1;
        }
        int val = CARD_VALUE(c);
        int playable = CARD_IS_WAGER(c) ? (st->exp_top[p][suit] == 0)
                                        : (val > st->exp_top[p][suit]);
        if (playable) {
            for (int k = 0; k < nsrc; k++) {
                Move m = { c, 0, src[k] };
                out[n++] = m;
            }
        }
        /* discarding to suit `suit` forbids drawing from that same pile */
        for (int k = 0; k < nsrc; k++) {
            if (src[k] == suit + 1) continue;
            Move m = { c, 1, src[k] };
            out[n++] = m;
        }
        /* if the pile was empty before, discarding makes it non-empty but the
         * fresh card may not be taken back, so no extra source appears */
    }
    return n;
}

uint8_t lc_permute_card(uint8_t card, const uint8_t perm[NSUIT])
{
    return (uint8_t)CARD_MAKE(perm[CARD_SUIT(card)], CARD_RANK(card));
}

static uint64_t permute_mask(uint64_t mask, const uint8_t perm[NSUIT])
{
    uint64_t out = 0;
    while (mask) {
        int card = __builtin_ctzll(mask);
        mask &= mask - 1;
        out |= 1ULL << lc_permute_card((uint8_t)card, perm);
    }
    return out;
}

Move lc_permute_move(Move m, const uint8_t perm[NSUIT])
{
    m.card = lc_permute_card(m.card, perm);
    if (m.draw > 0) m.draw = (uint8_t)(perm[m.draw - 1] + 1);
    return m;
}

void lc_permute_suits(const State *src, State *dst,
                      const uint8_t perm[NSUIT])
{
    State copy;
    if (src == dst) {
        copy = *src;
        src = &copy;
    }
    *dst = *src;
    for (int i = 0; i < NCARD; i++)
        dst->deck[i] = lc_permute_card(src->deck[i], perm);
    for (int p = 0; p < 2; p++) {
        dst->hand[p] = permute_mask(src->hand[p], perm);
        dst->played[p] = permute_mask(src->played[p], perm);
        dst->known[p] = permute_mask(src->known[p], perm);
    }
    dst->discarded = permute_mask(src->discarded, perm);
    for (int s = 0; s < NSUIT; s++) {
        int q = perm[s];
        dst->pile_n[q] = src->pile_n[s];
        for (int i = 0; i < NRANK; i++)
            dst->pile[q][i] = lc_permute_card(src->pile[s][i], perm);
        for (int p = 0; p < 2; p++) {
            dst->exp_wager[p][q] = src->exp_wager[p][s];
            dst->exp_top[p][q] = src->exp_top[p][s];
            dst->exp_n[p][q] = src->exp_n[p][s];
            dst->exp_sum[p][q] = src->exp_sum[p][s];
        }
    }
}

void lc_apply_play(State *st, Move m)
{
    const int p = st->turn;
    const int suit = CARD_SUIT(m.card);

    st->hand[p] &= ~(1ULL << m.card);
    st->hand_n[p]--;
    if (CARD_IS_WAGER(m.card)) {
        /* Wager copies have no observable identity.  If at least one wager
         * of this suit was publicly known in the hand, seeing any identical
         * wager leave discharges one such guarantee, regardless of which
         * internal card id the move happened to use. */
        uint64_t wm = (uint64_t)((1u << WAGERS_PER_SUIT) - 1u)
                    << (suit * NRANK);
        uint64_t known_wagers = st->known[p] & wm;
        if (known_wagers) {
            uint64_t clear = (known_wagers >> m.card) & 1ULL
                           ? (1ULL << m.card)
                           : (known_wagers & (~known_wagers + 1ULL));
            st->known[p] &= ~clear;
        }
    } else {
        st->known[p] &= ~(1ULL << m.card);
    }

    if (m.discard) {
        st->pile[suit][st->pile_n[suit]++] = m.card;
        st->discarded |= 1ULL << m.card;
    } else {
        st->played[p] |= 1ULL << m.card;
        st->exp_n[p][suit]++;
        if (CARD_IS_WAGER(m.card)) {
            st->exp_wager[p][suit]++;
        } else {
            st->exp_top[p][suit] = (uint8_t)CARD_VALUE(m.card);
            st->exp_sum[p][suit] += (uint8_t)CARD_VALUE(m.card);
        }
    }
}

void lc_apply_draw(State *st, Move m, int card)
{
    const int p = st->turn;
    if (m.draw == 0) {
        uint8_t c;
        if (card < 0) c = st->deck[st->deck_pos++];
        else { c = (uint8_t)card; st->deck_pos++; }
        st->deck_left--;
        st->hand[p] |= 1ULL << c;
    } else {
        int s = m.draw - 1;
        uint8_t c = st->pile[s][--st->pile_n[s]];
        st->hand[p] |= 1ULL << c;
        st->discarded &= ~(1ULL << c);
        st->known[p] |= 1ULL << c;       /* taken face up: everyone saw it */
    }
    st->hand_n[p]++;
    st->nply++;
    if (st->deck_left == 0 || st->nply >= LC_MAX_PLIES) st->over = 1;
    st->turn ^= 1;
}

void lc_apply(State *st, Move m)
{
    lc_apply_play(st, m);
    lc_apply_draw(st, m, -1);
}

/* Cards that can never legally enter play again, for EITHER player, provable
 * from public information alone: expedition tops only ascend, so a number at
 * or below both tops is dead, and a wager of a suit where both sides have
 * started numbers is dead.  (A dead card can still be drawn from a pile as a
 * stall, like any card -- dead means unplayable, not untouchable.) */
uint64_t lc_dead_cards(const State *st)
{
    uint64_t dead = 0;
    for (int s = 0; s < NSUIT; s++) {
        int t0 = st->exp_top[0][s], t1 = st->exp_top[1][s];
        int base = s * NRANK;
        if (t0 > 0 && t1 > 0)
            dead |= (uint64_t)((1 << WAGERS_PER_SUIT) - 1) << base;
        int lo = t0 < t1 ? t0 : t1;      /* dead iff value <= BOTH tops */
        for (int v = 2; v <= lo; v++)
            dead |= 1ULL << (base + v + 1);
    }
    return dead;
}

/* True if discard move m is dominated by discarding a dead card instead: a
 * both-ways-dead card is the safest possible gift, so with one in hand any
 * other discard (same draw source) only adds risk.  Guards that keep this
 * honest: the dead card must be discardable while preserving m's draw (a
 * card cannot go onto the pile that is drawn from in the same turn), and
 * among equally dead cards only the lowest id survives (they differ only in
 * which pile they cover).  Deliberately NOT strict dominance -- discarding a
 * live card as bait, or preferring which pile top gets buried, are real but
 * marginal lines this trades away for search focus. */
int lc_discard_dominated(const State *st, Move m, uint64_t dead)
{
    if (!m.discard) return 0;
    uint64_t h = st->hand[st->turn] & dead;
    int mdead = (int)((dead >> m.card) & 1ULL);
    while (h) {
        int d = __builtin_ctzll(h);
        h &= h - 1;
        if (d == m.card) continue;
        if (m.draw != 0 && m.draw - 1 == CARD_SUIT(d)) continue;
        if (!mdead || d < m.card) return 1;
    }
    return 0;
}

int lc_exp_score(const State *st, int p, int suit)
{
    if (st->exp_n[p][suit] == 0) return 0;
    int s = ((int)st->exp_sum[p][suit] - 20) * (1 + (int)st->exp_wager[p][suit]);
    if (st->exp_n[p][suit] >= 8) s += 20;
    return s;
}

int lc_score(const State *st, int p)
{
    int total = 0;
    for (int s = 0; s < NSUIT; s++) total += lc_exp_score(st, p, s);
    return total;
}

/* Cards not visible to player p: opponent's hand plus the undrawn deck. */
void lc_unseen(const State *st, int p, uint8_t *out, int *n)
{
    uint64_t hidden = ~(st->hand[p] | st->played[0] | st->played[1] | st->discarded
                        | st->known[p ^ 1]);
    hidden &= (NCARD == 64) ? ~0ULL : ((1ULL << NCARD) - 1);
    int k = 0;
    while (hidden) {
        int c = __builtin_ctzll(hidden);
        hidden &= hidden - 1;
        out[k++] = (uint8_t)c;
    }
    *n = k;
}

const char *lc_card_name(int card, char *buf)
{
    static const char suits[] = "YBWGR";
    int s = CARD_SUIT(card);
    if (CARD_IS_WAGER(card)) sprintf(buf, "%cx", suits[s]);
    else sprintf(buf, "%c%d", suits[s], CARD_VALUE(card));
    return buf;
}

void lc_move_name(const State *st, Move m, char *buf)
{
    static const char suits[] = "YBWGR";
    char cn[8];
    lc_card_name(m.card, cn);
    (void)st;
    if (m.draw == 0) sprintf(buf, "%s %s, draw deck", m.discard ? "disc" : "play", cn);
    else sprintf(buf, "%s %s, take %c", m.discard ? "disc" : "play", cn, suits[m.draw - 1]);
}
