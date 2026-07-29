#include "features.h"

void feat_extract(const State *st, int p, Features *f)
{
    const int o = p ^ 1;
    int n = 0;

    /* --- card planes -------------------------------------------------- */
    uint64_t mask;
    mask = st->hand[p];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(0 * NCARD + c); }
    mask = st->played[p];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(1 * NCARD + c); }
    mask = st->played[o];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(2 * NCARD + c); }
    mask = st->discarded;
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(3 * NCARD + c); }
    for (int s = 0; s < NSUIT; s++)
        if (st->pile_n[s] > 0) f->idx[n++] = (uint16_t)(4 * NCARD + st->pile[s][st->pile_n[s] - 1]);
    /* cards taken face up: I know the opponent holds these, and they know I
     * hold mine -- both facts shape what is worth discarding or racing for */
    mask = st->known[o];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(5 * NCARD + c); }
    mask = st->known[p];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(6 * NCARD + c); }
    f->nidx = n;

    /* --- dense ---------------------------------------------------------- */
    float *d = f->dense;
    for (int i = 0; i < FEAT_DENSE; i++) d[i] = 0.0f;

    uint64_t unseen = ~(st->hand[p] | st->played[0] | st->played[1] | st->discarded
                        | st->known[o])
                      & ((1ULL << NCARD) - 1);

    int my_started = 0, op_started = 0, my_score = 0, op_score = 0;

    for (int s = 0; s < NSUIT; s++) {
        float *v = d + s * SUIT_FEATS;
        int mytop = st->exp_top[p][s], optop = st->exp_top[o][s];
        int msc = lc_exp_score(st, p, s), osc = lc_exp_score(st, o, s);
        my_score += msc; op_score += osc;
        if (st->exp_n[p][s]) my_started++;
        if (st->exp_n[o][s]) op_started++;

        v[0]  = st->exp_n[p][s] ? 1.0f : 0.0f;
        v[1]  = st->exp_wager[p][s] * (1.0f / 3.0f);
        v[2]  = st->exp_n[p][s] * (1.0f / 12.0f);
        v[3]  = st->exp_sum[p][s] * (1.0f / 54.0f);
        v[4]  = mytop * (1.0f / 10.0f);
        v[5]  = msc * (1.0f / 50.0f);
        v[6]  = st->exp_n[o][s] ? 1.0f : 0.0f;
        v[7]  = st->exp_wager[o][s] * (1.0f / 3.0f);
        v[8]  = st->exp_n[o][s] * (1.0f / 12.0f);
        v[9]  = st->exp_sum[o][s] * (1.0f / 54.0f);
        v[10] = optop * (1.0f / 10.0f);
        v[11] = osc * (1.0f / 50.0f);
        v[12] = st->pile_n[s] * (1.0f / 12.0f);
        if (st->pile_n[s] > 0) {
            int tc = st->pile[s][st->pile_n[s] - 1];
            v[13] = CARD_VALUE(tc) * (1.0f / 10.0f);
            v[14] = CARD_IS_WAGER(tc) ? 1.0f : 0.0f;
        }

        int hand_cnt = 0, play_cnt = 0, play_sum = 0, hand_wag = 0;
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++) {
            if ((st->hand[p] >> c) & 1ULL) {
                hand_cnt++;
                if (CARD_IS_WAGER(c)) {
                    hand_wag++;
                    if (mytop == 0) { play_cnt++; }
                } else if (CARD_VALUE(c) > mytop) {
                    play_cnt++; play_sum += CARD_VALUE(c);
                }
            }
        }
        v[15] = hand_cnt * (1.0f / 12.0f);
        v[16] = play_cnt * (1.0f / 12.0f);
        v[17] = play_sum * (1.0f / 54.0f);
        v[18] = hand_wag * (1.0f / 3.0f);

        int uns_cnt = 0, uns_mine = 0, uns_opp = 0;
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++) {
            if ((unseen >> c) & 1ULL) {
                uns_cnt++;
                int val = CARD_VALUE(c);
                if (!CARD_IS_WAGER(c)) {
                    if (val > mytop) uns_mine += val;
                    if (val > optop) uns_opp += val;
                }
            }
        }
        v[19] = uns_cnt * (1.0f / 12.0f);
        v[20] = uns_mine * (1.0f / 54.0f);
        v[21] = uns_opp * (1.0f / 54.0f);
        /* how many more cards this expedition needs for the 8 card bonus */
        v[22] = st->exp_n[p][s] ? (8 - st->exp_n[p][s] > 0 ? (8 - st->exp_n[p][s]) * 0.125f : 0.0f) : 0.0f;
        v[23] = st->exp_n[o][s] ? (8 - st->exp_n[o][s] > 0 ? (8 - st->exp_n[o][s]) * 0.125f : 0.0f) : 0.0f;
    }

    float *g = d + NSUIT * SUIT_FEATS;
    g[0]  = st->deck_left * (1.0f / 44.0f);
    g[1]  = (st->turn == p) ? 1.0f : 0.0f;
    g[2]  = my_score * (1.0f / 50.0f);
    g[3]  = op_score * (1.0f / 50.0f);
    g[4]  = my_started * 0.2f;
    g[5]  = op_started * 0.2f;
    g[6]  = st->nply * 0.01f;
    g[7]  = st->hand_n[p] * 0.125f;
    g[8]  = st->hand_n[o] * 0.125f;
    g[9]  = 1.0f;
    g[10] = st->deck_left <= 5 ? 1.0f : 0.0f;
    g[11] = st->deck_left <= 12 ? 1.0f : 0.0f;
    /* match context: which round this is and where the match stands */
    g[12] = st->round == 0 ? 1.0f : 0.0f;
    g[13] = st->round == 1 ? 1.0f : 0.0f;
    g[14] = st->round >= 2 ? 1.0f : 0.0f;
    float cm = (float)(st->cum[p] - st->cum[o]) * 0.01f;
    if (cm > 1.5f) cm = 1.5f;
    if (cm < -1.5f) cm = -1.5f;
    g[15] = cm;

    /* --- ordered discard-pile continuation ---------------------------
     * The legacy planes/suit scalars already encode the pile set, count and
     * top.  Append the cards under that top in top-relative order so taking a
     * face-up card does not reveal information that was absent from the
     * network state.  Zero means "no card at this depth"; wagers are
     * distinguishable from absence through the second scalar. */
    float *po = d + FEAT_LEGACY_DENSE;
    for (int s = 0; s < NSUIT; s++) {
        for (int depth = 2; depth <= NRANK; depth++) {
            int pos = (int)st->pile_n[s] - depth;
            if (pos < 0) continue;
            int c = st->pile[s][pos];
            int k = ((s * FEAT_PILE_BURIED_DEPTHS) + (depth - 2))
                  * FEAT_PILE_CARD_FEATS;
            po[k] = CARD_VALUE(c) * (1.0f / 10.0f);
            po[k + 1] = CARD_IS_WAGER(c) ? 1.0f : 0.0f;
        }
    }
}
