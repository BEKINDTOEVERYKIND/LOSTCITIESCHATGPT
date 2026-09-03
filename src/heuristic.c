#include "heuristic.h"
#include <math.h>

/* Projection evaluation.
 *
 * For each suit we estimate the score the expedition will finish at, assuming
 * the owner plays the useful cards they can still reach.  Our own reachable
 * cards are the playable cards in hand; the opponent's are the unseen cards,
 * discounted by the chance they hold or will draw them.  Expeditions that are
 * not yet open only count when opening them projects positive.  A global turn
 * budget stops the projection from assuming every card gets played.
 */

static float project_side(const State *st, int p, int turns, int is_me)
{
    const int o = p ^ 1;
    (void)o;
    uint64_t reach; /* cards this player may still play */
    float acquire;  /* probability the player actually gets a reachable card */

    if (is_me) {
        reach = st->hand[p];
        acquire = 1.0f;
    } else {
        uint64_t unseen = ~(st->hand[p ^ 1] | st->played[0] | st->played[1] | st->discarded
                            | st->known[p])
                          & ((1ULL << NCARD) - 1);
        /* p here is the opponent index; unseen from the other player's view,
         * plus the cards p is known for certain to hold */
        reach = unseen | st->known[p];
        int nun = __builtin_popcountll(unseen);
        float have = (float)st->hand_n[p] + 0.55f * (float)turns;
        acquire = nun > 0 ? have / (float)nun : 0.0f;
        if (acquire > 1.0f) acquire = 1.0f;
    }

    float total = 0.0f;
    float want_cards = 0.0f;
    float suit_val[NSUIT];

    for (int s = 0; s < NSUIT; s++) {
        int top = st->exp_top[p][s];
        float add_sum = 0.0f, add_cnt = 0.0f, add_wag = 0.0f;
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++) {
            if (!((reach >> c) & 1ULL)) continue;
            if (CARD_IS_WAGER(c)) {
                if (top == 0) add_wag += acquire;
            } else if (CARD_VALUE(c) > top) {
                add_sum += CARD_VALUE(c) * acquire;
                add_cnt += acquire;
            }
        }
        int started = st->exp_n[p][s] > 0;
        float mult = 1.0f + st->exp_wager[p][s];
        /* extra wagers are only worth playing when the expedition is rich */
        float proj_sum = st->exp_sum[p][s] + add_sum;
        float best = -1e9f;
        float bestcards = 0.0f;
        for (float w = 0.0f; w <= add_wag + 0.001f; w += 1.0f) {
            float m = mult + w;
            float n = st->exp_n[p][s] + add_cnt + w;
            float sc = (proj_sum - 20.0f) * m + (n >= 8.0f ? 20.0f : 0.0f);
            if (sc > best) { best = sc; bestcards = add_cnt + w; }
            if (add_wag < 1.0f) break;
        }
        if (!started && best < 0.0f) { best = 0.0f; bestcards = 0.0f; }
        if (started) {
            /* we may also stop: the floor is the current score */
            float now = (float)lc_exp_score(st, p, s);
            if (best < now) { best = now; bestcards = 0.0f; }
        }
        suit_val[s] = best;
        want_cards += bestcards;
        total += best;
    }

    /* Not every projected card can be played: scale the optimistic part back
     * to fit the remaining turns. */
    if (want_cards > turns && want_cards > 0.01f) {
        float keep = (float)turns / want_cards;
        float floorv = 0.0f;
        for (int s = 0; s < NSUIT; s++)
            floorv += st->exp_n[p][s] > 0 ? (float)lc_exp_score(st, p, s) : 0.0f;
        total = floorv + (total - floorv) * keep;
        (void)suit_val;
    }
    return total;
}

float heur_eval(const State *st, int p)
{
    int o = p ^ 1;
    int my_turns = ((int)st->deck_left + (st->turn == p ? 1 : 0)) / 2;
    int op_turns = ((int)st->deck_left + (st->turn == o ? 1 : 0)) / 2;
    if (st->over) return (float)(lc_score(st, p) - lc_score(st, o));
    float me = project_side(st, p, my_turns, 1);
    float op = project_side(st, o, op_turns, 0);
    return me - op;
}

float heur_move_value_det(const State *st, Move m)
{
    const int p = st->turn;
    State s2 = *st;
    lc_apply(&s2, m);
    if (s2.over) return (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
    return heur_eval(&s2, p);
}
