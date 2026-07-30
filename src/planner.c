#include "planner.h"
#include <limits.h>
#include <string.h>

static int expedition_score(int n, int wagers, int sum)
{
    if (n == 0) return 0;
    int score = (sum - 20) * (1 + wagers);
    if (n >= 8) score += 20;
    return score;
}

void hand_plan_build(const State *st, int p, int turns, HandPlan *out)
{
    memset(out, 0, sizeof *out);
    out->base_score = lc_score(st, p);
    out->score = out->base_score;
    out->min_cards = 0;

    uint8_t card[HAND_SIZE];
    int ncard = lc_hand_cards(st, p, card);
    if (turns < 0) turns = 0;
    if (turns > ncard) turns = ncard;
    out->turns = turns;

    const unsigned limit = 1u << ncard;
    for (unsigned subset = 0; subset < limit; subset++) {
        if (__builtin_popcount(subset) > turns) continue;

        int n[NSUIT], wagers[NSUIT], sum[NSUIT];
        for (int s = 0; s < NSUIT; s++) {
            n[s] = st->exp_n[p][s];
            wagers[s] = st->exp_wager[p][s];
            sum[s] = st->exp_sum[p][s];
        }

        int valid = 1;
        for (int i = 0; i < ncard && valid; i++) {
            if (!((subset >> i) & 1u)) continue;
            int c = card[i], s = CARD_SUIT(c);
            if (CARD_IS_WAGER(c)) {
                if (st->exp_top[p][s] != 0) valid = 0;
                else {
                    wagers[s]++;
                    n[s]++;
                }
            } else {
                int v = CARD_VALUE(c);
                if (v <= st->exp_top[p][s]) valid = 0;
                else {
                    sum[s] += v;
                    n[s]++;
                }
            }
        }
        if (!valid) continue;

        int score = 0;
        for (int s = 0; s < NSUIT; s++)
            score += expedition_score(n[s], wagers[s], sum[s]);
        if (score < out->score) continue;
        int selected_count = __builtin_popcount(subset);

        uint64_t first = 0;
        for (int s = 0; s < NSUIT; s++) {
            uint64_t selected_wagers = 0;
            int lowest_value = INT_MAX, lowest_card = -1;
            for (int i = 0; i < ncard; i++) {
                if (!((subset >> i) & 1u) || CARD_SUIT(card[i]) != s)
                    continue;
                if (CARD_IS_WAGER(card[i]))
                    selected_wagers |= 1ULL << card[i];
                else if (CARD_VALUE(card[i]) < lowest_value) {
                    lowest_value = CARD_VALUE(card[i]);
                    lowest_card = card[i];
                }
            }
            if (selected_wagers)
                first |= selected_wagers;
            else if (lowest_card >= 0)
                first |= 1ULL << lowest_card;
        }

        if (score > out->score ||
            (score == out->score && selected_count < out->min_cards)) {
            out->score = score;
            out->min_cards = selected_count;
            out->first_cards = first;
            out->used_cards = 0;
            for (int i = 0; i < ncard; i++)
                if ((subset >> i) & 1u)
                    out->used_cards |= 1ULL << card[i];
        } else if (selected_count == out->min_cards) {
            out->first_cards |= first;
            for (int i = 0; i < ncard; i++)
                if ((subset >> i) & 1u)
                    out->used_cards |= 1ULL << card[i];
        }
    }

    if (out->score <= out->base_score) {
        out->first_cards = 0;
        out->used_cards = 0;
        out->min_cards = 0;
    }
}

int hand_plan_block_cost(const State *st, int p, int card)
{
    if (CARD_IS_WAGER(card)) return 0;
    const int suit = CARD_SUIT(card);
    const int top = st->exp_top[p][suit];
    const int value = CARD_VALUE(card);
    const int mult = 1 + st->exp_wager[p][suit];
    uint8_t unseen[NCARD];
    int n = 0, cost = 0;
    lc_unseen(st, p, unseen, &n);
    for (int i = 0; i < n; i++) {
        if (CARD_SUIT(unseen[i]) != suit || CARD_IS_WAGER(unseen[i]))
            continue;
        int v = CARD_VALUE(unseen[i]);
        if (v > top && v < value) cost += v * mult;
    }
    return cost;
}

static int immediate_gain(const State *st, int p, int card)
{
    int s = CARD_SUIT(card);
    int before = lc_exp_score(st, p, s);
    int n = st->exp_n[p][s] + 1;
    int wagers = st->exp_wager[p][s] + (CARD_IS_WAGER(card) ? 1 : 0);
    int sum = st->exp_sum[p][s] + (CARD_IS_WAGER(card) ? 0 : CARD_VALUE(card));
    return expedition_score(n, wagers, sum) - before;
}

int hand_plan_choose(const State *st, int p, const Move *mv, const float *prob,
                     const int *order, int norder, int turns)
{
    HandPlan plan;
    hand_plan_build(st, p, turns, &plan);
    if (!plan.first_cards) return -1;

    int best = -1, best_block = INT_MAX, best_gain = INT_MIN;
    for (int k = 0; k < norder; k++) {
        int i = order[k];
        if (mv[i].discard || mv[i].draw != 0 ||
            !((plan.first_cards >> mv[i].card) & 1ULL))
            continue;
        int block = hand_plan_block_cost(st, p, mv[i].card);
        int gain = immediate_gain(st, p, mv[i].card);
        if (best < 0 || block < best_block ||
            (block == best_block && gain > best_gain) ||
            (block == best_block && gain == best_gain &&
             prob[i] > prob[best])) {
            best = i;
            best_block = block;
            best_gain = gain;
        }
    }
    return best;
}

int hand_plan_score_after_play(const State *st, int p, int card, int turns)
{
    State after = *st;
    int suit = CARD_SUIT(card);
    after.hand[p] &= ~(1ULL << card);
    if (after.hand_n[p] > 0) after.hand_n[p]--;
    after.played[p] |= 1ULL << card;
    after.exp_n[p][suit]++;
    if (CARD_IS_WAGER(card)) {
        after.exp_wager[p][suit]++;
    } else {
        after.exp_top[p][suit] = (uint8_t)CARD_VALUE(card);
        after.exp_sum[p][suit] += (uint8_t)CARD_VALUE(card);
    }
    HandPlan spent;
    hand_plan_build(&after, p, turns > 0 ? turns - 1 : 0, &spent);
    return spent.score;
}

int hand_plan_conservative_choose(
    const State *st, int p, const Move *mv, const float *prob,
    const int *order, int norder, int turns, int minimum_block_reduction)
{
    if (norder <= 0 || minimum_block_reduction <= 0) return -1;
    HandPlan plan;
    hand_plan_build(st, p, turns, &plan);
    int top = order[0];
    if (mv[top].discard || mv[top].draw != 0)
        return -1;
    int pick = hand_plan_choose(st, p, mv, prob, order, norder, turns);
    if (pick < 0) return -1;

    /* If the policy leader cannot begin any best guaranteed schedule,
     * compare the visible-hand result after spending a turn on it.  This
     * catches commitments such as a third wager that lowers the best score
     * already secured by the current hand.  The simulated state deliberately
     * receives no deck card: neither side of this comparison may inspect an
     * unknown draw. */
    if (!((plan.first_cards >> mv[top].card) & 1ULL)) {
        int spent_score =
            hand_plan_score_after_play(st, p, mv[top].card, turns);
        if (plan.score - spent_score >= 2) return pick;
        return -1;
    }

    int reduction = hand_plan_block_cost(st, p, mv[top].card)
                  - hand_plan_block_cost(st, p, mv[pick].card);
    return reduction >= minimum_block_reduction ? pick : -1;
}
