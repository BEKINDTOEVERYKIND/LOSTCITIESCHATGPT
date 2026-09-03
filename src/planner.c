#include "planner.h"
#include <limits.h>
#include <math.h>
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

double hand_plan_expected_score_after_move(const State *st, int p, Move move)
{
    State played = *st;
    lc_apply_play(&played, move);

    if (move.draw != 0) {
        lc_apply_draw(&played, move, -1);
        HandPlan plan;
        hand_plan_build(&played, p, played.deck_left / 2, &plan);
        return (double)plan.score;
    }

    /* From p's information set, each unpinned card is equally likely to be
     * the next deck card under the uniform fixed-cardinality world model.
     * Enumerating that support is exact and, unlike applying the real deck
     * card from a determinization, cannot leak hidden information into a
     * downstream action. */
    uint8_t unseen[NCARD];
    int nunseen = 0;
    lc_unseen(&played, p, unseen, &nunseen);
    if (nunseen <= 0) {
        HandPlan plan;
        hand_plan_build(&played, p, 0, &plan);
        return (double)plan.score;
    }

    /* Enumerate subsets of the seven post-play cards once, then attach each
     * possible draw.  Re-running the eight-card scheduler independently for
     * every unseen card is equivalent but roughly an order of magnitude more
     * expensive inside a many-world rollout. */
    uint8_t hand[HAND_SIZE];
    int nhand = lc_hand_cards(&played, p, hand);
    int turns = played.deck_left > 0 ? (played.deck_left - 1) / 2 : 0;
    if (turns > nhand + 1) turns = nhand + 1;
    const unsigned limit = 1u << nhand;
    int subset_score[1u << HAND_SIZE];
    int subset_n[1u << HAND_SIZE][NSUIT];
    int subset_wager[1u << HAND_SIZE][NSUIT];
    int subset_sum[1u << HAND_SIZE][NSUIT];
    unsigned char subset_valid[1u << HAND_SIZE];
    int best_without = lc_score(&played, p);

    for (unsigned subset = 0; subset < limit; subset++) {
        int count = __builtin_popcount(subset);
        int valid = count <= turns;
        for (int s = 0; s < NSUIT; s++) {
            subset_n[subset][s] = played.exp_n[p][s];
            subset_wager[subset][s] = played.exp_wager[p][s];
            subset_sum[subset][s] = played.exp_sum[p][s];
        }
        for (int i = 0; i < nhand && valid; i++) {
            if (!((subset >> i) & 1u)) continue;
            int c = hand[i], s = CARD_SUIT(c);
            if ((CARD_IS_WAGER(c) && played.exp_top[p][s] != 0) ||
                (!CARD_IS_WAGER(c) &&
                 CARD_VALUE(c) <= played.exp_top[p][s])) {
                valid = 0;
                break;
            }
            subset_n[subset][s]++;
            if (CARD_IS_WAGER(c)) subset_wager[subset][s]++;
            else subset_sum[subset][s] += CARD_VALUE(c);
        }
        subset_valid[subset] = (unsigned char)valid;
        int score = 0;
        if (valid) {
            for (int s = 0; s < NSUIT; s++)
                score += expedition_score(subset_n[subset][s],
                                          subset_wager[subset][s],
                                          subset_sum[subset][s]);
            if (score > best_without) best_without = score;
        }
        subset_score[subset] = score;
    }

    double total = 0.0;
    for (int i = 0; i < nunseen; i++) {
        int c = unseen[i], s = CARD_SUIT(c);
        int best = best_without;
        if ((CARD_IS_WAGER(c) && played.exp_top[p][s] == 0) ||
            (!CARD_IS_WAGER(c) &&
             CARD_VALUE(c) > played.exp_top[p][s])) {
            for (unsigned subset = 0; subset < limit; subset++) {
                if (!subset_valid[subset] ||
                    __builtin_popcount(subset) >= turns)
                    continue;
                int before = expedition_score(subset_n[subset][s],
                                              subset_wager[subset][s],
                                              subset_sum[subset][s]);
                int after = expedition_score(
                    subset_n[subset][s] + 1,
                    subset_wager[subset][s] + CARD_IS_WAGER(c),
                    subset_sum[subset][s] +
                        (CARD_IS_WAGER(c) ? 0 : CARD_VALUE(c)));
                int score = subset_score[subset] - before + after;
                if (score > best) best = score;
            }
        }
        total += (double)best;
    }
    return total / (double)nunseen;
}

static int same_semantic_action(Move a, Move b)
{
    if (a.discard != b.discard) return 0;
    if (CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card))
        return CARD_SUIT(a.card) == CARD_SUIT(b.card);
    return a.card == b.card;
}

int hand_plan_choose_draw_source(const State *st, int p,
                                 const Move *mv, const float *prior,
                                 int n, int top)
{
    if (n <= 1 || top < 0 || top >= n) return top;
    if (st->deck_left == 1) {
        for (int i = 0; i < n; i++)
            if (mv[i].draw == 0 &&
                same_semantic_action(mv[i], mv[top]))
                return i;
    }

    int best = top;
    double best_score = hand_plan_expected_score_after_move(st, p, mv[top]);
    for (int i = 0; i < n; i++) {
        if (i == top || !same_semantic_action(mv[i], mv[top])) continue;
        double score = hand_plan_expected_score_after_move(st, p, mv[i]);
        if (score > best_score + 1e-12 ||
            (fabs(score - best_score) <= 1e-12 && prior[i] > prior[best])) {
            best = i;
            best_score = score;
        }
    }
    return best;
}
