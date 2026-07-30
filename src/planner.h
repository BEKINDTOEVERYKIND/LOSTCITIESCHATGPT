/* planner.h -- exact scheduling of the cards already in one player's hand.
 *
 * This is deliberately information-set safe: it never reads the opponent's
 * hand or the deck order.  It enumerates the current hand and computes the
 * best expedition score reachable within a fixed number of own turns.
 */
#ifndef PLANNER_H
#define PLANNER_H

#include "lc.h"

typedef struct {
    int score;
    int base_score;
    int turns;
    int min_cards;
    uint64_t first_cards;
    uint64_t used_cards;
} HandPlan;

void hand_plan_build(const State *st, int p, int turns, HandPlan *out);

/* order[] contains policy-ranked indices into mv/prob.  Only deck-draw plays
 * in that short prefix are eligible; discards and pile pickups stay under the
 * rollout evaluator.  Returns an mv[] index, or -1 when no improving plan is
 * represented in the prefix. */
int hand_plan_choose(const State *st, int p, const Move *mv, const float *prob,
                     const int *order, int norder, int turns);

/* Guaranteed visible-hand score after playing card now (without peeking at or
 * crediting the unknown draw), then optimally scheduling turns-1 more cards. */
int hand_plan_score_after_play(const State *st, int p, int card, int turns);

/* Conservative production gate.  When the policy leader begins an optimal
 * guaranteed schedule, replacement requires at least minimum_block_reduction
 * more unseen-card preservation.  A leader outside every best schedule may
 * instead be replaced when spending the turn on it has at least two points of
 * guaranteed visible-hand regret.  This avoids turning the scheduler into a
 * general heuristic policy. */
int hand_plan_conservative_choose(
    const State *st, int p, const Move *mv, const float *prob,
    const int *order, int norder, int turns, int minimum_block_reduction);

/* Cost of closing off still-unseen, lower number cards by playing card now. */
int hand_plan_block_cost(const State *st, int p, int card);

#endif
