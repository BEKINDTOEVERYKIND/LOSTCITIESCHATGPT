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

/* Expected best visible-hand finish after a complete legal move.  A face-up
 * pile draw is evaluated exactly.  A deck draw is averaged over every card
 * that can be on top from p's information set; the real hidden deck order and
 * the opponent's unknown hand are never inspected.  After the move the
 * opponent acts first, so floor(deck_left / 2) is the conservative number of
 * remaining turns credited to p. */
double hand_plan_expected_score_after_move(const State *st, int p, Move move);

/* Keep the policy's chosen card and play/discard disposition fixed, and
 * choose the best legal draw source attached to that semantic action.  Wager
 * copies of one suit are equivalent.  Ties retain the larger policy prior.
 * With one deck card left, drawing it is weakly dominant and is selected
 * directly. */
int hand_plan_choose_draw_source(const State *st, int p,
                                 const Move *mv, const float *prior,
                                 int n, int top);

#endif
