#ifndef TRAIN_TARGET_H
#define TRAIN_TARGET_H

#include "../src/search.h"
#include <math.h>

enum {
    ROLLOUT_TARGET_PRIMARY = 0,
    ROLLOUT_TARGET_COHERENT,
    ROLLOUT_TARGET_SELECTED
};

static inline int rollout_target_one_hot(const SearchStats *ss, int selected,
                                         Move *mv, float *weight,
                                         int *mode_out)
{
    for (int i = 0; i < ss->n; i++) {
        mv[i] = ss->mv[i];
        weight[i] = i == selected ? 1.0f : 0.0f;
    }
    if (mode_out) *mode_out = ROLLOUT_TARGET_SELECTED;
    return ss->n;
}

/* Convert rollout diagnostics into a policy-training target without teaching
 * a move that the deployed selection rule rejected.
 *
 * - A failed fresh prefix panel is authoritative: train the returned move.
 * - When both panels agree and that consensus move is what was played, retain
 *   useful relative-Q information by averaging their same-scale objectives.
 * - Otherwise retain the primary soft target only when its numerical leader
 *   is the move that was actually played; a gate/guard disagreement falls back
 *   to the selected move.
 *
 * The returned weights are intentionally unnormalised; train.c applies its
 * common finite/empty-distribution normaliser before storing the target. */
static inline int rollout_training_weights(const SearchStats *ss, Move played,
                                           float tau, Move *mv, float *weight,
                                           int *mode_out)
{
    if (!ss || ss->n <= 0 || ss->n > MAX_MOVES) return 0;

    int selected = -1;
    for (int i = 0; i < ss->n; i++) {
        mv[i] = ss->mv[i];
        weight[i] = 0.0f;
        if (MOVE_PACK(ss->mv[i]) == MOVE_PACK(played)) selected = i;
    }
    if (selected < 0) return 0;

    const int prefix_rejected =
        ss->prefix_confirm_worlds > 0 && !ss->prefix_confirmed;
    int limit = ss->n;
    int mode = ROLLOUT_TARGET_PRIMARY;
    double score[MAX_MOVES];

    if (ss->prefix_confirm_worlds > 0 && ss->prefix_confirmed &&
        selected == ss->selection_reference &&
        ss->trusted_candidates > 0 &&
        ss->trusted_candidates <= ss->n) {
        /* The primary and coherent panels selected this same trusted-prefix
         * leader, so their average cannot promote the rejected alternative. */
        limit = ss->trusted_candidates;
        mode = ROLLOUT_TARGET_COHERENT;
        for (int i = 0; i < limit; i++) {
            if (!lc_double_isfinite(ss->q[i]) ||
                !lc_double_isfinite(ss->prefix_q[i]))
                return rollout_target_one_hot(
                    ss, selected, mv, weight, mode_out);
            score[i] = 0.5 * (ss->q[i] + ss->prefix_q[i]);
        }
    } else {
        for (int i = 0; i < limit; i++) {
            if (!lc_double_isfinite(ss->q[i]))
                return rollout_target_one_hot(
                    ss, selected, mv, weight, mode_out);
            score[i] = ss->q[i];
        }
    }

    int leader = 0;
    for (int i = 1; i < limit; i++)
        if (score[i] > score[leader]) leader = i;
    if (prefix_rejected || leader != selected)
        return rollout_target_one_hot(ss, selected, mv, weight, mode_out);

    const int soft = tau > 0.0f && lc_float_isfinite(tau);
    const double mx = score[leader];
    for (int i = 0; i < limit; i++) {
        double w = soft ? exp((score[i] - mx) / (double)tau)
                        : (score[i] == mx ? 1.0 : 0.0);
        if (!lc_double_isfinite(w) || w < 0.0)
            return rollout_target_one_hot(
                ss, selected, mv, weight, mode_out);
        weight[i] = (float)w;
    }
    if (mode_out) *mode_out = mode;
    return ss->n;
}

#endif
