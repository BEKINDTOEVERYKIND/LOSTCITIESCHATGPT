/* net.h -- value, policy, and belief network with a sparse input layer.
 *
 * Input  : FEAT_DIM features (FEAT_BIN sparse binary + FEAT_DENSE scalars)
 * Trunk  : NET_H1 -> NET_H2, ReLU
 * Value  : one scalar continuation return for the perspective player, in
 *          units of VAL_SCALE. Finishing checkpoints use
 *          0.05 * match margin + 50 * signed match result.
 * Policy : a move is (card, play-or-discard, draw source).  The head retains
 *          the sample-efficient additive decomposition and adds a learned
 *          residual for the complete combination:
 *              ip = card*2 + disc
 *              ic = ip*NET_NDRAW + src
 *              logit(m) = play[ip] + draw[src] + combination[ic]
 *          The combination term lets draw preference depend on the card and
 *          disposition while the additive terms still share statistics across
 *          related moves.  The softmax runs over the legal moves only.
 *
 * The policy head exists because ranking moves by a value function alone is
 * hopeless here: candidate moves differ by one or two points, far below the
 * accuracy any regression on game outcomes can reach.  Predicting the choice
 * directly sidesteps that.
 *
 * Belief : one logit per card, trained to predict whether the opponent holds
 *          it (the true hand is known when self-play data is generated, so the
 *          labels are free).  This is where behavioural inference lives: the
 *          trunk sees what the opponent has committed to and thrown away, and
 *          the head learns what that implies about the cards they kept.  The
 *          determinized search samples opponent hands from this posterior
 *          instead of uniformly.
 */
#ifndef NET_H
#define NET_H

#include "features.h"

#define NET_H1 512
#define NET_H2 256
#define NET_NPLAY (NCARD * 2)   /* card x (play|discard) */
#define NET_NDRAW (NSUIT + 1)   /* deck or one of five piles */
#define NET_NCOMB (NET_NPLAY * NET_NDRAW)
#define VAL_SCALE 50.0f

typedef struct {
    /* Rows 0..FEAT_LEGACY_DIM-1 retain the historical layout.  Ordered-pile
     * rows are appended so upgraded v3-v5 nets can preserve exact outputs by
     * loading the old rows and zeroing only the additions. */
    float w1[FEAT_DIM][NET_H1];
    float b1[NET_H1];
    float w2[NET_H1][NET_H2];
    float b2[NET_H2];
    float w3[NET_H2];                  /* value head  */
    float b3;
    float wplay[NET_NPLAY][NET_H2];    /* policy: which card, and whether played */
    float bplay[NET_NPLAY];
    float wdraw[NET_NDRAW][NET_H2];    /* policy: where to draw from */
    float bdraw[NET_NDRAW];
    /* belief head (the final section of v4 model files) */
    float wbel[NCARD][NET_H2];
    float bbel[NCARD];
    /* Full-move interaction residual.  Appended for v5 file compatibility. */
    float wcomb[NET_NCOMB][NET_H2];
    float bcomb[NET_NCOMB];
} Net;

typedef struct {
    float a1[NET_H1];
    float a2[NET_H2];
} NetAct;

typedef struct {
    Net m, v;
    long t;
} Adam;

void  net_init(Net *n, uint64_t seed);
void  net_zero(Net *n);
/* Project every parameter that refers to one physical wager copy onto the
 * exact three-copy symmetry.  The cards are indistinguishable under the
 * rules; tying these rows removes arbitrary ID-specific behaviour. */
void  net_project_wager_symmetry(Net *n);
/* Convert ordinary per-row gradients into the gradient of the tied wager
 * parameter: sum the three copies, then give every copy that same update. */
void  net_tie_wager_gradients(Net *g);

/* trunk only; fills act */
void  net_trunk(const Net *n, const Features *f, NetAct *act);
float net_value_act(const Net *n, const NetAct *act);
/* logits for the given packed moves */
void  net_policy_act(const Net *n, const NetAct *act, const uint16_t *mv, int nmv, float *logits);
/* belief logits (opponent holds card?) for the given card ids */
void  net_belief_act(const Net *n, const NetAct *act, const uint8_t *cards, int nc, float *logits);

/* convenience: trunk + value */
float net_value(const Net *n, const Features *f);

/* Accumulate gradients for one sample.  dvalue is dLoss/dvalue_output;
 * dlogit[i] is dLoss/dlogit for move mv[i]; dbel[i] likewise for belief card
 * bc[i].  Either head may be skipped by passing NULL. */
void  net_backward(const Net *n, const Features *f, const NetAct *act,
                   float dvalue, const uint16_t *mv, const float *dlogit, int nmv,
                   const uint8_t *bc, const float *dbel, int nb,
                   Net *g);
void  net_adam_step(Net *n, const Net *g, Adam *a, float lr, float scale, float wd);
int   net_save(const Net *n, const char *path);
int   net_load(Net *n, const char *path);

#endif
