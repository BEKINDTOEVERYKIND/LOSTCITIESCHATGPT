/* history_belief_model.h -- causal opponent-action hand belief model.
 *
 * This component is deliberately separate from Net and every playing actor.
 * It consumes only an observer's scrubbed information view plus a transcript
 * of public opponent actions.  Candidate-card scores feed the same exact-K
 * exponential-family objective used by the deployed belief head, but no
 * policy, value, rollout, or match-strength parameter is touched.
 */
#ifndef HISTORY_BELIEF_MODEL_H
#define HISTORY_BELIEF_MODEL_H

#include "agent.h"
#include "lc.h"
#include <stdint.h>

#define HISTORY_BELIEF_VERSION 1U
#define HISTORY_BELIEF_BASE_ALPHA 1.15f
#define HISTORY_BELIEF_RANKS 10
#define HISTORY_BELIEF_PHASES 6
#define HISTORY_BELIEF_AGES 17
#define HISTORY_BELIEF_DRAW_RANKS 11
#define HISTORY_BELIEF_NO_DRAW_RANK 10
#define HISTORY_BELIEF_MAX_EVENTS LC_MAX_PLIES
#define HISTORY_BELIEF_EXCLUSION_SHA256_BYTES 32

/* Every learnable feature is anchored to an opponent action.  A trace keeps
 * all public actions so order and the immediately preceding public context
 * are available, but an own-action-only trace produces a zero residual.  The
 * last age bucket pools opponent actions older than the most recent 16 public
 * events. */
#define HISTORY_BELIEF_PAIR_FEATURES \
    (HISTORY_BELIEF_AGES * 2 * 3 * 2 * \
     HISTORY_BELIEF_RANKS * HISTORY_BELIEF_RANKS)
#define HISTORY_BELIEF_DRAW_FEATURES \
    (HISTORY_BELIEF_AGES * 3 * 2 * HISTORY_BELIEF_RANKS * \
     HISTORY_BELIEF_DRAW_RANKS)
#define HISTORY_BELIEF_EVENT_PHASE_FEATURES \
    (HISTORY_BELIEF_PHASES * HISTORY_BELIEF_PHASES * 2 * 2 * \
     HISTORY_BELIEF_RANKS * HISTORY_BELIEF_RANKS)
#define HISTORY_BELIEF_CONTEXT_FEATURES \
    (HISTORY_BELIEF_AGES * 4 * 2 * 2 * HISTORY_BELIEF_RANKS * \
     HISTORY_BELIEF_RANKS * HISTORY_BELIEF_RANKS)
#define HISTORY_BELIEF_FEATURES \
    (HISTORY_BELIEF_PAIR_FEATURES + HISTORY_BELIEF_DRAW_FEATURES + \
     HISTORY_BELIEF_EVENT_PHASE_FEATURES + \
     HISTORY_BELIEF_CONTEXT_FEATURES)
#define HISTORY_BELIEF_MAX_ACTIVE (4 * HISTORY_BELIEF_MAX_EVENTS)

typedef struct {
    uint16_t ply;
    uint8_t suit;
    uint8_t rank;       /* 0 wager, 1..9 represent values 2..10 */
    uint8_t discard;
    uint8_t draw;       /* 0 deck, 1..5 public discard pile */
    uint8_t draw_rank;  /* semantic public pile card, or NO_DRAW_RANK */
    uint8_t opponent;   /* relative to trace observer */
    uint8_t deck_left;  /* public action-time context */
} HistoryBeliefEvent;

typedef struct {
    uint8_t observer;
    uint16_t n;
    HistoryBeliefEvent event[HISTORY_BELIEF_MAX_EVENTS];
} HistoryBeliefTrace;

typedef struct {
    uint32_t version;
    uint32_t training_symmetries;
    uint64_t actor_fingerprint;
    uint64_t base_net_fingerprint;
    uint64_t train_seed;
    uint64_t train_states;
    float training_temperature;
    float base_alpha;
    unsigned char exclusion_sha256[HISTORY_BELIEF_EXCLUSION_SHA256_BYTES];
    float weight[HISTORY_BELIEF_FEATURES];
} HistoryBeliefModel;

typedef struct {
    BeliefDist base;
    float logit[NCARD];
    float marginal[NCARD];
} HistoryBeliefDist;

typedef struct {
    const HistoryBeliefModel *model;
    const Net *actor;
    const Net *base_net;
    uint64_t actor_fingerprint;
    uint64_t base_net_fingerprint;
    uint32_t symmetries;
    float temperature;
    float base_alpha;
} HistoryBeliefRuntime;

void history_belief_model_init(HistoryBeliefModel *model);
int history_belief_model_bind(HistoryBeliefModel *model, const Net *actor,
                              const Net *base_net,
                              int training_symmetries,
                              float training_temperature,
                              float base_alpha,
                              const unsigned char exclusion_sha256[
                                  HISTORY_BELIEF_EXCLUSION_SHA256_BYTES]);
int history_belief_model_validate(const HistoryBeliefModel *model);
int history_belief_model_compatible(const HistoryBeliefModel *model,
                                    const Net *actor,
                                    const Net *base_net,
                                    int training_symmetries,
                                    float training_temperature,
                                    float base_alpha,
                                    const unsigned char exclusion_sha256[
                                        HISTORY_BELIEF_EXCLUSION_SHA256_BYTES]);
uint64_t history_belief_actor_fingerprint(const Net *actor);

void history_belief_trace_init(HistoryBeliefTrace *trace, int observer);
/* Record public move fields for every ply so action-time chronology is
 * reconstructible.  Learned residual features are nevertheless emitted only
 * for opponent events.  state_before may be complete because this function
 * is forbidden from reading either hand or any hidden deck card. */
int history_belief_trace_push(HistoryBeliefTrace *trace,
                              const State *state_before, Move move);
int history_belief_trace_opponent_actions(const HistoryBeliefTrace *trace);

/* Return the sparse, suit-equivariant feature indices for one uncertain
 * candidate card.  view must be produced by agent_information_view() for the
 * same observer; complete hidden states fail closed. */
int history_belief_active_features(const HistoryBeliefTrace *trace,
                                   const State *view, int observer,
                                   uint8_t candidate,
                                   uint32_t out[HISTORY_BELIEF_MAX_ACTIVE]);

int history_belief_logits(const HistoryBeliefModel *model,
                          const HistoryBeliefTrace *trace,
                          const State *view, int observer,
                          const uint8_t *card, const float *base_log_weight,
                          int n, float *logit);

/* Exact-K exponential family for already calibrated log weights.  Unlike the
 * generic raw-logit helper, this log-space DP does not recenter or clip, and
 * its gradient is marginal-minus-label everywhere.  The high-level runtime
 * reconstructs the frozen base log weights from stored double weights through
 * float logits, so zero-residual parity is numerical (locked to 2e-6 in the
 * native test), not a claim of bitwise identity. */
int history_belief_exact_k_eval(const float *log_weight,
                                const uint8_t *held,
                                int n, int need,
                                float *marginal, double *nll);

/* Exact derivative through belief_exact_k_eval()'s mean-centering and
 * deterministic [-20,20] clamp.  Points exactly on a clamp boundary use the
 * zero subgradient. */
int history_belief_clipped_raw_gradient(const float *raw_logit,
                                        const float *marginal,
                                        const uint8_t *held,
                                        int n, float alpha,
                                        float *gradient);

/* High-level causal inference entry point.  It derives the frozen base-net
 * exact-K log weights internally and rejects provenance or trace mismatches. */
int history_belief_runtime_init(
    HistoryBeliefRuntime *runtime,
    const HistoryBeliefModel *model,
    const Net *actor, const Net *base_net,
    int symmetries, float temperature, float base_alpha,
    const unsigned char exclusion_sha256[
        HISTORY_BELIEF_EXCLUSION_SHA256_BYTES]);
int history_belief_dist_init(const HistoryBeliefRuntime *runtime,
                             const HistoryBeliefTrace *trace,
                             const State *view, int observer,
                             HistoryBeliefDist *dist);

/* Canonical little-endian float32 artifact. */
int history_belief_model_save(const HistoryBeliefModel *model,
                              const char *path);
int history_belief_model_load(HistoryBeliefModel *model,
                              const char *path);
uint64_t history_belief_model_fingerprint(const HistoryBeliefModel *model);

#endif
