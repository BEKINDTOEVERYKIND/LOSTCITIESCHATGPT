/* policy_cost.h -- calibrated policy-frequency arbitration for rollout roots.
 *
 * A policy-cost table turns a root policy probability into a deterministic
 * opportunity cost in the same units as the rollout objective.  The cost is
 * split into a semantic card/action term and a conditional draw-source term:
 *
 *   score(m) = beta * Q(m)
 *              + alpha_action * log(P_action(m))
 *              + alpha_draw * log(P(m) / P_action(m))
 *
 * beta is strictly positive, so deployment divides the whole scalar by the
 * interpolated beta and uses the equivalent opportunity cost
 *
 *   cost(m) = -(alpha_action / beta) * log(P_action(m))
 *             -(alpha_draw / beta) * log(P(m) / P_action(m)).
 *
 * Crucially beta and both alphas are interpolated before either ratio is
 * formed.  Ranking Q(m)-cost(m) is one scalar potential, so pairwise
 * thresholds telescope and cannot create preference cycles.  The historical practical-
 * effect hurdle remains a distinct legacy rollout mechanism; v1 policy-cost
 * actors bind its threshold to canonical +0 and never fold it into this
 * trusted selection.
 *
 * Artifacts are canonical, content-hashed, and bound to the exact root and
 * continuation checkpoints plus the rollout settings that generated their
 * calibration corpus.  A stale or partially written artifact fails closed.
 */
#ifndef POLICY_COST_V13_H
#define POLICY_COST_V13_H

#include "lc.h"
#include "net.h"
#include <stdint.h>

#define POLICY_COST_LEGACY_VERSION 1U
#define POLICY_COST_V3_VERSION 3U
#define POLICY_COST_V13_VERSION 4U
#define POLICY_COST_VERSION 5U
#define POLICY_COST_CONTROLLER_ABI 1U
#define POLICY_COST_ANCHORS 10
#define POLICY_COST_MAX_CANDIDATES 8
#define POLICY_COST_PRIMARY_Z 3.5
#define POLICY_COST_FRESH_Z 2.58
#define POLICY_COST_LEGACY_SOURCE_SEED UINT64_C(202611140101)
#define POLICY_COST_V2_SOURCE_SEED UINT64_C(202612140101)
#define POLICY_COST_V3_SOURCE_SEED UINT64_C(202701140101)
#define POLICY_COST_V13_SOURCE_SEED UINT64_C(202702140101)
#define POLICY_COST_SOURCE_SEED UINT64_C(202712140101)
/* One half of the least positive binary32 value.  The policy vector is
 * binary32, so every strictly positive policy mass clears this exact
 * binary64 domain assertion while an exact zero never enters log(). */
#define POLICY_COST_EPSILON 0x1p-150

struct Agent;

typedef struct {
    uint64_t root_net_fingerprint;
    uint64_t continuation_net_fingerprint;
    /* Zero for objectives 0-2.  Objective 3 requires the fingerprint of the
     * independently content-checked match-value table. */
    uint64_t match_value_fingerprint;
    uint32_t controller_abi;
    uint32_t build_profile;
    uint32_t objective;
    uint32_t root_symmetries;
    uint32_t playout_symmetries;
    uint32_t playout_sample;
    uint32_t playout_prune;
    uint32_t exact_terminal;
    uint32_t no_belief;
    uint32_t dets;
    uint32_t confirm_dets;
    uint32_t root_width;
    uint32_t action_core_count;
    uint32_t min_cand;
    uint32_t ply_lo;
    uint32_t ply_hi;
    uint32_t discard_guard;
    uint32_t root_prune;
    /* These originate in the rollout tail as binary32.  Persist their exact
     * bits rather than comparing decimal spellings after widening. */
    float cand_floor;
    float override_k;
    float override_min;
} PolicyCostController;

typedef struct PolicyCostTable {
    uint32_t version;
    uint64_t source_seed;
    uint64_t payload_fingerprint;
    double epsilon;
    double primary_z;
    double fresh_z;
    uint32_t ply_anchor[POLICY_COST_ANCHORS];
    PolicyCostController controller;
    /* One round-shared predictive schedule.  beta and both alpha arrays are
     * linearly interpolated independently at the integer pre-move State.nply
     * through ply 64, then held constant; State.round remains diagnostic
     * only.  The aliases retain source compatibility for legacy-v1 table
     * construction; in v1 the persisted lambdas map to alpha with beta=1. */
    double beta[POLICY_COST_ANCHORS];
    union {
        double alpha_action[POLICY_COST_ANCHORS];
        double lambda_action[POLICY_COST_ANCHORS];
    };
    union {
        double alpha_draw[POLICY_COST_ANCHORS];
        double lambda_draw[POLICY_COST_ANCHORS];
    };
} PolicyCostTable;

typedef struct {
    int leader;             /* numerical leader of the adjusted scalar */
    int selected;           /* leader after all-pair gates, else zero   */
    int anchor_interval;
    int prior_protected_rivals;
    int all_pair_passed;
    double beta;
    double alpha_action;
    double alpha_draw;
    /* Normalized alpha/beta coefficients actually applied to Q. */
    double lambda_action;
    double lambda_draw;
    double q[POLICY_COST_MAX_CANDIDATES];
    double semantic_prior[POLICY_COST_MAX_CANDIDATES];
    double conditional_draw_prior[POLICY_COST_MAX_CANDIDATES];
    double cost[POLICY_COST_MAX_CANDIDATES];
    double adjusted_q[POLICY_COST_MAX_CANDIDATES];
    double pair_delta[POLICY_COST_MAX_CANDIDATES];
    double pair_se[POLICY_COST_MAX_CANDIDATES];
} PolicyCostDecision;

int policy_cost_controller_valid(const PolicyCostController *controller);
int policy_cost_validate(const PolicyCostTable *table);
/* Exact deployment-profile binding used by both the actor parser and the
 * runtime.  The runtime check matters for programmatically constructed
 * Agents: attaching a valid table must not silently compose it with an
 * uncalibrated planner, resolver, ranker, veto, or pruned root policy. */
int policy_cost_matches_agent(const struct Agent *agent);
/* Return the independently interpolated predictive coefficients. */
int policy_cost_coefficients(const PolicyCostTable *table, int nply,
                             double *beta, double *alpha_action,
                             double *alpha_draw, int *anchor_interval);
/* Return normalized alpha/beta costs.  This is equivalent to dividing the
 * complete predictive score and its uncertainty by positive beta. */
int policy_cost_schedule(const PolicyCostTable *table, int nply,
                         double *lambda_action, double *lambda_draw,
                         int *anchor_interval);

/* Canonical little-endian, binary64 persistence.  Save is no-clobber. */
int policy_cost_save(const PolicyCostTable *table, const char *path);
PolicyCostTable *policy_cost_load(const char *path, int *error);
void policy_cost_free(PolicyCostTable *table);

/* Return the two policy factors and resulting deterministic cost for one
 * complete move.  `prob` must be the already suit-symmetrized joint policy
 * over the complete legal-move list; aggregation happens after symmetry. */
int policy_cost_move_terms(const PolicyCostTable *table, int round, int nply,
                           const Move *move, const float *prob, int n,
                           int index, double *semantic_prior,
                           double *conditional_draw_prior, double *cost);

/* Summary-form authority for frozen offline SELECT/TEST panels.  q contains
 * candidate means and pair_se is a row-major symmetric ncand-by-ncand (or
 * wider, via pair_stride) matrix of paired standard errors.  This is the one
 * implementation of scalar leadership, all-pair evidence, and directed raw
 * protection; the raw-world entry point below reduces to this function. */
int policy_cost_decide_summary(
    const PolicyCostTable *table, int round, int nply,
    const Move *legal_move, const float *legal_prob, int nlegal,
    const int *candidate_index, int ncand, const double *q,
    const double *pair_se, int pair_stride, double z,
    PolicyCostDecision *decision);

/* Select from a paired fixed-world panel.  values is candidate-major, with
 * `stride` entries reserved per candidate and `worlds` initialized entries.
 * The adjusted leader must beat every admitted rival by z paired SEs.  Its
 * raw paired mean must additionally be positive versus literal candidate
 * zero, every rival with higher semantic-action or complete-semantic-move
 * mass, and a same-core rival with higher conditional draw probability.  Any
 * malformed or ambiguous panel returns candidate zero. */
int policy_cost_decide(const PolicyCostTable *table, int round, int nply,
                       const Move *legal_move, const float *legal_prob,
                       int nlegal, const int *candidate_index, int ncand,
                       const double *values, int stride, int worlds, double z,
                       PolicyCostDecision *decision);

#endif
