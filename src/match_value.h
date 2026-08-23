/* match_value.h -- round-boundary continuation values for three-round play.
 *
 * A fresh Lost Cities round carries no cards or public history forward.  For
 * a frozen, information-safe continuation controller, its expected remaining
 * match return is therefore a function only of the next round, the cumulative
 * score lead, and whether the evaluated player starts that next round.  This
 * module stores that small Bellman table in a canonical, checked file format.
 *
 * The table is deliberately separate from Net checkpoints.  Its raw variant
 * evaluates one exact downstream controller; its explicitly flagged isotonic
 * variant regularizes that estimate toward optimal-value structure.  Both
 * record the controller model, ABI, build profile, and behavioral settings;
 * parsers reject mismatches rather than applying stale values to a new policy.
 */
#ifndef MATCH_VALUE_H
#define MATCH_VALUE_H

#include "lc.h"
#include "net.h"
#include <stdint.h>

#define MATCH_VALUE_VERSION 1U
/* Bump whenever frozen playout semantics change, even if Agent fields and
 * network bytes do not.  A stale transition table must then fail closed. */
#define MATCH_VALUE_CONTROLLER_ABI 1U
#define MATCH_VALUE_POLICY_LEAD_LIMIT 150
#define MATCH_VALUE_ROUND_MARGIN_LIMIT 1180
#define MATCH_VALUE_R1_LEAD_LIMIT MATCH_VALUE_ROUND_MARGIN_LIMIT
#define MATCH_VALUE_R2_LEAD_LIMIT (2 * MATCH_VALUE_ROUND_MARGIN_LIMIT)
#define MATCH_VALUE_R1_COUNT (2 * MATCH_VALUE_R1_LEAD_LIMIT + 1)
#define MATCH_VALUE_R2_COUNT (2 * MATCH_VALUE_R2_LEAD_LIMIT + 1)
#define MATCH_VALUE_MAX_MATCH_MARGIN \
    (MATCH_ROUNDS * MATCH_VALUE_ROUND_MARGIN_LIMIT)
#define MATCH_VALUE_MAX_ABS_UTILITY \
    (50.0 + 0.05 * MATCH_VALUE_MAX_MATCH_MARGIN)

/* Settings that define the frozen policy used to generate the transition
 * kernels.  The first implementation intentionally supports the measured
 * fixed-role, greedy continuation family; unsupported settings fail closed. */
typedef struct {
    uint64_t net_fingerprint;
    uint32_t controller_abi;
    uint32_t build_profile;
    uint32_t objective;              /* legacy downstream terminal mode 0..2 */
    uint32_t playout_symmetries;
    uint32_t playout_sample;          /* currently fixed-role argmax mode 4 */
    uint32_t playout_prune;
    uint32_t exact_terminal;
    uint32_t plan_deck_max;
    uint32_t plan_block_gap;
    uint32_t draw_playout_deck_max;
    uint32_t deck2_replan_worlds;
    uint32_t deck2_replan_cores;
    uint32_t max_plies;
} MatchValueController;

typedef struct MatchValueTable {
    uint32_t version;
    uint32_t samples_per_policy_lead;
    uint32_t role_cycle_size;
    uint32_t role_balance_complete;
    uint32_t isotonic_projected;
    uint64_t source_seed;
    uint64_t payload_fingerprint;
    double max_isotonic_adjustment[2]; /* before rounds 1 and 2 */
    MatchValueController controller;
    /* Indexed by whether the perspective player starts the indexed round.
     * before_round1 covers leads [-1180,1180]; before_round2 [-2360,2360]. */
    double before_round1[2][MATCH_VALUE_R1_COUNT];
    double before_round2[2][MATCH_VALUE_R2_COUNT];
} MatchValueTable;

uint64_t match_value_net_fingerprint(const Net *net);
uint32_t match_value_build_profile(void);
int match_value_controller_supported(const MatchValueController *controller);
int match_value_controller_equal(const MatchValueController *a,
                                 const MatchValueController *b);
/* True only for an integer number of the controller's independent
 * player-role product cycles.  Playing-agent parsers require this. */
int match_value_balanced_roles(const MatchValueTable *table);

/* Validate every structural, finite, range, and zero-sum invariant.  Isotonic
 * artifacts additionally require monotonicity; raw policy values do not. */
int match_value_validate(const MatchValueTable *table);

/* Canonical little-endian, binary64 persistence.  Save is no-clobber. */
int match_value_save(const MatchValueTable *table, const char *path);
MatchValueTable *match_value_load(const char *path, int *error);
void match_value_free(MatchValueTable *table);

/* Evaluate a genuinely completed round.  Returns zero on malformed state,
 * out-of-range score context, or invalid table.  Round two is evaluated by
 * the exact finishing utility; rounds zero/one look up the next-round value. */
int match_value_terminal(const MatchValueTable *table, const State *terminal,
                         int perspective, double *value);

/* Frozen downstream-controller primitive shared by the deterministic table
 * builder and production rollout.  `role[p]` is one fixed suit mapping for
 * player p for this complete fresh round.  The state must be a live, fully
 * determinized fresh-round deal (not an information-state view). */
int rollout_match_value_round(const Net *net, const NetEvalPlan *eval_plan,
                              const MatchValueController *controller,
                              State *state,
                              const uint8_t role[2][NSUIT], int *margin);

#endif
