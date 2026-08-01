/* search.h -- determinized Monte Carlo tree search.
 *
 * The hidden information (opponent hand, deck order) is resampled many times;
 * each sample turns the game into a perfect information problem that a small
 * PUCT tree search solves with network leaf values and heuristic priors.  Root
 * statistics are pooled over all samples.
 */
#ifndef SEARCH_H
#define SEARCH_H

#include "lc.h"
#include "net.h"

struct Agent;

enum {
    SEARCH_SKIP_NONE = 0,
    SEARCH_SKIP_FORCED,
    SEARCH_SKIP_PLY_WINDOW,
    SEARCH_SKIP_DECK_PHASE,
    SEARCH_SKIP_POLICY_CONFIDENCE,
    SEARCH_SKIP_ROOT_FOCUS,
    SEARCH_SKIP_VISIBLE_PLAN,
    SEARCH_SKIP_LAST_DECK
};

enum {
    SEARCH_METRIC_ROLLOUT = 0,
    SEARCH_METRIC_NETWORK_VALUE,
    SEARCH_METRIC_VISIBLE_PLAN,
    SEARCH_METRIC_LAST_DECK_RULE
};

typedef struct {
    int n;
    int nlegal;             /* legal moves before policy-shortlist filtering */
    int worlds;             /* paired worlds actually evaluated              */
    int max_worlds;         /* configured cap                                */
    int resolved;           /* highest mean clears the paired confidence bar */
    int raw_best;           /* index of highest mean in mv/q                 */
    int policy_top;         /* index of the unmodified network-policy leader */
    int planned_baseline;   /* candidate zero came from the exact scheduler  */
    int deck_end_baseline;  /* candidate zero came from the one-card deck
                               dominance rule                                */
    int semantic_candidates; /* targeted non-prefix candidates actually added */
    int draw_variant_candidates; /* bounded top-action pile variants added    */
    int trusted_candidates; /* ordinary policy-prefix moves eligible directly */
    int prefix_proposed;    /* primary panel's trusted-prefix leader             */
    int selection_reference; /* index low-prior confirmation is compared to   */
    int trusted_prefix_override; /* selected a nonzero trusted-prefix leader   */
    int prefix_confirmed; /* trusted-prefix leader repeated on fresh panel     */
    int prefix_confirm_worlds; /* balanced fixed-world panel size               */
    int metric_kind;        /* SEARCH_METRIC_* meaning of q[]               */
    int planner_turns;      /* own visible-card turns used by scheduler      */
    int planner_score;      /* best guaranteed current-hand score            */
    int planner_policy_score; /* score after spending this turn on policy top */
    int planner_regret;     /* planner_score - planner_policy_score          */
    int planner_policy_block; /* unseen lower-card value policy would close  */
    int planner_selected_block; /* same cost for selected schedule move      */
    int skip_reason;        /* SEARCH_SKIP_* when worlds == 0                */
    int confirmed;          /* a non-policy override passed an independent
                               stochastic continuation check                  */
    int confirm_worlds;     /* independent worlds used by confirmation        */
    double policy_mass;     /* policy probability covered by mv[]            */
    Move mv[MAX_MOVES];
    double visits[MAX_MOVES];
    double q[MAX_MOVES];   /* mean selection objective, mover's view */
    double se[MAX_MOVES];  /* standard error of each candidate's own mean */
    double delta[MAX_MOVES]; /* paired difference against deployed baseline 0 */
    double dse[MAX_MOVES];   /* SE of that paired difference                 */
    double rdelta[MAX_MOVES]; /* primary difference vs selection_reference   */
    double rdse[MAX_MOVES];   /* SE of that reference-relative difference    */
    double cdelta[MAX_MOVES]; /* fresh paired difference vs selection_reference */
    double cdse[MAX_MOVES];   /* SE of cdelta                                 */
    double prefix_q[MAX_MOVES]; /* fresh coherent-panel objective mean          */
    double prefix_se[MAX_MOVES]; /* SE of coherent-panel objective mean          */
    uint8_t pqualified[MAX_MOVES]; /* passed the primary significance gates   */
    uint8_t csupported[MAX_MOVES]; /* candidate passed both independent gates */
    uint8_t guard_rejected[MAX_MOVES]; /* supported but blocked structural risk */
    double prior[MAX_MOVES]; /* root policy probability                      */
    double qw[MAX_MOVES];  /* rollout, final round only: match wins as a
                              fraction of playouts (draws count half);
                              -1 when not applicable */
    float value;           /* pooled root value in points        */
} SearchStats;

Move search_move(const struct Agent *a, const State *st, Rng *rng,
                 float *out_value, SearchStats *stats);
/* Policy improvement by paired playouts from sampled worlds (rollout.c). */
/* out_value is the policy-network continuation value; stats->value/q use the
 * configured rollout selection objective. */
Move rollout_move(const struct Agent *a, const State *st, Rng *rng,
                  float *out_value, SearchStats *stats);
/* Exact terminal objective used by rollout mode 0/1/2. */
double rollout_terminal_objective(const State *terminal, int p, int mode);

#endif
