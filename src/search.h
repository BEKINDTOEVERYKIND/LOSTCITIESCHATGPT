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
    SEARCH_SKIP_ROOT_FOCUS
};

typedef struct {
    int n;
    int nlegal;             /* legal moves before policy-shortlist filtering */
    int worlds;             /* paired worlds actually evaluated              */
    int max_worlds;         /* configured cap                                */
    int resolved;           /* highest mean clears the paired confidence bar */
    int raw_best;           /* index of highest mean in mv/q                 */
    int skip_reason;        /* SEARCH_SKIP_* when worlds == 0                */
    int confirmed;          /* a non-policy override passed an independent
                               stochastic continuation check                  */
    int confirm_worlds;     /* independent worlds used by confirmation        */
    double policy_mass;     /* policy probability covered by mv[]            */
    Move mv[MAX_MOVES];
    double visits[MAX_MOVES];
    double q[MAX_MOVES];   /* mean selection objective, mover's view */
    double se[MAX_MOVES];  /* standard error of each candidate's own mean */
    double delta[MAX_MOVES]; /* paired difference against policy candidate 0 */
    double dse[MAX_MOVES];   /* SE of that paired difference                 */
    double cdelta[MAX_MOVES]; /* fresh stochastic paired difference vs policy */
    double cdse[MAX_MOVES];   /* SE of cdelta                                 */
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
