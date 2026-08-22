/* late_resolver.h -- bounded particle policy improvement for deck <= 3.
 *
 * This is deliberately not described as an exact game solver.  It exhausts
 * the current mover's finite hidden-card support, keeps every particle intact
 * for a complete trajectory, and improves a small observation-keyed policy.
 * Pile-draw cycles are made finite with two independently solved stall
 * horizons.  A move is returned only when those horizons agree.
 */
#ifndef LATE_RESOLVER_H
#define LATE_RESOLVER_H

#include "lc.h"
#include "net.h"

typedef struct {
    int support;             /* ordered root assignments: at most 90 / 990 */
    int root_candidates;     /* top semantic-core complete moves, <= 6     */
    int horizon2_best;
    int horizon4_best;
    int stable;              /* horizons two and four selected same move    */
    int passed;              /* stable and cleared practical override gate  */
    int unavailable;         /* allocation, table, cap, or invariant failure */
    Move horizon2_move;
    Move horizon4_move;
    double horizon2_value;
    double horizon4_value;
    double horizon2_delta;
    double horizon4_delta;
    Move candidate[6];
    float prior[6];
    double horizon2_q[6];
    double horizon4_q[6];
    uint64_t horizon2_nodes;
    uint64_t horizon4_nodes;
    uint64_t horizon2_root_nodes;
    uint64_t horizon4_root_nodes;
    uint64_t horizon2_frozen_opponent_nodes;
    uint64_t horizon4_frozen_opponent_nodes;
    uint64_t horizon2_transitions;
    uint64_t horizon4_transitions;
    uint64_t horizon2_deviation_evals;
    uint64_t horizon4_deviation_evals;
    uint64_t horizon2_exact_leaves;
    uint64_t horizon4_exact_leaves;
} LateResolverStats;

/* Return one only when both bounded policy-improvement solves complete, agree
 * on the same semantic complete move, and clear the practical gate.  `out` is
 * valid only on that passed result.  A zero return with stats->unavailable == 0
 * is still a completed authoritative panel: the caller must retain
 * stats->candidate[0], not continue into a second evaluator.  Only an
 * unavailable panel may fall through.  `cores` must be three; narrower
 * historical settings deliberately keep their old implementation.  Descendant
 * policy ranking uses a deterministic one- or five-symmetry group
 * (`policy_symmetries > 1` selects five) and is cached by information state.
 */
int late_resolver_choose(const Net *net, const State *st, int objective,
                         int cores, int policy_symmetries, int max_actions,
                         double practical_min, Move *out,
                         LateResolverStats *stats);

/* Planned form for callers which already proved inference shortcuts for this
 * immutable checkpoint.  A null, foreign-owner, or structurally invalid plan
 * fails closed to the complete evaluator; it never triggers a fresh parameter
 * scan.  The ordinary entry point above remains the self-contained fallback
 * and builds its own proof before delegating here. */
int late_resolver_choose_plan(const Net *net, const State *st, int objective,
                              int cores, int policy_symmetries,
                              int max_actions, double practical_min,
                              Move *out, LateResolverStats *stats,
                              const NetEvalPlan *eval_plan);

/* Small public invariant hook used by focused runtime tests. */
int late_resolver_assignment_count(const State *st, int p);

/* Focused invariant hook: expose the exact root/forced-progress shortlist
 * construction without running either bounded horizon. */
int late_resolver_policy_candidates(const Net *net, const State *st,
                                    int cores, int policy_symmetries,
                                    int max_actions, int deck_only,
                                    Move *out, float *prior);

#endif
