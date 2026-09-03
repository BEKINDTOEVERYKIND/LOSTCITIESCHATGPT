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

/* Rollout deliberately spends worlds on only a small policy-guided prefix. */
#define ROLLOUT_MAX_CANDIDATES 8

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

enum {
    POLICY_COST_GATE_NONE = 0,
    POLICY_COST_GATE_INVALID_PRIMARY,
    POLICY_COST_GATE_ADJUSTED_BASELINE,
    POLICY_COST_GATE_PRIMARY_EVIDENCE,
    POLICY_COST_GATE_INVALID_FRESH,
    POLICY_COST_GATE_FRESH_LEADER_MISMATCH,
    POLICY_COST_GATE_FRESH_EVIDENCE,
    POLICY_COST_GATE_DISCARD_GUARD,
    POLICY_COST_GATE_CAP_INVALID,
    POLICY_COST_GATE_SELECTED
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
    int draw_planned_baseline; /* candidate zero's draw source was repaired */
    int deck_end_baseline;  /* candidate zero came from the one-card deck
                               dominance rule                                */
    int semantic_candidates; /* targeted non-prefix candidates actually added */
    int draw_variant_candidates; /* bounded top-action pile variants added    */
    int action_core_candidates; /* distinct ordinary semantic cores selected */
    int action_draw_candidates; /* safe draw alternatives within core budget */
    int trusted_candidates; /* ordinary policy-prefix moves eligible directly */
    int prefix_proposed;    /* primary panel's trusted-prefix leader             */
    int selection_reference; /* index low-prior confirmation is compared to   */
    int trusted_prefix_override; /* selected a nonzero trusted-prefix leader   */
    int prefix_numerical_agreement; /* fresh panel repeated primary leader      */
    int prefix_gate_passed; /* configured paired evidence/effect gate passed   */
    int prefix_confirmed; /* every required fresh/controller panel passed       */
    int prefix_confirm_worlds; /* balanced fixed-world panel size               */
    int prefix_veto_attempted; /* independent continuation controller consulted */
    int prefix_veto_worlds; /* same fixed-world panel under veto controller      */
    int prefix_veto_numerical_agreement; /* veto repeated proposed leader       */
    int prefix_veto_gate_passed; /* veto controller's paired gate passed         */
    int prefix_veto_passed; /* veto retained the already confirmed proposal      */
    int prefix_ranker_attempted; /* direct signed proposal-vs-zero ranker consulted */
    int prefix_ranker_valid; /* ranker produced a finite information-safe score */
    int prefix_ranker_passed; /* ranker retained the confirmed proposal          */
    double prefix_ranker_score; /* ranker-minus-root residual proposal log-odds */
    double prefix_ranker_threshold; /* configured minimum residual support      */
    int policy_cost_active; /* calibrated rollout5 root selector was used */
    int policy_cost_anchor_interval; /* shared spline segment at root nply */
    int policy_cost_proposed; /* primary adjusted all-pair leader          */
    int policy_cost_selected; /* final selection after independent panel   */
    int policy_cost_primary_passed; /* primary 3.5-SE all-pair gate          */
    int policy_cost_fresh_passed; /* fresh 2.58-SE all-pair gate             */
    int policy_cost_primary_valid; /* complete finite primary fixed-world panel */
    int policy_cost_fresh_valid; /* complete finite independent panel          */
    int policy_cost_fresh_worlds; /* independent policy-cost evidence worlds   */
    int policy_cost_gate_reason; /* POLICY_COST_GATE_* final authority reason */
    int policy_cost_cap_valid; /* no primary/fresh continuation cap leaf      */
    int policy_cost_primary_protected; /* raw-positive policy rivals checked   */
    int policy_cost_fresh_protected; /* same count on independent panel       */
    uint64_t policy_cost_fingerprint; /* checked external artifact content    */
    double policy_cost_lambda_action;
    double policy_cost_lambda_draw;
    double policy_cost_override_min; /* v1 binds legacy low-prior floor off */
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
    uint64_t exact_terminal_leaves; /* continuation trajectories whose final
                                       turn was solved exactly                 */
    uint64_t unfinished_cap_leaves; /* trajectories stopped by LC_MAX_PLIES
                                       with deck cards remaining              */
    uint64_t cycle_breaks;       /* repeated late decision states redirected  */
    uint64_t cap_reserve_forces; /* draws redirected to preserve a real end   */
    uint64_t deck2_replans;      /* bounded recursive late information replans */
    uint64_t deck2_replan_worlds; /* fresh mover-view determinizations used     */
    uint64_t deck2_replan_evals; /* late candidate trajectories evaluated      */
    uint64_t deck2_replan_cap_hits; /* explicit depth/work-budget fallbacks      */
    uint64_t deck2_replan_cache_hits; /* semantic transposition decisions reused */
    uint64_t deck2_replan_cycle_closures; /* recursive information cycles closed   */
    uint64_t deck2_replan_max_depth; /* deepest completed recursive replan path  */
    uint64_t deck2_replan_root_calls; /* first late node panels evaluated          */
    uint64_t deck2_replan_root_worlds; /* worlds in those unabridged first panels   */
    uint64_t deck2_replan_max_stall_chain; /* longest recursive pile-draw chain     */
    uint64_t deck2_replan_low_world_fallbacks; /* child panels below safe minimum    */
    /* Finite-support late resolver diagnostics.  These are deliberately
     * separate from deck2_replan_*: the resolver carries one root particle
     * support through each trajectory and does not recursively redeterminize.
     * A completed panel may still fail its stability/practical-gain gate. */
    int late_resolver_attempted;
    int late_resolver_completed;
    int late_resolver_stable;
    int late_resolver_passed;
    int late_resolver_used;     /* completed legal panel owned root choice */
    int late_resolver_retained; /* authoritative panel returned candidate 0 */
    int late_resolver_override; /* authoritative panel returned challenger */
    double late_resolver_practical_min;
    int late_resolver_support;
    int late_resolver_candidates;
    int late_resolver_h2_best;
    int late_resolver_h4_best;
    double late_resolver_h2_value;
    double late_resolver_h4_value;
    double late_resolver_h2_delta;
    double late_resolver_h4_delta;
    uint64_t late_resolver_h2_nodes;
    uint64_t late_resolver_h4_nodes;
    uint64_t late_resolver_h2_root_nodes;
    uint64_t late_resolver_h4_root_nodes;
    uint64_t late_resolver_h2_frozen_opponent_nodes;
    uint64_t late_resolver_h4_frozen_opponent_nodes;
    uint64_t late_resolver_h2_transitions;
    uint64_t late_resolver_h4_transitions;
    uint64_t late_resolver_h2_deviation_evals;
    uint64_t late_resolver_h4_deviation_evals;
    uint64_t late_resolver_h2_exact_leaves;
    uint64_t late_resolver_h4_exact_leaves;
    Move late_resolver_candidate[6];
    double late_resolver_prior[6];
    double late_resolver_h2_q[6];
    double late_resolver_h4_q[6];
    double policy_mass;     /* complete-move mass, or aggregate semantic-core
                               mass for a hierarchical shortlist              */
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
    double prefix_delta[MAX_MOVES]; /* fresh paired delta against candidate zero */
    double prefix_dse[MAX_MOVES]; /* SE of fresh paired delta                    */
    double prefix_veto_q[MAX_MOVES]; /* independent-controller objective mean    */
    double prefix_veto_se[MAX_MOVES]; /* SE of independent-controller mean       */
    double prefix_veto_delta[MAX_MOVES]; /* paired delta against candidate zero   */
    double prefix_veto_dse[MAX_MOVES]; /* SE of independent-controller delta      */
    double policy_semantic_prior[MAX_MOVES]; /* P_A after suit averaging       */
    double policy_conditional_draw_prior[MAX_MOVES]; /* P_D = joint/P_A      */
    double policy_cost_penalty[MAX_MOVES]; /* deterministic policy cost        */
    double policy_adjusted_q[MAX_MOVES]; /* raw primary Q minus policy cost    */
    double policy_fresh_adjusted_q[MAX_MOVES]; /* independent-panel counterpart */
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

/* Deterministic, information-set-safe final gate for rollout4/rolloutu4.
 * The complete referee state is sanitized internally before either network
 * is evaluated.  Only baseline and proposal are scored, and the function has
 * no RNG input.  Returns 1 only when a finite score meets the configured
 * threshold; invalid inference fails closed to candidate zero. */
int rollout_action_ranker_veto(const struct Agent *a,
                               const State *complete, Move baseline,
                               Move proposal, double *score, int *valid);

/* Exact flat policy-prefix admission used by the production rollout actor.
 * Ties retain legal-move enumeration order.  baseline is always unique at
 * candidate zero, even when it lies below the floor; order must hold at least
 * ROLLOUT_MAX_CANDIDATES indices.  Returns the admitted count. */
int rollout_policy_prefix_indices(
    const Move *mv, const float *prob, int n, int baseline,
    int root_width, float cand_floor, float cand_mass, int min_cand,
    int *order);
/* Exact hierarchical policy admission used by rollout5 calibration.  The
 * supplied arrays must be the complete legal list and one normalized,
 * suit-symmetrized policy vector for st; baseline must be its literal first
 * argmax.  Returns -1 on malformed/subset input, otherwise the admitted count
 * (at most five), with baseline uniquely at order[0]. */
int rollout_action_core_indices(
    const State *st, const Move *mv, const float *prob, int n, int baseline,
    int root_width, int action_core_count, int min_cand,
    float cand_floor, float cand_mass, int *order,
    int *core_candidates, int *draw_candidates);

/* The policy-cost family has a stricter, campaign-frozen hierarchy than the
 * legacy action-core helper above.  It constructs the one-percent semantic-
 * core master exactly once, admits positive-probability safe draw variants
 * without a joint-probability floor, and obtains the two-percent support only
 * by filtering that master.  No candidate may be refilled or reordered. */
#define ROLLOUT_POLICY_COST_FLOORS 2
#define ROLLOUT_POLICY_COST_MAX_CANDIDATES 5
typedef struct {
    int n[ROLLOUT_POLICY_COST_FLOORS];
    int core_candidates[ROLLOUT_POLICY_COST_FLOORS];
    int draw_candidates[ROLLOUT_POLICY_COST_FLOORS];
    int order[ROLLOUT_POLICY_COST_FLOORS]
             [ROLLOUT_POLICY_COST_MAX_CANDIDATES];
} RolloutPolicyCostSupport;

/* `mv` and `prob` must be the complete legal suit-symmetrized policy and
 * baseline its literal first argmax.  Index 0 is the canonical 1% master;
 * index 1 is its exact no-refill 2% mask.  Returns 1 on success, 0 on any
 * malformed input or failed nesting proof. */
int rollout_policy_cost_support(
    const State *st, const Move *mv, const float *prob, int n, int baseline,
    RolloutPolicyCostSupport *support);
/* Exact terminal objective used by rollout mode 0/1/2. */
double rollout_terminal_objective(const State *terminal, int p, int mode);

/* Solve a one-card-deck information state exactly.  Every legal semantic
 * play/discard action is paired with a deck draw, which ends the round before
 * the hidden card can matter.  Returns an index into mv, or -1 unless
 * st->deck_left == 1.  prior is used only to break exact objective ties. */
int rollout_exact_terminal_choice(const State *st, const Move *mv,
                                  const float *prior, int n, int objective,
                                  double *out_objective);

/* Propagation-control counterpart: preserve selected's semantic card/action
 * and return its round-ending deck-draw variant.  This deliberately does not
 * optimize the card/action. */
int rollout_policy_terminal_choice(const Move *mv, int n, int selected);

/* Production forced-progress counterpart: choose the network's best legal
 * move conditional on drawing from the deck.  This matters when the policy's
 * card/action x draw interaction makes the unrestricted winner differ from
 * the best action under a required deck draw.  dead applies the ordinary
 * continuation-only dominated-discard filter; if it removes every deck move,
 * the function safely retries without that heuristic filter. */
int rollout_policy_deck_choice(const State *st, const Move *mv,
                               const float *score, int n, uint64_t dead);

/* Exact semantic-information cycle tracker used by production late playouts.
 * One instance belongs to one continuation trajectory.  The key deliberately
 * ignores nply and physical wager identity; repeated states are detected only
 * with at most three deck cards remaining, exactly as in rollout.c. */
typedef struct {
    uint64_t information_key[LC_MAX_PLIES];
    int n;
} RolloutLateCycleHistory;

void rollout_late_cycle_init(RolloutLateCycleHistory *history);
int rollout_late_cycle_repeated(RolloutLateCycleHistory *history,
                                const State *st);

/* Strategic equality used by late-rollout cycle detection.  Inactive pile
 * storage and the physical IDs of indistinguishable wagers are ignored. */
int rollout_same_late_state(const State *a, const State *b);

/* Near-greedy continuation sampler used by optional fresh confirmation.
 * Only the shortest policy prefix covering 99.5% mass is eligible.  Noise is
 * a pure function of seed/depth/player/packed move, so reordering or adding
 * unrelated candidates cannot change a shared move's Gumbel variate.  A
 * non-positive temperature is exact argmax. */
int rollout_near_greedy_pick(const Move *mv, const float *prob, int n,
                             float temperature, uint64_t seed,
                             int depth, int player);

/* Fixed-world, policy-focused diagnostic panel.  This is deliberately not a
 * second gameplay selector: callers supply at most eight complete moves that
 * were admitted from policy-ranked complete semantic moves.  Every candidate sees the
 * same sanitized hidden worlds, production continuation roles, cycle/cap
 * handling, and exact last-deck solver.  baseline is an index into candidate
 * and owns delta/delta_se.  With one to three deck cards, requesting at least
 * the full ordered hidden-deck support evaluates that support exactly. */
typedef struct {
    int n;
    int requested_worlds;
    int worlds;
    int baseline;
    int selected;
    int exact_hidden_support;
    int hidden_support;
    int objective;
    int panel_role; /* ROLLOUT_AUDIT_PANEL_* frozen evidence domain */
    uint64_t hidden_world_fingerprint; /* ordered hidden support/worlds */
    double q[ROLLOUT_MAX_CANDIDATES];
    double se[ROLLOUT_MAX_CANDIDATES];
    double delta[ROLLOUT_MAX_CANDIDATES];
    double delta_se[ROLLOUT_MAX_CANDIDATES];
    /* Candidate-minus-rival paired panels.  These make every all-pair gate
     * reproducible from the same fixed worlds; diagonal entries are zero. */
    double pair_delta[ROLLOUT_MAX_CANDIDATES][ROLLOUT_MAX_CANDIDATES];
    double pair_delta_se[ROLLOUT_MAX_CANDIDATES][ROLLOUT_MAX_CANDIDATES];
    uint64_t exact_terminal_leaves;
    uint64_t unfinished_cap_leaves;
    uint64_t cycle_breaks;
    uint64_t cap_reserve_forces;
} RolloutAuditPanel;

enum {
    ROLLOUT_AUDIT_PANEL_PRIMARY = 0,
    ROLLOUT_AUDIT_PANEL_FRESH = 1
};

int rollout_audit_panel(const struct Agent *a, const State *st,
                        const Move *candidate, int n, int baseline,
                        uint64_t seed, int worlds,
                        RolloutAuditPanel *out);
/* Role-explicit form for frozen policy-cost P/F evidence.  FRESH uses the
 * same late-support, suit-role, world-RNG, and continuation-seed domains as
 * rollout5's independent production panel.  The legacy entry point above is
 * exactly a PRIMARY wrapper. */
int rollout_audit_panel_role(
    const struct Agent *a, const State *st,
    const Move *candidate, int n, int baseline, int panel_role,
    uint64_t seed, int worlds, RolloutAuditPanel *out);

#endif
