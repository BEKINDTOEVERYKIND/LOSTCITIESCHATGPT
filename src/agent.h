/* agent.h -- move selection policies.
 *
 * Every agent sees only its own information set: the deck order and the
 * opponent's hand are never read, and deck draws are handled by sampling from
 * the set of cards the agent has not seen.
 */
#ifndef AGENT_H
#define AGENT_H

#include "lc.h"
#include "net.h"

typedef enum {
    AG_RANDOM = 0,
    AG_HEUR,     /* one-ply greedy on the hand-crafted evaluation */
    AG_NET,      /* one-ply greedy on the value head             */
    AG_POLICY,   /* single forward pass, argmax of the policy head */
    AG_MCTS,     /* determinized MCTS, network priors and values  */
    AG_ROLLOUT   /* candidate moves played out in sampled worlds   */
} AgentKind;

typedef struct Agent {
    AgentKind kind;
    const Net *net;
    int draw_samples;   /* deck-draw samples per decision (AG_NET)         */
    float temp;         /* >0: sample instead of taking the best move      */
    float eps;          /* probability of a uniformly random legal move    */
    int symmetries;     /* policy ensemble over exact suit relabellings:
                           1 (off), 5 (rotations), 10 (dihedral),
                           20 (affine), or 120 (all permutations). */
    int plan_deck_max;  /* visible-hand play-order scheduler threshold;
                           requires a positive plan_block_gap (0 = disabled) */
    int plan_block_gap; /* root correction: minimum unseen-value preservation
                           needed to replace the policy leader with an
                           equivalent optimal-plan first move (0 = disabled);
                           inside rollout continuations a positive value merely
                           enables the full optimal visible-hand schedule */
    int draw_root_deck_max; /* repair only the deployed root action's draw
                               source at or below this deck size (0 = off) */
    int draw_playout_deck_max; /* independently repair draw sources inside
                                  rollout continuations (0 = off).  Keeping
                                  this separate makes the root and world-model
                                  effects directly ablatable. */
    float confirm_temp; /* AG_ROLLOUT: optional near-greedy action temperature
                           used only by fresh confirmation playouts.  Sampling
                           is limited to the top 99.5% policy mass and uses
                           stateless move-keyed common Gumbel noise; 0 keeps
                           deterministic argmax confirmation. */
    int action_core_count; /* AG_ROLLOUT: optional hierarchical ordinary
                              shortlist.  Keep this many distinct semantic
                              card/play-discard cores, then use remaining room
                              in a five-candidate budget for at most one safe
                              policy-floor draw alternative per core.  0 keeps
                              complete-move policy ranking. */
    int semantic_cand;  /* AG_ROLLOUT: add at most one useful pile pickup for
                           each of a top policy play/discard action, plus one
                           isolated one-sided-wager discard (0/1); these are
                           targeted semantic challengers, never a scan of every
                           legal move */
    int confirm_exact5;  /* AG_ROLLOUT: use the exact five-rotation ensemble
                            for the fresh confirmation pass (0 keeps the
                            configured cheap continuation model) */
    int draw_variant_cores; /* AG_ROLLOUT: for this many top distinct
                               card/disposition actions, admit the best legal
                               pile-draw alternatives into the bounded root
                               audit (0 off, 1-2 supported) */
    int draw_variant_deck_max; /* enable those tempo/stall pile alternatives
                                  only at or below this deck count (0 = no
                                  phase limit) */
    int policy_prefix_mode; /* ordinary policy-floor prefix selection:
                               0 = every override gated; 1 = trust numerical
                               leader directly; 2 = require that leader to
                               repeat on fresh balanced fixed-world symmetry;
                               3 = same fresh-panel consensus, but assign the
                               two players independently stratified, coherent
                               suit mappings.  Mode 3 adds no network forwards
                               and avoids assuming an unknown opponent shares
                               our arbitrary network orientation.
                               Added low-prior challengers remain gated. */
    float prefix_confirm_k; /* AG_ROLLOUT: optional fresh trusted-prefix
                               evidence gate.  When enabled together with
                               prefix_confirm_min, the proposed leader must
                               beat candidate zero by this many paired SEs.
                               Both zero preserves numerical consensus. */
    float prefix_confirm_min; /* AG_ROLLOUT: paired fresh-panel improvement
                                 required in objective units.  The prefix gate
                                 is enabled only when both thresholds are >0. */
    /* AG_MCTS */
    int dets;           /* determinizations                                */
    int sims;           /* simulations per determinization                 */
    int root_width;     /* root moves kept after prior pruning             */
    int node_width;     /* interior moves kept                             */
    float cpuct;
    float cand_floor;   /* AG_ROLLOUT: ignore candidates below this policy  */
    float cand_mass;    /* AG_ROLLOUT: when >0, keep the shortest top-policy
                           prefix reaching this cumulative mass (subject to
                           min_cand/root_width), instead of a fixed floor */
    int min_cand;       /* AG_ROLLOUT: but always keep at least this many --
                           a sharp prior otherwise leaves the search a single
                           candidate, able to confirm the policy but never to
                           correct it (0/1 = floor applies unconditionally) */
    int ply_lo, ply_hi; /* AG_ROLLOUT: search only when
                           ply_lo <= nply (< ply_hi if ply_hi > 0); outside
                           the window the policy/planner baseline plays, except
                           for a focused one-sided-wager trigger.  For
                           measuring WHERE in a round ordinary search earns
                           its keep (0,0 = search everywhere) */
    int eval_cand;      /* AG_ROLLOUT: report at least this many policy-ranked
                           candidates.  Extra entries are diagnostic only and
                           cannot be selected (0 = off). */
    int batch_dets;     /* AG_ROLLOUT: paired worlds per adaptive batch.
                           dets is the cap; 0 evaluates exactly dets worlds */
    int playout_symmetries; /* AG_ROLLOUT continuation suit group.  Mode 0
                               averages the group exactly; modes 1/2 draw one
                               member per decision, while mode 3 fixes one
                               member for the complete hidden-world trajectory.
                               Mode 1 samples the resulting action; modes 2/3
                               take argmax. */
    int discard_guard; /* AG_ROLLOUT: do not override a nondominated policy
                          move with a discard that lc_discard_dominated marks
                          questionable.  Unlike prune_dom, the move remains in
                          the audit and in continuations. */
    int deck_max;      /* AG_ROLLOUT: search only with at most this many deck
                          cards remaining (0 = no deck-phase gate). */
    int confirm_dets;  /* AG_ROLLOUT: fresh confirmation worlds
                          (0 = the primary configured world cap). */
    int playout_prune; /* AG_ROLLOUT continuation-only dead-discard focus.
                          -1 follows prune_dom for backward compatibility;
                          0/1 explicitly disables/enables it. */
    int win_q;          /* AG_ROLLOUT objective: 0 = round margin; 1 = pure
                           match result in real round index 2; 2 = champion
                           hybrid (0.05 * final margin + 50 * result) there.
                           Rounds 0/1 always use margin, preserving the
                           intentional last-round-only win objective.
                           Default 0; SearchStats.qw always reports the raw
                           final-round win fraction. */
    int prune_dom;      /* AG_ROLLOUT: drop discards dominated by a dead-card
                           discard (lc_discard_dominated) from candidates and
                           playout argmax -- frees candidate slots and stops
                           playouts gifting live cards when a dead one is in
                           hand */
    float override_k;   /* AG_ROLLOUT: let an eligible challenger take the
                           move only after it beats the deployed baseline by
                           this many
                           paired standard errors and passes a fresh,
                           independently seeded continuation check
                           (0 = legacy behavior: take the numerical leader) */
    int playout_sample; /* AG_ROLLOUT continuation mode:
                           0 = exact suit-group average, then argmax;
                           1 = random group member, then sample its policy;
                           2 = random group member, then argmax.
                           Modes 1/2/3/4 cost one forward per decision and common
                           per-world seeds keep candidate comparisons paired.
                           Mode 2 changes the sampled member every decision;
                           mode 3 draws one member per hidden world and keeps
                           it fixed throughout that playout; mode 4 draws
                           separately stratified fixed members for the two
                           players.  Mode 1 is a high-variance robustness
                           ablation. */
    float override_min; /* AG_ROLLOUT: ...AND by at least this many points.
                           The SE gate alone is world-count-dependent in the
                           wrong direction: more worlds shrink the noise but
                           not the playout BIAS, so at 512 worlds a 3-SE
                           gate fires on ~1-point bias artifacts (measured:
                           stall- and discard-flavoured overrides an expert
                           reviewer graded as blunders).  Points are the
                           bias's own units.  Default 4. */
    float gate;         /* AG_ROLLOUT: skip the search entirely when the
                           policy's top move already has >= this probability
                           (0 = always search) */
    int no_belief;      /* AG_ROLLOUT ablation: sample worlds uniformly      */
    float belief_alpha; /* AG_ROLLOUT: temperature/strength of the coherent
                           fixed-cardinality opponent-hand posterior.
                           1 = checkpoint logits; 0 = uniform. */
    const char *name;
} Agent;

/* Cards standing in for the unknown top of the deck.
 *
 * The set of cards a player has not seen is the same after every candidate
 * move of a turn (the card that leaves the hand was never unseen), so one
 * sample can be shared by all of them.  Reusing it is a common-random-numbers
 * trick: it removes almost all of the sampling noise from the *comparison*
 * between moves, which is what the choice depends on. */
#define MAX_DRAW_SAMPLES 24
typedef struct {
    uint8_t card[MAX_DRAW_SAMPLES];
    int n;
} DrawSamples;

void  draw_samples_init(const State *st, int p, Rng *rng, int k, DrawSamples *ds);
float move_value_net(const Net *net, const State *st, Move m, const DrawSamples *ds);
float move_value_heur(const State *st, Move m, const DrawSamples *ds);

void agent_default(Agent *a, AgentKind k, const Net *net);

/* Build exactly the information state available to player p.  The returned
 * view retains p's hand, all public state and public hand sizes, but replaces
 * the opponent's hidden hand by only its known face-up cards and erases the
 * future deck order.  Use this boundary in diagnostics that also retain a
 * complete referee state for labels: no neural-network call should receive
 * that complete state. */
void agent_information_view(const State *complete, int p, State *view);

/* Policy head evaluated on st for the player to move.  Fills mv[] with the
 * legal moves and prob[] with the normalized policy; returns the count. */
int  policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value);
int  policy_probs_sym(const Net *net, const State *st, Move *mv, float *prob,
                      float *value, int symmetries);
/* Draw one member of a suit-permutation group and map its policy back to st.
 * Averaging repeated calls equals policy_probs_sym(), but each costs one
 * forward pass. */
int  policy_probs_random_sym(const Net *net, const State *st, Move *mv,
                             float *prob, Rng *rng, int symmetries);
/* Evaluate one explicitly supplied suit relabelling and map its legal-move
 * probabilities back to st.  This is the one-forward building block for a
 * temporally consistent sampled-symmetry rollout actor. */
int  policy_probs_perm(const Net *net, const State *st, Move *mv, float *prob,
                       float *value, const uint8_t perm[NSUIT]);
/* Fill an exact subgroup of suit relabellings.  Invalid sizes return only the
 * identity.  The returned maps send original suit -> permuted suit. */
int  suit_permutations(int requested, uint8_t out[120][NSUIT]);

/* Deterministically choose one member of an exact suit group for a global
 * trajectory id.  Supported group sizes are 1, 5, 10, 20 and 120; zero is an
 * explicit identity/off setting.  Selection depends only on seed and id, not
 * on the worker that happens to generate the trajectory. */
int  trajectory_suit_permutation(int symmetries, uint64_t seed,
                                 uint64_t trajectory,
                                 uint8_t perm[NSUIT]);

/* Evaluate a behaviour policy in one relabelled trajectory orientation.
 * `view` and view_mv are the exact state/action rows to store for PPO;
 * engine_mv maps each action back to the engine's original orientation.
 * raw_prob is the network softmax and behavior_prob applies `temperature`.
 * This keeps chosen action, old probability, value and belief labels in one
 * coherent coordinate system while the canonical engine advances normally. */
int  trajectory_policy_probs(const Net *net, const State *engine_state,
                             const uint8_t perm[NSUIT], float temperature,
                             State *view, Move *view_mv, Move *engine_mv,
                             float *raw_prob, float *behavior_prob);

/* Network value for an explicitly chosen perspective, averaged over the same
 * exact suit group as the policy and returned in points.  Unlike
 * policy_probs_sym(), this does not require p to be the player to move.
 *
 * Combining two calls as 0.5 * (V_p - V_{p^1}) is a centralized critic: it
 * may be used only when the complete state is already available (self-play
 * training or a sampled determinization), never to inspect the real hidden
 * hand while making an information-set decision. */
float net_value_state_sym(const Net *net, const State *st, int p,
                          int symmetries);

/* Fixed-cardinality opponent-hand posterior.  `marginal[i]` is the true
 * inclusion probability of card[i] under the same distribution sample()
 * uses, and the marginals sum to `need`.  alpha=0 is the exact uniform
 * card-count prior. */
typedef struct {
    int n, need;
    uint8_t card[NCARD];
    double weight[NCARD];
    double suffix[NCARD + 1][HAND_SIZE + 1];
    float marginal[NCARD];
} BeliefDist;

/* Score the true unknown part of an opponent hand under an already prepared
 * BeliefDist.  `opponent_hand` may include publicly known cards; only the
 * candidate cards in dist are scored.  The call rejects labels whose
 * cardinality is not exactly dist->need. */
int  belief_dist_true_nll(const BeliefDist *dist, uint64_t opponent_hand,
                          double *nll);

/* Evaluate the exact-cardinality exponential-family hand distribution used by
 * belief_dist_sample().  logits/held are indexed over one candidate array;
 * held may be NULL when only marginals are required.  When nll is requested,
 * held must contain exactly need ones.  The returned gradient of nll with
 * respect to logits (at alpha=1, away from the numerical clamp) is exactly
 * marginal[i] - held[i]. */
int  belief_exact_k_eval(const float *logits, const uint8_t *held,
                         int n, int need, float alpha,
                         float *marginal, double *nll);

int  belief_dist_init(const Net *net, const State *st, int p, int symmetries,
                      float alpha, BeliefDist *dist);
void belief_dist_sample(const State *st, int p, Rng *rng,
                        const BeliefDist *dist, State *out);

/* Evaluate every legal move from st for the player to move.  Returns the
 * number of moves and fills mv[] and val[] (values in points, mover's view). */
int  agent_move_values(const Agent *a, const State *st, Rng *rng, Move *mv, float *val);
Move agent_move(const Agent *a, const State *st, Rng *rng);

/* Sample an index from weights[0..n) (already non-negative, sum > 0). */
int  sample_index(const float *w, int n, Rng *rng);

/* Build a determinization of st consistent with p's information: cards the
 * opponent is known to hold are pinned, and the rest of their hand and the
 * deck order are resampled from the unseen cards.  With a network, the
 * opponent's unknown cards are drawn from the belief head's posterior (what
 * their play so far implies they kept) instead of uniformly; net == NULL
 * falls back to uniform. */
void determinize(const State *st, int p, Rng *rng, State *out);
void determinize_b(const State *st, int p, Rng *rng, const Net *net, State *out);

#endif
