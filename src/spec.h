/* spec.h -- agent command line specs.
 *
 *   random
 *   heur
 *   net:PATH[:draw_samples]
 *   policy:PATH[:temperature[:symmetries[:plan_deck_max[:plan_block_gap
 *          [:draw_root_deck_max]]]]]
 *   hrollout[:worlds[:candidates]]     (no network: heuristic + PIMC)
 *   rollout:PATH[:worlds[:candidates[:policy_floor[:gate[:min_candidates
 *            [:ply_lo[:ply_hi[:eval_candidates[:objective[:prune
 *            [:override_k[:override_min[:sample[:symmetries
 *            [:policy_mass[:batch_worlds[:playout_symmetries
 *            [:discard_guard[:deck_max[:confirm_worlds
 *            [:playout_prune[:plan_deck_max[:plan_block_gap
 *            [:semantic_candidates[:confirm_exact5
 *            [:draw_variant_cores[:draw_variant_deck_max
 *            [:policy_prefix_mode[:belief_alpha[:draw_root_deck_max
 *            [:draw_playout_deck_max[:prefix_confirm_k
 *            [:prefix_confirm_min[:confirm_temp
 *            [:action_core_count[:exact_terminal[:deck2_replan_worlds
 *            [:deck2_replan_cores[:bounded_late_root
 *            [:bounded_late_min]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
 *                 objective: 0 margin; 1 final match result; 2 final hybrid
 *                 symmetries: 1, 5, 10, 20, or 120 exact suit relabellings
 *                 sample: continuation mode 0 exact-group argmax, 1 random-
 *                   group policy sample, 2 random-group argmax
 *                 policy_mass: shortest top-policy prefix to cover this mass
 *                 batch_worlds: adaptive paired batch; worlds becomes the cap
 *                 playout_symmetries: suit group for continuations.  Mode 0
 *                   averages it exactly; modes 1/2 draw one group member per
 *                   decision; mode 3 draws one member per hidden world and
 *                   retains it for the full playout; mode 4 assigns the two
 *                   players separately stratified members, each fixed for
 *                   the full playout.  Mode 1 samples the action; modes 2-4
 *                   take argmax (1, 5, 10, 20, or 120).
 *                 discard_guard: block a questionable-discard challenger
 *                   without removing it from the audit (0 or 1)
 *                 deck_max: search only at or below this deck count (0 off)
 *                 confirm_worlds: fixed fresh stochastic confirmation worlds
 *                   (0 uses the configured primary-world cap)
 *                 playout_prune: continuation-only dead-discard focus;
 *                   -1 follows root prune, 0 disables, 1 enables.  This can
 *                   improve the world model without hiding root candidates.
 *                 plan_deck_max / plan_block_gap: enable visible-hand
 *                   play-order scheduling.  At the root the gap is the
 *                   required lower-card option preservation; inside rollouts
 *                   a positive gap enables full schedule normalization.
 *                 semantic_candidates: add only targeted draw variants for
 *                   top policy actions and a one-sided-wager discard;
 *                   an early wager trigger compares only that move with the
 *                   baseline and receives 16,384 worlds
 *                 confirm_exact5: fresh confirmation uses an exact five-way
 *                   rotation ensemble at every downstream decision
 *                 draw_variant_cores: add bounded pile-draw alternatives for
 *                   the top one or two distinct policy action cores; this
 *                   never scans unrelated legal plays/discards
 *                 draw_variant_deck_max: activate that expansion only at or
 *                   below this deck count (0 disables the phase limit)
 *                 policy_prefix_mode: 0 gates every override; 1 trusts the
 *                   numerical leader among ordinary policy-floor candidates;
 *                   2 adds a fresh balanced fixed-world-symmetry panel; 3
 *                   uses independently stratified, coherent suit mappings
 *                   for the two players.  With paired thresholds disabled,
 *                   modes 2/3 require the same numerical leader.  With both
 *                   thresholds enabled, the primary proposal instead has to
 *                   beat candidate zero by both fresh paired tests; a nearly
 *                   tied alternative leader does not erase that evidence.
 *                   Added low-prior challengers retain both statistical gates.
 *                 prefix_confirm_k / prefix_confirm_min: optional paired
 *                   evidence and practical-effect thresholds for the fresh
 *                   trusted-prefix panel.  Both must be positive to enable;
 *                   both zero preserves numerical-consensus behavior.
 *                 belief_alpha: strength/temperature of the coherent
 *                   fixed-cardinality opponent-hand posterior (default 1;
 *                   0 is the exact uniform prior).
 *                 draw_root_deck_max / draw_playout_deck_max: independently
 *                   repair the chosen semantic action's draw source at the
 *                   deployed root and inside rollout continuations.  Each
 *                   threshold is 0 (off) or a remaining-deck count.
 *                 confirm_temp: optional near-greedy fresh-confirmation
 *                   action temperature.  Sampling is restricted to the top
 *                   99.5% policy mass and uses move-keyed common Gumbel noise;
 *                   0 retains deterministic argmax confirmation.
 *                 action_core_count: optional hierarchical ordinary
 *                   shortlist over this many distinct card/play-discard
 *                   cores (1-5).  Remaining room in a hard five-candidate
 *                   budget admits at most one public-information draw-source
 *                   alternative per selected core when its complete-move
 *                   prior clears cand_floor; 0 retains complete-move policy
 *                   ranking.
 *                 exact_terminal: 1 solves one-card-deck decisions exactly at
 *                   the root and in every continuation (default); 2 solves
 *                   the real root only; 3 solves the real root and, in each
 *                   continuation, preserves the ordinary policy's card/action
 *                   while forcing its terminal deck draw.  Modes 2/3 are
 *                   propagation controls; 0 disables both and is retained
 *                   only for low-level regression.
 *                 deck2_replan_worlds / deck2_replan_cores: optional bounded
 *                   recursive information-set search at two or three deck
 *                   cards.  At every late node it compares only the requested
 *                   top policy semantic action cores, using common ordered
 *                   hidden assignments from the mover's sanitized view, and
 *                   recurses through later stalls to the exact one-card leaf.
 *                   The first late panel always receives its configured world
 *                   count (or complete support); deeper panels share a strict
 *                   work/depth bound and report every fallback.  Both zero is
 *                   off; enabling it requires exact_terminal mode 1 and the
 *                   uniform `rolloutu` world model.  A 128-world deck-two
 *                   panel exhausts all at-most-90 ordered assignments.
 *                 bounded_late_root: run the separate finite-support H2/H4
 *                   resolver only at the real root with two or three deck
 *                   cards (0/1).  It enumerates all ordered assignments (at
 *                   most 90/990), carries each particle through the complete
 *                   trajectory without child redeterminization, and either
 *                   authorizes one horizon-stable practical improvement or
 *                   conservatively retains the literal policy baseline.  It
 *                   requires uniform worlds, exact_terminal mode 1, disabled
 *                   root planners, and disabled recursive deck2 replanning.
 *                 bounded_late_min: objective gain a non-policy candidate
 *                   must exceed in both bounded horizons (default 1 point).
 *                   This is intentionally independent of the ordinary
 *                   policy-prefix confirmation thresholds.
 *   rolloutu:PATH[...]                 (same, but uniform world sampling: the
 *                                       ablation for the learned hand beliefs)
 *   rollout2:ROOT_PATH:CONT_PATH[...]  (root policy/value/belief/shortlist use
 *                                       ROOT_PATH; every policy decision after
 *                                       a root candidate uses CONT_PATH)
 *   rolloutu2:ROOT_PATH:CONT_PATH[...] (same two-network actor with uniform
 *                                       hidden-world sampling)
 *   mcts:PATH[:dets[:sims[:root_width[:node_width[:symmetries]]]]]
 *                                      (symmetry averaging is root-only)
 */
#ifndef SPEC_H
#define SPEC_H

#include "agent.h"

/*
 * Strongest locked play-time configuration.  The 20-way policy acts directly
 * for the first 14 actions of each round.  Beginning at zero-based round ply
 * 14 (the 15th action), a 512-world primary panel compares at most five policy
 * moves with at least 2% prior.  If it proposes a different
 * leader, both the primary worlds and a fresh balanced 512-world panel give
 * the two players independently stratified suit mappings that remain fixed
 * for each complete trajectory; both panels must select the same move.  The
 * exact one-card-deck solver is intrinsic to rollout and is also used at the
 * end of every simulated continuation, so earlier search values inherit the
 * optimal final action rather than a policy-network mistake.
 *
 * The visible-hand scheduler and focused semantic candidates remain available
 * for post-hoc review and component experiments, but neither demonstrated an
 * additive gain over this actor in the combined match screen.  Keep
 * user-facing C defaults on this measured configuration.
 */
#define LC_CHAMPION_AGENT_SPEC \
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:3.5:2:4:20:" \
    "0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1"

/* Higher-compute post-game review.  It retains the match-tested ply-14 phase
 * boundary, then spends its ordinary rollout worlds on at most three distinct
 * top-policy card/action cores plus two policy-supported draw alternatives.
 * Its primary and trusted-prefix continuations use the same player-specific,
 * trajectory-coherent suit roles as the promoted live actor.
 * At a real deck depth of two or three, the separate bounded late resolver
 * enumerates the complete ordered 90/990 information-state support, solves
 * two stall horizons and accepts only horizon-consistent practical gains.
 * The final boolean enables that panel independently; the older recursive
 * continuation replanner stays off and cannot multiply or overturn it.
 * This is an audit instrument, not a live self-play override.  Keeping the
 * spec here prevents the UI, analyzer and documentation from silently
 * drifting. */
#define LC_AUDIT_AGENT_SPEC \
    "rolloutu:data/champion.bin:2048:5:0.01:0:1:14:0:0:0:0:3.5:2:4:20:" \
    "0:0:20:1:0:2048:1:0:0:0:0:0:0:3:1:0:0:2:1:0:3:1:0:0:1"

/* Parses spec into *a, loading a network if needed.  Exits on error.  Call
 * spec_release() when the parsed actor's process lifetime does not own it. */
void spec_parse(const char *spec, Agent *a);

/* Release checkpoints owned by spec_parse().  Caller-owned live networks
 * passed to agent_default/spec_parse_selfrollout are never freed.  Aliased
 * root/continuation checkpoints are freed exactly once. */
void spec_release(Agent *a);

/* Parses selfrollout[:ROLLOUT_TAIL] into *a using the caller-owned live
 * network.  This is the training counterpart of rollout:PATH[...] and shares
 * the same complete optional tail; it never loads, frees, or replaces net. */
void spec_parse_selfrollout(const char *spec, const Net *net, Agent *a);

#endif
