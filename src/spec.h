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
 *            [:draw_playout_deck_max]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
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
 *                   2 additionally requires the same leader on a fresh,
 *                   balanced fixed-world-symmetry panel; 3 uses that same
 *                   consensus rule with independently stratified, coherent
 *                   suit mappings for the two players.  Added low-prior
 *                   challengers always retain both statistical gates.
 *                 belief_alpha: strength/temperature of the coherent
 *                   fixed-cardinality opponent-hand posterior (default 1;
 *                   0 is the exact uniform prior).
 *                 draw_root_deck_max / draw_playout_deck_max: independently
 *                   repair the chosen semantic action's draw source at the
 *                   deployed root and inside rollout continuations.  Each
 *                   threshold is 0 (off) or a remaining-deck count.
 *   rolloutu:PATH[...]                 (same, but uniform world sampling: the
 *                                       ablation for the learned hand beliefs)
 *   mcts:PATH[:dets[:sims[:root_width[:node_width[:symmetries]]]]]
 *                                      (symmetry averaging is root-only)
 */
#ifndef SPEC_H
#define SPEC_H

#include "agent.h"

/*
 * Strongest locked play-time configuration.  The 20-way policy acts directly
 * before round ply 14.  From ply 14, a 512-world primary panel compares at
 * most five policy moves with at least 2% prior.  If it proposes a different
 * leader, a fresh balanced 512-world panel keeps one suit mapping fixed for
 * each complete trajectory; both panels must select the same move.  The exact
 * one-card-deck rule is intrinsic to rollout.
 *
 * The visible-hand scheduler and focused semantic candidates remain available
 * for post-hoc review and component experiments, but neither demonstrated an
 * additive gain over this actor in the combined match screen.  Keep
 * user-facing C defaults on this measured configuration.
 */
#define LC_CHAMPION_AGENT_SPEC \
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:3.5:2:2:20:" \
    "0:0:20:1:0:512:1:0:0:0:0:0:0:2"

/* Parses spec into *a, loading a network if needed.  Exits on error. */
void spec_parse(const char *spec, Agent *a);

/* Parses selfrollout[:ROLLOUT_TAIL] into *a using the caller-owned live
 * network.  This is the training counterpart of rollout:PATH[...] and shares
 * the same complete optional tail; it never loads, frees, or replaces net. */
void spec_parse_selfrollout(const char *spec, const Net *net, Agent *a);

#endif
