/* spec.h -- agent command line specs.
 *
 *   random
 *   heur
 *   net:PATH[:draw_samples]
 *   policy:PATH[:temperature[:symmetries[:plan_deck_max[:plan_block_gap]]]]
 *   hrollout[:worlds[:candidates]]     (no network: heuristic + PIMC)
 *   rollout:PATH[:worlds[:candidates[:policy_floor[:gate[:min_candidates
 *            [:ply_lo[:ply_hi[:eval_candidates[:objective[:prune
 *            [:override_k[:override_min[:sample[:symmetries
 *            [:policy_mass[:batch_worlds[:playout_symmetries
 *            [:discard_guard[:deck_max[:confirm_worlds
 *            [:playout_prune[:plan_deck_max[:plan_block_gap
 *            [:semantic_candidates]]]]]]]]]]]]]]]]]]]]]]]]
 *                 objective: 0 margin; 1 final match result; 2 final hybrid
 *                 symmetries: 1, 5, 10, 20, or 120 exact suit relabellings
 *                 sample: continuation mode 0 exact-group argmax, 1 random-
 *                   group policy sample, 2 random-group argmax
 *                 policy_mass: shortest top-policy prefix to cover this mass
 *                 batch_worlds: adaptive paired batch; worlds becomes the cap
 *                 playout_symmetries: suit group for continuations.  Mode 0
 *                   averages it exactly; modes 1/2 draw one group member per
 *                   decision, then mode 1 samples the action and mode 2 takes
 *                   argmax (1, 5, 10, 20, or 120).
 *                 discard_guard: block a questionable-discard challenger
 *                   without removing it from the audit (0 or 1)
 *                 deck_max: search only at or below this deck count (0 off)
 *                 confirm_worlds: fixed fresh stochastic confirmation worlds
 *                   (0 uses the configured primary-world cap)
 *                 playout_prune: continuation-only dead-discard focus;
 *                   -1 follows root prune, 0 disables, 1 enables.  This can
 *                   improve the world model without hiding root candidates.
 *                 plan_deck_max / plan_block_gap: enable the exact visible-
 *                   hand scheduler at or below this deck count; at the root,
 *                   require this much lower-card option preservation.  A
 *                   positive gap enables full schedule normalization inside
 *                   rollout continuations (0/0 disables both).
 *                 semantic_candidates: add only targeted draw variants for
 *                   top policy actions and a one-sided-wager discard;
 *                   an early wager trigger compares only that move with the
 *                   baseline and receives 16,384 worlds
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
 * before round ply 20.  From ply 20, uniform-world rollout may replace the
 * baseline only after a 512-world primary comparison and a fresh 512-world
 * confirmation.  The exact one-card-deck rule is intrinsic to rollout.
 *
 * The visible-hand scheduler and focused semantic candidates remain available
 * for post-hoc review and component experiments, but neither demonstrated an
 * additive gain over this actor in the combined match screen.  Keep
 * user-facing C defaults on this measured configuration.
 */
#define LC_CHAMPION_AGENT_SPEC \
    "rolloutu:data/champion.bin:512:4:0.02:0:1:20:0:0:0:0:3.5:2:2:20:" \
    "0:0:20:1:0:512:1"

/* Parses spec into *a, loading a network if needed.  Exits on error. */
void spec_parse(const char *spec, Agent *a);

#endif
