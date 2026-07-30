/* spec.h -- agent command line specs.
 *
 *   random
 *   heur
 *   net:PATH[:draw_samples]
 *   policy:PATH[:temperature[:symmetries]]
 *   hrollout[:worlds[:candidates]]     (no network: heuristic + PIMC)
 *   rollout:PATH[:worlds[:candidates[:policy_floor[:gate[:min_candidates
 *            [:ply_lo[:ply_hi[:eval_candidates[:objective[:prune
 *            [:override_k[:override_min[:sample[:symmetries
 *            [:policy_mass[:batch_worlds[:playout_symmetries]]]]]]]]]]]]]]]]]
 *                 objective: 0 margin; 1 final match result; 2 final hybrid
 *                 symmetries: 1, 5, 10, 20, or 120 exact suit relabellings
 *                 policy_mass: shortest top-policy prefix to cover this mass
 *                 batch_worlds: adaptive paired batch; worlds becomes the cap
 *                 playout_symmetries: exact ensemble at every continuation
 *                   decision (1, 5, 10, 20, or 120)
 *   rolloutu:PATH[...]                 (same, but uniform world sampling: the
 *                                       ablation for the learned hand beliefs)
 *   mcts:PATH[:dets[:sims[:root_width[:node_width[:symmetries]]]]]
 *                                      (symmetry averaging is root-only)
 */
#ifndef SPEC_H
#define SPEC_H

#include "agent.h"

/* Parses spec into *a, loading a network if needed.  Exits on error. */
void spec_parse(const char *spec, Agent *a);

#endif
