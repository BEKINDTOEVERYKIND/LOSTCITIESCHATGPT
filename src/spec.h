/* spec.h -- agent command line specs.
 *
 *   random
 *   heur
 *   net:PATH[:draw_samples]
 *   policy:PATH[:temperature]
 *   hrollout[:worlds[:candidates]]     (no network: heuristic + PIMC)
 *   rollout:PATH[:worlds[:candidates[:policy_floor[:gate]]]]
 *                                      (gate: skip search when the policy's
 *                                       top probability is already >= gate)
 *   rolloutu:PATH[...]                 (same, but uniform world sampling: the
 *                                       ablation for the learned hand beliefs)
 *   mcts:PATH[:dets[:sims[:root_width[:node_width]]]]
 */
#ifndef SPEC_H
#define SPEC_H

#include "agent.h"

/* Parses spec into *a, loading a network if needed.  Exits on error. */
void spec_parse(const char *spec, Agent *a);

#endif
