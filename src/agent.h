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
    /* AG_MCTS */
    int dets;           /* determinizations                                */
    int sims;           /* simulations per determinization                 */
    int root_width;     /* root moves kept after prior pruning             */
    int node_width;     /* interior moves kept                             */
    float cpuct;
    float cand_floor;   /* AG_ROLLOUT: ignore candidates below this policy  */
    int min_cand;       /* AG_ROLLOUT: but always keep at least this many --
                           a sharp prior otherwise leaves the search a single
                           candidate, able to confirm the policy but never to
                           correct it (0/1 = floor applies unconditionally) */
    int ply_lo, ply_hi; /* AG_ROLLOUT: search only when
                           ply_lo <= nply (< ply_hi if ply_hi > 0); outside
                           the window the raw policy plays.  For measuring
                           WHERE in a round the search actually earns its
                           keep (0,0 = search everywhere) */
    int eval_cand;      /* AG_ROLLOUT: evaluate (and report in stats) at
                           least this many candidates, but selection stays
                           restricted to the floor-passing set -- analysis
                           gets Q values for written-off moves without the
                           measured strength cost of letting 96-world noise
                           overrule a near-certain policy (0 = off) */
    int win_q;          /* AG_ROLLOUT: in the final round, select by match
                           wins over the playouts (margin as tiebreak)
                           instead of by margin.  Principled -- the last
                           round's playouts decide the match exactly -- but
                           measured NO BETTER than margin selection (48.0%
                           +- 2.0% head-to-head, 300 pairs): decided finals
                           tie on win%, close finals make a 96-world win
                           fraction a noisy binomial, and the win-trained
                           policy already carries the clutch behaviour.
                           Default off; the win fraction is still computed
                           and reported (SearchStats.qw) either way. */
    int prune_dom;      /* AG_ROLLOUT: drop discards dominated by a dead-card
                           discard (lc_discard_dominated) from candidates and
                           playout argmax -- frees candidate slots and stops
                           playouts gifting live cards when a dead one is in
                           hand */
    float override_k;   /* AG_ROLLOUT: let an advisory (eval_cand) candidate
                           take the move when it beats the eligible best by
                           more than this many paired standard errors
                           (0 = advisory candidates never selected) */
    int playout_sample; /* AG_ROLLOUT: sample the policy in playouts instead
                           of argmaxing it (common per-world seeds keep the
                           candidate comparison paired).  Argmax repeats
                           every knife-edge downstream decision across all
                           worlds, which can manufacture large fake Q gaps
                           with tiny paired errors; sampling trades a
                           little variance for unbiasedness. */
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

/* Policy head evaluated on st for the player to move.  Fills mv[] with the
 * legal moves and prob[] with the normalized policy; returns the count. */
int  policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value);

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
