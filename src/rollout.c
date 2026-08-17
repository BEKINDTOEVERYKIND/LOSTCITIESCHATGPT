/* rollout.c -- policy improvement by playing candidate moves out.
 *
 * The policy network is sharp, so a turn usually has two to four moves worth
 * considering.  For each of them we sample a world consistent with what the
 * mover knows -- the opponent's hand and the deck order -- and play the game to
 * the end with the same policy driving both seats, then compare the final
 * margins.
 *
 * Two properties make this work where the value network did not:
 *
 *  - The estimate comes from real finished games, so it never inherits the
 *    value head's inability to separate moves that differ by a point or two.
 *  - Each sampled world is shared by every candidate, and the playouts are
 *    deterministic given the world, so the *difference* between candidates is
 *    measured on identical futures.  That pairing is what makes a few hundred
 *    samples enough to resolve small differences.
 *
 * Rollouts also avoid the strategy fusion that spoils determinized tree search:
 * inside a sampled world each side still chooses from its own information set,
 * because the policy only ever sees the features of the player to move.
 */
#include "search.h"
#include "agent.h"
#include "heuristic.h"
#include "late_resolver.h"
#include "planner.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CAND 8
#define URGENT_SEMANTIC_WORLDS 16384
#define CONFIRM_POLICY_MASS 0.995
#define ACTION_SHORTLIST_MAX 5
#define LATE_REPLAN_MAX_DECK 3
#define LATE_REPLAN_MAX_DEPTH 4
#define LATE_REPLAN_BUDGET_FACTOR 30
#define LATE_REPLAN_MIN_BUDGET 512
#define LATE_REPLAN_MIN_RECURSIVE_WORLDS 8
#define LATE_REPLAN_CACHE_SIZE 256
#define LATE_REPLAN_MAX_UNSEEN (HAND_SIZE + LATE_REPLAN_MAX_DECK)
#define LATE_REPLAN_MAX_ASSIGNMENTS                                      \
    (LATE_REPLAN_MAX_UNSEEN * (LATE_REPLAN_MAX_UNSEEN - 1) *            \
     (LATE_REPLAN_MAX_UNSEEN - 2))

typedef struct {
    uint64_t exact_terminal_leaves;
    uint64_t unfinished_cap_leaves;
    uint64_t cycle_breaks;
    uint64_t cap_reserve_forces;
    uint64_t deck2_replans;
    uint64_t deck2_replan_worlds;
    uint64_t deck2_replan_evals;
    uint64_t deck2_replan_cap_hits;
    uint64_t deck2_replan_cache_hits;
    uint64_t deck2_replan_cycle_closures;
    uint64_t deck2_replan_max_depth;
    uint64_t deck2_replan_root_calls;
    uint64_t deck2_replan_root_worlds;
    uint64_t deck2_replan_max_stall_chain;
    uint64_t deck2_replan_low_world_fallbacks;
} PlayoutWork;

typedef struct {
    uint64_t key;
    uint64_t budget;
    uint64_t logical_cost;
    double objective;
    Move move;
    uint8_t depth;
    uint8_t status; /* 0 empty, 1 being evaluated, 2 complete */
} LateReplanCacheEntry;

typedef struct {
    LateReplanCacheEntry entry[LATE_REPLAN_CACHE_SIZE];
} LateReplanCache;

typedef struct {
    State view;
    uint8_t unseen[LATE_REPLAN_MAX_UNSEEN];
    int nunseen;
    int ndeck;
    int support;
} LateAssignmentPlan;

/* One context belongs to one top-level continuation trajectory; its cache
 * belongs to the enclosing primary or confirmation panel and may be shared by
 * many such contexts.  Recursive candidate branches receive equal budget
 * slices and reuse only completed semantic decisions.  This makes the work
 * bound independent of candidate iteration order while avoiding repeated
 * information-state panels across outer determinizations. */
typedef struct {
    uint64_t eval_budget;
    uint64_t path_key[LATE_REPLAN_MAX_DEPTH];
    int depth;
    int path_n;
    int stall_chain;
    int hypothetical;
    int exhausted_reported;
    int panel_domain;
    LateReplanCache *cache;
} LateReplanContext;

enum {
    LATE_REPLAN_UNAVAILABLE = 0,
    LATE_REPLAN_SELECTED = 1,
    LATE_REPLAN_PATH_CYCLE = -1,
    LATE_REPLAN_EXHAUSTED = -2,
    LATE_REPLAN_LOW_WORLDS = -3
};

/* A rollout may otherwise shuttle public pile cards forever without reducing
 * the deck.  Progress behavior must be a function of the player-to-move's
 * information, never of the sampled opponent hand or deck order.  Store the
 * semantic information-state identities seen on this trajectory; nply is
 * intentionally absent so a literal shuttle cannot evade detection. */
typedef struct {
    uint64_t information_key[LC_MAX_PLIES];
    int n;
} LateCycleHistory;

static uint64_t late_information_hash(const State *st, int p);
static uint64_t late_decision_seed(const State *st, int p, int panel_domain);

static const uint64_t ALL_WAGER_BITS =
    (UINT64_C(7) << 0) | (UINT64_C(7) << 12) |
    (UINT64_C(7) << 24) | (UINT64_C(7) << 36) |
    (UINT64_C(7) << 48);

static int same_semantic_card_id(uint8_t a, uint8_t b)
{
    return a == b || (CARD_IS_WAGER(a) && CARD_IS_WAGER(b) &&
                      CARD_SUIT(a) == CARD_SUIT(b));
}

static int same_semantic_card_set(uint64_t a, uint64_t b)
{
    if ((a & ~ALL_WAGER_BITS) != (b & ~ALL_WAGER_BITS)) return 0;
    for (int s = 0; s < NSUIT; s++) {
        uint64_t mask = UINT64_C(7) << (s * NRANK);
        if (__builtin_popcountll(a & mask) != __builtin_popcountll(b & mask))
            return 0;
    }
    return 1;
}

int rollout_same_late_state(const State *a, const State *b)
{
    if (a->deck_pos != b->deck_pos || a->deck_left != b->deck_left)
        return 0;
    for (int i = 0; i < a->deck_left; i++)
        if (!same_semantic_card_id(a->deck[a->deck_pos + i],
                                   b->deck[b->deck_pos + i]))
            return 0;
    if (!same_semantic_card_set(a->hand[0], b->hand[0]) ||
        !same_semantic_card_set(a->hand[1], b->hand[1]) ||
        !same_semantic_card_set(a->played[0], b->played[0]) ||
        !same_semantic_card_set(a->played[1], b->played[1]) ||
        !same_semantic_card_set(a->discarded, b->discarded) ||
        !same_semantic_card_set(a->known[0], b->known[0]) ||
        !same_semantic_card_set(a->known[1], b->known[1]))
        return 0;
    if (memcmp(a->pile_n, b->pile_n, sizeof a->pile_n) != 0) return 0;
    for (int s = 0; s < NSUIT; s++)
        for (int i = 0; i < a->pile_n[s]; i++)
            if (!same_semantic_card_id(a->pile[s][i], b->pile[s][i]))
                return 0;
    return memcmp(a->hand_n, b->hand_n, sizeof a->hand_n) == 0 &&
           memcmp(a->exp_wager, b->exp_wager, sizeof a->exp_wager) == 0 &&
           memcmp(a->exp_top, b->exp_top, sizeof a->exp_top) == 0 &&
           memcmp(a->exp_n, b->exp_n, sizeof a->exp_n) == 0 &&
           memcmp(a->exp_sum, b->exp_sum, sizeof a->exp_sum) == 0 &&
           a->turn == b->turn &&
           a->round == b->round &&
           memcmp(a->cum, b->cum, sizeof a->cum) == 0;
}

static int late_cycle_repeated(LateCycleHistory *history, const State *st)
{
    if (!history || st->deck_left > 3) return 0;
    uint64_t key = late_information_hash(st, st->turn);
    for (int i = 0; i < history->n; i++)
        if (history->information_key[i] == key) return 1;
    if (history->n < LC_MAX_PLIES)
        history->information_key[history->n++] = key;
    return 0;
}

static int same_semantic_action(Move a, Move b);

static int playout(const Net *net, const NetEvalPlan *eval_plan,
                   State *s, int p, int prune, Rng *symrng,
                   const uint8_t fixed_perm[NSUIT],
                   const uint8_t other_perm[NSUIT], int other_player,
                   int sample_actions, int symmetries,
                   int plan_deck_max, int plan_block_gap,
                   int draw_deck_max, float confirm_temp,
                   uint64_t confirm_seed, int objective, int exact_terminal,
                   int deck2_replan_worlds, int deck2_replan_cores,
                   LateReplanContext *late, double *winpts,
                   PlayoutWork *work);

static int late_replan_choice(
    const Net *net, const NetEvalPlan *eval_plan,
    const State *st, const Move *mv, const float *score, int n,
    int prune, int random_sym,
    const uint8_t fixed_perm[NSUIT],
    const uint8_t other_perm[NSUIT], int other_player,
    int sample_actions, int symmetries,
    int plan_deck_max, int plan_block_gap, int draw_deck_max,
    float confirm_temp, int objective, int exact_terminal,
    int worlds, int cores, LateReplanContext *late,
    Move *out_move, double *out_objective,
    PlayoutWork *work);

static LateReplanContext late_replan_context_init(
    int worlds, int cores, int panel_domain, LateReplanCache *cache)
{
    LateReplanContext late;
    memset(&late, 0, sizeof late);
    if (cores > 3) cores = 3;
    if (cores < 1) cores = 1;
    uint64_t variants = (uint64_t)(2 * cores);
    uint64_t budget = worlds > 0 ? (uint64_t)worlds : UINT64_C(1);
    if (budget > UINT64_MAX / variants)
        budget = UINT64_MAX;
    else
        budget *= variants;
    if (budget <= UINT64_MAX / LATE_REPLAN_BUDGET_FACTOR)
        budget *= LATE_REPLAN_BUDGET_FACTOR;
    else
        budget = UINT64_MAX;
    if (budget < LATE_REPLAN_MIN_BUDGET)
        budget = LATE_REPLAN_MIN_BUDGET;
    late.eval_budget = budget;
    late.panel_domain = panel_domain;
    late.depth = LATE_REPLAN_MAX_DEPTH;
    late.cache = cache;
    return late;
}

int rollout_exact_terminal_choice(const State *st, const Move *mv,
                                  const float *prior, int n, int objective,
                                  double *out_objective)
{
    if (st->deck_left != 1 || st->over || n <= 0) return -1;
    const int p = st->turn;
    int best = -1;
    double best_objective = -INFINITY;
    for (int i = 0; i < n; i++) {
        /* Drawing the last deck card ends the round immediately.  For every
         * semantic card/action core this is the guaranteed branch: taking a
         * pile instead gives the opponent the option to end after making its
         * own best play and therefore cannot improve the mover's minimax
         * result.  Evaluating every deck variant is both exact and cheaper
         * than one network rollout. */
        if (mv[i].draw != 0) continue;
        State terminal = *st;
        lc_apply(&terminal, mv[i]);
        if (!terminal.over || terminal.deck_left != 0) continue;
        double value = rollout_terminal_objective(&terminal, p, objective);
        float pi = prior ? prior[i] : 0.0f;
        float best_pi = best >= 0 && prior ? prior[best] : 0.0f;
        if (best < 0 || value > best_objective ||
            (value == best_objective && pi > best_pi) ||
            (value == best_objective && pi == best_pi &&
             MOVE_PACK(mv[i]) < MOVE_PACK(mv[best]))) {
            best = i;
            best_objective = value;
        }
    }
    if (best >= 0 && out_objective) *out_objective = best_objective;
    return best;
}

int rollout_policy_terminal_choice(const Move *mv, int n, int selected)
{
    if (selected < 0 || selected >= n) return -1;
    for (int i = 0; i < n; i++)
        if (mv[i].draw == 0 && same_semantic_action(mv[i], mv[selected]))
            return i;
    return -1;
}

int rollout_policy_deck_choice(const State *st, const Move *mv,
                               const float *score, int n, uint64_t dead)
{
    if (!st || !mv || !score || n <= 0) return -1;
    for (int pass = 0; pass < 2; pass++) {
        int best = -1;
        float best_score = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (mv[i].draw != 0) continue;
            if (pass == 0 && dead &&
                lc_discard_dominated(st, mv[i], dead))
                continue;
            float candidate_score = lc_float_isfinite(score[i])
                ? score[i] : -INFINITY;
            if (best < 0 || candidate_score > best_score ||
                (candidate_score == best_score &&
                 MOVE_PACK(mv[i]) < MOVE_PACK(mv[best]))) {
                best = i;
                best_score = candidate_score;
            }
        }
        if (best >= 0 || !dead) return best;
    }
    return -1;
}

/* Rank the legal moves of s for the player to move.  With a network that is
 * the policy head; without one it is the hand-crafted evaluation, which gives
 * the classical "heuristic + perfect-information Monte Carlo" baseline. */
static int rank_moves(const Net *net, const NetEvalPlan *eval_plan,
                      const State *s, Move *mv, float *score,
                      int symmetries, Rng *symrng,
                      const uint8_t fixed_perm[NSUIT])
{
    if (net) {
        if (fixed_perm)
            return policy_probs_perm_plan(
                net, s, mv, score, NULL, fixed_perm, eval_plan);
        if (symrng && symmetries > 1)
            return policy_probs_random_sym_plan(
                net, s, mv, score, symrng, symmetries, eval_plan);
        return policy_probs_sym_plan(
            net, s, mv, score, NULL, symmetries, eval_plan);
    }
    int n = lc_moves(s, mv);
    for (int i = 0; i < n; i++) score[i] = heur_move_value_det(s, mv[i]);
    return n;
}

static int rank_playout_moves(
    const Net *net, const NetEvalPlan *eval_plan,
    const State *s, Move *mv, float *score,
    int symmetries, Rng *symrng, const uint8_t fixed_perm[NSUIT],
    int exact_late_ensemble)
{
    if (exact_late_ensemble)
        return rank_moves(
            net, eval_plan, s, mv, score, symmetries, NULL, NULL);
    return rank_moves(
        net, eval_plan, s, mv, score, symmetries, symrng, fixed_perm);
}

static int late_exact_ensemble_enabled(
    int late_replan_enabled, const LateReplanContext *late)
{
    return late_replan_enabled && late && !late->hypothetical;
}

static uint64_t confirmation_mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x *= UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

int rollout_near_greedy_pick(const Move *mv, const float *prob, int n,
                             float temperature, uint64_t seed,
                             int depth, int player)
{
    if (n <= 0) return -1;
    int order[MAX_MOVES];
    int argmax = 0;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        order[i] = i;
        if (prob[i] > prob[argmax]) argmax = i;
        if (prob[i] > 0.0f && isfinite(prob[i])) total += prob[i];
    }
    if (!(temperature > 0.0f) || !isfinite(temperature) || !(total > 0.0))
        return argmax;

    for (int i = 0; i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (prob[order[j]] > prob[order[best]]) best = j;
        int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
    }
    int keep = 0;
    double mass = 0.0;
    while (keep < n && mass / total < CONFIRM_POLICY_MASS) {
        int i = order[keep++];
        if (prob[i] > 0.0f && isfinite(prob[i])) mass += prob[i];
    }
    /* Do not make an arbitrary legal-move ordering decide which member of an
     * exact probability tie is allowed into the confirmation policy. */
    if (keep > 0) {
        float cutoff = prob[order[keep - 1]];
        while (keep < n && prob[order[keep]] == cutoff) keep++;
    }
    if (keep < 1) return argmax;

    int chosen = order[0];
    double best_utility = -INFINITY;
    for (int k = 0; k < keep; k++) {
        int i = order[k];
        if (!(prob[i] > 0.0f) || !isfinite(prob[i])) continue;
        /* Physical wager copies are one observable card type.  Canonicalize
         * the packed key so an otherwise identical information state cannot
         * receive different confirmation noise merely because the engine
         * happens to hold wager copy 0, 1, or 2. */
        Move semantic = mv[i];
        if (CARD_IS_WAGER(semantic.card))
            semantic.card = (uint8_t)CARD_MAKE(CARD_SUIT(semantic.card), 0);
        uint64_t key = seed
            ^ (UINT64_C(0x9E3779B97F4A7C15) * (uint64_t)(depth + 1))
            ^ (UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(player + 1))
            ^ (UINT64_C(0x94D049BB133111EB) *
               (uint64_t)(MOVE_PACK(semantic) + 1));
        uint64_t bits = confirmation_mix64(key);
        double u = ((double)(bits >> 11) + 0.5) /
                   9007199254740992.0;
        double gumbel = -log(-log(u));
        double utility = log((double)prob[i] / total) /
                         (double)temperature + gumbel;
        if (utility > best_utility) {
            best_utility = utility;
            chosen = i;
        }
    }
    return chosen;
}

/* Play s out to the end of the round, returning the round margin for player
 * p.  In the final round of a match the round's end decides the match, so
 * *winpts gets the match result (1 win, 0.5 draw, 0 loss) from the carried
 * cumulative totals; in earlier rounds it gets -1 (margin is the only
 * available objective there, and it doubles as the natural proxy).
 * symrng without fixed_perm draws one suit-group member at each downstream
 * decision.  fixed_perm instead retains one member for the full sampled
 * world, avoiding a temporally inconsistent change of network orientation.
 * sample_actions is a separate robustness ablation: it samples from that
 * member's full policy rather than taking its best move.  Conflating these
 * two sources of randomness made the old fast mode evaluate a much weaker,
 * high-entropy continuation policy instead of approximating the champion. */
static int playout(const Net *net, const NetEvalPlan *eval_plan,
                   State *s, int p, int prune, Rng *symrng,
                   const uint8_t fixed_perm[NSUIT],
                   const uint8_t other_perm[NSUIT], int other_player,
                   int sample_actions,
                   int symmetries, int plan_deck_max, int plan_block_gap,
                   int draw_deck_max, float confirm_temp,
                   uint64_t confirm_seed, int objective, int exact_terminal,
                   int deck2_replan_worlds, int deck2_replan_cores,
                   LateReplanContext *late, double *winpts,
                   PlayoutWork *work)
{
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    /* Only entries below n are ever read.  Zeroing the full history here
     * clears roughly 100 KiB for every sampled continuation and needlessly
     * dominates high-world audits. */
    LateCycleHistory cycle_history;
    cycle_history.n = 0;
    int depth = 0;
    while (!s->over) {
        if (exact_terminal == 1 && s->deck_left == 1) {
            /* The last turn is a rules calculation.  Asking the 20-way
             * network ensemble to rank it first cannot change the objective
             * and dominated the cost of bounded deck-two replanning.  Equal
             * terminal scores are interchangeable, so the exact solver's
             * stable packed-move tie break is sufficient here. */
            int terminal_n = lc_moves(s, mv);
            int exact = rollout_exact_terminal_choice(
                s, mv, NULL, terminal_n, objective, NULL);
            if (exact >= 0) {
                lc_apply(s, mv[exact]);
                if (work) work->exact_terminal_leaves++;
                depth++;
                continue;
            }
        }
        /* LC_MAX_PLIES is only a defensive engine fuse.  Reserve exactly one
         * remaining turn per deck card before reaching it, so a pathological
         * but non-repeating pile walk still finishes by the real deck rule
         * instead of being assigned a fake terminal score.  The maintained
         * greedy policy is conditioned on that required deck draw; optional
         * sampled/planned controls preserve their semantic action.  Because
         * this rule is inside every playout, earlier candidate values see the
         * same completed continuation. */
        int force_cap_reserve =
            (int)s->nply + (int)s->deck_left >= LC_MAX_PLIES;
        int force_cycle_deck = late_cycle_repeated(&cycle_history, s);
        const uint8_t *turn_perm =
            other_perm && s->turn == other_player ? other_perm : fixed_perm;
        const uint8_t *replan_fixed_perm = fixed_perm;
        const uint8_t *replan_other_perm = other_perm;
        int replan_other_player = other_player;
        Rng *rank_rng = symrng;
        int late_replan_enabled =
            !force_cycle_deck && !force_cap_reserve && net &&
            deck2_replan_worlds > 0 && deck2_replan_cores > 0 &&
            s->deck_left >= 2 && s->deck_left <= LATE_REPLAN_MAX_DECK && late;
        /* A sampled suit orientation is part of the modeled policy.  At a
         * hypothetical late node it must be selected from the mover's
         * information state, not from an outer-world RNG position that can
         * encode hidden cards.  Actual selected-path nodes use the complete
         * ensemble below; bounded descendants use one deterministic member of
         * that same group and therefore remain cacheable and information-safe. */
        uint8_t state_keyed_perm[NSUIT];
        if (late_replan_enabled && late->hypothetical && symmetries > 1) {
            uint8_t perms[120][NSUIT];
            int nperm = suit_permutations(symmetries, perms);
            if (nperm > 0) {
                uint64_t state_seed = late_decision_seed(
                    s, s->turn, late->panel_domain);
                int pick = (int)(state_seed % (uint64_t)nperm);
                memcpy(state_keyed_perm, perms[pick], sizeof state_keyed_perm);
                turn_perm = state_keyed_perm;
                replan_fixed_perm = state_keyed_perm;
                replan_other_perm = NULL;
                replan_other_player = s->turn ^ 1;
                rank_rng = NULL;
            }
        }
        /* The deployed policy ranks each real selected-path information state
         * with its complete configured suit ensemble.  Hypothetical candidate
         * descendants already have a top-one-core, minimum-eight-world bound;
         * their state-keyed member above avoids multiplying the dominant
         * network cost by the entire suit group without making the policy a
         * function of hidden assignment order. */
        int exact_late_ensemble = late_exact_ensemble_enabled(
            late_replan_enabled, late);
        int n = rank_playout_moves(
            net, eval_plan, s, mv, score, symmetries, rank_rng, turn_perm,
            exact_late_ensemble);
        if (n <= 0) break;
        /* Replan every late information state, including states reached from
         * an earlier root.  Candidate branches recursively use the same rule;
         * the context divides a strict work budget equally among them.  A
         * semantic path cycle is closed through a deck draw.  Exhausting the
         * optional search budget instead falls back to the configured policy:
         * a compute limit is not a game rule and must not manufacture a deck
         * draw that the player would never choose.  Real repeated states and
         * the engine-fuse reserve still guarantee eventual progress. */
        int force_late_cycle = 0;
        if (late_replan_enabled) {
            Move replanned;
            int replanned_status = late_replan_choice(
                net, eval_plan, s, mv, score, n, prune, rank_rng != NULL,
                replan_fixed_perm, replan_other_perm, replan_other_player,
                sample_actions, symmetries,
                plan_deck_max, plan_block_gap, draw_deck_max,
                confirm_temp, objective, exact_terminal,
                deck2_replan_worlds, deck2_replan_cores, late,
                &replanned, NULL, work);
            if (replanned_status == LATE_REPLAN_SELECTED) {
                lc_apply(s, replanned);
                depth++;
                continue;
            }
            force_late_cycle = replanned_status == LATE_REPLAN_PATH_CYCLE;
        }
        uint64_t dead = prune ? (lc_dead_cards(s) & s->hand[s->turn]) : 0;
        int best = -1;
        if (sample_actions && symrng && net) {
            float w[MAX_MOVES];
            float tot = 0.0f;
            for (int i = 0; i < n; i++) {
                w[i] = (dead && lc_discard_dominated(s, mv[i], dead)) ? 0.0f : score[i];
                tot += w[i];
            }
            if (tot > 0.0f) best = sample_index(w, n, symrng);
        }
        if (best < 0 && net && confirm_temp > 0.0f) {
            float w[MAX_MOVES];
            for (int i = 0; i < n; i++)
                w[i] = (dead && lc_discard_dominated(s, mv[i], dead))
                    ? 0.0f : score[i];
            best = rollout_near_greedy_pick(
                mv, w, n, confirm_temp, confirm_seed, depth, s->turn);
        }
        if (best < 0) {
            for (int i = 0; i < n; i++) {
                if (dead && lc_discard_dominated(s, mv[i], dead)) continue;
                if (best < 0 || score[i] > score[best]) best = i;
            }
        }
        /* Continuation values need a horizon-aware lower bound, not another
         * myopic policy step.  Once the deck is short, follow an optimal
         * schedule of cards already visible in the current player's hand.
         * This is information-set safe and normalizes commuting play orders;
         * the production actor remains more conservative at the root. */
        if (!sample_actions && net && plan_deck_max > 0 &&
            plan_block_gap > 0 && s->deck_left <= plan_deck_max) {
            int order[MAX_MOVES];
            for (int i = 0; i < n; i++) order[i] = i;
            int keep = n < 8 ? n : 8;
            for (int i = 0; i < keep; i++) {
                int top = i;
                for (int j = i + 1; j < n; j++)
                    if (score[order[j]] > score[order[top]]) top = j;
                int tmp = order[i]; order[i] = order[top]; order[top] = tmp;
            }
            int planned = hand_plan_choose(
                s, s->turn, mv, score, order, keep,
                (s->deck_left + 1) / 2);
            if (planned >= 0) best = planned;
        }
        /* Repair only the chosen card/action's draw source.  Deck value is
         * averaged across the mover's information set, never read from this
         * determinization's hidden top card.  This is deliberately separate
         * from the optional play-order scheduler above. */
        if (!sample_actions && net && draw_deck_max > 0 &&
            s->deck_left <= draw_deck_max)
            best = hand_plan_choose_draw_source(
                s, s->turn, mv, score, n, best);
        if (best < 0) best = 0;

        int force_progress = force_late_cycle || force_cap_reserve ||
                             force_cycle_deck;
        if (force_progress) {
            int previous = best;
            /* The maintained greedy network actor should choose its best
             * action conditional on the now-required deck draw.  Merely
             * replacing the unrestricted winner's draw source is wrong once
             * card/action x draw interactions are learned.  Optional sampled,
             * near-greedy and visible-hand-planned actors retain their chosen
             * semantic action because that choice represents their configured
             * continuation model.  Mode 3 also deliberately retains it as the
             * propagation control. */
            int conditional_policy =
                net && !sample_actions && !(confirm_temp > 0.0f) &&
                !(plan_deck_max > 0 && plan_block_gap > 0 &&
                  s->deck_left <= plan_deck_max) &&
                !(exact_terminal == 3 && s->deck_left == 1);
            int deck = conditional_policy
                ? rollout_policy_deck_choice(s, mv, score, n, dead)
                : rollout_policy_terminal_choice(mv, n, best);
            if (deck >= 0) best = deck;
            if (previous >= 0 && mv[previous].draw != 0) {
                if (force_cap_reserve) {
                    if (work) work->cap_reserve_forces++;
                } else if (force_cycle_deck || force_late_cycle) {
                    /* Cycle matching treats the three physical wager IDs as
                     * one observable card type and consumes no extra RNG. */
                    if (work) work->cycle_breaks++;
                }
            }
        }

        if (exact_terminal == 3 && s->deck_left == 1) {
            /* Propagation control: retain the ordinary continuation actor's
             * chosen card and play/discard decision, but end through the deck.
             * This removes pathological pile-draw loops from the ablation
             * without granting it the production solver's optimization over
             * all action cores.  Comparing mode 1 with mode 3 therefore tests
             * whether exact terminal values improve decisions made on earlier
             * turns, rather than merely changing the final move itself. */
            int terminal = rollout_policy_terminal_choice(mv, n, best);
            if (terminal >= 0) best = terminal;
        }
        lc_apply(s, mv[best]);
        depth++;
    }
    if (work && s->over && s->deck_left > 0)
        work->unfinished_cap_leaves++;
    int sp = lc_score(s, p), so = lc_score(s, p ^ 1);
    if (winpts) {
        if (s->round == MATCH_ROUNDS - 1) {
            int tp = s->cum[p] + sp, to = s->cum[p ^ 1] + so;
            *winpts = tp > to ? 1.0 : (tp == to ? 0.5 : 0.0);
        } else *winpts = -1.0;
    }
    return sp - so;
}

/* Objective used to compare completed playouts.  Modes:
 *   0: round margin (historical default)
 *   1: pure match result in real round index 2
 *   2: 0.05 * final match margin + 50 * signed match result
 * Rounds 0 and 1 always retain margin semantics, independent of mode.  Mode 2
 * matches the strongest checkpoint's finishing reward while preserving the
 * intentional last-round-only switch to match-winning play. */
double rollout_terminal_objective(const State *terminal, int p, int mode)
{
    int round_margin = lc_score(terminal, p) - lc_score(terminal, p ^ 1);
    if (terminal->round != MATCH_ROUNDS - 1 || mode <= 0)
        return (double)round_margin;

    int total_margin = (int)terminal->cum[p] - (int)terminal->cum[p ^ 1]
                     + round_margin;
    int result = (total_margin > 0) - (total_margin < 0);
    if (mode == 1)
        return 50.0 * (double)result;
    return 0.05 * (double)total_margin + 50.0 * (double)result;
}

static uint64_t late_mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x *= UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static void late_hash_word(uint64_t *h, uint64_t x)
{
    *h = late_mix64(*h ^ (x + UINT64_C(0x9E3779B97F4A7C15)));
}

/* Hash only facts available to p.  Wager IDs are collapsed to per-suit counts
 * because their three physical engine IDs are not observable card identities. */
static void late_hash_semantic_mask(uint64_t *h, uint64_t mask)
{
    for (int s = 0; s < NSUIT; s++) {
        uint64_t wager_mask = UINT64_C(7) << (s * NRANK);
        late_hash_word(h, (uint64_t)__builtin_popcountll(mask & wager_mask));
        for (int r = WAGERS_PER_SUIT; r < NRANK; r++)
            late_hash_word(h, (mask >> CARD_MAKE(s, r)) & UINT64_C(1));
    }
}

static uint64_t late_information_hash(const State *st, int p)
{
    State view;
    agent_information_view(st, p, &view);
    uint64_t h = UINT64_C(0x243F6A8885A308D3);
    late_hash_word(&h, view.turn);
    late_hash_word(&h, view.deck_left);
    /* nply is deliberately absent from the strategic information identity.
     * Including it would let a public pile shuttle evade recursion-path cycle
     * detection forever.  It is included separately in the cache-only key,
     * where cap reserve and nply-dependent network features make it relevant. */
    late_hash_word(&h, view.round);
    late_hash_word(&h, (uint16_t)view.cum[0]);
    late_hash_word(&h, (uint16_t)view.cum[1]);
    late_hash_word(&h, view.hand_n[0]);
    late_hash_word(&h, view.hand_n[1]);
    late_hash_semantic_mask(&h, view.hand[p]);
    late_hash_semantic_mask(&h, view.known[0]);
    late_hash_semantic_mask(&h, view.known[1]);
    late_hash_semantic_mask(&h, view.played[0]);
    late_hash_semantic_mask(&h, view.played[1]);
    late_hash_semantic_mask(&h, view.discarded);
    for (int q = 0; q < 2; q++)
        for (int s = 0; s < NSUIT; s++) {
            late_hash_word(&h, view.exp_wager[q][s]);
            late_hash_word(&h, view.exp_top[q][s]);
            late_hash_word(&h, view.exp_n[q][s]);
            late_hash_word(&h, view.exp_sum[q][s]);
        }
    for (int s = 0; s < NSUIT; s++) {
        late_hash_word(&h, view.pile_n[s]);
        for (int i = 0; i < view.pile_n[s]; i++) {
            int c = view.pile[s][i];
            int semantic_rank = CARD_IS_WAGER(c)
                ? 0 : CARD_RANK(c) - WAGERS_PER_SUIT + 1;
            late_hash_word(&h, (uint64_t)(s * 16 + semantic_rank));
        }
    }
    return h;
}

/* A recursive late-search policy must be a function of the mover's observable
 * information, not of the outer determinization that happened to reach it.
 * Give each independently interpreted panel its own domain, while keeping one
 * seed common to every hidden world and root candidate inside that panel.
 * nply is public and affects both network features and the engine-fuse reserve,
 * so it belongs in the decision seed even though strategic cycle identity
 * deliberately omits it. */
enum {
    LATE_PANEL_PRIMARY = 1,
    LATE_PANEL_TRUSTED_CONFIRM = 2,
    LATE_PANEL_CHALLENGER_CONFIRM = 3,
    LATE_PANEL_PRIMARY_ASSIGNMENTS = 4,
    LATE_PANEL_TRUSTED_ASSIGNMENTS = 5,
    LATE_PANEL_CHALLENGER_ASSIGNMENTS = 6
};

static uint64_t late_decision_seed(const State *st, int p, int panel_domain)
{
    uint64_t h = late_information_hash(st, p);
    late_hash_word(&h, (uint64_t)(uint32_t)st->nply);
    late_hash_word(&h,
        UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(panel_domain + 1));
    return h;
}

static uint64_t late_replan_node_seed(
    const State *st, int p, const LateReplanContext *late)
{
    return late_decision_seed(st, p, late->panel_domain);
}

static uint64_t late_candidate_signature(
    const Move *mv, const float *score, const int *candidate, int ncand,
    const uint8_t fixed_perm[NSUIT],
    const uint8_t other_perm[NSUIT], int other_player)
{
    uint64_t h = UINT64_C(0x13198A2E03707344);
    for (int c = 0; c < ncand; c++) {
        uint32_t bits;
        memcpy(&bits, &score[candidate[c]], sizeof bits);
        late_hash_word(&h, MOVE_PACK(mv[candidate[c]]));
        late_hash_word(&h, bits);
    }
    late_hash_word(&h, (uint64_t)(other_player + 1));
    for (int s = 0; s < NSUIT; s++) {
        late_hash_word(&h, fixed_perm ? (uint64_t)(fixed_perm[s] + 1) : 0);
        late_hash_word(&h, other_perm ? (uint64_t)(other_perm[s] + 1) : 0);
    }
    return h;
}

static LateReplanCacheEntry *late_cache_entry(
    LateReplanCache *cache, uint64_t key)
{
    if (!cache) return NULL;
    uint64_t start = key % LATE_REPLAN_CACHE_SIZE;
    for (uint64_t step = 0; step < LATE_REPLAN_CACHE_SIZE; step++) {
        LateReplanCacheEntry *entry =
            &cache->entry[(start + step) % LATE_REPLAN_CACHE_SIZE];
        if (entry->status == 0 || entry->key == key) return entry;
    }
    return NULL;
}

static int late_path_contains(const LateReplanContext *late, uint64_t key)
{
    for (int i = 0; late && i < late->path_n; i++)
        if (late->path_key[i] == key) return 1;
    return 0;
}

static uint64_t late_path_signature(const LateReplanContext *late)
{
    uint64_t h = UINT64_C(0xA4093822299F31D0);
    late_hash_word(&h, (uint64_t)late->path_n);
    for (int i = 0; i < late->path_n; i++) {
        /* Preserve path order: the same set of ancestors reached in a
         * different sequence can encounter a different first cycle cutoff. */
        late_hash_word(&h, (uint64_t)(i + 1));
        late_hash_word(&h, late->path_key[i]);
    }
    late_hash_word(&h, (uint64_t)late->stall_chain);
    late_hash_word(&h, (uint64_t)late->hypothetical);
    return h;
}

static int late_falling_factorial(int n, int k)
{
    int result = 1;
    for (int i = 0; i < k; i++) result *= n - i;
    return result;
}

static void late_assignment_cards(
    int rank, int nunseen, int ndeck, int chosen[LATE_REPLAN_MAX_DECK])
{
    int available[LATE_REPLAN_MAX_UNSEEN];
    for (int i = 0; i < nunseen; i++) available[i] = i;
    int navailable = nunseen;
    for (int d = 0; d < ndeck; d++) {
        int block = late_falling_factorial(
            navailable - 1, ndeck - d - 1);
        int pick = block > 0 ? rank / block : 0;
        if (block > 0) rank %= block;
        chosen[d] = available[pick];
        for (int i = pick + 1; i < navailable; i++)
            available[i - 1] = available[i];
        navailable--;
    }
}

static int late_assignment_plan_init(
    const State *st, int p, LateAssignmentPlan *plan)
{
    if (!st || !plan || p < 0 || p > 1 || st->deck_left < 2 ||
        st->deck_left > LATE_REPLAN_MAX_DECK)
        return 0;
    memset(plan, 0, sizeof *plan);
    agent_information_view(st, p, &plan->view);
    lc_unseen(&plan->view, p, plan->unseen, &plan->nunseen);
    int opponent_need = (int)plan->view.hand_n[p ^ 1] -
                        __builtin_popcountll(plan->view.known[p ^ 1]);
    plan->ndeck = st->deck_left;
    if (plan->nunseen != opponent_need + plan->ndeck ||
        plan->nunseen < plan->ndeck ||
        plan->nunseen > LATE_REPLAN_MAX_UNSEEN)
        return 0;
    plan->support = late_falling_factorial(plan->nunseen, plan->ndeck);
    return plan->support > 0 &&
           plan->support <= LATE_REPLAN_MAX_ASSIGNMENTS;
}

static void late_assignment_world(
    const LateAssignmentPlan *plan, int p, int rank, State *world)
{
    int chosen[LATE_REPLAN_MAX_DECK] = { 0 };
    late_assignment_cards(
        rank, plan->nunseen, plan->ndeck, chosen);
    *world = plan->view;
    world->hand[p ^ 1] = plan->view.known[p ^ 1];
    for (int i = 0; i < plan->nunseen; i++) {
        int in_deck = 0;
        for (int k = 0; k < plan->ndeck; k++)
            if (i == chosen[k]) in_deck = 1;
        if (!in_deck)
            world->hand[p ^ 1] |= UINT64_C(1) << plan->unseen[i];
    }
    world->deck_pos = 0;
    memset(world->deck, 0, sizeof world->deck);
    for (int k = 0; k < plan->ndeck; k++)
        world->deck[k] = plan->unseen[chosen[k]];
    world->deck_left = (uint8_t)plan->ndeck;
}

static int late_unique_assignment_order(
    const State *st, int p, int requested, int panel_domain,
    LateAssignmentPlan *plan,
    int order[LATE_REPLAN_MAX_ASSIGNMENTS])
{
    if (requested <= 0 ||
        !late_assignment_plan_init(st, p, plan))
        return 0;
    for (int i = 0; i < plan->support; i++) order[i] = i;
    Rng panel_rng;
    rng_seed(&panel_rng, late_decision_seed(st, p, panel_domain));
    int count = requested < plan->support ? requested : plan->support;
    for (int i = 0; i < count; i++) {
        int j = i + (int)rng_below(
            &panel_rng, (uint32_t)(plan->support - i));
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    return count;
}

static int late_cached_candidate(
    const LateReplanCacheEntry *entry, const Move *mv,
    const int *candidate, int ncand)
{
    if (!entry || entry->status != 2) return -1;
    for (int c = 0; c < ncand; c++)
        if (mv[candidate[c]].draw == entry->move.draw &&
            same_semantic_action(mv[candidate[c]], entry->move))
            return c;
    return -1;
}

/* Bounded recursive information-set improvement for every state with two or
 * three deck cards.  At each node the policy supplies a small set of semantic
 * card/action cores.  Every retained core gets its deck exit; the remaining
 * slots go to the highest-prior pile variants across those cores rather than
 * guaranteeing compute to one weak pile draw per core.  Ordered remaining-
 * deck assignments come exclusively from
 * the current mover's sanitized view and are common to every candidate.
 *
 * The caller provides a strict candidate-trajectory budget.  One quarter is
 * reserved for later states on the selected path; the rest is split equally
 * across the current common panel and recursive child panels.  Thus recursion
 * can never grow as worlds^depth.  A truncated branch falls back to the
 * configured continuation policy and remains visible in diagnostics. */
static int late_replan_choice(
    const Net *net, const NetEvalPlan *eval_plan,
    const State *st, const Move *mv, const float *score, int n,
    int prune, int random_sym,
    const uint8_t fixed_perm[NSUIT],
    const uint8_t other_perm[NSUIT], int other_player,
    int sample_actions, int symmetries,
    int plan_deck_max, int plan_block_gap, int draw_deck_max,
    float confirm_temp, int objective, int exact_terminal,
    int worlds, int cores, LateReplanContext *late,
    Move *out_move, double *out_objective,
    PlayoutWork *work)
{
    if (!net || !st || st->over || st->deck_left < 2 ||
        st->deck_left > LATE_REPLAN_MAX_DECK || n <= 0 ||
        worlds <= 0 || cores <= 0 || !late || !out_move)
        return LATE_REPLAN_UNAVAILABLE;
    if (cores > 3) cores = 3;

    int first[MAX_MOVES], deck[MAX_MOVES];
    double mass[MAX_MOVES];
    int ncore = 0;
    for (int i = 0; i < n; i++) {
        int c = -1;
        for (int j = 0; j < ncore; j++)
            if (same_semantic_action(mv[i], mv[first[j]])) {
                c = j;
                break;
            }
        if (c < 0) {
            c = ncore++;
            first[c] = i;
            deck[c] = -1;
            mass[c] = 0.0;
        } else if (MOVE_PACK(mv[i]) < MOVE_PACK(mv[first[c]])) {
            /* Physical wager IDs and legal-move enumeration order must not
             * decide a semantic core's stable representative. */
            first[c] = i;
        }
        float pi = lc_float_isfinite(score[i]) && score[i] > 0.0f
            ? score[i] : 0.0f;
        mass[c] += pi;
        if (mv[i].draw == 0) {
            if (deck[c] < 0 || pi > score[deck[c]] ||
                (pi == score[deck[c]] &&
                 MOVE_PACK(mv[i]) < MOVE_PACK(mv[deck[c]])))
                deck[c] = i;
        }
    }
    if (ncore <= 0) return LATE_REPLAN_UNAVAILABLE;

    int ranked[MAX_MOVES];
    for (int c = 0; c < ncore; c++) ranked[c] = c;
    for (int i = 0; i < ncore; i++) {
        int best = i;
        for (int j = i + 1; j < ncore; j++) {
            int a = ranked[j], b = ranked[best];
            uint16_t ap = MOVE_PACK(mv[first[a]]);
            uint16_t bp = MOVE_PACK(mv[first[b]]);
            if (mass[a] > mass[b] || (mass[a] == mass[b] && ap < bp))
                best = j;
        }
        int tmp = ranked[i]; ranked[i] = ranked[best]; ranked[best] = tmp;
    }
    if (cores > ncore) cores = ncore;

    int candidate[6], ncand = 0;
    for (int r = 0; r < cores; r++) {
        int c = ranked[r];
        if (deck[c] >= 0) candidate[ncand++] = deck[c];
    }
    const int candidate_limit = 2 * cores;
    while (ncand < candidate_limit) {
        int best = -1;
        float best_pi = -1.0f;
        for (int i = 0; i < n; i++) {
            if (mv[i].draw == 0) continue;
            int retained_core = 0;
            for (int r = 0; r < cores; r++)
                if (same_semantic_action(mv[i], mv[first[ranked[r]]])) {
                    retained_core = 1;
                    break;
                }
            if (!retained_core) continue;
            int duplicate = 0;
            for (int c = 0; c < ncand; c++)
                if (mv[candidate[c]].draw == mv[i].draw &&
                    same_semantic_action(mv[candidate[c]], mv[i])) {
                    duplicate = 1;
                    break;
                }
            if (duplicate) continue;
            float pi = lc_float_isfinite(score[i]) && score[i] > 0.0f
                ? score[i] : 0.0f;
            if (best < 0 || pi > best_pi ||
                (pi == best_pi &&
                 MOVE_PACK(mv[i]) < MOVE_PACK(mv[best]))) {
                best = i;
                best_pi = pi;
            }
        }
        if (best < 0) break;
        candidate[ncand++] = best;
    }
    if (ncand <= 0) return LATE_REPLAN_UNAVAILABLE;

    const int p = st->turn;
    LateAssignmentPlan assignment_plan;
    if (!late_assignment_plan_init(st, p, &assignment_plan))
        return LATE_REPLAN_UNAVAILABLE;

    uint64_t info_key = late_information_hash(st, p);
    if (late_path_contains(late, info_key)) {
        if (work) work->deck2_replan_cycle_closures++;
        return LATE_REPLAN_PATH_CYCLE;
    }

    if (late->depth <= 0 || late->eval_budget < (uint64_t)ncand) {
        if (!late->exhausted_reported && work) {
            work->deck2_replan_cap_hits++;
            late->exhausted_reported = 1;
        }
        return LATE_REPLAN_EXHAUSTED;
    }

    uint64_t signature = late_candidate_signature(
        mv, score, candidate, ncand, fixed_perm, other_perm, other_player);
    uint64_t node_budget = late->eval_budget;
    uint64_t path_signature = late_path_signature(late);
    uint64_t cache_key = late_mix64(
        info_key ^ signature ^ path_signature ^
        late_mix64((uint64_t)(uint32_t)late->panel_domain) ^
        late_mix64((uint64_t)(uint32_t)st->nply) ^
        late_mix64(node_budget) ^
        (UINT64_C(0x9E3779B97F4A7C15) * (uint64_t)late->depth));
    LateReplanCacheEntry *cache_entry =
        late_cache_entry(late->cache, cache_key);
    int cached = late_cached_candidate(cache_entry, mv, candidate, ncand);
    if (cached >= 0 && cache_entry->budget == node_budget &&
        cache_entry->logical_cost <= node_budget) {
        *out_move = mv[candidate[cached]];
        if (out_objective) *out_objective = cache_entry->objective;
        late->eval_budget = node_budget - cache_entry->logical_cost;
        late->stall_chain = out_move->draw == 0
            ? 0 : late->stall_chain + 1;
        if (late->path_n < LATE_REPLAN_MAX_DEPTH)
            late->path_key[late->path_n++] = info_key;
        if (late->depth > 0) late->depth--;
        if (work) {
            work->deck2_replan_cache_hits++;
            uint64_t used_depth = (uint64_t)late->path_n;
            if (used_depth > work->deck2_replan_max_depth)
                work->deck2_replan_max_depth = used_depth;
            if ((uint64_t)late->stall_chain >
                work->deck2_replan_max_stall_chain)
                work->deck2_replan_max_stall_chain =
                    (uint64_t)late->stall_chain;
        }
        return LATE_REPLAN_SELECTED;
    }
    if (cache_entry && cache_entry->status == 1) {
        if (work) work->deck2_replan_cycle_closures++;
        return LATE_REPLAN_PATH_CYCLE;
    }

    if (cache_entry) {
        cache_entry->key = cache_key;
        cache_entry->budget = node_budget;
        cache_entry->depth = (uint8_t)late->depth;
        cache_entry->status = 1;
    }

    int assignment[LATE_REPLAN_MAX_ASSIGNMENTS];
    int nassignment = assignment_plan.support;
    for (int i = 0; i < nassignment; i++) assignment[i] = i;
    Rng assignment_rng;
    /* Stateless information-keyed worlds make a cached decision independent
     * of the path and outer determinization that first reached it.  Derive the
     * seed from this node mover's view plus the enclosing panel domain.  In
     * particular, never carry the original root mover's private hand into an
     * opponent decision through a root-derived salt. */
    uint64_t node_seed = late_replan_node_seed(st, p, late);
    uint64_t panel_seed = late_mix64(
        node_seed ^
        (UINT64_C(0x9E3779B97F4A7C15) * (uint64_t)late->depth) ^
        UINT64_C(0xE7037ED1A0B428DB));
    rng_seed(&assignment_rng, panel_seed);

    int top_late_node = late->path_n == 0;
    uint64_t reserve = late->depth > 1 ? node_budget / 4 : 0;
    uint64_t planning_budget = node_budget - reserve;
    if (planning_budget < (uint64_t)ncand) {
        reserve = 0;
        planning_budget = node_budget;
    }
    uint64_t budget_worlds = planning_budget / (uint64_t)ncand;
    int desired_worlds = worlds < nassignment ? worlds : nassignment;
    /* The configured panel is a promise at the first late node reached by a
     * continuation.  Never silently turn 128 requested worlds into a smaller
     * early-game audit.  Deeper recursive nodes may receive a smaller, fully
     * reported panel when their equal branch allocation requires it. */
    if (top_late_node && node_budget >=
            (uint64_t)desired_worlds * (uint64_t)ncand)
        budget_worlds = (uint64_t)desired_worlds;
    if (budget_worlds == 0) {
        if (cache_entry) cache_entry->status = 0;
        if (!late->exhausted_reported && work) {
            work->deck2_replan_cap_hits++;
            late->exhausted_reported = 1;
        }
        return LATE_REPLAN_EXHAUSTED;
    }
    int eval_worlds = desired_worlds;
    if ((uint64_t)eval_worlds > budget_worlds)
        eval_worlds = (int)budget_worlds;
    if (!top_late_node &&
        eval_worlds < LATE_REPLAN_MIN_RECURSIVE_WORLDS) {
        if (cache_entry) cache_entry->status = 0;
        if (work) {
            work->deck2_replan_low_world_fallbacks++;
            work->deck2_replan_cap_hits++;
        }
        return LATE_REPLAN_LOW_WORLDS;
    }
    for (int i = 0; i < eval_worlds; i++) {
        int j = i + (int)rng_below(
            &assignment_rng, (uint32_t)(nassignment - i));
        int tmp = assignment[i]; assignment[i] = assignment[j];
        assignment[j] = tmp;
    }
    double total[6] = { 0 };
    uint64_t current_cost = (uint64_t)eval_worlds * (uint64_t)ncand;
    uint64_t child_pool = planning_budget > current_cost
        ? planning_budget - current_cost : 0;
    uint64_t child_budget = current_cost > 0 && late->depth > 1
        ? child_pool / current_cost : 0;
    uint64_t child_spent = 0;
    for (int d = 0; d < eval_worlds; d++) {
        State world;
        late_assignment_world(
            &assignment_plan, p, assignment[d], &world);
        uint64_t wseed = late_mix64(
            panel_seed ^ UINT64_C(0xD1B54A32D192ED03) *
                         (uint64_t)(d + 1));
        for (int c = 0; c < ncand; c++) {
            State child = world;
            lc_apply(&child, mv[candidate[c]]);
            Rng continuation_rng;
            rng_seed(&continuation_rng, wseed);
            LateReplanContext child_late = *late;
            child_late.eval_budget = child_budget;
            child_late.depth = late->depth - 1;
            child_late.exhausted_reported = 0;
            child_late.hypothetical = 1;
            if (child_late.path_n < LATE_REPLAN_MAX_DEPTH)
                child_late.path_key[child_late.path_n++] = info_key;
            uint64_t child_before = child_late.eval_budget;
            (void)playout(
                net, eval_plan, &child, p, prune,
                random_sym ? &continuation_rng : NULL,
                fixed_perm, other_perm, other_player,
                sample_actions, symmetries,
                plan_deck_max, plan_block_gap, draw_deck_max,
                confirm_temp, wseed, objective, exact_terminal,
                worlds, 1, &child_late, NULL, work);
            child_spent += child_before - child_late.eval_budget;
            total[c] += rollout_terminal_objective(&child, p, objective);
        }
    }

    int best = 0;
    for (int c = 1; c < ncand; c++) {
        int ci = candidate[c], bi = candidate[best];
        float cp = score[ci], bp = score[bi];
        if (total[c] > total[best] ||
            (total[c] == total[best] && cp > bp) ||
            (total[c] == total[best] && cp == bp &&
             MOVE_PACK(mv[ci]) < MOVE_PACK(mv[bi])))
            best = c;
    }
    *out_move = mv[candidate[best]];
    double best_objective = total[best] / eval_worlds;
    if (out_objective) *out_objective = best_objective;
    uint64_t spent = current_cost + child_spent;
    if (spent > node_budget) spent = node_budget;
    late->eval_budget = node_budget - spent;
    late->stall_chain = out_move->draw == 0 ? 0 : late->stall_chain + 1;
    if (late->path_n < LATE_REPLAN_MAX_DEPTH)
        late->path_key[late->path_n++] = info_key;
    if (late->depth > 0) late->depth--;
    if (work) {
        work->deck2_replans++;
        work->deck2_replan_worlds += (uint64_t)eval_worlds;
        work->deck2_replan_evals += current_cost;
        if (top_late_node) {
            work->deck2_replan_root_calls++;
            work->deck2_replan_root_worlds += (uint64_t)eval_worlds;
        }
        uint64_t used_depth = (uint64_t)late->path_n;
        if (used_depth > work->deck2_replan_max_depth)
            work->deck2_replan_max_depth = used_depth;
        if ((uint64_t)late->stall_chain >
            work->deck2_replan_max_stall_chain)
            work->deck2_replan_max_stall_chain =
                (uint64_t)late->stall_chain;
    }
    if (cache_entry) {
        cache_entry->logical_cost = spent;
        cache_entry->objective = best_objective;
        cache_entry->move = *out_move;
        cache_entry->status = 2;
    }
    return LATE_REPLAN_SELECTED;
}

/* Runtime-only white-box regression hook.  It verifies that semantic
 * candidate enumeration is order-independent, that cache hits consume the
 * same logical branch budget without pretending to perform fresh work, and
 * that an entry produced under one ordered ancestor path cannot serve another
 * path.  Kept here so the test exercises the real private helper and key. */
int rollout_test_late_cache_order(const Net *net, const State *st)
{
    if (!net || !st || st->over || st->deck_left < 2 ||
        st->deck_left > LATE_REPLAN_MAX_DECK)
        return 0;
    Move forward_mv[MAX_MOVES], reverse_mv[MAX_MOVES];
    float forward_score[MAX_MOVES], reverse_score[MAX_MOVES];
    int n = rank_moves(
        net, NULL, st, forward_mv, forward_score, 1, NULL, NULL);
    if (n <= 1) return 0;

    /* Actual selected-path late nodes must ignore the cheap continuation's
     * arbitrary sampled/fixed symmetry member and recover the exact configured
     * ensemble.  Hypothetical descendants must do the opposite: preserve the
     * configured bounded orientation rather than falsely claiming exact-20
     * compute.  Use a deliberately suit-asymmetric policy so the two paths are
     * observably different. */
    uint8_t perms[120][NSUIT];
    int nperm = suit_permutations(20, perms);
    if (nperm <= 1) return 0;
    Net *oriented = (Net *)malloc(sizeof *oriented);
    if (!oriented) return 0;
    net_zero(oriented);
    Move target = forward_mv[0];
    int mapped_card = lc_permute_card(target.card, perms[1]);
    oriented->bplay[mapped_card * 2 + target.discard] = 12.0f;

    Move actual_a[MAX_MOVES], actual_b[MAX_MOVES];
    Move expected_exact[MAX_MOVES], hypothetical[MAX_MOVES];
    Move expected_hypothetical[MAX_MOVES];
    float actual_a_score[MAX_MOVES], actual_b_score[MAX_MOVES];
    float expected_exact_score[MAX_MOVES], hypothetical_score[MAX_MOVES];
    float expected_hypothetical_score[MAX_MOVES];
    LateReplanContext actual_rank_context = { 0 };
    LateReplanContext hypothetical_rank_context = { 0 };
    hypothetical_rank_context.hypothetical = 1;
    int actual_exact = late_exact_ensemble_enabled(
        1, &actual_rank_context);
    int hypothetical_exact = late_exact_ensemble_enabled(
        1, &hypothetical_rank_context);
    if (!actual_exact || hypothetical_exact) {
        free(oriented);
        return 0;
    }
    Rng actual_rng_a, actual_rng_b;
    rng_seed(&actual_rng_a, UINT64_C(0x1111222233334444));
    rng_seed(&actual_rng_b, UINT64_C(0xAAAABBBBCCCCDDDD));
    int actual_an = rank_playout_moves(
        oriented, NULL, st, actual_a, actual_a_score, 20,
        &actual_rng_a, perms[1], actual_exact);
    int actual_bn = rank_playout_moves(
        oriented, NULL, st, actual_b, actual_b_score, 20,
        &actual_rng_b, perms[nperm - 1], actual_exact);
    int expected_exact_n = policy_probs_sym(
        oriented, st, expected_exact, expected_exact_score, NULL, 20);
    int hypothetical_n = rank_playout_moves(
        oriented, NULL, st, hypothetical, hypothetical_score, 20,
        &actual_rng_a, perms[1], hypothetical_exact);
    int expected_hypothetical_n = policy_probs_perm(
        oriented, st, expected_hypothetical,
        expected_hypothetical_score, NULL, perms[1]);
    int orientation_ok =
        actual_an == n && actual_bn == n && expected_exact_n == n &&
        hypothetical_n == n && expected_hypothetical_n == n;
    int hypothetical_differs = 0;
    for (int i = 0; i < n && orientation_ok; i++) {
        if (MOVE_PACK(actual_a[i]) != MOVE_PACK(actual_b[i]) ||
            MOVE_PACK(actual_a[i]) != MOVE_PACK(expected_exact[i]) ||
            MOVE_PACK(hypothetical[i]) !=
                MOVE_PACK(expected_hypothetical[i]) ||
            actual_a_score[i] != actual_b_score[i] ||
            actual_a_score[i] != expected_exact_score[i] ||
            hypothetical_score[i] != expected_hypothetical_score[i])
            orientation_ok = 0;
        if (fabsf(hypothetical_score[i] - actual_a_score[i]) > 1e-6f)
            hypothetical_differs = 1;
    }
    free(oriented);
    if (!orientation_ok || !hypothetical_differs) return 0;

    Move exact_a[MAX_MOVES], exact_b[MAX_MOVES];
    float exact_a_score[MAX_MOVES], exact_b_score[MAX_MOVES];
    Rng exact_rng_a, exact_rng_b;
    rng_seed(&exact_rng_a, UINT64_C(0x1111222233334444));
    rng_seed(&exact_rng_b, UINT64_C(0xAAAABBBBCCCCDDDD));
    int exact_an = rank_playout_moves(
        net, NULL, st, exact_a, exact_a_score, 20, &exact_rng_a, perms[1],
        actual_exact);
    int exact_bn = rank_playout_moves(
        net, NULL, st, exact_b, exact_b_score, 20, &exact_rng_b,
        perms[nperm - 1], actual_exact);
    if (exact_an != n || exact_bn != n ||
        memcmp(exact_a, exact_b, sizeof(Move) * (size_t)n) != 0 ||
        memcmp(exact_a_score, exact_b_score,
               sizeof(float) * (size_t)n) != 0)
        return 0;
    for (int i = 0; i < n; i++) {
        reverse_mv[i] = forward_mv[n - 1 - i];
        reverse_score[i] = forward_score[n - 1 - i];
    }

    uint64_t info = late_information_hash(st, st->turn);
    uint64_t ancestor_a = late_mix64(info ^ UINT64_C(0x243F6A8885A308D3));
    uint64_t ancestor_b = late_mix64(info ^ UINT64_C(0x13198A2E03707344));
    if (ancestor_a == info) ancestor_a ^= UINT64_C(1);
    if (ancestor_b == info || ancestor_b == ancestor_a)
        ancestor_b ^= UINT64_C(0x9E3779B97F4A7C15);
#define LATE_TEST_CONTEXT(name, cache_ptr, ancestor)                     \
    LateReplanContext name = late_replan_context_init(                   \
        8, 2, LATE_PANEL_PRIMARY, cache_ptr);                             \
    name.path_key[0] = ancestor;                                         \
    name.path_n = 1;                                                     \
    name.stall_chain = 1
#define LATE_TEST_CALL(state, moves, scores, context, move_out, q_out, work_out) \
    late_replan_choice(                                                  \
        net, NULL, state, moves, scores, n, 0, 0, NULL, NULL,           \
        (state)->turn ^ 1,                                               \
        0, 1, 0, 0, 0, 0.0f, 0, 1, 8, 2, context,                     \
        move_out, q_out, work_out)

    LateReplanCache shared = { 0 };
    LATE_TEST_CONTEXT(forward, &shared, ancestor_a);
    PlayoutWork forward_work = { 0 };
    Move forward_choice;
    double forward_q = 0.0;
    if (LATE_TEST_CALL(
            st, forward_mv, forward_score, &forward,
            &forward_choice, &forward_q, &forward_work) !=
        LATE_REPLAN_SELECTED)
        return 0;

    LATE_TEST_CONTEXT(cached_reverse, &shared, ancestor_a);
    PlayoutWork cached_work = { 0 };
    Move cached_choice;
    double cached_q = 0.0;
    if (LATE_TEST_CALL(
            st, reverse_mv, reverse_score, &cached_reverse,
            &cached_choice, &cached_q, &cached_work) !=
        LATE_REPLAN_SELECTED)
        return 0;

    LateReplanCache reverse_cache = { 0 };
    LATE_TEST_CONTEXT(fresh_reverse, &reverse_cache, ancestor_a);
    PlayoutWork reverse_work = { 0 };
    Move reverse_choice;
    double reverse_q = 0.0;
    if (LATE_TEST_CALL(
            st, reverse_mv, reverse_score, &fresh_reverse,
            &reverse_choice, &reverse_q, &reverse_work) !=
        LATE_REPLAN_SELECTED)
        return 0;

    if (MOVE_PACK(forward_choice) != MOVE_PACK(reverse_choice) ||
        MOVE_PACK(forward_choice) != MOVE_PACK(cached_choice) ||
        fabs(forward_q - reverse_q) > 1e-12 ||
        fabs(forward_q - cached_q) > 1e-12 ||
        forward.eval_budget != fresh_reverse.eval_budget ||
        forward.eval_budget != cached_reverse.eval_budget ||
        memcmp(&forward_work, &reverse_work, sizeof forward_work) != 0 ||
        cached_work.deck2_replan_cache_hits != 1 ||
        cached_work.deck2_replans != 0 ||
        cached_work.deck2_replan_worlds != 0 ||
        cached_work.deck2_replan_evals != 0 ||
        cached_work.exact_terminal_leaves != 0 ||
        cached_work.unfinished_cap_leaves != 0)
        return 0;

    /* Treat p as a downstream mover and o as the original root mover.  Swap a
     * card private to o with an unobserved deck card.  The downstream node's
     * panel seed, ranking, and completed cached policy must remain identical:
     * no root-private fact may arrive through its recursive context.  Repeat
     * in both outer-world orders to prove that whichever determinization
     * populates the panel cache first cannot alter the result. */
    State hidden = *st;
    int p = st->turn, o = p ^ 1;
    uint64_t private_hand = hidden.hand[o] & ~hidden.known[o];
    int hc = private_hand ? __builtin_ctzll(private_hand) : -1;
    int dc = hidden.deck_left > 0 ? hidden.deck[hidden.deck_pos] : -1;
    if (hc < 0 || dc < 0 || hc == dc) return 0;
    hidden.hand[o] &= ~(UINT64_C(1) << hc);
    hidden.hand[o] |= UINT64_C(1) << dc;
    hidden.deck[hidden.deck_pos] = (uint8_t)hc;
    if (late_information_hash(st, p) != late_information_hash(&hidden, p) ||
        late_decision_seed(st, p, LATE_PANEL_PRIMARY) !=
            late_decision_seed(&hidden, p, LATE_PANEL_PRIMARY))
        return 0;

    LateReplanCache seed_cache = { 0 };
    LateReplanContext primary_seed = late_replan_context_init(
        8, 2, LATE_PANEL_PRIMARY, &seed_cache);
    LateReplanContext trusted_seed = late_replan_context_init(
        8, 2, LATE_PANEL_TRUSTED_CONFIRM, &seed_cache);
    LateReplanContext challenger_seed = late_replan_context_init(
        8, 2, LATE_PANEL_CHALLENGER_CONFIRM, &seed_cache);
    uint64_t primary_node_seed = late_replan_node_seed(st, p, &primary_seed);
    uint64_t trusted_node_seed = late_replan_node_seed(st, p, &trusted_seed);
    uint64_t challenger_node_seed =
        late_replan_node_seed(st, p, &challenger_seed);
    if (primary_node_seed !=
            late_replan_node_seed(&hidden, p, &primary_seed) ||
        trusted_node_seed !=
            late_replan_node_seed(&hidden, p, &trusted_seed) ||
        challenger_node_seed !=
            late_replan_node_seed(&hidden, p, &challenger_seed) ||
        primary_node_seed == trusted_node_seed ||
        primary_node_seed == challenger_node_seed ||
        trusted_node_seed == challenger_node_seed)
        return 0;

    Move hidden_mv[MAX_MOVES];
    float hidden_score[MAX_MOVES];
    int hidden_n = rank_playout_moves(
        net, NULL, &hidden, hidden_mv, hidden_score, 20, NULL, NULL,
        actual_exact);
    if (hidden_n != exact_an ||
        memcmp(exact_a, hidden_mv, sizeof(Move) * (size_t)hidden_n) != 0 ||
        memcmp(exact_a_score, hidden_score,
               sizeof(float) * (size_t)hidden_n) != 0)
        return 0;

    LateReplanCache visible_first_cache = { 0 };
    LATE_TEST_CONTEXT(visible_first, &visible_first_cache, ancestor_a);
    PlayoutWork visible_first_work = { 0 };
    Move visible_first_choice;
    double visible_first_q = 0.0;
    if (late_replan_choice(
            net, NULL, st, exact_a, exact_a_score, exact_an,
            0, 0, NULL, NULL,
            o, 0, 20, 0, 0, 0, 0.0f, 0, 1, 8, 2,
            &visible_first, &visible_first_choice, &visible_first_q,
            &visible_first_work) != LATE_REPLAN_SELECTED)
        return 0;
    LATE_TEST_CONTEXT(hidden_second, &visible_first_cache, ancestor_a);
    PlayoutWork hidden_second_work = { 0 };
    Move hidden_second_choice;
    double hidden_second_q = 0.0;
    if (late_replan_choice(
            net, NULL, &hidden, hidden_mv, hidden_score, hidden_n, 0, 0,
            NULL, NULL, o, 0, 20, 0, 0, 0, 0.0f, 0, 1, 8, 2,
            &hidden_second, &hidden_second_choice, &hidden_second_q,
            &hidden_second_work) != LATE_REPLAN_SELECTED ||
        hidden_second_work.deck2_replan_cache_hits != 1 ||
        hidden_second_work.deck2_replans != 0)
        return 0;

    LateReplanCache hidden_first_cache = { 0 };
    LATE_TEST_CONTEXT(hidden_first, &hidden_first_cache, ancestor_a);
    PlayoutWork hidden_first_work = { 0 };
    Move hidden_first_choice;
    double hidden_first_q = 0.0;
    if (late_replan_choice(
            net, NULL, &hidden, hidden_mv, hidden_score, hidden_n, 0, 0,
            NULL, NULL, o, 0, 20, 0, 0, 0, 0.0f, 0, 1, 8, 2,
            &hidden_first, &hidden_first_choice, &hidden_first_q,
            &hidden_first_work) != LATE_REPLAN_SELECTED)
        return 0;
    LATE_TEST_CONTEXT(visible_second, &hidden_first_cache, ancestor_a);
    PlayoutWork visible_second_work = { 0 };
    Move visible_second_choice;
    double visible_second_q = 0.0;
    if (late_replan_choice(
            net, NULL, st, exact_a, exact_a_score, exact_an,
            0, 0, NULL, NULL,
            o, 0, 20, 0, 0, 0, 0.0f, 0, 1, 8, 2,
            &visible_second, &visible_second_choice, &visible_second_q,
            &visible_second_work) != LATE_REPLAN_SELECTED ||
        visible_second_work.deck2_replan_cache_hits != 1 ||
        visible_second_work.deck2_replans != 0 ||
        MOVE_PACK(visible_first_choice) != MOVE_PACK(hidden_second_choice) ||
        MOVE_PACK(visible_first_choice) != MOVE_PACK(hidden_first_choice) ||
        MOVE_PACK(visible_first_choice) != MOVE_PACK(visible_second_choice) ||
        fabs(visible_first_q - hidden_second_q) > 1e-12 ||
        fabs(visible_first_q - hidden_first_q) > 1e-12 ||
        fabs(visible_first_q - visible_second_q) > 1e-12 ||
        memcmp(&visible_first_work, &hidden_first_work,
               sizeof visible_first_work) != 0)
        return 0;

    /* The shared cache contains the complete ancestor-A subtree.  Ancestor B
     * must behave exactly like a clean cache, including physical work counts. */
    LATE_TEST_CONTEXT(shared_b, &shared, ancestor_b);
    PlayoutWork shared_b_work = { 0 };
    Move shared_b_choice;
    double shared_b_q = 0.0;
    if (LATE_TEST_CALL(
            st, reverse_mv, reverse_score, &shared_b,
            &shared_b_choice, &shared_b_q, &shared_b_work) !=
        LATE_REPLAN_SELECTED)
        return 0;
    LateReplanCache clean_b_cache = { 0 };
    LATE_TEST_CONTEXT(clean_b, &clean_b_cache, ancestor_b);
    PlayoutWork clean_b_work = { 0 };
    Move clean_b_choice;
    double clean_b_q = 0.0;
    if (LATE_TEST_CALL(
            st, reverse_mv, reverse_score, &clean_b,
            &clean_b_choice, &clean_b_q, &clean_b_work) !=
        LATE_REPLAN_SELECTED)
        return 0;

    /* Round ply is deliberately absent from cycle identity, but it changes
     * cap reserve and the network input.  A cache built for the otherwise
     * identical preceding ply therefore must behave like an empty cache. */
    State next_ply = *st;
    next_ply.nply++;
    LATE_TEST_CONTEXT(shared_next_ply, &shared, ancestor_a);
    PlayoutWork shared_next_ply_work = { 0 };
    Move shared_next_ply_choice;
    double shared_next_ply_q = 0.0;
    if (LATE_TEST_CALL(
            &next_ply, reverse_mv, reverse_score, &shared_next_ply,
            &shared_next_ply_choice, &shared_next_ply_q,
            &shared_next_ply_work) != LATE_REPLAN_SELECTED)
        return 0;
    LateReplanCache clean_next_ply_cache = { 0 };
    LATE_TEST_CONTEXT(clean_next_ply, &clean_next_ply_cache, ancestor_a);
    PlayoutWork clean_next_ply_work = { 0 };
    Move clean_next_ply_choice;
    double clean_next_ply_q = 0.0;
    if (LATE_TEST_CALL(
            &next_ply, reverse_mv, reverse_score, &clean_next_ply,
            &clean_next_ply_choice, &clean_next_ply_q,
            &clean_next_ply_work) != LATE_REPLAN_SELECTED ||
        MOVE_PACK(shared_next_ply_choice) !=
            MOVE_PACK(clean_next_ply_choice) ||
        fabs(shared_next_ply_q - clean_next_ply_q) > 1e-12 ||
        shared_next_ply.eval_budget != clean_next_ply.eval_budget ||
        memcmp(&shared_next_ply_work, &clean_next_ply_work,
               sizeof shared_next_ply_work) != 0)
        return 0;

    LateReplanCache low_cache = { 0 };
    LATE_TEST_CONTEXT(low, &low_cache, ancestor_b);
    low.hypothetical = 1;
    low.eval_budget = 4;
    PlayoutWork low_work = { 0 };
    Move low_choice;
    double low_q = 0.0;
    if (LATE_TEST_CALL(
            st, forward_mv, forward_score, &low,
            &low_choice, &low_q, &low_work) !=
            LATE_REPLAN_LOW_WORLDS ||
        low_work.deck2_replan_low_world_fallbacks != 1 ||
        low_work.deck2_replan_cap_hits != 1 ||
        low_work.deck2_replans != 0 ||
        low_work.deck2_replan_worlds != 0 ||
        low_work.deck2_replan_evals != 0)
        return 0;

#undef LATE_TEST_CALL
#undef LATE_TEST_CONTEXT
    return MOVE_PACK(shared_b_choice) == MOVE_PACK(clean_b_choice) &&
           fabs(shared_b_q - clean_b_q) <= 1e-12 &&
           shared_b.eval_budget == clean_b.eval_budget &&
           memcmp(&shared_b_work, &clean_b_work,
                  sizeof shared_b_work) == 0;
}

/* Runtime-only verification of the finite outer information-set panel.  The
 * assignment rank is a bijection with an ordered deck; checking rank uniqueness
 * therefore checks complete-world uniqueness without a large world hash set. */
int rollout_test_unique_late_assignments(
    const State *st, int requested, int *support_out)
{
    if (!st || requested <= 0) return -1;
    const int p = st->turn;
    LateAssignmentPlan primary_plan, trusted_plan, challenger_plan;
    int primary[LATE_REPLAN_MAX_ASSIGNMENTS];
    int trusted[LATE_REPLAN_MAX_ASSIGNMENTS];
    int challenger[LATE_REPLAN_MAX_ASSIGNMENTS];
    int np = late_unique_assignment_order(
        st, p, requested, LATE_PANEL_PRIMARY_ASSIGNMENTS,
        &primary_plan, primary);
    int nt = late_unique_assignment_order(
        st, p, requested, LATE_PANEL_TRUSTED_ASSIGNMENTS,
        &trusted_plan, trusted);
    int nc = late_unique_assignment_order(
        st, p, requested, LATE_PANEL_CHALLENGER_ASSIGNMENTS,
        &challenger_plan, challenger);
    if (np <= 0 || nt != np || nc != np ||
        trusted_plan.support != primary_plan.support ||
        challenger_plan.support != primary_plan.support)
        return -1;
    if (support_out) *support_out = primary_plan.support;

    uint8_t seen[LATE_REPLAN_MAX_ASSIGNMENTS] = { 0 };
    int trusted_differs = 0, challenger_differs = 0;
    for (int i = 0; i < np; i++) {
        if (primary[i] < 0 || primary[i] >= primary_plan.support ||
            seen[primary[i]])
            return -1;
        seen[primary[i]] = 1;
        if (trusted[i] != primary[i]) trusted_differs = 1;
        if (challenger[i] != primary[i]) challenger_differs = 1;
        State world;
        late_assignment_world(&primary_plan, p, primary[i], &world);
        if (world.deck_left != st->deck_left ||
            world.hand_n[p ^ 1] != st->hand_n[p ^ 1] ||
            late_information_hash(&world, p) !=
                late_information_hash(st, p))
            return -1;
    }
    if (np > 1 && (!trusted_differs || !challenger_differs)) return -1;

    /* The input may be a complete engine state.  Exchanging two facts hidden
     * from the mover must not change support, panel order, or rebuilt worlds. */
    State hidden = *st;
    int o = p ^ 1;
    uint64_t private_hand = hidden.hand[o] & ~hidden.known[o];
    int hc = private_hand ? __builtin_ctzll(private_hand) : -1;
    int dc = hidden.deck_left > 0 ? hidden.deck[hidden.deck_pos] : -1;
    if (hc >= 0 && dc >= 0 && hc != dc) {
        hidden.hand[o] &= ~(UINT64_C(1) << hc);
        hidden.hand[o] |= UINT64_C(1) << dc;
        hidden.deck[hidden.deck_pos] = (uint8_t)hc;
        LateAssignmentPlan hidden_plan;
        int hidden_order[LATE_REPLAN_MAX_ASSIGNMENTS];
        int nh = late_unique_assignment_order(
            &hidden, p, requested, LATE_PANEL_PRIMARY_ASSIGNMENTS,
            &hidden_plan, hidden_order);
        if (nh != np || hidden_plan.support != primary_plan.support ||
            memcmp(hidden_order, primary, sizeof(int) * (size_t)np) != 0)
            return -1;
        for (int i = 0; i < np; i++) {
            State a, b;
            late_assignment_world(&primary_plan, p, primary[i], &a);
            late_assignment_world(&hidden_plan, p, hidden_order[i], &b);
            if (memcmp(&a, &b, sizeof a) != 0) return -1;
        }
    }
    return np;
}

static int move_equal(Move a, Move b)
{
    return MOVE_PACK(a) == MOVE_PACK(b);
}

static int find_move(const Move *mv, int n, Move target)
{
    for (int i = 0; i < n; i++)
        if (move_equal(mv[i], target)) return i;
    return -1;
}

static int append_unique(int *order, int *count, int limit, int index)
{
    if (index < 0) return 0;
    for (int i = 0; i < *count; i++)
        if (order[i] == index) return 0;
    if (*count >= limit) return 0;
    order[(*count)++] = index;
    return 1;
}

static int same_semantic_action(Move a, Move b)
{
    if (a.discard != b.discard) return 0;
    if (CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card))
        return CARD_SUIT(a.card) == CARD_SUIT(b.card);
    return a.card == b.card;
}

static int same_semantic_move(Move a, Move b)
{
    return a.draw == b.draw && same_semantic_action(a, b);
}

static int find_semantic_move(const Move *mv, int n, Move target)
{
    for (int i = 0; i < n; i++)
        if (same_semantic_move(mv[i], target)) return i;
    return -1;
}

/* The finite-support resolver is a distinct audit method, not a special
 * setting of the historical recursive redeterminizing replan.  Its explicit
 * flag therefore owns the real root and suppresses the old method inside all
 * ordinary and confirmation trajectories.  Parsed specs enforce these
 * prerequisites too; keeping the checks here makes programmatic Agent users
 * fail closed instead of changing the deployed baseline underneath the
 * bounded comparison. */
static int bounded_late_root_enabled(const Agent *a)
{
    return a && a->bounded_late_root && a->net && a->exact_terminal == 1 &&
           a->no_belief &&
           !(a->plan_deck_max > 0 && a->plan_block_gap > 0) &&
           a->draw_root_deck_max == 0;
}

static void historical_replan_config(const Agent *a, int *worlds, int *cores)
{
    if (a->bounded_late_root) {
        *worlds = 0;
        *cores = 0;
        return;
    }
    *worlds = a->deck2_replan_worlds;
    *cores = *worlds > 0 ? a->deck2_replan_cores : 0;
}

/* Add only pile-draw variants of the highest-prior distinct play/discard
 * actions.  The complete-move policy factorization can make one global draw
 * preference suppress every pile alternative at once; ranking action cores
 * first preserves the policy's card decision while allowing rollout to value
 * late-round tempo.  First guarantee one best pile option per requested core,
 * then fill the audit's remaining slots by prior.  The caller adds planner
 * and semantic challengers first so this optional expansion cannot crowd out
 * the more targeted corrections. */
static int add_top_action_pile_variants(
    const Move *mv, const float *prob, int n,
    const int *ranked, int nsorted, int wanted_cores,
    int *order, int *ncand)
{
    if (wanted_cores < 1) return 0;
    if (wanted_cores > 2) wanted_cores = 2;
    int core[2], ncore = 0;
    for (int r = 0; r < nsorted && ncore < wanted_cores; r++) {
        int distinct = 1;
        for (int c = 0; c < ncore; c++)
            if (same_semantic_action(mv[ranked[r]], mv[core[c]])) {
                distinct = 0;
                break;
            }
        if (distinct) core[ncore++] = ranked[r];
    }

    int added = 0;
    for (int c = 0; c < ncore && *ncand < MAX_CAND; c++) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (mv[i].draw == 0 ||
                !same_semantic_action(mv[i], mv[core[c]]))
                continue;
            int present = 0;
            for (int k = 0; k < *ncand; k++)
                if (order[k] == i) { present = 1; break; }
            if (!present && (best < 0 || prob[i] > prob[best])) best = i;
        }
        added += append_unique(order, ncand, MAX_CAND, best);
    }

    while (*ncand < MAX_CAND) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (mv[i].draw == 0) continue;
            int selected_core = 0;
            for (int c = 0; c < ncore; c++)
                if (same_semantic_action(mv[i], mv[core[c]])) {
                    selected_core = 1;
                    break;
                }
            if (!selected_core) continue;
            int present = 0;
            for (int k = 0; k < *ncand; k++)
                if (order[k] == i) { present = 1; break; }
            if (!present && (best < 0 || prob[i] > prob[best])) best = i;
        }
        if (best < 0) break;
        added += append_unique(order, ncand, MAX_CAND, best);
    }
    return added;
}

static int top_action_group(const Move *mv, const float *prob, int n,
                            int index, int limit)
{
    double candidate = 0.0;
    for (int i = 0; i < n; i++)
        if (same_semantic_action(mv[i], mv[index]))
            candidate += prob[i];
    int better = 0;
    for (int i = 0; i < n; i++) {
        int first = 1;
        for (int j = 0; j < i; j++)
            if (same_semantic_action(mv[j], mv[i])) {
                first = 0;
                break;
            }
        if (!first) continue;
        double action_prob = 0.0;
        for (int j = 0; j < n; j++)
            if (same_semantic_action(mv[j], mv[i]))
                action_prob += prob[j];
        if (action_prob > candidate) better++;
    }
    return better < limit;
}

/* Build an ordinary shortlist hierarchically.  Complete-move softmax mass can
 * split one card/play-discard decision across several draw sources, allowing
 * those variants to consume most of a flat top-five prefix.  Group by the
 * semantic action first, retain one highest-prior complete move per selected
 * core, then spend only the remaining slots on one information-set-safe draw
 * alternative per core.  Candidate zero is always the deployed baseline.
 *
 * action_core_count controls distinct cores, while root_width controls total
 * compute.  This mode additionally hard-caps the ordinary prefix at five.
 * Purposeful research challengers are appended later under their own gates. */
static int build_action_core_shortlist(
    const State *st, const Move *mv, const float *prob, int n,
    int baseline, int root_width, int action_core_count,
    int min_cand, float cand_floor, float cand_mass,
    int *order, int *core_candidates, int *draw_candidates)
{
    double mass[MAX_MOVES] = { 0 };
    int best_move[MAX_MOVES];
    int first_move[MAX_MOVES];
    int ncore = 0;
    for (int i = 0; i < n; i++) {
        int c = -1;
        for (int j = 0; j < ncore; j++)
            if (same_semantic_action(mv[i], mv[first_move[j]])) {
                c = j;
                break;
            }
        if (c < 0) {
            c = ncore++;
            first_move[c] = i;
            best_move[c] = i;
        }
        mass[c] += prob[i];
        if (prob[i] > prob[best_move[c]]) best_move[c] = i;
    }

    int ranked[MAX_MOVES];
    for (int c = 0; c < ncore; c++) ranked[c] = c;
    for (int i = 0; i < ncore; i++) {
        int top = i;
        for (int j = i + 1; j < ncore; j++) {
            int a = ranked[j], b = ranked[top];
            if (mass[a] > mass[b] ||
                (mass[a] == mass[b] &&
                 prob[best_move[a]] > prob[best_move[b]]))
                top = j;
        }
        int tmp = ranked[i]; ranked[i] = ranked[top]; ranked[top] = tmp;
    }

    int budget = root_width;
    if (budget > ACTION_SHORTLIST_MAX) budget = ACTION_SHORTLIST_MAX;
    if (budget < 1) budget = 1;
    int wanted = action_core_count;
    if (wanted > budget) wanted = budget;
    if (wanted > ncore) wanted = ncore;
    if (wanted < 1) wanted = 1;
    int required = min_cand > 1 ? min_cand : 1;
    if (required > wanted) required = wanted;

    int baseline_core = 0;
    for (int c = 0; c < ncore; c++)
        if (same_semantic_action(mv[baseline], mv[first_move[c]])) {
            baseline_core = c;
            break;
        }

    int selected_rep[MAX_MOVES];
    int nselected = 1, ncand = 1;
    selected_rep[0] = baseline;
    order[0] = baseline;
    double covered = mass[baseline_core];

    for (int r = 0; r < ncore && nselected < wanted; r++) {
        int c = ranked[r];
        if (c == baseline_core) continue;
        int eligible = nselected < required;
        if (!eligible) {
            if (cand_mass > 0.0f)
                eligible = covered < (double)cand_mass;
            else {
                float floor_p = cand_floor > 0.0f ? cand_floor : 0.02f;
                eligible = mass[c] >= (double)floor_p;
            }
        }
        if (!eligible) break;
        selected_rep[nselected] = best_move[c];
        order[ncand++] = best_move[c];
        covered += mass[c];
        nselected++;
    }

    *core_candidates = nselected;
    *draw_candidates = 0;
    const float draw_floor = cand_floor > 0.0f ? cand_floor : 0.02f;
    for (int k = 0; k < nselected && ncand < budget; k++) {
        int alternative = hand_plan_choose_draw_source(
            st, st->turn, mv, prob, n, selected_rep[k]);
        /* This mode is intentionally a top-policy audit.  The planner may
         * identify a legal, information-safe pile draw whose factorized
         * full-move prior is effectively zero; admitting it here would spend
         * scarce worlds on precisely the low-prior combinations this
         * shortlist is meant to avoid.  Such moves belong in the separately
         * gated semantic/draw-variant research paths. */
        if (alternative < 0 || alternative == selected_rep[k] ||
            prob[alternative] < draw_floor)
            continue;
        int present = 0;
        for (int i = 0; i < ncand; i++)
            if (order[i] == alternative) {
                present = 1;
                break;
            }
        if (!present) {
            order[ncand++] = alternative;
            (*draw_candidates)++;
        }
    }
    return ncand;
}

/* A flat shortlist covers exactly the probability of its complete moves.  A
 * hierarchical shortlist instead represents whole card/play-discard action
 * cores: every draw source belonging to a selected core is part of the
 * policy evidence that admitted that core.  Report that aggregate rather
 * than understating coverage with only the chosen representative logits. */
static double shortlist_policy_mass(
    const Move *mv, const float *prob, int n, const int *order,
    int ncand, int action_core_candidates)
{
    double covered = 0.0;
    if (action_core_candidates > 0) {
        for (int i = 0; i < n; i++) {
            int represented = 0;
            for (int c = 0; c < action_core_candidates; c++)
                if (same_semantic_action(mv[i], mv[order[c]])) {
                    represented = 1;
                    break;
                }
            if (represented) covered += prob[i];
        }
    } else {
        for (int c = 0; c < ncand; c++) covered += prob[order[c]];
    }
    return covered;
}

/* A wager that is the only card of its suit in our hand, before we have
 * started that suit, is a high-value semantic discard candidate once the
 * opponent has played a number and therefore cannot score the wager.  It is
 * not literally cost-free—the opponent could pick it up to stall—so it still
 * has to beat the baseline in both large stochastic comparisons.  This is
 * deliberately much narrower than evaluating every legal discard. */
static int isolated_one_sided_wager_discard(
    const State *st, const Move *mv, const float *prob, int n)
{
    const int p = st->turn, o = p ^ 1;
    int best = -1, best_score = -1000000;
    for (int i = 0; i < n; i++) {
        int c = mv[i].card, s = CARD_SUIT(c);
        if (!mv[i].discard || !CARD_IS_WAGER(c) ||
            st->exp_n[p][s] != 0 || st->exp_top[o][s] == 0)
            continue;
        int suit_cards = 0;
        uint64_t hand = st->hand[p];
        while (hand) {
            int h = __builtin_ctzll(hand);
            hand &= hand - 1;
            if (CARD_SUIT(h) == s) suit_cards++;
        }
        if (suit_cards != 1) continue;

        /* Compare the known post-move hands without ever reading a hidden
         * deck card.  This lets a one-sided wager discard pair naturally with a
         * useful face-up pickup (p59/p61) instead of forcing the deck variant
         * and spending a second semantic slot on the Cartesian combination. */
        State after = *st;
        if (mv[i].draw == 0) {
            after.hand[p] &= ~(1ULL << c);
            if (after.hand_n[p] > 0) after.hand_n[p]--;
            if (after.deck_left > 0) after.deck_left--;
        } else {
            lc_apply(&after, mv[i]);
        }
        HandPlan plan;
        hand_plan_build(&after, p, after.deck_left / 2, &plan);
        if (best < 0 || plan.score > best_score ||
            (plan.score == best_score &&
             mv[i].draw == 0 && mv[best].draw != 0) ||
            (plan.score == best_score &&
             mv[i].draw == mv[best].draw && prob[i] > prob[best])) {
            best = i;
            best_score = plan.score;
        }
    }
    return best;
}

/* Add one purposeful pile pickup, not one move per draw source.  Number cards
 * must improve the exact visible-hand schedule (for example R4 before held
 * R8/R9).  A wager may instead be denied when public knowledge proves that
 * the opponent has already picked up or committed to that suit. */
static int useful_pile_pickup(
    const State *st, const Move *mv, const float *prob, int n,
    int discard)
{
    const int p = st->turn, o = p ^ 1;
    int best = -1, best_gain = -1000000, best_score = -1000000;
    for (int s = 0; s < NSUIT; s++) {
        if (st->pile_n[s] == 0) continue;
        int pickup = st->pile[s][st->pile_n[s] - 1];
        int wager = CARD_IS_WAGER(pickup);
        if (wager && st->exp_top[p][s] != 0) continue;
        uint64_t suit_mask =
            ((UINT64_C(1) << NRANK) - 1) << (s * NRANK);
        uint64_t wager_mask =
            ((UINT64_C(1) << WAGERS_PER_SUIT) - 1) << (s * NRANK);
        int public_signal =
            (st->known[o] & wager_mask) != 0 || st->exp_n[o][s] != 0;
        int own_numbers =
            __builtin_popcountll(st->hand[p] & suit_mask & ~wager_mask);
        if (wager && !public_signal && own_numbers < 2) continue;

        for (int i = 0; i < n; i++) {
            if (mv[i].draw != s + 1 || mv[i].discard != discard ||
                !top_action_group(mv, prob, n, i, 3))
                continue;
            State after = *st;
            lc_apply(&after, mv[i]);
            HandPlan plan;
            hand_plan_build(&after, p, after.deck_left / 2, &plan);
            int uses_pickup = (plan.used_cards >> pickup) & 1ULL;
            State without = after;
            without.hand[p] &= ~(1ULL << pickup);
            if (without.hand_n[p] > 0) without.hand_n[p]--;
            HandPlan no_pickup;
            hand_plan_build(&without, p, after.deck_left / 2, &no_pickup);
            int gain = plan.score - no_pickup.score;
            if ((!wager && (!uses_pickup || gain <= 0)) ||
                (wager && !uses_pickup && !public_signal))
                continue;
            if (best < 0 || gain > best_gain ||
                (gain == best_gain && plan.score > best_score) ||
                (gain == best_gain && plan.score == best_score &&
                 prob[i] > prob[best])) {
                best = i;
                best_gain = gain;
                best_score = plan.score;
            }
        }
    }
    return best;
}

/* Recheck a proposed ordinary policy-prefix override on fresh worlds with a
 * coherent continuation actor.  Suit mappings are balanced (counts differ by
 * at most one), fixed for each complete trajectory, and shared by every root
 * candidate.  In policy-prefix mode 3, the two players receive independently
 * stratified mappings.  This is a zero-extra-forward robustness panel: it no
 * longer assumes that an unknown opponent shares the root player's arbitrary
 * network orientation, while preserving one temporally coherent policy per
 * player and common random numbers across candidates.  Without an explicit
 * evidence gate the proposal survives only when the independent panel selects
 * the same numerical leader.  With paired evidence/effect thresholds, the
 * proposal instead has to clear both a standard-error bar and a practical
 * objective floor against candidate zero; a statistically tied alternative
 * leader does not erase an independently verified improvement. */
static int confirm_trusted_prefix(
    const Agent *a, const NetEvalPlan *eval_plan,
    const State *st, int p,
    const Move *mv, const int *order, int ntrusted, int proposed,
    int cont_prune, int cont_sym, const BeliefDist *belief, int have_belief,
    uint64_t seed, int cap, int *worlds_out,
    double *q_out, double *se_out, double *delta_out, double *dse_out,
    int *agreement_out, int *gate_passed_out, PlayoutWork *work)
{
    *worlds_out = 0;
    *agreement_out = 0;
    *gate_passed_out = 0;
    if (proposed <= 0 || ntrusted <= 1) return proposed;
    int worlds = a->confirm_dets > 0 ? a->confirm_dets : cap;
    if (worlds < 2) worlds = 2;
    int replan_worlds, replan_cores;
    historical_replan_config(a, &replan_worlds, &replan_cores);
    LateAssignmentPlan assignment_plan;
    int assignment_order[LATE_REPLAN_MAX_ASSIGNMENTS];
    int unique_worlds = 0;
    if (a->no_belief && !have_belief && st->deck_left >= 2 &&
        st->deck_left <= LATE_REPLAN_MAX_DECK) {
        unique_worlds = late_unique_assignment_order(
            st, p, worlds, LATE_PANEL_TRUSTED_ASSIGNMENTS,
            &assignment_plan, assignment_order);
        if (unique_worlds > 0) worlds = unique_worlds;
    }

    uint8_t perms[120][NSUIT];
    int nperm = suit_permutations(cont_sym, perms);
    Rng perm_rng;
    rng_seed(&perm_rng, seed ^ UINT64_C(0x8CB92BA72F3D8DD7));
    int offset = (int)rng_below(&perm_rng, (uint32_t)nperm);
    int other_offset = (int)rng_below(&perm_rng, (uint32_t)nperm);
    Rng world_rng;
    rng_seed(&world_rng, seed ^ UINT64_C(0xDB4F0B9175AE2165));
    LateReplanCache replan_cache = { 0 };
    double total[MAX_CAND] = { 0 };
    double total2[MAX_CAND] = { 0 };
    double diff_total[MAX_CAND] = { 0 };
    double diff_total2[MAX_CAND] = { 0 };
    double margin[MAX_CAND] = { 0 };
    for (int d = 0; d < worlds; d++) {
        State world;
        if (unique_worlds > 0)
            late_assignment_world(
                &assignment_plan, p, assignment_order[d], &world);
        else if (have_belief)
            belief_dist_sample(st, p, &world_rng, belief, &world);
        else
            determinize(st, p, &world_rng, &world);
        const uint8_t *perm = perms[(offset + d) % nperm];
        const uint8_t *other_perm = NULL;
        if (a->policy_prefix_mode == 3) {
            /* Enumerate the product group in row-major order, up to independent
             * seed-dependent offsets.  Every complete nperm^2 block contains
             * every ordered pair exactly once; partial blocks keep each
             * marginal balanced to within one trajectory. */
            int other_index =
                (other_offset + (d / nperm) + (d % nperm)) % nperm;
            other_perm = perms[other_index];
        }
        double world_obj[MAX_CAND] = { 0 };
        for (int c = 0; c < ntrusted; c++) {
            State s = world;
            lc_apply(&s, mv[order[c]]);
            uint64_t wseed = seed ^
                (UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(d + 1));
            LateReplanContext replan = late_replan_context_init(
                replan_worlds, replan_cores,
                LATE_PANEL_TRUSTED_CONFIRM, &replan_cache);
            int m = playout(a->net, eval_plan, &s, p, cont_prune, NULL, perm,
                            other_perm, p ^ 1, 0,
                            cont_sym, a->plan_deck_max,
                            a->plan_block_gap,
                            a->draw_playout_deck_max, a->confirm_temp,
                            wseed, a->win_q,
                            a->exact_terminal,
                            replan_worlds, replan_cores,
                            replan_worlds > 0 ? &replan : NULL,
                            NULL, work);
            double obj = rollout_terminal_objective(&s, p, a->win_q);
            world_obj[c] = obj;
            total[c] += obj;
            total2[c] += obj * obj;
            margin[c] += m;
        }
        for (int c = 1; c < ntrusted; c++) {
            double diff = world_obj[c] - world_obj[0];
            diff_total[c] += diff;
            diff_total2[c] += diff * diff;
        }
    }
    *worlds_out = worlds;
    for (int c = 0; c < ntrusted; c++) {
        q_out[c] = total[c] / worlds;
        double centered = total2[c] - total[c] * total[c] / worlds;
        if (centered < 0.0) centered = 0.0;
        se_out[c] = sqrt(centered / (worlds - 1) / worlds);
        delta_out[c] = diff_total[c] / worlds;
        double diff_centered =
            diff_total2[c] - diff_total[c] * diff_total[c] / worlds;
        if (diff_centered < 0.0) diff_centered = 0.0;
        dse_out[c] = c == 0
            ? 0.0 : sqrt(diff_centered / (worlds - 1) / worlds);
    }
    int leader = 0;
    for (int c = 1; c < ntrusted; c++)
        if (total[c] > total[leader] ||
            (total[c] == total[leader] && margin[c] > margin[leader]))
            leader = c;
    *agreement_out = leader == proposed;
    int gate_requested = a->prefix_confirm_k > 0.0f ||
                         a->prefix_confirm_min > 0.0f;
    int gate_complete = a->prefix_confirm_k > 0.0f &&
                        a->prefix_confirm_min > 0.0f;
    *gate_passed_out = gate_requested
        ? (gate_complete &&
           delta_out[proposed] > a->prefix_confirm_k * dse_out[proposed] &&
           delta_out[proposed] > a->prefix_confirm_min)
        : *agreement_out;
    /* With an explicit paired gate, the fresh panel's job is to verify that
     * the proposed move beats the deployed baseline—not to reproduce an exact
     * argmax among several statistically tied improvements.  Requiring the
     * same numerical leader made two equally good stalls cancel each other and
     * fall back to a clearly losing deck-end move.  The match-tested live actor
     * has no paired gate and therefore retains its stricter exact-leader
     * agreement behavior. */
    if (gate_requested) return *gate_passed_out ? proposed : 0;
    return *agreement_out ? proposed : 0;
}

static Move rollout_move_impl(const struct Agent *a, const State *st,
                              Rng *rng, float *out_value,
                              SearchStats *stats, int use_eval_plan)
{
    NetEvalPlan eval_plan_storage;
    const NetEvalPlan *eval_plan = NULL;
    if (a->net && use_eval_plan) {
        net_eval_plan_init(a->net, &eval_plan_storage);
        eval_plan = &eval_plan_storage;
    }
    if (stats) {
        memset(stats, 0, sizeof *stats);
        stats->policy_top = -1;
        stats->metric_kind = SEARCH_METRIC_NETWORK_VALUE;
        stats->late_resolver_practical_min = a->bounded_late_min;
        for (int i = 0; i < MAX_MOVES; i++) stats->qw[i] = -1.0;
    }
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    float value = 0.0f;
    int n;
    if (a->net) {
        n = policy_probs_sym_plan(
            a->net, st, mv, prob, &value, a->symmetries, eval_plan);
    } else if (a->exact_terminal && st->deck_left == 1 && !st->over) {
        /* The exact turn does not need sampled deck values.  In particular,
         * hrollout must not consume its gameplay RNG while reporting a
         * deterministic zero-world terminal decision. */
        n = lc_moves(st, mv);
        for (int i = 0; i < n; i++) prob[i] = 0.0f;
    } else {
        DrawSamples ds;
        draw_samples_init(st, st->turn, rng, 6, &ds);
        n = lc_moves(st, mv);
        for (int i = 0; i < n; i++) prob[i] = move_value_heur(st, mv[i], &ds);
    }
    const int nlegal = n;
    if (n <= 1) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = n;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = n == 1;
            stats->skip_reason = SEARCH_SKIP_FORCED;
            stats->raw_best = 0;
            stats->policy_mass = n == 1 ? 1.0 : 0.0;
            if (n == 1) {
                stats->mv[0] = mv[0]; stats->visits[0] = 0; stats->q[0] = value;
                stats->se[0] = 0.0; stats->prior[0] = 1.0;
            }
            stats->value = value;
        }
        Move none = { 0, 0, 0 };
        return n == 1 ? mv[0] : none;
    }

    int policy_top_index = 0;
    for (int i = 1; i < n; i++)
        if (prob[i] > prob[policy_top_index]) policy_top_index = i;
    Move policy_top_move = mv[policy_top_index];

    /* The last turn is a solved game, not a Monte Carlo estimate.  Search all
     * semantic card/action cores, end through the deck, and optimize the same
     * round/final-match objective used by rollout.  Doing this before every
     * phase or confidence gate guarantees that the playing actor cannot skip
     * the exact result. */
    double exact_objective = 0.0;
    int exact_index = a->exact_terminal ? rollout_exact_terminal_choice(
        st, mv, prob, n, a->win_q, &exact_objective) : -1;
    if (exact_index >= 0) {
        int policy_terminal_index = -1;
        for (int i = 0; i < n; i++)
            if (mv[i].draw == 0 &&
                mv[i].card == policy_top_move.card &&
                mv[i].discard == policy_top_move.discard) {
                policy_terminal_index = i;
                break;
            }
        if (policy_terminal_index < 0) policy_terminal_index = exact_index;
        State policy_terminal = *st;
        lc_apply_play(&policy_terminal, mv[policy_terminal_index]);
        policy_terminal.deck_left = 0;
        policy_terminal.over = 1;
        double policy_terminal_objective = rollout_terminal_objective(
            &policy_terminal, st->turn, a->win_q);
        int rows = exact_index == policy_terminal_index ? 1 : 2;
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = rows;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 1;
            stats->skip_reason = SEARCH_SKIP_LAST_DECK;
            stats->raw_best = 0;
            /* A pile-draw policy leader is shown separately in the policy
             * table.  Do not label its canonical terminal deck variant as
             * the literal policy move. */
            stats->policy_top = policy_top_index == policy_terminal_index
                ? (rows == 1 ? 0 : 1) : -1;
            stats->deck_end_baseline = 1;
            stats->metric_kind = SEARCH_METRIC_LAST_DECK_RULE;
            stats->policy_mass = prob[exact_index] +
                (rows == 2 ? prob[policy_terminal_index] : 0.0f);
            stats->mv[0] = mv[exact_index];
            stats->q[0] = exact_objective;
            stats->prior[0] = prob[exact_index];
            stats->visits[0] = 0;
            if (rows == 2) {
                stats->mv[1] = mv[policy_terminal_index];
                stats->q[1] = policy_terminal_objective;
                stats->delta[1] =
                    policy_terminal_objective - exact_objective;
                stats->prior[1] = prob[policy_terminal_index];
                stats->visits[1] = 0;
            }
            stats->value = (float)exact_objective;
        }
        return mv[exact_index];
    }

    /* Experimental bounded late resolver.  Unlike the historical recursive
     * evaluator, it constructs the mover's finite root support once and
     * carries each particle through the complete continuation.  Once both
     * horizon panels complete, this is the authoritative root gate: a stable
     * challenger must clear the practical floor in both horizons, otherwise
     * the literal policy baseline is retained.  Falling into a different
     * evaluator after a completed rejection would silently undo that gate. */
    if (bounded_late_root_enabled(a) && st->deck_left >= 2 &&
        st->deck_left <= LATE_REPLAN_MAX_DECK) {
        Move resolved = { 0 };
        LateResolverStats late_stats;
        int late_passed = late_resolver_choose(
            a->net, st, a->win_q, 3, a->symmetries,
            a->root_width, a->bounded_late_min,
            &resolved, &late_stats);
        if (stats) {
            stats->late_resolver_attempted = 1;
            stats->late_resolver_stable = late_stats.stable;
            stats->late_resolver_passed = late_stats.passed;
            stats->late_resolver_support = late_stats.support;
            stats->late_resolver_candidates = late_stats.root_candidates;
            stats->late_resolver_h2_best = late_stats.horizon2_best;
            stats->late_resolver_h4_best = late_stats.horizon4_best;
            stats->late_resolver_h2_value = late_stats.horizon2_value;
            stats->late_resolver_h4_value = late_stats.horizon4_value;
            stats->late_resolver_h2_delta = late_stats.horizon2_delta;
            stats->late_resolver_h4_delta = late_stats.horizon4_delta;
            stats->late_resolver_h2_nodes = late_stats.horizon2_nodes;
            stats->late_resolver_h4_nodes = late_stats.horizon4_nodes;
            stats->late_resolver_h2_root_nodes =
                late_stats.horizon2_root_nodes;
            stats->late_resolver_h4_root_nodes =
                late_stats.horizon4_root_nodes;
            stats->late_resolver_h2_frozen_opponent_nodes =
                late_stats.horizon2_frozen_opponent_nodes;
            stats->late_resolver_h4_frozen_opponent_nodes =
                late_stats.horizon4_frozen_opponent_nodes;
            stats->late_resolver_h2_transitions =
                late_stats.horizon2_transitions;
            stats->late_resolver_h4_transitions =
                late_stats.horizon4_transitions;
            stats->late_resolver_h2_deviation_evals =
                late_stats.horizon2_deviation_evals;
            stats->late_resolver_h4_deviation_evals =
                late_stats.horizon4_deviation_evals;
            stats->late_resolver_h2_exact_leaves =
                late_stats.horizon2_exact_leaves;
            stats->late_resolver_h4_exact_leaves =
                late_stats.horizon4_exact_leaves;
            for (int i = 0; i < late_stats.root_candidates && i < 6; i++) {
                stats->late_resolver_candidate[i] =
                    late_stats.candidate[i];
                stats->late_resolver_prior[i] = late_stats.prior[i];
                stats->late_resolver_h2_q[i] = late_stats.horizon2_q[i];
                stats->late_resolver_h4_q[i] = late_stats.horizon4_q[i];
            }
        }

        /* Candidate zero is promised to be the literal complete-move policy
         * argmax.  Resolve it back through the engine's current legal list so
         * physical wager IDs cannot make a semantically legal move look
         * invalid, and verify the promised baseline before granting authority. */
        int baseline = !late_stats.unavailable &&
                       late_stats.root_candidates > 0
            ? find_semantic_move(mv, n, late_stats.candidate[0]) : -1;
        int valid_baseline = baseline >= 0 &&
            same_semantic_move(mv[baseline], policy_top_move);
        if (!late_stats.unavailable && valid_baseline) {
            int selected = 0;
            int selected_legal = baseline;
            int challenger = late_stats.horizon4_best;
            int authorize_override = late_passed && late_stats.stable &&
                late_stats.horizon2_best == challenger && challenger > 0 &&
                challenger < late_stats.root_candidates &&
                late_stats.horizon2_delta > a->bounded_late_min &&
                late_stats.horizon4_delta > a->bounded_late_min;
            if (authorize_override) {
                int legal = find_semantic_move(
                    mv, n, late_stats.candidate[challenger]);
                if (legal >= 0) {
                    selected = challenger;
                    selected_legal = legal;
                }
            }
            /* Return the engine's canonical legal representative rather than
             * the resolver's semantic card ID.  They normally coincide, but
             * wager copies are intentionally identity-free. */
            Move selected_move = mv[selected_legal];
            if (out_value) *out_value = value;
            if (stats) {
                stats->late_resolver_completed = 1;
                stats->late_resolver_used = 1;
                stats->late_resolver_retained = selected == 0;
                stats->late_resolver_override = selected != 0;
                stats->n = late_stats.root_candidates;
                stats->nlegal = nlegal;
                stats->worlds = late_stats.support;
                stats->max_worlds = late_stats.support;
                stats->resolved = 1;
                stats->raw_best = late_stats.horizon4_best;
                stats->policy_top = 0;
                stats->selection_reference = 0;
                stats->metric_kind = SEARCH_METRIC_ROLLOUT;
                stats->skip_reason = SEARCH_SKIP_NONE;
                stats->policy_mass = 0.0;
                for (int i = 0; i < late_stats.root_candidates; i++) {
                    stats->mv[i] = late_stats.candidate[i];
                    stats->q[i] = late_stats.horizon4_q[i];
                    stats->delta[i] = late_stats.horizon4_q[i] -
                                      late_stats.horizon4_q[0];
                    stats->rdelta[i] = stats->delta[i];
                    stats->prior[i] = late_stats.prior[i];
                    stats->visits[i] = late_stats.support;
                    stats->policy_mass += late_stats.prior[i];
                }
                stats->value = (float)late_stats.horizon4_q[selected];
                stats->exact_terminal_leaves =
                    late_stats.horizon2_exact_leaves +
                    late_stats.horizon4_exact_leaves;
            }
            return selected_move;
        }
    }

    int baseline_index = -1;
    int deck_end_baseline = 0;
    int baseline_from_planner = 0;
    if (baseline_index < 0 && a->net && a->plan_deck_max > 0 &&
        a->plan_block_gap > 0 && st->deck_left <= a->plan_deck_max) {
        int planned_order[MAX_MOVES];
        for (int i = 0; i < n; i++) planned_order[i] = i;
        int planned_keep = n < 8 ? n : 8;
        for (int i = 0; i < planned_keep; i++) {
            int top = i;
            for (int j = i + 1; j < n; j++)
                if (prob[planned_order[j]] > prob[planned_order[top]])
                    top = j;
            int tmp = planned_order[i];
            planned_order[i] = planned_order[top];
            planned_order[top] = tmp;
        }
        baseline_index = hand_plan_conservative_choose(
            st, st->turn, mv, prob, planned_order, planned_keep,
            (st->deck_left + 1) / 2, a->plan_block_gap);
        baseline_from_planner =
            baseline_index >= 0 && baseline_index != policy_top_index;
    }
    if (baseline_index < 0) baseline_index = policy_top_index;
    int before_draw_plan = baseline_index;
    if (a->net && a->draw_root_deck_max > 0 &&
        st->deck_left <= a->draw_root_deck_max)
        baseline_index = hand_plan_choose_draw_source(
            st, st->turn, mv, prob, n, baseline_index);
    int draw_planned_baseline = baseline_index != before_draw_plan;
    int stop_on_baseline = baseline_from_planner;
    int changed_baseline = stop_on_baseline || draw_planned_baseline;
    Move baseline_move = mv[baseline_index];
    int safe_discard_index = a->semantic_cand
        ? isolated_one_sided_wager_discard(st, mv, prob, n) : -1;
    int pile_play_index = a->semantic_cand
        ? useful_pile_pickup(st, mv, prob, n, 0) : -1;
    int pile_discard_index = a->semantic_cand
        ? useful_pile_pickup(st, mv, prob, n, 1) : -1;
    const int have_safe_discard = safe_discard_index >= 0;
    const int have_pile_play = pile_play_index >= 0;
    const int have_pile_discard = pile_discard_index >= 0;
    const Move safe_discard_move =
        have_safe_discard ? mv[safe_discard_index] : policy_top_move;
    const Move pile_play_move =
        have_pile_play ? mv[pile_play_index] : policy_top_move;
    const Move pile_discard_move =
        have_pile_discard ? mv[pile_discard_index] : policy_top_move;

    /* A validated visible-hand correction is the deployed baseline, not
     * another noisy rollout candidate.  Continuing to simulate here allowed
     * the same myopic scheduler error to "confirm" itself and undo the exact
     * correction.  Report both the corrected baseline and raw policy, but
     * spend no hidden worlds. */
    if (stop_on_baseline) {
        int turns = (st->deck_left + 1) / 2;
        HandPlan plan;
        hand_plan_build(st, st->turn, turns, &plan);
        int selected_score = baseline_from_planner
            ? hand_plan_score_after_play(
                  st, st->turn, baseline_move.card, turns)
            : plan.score;
        int policy_score = baseline_from_planner
            ? hand_plan_score_after_play(
                  st, st->turn, policy_top_move.card, turns)
            : plan.score;
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 2;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 1;
            stats->skip_reason = SEARCH_SKIP_VISIBLE_PLAN;
            stats->raw_best = 0;
            stats->policy_top = 1;
            stats->planned_baseline = baseline_from_planner;
            stats->draw_planned_baseline = draw_planned_baseline;
            stats->deck_end_baseline = deck_end_baseline;
            stats->metric_kind = SEARCH_METRIC_VISIBLE_PLAN;
            stats->planner_turns = baseline_from_planner ? turns : 0;
            stats->planner_score = baseline_from_planner ? plan.score : 0;
            stats->planner_policy_score =
                baseline_from_planner ? policy_score : 0;
            stats->planner_regret =
                baseline_from_planner ? plan.score - policy_score : 0;
            stats->planner_policy_block = baseline_from_planner
                ? hand_plan_block_cost(st, st->turn, policy_top_move.card) : 0;
            stats->planner_selected_block = baseline_from_planner
                ? hand_plan_block_cost(st, st->turn, baseline_move.card) : 0;
            stats->policy_mass =
                prob[baseline_index] + prob[policy_top_index];
            stats->mv[0] = baseline_move;
            stats->mv[1] = policy_top_move;
            stats->q[0] = baseline_from_planner ? selected_score : value;
            stats->q[1] = baseline_from_planner ? policy_score : value;
            stats->delta[0] = 0.0;
            stats->delta[1] = baseline_from_planner
                ? (double)(policy_score - selected_score) : 0.0;
            stats->prior[0] = prob[baseline_index];
            stats->prior[1] = prob[policy_top_index];
            stats->value = value;
        }
        return baseline_move;
    }

    /* Outside the rollout window, the validated scheduler remains part of
     * the policy actor.  Only the Monte Carlo comparison is phase-gated,
     * except for the narrow one-sided-wager signal: that public constraint is
     * how an otherwise low-prior correction at an early round ply earns a
     * focused comparison. */
    int outside_ply =
        (a->ply_lo > 0 && st->nply < a->ply_lo) ||
        (a->ply_hi > 0 && st->nply >= a->ply_hi);
    int outside_deck = a->deck_max > 0 && st->deck_left > a->deck_max;
    int urgent_semantic =
        (outside_ply || outside_deck) && safe_discard_index >= 0 &&
        !move_equal(safe_discard_move, baseline_move);
    if ((outside_ply || outside_deck) && !urgent_semantic) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 0;
            stats->skip_reason = outside_ply ? SEARCH_SKIP_PLY_WINDOW
                                             : SEARCH_SKIP_DECK_PHASE;
            stats->raw_best = 0;
            stats->policy_mass = prob[baseline_index];
            stats->policy_top = changed_baseline ? -1 : 0;
            stats->planned_baseline = baseline_from_planner;
            stats->draw_planned_baseline = draw_planned_baseline;
            stats->deck_end_baseline = deck_end_baseline;
            stats->mv[0] = baseline_move;
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0;
            stats->prior[0] = prob[baseline_index];
            stats->value = value;
        }
        return baseline_move;
    }

    /* confidence gate: when the policy is already near-certain, searching can
     * only confirm it or override it with noise -- return the policy move and
     * spend the compute where decisions are actually contested */
    if (a->gate > 0.0f) {
        if (prob[policy_top_index] >= a->gate) {
            if (out_value) *out_value = value;
            if (stats) {
                stats->n = 1;
                stats->nlegal = nlegal;
                stats->worlds = 0;
                stats->max_worlds = a->dets;
                stats->resolved = 0;
                stats->skip_reason = SEARCH_SKIP_POLICY_CONFIDENCE;
                stats->raw_best = 0;
                stats->policy_mass = prob[baseline_index];
                stats->policy_top = changed_baseline ? -1 : 0;
                stats->planned_baseline = baseline_from_planner;
                stats->draw_planned_baseline = draw_planned_baseline;
                stats->deck_end_baseline = deck_end_baseline;
                stats->mv[0] = baseline_move;
                stats->visits[0] = 0;
                stats->q[0] = value;
                stats->se[0] = 0.0;
                stats->prior[0] = prob[baseline_index];
                stats->value = value;
            }
            return baseline_move;
        }
    }

    /* Optional dead-discard search-focus heuristic.  Apply it only after the
     * raw-policy phase and confidence gates: a late-only search must return
     * the untouched champion early, and confidence means the champion's
     * actual probability rather than a renormalized conditional one. */
    if (a->prune_dom && n > 1) {
        uint64_t dead = lc_dead_cards(st);
        if (dead & st->hand[st->turn]) {
            int k = 0;
            for (int i = 0; i < n; i++) {
                if (lc_discard_dominated(st, mv[i], dead)) continue;
                mv[k] = mv[i];
                prob[k] = prob[i];
                k++;
            }
            if (k > 0) {
                n = k;
                if (a->net) {
                    double kept = 0.0;
                    for (int i = 0; i < n; i++) kept += prob[i];
                    if (kept > 0.0)
                        for (int i = 0; i < n; i++)
                            prob[i] = (float)(prob[i] / kept);
                }
            }
        }
    }
    /* Root-focus pruning compacts mv[] and prob[].  Semantic candidates were
     * identified before the phase gate, so remap their move identities rather
     * than retaining now-stale array indices. */
    safe_discard_index =
        have_safe_discard ? find_move(mv, n, safe_discard_move) : -1;
    pile_play_index =
        have_pile_play ? find_move(mv, n, pile_play_move) : -1;
    pile_discard_index =
        have_pile_discard ? find_move(mv, n, pile_discard_move) : -1;
    if (n == 1) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 1;
            stats->skip_reason = SEARCH_SKIP_ROOT_FOCUS;
            stats->raw_best = 0;
            stats->policy_mass = 1.0;
            stats->mv[0] = mv[0];
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0;
            stats->prior[0] = 1.0;
            stats->value = value;
        }
        return mv[0];
    }

    /* Candidates are a policy-guided prefix.  Analysis compute belongs on
     * moves the champion genuinely considers, not forced variants with
     * effectively zero prior.  A cumulative-mass target is more stable than a
     * fixed width when the policy ranges from near-certain to genuinely broad. */
    int order[MAX_MOVES];
    for (int i = 0; i < n; i++) order[i] = i;
    int maxcand = a->root_width < MAX_CAND ? a->root_width : MAX_CAND;
    if (maxcand < 1) maxcand = 1;
    if (maxcand > n) maxcand = n;
    int nsorted = maxcand;
    if (a->eval_cand > nsorted) {
        nsorted = a->eval_cand < MAX_CAND ? a->eval_cand : MAX_CAND;
        if (nsorted > n) nsorted = n;
    }
    if ((a->semantic_cand ||
         (a->plan_deck_max > 0 && a->plan_block_gap > 0) ||
         a->draw_variant_cores > 0 ||
         (a->deck2_replan_worlds > 0 && st->deck_left >= 2 &&
          st->deck_left <= LATE_REPLAN_MAX_DECK)) &&
        nsorted < MAX_CAND) {
        nsorted = n < MAX_CAND ? n : MAX_CAND;
    }
    for (int i = 0; i < nsorted; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (prob[order[j]] > prob[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    int ranked[MAX_CAND];
    for (int i = 0; i < nsorted; i++) ranked[i] = order[i];
    int current_baseline = -1, current_policy_top = -1;
    for (int i = 0; i < n; i++) {
        if (move_equal(mv[i], baseline_move)) current_baseline = i;
        if (move_equal(mv[i], policy_top_move)) current_policy_top = i;
    }
    if (current_baseline < 0) {
        current_baseline = current_policy_top >= 0
            ? current_policy_top : order[0];
        baseline_from_planner = 0;
        draw_planned_baseline = 0;
        deck_end_baseline = 0;
        stop_on_baseline = 0;
        changed_baseline = 0;
    }

    int keep = a->min_cand > 1 ? a->min_cand : 1;
    if (keep > maxcand) keep = maxcand;
    int ncand;
    int action_core_candidates = 0;
    int action_draw_candidates = 0;
    if (a->net && a->action_core_count > 0) {
        ncand = build_action_core_shortlist(
            st, mv, prob, n, current_baseline, maxcand,
            a->action_core_count, keep, a->cand_floor, a->cand_mass,
            order, &action_core_candidates, &action_draw_candidates);
    } else {
        ncand = maxcand;
        if (a->net && a->cand_mass > 0.0f) {
            ncand = 0;
            double mass = 0.0;
            while (ncand < maxcand &&
                   (ncand < keep || mass < (double)a->cand_mass)) {
                mass += prob[order[ncand]];
                ncand++;
            }
        } else if (a->net) {
            float floor_p = a->cand_floor > 0.0f ? a->cand_floor : 0.02f;
            while (ncand > keep && prob[order[ncand - 1]] < floor_p) ncand--;
        }
        append_unique(order, &ncand, MAX_CAND, current_baseline);
    }
    /* Enabling the late method promises an actual comparison.  A sharp
     * complete-move prior must not collapse the root to one candidate before
     * the exact-leaf-aware rollout can run.  Admit only the highest-prior
     * different semantic action core; this is still a policy-focused audit. */
    if (a->net && a->deck2_replan_worlds > 0 &&
        a->deck2_replan_cores > 0 && st->deck_left >= 2 &&
        st->deck_left <= LATE_REPLAN_MAX_DECK && ncand == 1) {
        for (int r = 0; r < nsorted; r++)
            if (!same_semantic_action(mv[ranked[r]], mv[order[0]])) {
                append_unique(order, &ncand, MAX_CAND, ranked[r]);
                break;
            }
    }
    int trusted_candidates = ncand;

    /* One exact hand-scheduling challenger is enough.  It comes from the
     * policy's top eight, so this remains a focused audit rather than a scan
     * of every legal move. */
    if (a->net && a->plan_deck_max > 0 && a->plan_block_gap > 0 &&
        st->deck_left <= a->plan_deck_max) {
        int planned = hand_plan_choose(
            st, st->turn, mv, prob, ranked, nsorted,
            (st->deck_left + 1) / 2);
        append_unique(order, &ncand, MAX_CAND, planned);
    }
    int semantic_added = 0;
    if (a->semantic_cand) {
        semantic_added +=
            append_unique(order, &ncand, MAX_CAND, safe_discard_index);
        semantic_added +=
            append_unique(order, &ncand, MAX_CAND, pile_play_index);
        semantic_added +=
            append_unique(order, &ncand, MAX_CAND, pile_discard_index);
    }
    int draw_variant_added = 0;
    if (a->net && a->draw_variant_cores > 0 &&
        (a->draw_variant_deck_max <= 0 ||
         st->deck_left <= a->draw_variant_deck_max)) {
        draw_variant_added = add_top_action_pile_variants(
            mv, prob, n, ranked, nsorted, a->draw_variant_cores,
            order, &ncand);
    }
    /*
     * An early one-sided-wager trigger is deliberately a two-move test.  Its
     * purpose is to rescue one low-prior semantic blind spot, not to turn the
     * entire early policy prefix into a 16,384-world audit.  This keeps the
     * expensive path at baseline versus wager-discard only.
     */
    if (urgent_semantic) {
        order[0] = current_baseline;
        ncand = 1;
        semantic_added =
            append_unique(order, &ncand, MAX_CAND, safe_discard_index);
        draw_variant_added = 0;
        trusted_candidates = 1;
        action_core_candidates = a->action_core_count > 0 ? 1 : 0;
        action_draw_candidates = 0;
    }

    /* Candidate zero is the deployed actor's baseline: ordinarily the raw
     * policy, but occasionally the validated scheduler or exact deck=1 rule.
     * Keep the untouched policy leader in the list and report its true index. */
    int baseline_pos = -1;
    for (int i = 0; i < ncand; i++)
        if (order[i] == current_baseline) baseline_pos = i;
    if (baseline_pos > 0) {
        int tmp = order[0]; order[0] = order[baseline_pos];
        order[baseline_pos] = tmp;
    }

    /* Optional advisory rows are the next policy-ranked moves.  They remain
     * diagnostic-only; semantic additions above are eligible because each is
     * a single rule-derived challenger. */
    int neval = ncand;
    int advisory_target = urgent_semantic ? 0 :
        (a->eval_cand < nsorted ? a->eval_cand : nsorted);
    for (int i = 0; i < advisory_target; i++)
        append_unique(order, &neval, MAX_CAND, ranked[i]);

    int policy_pos = -1;
    for (int i = 0; i < neval; i++)
        if (order[i] == current_policy_top) policy_pos = i;
    if (current_policy_top >= 0 && policy_pos < 0) {
        append_unique(order, &neval, MAX_CAND, current_policy_top);
        for (int i = 0; i < neval; i++)
            if (order[i] == current_policy_top) policy_pos = i;
    }
    /* A mass-based shortlist can legitimately contain only the policy leader.
     * With neither an eligible challenger nor requested advisory rows, paired
     * rollouts cannot change or teach anything, so spending hundreds of
     * continuation worlds would be pure waste.  Keep the ordinary path when
     * eval_cand requested advisory Q targets even though they cannot play.
     *
     * Consume only the hidden-world draws that the old one-candidate path
     * consumed.  Continuation playouts use forked RNGs, so this preserves the
     * actor's later decision stream exactly while removing nearly all of the
     * wasted forwards. */
    if (ncand == 1 && neval == 1) {
        const int p = st->turn;
        int cap = a->dets > 0 ? a->dets : 1;
        int reps = cap;
        if (a->batch_dets > 0) {
            reps = a->batch_dets < 2 ? 2 : a->batch_dets;
            if (reps > cap) reps = cap;
        }
        BeliefDist singleton_belief;
        int singleton_have_belief =
            a->net && !a->no_belief &&
            belief_dist_init(a->net, st, p, a->symmetries,
                             a->belief_alpha,
                             &singleton_belief);
        for (int d = 0; d < reps; d++) {
            State ignored;
            if (singleton_have_belief)
                belief_dist_sample(st, p, rng, &singleton_belief, &ignored);
            else
                determinize(st, p, rng, &ignored);
        }
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 0;
            stats->skip_reason = SEARCH_SKIP_POLICY_CONFIDENCE;
            stats->raw_best = 0;
            stats->policy_top = policy_pos == 0 ? 0 : -1;
            stats->planned_baseline = baseline_from_planner;
            stats->draw_planned_baseline = draw_planned_baseline;
            stats->deck_end_baseline = deck_end_baseline;
            stats->semantic_candidates = semantic_added;
            stats->draw_variant_candidates = draw_variant_added;
            stats->action_core_candidates = action_core_candidates;
            stats->action_draw_candidates = action_draw_candidates;
            stats->trusted_candidates = trusted_candidates;
            stats->selection_reference = 0;
            stats->policy_mass = shortlist_policy_mass(
                mv, prob, n, order, 1, action_core_candidates);
            stats->mv[0] = mv[order[0]];
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0;
            stats->prior[0] = prob[order[0]];
            stats->value = value;
        }
        return mv[order[0]];
    }

    double sum[MAX_CAND], sumw[MAX_CAND], sumobj[MAX_CAND];
    for (int i = 0; i < neval; i++) {
        sum[i] = 0.0;
        sumw[i] = 0.0;
        sumobj[i] = 0.0;
    }
    const int p = st->turn;
    int cap = a->dets > 0 ? a->dets : 1;
    if (urgent_semantic && cap < URGENT_SEMANTIC_WORLDS)
        cap = URGENT_SEMANTIC_WORLDS;
    int requested_cap = cap;
    LateAssignmentPlan primary_assignment_plan;
    int primary_assignment_order[LATE_REPLAN_MAX_ASSIGNMENTS];
    int primary_unique_worlds = 0;
    int primary_exhaust_support = 0;
    if (a->no_belief && st->deck_left >= 2 &&
        st->deck_left <= LATE_REPLAN_MAX_DECK) {
        primary_unique_worlds = late_unique_assignment_order(
            st, p, cap, LATE_PANEL_PRIMARY_ASSIGNMENTS,
            &primary_assignment_plan, primary_assignment_order);
        if (primary_unique_worlds > 0) {
            primary_exhaust_support =
                requested_cap >= primary_assignment_plan.support;
            cap = primary_unique_worlds;
        }
    }
    int batch = a->batch_dets > 0 ? a->batch_dets : cap;
    if (batch > cap) batch = cap;
    int lastround = st->round == MATCH_ROUNDS - 1;
    double *val = (double *)malloc(sizeof(double) * (size_t)neval * (size_t)cap);
    if (!val) {
        if (out_value) *out_value = value;
        return mv[order[0]];
    }

    /* The root belief distribution is constant across all worlds.  Preparing
     * it once removes hundreds of duplicate network forwards. */
    BeliefDist belief;
    int have_belief = a->net && !a->no_belief &&
                      belief_dist_init(a->net, st, p, a->symmetries,
                                       a->belief_alpha, &belief);
    /* Fork confirmation randomness before the adaptive primary loop.  Its
     * worlds must not depend on which batch boundary happened to stop the
     * discovery estimate, and running a confirmation must not perturb the
     * agent's future decision RNG stream. */
    uint64_t confirm_seed_base =
        rng->s[0] ^ rotl64(rng->s[1], 13) ^
        rotl64(rng->s[2], 29) ^ rotl64(rng->s[3], 47) ^
        UINT64_C(0xA0761D6478BD642F);
    int reps = 0;
    int resolved = 0;
    int rawbest = 0;
    /* Ordinary 1.96-SE intervals are not valid after repeatedly checking up
     * to eight leaders at several batch boundaries.  3.5 is a conservative
     * family-wise guard for as many as 16 looks across eight candidates. */
    double resolve_z = a->override_k > 3.5f ? a->override_k : 3.5;
    int cont_sym = a->playout_symmetries > 0 ? a->playout_symmetries : 1;
    int cont_prune =
        a->playout_prune < 0 ? a->prune_dom : a->playout_prune != 0;
    int random_cont_sym = a->playout_sample > 0;
    int sample_cont_actions = a->playout_sample == 1;
    int fixed_world_sym =
        (a->playout_sample == 3 || a->playout_sample == 4) && cont_sym > 1;
    int role_fixed_world_sym =
        a->playout_sample == 4 && cont_sym > 1;
    uint8_t cont_perms[120][NSUIT];
    int ncont_perms = fixed_world_sym
        ? suit_permutations(cont_sym, cont_perms) : 0;
    /* Modes 3/4 use coherent sampled members of the suit ensemble.  Stratify
     * the panel instead of independently redrawing a member in every world:
     * over any complete group cycle each relabelling receives exactly the
     * same number of hidden worlds, while the offset remains seed-dependent.
     * Mode 4 walks the ordered product group for the second player.  This
     * removes needless audit variance and arbitrary cross-player orientation
     * correlation without changing hidden cards or adding network forwards. */
    int fixed_perm_offset = fixed_world_sym
        ? (int)(confirm_seed_base % (uint64_t)ncont_perms) : 0;
    int other_fixed_perm_offset = role_fixed_world_sym
        ? (int)(rotl64(confirm_seed_base, 31) % (uint64_t)ncont_perms) : 0;
    PlayoutWork work = { 0 };
    int replan_worlds, replan_cores;
    historical_replan_config(a, &replan_worlds, &replan_cores);
    LateReplanCache primary_replan_cache = { 0 };

    for (int d = 0; d < cap; d++) {
        State world;
        if (primary_unique_worlds > 0)
            late_assignment_world(
                &primary_assignment_plan, p,
                primary_assignment_order[d], &world);
        else if (have_belief)
            belief_dist_sample(st, p, rng, &belief, &world);
        else
            determinize(st, p, rng, &world);
        uint64_t wseed = 0x9E3779B97F4A7C15ULL * (uint64_t)(d + 1) ^ rng->s[0];
        for (int c = 0; c < neval; c++) {
            State s = world;                 /* same world for every candidate */
            lc_apply(&s, mv[order[c]]);
            double w;
            Rng pr;
            if (random_cont_sym) rng_seed(&pr, wseed); /* same seed per world */
            const uint8_t *fixed_perm = NULL;
            const uint8_t *other_fixed_perm = NULL;
            if (fixed_world_sym) {
                fixed_perm =
                    cont_perms[(fixed_perm_offset + d) % ncont_perms];
                if (role_fixed_world_sym) {
                    int other_index =
                        (other_fixed_perm_offset + d / ncont_perms +
                         d % ncont_perms) % ncont_perms;
                    other_fixed_perm = cont_perms[other_index];
                }
            }
            LateReplanContext replan = late_replan_context_init(
                replan_worlds, replan_cores, LATE_PANEL_PRIMARY,
                &primary_replan_cache);
            int m = playout(a->net, eval_plan, &s, p, cont_prune,
                            random_cont_sym ? &pr : NULL,
                            fixed_perm, other_fixed_perm, p ^ 1,
                            sample_cont_actions, cont_sym,
                            a->plan_deck_max, a->plan_block_gap,
                            a->draw_playout_deck_max, 0.0f, wseed,
                            a->win_q, a->exact_terminal,
                            replan_worlds, replan_cores,
                            replan_worlds > 0 ? &replan : NULL,
                            &w, &work);
            double obj = rollout_terminal_objective(&s, p, a->win_q);
            if (val) val[(size_t)c * cap + d] = obj;
            sum[c] += m;
            if (w >= 0.0) sumw[c] += w;
            sumobj[c] += obj;
        }
        reps = d + 1;

        /* Sequential paired evaluation: stop once the numerical leader clears
         * every alternative's two-sided confidence bar.  Ambiguous decisions
         * receive more worlds up to cap; obvious ones release the compute. */
        if (reps % batch == 0 || reps == cap) {
            rawbest = 0;
            for (int c = 1; c < neval; c++)
                if (sumobj[c] > sumobj[rawbest]) rawbest = c;
            resolved = reps > 1;
            for (int c = 0; c < neval && resolved; c++) {
                if (c == rawbest) continue;
                double dm = (sumobj[rawbest] - sumobj[c]) / reps;
                double v2 = 0.0;
                for (int j = 0; j < reps; j++) {
                    double x = val[(size_t)rawbest * cap + j]
                             - val[(size_t)c * cap + j] - dm;
                    v2 += x * x;
                }
                double sed = sqrt(v2 / (reps - 1) / reps);
                if (!(dm > resolve_z * sed)) resolved = 0;
            }
            if (a->batch_dets > 0 && resolved && !primary_exhaust_support)
                break;
        }
    }

    /* The raw leader is descriptive.  Move selection is deliberately more
     * conservative: only eligible candidates can win, and every challenger is
     * tested directly against candidate zero, the deployed policy/planner
     * baseline. */
    int eligible_best = 0;
    for (int c = 1; c < ncand; c++) {
        if (sumobj[c] > sumobj[eligible_best] ||
            (sumobj[c] == sumobj[eligible_best] && sum[c] > sum[eligible_best]))
            eligible_best = c;
    }
    /* The optional two-tier mode trusts only the ordinary policy-floor prefix
     * as a direct numerical comparison.  Purposefully admitted low-prior pile
     * and semantic variants must still clear both statistical gates. */
    int reference = 0;
    if (a->policy_prefix_mode > 0) {
        for (int c = 1; c < trusted_candidates; c++)
            if (sumobj[c] > sumobj[reference] ||
                (sumobj[c] == sumobj[reference] && sum[c] > sum[reference]))
                reference = c;
    }
    int proposed_reference = reference;
    int prefix_confirm_worlds = 0;
    int prefix_confirmed = 0;
    double prefix_q[MAX_CAND] = { 0 };
    double prefix_se[MAX_CAND] = { 0 };
    double prefix_delta[MAX_CAND] = { 0 };
    double prefix_dse[MAX_CAND] = { 0 };
    int prefix_numerical_agreement = 0;
    int prefix_gate_passed = 0;
    if (a->policy_prefix_mode >= 2 && proposed_reference != 0) {
        reference = confirm_trusted_prefix(
            a, eval_plan, st, p, mv, order, trusted_candidates,
            proposed_reference,
            cont_prune, cont_sym, &belief, have_belief,
            confirm_seed_base ^ UINT64_C(0xE7037ED1A0B428DB),
            cap, &prefix_confirm_worlds, prefix_q, prefix_se,
            prefix_delta, prefix_dse, &prefix_numerical_agreement,
            &prefix_gate_passed, &work);
        prefix_confirmed = reference == proposed_reference;
    }
    /* override_k == 0 preserves the fully ungated historical rollout agent
     * only when prefix selection is disabled.  Once a prefix mode is enabled,
     * its numerical or fresh-panel consensus result owns the trusted-prefix
     * decision even when the low-prior challenger gate is off. */
    int best = a->override_k <= 0.0f && a->policy_prefix_mode == 0
        ? eligible_best : reference;
    int confirmed = 0, confirm_worlds = 0;
    int qual[MAX_CAND], nqual = 0;
    double csum[MAX_CAND] = { 0 }, csum2[MAX_CAND] = { 0 };
    double cdelta[MAX_CAND] = { 0 }, cdse[MAX_CAND] = { 0 };
    uint8_t pqualified[MAX_CAND] = { 0 };
    uint8_t csupported[MAX_CAND] = { 0 };
    uint8_t guard_rejected[MAX_CAND] = { 0 };

    if (a->override_k > 0.0f && reps > 1) {
        /* Qualify each challenger independently.  Requiring the raw numerical
         * leader to survive made a biased leader hide a smaller real
         * correction: when that leader failed confirmation the old code fell
         * all the way back to the baseline. */
        for (int c = 0; c < ncand; c++) {
            if (c == reference ||
                (a->policy_prefix_mode > 0 && c < trusted_candidates))
                continue;
            double dm = (sumobj[c] - sumobj[reference]) / reps;
            double v2 = 0.0;
            for (int d = 0; d < reps; d++) {
                double x = val[(size_t)c * cap + d]
                         - val[(size_t)reference * cap + d] - dm;
                v2 += x * x;
            }
            double sed = sqrt(v2 / (reps - 1) / reps);
            if (dm > resolve_z * sed && dm > a->override_min) {
                pqualified[c] = 1;
                qual[nqual++] = c;
            }
        }

        if (nqual > 0) {
            /* An exact-ensemble continuation can repeat one downstream policy
             * discontinuity in every paired world, producing a large mean
             * with a misleadingly tiny SE.  Validate all primary qualifiers
             * on one fresh, fixed-size batch of hidden worlds.  Each decision
             * uses a random member of the requested suit group but remains
             * greedy: this perturbs the discontinuity without changing the
             * target into a weak high-entropy policy.  Candidate zero is
             * played out once per world, so no confirmation compute is spent
             * on candidates that failed the primary screen. */
            confirm_worlds =
                a->confirm_dets > 0 ? a->confirm_dets : cap;
            if (urgent_semantic &&
                confirm_worlds < URGENT_SEMANTIC_WORLDS)
                confirm_worlds = URGENT_SEMANTIC_WORLDS;
            if (confirm_worlds < 2) confirm_worlds = 2;
            LateAssignmentPlan confirm_assignment_plan;
            int confirm_assignment_order[LATE_REPLAN_MAX_ASSIGNMENTS];
            int confirm_unique_worlds = 0;
            if (a->no_belief && !have_belief && st->deck_left >= 2 &&
                st->deck_left <= LATE_REPLAN_MAX_DECK) {
                confirm_unique_worlds = late_unique_assignment_order(
                    st, p, confirm_worlds,
                    LATE_PANEL_CHALLENGER_ASSIGNMENTS,
                    &confirm_assignment_plan, confirm_assignment_order);
                if (confirm_unique_worlds > 0)
                    confirm_worlds = confirm_unique_worlds;
            }
            Rng confirm_rng;
            rng_seed(&confirm_rng, confirm_seed_base);
            int confirm_perm_offset = role_fixed_world_sym
                ? (int)(rotl64(confirm_seed_base, 17) %
                        (uint64_t)ncont_perms) : 0;
            int confirm_other_perm_offset = role_fixed_world_sym
                ? (int)(rotl64(confirm_seed_base, 43) %
                        (uint64_t)ncont_perms) : 0;
            LateReplanCache confirm_replan_cache = { 0 };
            for (int d = 0; d < confirm_worlds; d++) {
                State world;
                if (confirm_unique_worlds > 0)
                    late_assignment_world(
                        &confirm_assignment_plan, p,
                        confirm_assignment_order[d], &world);
                else if (have_belief)
                    belief_dist_sample(st, p, &confirm_rng, &belief, &world);
                else
                    determinize(st, p, &confirm_rng, &world);
                uint64_t wseed =
                    UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(d + 1)
                    ^ confirm_rng.s[0];
                State baseline = world;
                lc_apply(&baseline, mv[order[reference]]);
                LateReplanContext baseline_replan = late_replan_context_init(
                    replan_worlds, replan_cores,
                    LATE_PANEL_CHALLENGER_CONFIRM,
                    &confirm_replan_cache);
                Rng brng;
                rng_seed(&brng, wseed);
                const uint8_t *confirm_perm = NULL;
                const uint8_t *confirm_other_perm = NULL;
                if (fixed_world_sym && !a->confirm_exact5) {
                    int pk = role_fixed_world_sym
                        ? (confirm_perm_offset + d) % ncont_perms
                        : (int)rng_below(&brng, (uint32_t)ncont_perms);
                    confirm_perm = cont_perms[pk];
                    if (role_fixed_world_sym) {
                        int other_pk =
                            (confirm_other_perm_offset + d / ncont_perms +
                             d % ncont_perms) % ncont_perms;
                        confirm_other_perm = cont_perms[other_pk];
                    }
                }
                (void)playout(a->net, eval_plan, &baseline, p, cont_prune,
                              a->confirm_exact5 || confirm_perm ? NULL : &brng,
                              confirm_perm, confirm_other_perm, p ^ 1, 0,
                              a->confirm_exact5 ? 5 : cont_sym,
                              a->plan_deck_max, a->plan_block_gap,
                              a->draw_playout_deck_max, a->confirm_temp,
                              wseed, a->win_q, a->exact_terminal,
                              replan_worlds, replan_cores,
                              replan_worlds > 0 ? &baseline_replan : NULL,
                              NULL, &work);
                double bobj =
                    rollout_terminal_objective(&baseline, p, a->win_q);

                for (int q = 0; q < nqual; q++) {
                    int c = qual[q];
                    State challenger = world;
                    lc_apply(&challenger, mv[order[c]]);
                    Rng crng;
                    rng_seed(&crng, wseed);
                    if (confirm_perm && !role_fixed_world_sym)
                        (void)rng_below(&crng, (uint32_t)ncont_perms);
                    LateReplanContext challenger_replan =
                        late_replan_context_init(
                            replan_worlds, replan_cores,
                            LATE_PANEL_CHALLENGER_CONFIRM,
                            &confirm_replan_cache);
                    (void)playout(
                        a->net, eval_plan, &challenger, p, cont_prune,
                        a->confirm_exact5 || confirm_perm ? NULL : &crng,
                        confirm_perm, confirm_other_perm, p ^ 1, 0,
                        a->confirm_exact5 ? 5 : cont_sym,
                        a->plan_deck_max, a->plan_block_gap,
                        a->draw_playout_deck_max, a->confirm_temp,
                        wseed, a->win_q, a->exact_terminal,
                        replan_worlds, replan_cores,
                        replan_worlds > 0 ? &challenger_replan : NULL,
                        NULL, &work);
                    double x =
                        rollout_terminal_objective(&challenger, p, a->win_q)
                        - bobj;
                    csum[c] += x;
                    csum2[c] += x * x;
                }
            }

            for (int q = 0; q < nqual; q++) {
                int c = qual[q];
                cdelta[c] = csum[c] / confirm_worlds;
                if (confirm_worlds > 1) {
                    double cv =
                        csum2[c] - csum[c] * csum[c] / confirm_worlds;
                    if (cv < 0.0) cv = 0.0;
                    cdse[c] =
                        sqrt(cv / (confirm_worlds - 1) / confirm_worlds);
                }
                /* The discovery pass already paid the family-wise 3.5-SE
                 * guard.  This independent two-move check asks for 99%
                 * confirmation and at least half the practical-effect floor. */
                if (cdelta[c] > 2.58 * cdse[c] &&
                    cdelta[c] > 0.5 * a->override_min) {
                    /* Report the evaluator's independent statistical support
                     * separately from the optional strategic safety rail.
                     * Consumers can then distinguish "search supports this"
                     * from "the guard declined to execute it". */
                    csupported[c] = 1;
                    uint64_t dead = lc_dead_cards(st);
                    int baseline_guarded =
                        lc_discard_dominated(st, mv[order[reference]], dead);
                    /*
                     * The generic discard guard predates the semantic
                     * shortlist and can reject a one-sided wager even after
                     * the focused primary and independent confirmation both
                     * support it.  In this narrow public-information case the
                     * opponent cannot score the wager, while any remaining
                     * stall cost is already represented in both 16,384-world
                     * continuation batches.  Let that evidence supersede the
                     * coarse generic guard.  The maintained urgent path uses
                     * 16,384 primary and 16,384 fresh worlds because the
                     * reviewed ply-61 signal was not stable at 4,096.
                     */
                    int confirmed_one_sided_wager =
                        have_safe_discard &&
                        move_equal(mv[order[c]], safe_discard_move);
                    if (a->discard_guard && !baseline_guarded &&
                        !confirmed_one_sided_wager &&
                        lc_discard_dominated(st, mv[order[c]], dead)) {
                        guard_rejected[c] = 1;
                        continue;
                    }
                    if (best == reference || cdelta[c] > cdelta[best] ||
                        (cdelta[c] == cdelta[best] &&
                         sumobj[c] > sumobj[best]))
                        best = c;
                }
            }
            confirmed = best != reference;
        }
    }
    /* The engine cap is a safety fuse, not a game-theoretic terminal.  Never
     * let an unfinished capped continuation authorize an override.  Retain
     * its rows for diagnosis, but fall back to the deployed baseline and mark
     * the comparison unresolved. */
    if (work.unfinished_cap_leaves > 0) {
        best = 0;
        reference = 0;
        resolved = 0;
        confirmed = 0;
        prefix_confirmed = 0;
    }
    float bestq = (float)(sumobj[best] / reps);
    if (stats) {
        stats->n = neval;
        stats->nlegal = nlegal;
        stats->worlds = reps;
        stats->max_worlds = cap;
        stats->resolved = resolved;
        stats->raw_best = rawbest;
        stats->policy_top = policy_pos;
        stats->planned_baseline = baseline_from_planner;
        stats->draw_planned_baseline = draw_planned_baseline;
        stats->deck_end_baseline = deck_end_baseline;
        stats->semantic_candidates = semantic_added;
        stats->draw_variant_candidates = draw_variant_added;
        stats->action_core_candidates = action_core_candidates;
        stats->action_draw_candidates = action_draw_candidates;
        stats->trusted_candidates = trusted_candidates;
        stats->prefix_proposed = proposed_reference;
        stats->selection_reference = reference;
        stats->trusted_prefix_override =
            a->policy_prefix_mode > 0 && reference != 0;
        stats->prefix_numerical_agreement = prefix_numerical_agreement;
        stats->prefix_gate_passed = prefix_gate_passed;
        stats->prefix_confirmed = prefix_confirmed;
        stats->prefix_confirm_worlds = prefix_confirm_worlds;
        stats->metric_kind = SEARCH_METRIC_ROLLOUT;
        stats->confirmed = confirmed;
        stats->confirm_worlds = confirm_worlds;
        stats->exact_terminal_leaves = work.exact_terminal_leaves;
        stats->unfinished_cap_leaves = work.unfinished_cap_leaves;
        stats->cycle_breaks = work.cycle_breaks;
        stats->cap_reserve_forces = work.cap_reserve_forces;
        stats->deck2_replans = work.deck2_replans;
        stats->deck2_replan_worlds = work.deck2_replan_worlds;
        stats->deck2_replan_evals = work.deck2_replan_evals;
        stats->deck2_replan_cap_hits = work.deck2_replan_cap_hits;
        stats->deck2_replan_cache_hits = work.deck2_replan_cache_hits;
        stats->deck2_replan_cycle_closures =
            work.deck2_replan_cycle_closures;
        stats->deck2_replan_max_depth = work.deck2_replan_max_depth;
        stats->deck2_replan_root_calls = work.deck2_replan_root_calls;
        stats->deck2_replan_root_worlds = work.deck2_replan_root_worlds;
        stats->deck2_replan_max_stall_chain =
            work.deck2_replan_max_stall_chain;
        stats->deck2_replan_low_world_fallbacks =
            work.deck2_replan_low_world_fallbacks;
        stats->policy_mass = shortlist_policy_mass(
            mv, prob, n, order, ncand, action_core_candidates);
        for (int c = 0; c < neval; c++) {
            stats->mv[c] = mv[order[c]];
            stats->visits[c] = reps;
            stats->q[c] = sumobj[c] / reps;
            stats->qw[c] = lastround ? sumw[c] / reps : -1.0;
            stats->prior[c] = prob[order[c]];
            double qv = 0.0, dv = 0.0, rdv = 0.0;
            double dm = (sumobj[c] - sumobj[0]) / reps;
            double rdm =
                (sumobj[c] - sumobj[reference]) / reps;
            if (reps > 1) {
                for (int d = 0; d < reps; d++) {
                    double qx = val[(size_t)c * cap + d] - stats->q[c];
                    double dx = val[(size_t)c * cap + d] - val[d] - dm;
                    double rdx = val[(size_t)c * cap + d]
                               - val[(size_t)reference * cap + d] - rdm;
                    qv += qx * qx;
                    dv += dx * dx;
                    rdv += rdx * rdx;
                }
                qv = sqrt(qv / (reps - 1) / reps);
                dv = sqrt(dv / (reps - 1) / reps);
                rdv = sqrt(rdv / (reps - 1) / reps);
            }
            stats->se[c] = qv;
            stats->delta[c] = dm;
            stats->dse[c] = c == 0 ? 0.0 : dv;
            stats->rdelta[c] = rdm;
            stats->rdse[c] = c == reference ? 0.0 : rdv;
            stats->cdelta[c] = cdelta[c];
            stats->cdse[c] = cdse[c];
            stats->prefix_q[c] = prefix_q[c];
            stats->prefix_se[c] = prefix_se[c];
            stats->prefix_delta[c] = prefix_delta[c];
            stats->prefix_dse[c] = prefix_dse[c];
            stats->pqualified[c] = pqualified[c];
            stats->csupported[c] = csupported[c];
            stats->guard_rejected[c] = guard_rejected[c];
        }
        stats->value = bestq;
    }
    free(val);
    /* Keep out_value on one stable scale across searched and skipped moves:
     * it is always the policy-network continuation value (ensemble-averaged
     * when enabled). SearchStats.value/q carry the rollout objective. */
    if (out_value) *out_value = value;
    return mv[order[best]];
}

Move rollout_move(const struct Agent *a, const State *st, Rng *rng,
                  float *out_value, SearchStats *stats)
{
    return rollout_move_impl(a, st, rng, out_value, stats, 1);
}

/* Runtime-regression oracle: run the identical actor through the ordinary
 * inference path.  Kept out of the public header because gameplay should
 * always use the checkpoint-proven plan above. */
Move rollout_move_reference_for_test(
    const struct Agent *a, const State *st, Rng *rng,
    float *out_value, SearchStats *stats)
{
    return rollout_move_impl(a, st, rng, out_value, stats, 0);
}
