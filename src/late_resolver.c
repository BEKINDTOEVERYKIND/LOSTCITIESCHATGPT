/* late_resolver.c -- finite-support, bounded late policy improvement.
 *
 * The old recursive rollout re-determinizes at child decisions.  That can
 * make an earlier stall depend on a continuation policy which is not the one
 * later used by the mover.  This module instead constructs the current
 * mover's complete ordered hidden support once and carries each world through
 * the whole trajectory.  Policies are keyed only by the mover's observation
 * and the public action path.
 *
 * The restricted policy is initialized from the champion: top three semantic
 * card/action cores, with each core's deck exit and best policy-supported pile
 * variant (at most six complete moves).  Two frozen-policy improvement sweeps
 * evaluate every local deviation on common root particles.  This is bounded
 * particle policy improvement, not CFR and not a claim of equilibrium play.
 * The official rules allow endless pile cycles, so horizons H=2 and H=4 are
 * solved separately; a completed disagreement authoritatively retains the
 * literal champion-policy baseline, while only an unavailable panel falls
 * through to the ordinary evaluator.
 */
#include "late_resolver.h"
#include "agent.h"
#include "search.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LR_MAX_DECK 3
#define LR_MAX_UNSEEN (HAND_SIZE + LR_MAX_DECK)
#define LR_MAX_SUPPORT 990
#define LR_MAX_ACTIONS 6
#define LR_MAX_CORES 3
#define LR_TABLE_SIZE 65536
#define LR_POLICY_SWEEPS 2

typedef struct {
    uint8_t suit;
    uint8_t rank;       /* zero is the semantic wager rank; otherwise 2..10 */
    uint8_t discard;
    uint8_t draw;
} LRAction;

typedef struct {
    uint64_t key;
    LRAction action[LR_MAX_ACTIONS];
    float prior[LR_MAX_ACTIONS];
    double total[LR_MAX_ACTIONS];
    uint32_t visits[LR_MAX_ACTIONS];
    uint8_t used;
    uint8_t nact;
    uint8_t chosen;
    uint8_t actor;
} LRNode;

typedef struct {
    State view;
    uint8_t unseen[LR_MAX_UNSEEN];
    int nunseen;
    int ndeck;
    int support;
} LRAssignmentPlan;

typedef struct {
    const Net *net;
    const NetEvalPlan *eval_plan;
    int objective;
    int cores;
    int max_actions;
    int symmetries;
    int horizon;
    int root_player;
    int failed;
    LRNode *table;
    uint64_t nnodes;
    uint64_t root_nodes;
    uint64_t frozen_opponent_nodes;
    uint64_t transitions;
    uint64_t deviation_evals;
    uint64_t exact_leaves;
} LRContext;

static uint64_t lr_mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x *= UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static void lr_hash_word(uint64_t *h, uint64_t x)
{
    *h = lr_mix64(*h ^ (x + UINT64_C(0x9E3779B97F4A7C15)));
}

static void lr_hash_mask(uint64_t *h, uint64_t mask)
{
    for (int s = 0; s < NSUIT; s++) {
        uint64_t wagers = UINT64_C(7) << (s * NRANK);
        lr_hash_word(h, (uint64_t)__builtin_popcountll(mask & wagers));
        for (int r = WAGERS_PER_SUIT; r < NRANK; r++)
            lr_hash_word(h, (mask >> CARD_MAKE(s, r)) & UINT64_C(1));
    }
}

/* Hash exactly what p can observe.  The public path is mixed separately in
 * lr_node_key so histories which reach the same board are not merged. */
static uint64_t lr_information_hash(const State *st, int p)
{
    State view;
    agent_information_view(st, p, &view);
    uint64_t h = UINT64_C(0x6A09E667F3BCC909);
    lr_hash_word(&h, view.turn);
    lr_hash_word(&h, view.deck_left);
    lr_hash_word(&h, view.nply);
    lr_hash_word(&h, view.round);
    lr_hash_word(&h, (uint16_t)view.cum[0]);
    lr_hash_word(&h, (uint16_t)view.cum[1]);
    lr_hash_word(&h, view.hand_n[0]);
    lr_hash_word(&h, view.hand_n[1]);
    lr_hash_mask(&h, view.hand[p]);
    lr_hash_mask(&h, view.known[0]);
    lr_hash_mask(&h, view.known[1]);
    lr_hash_mask(&h, view.played[0]);
    lr_hash_mask(&h, view.played[1]);
    lr_hash_mask(&h, view.discarded);
    for (int q = 0; q < 2; q++)
        for (int s = 0; s < NSUIT; s++) {
            lr_hash_word(&h, view.exp_wager[q][s]);
            lr_hash_word(&h, view.exp_top[q][s]);
            lr_hash_word(&h, view.exp_n[q][s]);
            lr_hash_word(&h, view.exp_sum[q][s]);
        }
    for (int s = 0; s < NSUIT; s++) {
        lr_hash_word(&h, view.pile_n[s]);
        for (int i = 0; i < view.pile_n[s]; i++) {
            int c = view.pile[s][i];
            int rank = CARD_IS_WAGER(c) ? 0 : CARD_VALUE(c);
            lr_hash_word(&h, (uint64_t)(s * 16 + rank));
        }
    }
    return h;
}

static uint64_t lr_node_key(const State *st, uint64_t path, int stall_left)
{
    uint64_t h = lr_information_hash(st, st->turn);
    lr_hash_word(&h, path);
    lr_hash_word(&h, (uint64_t)(stall_left + 1));
    return h ? h : UINT64_C(1);
}

static int lr_falling_factorial(int n, int k)
{
    int out = 1;
    for (int i = 0; i < k; i++) out *= n - i;
    return out;
}

static int lr_assignment_plan_init(const State *st, int p,
                                   LRAssignmentPlan *plan)
{
    if (!st || !plan || p < 0 || p > 1 || st->over ||
        st->deck_left < 2 || st->deck_left > LR_MAX_DECK)
        return 0;
    memset(plan, 0, sizeof *plan);
    agent_information_view(st, p, &plan->view);
    lc_unseen(&plan->view, p, plan->unseen, &plan->nunseen);
    int opponent_need = (int)plan->view.hand_n[p ^ 1] -
                        __builtin_popcountll(plan->view.known[p ^ 1]);
    plan->ndeck = plan->view.deck_left;
    if (opponent_need < 0 || plan->nunseen != opponent_need + plan->ndeck ||
        plan->nunseen < plan->ndeck || plan->nunseen > LR_MAX_UNSEEN)
        return 0;
    plan->support = lr_falling_factorial(plan->nunseen, plan->ndeck);
    return plan->support > 0 && plan->support <= LR_MAX_SUPPORT;
}

int late_resolver_assignment_count(const State *st, int p)
{
    LRAssignmentPlan plan;
    return lr_assignment_plan_init(st, p, &plan) ? plan.support : 0;
}

static void lr_assignment_indices(int rank, int n, int k,
                                  int chosen[LR_MAX_DECK])
{
    int available[LR_MAX_UNSEEN];
    for (int i = 0; i < n; i++) available[i] = i;
    int navailable = n;
    for (int d = 0; d < k; d++) {
        int block = lr_falling_factorial(navailable - 1, k - d - 1);
        int pick = block > 0 ? rank / block : 0;
        if (block > 0) rank %= block;
        chosen[d] = available[pick];
        for (int i = pick + 1; i < navailable; i++)
            available[i - 1] = available[i];
        navailable--;
    }
}

static void lr_assignment_world(const LRAssignmentPlan *plan, int p,
                                int rank, State *world)
{
    int chosen[LR_MAX_DECK] = { 0 };
    lr_assignment_indices(rank, plan->nunseen, plan->ndeck, chosen);
    *world = plan->view;
    world->hand[p ^ 1] = plan->view.known[p ^ 1];
    for (int i = 0; i < plan->nunseen; i++) {
        int in_deck = 0;
        for (int d = 0; d < plan->ndeck; d++)
            if (chosen[d] == i) in_deck = 1;
        if (!in_deck)
            world->hand[p ^ 1] |= UINT64_C(1) << plan->unseen[i];
    }
    memset(world->deck, 0, sizeof world->deck);
    world->deck_pos = 0;
    for (int d = 0; d < plan->ndeck; d++)
        world->deck[d] = plan->unseen[chosen[d]];
    world->deck_left = (uint8_t)plan->ndeck;
}

static LRAction lr_action_from_move(Move m)
{
    LRAction a;
    a.suit = (uint8_t)CARD_SUIT(m.card);
    a.rank = (uint8_t)(CARD_IS_WAGER(m.card) ? 0 : CARD_VALUE(m.card));
    a.discard = m.discard;
    a.draw = m.draw;
    return a;
}

static int lr_action_equal(LRAction a, LRAction b)
{
    return a.suit == b.suit && a.rank == b.rank &&
           a.discard == b.discard && a.draw == b.draw;
}

static int lr_core_equal(LRAction a, LRAction b)
{
    return a.suit == b.suit && a.rank == b.rank &&
           a.discard == b.discard;
}

static uint16_t lr_action_pack(LRAction a)
{
    return (uint16_t)(a.suit * 11u + a.rank + 64u * a.discard +
                      128u * a.draw);
}

static uint64_t lr_path_append(uint64_t path, LRAction a)
{
    lr_hash_word(&path, lr_action_pack(a));
    return path;
}

static int lr_resolve_action(const State *st, LRAction action, Move *out)
{
    Move mv[MAX_MOVES];
    int n = lc_moves(st, mv);
    for (int i = 0; i < n; i++) {
        LRAction candidate = lr_action_from_move(mv[i]);
        if (lr_action_equal(candidate, action)) {
            *out = mv[i];
            return 1;
        }
    }
    return 0;
}

typedef struct {
    LRAction core;
    double mass;
    int deck;
    int pile;
} LRCore;

static int lr_build_actions(LRContext *ctx, const State *st, int deck_only,
                            LRAction out[LR_MAX_ACTIONS],
                            float prior[LR_MAX_ACTIONS])
{
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    int n = policy_probs_sym_plan(
        ctx->net, st, mv, prob, NULL, ctx->symmetries, ctx->eval_plan);
    if (n <= 0) return 0;
    int global_baseline = -1;
    if (!deck_only) {
        for (int i = 0; i < n; i++)
            /* Match the deployed zero-temperature policy exactly: its stable
             * tie behavior is the first legal move, not the lowest packed
             * representation.  Candidate zero is the baseline the resolver
             * may retain, so even an exact-probability tie must not drift. */
            if (global_baseline < 0 || prob[i] > prob[global_baseline])
                global_baseline = i;
    }
    LRCore core[2 * HAND_SIZE];
    int ncore = 0;
    for (int i = 0; i < n; i++) {
        LRAction action = lr_action_from_move(mv[i]);
        int k = -1;
        for (int j = 0; j < ncore; j++)
            if (lr_core_equal(action, core[j].core)) {
                k = j;
                break;
            }
        if (k < 0) {
            if (ncore >= 2 * HAND_SIZE) return 0;
            k = ncore++;
            core[k].core = action;
            core[k].mass = 0.0;
            core[k].deck = -1;
            core[k].pile = -1;
        }
        double pi = lc_float_isfinite(prob[i]) && prob[i] > 0.0f
            ? prob[i] : 0.0;
        core[k].mass += pi;
        if (mv[i].draw == 0) {
            if (core[k].deck < 0 || prob[i] > prob[core[k].deck])
                core[k].deck = i;
        } else if (core[k].pile < 0 || prob[i] > prob[core[k].pile]) {
            core[k].pile = i;
        }
    }
    for (int i = 0; i < ncore; i++) {
        int best = i;
        for (int j = i + 1; j < ncore; j++) {
            double sj = deck_only
                ? (core[j].deck >= 0 ? prob[core[j].deck] : 0.0)
                : core[j].mass;
            double sb = deck_only
                ? (core[best].deck >= 0 ? prob[core[best].deck] : 0.0)
                : core[best].mass;
            if (sj > sb)
                best = j;
        }
        LRCore tmp = core[i]; core[i] = core[best]; core[best] = tmp;
    }
    int keep = ctx->cores < ncore ? ctx->cores : ncore;
    /* Aggregate semantic-core mass is useful for breadth, but it must never
     * evict the literal complete-move policy argmax.  Candidate zero is the
     * deployed policy baseline used by the practical-improvement gate. */
    if (!deck_only && global_baseline >= 0 && keep > 0) {
        LRAction baseline_action = lr_action_from_move(mv[global_baseline]);
        int baseline_core = -1;
        for (int k = 0; k < ncore; k++)
            if (lr_core_equal(baseline_action, core[k].core)) {
                baseline_core = k;
                break;
            }
        if (baseline_core < 0) return 0;
        if (baseline_core >= keep) {
            LRCore tmp = core[keep - 1];
            core[keep - 1] = core[baseline_core];
            core[baseline_core] = tmp;
        }
    }
    int limit = ctx->max_actions;
    if (limit < 1) limit = 1;
    if (limit > LR_MAX_ACTIONS) limit = LR_MAX_ACTIONS;
    int count = 0;
    /* Candidate zero must be the actual complete-move policy baseline, not a
     * deck-only surrogate created by reserving slots for every core first. */
    if (!deck_only) {
        if (global_baseline >= 0) {
            out[count] = lr_action_from_move(mv[global_baseline]);
            prior[count] = prob[global_baseline];
            count++;
        }
    }
    /* Retain as many selected cores' guaranteed progress actions as the hard
     * caller budget permits. */
    for (int k = 0; k < keep && count < limit; k++) {
        int i = core[k].deck;
        if (i < 0) continue;
        LRAction action = lr_action_from_move(mv[i]);
        int duplicate = 0;
        for (int q = 0; q < count; q++)
            if (lr_action_equal(out[q], action)) duplicate = 1;
        if (duplicate) continue;
        out[count] = action;
        prior[count] = prob[i];
        count++;
    }
    /* Allocate the remaining slots globally by complete-move prior.  A weak
     * pile pickup is not admitted merely because its core ranked third, and a
     * genuinely plausible core may keep more than one distinct public stall. */
    if (!deck_only && count < limit) {
        int pile_order[MAX_MOVES], npile = 0;
        for (int i = 0; i < n; i++) {
            if (mv[i].draw == 0) continue;
            LRAction action = lr_action_from_move(mv[i]);
            int retained = 0;
            for (int k = 0; k < keep; k++)
                if (lr_core_equal(action, core[k].core)) retained = 1;
            if (retained) pile_order[npile++] = i;
        }
        for (int i = 0; i < npile; i++) {
            int best = i;
            for (int j = i + 1; j < npile; j++)
                if (prob[pile_order[j]] > prob[pile_order[best]] ||
                    (prob[pile_order[j]] == prob[pile_order[best]] &&
                     MOVE_PACK(mv[pile_order[j]]) <
                     MOVE_PACK(mv[pile_order[best]])))
                    best = j;
            int tmp = pile_order[i];
            pile_order[i] = pile_order[best];
            pile_order[best] = tmp;
        }
        for (int k = 0; k < npile && count < limit; k++) {
            int i = pile_order[k];
            LRAction action = lr_action_from_move(mv[i]);
            int duplicate = 0;
            for (int q = 0; q < count; q++)
                if (lr_action_equal(out[q], action)) duplicate = 1;
            if (duplicate) continue;
            out[count] = action;
            prior[count] = prob[i];
            count++;
        }
    }
    /* Candidate zero is already the deployed complete-move baseline.  Keep it
     * fixed even when another retained move has exactly equal probability;
     * only forced-progress lists (which have no deployed root baseline) and
     * the remaining diagnostic challengers are sorted here. */
    int sort_from = !deck_only && count > 0 ? 1 : 0;
    for (int i = sort_from; i < count; i++) {
        int best = i;
        for (int j = i + 1; j < count; j++)
            if (prior[j] > prior[best])
                best = j;
        LRAction action_tmp = out[i]; out[i] = out[best]; out[best] = action_tmp;
        float prior_tmp = prior[i]; prior[i] = prior[best]; prior[best] = prior_tmp;
    }
    return count;
}

int late_resolver_policy_candidates(const Net *net, const State *st,
                                    int cores, int policy_symmetries,
                                    int max_actions, int deck_only,
                                    Move *out, float *prior)
{
    if (!net || !st || !out || !prior || cores < 1 ||
        cores > LR_MAX_CORES || max_actions < 1)
        return 0;
    LRContext ctx;
    memset(&ctx, 0, sizeof ctx);
    NetEvalPlan eval_plan;
    net_eval_plan_init(net, &eval_plan);
    ctx.net = net;
    ctx.eval_plan = &eval_plan;
    ctx.cores = cores;
    ctx.max_actions = max_actions > LR_MAX_ACTIONS
        ? LR_MAX_ACTIONS : max_actions;
    ctx.symmetries = policy_symmetries;
    LRAction action[LR_MAX_ACTIONS];
    int n = lr_build_actions(&ctx, st, deck_only != 0, action, prior);
    for (int i = 0; i < n; i++)
        if (!lr_resolve_action(st, action[i], &out[i])) return 0;
    return n;
}

static LRNode *lr_get_node(LRContext *ctx, const State *st, uint64_t path,
                           int stall_left)
{
    uint64_t key = lr_node_key(st, path, stall_left);
    uint64_t slot = key % LR_TABLE_SIZE;
    for (uint64_t step = 0; step < LR_TABLE_SIZE; step++) {
        LRNode *node = &ctx->table[(slot + step) % LR_TABLE_SIZE];
        if (node->used && node->key == key) {
            if (node->actor != st->turn ||
                (node->actor != ctx->root_player &&
                 node->actor != (ctx->root_player ^ 1))) {
                ctx->failed = 1;
                return NULL;
            }
            return node;
        }
        if (node->used) continue;
        LRAction action[LR_MAX_ACTIONS];
        float prior[LR_MAX_ACTIONS];
        int n = lr_build_actions(ctx, st, stall_left <= 0, action, prior);
        if (n <= 0) {
            ctx->failed = 1;
            return NULL;
        }
        memset(node, 0, sizeof *node);
        node->used = 1;
        node->key = key;
        node->nact = (uint8_t)n;
        node->actor = st->turn;
        node->chosen = 0;
        for (int i = 0; i < n; i++) {
            node->action[i] = action[i];
            node->prior[i] = prior[i];
            /* Match deployed policy semantics: exact ties retain the first
             * legal/ranked candidate instead of inventing a packed-move
             * preference that the champion never uses. */
            if (prior[i] > prior[node->chosen])
                node->chosen = (uint8_t)i;
        }
        ctx->nnodes++;
        if (node->actor == ctx->root_player)
            ctx->root_nodes++;
        else
            ctx->frozen_opponent_nodes++;
        return node;
    }
    ctx->failed = 1;
    return NULL;
}

static int lr_apply(LRContext *ctx, State *st, LRAction action,
                    uint64_t *path, int *stall_left)
{
    Move move;
    if (!lr_resolve_action(st, action, &move)) {
        ctx->failed = 1;
        return 0;
    }
    int pile = move.draw != 0;
    lc_apply(st, move);
    ctx->transitions++;
    *path = lr_path_append(*path, action);
    *stall_left = pile ? *stall_left - 1 : ctx->horizon;
    if (*stall_left < 0 || (st->over && st->deck_left > 0)) {
        ctx->failed = 1;
        return 0;
    }
    return 1;
}

static int lr_apply_exact_leaf(LRContext *ctx, State *st, uint64_t *path,
                               int *stall_left)
{
    Move mv[MAX_MOVES];
    int n = lc_moves(st, mv);
    int selected = rollout_exact_terminal_choice(
        st, mv, NULL, n, ctx->objective, NULL);
    if (selected < 0) {
        ctx->failed = 1;
        return 0;
    }
    LRAction action = lr_action_from_move(mv[selected]);
    if (!lr_apply(ctx, st, action, path, stall_left)) return 0;
    ctx->exact_leaves++;
    return st->over && st->deck_left == 0;
}

static int lr_finish_policy(LRContext *ctx, State *st, uint64_t path,
                            int stall_left, int perspective, double *value)
{
    int max_steps = (int)st->deck_left * (ctx->horizon + 1) + 1;
    for (int step = 0; step < max_steps && !st->over; step++) {
        if (st->deck_left == 1) {
            if (!lr_apply_exact_leaf(ctx, st, &path, &stall_left)) return 0;
            break;
        }
        LRNode *node = lr_get_node(ctx, st, path, stall_left);
        if (!node || node->chosen >= node->nact) return 0;
        if (!lr_apply(ctx, st, node->action[node->chosen],
                      &path, &stall_left))
            return 0;
    }
    if (!st->over || st->deck_left != 0) {
        ctx->failed = 1;
        return 0;
    }
    *value = rollout_terminal_objective(st, perspective, ctx->objective);
    return 1;
}

static void lr_reset_accumulators(LRContext *ctx)
{
    for (int i = 0; i < LR_TABLE_SIZE; i++) {
        LRNode *node = &ctx->table[i];
        if (!node->used) continue;
        memset(node->total, 0, sizeof node->total);
        memset(node->visits, 0, sizeof node->visits);
    }
}

/* Visit the frozen policy path and evaluate every one-step deviation against
 * that same frozen observation-keyed continuation. */
static int lr_improve_trajectory(LRContext *ctx, State *st, uint64_t path,
                                 int stall_left)
{
    int max_steps = (int)st->deck_left * (ctx->horizon + 1) + 1;
    for (int step = 0; step < max_steps && !st->over; step++) {
        if (st->deck_left == 1)
            return lr_apply_exact_leaf(ctx, st, &path, &stall_left);
        LRNode *node = lr_get_node(ctx, st, path, stall_left);
        if (!node || node->chosen >= node->nact) return 0;
        /* The root support is conditioned on the root mover's actual hand.
         * Improving an opponent policy on that one-sided range would teach it
         * the private hand despite an observation-safe node key.  Opponent
         * nonterminal nodes therefore remain the champion prior; only the
         * root mover's later information sets receive policy improvement. */
        if (node->actor != ctx->root_player) {
            LRAction selected = node->action[node->chosen];
            if (!lr_apply(ctx, st, selected, &path, &stall_left)) return 0;
            continue;
        }
        const int actor = st->turn;
        for (int a = 0; a < node->nact; a++) {
            State child = *st;
            uint64_t child_path = path;
            int child_stall = stall_left;
            if (!lr_apply(ctx, &child, node->action[a],
                          &child_path, &child_stall))
                return 0;
            double q;
            if (!lr_finish_policy(ctx, &child, child_path, child_stall,
                                  actor, &q))
                return 0;
            node->total[a] += q;
            node->visits[a]++;
            ctx->deviation_evals++;
        }
        LRAction selected = node->action[node->chosen];
        if (!lr_apply(ctx, st, selected, &path, &stall_left)) return 0;
    }
    if (!st->over || st->deck_left != 0) {
        ctx->failed = 1;
        return 0;
    }
    return 1;
}

static void lr_commit_improvement(LRContext *ctx)
{
    for (int i = 0; i < LR_TABLE_SIZE; i++) {
        LRNode *node = &ctx->table[i];
        if (!node->used || node->nact == 0 ||
            node->actor != ctx->root_player)
            continue;
        int best = node->chosen;
        int have = 0;
        for (int a = 0; a < node->nact; a++) {
            if (node->visits[a] == 0) continue;
            if (!have) {
                best = a;
                have = 1;
                continue;
            }
            double aq = node->total[a] / node->visits[a];
            double bq = node->total[best] / node->visits[best];
            if (aq > bq ||
                (aq == bq && node->prior[a] > node->prior[best]) ||
                (aq == bq && node->prior[a] == node->prior[best] &&
                 lr_action_pack(node->action[a]) <
                 lr_action_pack(node->action[best])))
                best = a;
        }
        if (have) node->chosen = (uint8_t)best;
    }
}

static uint64_t lr_root_path(void)
{
    /* The subgame begins at one public history.  Never seed this path with the
     * root mover's private observation: doing so would let the opponent's
     * information-node identity depend on a hand it cannot see. */
    return UINT64_C(0xBB67AE8584CAA73B);
}

static int lr_solve_horizon(const Net *net, const NetEvalPlan *eval_plan,
                            const State *root,
                            const LRAssignmentPlan *plan,
                            const LRAction *root_action,
                            const float *root_prior, int nroot,
                            int objective, int cores, int symmetries,
                            int max_actions, int horizon, LRAction *selected,
                            double *selected_value,
                            double action_value[LR_MAX_ACTIONS],
                            int *best_index, LRContext *summary)
{
    LRContext ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.net = net;
    ctx.eval_plan = eval_plan;
    ctx.objective = objective;
    ctx.cores = cores;
    ctx.max_actions = max_actions;
    ctx.symmetries = symmetries;
    ctx.horizon = horizon;
    ctx.root_player = root->turn;
    ctx.table = calloc(LR_TABLE_SIZE, sizeof *ctx.table);
    if (!ctx.table) return 0;
    uint64_t root_path = lr_root_path();

    /* Sweep every root candidate over every ordered assignment.  This costs
     * more than assigning one candidate per particle, but removes candidate
     * order and hidden-assignment order from the learned continuation. */
    for (int sweep = 0; sweep < LR_POLICY_SWEEPS && !ctx.failed; sweep++) {
        lr_reset_accumulators(&ctx);
        for (int r = 0; r < nroot && !ctx.failed; r++) {
            for (int rank = 0; rank < plan->support && !ctx.failed; rank++) {
                State world;
                lr_assignment_world(plan, root->turn, rank, &world);
                uint64_t path = root_path;
                int stall_left = horizon;
                if (!lr_apply(&ctx, &world, root_action[r],
                              &path, &stall_left) ||
                    !lr_improve_trajectory(
                        &ctx, &world, path, stall_left))
                    ctx.failed = 1;
            }
        }
        if (!ctx.failed) lr_commit_improvement(&ctx);
    }

    double total[LR_MAX_ACTIONS] = { 0 };
    if (!ctx.failed) {
        for (int r = 0; r < nroot && !ctx.failed; r++) {
            for (int rank = 0; rank < plan->support && !ctx.failed; rank++) {
                State world;
                lr_assignment_world(plan, root->turn, rank, &world);
                uint64_t path = root_path;
                int stall_left = horizon;
                if (!lr_apply(&ctx, &world, root_action[r],
                              &path, &stall_left)) {
                    ctx.failed = 1;
                    break;
                }
                double q;
                if (!lr_finish_policy(&ctx, &world, path, stall_left,
                                      root->turn, &q)) {
                    ctx.failed = 1;
                    break;
                }
                total[r] += q;
            }
        }
    }

    int best = 0;
    if (!ctx.failed) {
        for (int r = 0; r < nroot; r++)
            action_value[r] = total[r] / plan->support;
        for (int r = 1; r < nroot; r++) {
            if (total[r] > total[best] ||
                (total[r] == total[best] &&
                 root_prior[r] > root_prior[best]) ||
                (total[r] == total[best] &&
                 root_prior[r] == root_prior[best] &&
                 lr_action_pack(root_action[r]) <
                 lr_action_pack(root_action[best])))
                best = r;
        }
        *selected = root_action[best];
        *selected_value = action_value[best];
        if (best_index) *best_index = best;
    }
    if (summary) {
        *summary = ctx;
        summary->table = NULL;
        summary->eval_plan = NULL;
    }
    free(ctx.table);
    return !ctx.failed;
}

static const NetEvalPlan *lr_safe_eval_plan(
    const Net *net, const NetEvalPlan *eval_plan)
{
    if (!net || !eval_plan || eval_plan->owner != net ||
        (eval_plan->dense_count != FEAT_LEGACY_DENSE &&
         eval_plan->dense_count != FEAT_DENSE) ||
        eval_plan->zero_combination > 1)
        return NULL;
    return eval_plan;
}

int late_resolver_choose_dual_plan(
    const Net *root_net, const Net *continuation_net,
    const State *st, int objective, int cores, int policy_symmetries,
    int max_actions, double practical_min, Move *out,
    LateResolverStats *stats, const NetEvalPlan *root_eval_plan,
    const NetEvalPlan *continuation_eval_plan)
{
    if (stats) {
        memset(stats, 0, sizeof *stats);
        stats->horizon2_best = -1;
        stats->horizon4_best = -1;
        stats->unavailable = 1;
    }
    if (!root_net || !st || !out || st->over || st->deck_left < 2 ||
        st->deck_left > LR_MAX_DECK || cores != LR_MAX_CORES ||
        max_actions < 1)
        return 0;
    if (!continuation_net) continuation_net = root_net;
    if (max_actions > LR_MAX_ACTIONS) max_actions = LR_MAX_ACTIONS;
    if (!(practical_min >= 0.0) || !lc_double_isfinite(practical_min))
        return 0;
    LRAssignmentPlan plan;
    if (!lr_assignment_plan_init(st, st->turn, &plan)) return 0;
    if (stats) stats->support = plan.support;

    LRContext root_ctx;
    memset(&root_ctx, 0, sizeof root_ctx);
    /* Do not let a stale/foreign or malformed proof select a shortcut.  A
     * null plan deliberately takes the complete feature/head path without
     * rescanning the checkpoint; the owner is responsible for constructing a
     * fresh plan after every model mutation. */
    const NetEvalPlan *safe_root_eval_plan =
        lr_safe_eval_plan(root_net, root_eval_plan);
    const NetEvalPlan *safe_continuation_eval_plan =
        lr_safe_eval_plan(continuation_net, continuation_eval_plan);
    root_ctx.net = root_net;
    root_ctx.eval_plan = safe_root_eval_plan;
    root_ctx.cores = cores;
    root_ctx.max_actions = max_actions;
    /* Rank the root with the configured deployed ensemble.  Exact-20 at every
     * counterfactual node is prohibitively expensive, so descendants use five
     * deterministic rotations (or one for explicitly symmetry-free runs). */
    root_ctx.symmetries = policy_symmetries;
    LRAction root_action[LR_MAX_ACTIONS];
    float root_prior[LR_MAX_ACTIONS];
    int nroot = lr_build_actions(
        &root_ctx, st, 0, root_action, root_prior);
    if (nroot <= 0 || nroot > LR_MAX_ACTIONS) return 0;
    if (stats) stats->root_candidates = nroot;

    LRAction h2, h4;
    double v2 = 0.0, v4 = 0.0;
    double q2[LR_MAX_ACTIONS] = { 0 }, q4[LR_MAX_ACTIONS] = { 0 };
    int best2 = -1, best4 = -1;
    LRContext s2, s4;
    memset(&s2, 0, sizeof s2);
    memset(&s4, 0, sizeof s4);
    int ok2 = lr_solve_horizon(
        continuation_net, safe_continuation_eval_plan,
        st, &plan, root_action, root_prior, nroot,
        objective, cores, policy_symmetries > 1 ? 5 : 1, max_actions,
        2, &h2, &v2,
        q2, &best2, &s2);
    int ok4 = ok2 && lr_solve_horizon(
        continuation_net, safe_continuation_eval_plan,
        st, &plan, root_action, root_prior, nroot,
        objective, cores, policy_symmetries > 1 ? 5 : 1, max_actions,
        4, &h4, &v4,
        q4, &best4, &s4);
    int stable = ok2 && ok4 && lr_action_equal(h2, h4);
    double delta2 = best2 >= 0 ? q2[best2] - q2[0] : 0.0;
    double delta4 = best4 >= 0 ? q4[best4] - q4[0] : 0.0;
    int passed = stable && best2 == best4 &&
        (best4 == 0 || (delta2 > practical_min && delta4 > practical_min));

    if (stats) {
        stats->support = plan.support;
        stats->root_candidates = nroot;
        stats->horizon2_best = best2;
        stats->horizon4_best = best4;
        stats->stable = stable;
        stats->passed = passed;
        stats->unavailable = !(ok2 && ok4);
        stats->horizon2_delta = delta2;
        stats->horizon4_delta = delta4;
        for (int i = 0; i < nroot; i++) {
            (void)lr_resolve_action(st, root_action[i], &stats->candidate[i]);
            stats->prior[i] = root_prior[i];
            stats->horizon2_q[i] = q2[i];
            stats->horizon4_q[i] = q4[i];
        }
        if (ok2) {
            (void)lr_resolve_action(st, h2, &stats->horizon2_move);
            stats->horizon2_value = v2;
            stats->horizon2_nodes = s2.nnodes;
            stats->horizon2_root_nodes = s2.root_nodes;
            stats->horizon2_frozen_opponent_nodes =
                s2.frozen_opponent_nodes;
            stats->horizon2_transitions = s2.transitions;
            stats->horizon2_deviation_evals = s2.deviation_evals;
            stats->horizon2_exact_leaves = s2.exact_leaves;
        }
        if (ok4) {
            (void)lr_resolve_action(st, h4, &stats->horizon4_move);
            stats->horizon4_value = v4;
            stats->horizon4_nodes = s4.nnodes;
            stats->horizon4_root_nodes = s4.root_nodes;
            stats->horizon4_frozen_opponent_nodes =
                s4.frozen_opponent_nodes;
            stats->horizon4_transitions = s4.transitions;
            stats->horizon4_deviation_evals = s4.deviation_evals;
            stats->horizon4_exact_leaves = s4.exact_leaves;
        }
    }
    if (!passed || !lr_resolve_action(st, h4, out)) return 0;
    return 1;
}

int late_resolver_choose_plan(const Net *net, const State *st, int objective,
                              int cores, int policy_symmetries,
                              int max_actions, double practical_min,
                              Move *out, LateResolverStats *stats,
                              const NetEvalPlan *eval_plan)
{
    return late_resolver_choose_dual_plan(
        net, net, st, objective, cores, policy_symmetries, max_actions,
        practical_min, out, stats, eval_plan, eval_plan);
}

int late_resolver_choose(const Net *net, const State *st, int objective,
                         int cores, int policy_symmetries, int max_actions,
                         double practical_min, Move *out,
                         LateResolverStats *stats)
{
    NetEvalPlan eval_plan;
    const NetEvalPlan *use_plan = NULL;
    /* Keep malformed/unavailable calls cheap just as the historical public
     * entry point was: the implementation below owns authoritative argument
     * validation and diagnostics, while a scan is useful only once that
     * coarse gate can possibly pass. */
    if (net && st && out && !st->over && st->deck_left >= 2 &&
        st->deck_left <= LR_MAX_DECK && cores == LR_MAX_CORES &&
        max_actions >= 1 && practical_min >= 0.0 &&
        lc_double_isfinite(practical_min)) {
        net_eval_plan_init(net, &eval_plan);
        use_plan = &eval_plan;
    }
    return late_resolver_choose_plan(
        net, st, objective, cores, policy_symmetries, max_actions,
        practical_min, out, stats, use_plan);
}
