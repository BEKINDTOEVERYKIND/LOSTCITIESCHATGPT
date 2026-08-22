/* White-box regressions for the coherent two-role rollout ensembles.
 *
 * Include rollout.c in this test translation unit so the test can observe the
 * exact fixed-permutation policy calls made by the real playout code without
 * adding a diagnostic hook to the gameplay ABI.  All other modules are linked
 * normally. */
#include "../src/agent.h"
#include "../src/net.h"
#include "../src/search.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ROLE_GROUP = 20,
    ROLE_PRODUCT = ROLE_GROUP * ROLE_GROUP,
    TRACE_MAX_TRAJECTORIES = 2048,
    TRACE_MAX_CALLS = 32
};

typedef struct {
    int perm[2];
    int calls[2];
    int stable;
    int ncall;
    uint8_t player[TRACE_MAX_CALLS];
    uint16_t move[TRACE_MAX_CALLS];
} RoleTrajectory;

/* suit_permutations() has one fixed 120-row output contract even when a
 * subgroup is requested; only the first ROLE_GROUP rows are used here. */
static uint8_t trace_group[120][NSUIT];
static RoleTrajectory trace_trajectory[TRACE_MAX_TRAJECTORIES];
static int trace_enabled;
static int trace_n;
static int trace_last_nply;
static int trace_bad_perm;
static int trace_overflow;

static int traced_policy_probs_perm_plan(
    const Net *net, const State *st, Move *mv, float *prob, float *value,
    const uint8_t perm[NSUIT], const NetEvalPlan *plan);

/* Interpose only calls made from the included rollout implementation. */
#define policy_probs_perm_plan traced_policy_probs_perm_plan
#include "../src/rollout.c"
#undef policy_probs_perm_plan

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) {                         \
    printf("FAIL %s:%d: ", __FILE__, __LINE__);                    \
    printf(__VA_ARGS__); printf("\n"); failures++;                 \
} } while (0)

static int permutation_index(const uint8_t perm[NSUIT])
{
    for (int i = 0; i < ROLE_GROUP; i++)
        if (memcmp(perm, trace_group[i], NSUIT) == 0) return i;
    return -1;
}

static void trace_reset(void)
{
    memset(trace_trajectory, 0, sizeof trace_trajectory);
    trace_enabled = 1;
    trace_n = 0;
    trace_last_nply = -1;
    trace_bad_perm = 0;
    trace_overflow = 0;
}

static void trace_stop(void)
{
    trace_enabled = 0;
}

static void trace_record(const State *st, const uint8_t perm[NSUIT],
                         const Move *mv, const float *prob, int n)
{
    if (!trace_enabled) return;
    if (trace_n == 0 || (int)st->nply <= trace_last_nply) {
        if (trace_n >= TRACE_MAX_TRAJECTORIES) {
            trace_overflow = 1;
            return;
        }
        RoleTrajectory *t = &trace_trajectory[trace_n++];
        t->perm[0] = t->perm[1] = -1;
        t->stable = 1;
    }
    trace_last_nply = st->nply;
    RoleTrajectory *t = &trace_trajectory[trace_n - 1];
    int p = st->turn;
    int k = permutation_index(perm);
    if (k < 0) trace_bad_perm++;
    if (t->perm[p] < 0) t->perm[p] = k;
    else if (t->perm[p] != k) t->stable = 0;
    t->calls[p]++;
    if (t->ncall < TRACE_MAX_CALLS) {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (prob[i] > prob[best]) best = i;
        t->player[t->ncall] = (uint8_t)p;
        t->move[t->ncall] = MOVE_PACK(mv[best]);
        t->ncall++;
    } else {
        trace_overflow = 1;
    }
}

static int traced_policy_probs_perm_plan(
    const Net *net, const State *st, Move *mv, float *prob, float *value,
    const uint8_t perm[NSUIT], const NetEvalPlan *plan)
{
    int n = policy_probs_perm_plan(net, st, mv, prob, value, perm, plan);
    trace_record(st, perm, mv, prob, n);
    return n;
}

static State reachable_late_state(int deck_left)
{
    Rng rng;
    rng_seed(&rng, UINT64_C(0xD00DFEED12345678));
    State st;
    lc_deal(&st, &rng);
    while (!st.over && st.deck_left > deck_left) {
        Move mv[MAX_MOVES];
        int n = lc_moves(&st, mv);
        int chosen = -1;
        for (int i = 0; i < n; i++)
            if (mv[i].draw == 0) {
                chosen = i;
                break;
            }
        if (chosen < 0) break;
        lc_apply(&st, mv[chosen]);
    }
    CHECK(!st.over && st.deck_left == deck_left,
          "could not build reachable deck-%d fixture", deck_left);
    return st;
}

static uint64_t rollout_seed_fork(const Rng *rng)
{
    return rng->s[0] ^ rotl64(rng->s[1], 13) ^
           rotl64(rng->s[2], 29) ^ rotl64(rng->s[3], 47) ^
           UINT64_C(0xA0761D6478BD642F);
}

static void check_role_schedule(const char *label, int root_player,
                                int worlds, int candidates,
                                int fixed_offset, int other_offset)
{
    CHECK(!trace_overflow, "%s trace overflowed", label);
    CHECK(trace_bad_perm == 0, "%s used a permutation outside the 20-group",
          label);
    CHECK(trace_n == worlds * candidates,
          "%s traced %d trajectories, expected %d", label, trace_n,
          worlds * candidates);
    int count[ROLE_GROUP][ROLE_GROUP] = { { 0 } };
    int limit = trace_n < worlds * candidates
              ? trace_n : worlds * candidates;
    for (int t = 0; t < limit; t++) {
        int d = t / candidates;
        int expected_fixed = (fixed_offset + d) % ROLE_GROUP;
        int expected_other =
            (other_offset + d / ROLE_GROUP + d % ROLE_GROUP) % ROLE_GROUP;
        RoleTrajectory *trajectory = &trace_trajectory[t];
        CHECK(trajectory->stable,
              "%s changed a player's mapping within trajectory %d", label, t);
        CHECK(trajectory->calls[root_player] >= 2 &&
              trajectory->calls[root_player ^ 1] >= 2,
              "%s trajectory %d did not exercise both roles repeatedly "
              "(calls=%d/%d)", label, t, trajectory->calls[root_player],
              trajectory->calls[root_player ^ 1]);
        CHECK(trajectory->perm[root_player] == expected_fixed,
              "%s trajectory %d assigned fixed role %d to player %d "
              "(expected %d)", label, t, trajectory->perm[root_player],
              root_player, expected_fixed);
        CHECK(trajectory->perm[root_player ^ 1] == expected_other,
              "%s trajectory %d assigned other role %d to player %d "
              "(expected %d)", label, t,
              trajectory->perm[root_player ^ 1], root_player ^ 1,
              expected_other);
        if (trajectory->perm[root_player] >= 0 &&
            trajectory->perm[root_player] < ROLE_GROUP &&
            trajectory->perm[root_player ^ 1] >= 0 &&
            trajectory->perm[root_player ^ 1] < ROLE_GROUP)
            count[trajectory->perm[root_player]]
                 [trajectory->perm[root_player ^ 1]]++;
    }
    for (int i = 0; i < ROLE_GROUP; i++)
        for (int j = 0; j < ROLE_GROUP; j++)
            CHECK(count[i][j] == candidates,
                  "%s ordered role pair (%d,%d) appeared %d times, expected %d",
                  label, i, j, count[i][j], candidates);
}

static void test_rollout_mode4_product(void)
{
    Net *zero = calloc(1, sizeof *zero);
    CHECK(zero != NULL, "allocate mode-4 zero network");
    if (!zero) return;
    State st = reachable_late_state(6);
    const int p = st.turn;
    Agent a;
    agent_default(&a, AG_ROLLOUT, zero);
    a.dets = ROLE_PRODUCT;
    a.root_width = 2;
    a.min_cand = 2;
    a.cand_floor = 0.0f;
    a.batch_dets = ROLE_PRODUCT;
    a.playout_sample = 4;
    a.playout_symmetries = ROLE_GROUP;
    a.playout_prune = 0;
    a.no_belief = 1;
    a.exact_terminal = 1;

    Rng rng;
    rng_seed(&rng, UINT64_C(202608220401));
    uint64_t fork = rollout_seed_fork(&rng);
    int fixed_offset = (int)(fork % ROLE_GROUP);
    int other_offset = (int)(rotl64(fork, 31) % ROLE_GROUP);
    SearchStats stats_a;
    trace_reset();
    Move move_a = rollout_move(&a, &st, &rng, NULL, &stats_a);
    Rng rng_a = rng;
    trace_stop();
    CHECK(stats_a.worlds == ROLE_PRODUCT && stats_a.n == 2,
          "mode 4 did not execute the locked 400x2 panel (worlds=%d n=%d)",
          stats_a.worlds, stats_a.n);
    check_role_schedule("rollout mode 4", p, ROLE_PRODUCT, 2,
                        fixed_offset, other_offset);

    RoleTrajectory first_trace[TRACE_MAX_TRAJECTORIES];
    int first_n = trace_n;
    memcpy(first_trace, trace_trajectory, sizeof first_trace);
    SearchStats stats_b;
    rng_seed(&rng, UINT64_C(202608220401));
    trace_reset();
    Move move_b = rollout_move(&a, &st, &rng, NULL, &stats_b);
    trace_stop();
    CHECK(MOVE_PACK(move_a) == MOVE_PACK(move_b) &&
          memcmp(&stats_a, &stats_b, sizeof stats_a) == 0 &&
          memcmp(&rng_a, &rng, sizeof rng) == 0 && trace_n == first_n &&
          memcmp(first_trace, trace_trajectory, sizeof first_trace) == 0,
          "mode-4 decision, diagnostics, RNG, or role trajectories are not "
          "deterministic");
    free(zero);
}

typedef struct {
    int selected;
    int worlds;
    int agreement;
    int gate;
    double q[2], se[2], delta[2], dse[2];
    PlayoutWork work;
} PrefixResult;

static PrefixResult run_prefix_panel(const Agent *a,
                                     const NetEvalPlan *plan,
                                     const State *st, uint64_t seed)
{
    Move legal[MAX_MOVES], chosen[2];
    int n = lc_moves(st, legal), keep = 0;
    for (int i = 0; i < n && keep < 2; i++)
        if (legal[i].draw == 0) chosen[keep++] = legal[i];
    CHECK(keep == 2, "prefix fixture lacks two deck-draw candidates");
    int order[2] = { 0, 1 };
    BeliefDist unused_belief;
    memset(&unused_belief, 0, sizeof unused_belief);
    PrefixResult out;
    memset(&out, 0, sizeof out);
    out.selected = confirm_trusted_prefix(
        a, plan, st, st->turn, chosen, order, 2, 1, 0, ROLE_GROUP,
        &unused_belief, 0, seed, ROLE_PRODUCT, &out.worlds,
        out.q, out.se, out.delta, out.dse, &out.agreement, &out.gate,
        &out.work);
    return out;
}

static void test_prefix_mode3_product(void)
{
    Net *zero = calloc(1, sizeof *zero);
    CHECK(zero != NULL, "allocate prefix-mode-3 zero network");
    if (!zero) return;
    NetEvalPlan plan;
    net_eval_plan_init(zero, &plan);
    State st = reachable_late_state(7); /* odd ply: root player is seat 1 */
    const int p = st.turn;
    CHECK(p == 1, "prefix role fixture no longer starts with player 1");
    Agent a;
    agent_default(&a, AG_ROLLOUT, zero);
    a.confirm_dets = ROLE_PRODUCT;
    a.policy_prefix_mode = 3;
    a.playout_symmetries = ROLE_GROUP;
    a.no_belief = 1;
    a.exact_terminal = 1;

    const uint64_t seed = UINT64_C(202608220303);
    Rng perm_rng;
    rng_seed(&perm_rng, seed ^ UINT64_C(0x8CB92BA72F3D8DD7));
    int fixed_offset = (int)rng_below(&perm_rng, ROLE_GROUP);
    int other_offset = (int)rng_below(&perm_rng, ROLE_GROUP);
    trace_reset();
    PrefixResult first = run_prefix_panel(&a, &plan, &st, seed);
    trace_stop();
    CHECK(first.worlds == ROLE_PRODUCT,
          "prefix mode 3 used %d worlds, expected 400", first.worlds);
    check_role_schedule("prefix mode 3", p, ROLE_PRODUCT, 2,
                        fixed_offset, other_offset);
    RoleTrajectory first_trace[TRACE_MAX_TRAJECTORIES];
    int first_n = trace_n;
    memcpy(first_trace, trace_trajectory, sizeof first_trace);

    trace_reset();
    PrefixResult second = run_prefix_panel(&a, &plan, &st, seed);
    trace_stop();
    CHECK(memcmp(&first, &second, sizeof first) == 0 && trace_n == first_n &&
          memcmp(first_trace, trace_trajectory, sizeof first_trace) == 0,
          "prefix-mode-3 result or role trajectories are not deterministic");
    free(zero);
}

static void inverse_permutation(const uint8_t in[NSUIT],
                                uint8_t out[NSUIT])
{
    for (int s = 0; s < NSUIT; s++) out[in[s]] = (uint8_t)s;
}

/* Return left o right: output[s] = left[right[s]]. */
static void compose_permutations(const uint8_t left[NSUIT],
                                 const uint8_t right[NSUIT],
                                 uint8_t out[NSUIT])
{
    for (int s = 0; s < NSUIT; s++) out[s] = left[right[s]];
}

static int run_fixed_role_playout(const Net *net, const NetEvalPlan *plan,
                                  const State *start,
                                  const uint8_t fixed[NSUIT],
                                  const uint8_t other[NSUIT], State *finish,
                                  PlayoutWork *work)
{
    *finish = *start;
    memset(work, 0, sizeof *work);
    return playout(net, plan, finish, start->turn, 0, NULL, fixed, other,
                   start->turn ^ 1, 0, ROLE_GROUP, 0, 0, 0, 0.0f,
                   UINT64_C(202608220404), 0, 3, 0, 0, NULL, NULL, work);
}

static void test_fixed_role_trajectory_equivariance(void)
{
    Net *net = malloc(sizeof *net);
    CHECK(net != NULL, "allocate role-equivariance network");
    if (!net) return;
    CHECK(net_load(net, "data/champion.bin") == 0,
          "load champion for role-equivariance test");
    NetEvalPlan plan;
    net_eval_plan_init(net, &plan);
    State start = reachable_late_state(7);
    const int p = start.turn;
    const uint8_t *fixed = trace_group[3];
    const uint8_t *other = trace_group[14];

    State finish_a;
    PlayoutWork work_a;
    trace_reset();
    int margin_a = run_fixed_role_playout(
        net, &plan, &start, fixed, other, &finish_a, &work_a);
    trace_stop();
    CHECK(trace_n == 1 && trace_trajectory[0].stable &&
          trace_trajectory[0].perm[p] == 3 &&
          trace_trajectory[0].perm[p ^ 1] == 14 &&
          trace_trajectory[0].calls[p] >= 2 &&
          trace_trajectory[0].calls[p ^ 1] >= 2,
          "fixed roles were not retained by actual player for one trajectory");
    RoleTrajectory original_trace = trace_trajectory[0];

    State repeat;
    PlayoutWork repeat_work;
    trace_reset();
    int repeat_margin = run_fixed_role_playout(
        net, &plan, &start, fixed, other, &repeat, &repeat_work);
    trace_stop();
    CHECK(margin_a == repeat_margin &&
          memcmp(&finish_a, &repeat, sizeof finish_a) == 0 &&
          memcmp(&work_a, &repeat_work, sizeof work_a) == 0 &&
          trace_n == 1 &&
          memcmp(&original_trace, &trace_trajectory[0],
                 sizeof original_trace) == 0,
          "fixed-role complete trajectory is not deterministic");

    /* Relabelling the engine by R requires conjugating each network
     * orientation to P o R^-1.  The oriented network state is then identical,
     * so every action in both players' persistent trajectories must map. */
    const uint8_t *relabel = trace_group[7];
    uint8_t inverse[NSUIT], mapped_fixed[NSUIT], mapped_other[NSUIT];
    inverse_permutation(relabel, inverse);
    compose_permutations(fixed, inverse, mapped_fixed);
    compose_permutations(other, inverse, mapped_other);
    int mapped_fixed_index = permutation_index(mapped_fixed);
    int mapped_other_index = permutation_index(mapped_other);
    CHECK(mapped_fixed_index >= 0 && mapped_other_index >= 0,
          "conjugated role mapping left the affine 20-group");
    State relabelled_start;
    lc_permute_suits(&start, &relabelled_start, relabel);
    State relabelled_finish;
    PlayoutWork relabelled_work;
    trace_reset();
    int relabelled_margin = run_fixed_role_playout(
        net, &plan, &relabelled_start, mapped_fixed, mapped_other,
        &relabelled_finish, &relabelled_work);
    trace_stop();
    State expected_finish;
    lc_permute_suits(&finish_a, &expected_finish, relabel);
    CHECK(relabelled_margin == margin_a &&
          memcmp(&expected_finish, &relabelled_finish,
                 sizeof expected_finish) == 0,
          "persistent two-role playout is not suit-equivariant");
    CHECK(trace_n == 1 && trace_trajectory[0].perm[p] == mapped_fixed_index &&
          trace_trajectory[0].perm[p ^ 1] == mapped_other_index &&
          trace_trajectory[0].ncall == original_trace.ncall,
          "relabelled trajectory did not preserve role assignment/call count");
    if (trace_n == 1 &&
        trace_trajectory[0].ncall == original_trace.ncall) {
        for (int i = 0; i < original_trace.ncall; i++) {
            Move original = {
                MOVE_CARD(original_trace.move[i]),
                MOVE_DISC(original_trace.move[i]),
                MOVE_DRAW(original_trace.move[i])
            };
            Move mapped = lc_permute_move(original, relabel);
            CHECK(original_trace.player[i] == trace_trajectory[0].player[i] &&
                  MOVE_PACK(mapped) == trace_trajectory[0].move[i],
                  "relabelled role action %d did not map equivariantly", i);
        }
    }
    free(net);
}

int main(void)
{
    CHECK(suit_permutations(ROLE_GROUP, trace_group) == ROLE_GROUP,
          "could not construct exact 20-way suit group");
    test_rollout_mode4_product();
    test_prefix_mode3_product();
    test_fixed_role_trajectory_equivariance();
    if (failures) {
        printf("%d role-coherence test(s) failed\n", failures);
        return 1;
    }
    printf("role-coherent rollout tests passed\n");
    return 0;
}
