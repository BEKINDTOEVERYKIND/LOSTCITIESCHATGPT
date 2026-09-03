/* Focused regressions for rollout4's direct signed pairwise veto. */
#include "../src/agent.h"
#include "../src/match_value.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); failures++; \
} } while (0)

static int combination_index(Move m)
{
    return (m.card * 2 + m.discard) * NET_NDRAW + m.draw;
}

static State dealt_state(uint64_t seed)
{
    Rng rng;
    rng_seed(&rng, seed);
    State st;
    lc_deal(&st, &rng);
    return st;
}

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

/* A stable, real information state from the reviewed showcase replay. */
static State reviewed_state(const Net *net, int target_ply)
{
    const uint64_t seed = UINT64_C(2214615196);
    Rng deal_rng, actor_rng;
    rng_seed(&deal_rng, seed);
    rng_seed(&actor_rng, mix64(seed ^ UINT64_C(0xA17C0A)));
    State st;
    lc_deal(&st, &deal_rng);
    st.round = 0;
    st.turn = 0;
    Agent actor;
    agent_default(&actor, AG_POLICY, net);
    actor.symmetries = 20;
    for (int ply = 1; ply < target_ply && !st.over; ply++)
        lc_apply(&st, agent_move(&actor, &st, &actor_rng));
    return st;
}

static void init_play_continuation(Net *net)
{
    net_zero(net);
    for (int c = 0; c < NCARD; c++) {
        net->bplay[c * 2] = 16.0f;
        net->bplay[c * 2 + 1] = -16.0f;
    }
    net->bdraw[0] = 8.0f;
    for (int d = 1; d < NET_NDRAW; d++) net->bdraw[d] = -8.0f;
}

/* A monotone, controller-bound table whose early-round utility is exactly
 * 0.05 times the carried lead.  Comparing it with objective zero proves that
 * the integrated runtime reports table units rather than round points. */
static MatchValueTable *linear_match_value_table(const Net *continuation)
{
    MatchValueTable *table = calloc(1, sizeof *table);
    CHECK(table != NULL, "match-value fixture allocation failed");
    if (!table) return NULL;
    table->version = MATCH_VALUE_VERSION;
    table->samples_per_policy_lead = 25;
    table->role_cycle_size = 25;
    table->role_balance_complete = 1;
    table->isotonic_projected = 1;
    table->source_seed = UINT64_C(0x52414e4b45524d56);
    table->controller = (MatchValueController){
        .net_fingerprint = match_value_net_fingerprint(continuation),
        .controller_abi = MATCH_VALUE_CONTROLLER_ABI,
        .build_profile = match_value_build_profile(),
        .objective = 0,
        .playout_symmetries = 5,
        .playout_sample = 4,
        .playout_prune = 0,
        .exact_terminal = 1,
        .max_plies = LC_MAX_PLIES
    };
    for (int lead = -MATCH_VALUE_R1_LEAD_LIMIT;
         lead <= MATCH_VALUE_R1_LEAD_LIMIT; lead++) {
        int i = lead + MATCH_VALUE_R1_LEAD_LIMIT;
        table->before_round1[0][i] = 0.05 * lead;
        table->before_round1[1][i] = 0.05 * lead;
    }
    for (int lead = -MATCH_VALUE_R2_LEAD_LIMIT;
         lead <= MATCH_VALUE_R2_LEAD_LIMIT; lead++) {
        int i = lead + MATCH_VALUE_R2_LEAD_LIMIT;
        table->before_round2[0][i] = 0.05 * lead;
        table->before_round2[1][i] = 0.05 * lead;
    }
    CHECK(match_value_validate(table) && match_value_balanced_roles(table),
          "controller-bound linear match-value fixture is invalid");
    return table;
}

static void first_three_moves(const State *st, Move out[3])
{
    Move legal[MAX_MOVES];
    int n = lc_moves(st, legal);
    CHECK(n >= 3, "test state has only %d legal moves", n);
    for (int i = 0; i < 3; i++) out[i] = legal[i];
}

static void test_raw_pair_score_and_gate(void)
{
    Net *root = (Net *)malloc(sizeof *root);
    Net *ranker = (Net *)malloc(sizeof *ranker);
    CHECK(root && ranker, "network allocation failed");
    if (!root || !ranker) { free(root); free(ranker); return; }
    net_zero(root);
    net_zero(ranker);

    State st = dealt_state(UINT64_C(0x725402798));
    Move pair[3];
    first_three_moves(&st, pair);
    double score = 99.0;
    CHECK(policy_residual_log_odds_sym(
              root, root, &st, pair[0], pair[1], 1, &score) && score == 0.0,
          "root/ranker alias was not exactly neutral (%.9f)", score);

    ranker->bcomb[combination_index(pair[1])] = 2.0f;
    CHECK(policy_residual_log_odds_sym(
              root, ranker, &st, pair[0], pair[1], 1, &score) &&
              fabs(score - 2.0) < 1e-9,
          "positive residual pair score is %.9f, expected 2", score);
    double reverse = 0.0;
    CHECK(policy_residual_log_odds_sym(
              root, ranker, &st, pair[1], pair[0], 1, &reverse) &&
              fabs(score + reverse) < 1e-12,
          "pair score is not antisymmetric: %.9f vs %.9f", score, reverse);

    Agent actor;
    agent_default(&actor, AG_ROLLOUT, root);
    actor.action_ranker_net = ranker;
    actor.symmetries = 1;
    actor.action_ranker_min = 1.5f;
    int valid = 0;
    double gated_score = 0.0;
    CHECK(rollout_action_ranker_veto(
              &actor, &st, pair[0], pair[1], &gated_score, &valid) && valid &&
              fabs(gated_score - 2.0) < 1e-9,
          "positive ranker support did not retain proposal");
    actor.action_ranker_min = 2.5f;
    CHECK(!rollout_action_ranker_veto(
               &actor, &st, pair[0], pair[1], &gated_score, &valid) && valid,
          "below-threshold ranker support did not veto proposal");

    /* The helper evaluates exactly two logits; an unrelated legal action is
     * absent from the normalization and cannot change their signed score. */
    double before = gated_score;
    ranker->bcomb[combination_index(pair[2])] = 1.0e20f;
    CHECK(rollout_action_ranker_veto(
              &actor, &st, pair[0], pair[1], &gated_score, &valid) == 0 &&
              valid && gated_score == before,
          "third-move logit changed the direct pair score");

    ranker->bcomb[combination_index(pair[1])] = NAN;
    CHECK(!rollout_action_ranker_veto(
               &actor, &st, pair[0], pair[1], &gated_score, &valid) && !valid,
          "non-finite ranker score did not fail closed");
    free(root);
    free(ranker);
}

static void test_suit_equivariance(void)
{
    Net *root = (Net *)malloc(sizeof *root);
    Net *ranker = (Net *)malloc(sizeof *ranker);
    CHECK(root && ranker, "network allocation failed");
    if (!root || !ranker) { free(root); free(ranker); return; }
    net_init(root, UINT64_C(101));
    net_init(ranker, UINT64_C(202));
    State st = dealt_state(UINT64_C(303));
    Move pair[3];
    first_three_moves(&st, pair);
    double original = 0.0;
    CHECK(policy_residual_log_odds_sym(
              root, ranker, &st, pair[0], pair[1], 120, &original),
          "full-group ranker score failed");

    const uint8_t perm[NSUIT] = { 2, 4, 1, 0, 3 };
    State ps;
    lc_permute_suits(&st, &ps, perm);
    Move pbase = lc_permute_move(pair[0], perm);
    Move pproposal = lc_permute_move(pair[1], perm);
    double transformed = 0.0;
    CHECK(policy_residual_log_odds_sym(
              root, ranker, &ps, pbase, pproposal, 120, &transformed) &&
              fabs(original - transformed) < 1e-7,
          "120-way score changed under suit relabelling: %.9f vs %.9f",
          original, transformed);
    free(root);
    free(ranker);
}

static void test_hidden_information_sanitized(void)
{
    Net *root = (Net *)malloc(sizeof *root);
    Net *ranker = (Net *)malloc(sizeof *ranker);
    CHECK(root && ranker, "network allocation failed");
    if (!root || !ranker) { free(root); free(ranker); return; }
    net_init(root, UINT64_C(404));
    net_init(ranker, UINT64_C(505));
    State a = dealt_state(UINT64_C(606));
    State b = a;
    Move pair[3];
    first_three_moves(&a, pair);

    int opponent = a.turn ^ 1;
    int hidden_hand = __builtin_ctzll(a.hand[opponent]);
    int hidden_deck = a.deck[a.deck_pos];
    b.hand[opponent] &= ~(UINT64_C(1) << hidden_hand);
    b.hand[opponent] |= UINT64_C(1) << hidden_deck;
    b.deck[b.deck_pos] = (uint8_t)hidden_hand;

    Agent actor;
    agent_default(&actor, AG_ROLLOUT, root);
    actor.action_ranker_net = ranker;
    actor.symmetries = 20;
    double sa = 0.0, sb = 0.0;
    int va = 0, vb = 0;
    int pa = rollout_action_ranker_veto(
        &actor, &a, pair[0], pair[1], &sa, &va);
    int pb = rollout_action_ranker_veto(
        &actor, &b, pair[0], pair[1], &sb, &vb);
    CHECK(va && vb && pa == pb && sa == sb,
          "hidden opponent/deck assignment changed gate: %d %.9f vs %d %.9f",
          pa, sa, pb, sb);
    free(root);
    free(ranker);
}

static void test_rolloutu4_spec_roles(void)
{
    const char *tail =
        "32:3:0.02:0:1:14:0:0:0:0:3.5:2:4:20:0:0:20:1:0:"
        "32:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1:0:0:0:1:0.25";
    char spec[512];
    snprintf(spec, sizeof spec,
             "rolloutu4:data/champion.bin:data/c8.bin:data/best.bin:%s",
             tail);
    Agent ranker;
    spec_parse(spec, &ranker);
    CHECK(ranker.kind == AG_ROLLOUT && ranker.no_belief &&
          ranker.net && ranker.continuation_net && ranker.action_ranker_net &&
          !ranker.veto_continuation_net &&
          ranker.net != ranker.continuation_net &&
          ranker.net != ranker.action_ranker_net &&
          ranker.continuation_net != ranker.action_ranker_net &&
          ranker.owns_net && ranker.owns_continuation_net &&
          ranker.owns_action_ranker_net &&
          !ranker.owns_veto_continuation_net &&
          fabsf(ranker.action_ranker_min - 0.25f) < 1e-6f,
          "rolloutu4 did not bind three distinct checkpoint roles");
    spec_release(&ranker);
    CHECK(!ranker.net && !ranker.continuation_net &&
          !ranker.action_ranker_net && !ranker.veto_continuation_net &&
          !ranker.owns_net && !ranker.owns_continuation_net &&
          !ranker.owns_action_ranker_net,
          "rolloutu4 release did not clear owned roles");

    snprintf(spec, sizeof spec,
             "rolloutu4:data/champion.bin:data/champion.bin:"
             "data/champion.bin:%s", tail);
    spec_parse(spec, &ranker);
    CHECK(ranker.net == ranker.continuation_net &&
          ranker.net == ranker.action_ranker_net &&
          !ranker.owns_continuation_net &&
          !ranker.owns_action_ranker_net &&
          !ranker.veto_continuation_net,
          "rolloutu4 same-path roles did not safely alias root");
    spec_release(&ranker);

    snprintf(spec, sizeof spec,
             "rolloutu3:data/champion.bin:data/champion.bin:"
             "data/champion.bin:%s", tail);
    /* rollout3 has no ranker threshold field, so remove field 41. */
    char *last = strrchr(spec, ':');
    CHECK(last != NULL, "could not trim rollout3 tail");
    if (last) *last = '\0';
    Agent controller;
    spec_parse(spec, &controller);
    CHECK(controller.veto_continuation_net == controller.net &&
          controller.action_ranker_net == NULL &&
          !controller.owns_action_ranker_net,
          "rolloutu3 semantics changed while adding rolloutu4");
    spec_release(&controller);
}

static void test_match_value_ranker_runtime(void)
{
    Net *root = malloc(sizeof *root);
    Net *continuation = malloc(sizeof *continuation);
    Net *ranker = malloc(sizeof *ranker);
    CHECK(root && continuation && ranker,
          "combined match-value/ranker allocation failed");
    if (!root || !continuation || !ranker) {
        free(root); free(continuation); free(ranker);
        return;
    }
    if (net_load(root, "data/champion.bin") != 0) {
        CHECK(0, "cannot load combined-runtime root checkpoint");
        free(root); free(continuation); free(ranker);
        return;
    }
    init_play_continuation(continuation);
    memcpy(ranker, root, sizeof *ranker);
    MatchValueTable *table = linear_match_value_table(continuation);
    if (!table) {
        free(root); free(continuation); free(ranker);
        return;
    }

    State st = reviewed_state(root, 20);
    CHECK(!st.over && st.deck_left > 1,
          "fixed combined-runtime state is not live/non-deck1");
    Agent actor;
    agent_default(&actor, AG_ROLLOUT, root);
    actor.continuation_net = continuation;
    actor.match_value = table;
    actor.win_q = 3;
    actor.no_belief = 1;
    actor.dets = 8;
    actor.confirm_dets = 6;
    actor.root_width = 3;
    actor.min_cand = 3;
    actor.cand_floor = 0.02f;
    actor.symmetries = 1;
    actor.playout_symmetries = 5;
    actor.playout_sample = 4;
    actor.playout_prune = 0;
    actor.override_k = 0.0f;
    actor.policy_prefix_mode = 2;
    actor.exact_terminal = 1;

    /* Discover the deterministic, independently confirmed proposal before
     * installing a direct ranker.  This is still the full objective-3 path. */
    Rng discovery_rng;
    rng_seed(&discovery_rng, UINT64_C(0x52414e4b45524d56));
    SearchStats discovery;
    Move discovery_move = rollout_move(
        &actor, &st, &discovery_rng, NULL, &discovery);
    CHECK(discovery.prefix_proposed > 0 &&
          discovery.prefix_numerical_agreement &&
          discovery.prefix_gate_passed && discovery.prefix_confirmed &&
          discovery.selection_reference == discovery.prefix_proposed &&
          MOVE_PACK(discovery_move) ==
              MOVE_PACK(discovery.mv[discovery.prefix_proposed]),
          "fixed objective-3 state did not reach confirmed proposal "
          "(proposal %d, agreement %d, gate %d, selected %d)",
          discovery.prefix_proposed,
          discovery.prefix_numerical_agreement,
          discovery.prefix_gate_passed,
          discovery.selection_reference);
    if (discovery.prefix_proposed <= 0) {
        free(table); free(root); free(continuation); free(ranker);
        return;
    }

    /* Objective zero uses the identical worlds/controller.  The fixture's
     * exact 0.05 slope makes every objective-3 Q and paired delta a direct,
     * testable table-unit transform of its round-point counterpart. */
    Agent points_actor = actor;
    points_actor.match_value = NULL;
    points_actor.win_q = 0;
    Rng points_rng;
    rng_seed(&points_rng, UINT64_C(0x52414e4b45524d56));
    SearchStats points;
    Move points_move = rollout_move(
        &points_actor, &st, &points_rng, NULL, &points);
    CHECK(MOVE_PACK(points_move) == MOVE_PACK(discovery_move) &&
          points.n == discovery.n &&
          points.prefix_proposed == discovery.prefix_proposed,
          "linear table changed the comparison rather than only its units");
    CHECK(fabs(points.q[0]) > 1e-9 &&
          fabs(points.delta[discovery.prefix_proposed]) > 1e-9,
          "fixed state did not produce nonzero Q/delta evidence");
    for (int i = 0; i < discovery.n && i < points.n; i++) {
        CHECK(MOVE_PACK(discovery.mv[i]) == MOVE_PACK(points.mv[i]) &&
              fabs(discovery.q[i] - 0.05 * points.q[i]) < 1e-9 &&
              fabs(discovery.delta[i] - 0.05 * points.delta[i]) < 1e-9,
              "candidate %d is not in match-table units: "
              "q %.9f/%.9f delta %.9f/%.9f", i,
              discovery.q[i], points.q[i],
              discovery.delta[i], points.delta[i]);
    }
    for (int i = 0; i < discovery.trusted_candidates &&
                        i < points.trusted_candidates; i++) {
        CHECK(fabs(discovery.prefix_q[i] -
                       0.05 * points.prefix_q[i]) < 1e-9 &&
              fabs(discovery.prefix_delta[i] -
                       0.05 * points.prefix_delta[i]) < 1e-9,
              "fresh candidate %d is not in match-table units: "
              "q %.9f/%.9f delta %.9f/%.9f", i,
              discovery.prefix_q[i], points.prefix_q[i],
              discovery.prefix_delta[i], points.prefix_delta[i]);
    }

    int proposal = discovery.prefix_proposed;
    ranker->bcomb[combination_index(discovery.mv[proposal])] += 2.0f;
    actor.action_ranker_net = ranker;
    actor.action_ranker_min = 1.5f;
    Rng accepted_rng;
    rng_seed(&accepted_rng, UINT64_C(0x52414e4b45524d56));
    SearchStats accepted;
    Move accepted_move = rollout_move(
        &actor, &st, &accepted_rng, NULL, &accepted);
    CHECK(accepted.prefix_ranker_attempted &&
          accepted.prefix_ranker_valid && accepted.prefix_ranker_passed &&
          fabs(accepted.prefix_ranker_score - 2.0) < 1e-9 &&
          accepted.selection_reference == proposal &&
          MOVE_PACK(accepted_move) == MOVE_PACK(accepted.mv[proposal]),
          "combined objective-3 ranker did not retain proposal "
          "(attempted %d valid %d passed %d score %.9f selected %d)",
          accepted.prefix_ranker_attempted,
          accepted.prefix_ranker_valid,
          accepted.prefix_ranker_passed,
          accepted.prefix_ranker_score,
          accepted.selection_reference);

    Agent rejected_actor = actor;
    rejected_actor.action_ranker_min = 2.5f;
    Rng rejected_rng;
    rng_seed(&rejected_rng, UINT64_C(0x52414e4b45524d56));
    SearchStats rejected;
    Move rejected_move = rollout_move(
        &rejected_actor, &st, &rejected_rng, NULL, &rejected);
    CHECK(rejected.prefix_ranker_attempted &&
          rejected.prefix_ranker_valid && !rejected.prefix_ranker_passed &&
          rejected.selection_reference == 0 &&
          MOVE_PACK(rejected_move) == MOVE_PACK(rejected.mv[0]),
          "below-threshold combined ranker did not return baseline");
    CHECK(memcmp(&accepted_rng, &rejected_rng, sizeof accepted_rng) == 0,
          "direct ranker threshold changed caller RNG consumption");
    CHECK(memcmp(&accepted_rng, &discovery_rng, sizeof accepted_rng) == 0,
          "consulting the direct ranker consumed caller RNG");

    /* An unfinished engine-cap trajectory must fail closed before the ranker
     * is consulted, even if the primary numerical panel proposed a move. */
    State capped = st;
    capped.nply = LC_MAX_PLIES - 1;
    Rng capped_rng;
    rng_seed(&capped_rng, UINT64_C(0x43415046414c4c42));
    SearchStats capped_stats;
    Move capped_move = rollout_move(
        &actor, &capped, &capped_rng, NULL, &capped_stats);
    CHECK(capped_stats.unfinished_cap_leaves > 0 &&
          !capped_stats.prefix_ranker_attempted &&
          !capped_stats.prefix_ranker_valid &&
          !capped_stats.prefix_ranker_passed &&
          capped_stats.selection_reference == 0 &&
          MOVE_PACK(capped_move) == MOVE_PACK(capped_stats.mv[0]),
          "unfinished-cap fallback reached ranker or escaped baseline "
          "(leaves %llu attempted %d selected %d)",
          (unsigned long long)capped_stats.unfinished_cap_leaves,
          capped_stats.prefix_ranker_attempted,
          capped_stats.selection_reference);

    free(table);
    free(root);
    free(continuation);
    free(ranker);
}

int main(void)
{
    test_raw_pair_score_and_gate();
    test_suit_equivariance();
    test_hidden_information_sanitized();
    test_rolloutu4_spec_roles();
    test_match_value_ranker_runtime();
    if (failures) {
        printf("%d action-ranker test(s) failed\n", failures);
        return 1;
    }
    printf("action-ranker tests passed\n");
    return 0;
}
