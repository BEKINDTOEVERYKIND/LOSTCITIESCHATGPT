#define _XOPEN_SOURCE 700

#include "../src/agent.h"
#include "../src/match_value.h"
#include "../src/search.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

#define CHECK(condition, ...) do {                                       \
    if (!(condition)) {                                                  \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);           \
        fprintf(stderr, __VA_ARGS__);                                   \
        fputc('\n', stderr);                                             \
        failures++;                                                      \
    }                                                                    \
} while (0)

static void fill_linear_values(double value[2][MATCH_VALUE_R2_COUNT],
                               int limit)
{
    int count = 2 * limit + 1;
    for (int lead = -limit; lead <= limit; lead++) {
        int index = lead + limit;
        value[0][index] = 0.09 * lead - 2.0;
        value[1][index] = 0.09 * lead + 2.0;
    }
    /* Make the intended zero-sum relationship explicit in the fixture. */
    for (int i = 0; i < count; i++)
        CHECK(fabs(value[0][i] + value[1][count - 1 - i]) < 1e-12,
              "linear fixture is not antisymmetric at %d", i);
}

static MatchValueTable *make_table(void)
{
    MatchValueTable *table = calloc(1, sizeof *table);
    CHECK(table != NULL, "cannot allocate match-value table");
    if (!table) return NULL;
    table->version = MATCH_VALUE_VERSION;
    table->samples_per_policy_lead = 400;
    table->role_cycle_size = 400;
    table->role_balance_complete = 1;
    table->isotonic_projected = 1;
    table->source_seed = UINT64_C(7331001);
    table->max_isotonic_adjustment[0] = 0.25;
    table->max_isotonic_adjustment[1] = 0.5;
    table->controller = (MatchValueController){
        .net_fingerprint = UINT64_C(0x123456789abcdef0),
        .controller_abi = MATCH_VALUE_CONTROLLER_ABI,
        .build_profile = match_value_build_profile(),
        .objective = 0,
        .playout_symmetries = 20,
        .playout_sample = 4,
        .playout_prune = 1,
        .exact_terminal = 1,
        .max_plies = LC_MAX_PLIES
    };
    fill_linear_values(
        (double (*)[MATCH_VALUE_R2_COUNT])table->before_round2,
        MATCH_VALUE_R2_LEAD_LIMIT);
    for (int lead = -MATCH_VALUE_R1_LEAD_LIMIT;
         lead <= MATCH_VALUE_R1_LEAD_LIMIT; lead++) {
        int index = lead + MATCH_VALUE_R1_LEAD_LIMIT;
        table->before_round1[0][index] = 0.1 * lead - 2.0;
        table->before_round1[1][index] = 0.1 * lead + 2.0;
    }
    return table;
}

static State terminal_state(int round)
{
    State state;
    memset(&state, 0, sizeof state);
    state.over = 1;
    state.deck_left = 0;
    state.round = (uint8_t)round;
    state.cum[0] = 7;
    /* One +10 expedition gives player zero a total carried lead of 17. */
    state.exp_n[0][0] = 1;
    state.exp_sum[0][0] = 30;
    return state;
}

static void test_lookup(MatchValueTable *table)
{
    double value = NAN;
    State state = terminal_state(0);
    CHECK(match_value_terminal(table, &state, 0, &value) &&
          fabs(value - (-0.3)) < 1e-12,
          "round-zero p0 lookup got %.17g", value);
    CHECK(match_value_terminal(table, &state, 1, &value) &&
          fabs(value - 0.3) < 1e-12,
          "round-zero p1 lookup got %.17g", value);

    state.round = 1;
    CHECK(match_value_terminal(table, &state, 0, &value) &&
          fabs(value - 3.53) < 1e-12,
          "round-one p0 lookup got %.17g", value);
    CHECK(match_value_terminal(table, &state, 1, &value) &&
          fabs(value - (-3.53)) < 1e-12,
          "round-one p1 lookup got %.17g", value);

    state.round = 2;
    CHECK(match_value_terminal(table, &state, 0, &value) &&
          fabs(value - 50.85) < 1e-12,
          "final-round hybrid utility got %.17g", value);
    CHECK(match_value_terminal(table, &state, 1, &value) &&
          fabs(value + 50.85) < 1e-12,
          "final-round utility is not zero-sum: %.17g", value);

    state = terminal_state(0);
    state.over = 0;
    CHECK(!match_value_terminal(table, &state, 0, &value),
          "unfinished state was accepted");
    state.over = 1;
    state.deck_left = 1;
    CHECK(!match_value_terminal(table, &state, 0, &value),
          "nonempty deck was accepted as a completed round");
    state.deck_left = 0;
    state.cum[0] = INT16_MAX;
    CHECK(!match_value_terminal(table, &state, 0, &value),
          "out-of-range match lead was accepted");

    state = terminal_state(0);
    int index = 17 + MATCH_VALUE_R1_LEAD_LIMIT;
    double saved = table->before_round1[0][index];
    table->before_round1[0][index] = NAN;
    CHECK(!match_value_terminal(table, &state, 0, &value),
          "non-finite selected entry was accepted by the cheap lookup");
    table->before_round1[0][index] =
        MATCH_VALUE_MAX_ABS_UTILITY + 1.0;
    CHECK(!match_value_terminal(table, &state, 0, &value),
          "out-of-range selected entry was accepted by the cheap lookup");
    table->before_round1[0][index] = saved;
    uint32_t saved_max_plies = table->controller.max_plies;
    table->controller.max_plies = LC_MAX_PLIES - 1;
    CHECK(!match_value_terminal(table, &state, 0, &value),
          "unsupported controller was accepted by the cheap lookup");
    table->controller.max_plies = saved_max_plies;
}

static void test_validation(MatchValueTable *table)
{
    CHECK(match_value_validate(table), "valid fixture was rejected");
    double saved = table->before_round1[0][10];
    table->before_round1[0][10] = NAN;
    CHECK(!match_value_validate(table), "NaN table entry was accepted");
    table->before_round1[0][10] = saved;

    double saved_adjustment = table->max_isotonic_adjustment[0];
    table->max_isotonic_adjustment[0] =
        2.0 * MATCH_VALUE_MAX_ABS_UTILITY + 1.0;
    CHECK(!match_value_validate(table),
          "impossible isotonic adjustment was accepted");
    table->max_isotonic_adjustment[0] = saved_adjustment;

    saved = table->before_round2[0][100];
    table->before_round2[0][100] =
        table->before_round2[0][99] - 1.0;
    CHECK(!match_value_validate(table), "decreasing table was accepted");
    table->before_round2[0][100] = saved;
    CHECK(match_value_validate(table), "restored fixture stayed invalid");

    int raw_index = 100;
    int raw_mirror = MATCH_VALUE_R2_COUNT - 1 - raw_index;
    double raw_not_saved = table->before_round2[0][raw_index];
    double raw_start_saved = table->before_round2[1][raw_mirror];
    table->before_round2[0][raw_index] =
        table->before_round2[0][raw_index - 1] - 1.0;
    table->before_round2[1][raw_mirror] =
        -table->before_round2[0][raw_index];
    table->isotonic_projected = 0;
    CHECK(match_value_validate(table),
          "raw fixed-policy table incorrectly required monotonicity");
    table->isotonic_projected = 1;
    CHECK(!match_value_validate(table),
          "projected table accepted a monotonicity violation");
    table->before_round2[0][raw_index] = raw_not_saved;
    table->before_round2[1][raw_mirror] = raw_start_saved;

    saved = table->before_round1[0][10];
    table->before_round1[0][10] = MATCH_VALUE_MAX_ABS_UTILITY + 1.0;
    CHECK(!match_value_validate(table),
          "value outside the theoretical utility range was accepted");
    table->before_round1[0][10] = saved;

    MatchValueController other = table->controller;
    CHECK(match_value_controller_equal(&table->controller, &other),
          "identical controller metadata differs");
    other.playout_prune ^= 1U;
    CHECK(!match_value_controller_equal(&table->controller, &other),
          "controller mismatch was ignored");
    other = table->controller;
    other.playout_sample = 1;
    CHECK(!match_value_controller_supported(&other),
          "unsupported stochastic controller was accepted");
    other = table->controller;
    other.plan_block_gap = 1000001U;
    CHECK(!match_value_controller_supported(&other),
          "out-of-range plan block gap was accepted");
    other = table->controller;
    other.controller_abi++;
    CHECK(!match_value_controller_supported(&other),
          "stale controller ABI was accepted");
    other = table->controller;
    other.build_profile ^= 1U;
    CHECK(!match_value_controller_supported(&other),
          "mismatched floating-point build profile was accepted");

    CHECK(match_value_balanced_roles(table),
          "complete 20x20 role cycle was not recognized");
    table->samples_per_policy_lead = 401;
    table->role_balance_complete = 0;
    CHECK(match_value_validate(table),
          "explicit unbalanced development artifact is structurally invalid");
    CHECK(!match_value_balanced_roles(table),
          "partial role cycle was marked production-balanced");
    table->samples_per_policy_lead = 400;
    table->role_balance_complete = 1;
}

static int temporary_path(char path[64])
{
    strcpy(path, "/tmp/lostcities-match-value-XXXXXX");
    int descriptor = mkstemp(path);
    if (descriptor < 0) return 0;
    close(descriptor);
    return unlink(path) == 0;
}

static void test_persistence(MatchValueTable *table)
{
    char path[64];
    CHECK(temporary_path(path), "cannot allocate temporary filename");
    if (failures) return;
    CHECK(match_value_save(table, path) == 0, "cannot save valid table");
    CHECK(match_value_save(table, path) == -2,
          "no-clobber save overwrote an existing table");

    int error = -99;
    MatchValueTable *loaded = match_value_load(path, &error);
    CHECK(loaded != NULL && error == 0, "load failed with error %d", error);
    if (loaded) {
        CHECK(match_value_controller_equal(&table->controller,
                                           &loaded->controller),
              "controller metadata changed across persistence");
        CHECK(loaded->source_seed == table->source_seed &&
              loaded->samples_per_policy_lead ==
                  table->samples_per_policy_lead &&
              loaded->role_cycle_size == table->role_cycle_size &&
              loaded->role_balance_complete ==
                  table->role_balance_complete &&
              loaded->isotonic_projected == table->isotonic_projected,
              "build provenance changed across persistence");
        CHECK(memcmp(table->before_round1, loaded->before_round1,
                     sizeof table->before_round1) == 0 &&
              memcmp(table->before_round2, loaded->before_round2,
                     sizeof table->before_round2) == 0,
              "table payload changed across persistence");
        match_value_free(loaded);
    }

    FILE *file = fopen(path, "ab");
    CHECK(file != NULL, "cannot open table for trailing-byte test");
    if (file) {
        fputc(0, file);
        fclose(file);
    }
    loaded = match_value_load(path, &error);
    CHECK(loaded == NULL, "table with trailing bytes was accepted");
    match_value_free(loaded);
    unlink(path);

    CHECK(temporary_path(path), "cannot allocate corruption filename");
    CHECK(match_value_save(table, path) == 0,
          "cannot save corruption fixture");
    file = fopen(path, "r+b");
    CHECK(file != NULL, "cannot open corruption fixture");
    if (file) {
        /* Reserved header bytes are covered by the artifact checksum too. */
        CHECK(fseek(file, 127, SEEK_SET) == 0,
              "cannot seek in corruption fixture");
        fputc(1, file);
        fclose(file);
    }
    loaded = match_value_load(path, &error);
    CHECK(loaded == NULL, "header corruption escaped the checksum");
    match_value_free(loaded);
    unlink(path);
}

static void test_net_fingerprint(void)
{
    Net *net = malloc(sizeof *net);
    CHECK(net != NULL, "cannot allocate fingerprint network");
    if (!net) return;
    net_zero(net);
    uint64_t first = match_value_net_fingerprint(net);
    uint64_t repeat = match_value_net_fingerprint(net);
    net->b3 = 1.0f;
    uint64_t changed = match_value_net_fingerprint(net);
    CHECK(first != 0 && first == repeat,
          "network fingerprint is not deterministic");
    CHECK(first != changed, "network fingerprint ignored a parameter bit");
    free(net);
}

static void test_exact_deck_one_uses_match_value(
    MatchValueTable *table)
{
    Net *net = malloc(sizeof *net);
    CHECK(net != NULL, "cannot allocate exact-deck network");
    if (!net) return;
    net_zero(net);
    uint64_t saved_fingerprint = table->controller.net_fingerprint;
    table->controller.net_fingerprint = match_value_net_fingerprint(net);

    State state;
    memset(&state, 0, sizeof state);
    state.round = 0;
    state.turn = 0;
    state.deck_left = 1;
    state.deck[0] = (uint8_t)CARD_MAKE(4, 3);
    int yellow2 = CARD_MAKE(0, 3);
    int yellow10 = CARD_MAKE(0, 11);
    int red2 = CARD_MAKE(1, 3);
    state.played[0] = UINT64_C(1) << yellow2;
    state.exp_top[0][0] = 2;
    state.exp_n[0][0] = 1;
    state.exp_sum[0][0] = 2;
    state.hand[0] = (UINT64_C(1) << yellow10) |
                    (UINT64_C(1) << red2);
    state.hand_n[0] = 2;

    Agent legacy;
    agent_default(&legacy, AG_ROLLOUT, net);
    legacy.exact_terminal = 1;
    Rng legacy_rng;
    rng_seed(&legacy_rng, UINT64_C(7331013));
    Rng legacy_rng_before = legacy_rng;
    Move legacy_selected = rollout_move(
        &legacy, &state, &legacy_rng, NULL, NULL);
    CHECK(legacy_selected.card == yellow10 &&
          legacy_selected.discard == 0 && legacy_selected.draw == 0,
          "legacy deck-one selector changed");
    CHECK(memcmp(&legacy_rng, &legacy_rng_before, sizeof legacy_rng) == 0,
          "legacy exact deck-one selector consumed RNG");

    Agent agent;
    agent_default(&agent, AG_ROLLOUT, net);
    agent.win_q = 3;
    agent.match_value = table;
    agent.exact_terminal = 1;
    Rng rng;
    rng_seed(&rng, UINT64_C(7331013));
    SearchStats stats;
    Move selected = rollout_move(&agent, &state, &rng, NULL, &stats);
    CHECK(selected.card == yellow10 && selected.discard == 0 &&
          selected.draw == 0,
          "deck-one table actor did not choose the maximum-margin action");

    State terminal = state;
    lc_apply_play(&terminal, selected);
    terminal.deck_left = 0;
    terminal.over = 1;
    double expected = NAN;
    CHECK(match_value_terminal(table, &terminal, 0, &expected),
          "cannot evaluate exact-deck selected state");
    CHECK(stats.n >= 1 && fabs(stats.q[0] - expected) < 1e-12,
          "deck-one diagnostic did not report table utility");

    /* A raw fixed-policy table can genuinely be nonmonotone because changing
     * carried lead changes controller behavior.  Prove mode 3 ranks the table
     * directly rather than silently retaining the legacy margin maximizer. */
    int lead_a = -18, lead_b = -8;
    int index_a = lead_a + MATCH_VALUE_R1_LEAD_LIMIT;
    int index_b = lead_b + MATCH_VALUE_R1_LEAD_LIMIT;
    int mirror_a = MATCH_VALUE_R1_COUNT - 1 - index_a;
    int mirror_b = MATCH_VALUE_R1_COUNT - 1 - index_b;
    double saved_a = table->before_round1[0][index_a];
    double saved_b = table->before_round1[0][index_b];
    double saved_ma = table->before_round1[1][mirror_a];
    double saved_mb = table->before_round1[1][mirror_b];
    table->before_round1[0][index_a] = 10.0;
    table->before_round1[1][mirror_a] = -10.0;
    table->before_round1[0][index_b] = 0.0;
    table->before_round1[1][mirror_b] = 0.0;
    table->isotonic_projected = 0;
    CHECK(match_value_validate(table),
          "nonmonotone raw deck-one fixture is invalid");
    rng_seed(&rng, UINT64_C(7331014));
    selected = rollout_move(&agent, &state, &rng, NULL, &stats);
    CHECK(selected.discard == 1 && selected.draw == 0,
          "raw table utility did not override the maximum-margin action");
    CHECK(fabs(stats.q[0] - 10.0) < 1e-12,
          "raw table deck-one diagnostic has wrong objective %.17g",
          stats.q[0]);
    table->before_round1[0][index_a] = saved_a;
    table->before_round1[0][index_b] = saved_b;
    table->before_round1[1][mirror_a] = saved_ma;
    table->before_round1[1][mirror_b] = saved_mb;
    table->isotonic_projected = 1;

    table->controller.net_fingerprint = saved_fingerprint;
    free(net);
}

int main(void)
{
    MatchValueTable *table = make_table();
    if (table) {
        test_validation(table);
        test_lookup(table);
        test_persistence(table);
        test_exact_deck_one_uses_match_value(table);
        free(table);
    }
    test_net_fingerprint();
    if (failures) {
        fprintf(stderr, "%d match-value test(s) failed\n", failures);
        return 1;
    }
    puts("match-value tests passed");
    return 0;
}
