/* Focused v5 regressions for policy-frequency artifacts and rollout5. */
#include "../src/match_value.h"
#include "../src/policy_cost_v17.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); failures++; \
} } while (0)

static const uint32_t anchors[POLICY_COST_ANCHORS] = {
    0, 4, 8, 12, 16, 24, 32, 40, 48, 64
};

static PolicyCostTable fixture(void)
{
    PolicyCostTable table;
    memset(&table, 0, sizeof table);
    table.version = POLICY_COST_VERSION;
    table.source_seed = POLICY_COST_SOURCE_SEED;
    table.epsilon = POLICY_COST_EPSILON;
    table.primary_z = POLICY_COST_PRIMARY_Z;
    table.fresh_z = POLICY_COST_FRESH_Z;
    table.controller = (PolicyCostController){
        .root_net_fingerprint = UINT64_C(0x1111222233334444),
        .continuation_net_fingerprint = UINT64_C(0x1111222233334444),
        .controller_abi = POLICY_COST_CONTROLLER_ABI,
        .build_profile = match_value_build_profile(),
        .objective = 0,
        .root_symmetries = 20,
        .playout_symmetries = 20,
        .playout_sample = 4,
        .playout_prune = 1,
        .exact_terminal = 1,
        .no_belief = 1,
        .dets = 800,
        .confirm_dets = 800,
        .root_width = 5,
        .action_core_count = 3,
        .min_cand = 1,
        .ply_lo = 0,
        .ply_hi = 0,
        .discard_guard = 1,
        .root_prune = 0,
        .cand_floor = 0.01f,
        .override_k = 3.5f,
        .override_min = 0.0f,
    };
    for (int a = 0; a < POLICY_COST_ANCHORS; a++) {
        table.ply_anchor[a] = anchors[a];
        table.beta[a] = 1.0;
        table.lambda_action[a] = 1.0 + 0.1 * anchors[a];
        table.lambda_draw[a] = 0.25 + 0.01 * anchors[a];
    }
    for (int a = 4; a <= 6; a++) {
        table.alpha_action[a] = 0.0;
        table.alpha_draw[a] = 0.0;
    }
    return table;
}

static MatchValueTable *match_fixture(uint64_t net_fingerprint)
{
    MatchValueTable *table = calloc(1, sizeof *table);
    if (!table) return NULL;
    table->version = MATCH_VALUE_VERSION;
    table->samples_per_policy_lead = 400;
    table->role_cycle_size = 400;
    table->role_balance_complete = 1;
    table->isotonic_projected = 1;
    table->source_seed = UINT64_C(7331001);
    table->controller = (MatchValueController){
        .net_fingerprint = net_fingerprint,
        .controller_abi = MATCH_VALUE_CONTROLLER_ABI,
        .build_profile = match_value_build_profile(),
        .objective = 0,
        .playout_symmetries = 20,
        .playout_sample = 4,
        .playout_prune = 1,
        .exact_terminal = 1,
        .max_plies = LC_MAX_PLIES,
    };
    return table;
}

static void temporary_path(char path[128], const char *tag)
{
    snprintf(path, 128, "/tmp/lc_policy_cost_%ld_%s.lcpc",
             (long)getpid(), tag);
    (void)remove(path);
}

static int files_equal(const char *left_path, const char *right_path)
{
    FILE *left = fopen(left_path, "rb");
    FILE *right = fopen(right_path, "rb");
    if (!left || !right) {
        if (left) fclose(left);
        if (right) fclose(right);
        return 0;
    }
    int equal = 1;
    for (;;) {
        int a = fgetc(left), b = fgetc(right);
        if (a != b) equal = 0;
        if (a == EOF || b == EOF) break;
    }
    if (ferror(left) || ferror(right)) equal = 0;
    fclose(left);
    fclose(right);
    return equal;
}

static uint64_t test_fingerprint(const unsigned char *data, size_t size)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void test_store_u64(unsigned char out[8], uint64_t value)
{
    for (int i = 0; i < 8; i++) out[i] = (unsigned char)(value >> (8 * i));
}

static void test_schedule_and_format(void)
{
    PolicyCostTable table = fixture();
    CHECK(policy_cost_validate(&table), "valid fixture was rejected");
    PolicyCostTable legacy = table;
    PolicyCostTable v4 = table;
    PolicyCostTable v3 = table;
    v4.version = POLICY_COST_V17_VERSION;
    v4.source_seed = POLICY_COST_V17_SOURCE_SEED;
    CHECK(policy_cost_validate(&v4),
          "v4 predictive identity was rejected");
    char v4_path[128];
    temporary_path(v4_path, "v4-format");
    CHECK(policy_cost_save(&v4, v4_path) == 0,
          "v4 predictive artifact save failed");
    int v4_error = 99;
    PolicyCostTable *v4_loaded = policy_cost_load(v4_path, &v4_error);
    CHECK(v4_loaded && v4_error == 0 &&
          v4_loaded->version == POLICY_COST_V17_VERSION &&
          v4_loaded->source_seed == POLICY_COST_V17_SOURCE_SEED &&
          v4_loaded->beta[7] == v4.beta[7] &&
          v4_loaded->alpha_action[7] == v4.alpha_action[7],
          "v4 predictive artifact lost read compatibility");
    char v4_copy_path[128];
    temporary_path(v4_copy_path, "v4-format-copy");
    CHECK(v4_loaded && policy_cost_save(v4_loaded, v4_copy_path) == 0 &&
          files_equal(v4_path, v4_copy_path),
          "v4 predictive artifact did not byte-round-trip");
    (void)remove(v4_copy_path);
    policy_cost_free(v4_loaded);
    (void)remove(v4_path);
    PolicyCostTable v2 = table;
    v3.version = POLICY_COST_V3_VERSION;
    v3.source_seed = POLICY_COST_V3_SOURCE_SEED;
    CHECK(policy_cost_validate(&v3),
          "v3 predictive identity was rejected");
    char v3_path[128];
    temporary_path(v3_path, "v3-format");
    CHECK(policy_cost_save(&v3, v3_path) == 0,
          "v3 predictive artifact save failed");
    int v3_error = 99;
    PolicyCostTable *v3_loaded = policy_cost_load(v3_path, &v3_error);
    CHECK(v3_loaded && v3_error == 0 &&
          v3_loaded->version == POLICY_COST_V3_VERSION &&
          v3_loaded->source_seed == POLICY_COST_V3_SOURCE_SEED &&
          v3_loaded->beta[7] == v3.beta[7] &&
          v3_loaded->alpha_action[7] == v3.alpha_action[7],
          "v3 predictive artifact lost read compatibility");
    char v3_copy_path[128];
    temporary_path(v3_copy_path, "v3-format-copy");
    CHECK(v3_loaded && policy_cost_save(v3_loaded, v3_copy_path) == 0 &&
          files_equal(v3_path, v3_copy_path),
          "v3 predictive artifact did not byte-round-trip");
    (void)remove(v3_copy_path);
    policy_cost_free(v3_loaded);
    (void)remove(v3_path);

    v2.version = POLICY_COST_V3_VERSION;
    v2.source_seed = POLICY_COST_V2_SOURCE_SEED;
    CHECK(policy_cost_validate(&v2),
          "v2 predictive source seed was rejected");
    char v2_path[128];
    temporary_path(v2_path, "v2-format");
    CHECK(policy_cost_save(&v2, v2_path) == 0,
          "v2 predictive artifact save failed");
    int v2_error = 99;
    PolicyCostTable *v2_loaded = policy_cost_load(v2_path, &v2_error);
    CHECK(v2_loaded && v2_error == 0 &&
          v2_loaded->version == POLICY_COST_V3_VERSION &&
          v2_loaded->source_seed == POLICY_COST_V2_SOURCE_SEED &&
          v2_loaded->beta[7] == v2.beta[7] &&
          v2_loaded->alpha_action[7] == v2.alpha_action[7],
          "v2 predictive artifact lost read compatibility");
    char v2_copy_path[128];
    temporary_path(v2_copy_path, "v2-format-copy");
    CHECK(v2_loaded && policy_cost_save(v2_loaded, v2_copy_path) == 0 &&
          files_equal(v2_path, v2_copy_path),
          "v2 predictive artifact did not byte-round-trip");
    (void)remove(v2_copy_path);
    policy_cost_free(v2_loaded);
    (void)remove(v2_path);

    legacy.version = POLICY_COST_LEGACY_VERSION;
    legacy.source_seed = POLICY_COST_LEGACY_SOURCE_SEED;
    CHECK(policy_cost_validate(&legacy),
          "legacy version/source-seed pair was rejected");
    legacy.controller.ply_lo = 14;
    CHECK(policy_cost_validate(&legacy),
          "legacy version lost delayed-onset compatibility");
    legacy.controller.ply_lo = 0;
    legacy.source_seed = POLICY_COST_SOURCE_SEED;
    CHECK(!policy_cost_validate(&legacy),
          "legacy version accepted the v17 source seed");
    legacy.version = POLICY_COST_VERSION;
    legacy.source_seed = POLICY_COST_LEGACY_SOURCE_SEED;
    CHECK(!policy_cost_validate(&legacy),
          "predictive version accepted the legacy source seed");
    legacy = table;
    legacy.source_seed = POLICY_COST_V3_SOURCE_SEED;
    CHECK(!policy_cost_validate(&legacy),
          "v5 version accepted the v3 source seed");
    legacy.version = POLICY_COST_V3_VERSION;
    legacy.source_seed = POLICY_COST_SOURCE_SEED;
    CHECK(!policy_cost_validate(&legacy),
          "v3 version accepted the v17 source seed");
    legacy.source_seed = UINT64_C(202799999999);
    CHECK(!policy_cost_validate(&legacy),
          "predictive version accepted an unknown source seed");
    legacy.version = 2;
    legacy.source_seed = POLICY_COST_V2_SOURCE_SEED;
    CHECK(!policy_cost_validate(&legacy),
          "unknown artifact version was accepted");
    legacy = table;
    legacy.version = POLICY_COST_LEGACY_VERSION;
    legacy.source_seed = POLICY_COST_LEGACY_SOURCE_SEED;
    char legacy_path[128];
    temporary_path(legacy_path, "legacy-format");
    CHECK(policy_cost_save(&legacy, legacy_path) == 0,
          "legacy artifact save failed");
    unsigned char legacy_header[40] = {0};
    FILE *legacy_file = fopen(legacy_path, "rb");
    CHECK(legacy_file &&
          fread(legacy_header, 1, sizeof legacy_header, legacy_file) ==
              sizeof legacy_header,
          "cannot read legacy artifact header");
    if (legacy_file) fclose(legacy_file);
    CHECK(legacy_header[8] == POLICY_COST_LEGACY_VERSION &&
          legacy_header[9] == 0 && legacy_header[10] == 0 &&
          legacy_header[11] == 0,
          "legacy artifact was rewritten with a predictive header");
    int legacy_error = 99;
    PolicyCostTable *legacy_loaded = policy_cost_load(
        legacy_path, &legacy_error);
    CHECK(legacy_loaded && legacy_error == 0 &&
          legacy_loaded->version == POLICY_COST_LEGACY_VERSION &&
          legacy_loaded->source_seed == POLICY_COST_LEGACY_SOURCE_SEED,
          "legacy artifact did not round-trip byte-compatible identity");
    char legacy_copy_path[128];
    temporary_path(legacy_copy_path, "legacy-format-copy");
    CHECK(legacy_loaded &&
          policy_cost_save(legacy_loaded, legacy_copy_path) == 0 &&
          files_equal(legacy_path, legacy_copy_path),
          "legacy artifact did not byte-round-trip");
    (void)remove(legacy_copy_path);
    policy_cost_free(legacy_loaded);
    (void)remove(legacy_path);
    legacy.controller.ply_lo = 14;
    temporary_path(legacy_path, "legacy-delayed-onset");
    CHECK(policy_cost_save(&legacy, legacy_path) == 0,
          "legacy delayed-onset artifact save failed");
    legacy_loaded = policy_cost_load(legacy_path, &legacy_error);
    CHECK(legacy_loaded && legacy_error == 0 &&
          legacy_loaded->controller.ply_lo == 14,
          "legacy delayed-onset artifact did not remain loadable");
    policy_cost_free(legacy_loaded);
    (void)remove(legacy_path);
    /* Interpolate beta and alpha independently.  At ply 20 the normalized
     * action anchors are 2 and 1, whose incorrectly interpolated ratio would
     * be 1.5; the correct ratio of interpolants is 4/3. */
    PolicyCostTable interpolated = v4;
    interpolated.beta[4] = 2.0;
    interpolated.beta[5] = 4.0;
    interpolated.alpha_action[4] = 4.0;
    interpolated.alpha_action[5] = 4.0;
    interpolated.alpha_draw[4] = 2.0;
    interpolated.alpha_draw[5] = 2.0;
    double beta = 0.0, aa = 0.0, ad = 0.0;
    double la = 0.0, ld = 0.0;
    int interval = -1;
    CHECK(policy_cost_coefficients(
              &interpolated, 20, &beta, &aa, &ad, &interval) &&
          interval == 4 && fabs(beta - 3.0) < 1e-12 &&
          fabs(aa - 4.0) < 1e-12 && fabs(ad - 2.0) < 1e-12,
          "predictive spline interpolation is wrong: "
          "i=%d beta=%.12f aa=%.12f ad=%.12f",
          interval, beta, aa, ad);
    CHECK(policy_cost_schedule(&interpolated, 20, &la, &ld, &interval) &&
          interval == 4 && fabs(la - 4.0 / 3.0) < 1e-12 &&
          fabs(ld - 2.0 / 3.0) < 1e-12,
          "shared spline interpolation is wrong: i=%d la=%.12f ld=%.12f",
          interval, la, ld);
    Move interpolation_move[2] = {
        { CARD_MAKE(0, 3), 0, 0 },
        { CARD_MAKE(1, 3), 0, 0 },
    };
    float interpolation_prob[2] = { 0.75f, 0.25f };
    double pa = 0.0, pd = 0.0, cost = 0.0;
    CHECK(policy_cost_move_terms(
              &interpolated, 0, 20, interpolation_move, interpolation_prob, 2, 1,
              &pa, &pd, &cost) && pa == 0.25 && pd == 1.0 &&
          fabs(cost + (4.0 / 3.0) * log(0.25)) < 1e-12,
          "runtime did not score with the ratio of interpolated alpha/beta: "
          "PA=%.12f PD=%.12f cost=%.12f", pa, pd, cost);
    double q = 5.0;
    double scaled = beta * q + aa * log(pa) + ad * log(pd);
    CHECK(fabs(scaled - beta * (q - cost)) < 1e-12,
          "scaled predictive score and normalized runtime score diverged");
    for (int ply = 16; ply < 40; ply++) {
        CHECK(policy_cost_coefficients(
                  &table, ply, &beta, &aa, &ad, &interval) &&
              aa == 0.0 && ad == 0.0,
              "current phase-selective table was nonzero at ply %d: "
              "aa=%.12f ad=%.12f", ply, aa, ad);
    }
    CHECK(policy_cost_coefficients(
              &table, 40, &beta, &aa, &ad, &interval) &&
          aa == table.alpha_action[7] && ad == table.alpha_draw[7] &&
          aa > 0.0 && ad > 0.0,
          "late policy costs did not begin independently at ply 40");
    CHECK(policy_cost_schedule(&table, 299, &la, &ld, &interval) &&
          interval == POLICY_COST_ANCHORS - 1 &&
          fabs(la - 7.4) < 1e-12 && fabs(ld - 0.89) < 1e-12,
          "post-64 schedule did not clamp: i=%d la=%.12f ld=%.12f",
          interval, la, ld);

    PolicyCostTable bad = table;
    bad.controller.ply_lo = 14;
    CHECK(!policy_cost_validate(&bad),
          "predictive artifact accepted a delayed search onset");
    bad = table;
    bad.beta[3] = 0.0;
    CHECK(!policy_cost_validate(&bad), "zero beta was accepted");
    bad = table;
    bad.alpha_action[3] = -1.0;
    CHECK(!policy_cost_validate(&bad), "negative action alpha was accepted");
    bad = table;
    bad.alpha_action[6] = 0x1p-40;
    CHECK(!policy_cost_validate(&bad),
          "current table accepted a nonzero midgame action alpha");
    bad = table;
    bad.alpha_draw[4] = -0.0;
    CHECK(!policy_cost_validate(&bad),
          "current table accepted a negative-zero midgame draw alpha");
    bad = table;
    bad.ply_anchor[5]++;
    CHECK(!policy_cost_validate(&bad), "noncanonical anchor was accepted");
    bad = table;
    bad.controller.objective = 3;
    bad.controller.match_value_fingerprint = UINT64_C(0x1234);
    CHECK(policy_cost_validate(&bad),
          "objective-3 artifact with a table binding was rejected");
    bad.controller.match_value_fingerprint = 0;
    CHECK(!policy_cost_validate(&bad),
          "objective-3 artifact without a table binding was accepted");
    bad = table;
    bad.controller.root_prune = 1;
    CHECK(!policy_cost_validate(&bad),
          "policy-cost artifact accepted a pruned root policy");
    bad = table;
    bad.controller.no_belief = 0;
    CHECK(!policy_cost_validate(&bad),
          "policy-cost accepted an unbound learned-belief sampler");
    bad = table;
    bad.controller.override_min = 2.0f;
    CHECK(!policy_cost_validate(&bad),
          "policy-cost accepted the legacy low-prior practical floor");
    bad.controller.override_min = -0.0f;
    CHECK(!policy_cost_validate(&bad),
          "policy-cost accepted a noncanonical negative-zero floor");

    char path[128];
    temporary_path(path, "format");
    CHECK(policy_cost_save(&table, path) == 0,
          "canonical artifact save failed");
    CHECK(policy_cost_save(&table, path) == -2,
          "save clobbered an existing artifact");
    int error = 99;
    PolicyCostTable *loaded = policy_cost_load(path, &error);
    CHECK(loaded && error == 0 && loaded->payload_fingerprint != 0 &&
          loaded->controller.cand_floor == table.controller.cand_floor &&
          loaded->controller.override_k == table.controller.override_k &&
          loaded->beta[7] == table.beta[7] &&
          loaded->alpha_action[7] == table.alpha_action[7],
          "canonical round trip lost content or binary32 bindings");
    char current_copy_path[128];
    temporary_path(current_copy_path, "v5-format-copy");
    CHECK(loaded && policy_cost_save(loaded, current_copy_path) == 0 &&
          files_equal(path, current_copy_path),
          "v5 predictive artifact did not byte-round-trip");
    (void)remove(current_copy_path);
    policy_cost_free(loaded);

    enum { ARTIFACT_BYTES = 256 + 24 * POLICY_COST_ANCHORS + 8 };
    unsigned char raw[ARTIFACT_BYTES];
    FILE *file = fopen(path, "r+b");
    CHECK(file && fread(raw, 1, sizeof raw, file) == sizeof raw,
          "cannot read artifact for reserved-byte test");
    if (file) {
        raw[28] = 1;
        test_store_u64(raw + sizeof raw - 8,
                       test_fingerprint(raw, sizeof raw - 8));
        rewind(file);
        CHECK(fwrite(raw, 1, sizeof raw, file) == sizeof raw,
              "cannot write reserved-byte mutation");
        fclose(file);
    }
    loaded = policy_cost_load(path, &error);
    CHECK(!loaded && error == -2,
          "checksummed nonzero reserved header byte was accepted");
    (void)remove(path);
    CHECK(policy_cost_save(&table, path) == 0,
          "cannot recreate artifact after reserved-byte test");

    file = fopen(path, "r+b");
    CHECK(file != NULL, "cannot reopen artifact for corruption test");
    if (file) {
        CHECK(fseek(file, 260, SEEK_SET) == 0,
              "cannot seek to artifact payload");
        int byte = fgetc(file);
        CHECK(byte != EOF && fseek(file, 260, SEEK_SET) == 0 &&
              fputc(byte ^ 0x01, file) != EOF,
              "cannot corrupt artifact payload");
        fclose(file);
    }
    loaded = policy_cost_load(path, &error);
    CHECK(!loaded && error == -4,
          "content-hash mismatch did not fail closed (error %d)", error);
    (void)remove(path);
}

static void test_wager_semantic_complete_mass(void)
{
    PolicyCostTable table = fixture();
    for (int a = 0; a < POLICY_COST_ANCHORS; a++) {
        if (a >= 4 && a <= 6) continue;
        table.lambda_action[a] = 1.0;
        table.lambda_draw[a] = 1.0;
    }
    /* The three same-suit wager cards are physical identities for dealing,
     * but one semantic action at decision time.  Their same-draw mass must be
     * summed in both P_A and P_D's numerator. */
    Move legal[5] = {
        { CARD_MAKE(0, 0), 0, 0 },
        { CARD_MAKE(0, 1), 0, 0 },
        { CARD_MAKE(0, 2), 0, 0 },
        { CARD_MAKE(0, 0), 0, 1 },
        { CARD_MAKE(1, 3), 0, 0 },
    };
    float prob[5] = { 0.01f, 0.02f, 0.03f, 0.04f, 0.90f };
    double pa[3], pd[3], cost[3];
    for (int i = 0; i < 3; i++)
        CHECK(policy_cost_move_terms(
                  &table, 0, 8, legal, prob, 5, i,
                  &pa[i], &pd[i], &cost[i]),
              "wager copy %d had invalid semantic terms", i);
    for (int i = 0; i < 3; i++)
        CHECK(fabs(pa[i] - 0.10) < 2e-7 &&
              fabs(pd[i] - 0.60) < 2e-7 &&
              fabs(cost[i] - cost[0]) < 1e-12,
              "wager copy %d terms diverged: PA=%.9f PD=%.9f cost=%.9f",
              i, pa[i], pd[i], cost[i]);

    /* A shortlist must not spend two candidate rows on indistinguishable
     * physical copies of that same complete semantic move. */
    int candidate[2] = { 0, 1 };
    double values[2 * 4] = { 0 };
    PolicyCostDecision decision;
    CHECK(!policy_cost_decide(
              &table, 0, 8, legal, prob, 5, candidate, 2,
              values, 4, 4, POLICY_COST_PRIMARY_Z, &decision),
          "semantic-complete duplicate wager candidates were accepted");
}

static void constant_candidate(double *values, int stride, int candidate,
                               double value)
{
    for (int d = 0; d < stride; d++)
        values[(size_t)candidate * stride + d] = value;
}

static int same_complete_semantic_move(Move a, Move b)
{
    int same_action = a.discard == b.discard &&
        ((CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card) &&
          CARD_SUIT(a.card) == CARD_SUIT(b.card)) || a.card == b.card);
    return same_action && a.draw == b.draw;
}

static int same_semantic_action_test(Move a, Move b)
{
    return a.discard == b.discard &&
        ((CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card) &&
          CARD_SUIT(a.card) == CARD_SUIT(b.card)) || a.card == b.card);
}

static uint16_t semantic_move_pack_test(Move move)
{
    if (CARD_IS_WAGER(move.card))
        move.card = (uint8_t)CARD_MAKE(CARD_SUIT(move.card), 0);
    return MOVE_PACK(move);
}

static void test_policy_cost_master_and_no_refill_mask(void)
{
    Rng rng;
    rng_seed(&rng, UINT64_C(0x4e4f524546494c4c));
    State state;
    lc_deal(&state, &rng);
    while (!state.over && state.deck_left > 1) {
        Move turn[MAX_MOVES];
        int nturn = lc_moves(&state, turn), chosen = -1;
        for (int i = 0; i < nturn; i++)
            if (turn[i].discard && turn[i].draw == 0) {
                chosen = i;
                break;
            }
        CHECK(chosen >= 0, "cannot build late shortlist fixture");
        if (chosen < 0) return;
        lc_apply(&state, turn[chosen]);
    }
    CHECK(!state.over && state.deck_left == 1,
          "late shortlist fixture did not stop before the terminal draw");
    if (state.over || state.deck_left != 1) return;

    Move move[MAX_MOVES];
    int n = lc_moves(&state, move);
    int first[MAX_MOVES], deck[MAX_MOVES], pile[MAX_MOVES], ncore = 0;
    for (int i = 0; i < n; i++) {
        int core = -1;
        for (int c = 0; c < ncore; c++)
            if (same_semantic_action_test(move[i], move[first[c]])) {
                core = c;
                break;
            }
        if (core < 0) {
            core = ncore++;
            first[core] = i;
            deck[core] = -1;
            pile[core] = -1;
        }
        if (move[i].draw == 0) deck[core] = i;
        else if (pile[core] < 0) pile[core] = i;
    }
    int selected[3] = { -1, -1, -1 };
    for (int c = 0; c < ncore; c++) {
        if (deck[c] < 0) continue;
        if (selected[0] < 0) selected[0] = c;
        else if (selected[1] < 0) selected[1] = c;
        else if (pile[c] >= 0) {
            selected[2] = c;
            break;
        }
    }
    CHECK(ncore >= 6 && selected[2] >= 0,
          "late state lacks the required semantic-core fixture");
    if (ncore < 6 || selected[2] < 0) return;

    float probability[MAX_MOVES];
    const float tiny = ldexpf(1.0f, -30);
    for (int i = 0; i < n; i++) probability[i] = tiny;
    const double target[3] = { 0.70, 0.25, 0.015 };
    int representative[3] = {
        deck[selected[0]], deck[selected[1]], pile[selected[2]]
    };
    for (int s = 0; s < 3; s++) {
        int count = 0;
        for (int i = 0; i < n; i++)
            if (same_semantic_action_test(
                    move[i], move[first[selected[s]]])) count++;
        probability[representative[s]] =
            (float)(target[s] - tiny * (count - 1));
    }
    int other_cores = ncore - 3;
    double other_mass = (1.0 - target[0] - target[1] - target[2]) /
                        other_cores;
    for (int c = 0; c < ncore; c++) {
        int special = c == selected[0] || c == selected[1] ||
                      c == selected[2];
        if (special) continue;
        int count = 0, rep = first[c];
        for (int i = 0; i < n; i++)
            if (same_semantic_action_test(move[i], move[first[c]])) {
                count++;
                if (move[i].draw == 0) rep = i;
            }
        probability[rep] = (float)(other_mass - tiny * (count - 1));
    }
    double total = 0.0;
    for (int i = 0; i < n; i++) total += probability[i];
    probability[representative[0]] += (float)(1.0 - total);

    RolloutPolicyCostSupport support;
    CHECK(rollout_policy_cost_support(
              &state, move, probability, n, representative[0], &support),
          "canonical policy-cost support construction failed");
    CHECK(support.n[0] == 4 && support.core_candidates[0] == 3 &&
          support.draw_candidates[0] == 1 &&
          support.order[0][0] == representative[0] &&
          support.order[0][1] == representative[1] &&
          support.order[0][2] == representative[2] &&
          support.order[0][3] == deck[selected[2]] &&
          probability[support.order[0][3]] < 0.01f,
          "1%% master omitted the rare positive conditional-draw variant");
    CHECK(support.n[1] == 2 && support.core_candidates[1] == 2 &&
          support.draw_candidates[1] == 0 &&
          support.order[1][0] == support.order[0][0] &&
          support.order[1][1] == support.order[0][1],
          "2%% mask refilled, reordered, or retained a removed-core draw");

    /* Exact aggregate/core-representative ties in the new controller have a
     * semantic packed-key order.  This is an explicit policy-cost contract,
     * rather than an accidental dependency on lc_moves() enumeration.  The
     * baseline remains the deployed literal complete-move argmax. */
    float tied[MAX_MOVES] = { 0 };
    int tied_core[2] = { -1, -1 }, ntied_core = 0;
    for (int c = 0; c < ncore && ntied_core < 2; c++)
        if (c != selected[0] && deck[c] >= 0 && pile[c] >= 0)
            tied_core[ntied_core++] = c;
    CHECK(ntied_core == 2,
          "late state lacks two draw-bearing cores for semantic tie fixture");
    if (ntied_core != 2) return;
    int tied_first = tied_core[0], tied_second = tied_core[1];
    tied[deck[selected[0]]] = 0.55f;
    tied[deck[tied_first]] = 0.075f;
    tied[pile[tied_first]] = 0.075f;
    tied[deck[tied_second]] = 0.075f;
    tied[pile[tied_second]] = 0.075f;
    int ordinary = ncore - 3;
    for (int c = 0; c < ncore; c++) {
        if (c == selected[0] || c == tied_first || c == tied_second)
            continue;
        int rep = deck[c] >= 0 ? deck[c] : first[c];
        tied[rep] = 0.15f / (float)ordinary;
    }
    double tied_total = 0.0;
    for (int i = 0; i < n; i++) tied_total += tied[i];
    tied[deck[selected[0]]] += (float)(1.0 - tied_total);
    int expected_first =
        MOVE_PACK(move[deck[tied_first]]) <
                MOVE_PACK(move[deck[tied_second]])
            ? deck[tied_first] : deck[tied_second];
    int expected_second = expected_first == deck[tied_first]
        ? deck[tied_second] : deck[tied_first];
    CHECK(rollout_policy_cost_support(
              &state, move, tied, n, deck[selected[0]], &support) &&
          support.core_candidates[0] == 3 &&
          support.order[0][1] == expected_first &&
          support.order[0][2] == expected_second,
          "policy-cost exact ties did not use the semantic packed order "
          "(ok=%d cores=%d got=%d,%d expected=%d,%d)",
          support.n[0] > 0, support.core_candidates[0],
          support.order[0][1], support.order[0][2],
          expected_first, expected_second);

    float damaged[MAX_MOVES];
    memcpy(damaged, probability, sizeof(float) * (size_t)n);
    damaged[0] *= 0.5f;
    CHECK(!rollout_policy_cost_support(
              &state, move, damaged, n, representative[0], &support),
          "policy-cost support accepted a non-normalized/subset policy");

    /* Candidate zero cannot displace a true top-three semantic core.  If a
     * literal complete-move argmax is only the fourth aggregate core, retain
     * it as the mandatory comparator plus all three nominated cores. */
    int multi[3] = { -1, -1, -1 }, nmulti = 0;
    for (int c = 0; c < ncore && nmulti < 3; c++) {
        if (c == selected[0]) continue;
        int count = 0;
        for (int i = 0; i < n; i++)
            if (same_semantic_action_test(move[i], move[first[c]])) count++;
        if (count >= 2) multi[nmulti++] = c;
    }
    CHECK(nmulti == 3,
          "late state lacks three split-probability cores for top-three guard");
    if (nmulti == 3) {
        float fourth[MAX_MOVES];
        for (int i = 0; i < n; i++) fourth[i] = tiny;
        int baseline_rep = deck[selected[0]] >= 0
            ? deck[selected[0]] : first[selected[0]];
        int baseline_count = 0;
        for (int i = 0; i < n; i++)
            if (same_semantic_action_test(move[i], move[first[selected[0]]]))
                baseline_count++;
        fourth[baseline_rep] = (float)(0.20 - tiny * (baseline_count - 1));
        const double rival_mass[3] = { 0.22, 0.21, 0.205 };
        int rival_rep[3] = { -1, -1, -1 };
        for (int r = 0; r < 3; r++) {
            int first_two[2] = { -1, -1 }, count = 0, total_count = 0;
            for (int i = 0; i < n; i++)
                if (same_semantic_action_test(move[i], move[first[multi[r]]])) {
                    if (count < 2) first_two[count++] = i;
                    total_count++;
                }
            fourth[first_two[0]] = (float)(rival_mass[r] * 0.5);
            fourth[first_two[1]] = (float)(rival_mass[r] * 0.5 -
                                            tiny * (total_count - 2));
            rival_rep[r] = semantic_move_pack_test(move[first_two[0]]) <
                    semantic_move_pack_test(move[first_two[1]])
                ? first_two[0] : first_two[1];
        }
        double subtotal = 0.0;
        for (int i = 0; i < n; i++) subtotal += fourth[i];
        int remainder = -1;
        for (int c = 0; c < ncore; c++) {
            if (c == selected[0] || c == multi[0] ||
                c == multi[1] || c == multi[2]) continue;
            remainder = first[c];
            break;
        }
        CHECK(remainder >= 0, "top-three fixture lacks a remainder core");
        if (remainder >= 0) {
            fourth[remainder] += (float)(1.0 - subtotal);
            CHECK(fourth[baseline_rep] > fourth[remainder],
                  "top-three fixture lost the literal complete-move argmax");
            CHECK(rollout_policy_cost_support(
                      &state, move, fourth, n, baseline_rep, &support) &&
                  support.core_candidates[0] == 4 &&
                  support.order[0][0] == baseline_rep &&
                  support.order[0][1] == rival_rep[0] &&
                  support.order[0][2] == rival_rep[1] &&
                  support.order[0][3] == rival_rep[2] &&
                  support.n[0] <= 5 && support.draw_candidates[0] <= 1,
                  "policy-cost support did not retain candidate zero plus "
                  "the true semantic top three");
        }
    }
}

static void test_full_policy_mass_and_all_pair_gates(void)
{
    PolicyCostTable table = fixture();
    for (int a = 0; a < POLICY_COST_ANCHORS; a++) {
        if (a >= 4 && a <= 6) continue;
        table.lambda_action[a] = 1.0;
        table.lambda_draw[a] = 0.0;
    }
    /* Candidate rows are 0/1/2.  Row 3 is intentionally not admitted but is
     * another draw source for row 0's semantic action.  P_A must be .95, not
     * the shortlisted .90. */
    Move legal[4] = {
        { CARD_MAKE(0, 3), 0, 0 },
        { CARD_MAKE(1, 3), 0, 0 },
        { CARD_MAKE(2, 3), 0, 0 },
        { CARD_MAKE(0, 3), 0, 1 },
    };
    float prob[4] = { 0.90f, 0.04f, 0.01f, 0.05f };
    int candidate[3] = { 0, 1, 2 };
    double pa = 0.0, pd = 0.0, cost = 0.0;
    CHECK(policy_cost_move_terms(
              &table, 0, 0, legal, prob, 4, 0, &pa, &pd, &cost) &&
          fabs(pa - 0.95) < 2e-7 &&
          fabs(pd - (0.90 / 0.95)) < 2e-7,
          "excluded draw mass was lost: PA=%.9f PD=%.9f", pa, pd);
    double pa2 = 0.0, pd2 = 0.0, cost2 = 0.0;
    CHECK(policy_cost_move_terms(
              &table, 2, 0, legal, prob, 4, 0, &pa2, &pd2, &cost2) &&
          pa2 == pa && pd2 == pd && cost2 == cost,
          "round changed the shared policy-cost schedule");

    enum { W = 8 };
    double values[3 * W];
    constant_candidate(values, W, 0, 0.0);
    constant_candidate(values, W, 1, 4.0);
    constant_candidate(values, W, 2, 6.0);
    PolicyCostDecision decision;
    CHECK(policy_cost_decide(
              &table, 0, 0, legal, prob, 4, candidate, 3,
              values, W, W, POLICY_COST_PRIMARY_Z, &decision) &&
          decision.leader == 2 && decision.selected == 2 &&
          decision.all_pair_passed &&
          decision.prior_protected_rivals == 2,
          "strong 1%% move did not clear both 95%% and 4%% plays "
          "(leader %d selected %d protected %d)",
          decision.leader, decision.selected,
          decision.prior_protected_rivals);
    double q[3] = { 0.0, 4.0, 6.0 };
    double paired_se[3][3] = { { 0 } };
    PolicyCostDecision summary;
    CHECK(policy_cost_decide_summary(
              &table, 0, 0, legal, prob, 4, candidate, 3, q,
              &paired_se[0][0], 3, POLICY_COST_PRIMARY_Z, &summary) &&
          summary.leader == decision.leader &&
          summary.selected == decision.selected &&
          summary.prior_protected_rivals ==
              decision.prior_protected_rivals &&
          summary.adjusted_q[2] == decision.adjusted_q[2],
          "summary-form selector diverged from raw fixed-world authority");
    paired_se[0][1] = 1.0;
    CHECK(!policy_cost_decide_summary(
              &table, 0, 0, legal, prob, 4, candidate, 3, q,
              &paired_se[0][0], 3, POLICY_COST_PRIMARY_Z, &summary),
          "asymmetric paired-SE summary was accepted");
    paired_se[0][1] = 0.0;

    /* Keep the same mean but add enough paired dispersion that the adjusted
     * 1%-vs-4% edge is not 3.5-SE unique.  Candidate zero must own fallback. */
    for (int d = 0; d < W; d++)
        values[2 * W + d] = d & 1 ? 8.0 : 4.0;
    CHECK(policy_cost_decide(
              &table, 0, 0, legal, prob, 4, candidate, 3,
              values, W, W, POLICY_COST_PRIMARY_Z, &decision) &&
          decision.leader == 2 && decision.selected == 0 &&
          !decision.all_pair_passed,
          "ambiguous 1%%-vs-4%% comparison escaped candidate-zero fallback");

    /* Passing only the admitted subset is not a legal replacement for the
     * full normalized policy vector. */
    CHECK(!policy_cost_decide(
              &table, 0, 0, legal, prob, 3, candidate, 3,
              values, W, W, POLICY_COST_PRIMARY_Z, &decision),
          "unnormalized shortlisted probability vector was accepted");
}

static void test_directed_raw_guard_and_strict_floor(void)
{
    PolicyCostTable table = fixture();
    for (int a = 0; a < POLICY_COST_ANCHORS; a++) {
        if (a >= 4 && a <= 6) continue;
        table.lambda_action[a] = 0.0;
        table.lambda_draw[a] = 1.0;
    }
    Move legal[3] = {
        { CARD_MAKE(0, 4), 0, 0 },
        { CARD_MAKE(1, 4), 0, 0 },
        { CARD_MAKE(0, 4), 0, 1 },
    };
    float prob[3] = { 0.20f, 0.19f, 0.61f };
    int candidate[2] = { 0, 1 };
    enum { W = 8 };
    double values[2 * W];
    constant_candidate(values, W, 0, 0.0);
    constant_candidate(values, W, 1, -0.1);
    PolicyCostDecision decision;
    CHECK(policy_cost_decide(
              &table, 1, 12, legal, prob, 3, candidate, 2,
              values, W, W, POLICY_COST_PRIMARY_Z, &decision) &&
          decision.leader == 0 && decision.selected == 0,
          "draw-only hierarchy altered the raw-search action core");

    /* Raw protection is intentionally directed by prior.  A proposal must
     * beat candidate zero and every higher-prior rival in raw Q, but a still
     * rarer move may have a small noisy raw edge without vetoing the proposal:
     * the all-pair adjusted-confidence condition remains authoritative there.
     * Requiring raw positivity against this lower-prior row would erase the
     * policy-frequency preference the controller is designed to express. */
    PolicyCostTable directed = fixture();
    for (int a = 0; a < POLICY_COST_ANCHORS; a++) {
        if (a >= 4 && a <= 6) continue;
        directed.lambda_action[a] = 1.0;
        directed.lambda_draw[a] = 0.0;
    }
    Move ordered[3] = {
        { CARD_MAKE(0, 4), 0, 0 },
        { CARD_MAKE(1, 4), 0, 0 },
        { CARD_MAKE(2, 4), 0, 0 },
    };
    float ordered_prob[3] = { 0.50f, 0.40f, 0.10f };
    int ordered_candidate[3] = { 0, 1, 2 };
    double ordered_values[3 * W];
    constant_candidate(ordered_values, W, 0, 0.0);
    constant_candidate(ordered_values, W, 1, 1.0);
    constant_candidate(ordered_values, W, 2, 1.1);
    CHECK(policy_cost_decide(
              &directed, 1, 12, ordered, ordered_prob, 3,
              ordered_candidate, 3, ordered_values, W, W,
              POLICY_COST_PRIMARY_Z, &decision) &&
          decision.leader == 1 && decision.selected == 1 &&
          decision.q[1] < decision.q[2] &&
          decision.adjusted_q[1] > decision.adjusted_q[2] &&
          decision.prior_protected_rivals == 1,
          "lower-prior noisy raw edge incorrectly vetoed the adjusted leader");

    PolicyCostTable floor = fixture();
    for (int a = 0; a < POLICY_COST_ANCHORS; a++) {
        if (a >= 4 && a <= 6) continue;
        floor.lambda_action[a] = 1.0;
        floor.lambda_draw[a] = 1.0;
    }
    Move tiny[2] = {
        { CARD_MAKE(0, 5), 0, 0 },
        { CARD_MAKE(1, 5), 0, 0 },
    };
    float tiny_prob[2] = { 1.0f, 0.0f };
    double tiny_values[2 * W];
    constant_candidate(tiny_values, W, 0, 0.0);
    constant_candidate(tiny_values, W, 1, 100.0);
    CHECK(!policy_cost_decide(
              &floor, 0, 0, tiny, tiny_prob, 2, candidate, 2,
              tiny_values, W, W, POLICY_COST_PRIMARY_Z, &decision),
          "exactly zero policy mass was silently clamped");
    tiny_prob[1] = ldexpf(1.0f, -149);
    double pa = 0.0, pd = 0.0, cost = 0.0;
    CHECK(policy_cost_move_terms(
              &floor, 0, 0, tiny, tiny_prob, 2, 1,
              &pa, &pd, &cost) && pa > POLICY_COST_EPSILON &&
          pd > POLICY_COST_EPSILON && isfinite(cost),
          "least positive binary32 policy mass was not admitted");
}

static void test_crossed_core_and_complete_mass_guard(void)
{
    PolicyCostTable table = fixture();
    for (int a = 0; a < POLICY_COST_ANCHORS; a++) {
        if (a >= 4 && a <= 6) continue;
        table.lambda_action[a] = 1.0;
        table.lambda_draw[a] = 0.0;
    }
    /* The proposed wager core has 50% semantic action mass, but this complete
     * move has only 1% after summing two physical wager copies.  The rival's
     * core is only 4%, all on one complete move.  A core-only guard would let
     * the 1% move regress raw Q against the 4% move when lambda_action is much
     * stronger than lambda_draw; the user's complete-play ordering forbids
     * that crossed-factor escape. */
    Move legal[5] = {
        { CARD_MAKE(0, 0), 0, 0 },
        { CARD_MAKE(0, 1), 0, 0 },
        { CARD_MAKE(0, 0), 0, 1 },
        { CARD_MAKE(1, 4), 0, 0 },
        { CARD_MAKE(2, 4), 0, 0 },
    };
    float prob[5] = { 0.004f, 0.006f, 0.49f, 0.04f, 0.46f };
    int candidate[3] = { 2, 0, 3 };
    enum { W = 8 };
    double values[3 * W];
    constant_candidate(values, W, 0, 0.0);
    constant_candidate(values, W, 1, 1.0);
    constant_candidate(values, W, 2, 2.0);
    PolicyCostDecision decision;
    CHECK(policy_cost_decide(
              &table, 0, 0, legal, prob, 5, candidate, 3,
              values, W, W, POLICY_COST_PRIMARY_Z, &decision) &&
          decision.leader == 1 && decision.selected == 0 &&
          decision.semantic_prior[1] > decision.semantic_prior[2] &&
          decision.semantic_prior[1] *
                  decision.conditional_draw_prior[1] <
              decision.semantic_prior[2] *
                  decision.conditional_draw_prior[2],
          "higher joint play escaped crossed-factor raw protection");
}

static int parse_fails(const char *spec)
{
    fflush(NULL);
    pid_t child = fork();
    if (child == 0) {
        int nullfd = open("/dev/null", O_WRONLY);
        if (nullfd >= 0) {
            (void)dup2(nullfd, STDERR_FILENO);
            close(nullfd);
        }
        Agent actor;
        spec_parse(spec, &actor);
        spec_release(&actor);
        _exit(0);
    }
    if (child < 0) return 0;
    int status = 0;
    if (waitpid(child, &status, 0) != child) return 0;
    return !WIFEXITED(status) || WEXITSTATUS(status) != 0;
}

static void test_rolloutu5_header_and_guards(void)
{
    Net *net = malloc(sizeof *net);
    CHECK(net != NULL && net_load(net, "data/champion.bin") == 0,
          "cannot load rollout5 parser checkpoint");
    if (!net) return;
    const char *neutral_objective0_tail =
        "800:5:0.01:0:1:0:0:0:0:0:3.5:0:4:20:0:0:20:1:0:800:1:"
        "0:0:0:0:0:0:0:1:0:0:0:0:0:3:1:0:0:0:1";
    char neutral_spec[512];
    snprintf(neutral_spec, sizeof neutral_spec,
             "rolloutu2:data/champion.bin:data/champion.bin:%s",
             neutral_objective0_tail);
    Agent neutral;
    spec_parse(neutral_spec, &neutral);
    CHECK(neutral.kind == AG_ROLLOUT && neutral.no_belief &&
          neutral.dets == 800 && neutral.confirm_dets == 800 &&
          neutral.root_width == 5 && neutral.action_core_count == 3 &&
          neutral.action_ranker_net == NULL && neutral.match_value == NULL &&
          neutral.action_ranker_min == 0.0f,
          "canonical 40-field objective-0 neutral actor did not parse");
    spec_release(&neutral);
    PolicyCostTable table = fixture();
    table.controller.root_net_fingerprint = match_value_net_fingerprint(net);
    table.controller.continuation_net_fingerprint =
        table.controller.root_net_fingerprint;
    char path[128];
    temporary_path(path, "parser");
    CHECK(policy_cost_save(&table, path) == 0,
          "cannot save rollout5 parser artifact");
    const char *tail =
        "800:5:0.01:0:1:0:0:0:0:0:3.5:0:4:20:0:0:20:1:0:800:1:"
        "0:0:0:0:0:0:0:0:0:0:0:0:0:3:1";
    char spec[512];
    snprintf(spec, sizeof spec,
             "rolloutu5:data/champion.bin:data/champion.bin:%s:%s",
             path, tail);
    CHECK(strlen(spec) < 511, "rollout5 spec exceeds parser ABI");
    Agent actor;
    spec_parse(spec, &actor);
    CHECK(actor.kind == AG_ROLLOUT && actor.no_belief && actor.policy_cost &&
          actor.owns_policy_cost && actor.net == actor.continuation_net &&
          actor.action_ranker_net == NULL &&
          actor.veto_continuation_net == NULL,
          "rolloutu5 header did not bind exact root/continuation/cost roles");
    Rng deal_rng, move_rng;
    rng_seed(&deal_rng, UINT64_C(0x5052434f53545254));
    rng_seed(&move_rng, UINT64_C(0x465245534850414e));
    State state;
    lc_deal(&state, &deal_rng);
    Move policy_move[MAX_MOVES];
    float policy_prob[MAX_MOVES], policy_value = 0.0f;
    int npolicy = policy_probs_sym(
        actor.net, &state, policy_move, policy_prob, &policy_value,
        actor.symmetries);
    int literal_top = 0;
    for (int i = 1; i < npolicy; i++)
        if (policy_prob[i] > policy_prob[literal_top]) literal_top = i;
    int shortlist[5], ncore = 0, ndraw = 0;
    int nshort = rollout_action_core_indices(
        &state, policy_move, policy_prob, npolicy, literal_top,
        actor.root_width, actor.action_core_count, actor.min_cand,
        actor.cand_floor, actor.cand_mass, shortlist, &ncore, &ndraw);
    CHECK(nshort >= 1 && nshort <= 5 && shortlist[0] == literal_top &&
          ncore >= 1 && ncore <= actor.action_core_count,
          "public hierarchical shortlist did not preserve literal argmax");
    float damaged[MAX_MOVES];
    memcpy(damaged, policy_prob, sizeof(float) * (size_t)npolicy);
    damaged[0] *= 0.5f;
    CHECK(rollout_action_core_indices(
              &state, policy_move, damaged, npolicy, literal_top,
              actor.root_width, actor.action_core_count, actor.min_cand,
              actor.cand_floor, actor.cand_mass, shortlist, &ncore, &ndraw)
              == -1,
          "hierarchical shortlist accepted a subset/unnormalized policy");

    Move audit_move[5];
    for (int c = 0; c < nshort; c++) audit_move[c] = policy_move[shortlist[c]];
    RolloutAuditPanel audit;
    CHECK(rollout_audit_panel(
              &actor, &state, audit_move, nshort, 0,
              UINT64_C(0x414c4c5041495253), 4, &audit) == 0,
          "all-pair audit panel failed");
    for (int c = 0; c < nshort; c++)
        for (int rival = 0; rival < nshort; rival++) {
            CHECK(fabs(audit.pair_delta[c][rival] +
                       audit.pair_delta[rival][c]) < 1e-12 &&
                  audit.pair_delta_se[c][rival] ==
                      audit.pair_delta_se[rival][c],
                  "audit all-pair matrix lost antisymmetry (%d,%d)",
                  c, rival);
            if (rival == 0)
                CHECK(audit.pair_delta[c][0] == audit.delta[c] &&
                      audit.pair_delta_se[c][0] == audit.delta_se[c],
                      "audit baseline column changed legacy delta (%d)", c);
        }

    Move union_move[ROLLOUT_MAX_CANDIDATES];
    int nunion = 0;
    for (int i = 0; i < npolicy && nunion < ROLLOUT_MAX_CANDIDATES; i++) {
        int duplicate = 0;
        for (int c = 0; c < nunion; c++)
            if (same_complete_semantic_move(policy_move[i], union_move[c])) {
                duplicate = 1;
                break;
            }
        if (!duplicate) union_move[nunion++] = policy_move[i];
    }
    CHECK(nunion == ROLLOUT_MAX_CANDIDATES &&
          rollout_audit_panel(
              &actor, &state, union_move, nunion, 0,
              UINT64_C(0x554e494f4e50414e), 2, &audit) == 0 &&
          audit.n == ROLLOUT_MAX_CANDIDATES,
          "diagnostic panel did not admit an eight-row mask union");
    CHECK(rollout_audit_panel(
              &actor, &state, union_move, ROLLOUT_MAX_CANDIDATES + 1, 0,
              UINT64_C(0x554e494f4e50414e), 2, &audit) == -1,
          "diagnostic panel accepted more than eight rows");

    /* PRIMARY and FRESH are separate frozen finite-support domains.  Repeat
     * calls within a role must be bit-stable; the same external seed must not
     * collapse the trusted panel back onto PRIMARY's assignment order. */
    State late = state;
    while (!late.over && late.deck_left > 3) {
        Move legal[MAX_MOVES];
        int nlegal = lc_moves(&late, legal), deck_move = -1;
        for (int i = 0; i < nlegal; i++)
            if (legal[i].draw == 0) { deck_move = i; break; }
        if (deck_move < 0) break;
        lc_apply(&late, legal[deck_move]);
    }
    Move late_legal[MAX_MOVES], late_candidate[2];
    int nlate = lc_moves(&late, late_legal), nlate_candidate = 0;
    for (int i = 0; i < nlate && nlate_candidate < 2; i++) {
        int duplicate = 0;
        for (int c = 0; c < nlate_candidate; c++)
            if (same_complete_semantic_move(
                    late_legal[i], late_candidate[c])) duplicate = 1;
        if (!duplicate) late_candidate[nlate_candidate++] = late_legal[i];
    }
    RolloutAuditPanel primary_a, primary_b, fresh_a, fresh_b;
    const uint64_t role_seed = UINT64_C(0x50414e454c524f4c);
    CHECK(late.deck_left == 3 && nlate_candidate == 2,
          "cannot construct finite-support audit fixture");
    CHECK(rollout_audit_panel_role(
              &actor, &late, late_candidate, nlate_candidate, 0,
              ROLLOUT_AUDIT_PANEL_PRIMARY, role_seed, 8, &primary_a) == 0 &&
          rollout_audit_panel_role(
              &actor, &late, late_candidate, nlate_candidate, 0,
              ROLLOUT_AUDIT_PANEL_PRIMARY, role_seed, 8, &primary_b) == 0 &&
          rollout_audit_panel_role(
              &actor, &late, late_candidate, nlate_candidate, 0,
              ROLLOUT_AUDIT_PANEL_FRESH, role_seed, 8, &fresh_a) == 0 &&
          rollout_audit_panel_role(
              &actor, &late, late_candidate, nlate_candidate, 0,
              ROLLOUT_AUDIT_PANEL_FRESH, role_seed, 8, &fresh_b) == 0,
          "role-explicit finite-support audit failed");
    CHECK(primary_a.panel_role == ROLLOUT_AUDIT_PANEL_PRIMARY &&
          fresh_a.panel_role == ROLLOUT_AUDIT_PANEL_FRESH &&
          primary_a.hidden_support > 8 && fresh_a.hidden_support > 8,
          "audit evidence did not bind role/support");
    CHECK(primary_a.hidden_world_fingerprint ==
              primary_b.hidden_world_fingerprint &&
          fresh_a.hidden_world_fingerprint ==
              fresh_b.hidden_world_fingerprint &&
          primary_a.q[0] == primary_b.q[0] &&
          fresh_a.q[0] == fresh_b.q[0],
          "audit role was not deterministic");
    CHECK(primary_a.hidden_world_fingerprint !=
              fresh_a.hidden_world_fingerprint,
          "PRIMARY and FRESH reused one finite-support assignment order");
    CHECK(rollout_audit_panel_role(
              &actor, &late, late_candidate, nlate_candidate, 0,
              99, role_seed, 8, &audit) == -1,
          "audit accepted an unknown evidence role");
    actor.no_belief = 0;
    CHECK(rollout_audit_panel_role(
              &actor, &late, late_candidate, nlate_candidate, 0,
              ROLLOUT_AUDIT_PANEL_FRESH, role_seed, 8, &audit) == -2,
          "FRESH audit accepted an unbound learned-belief distribution");
    actor.no_belief = 1;

    SearchStats stats;
    Move selected = rollout_move(&actor, &state, &move_rng, NULL, &stats);
    CHECK(stats.policy_cost_active && stats.policy_cost_fingerprint != 0 &&
          stats.policy_cost_anchor_interval == 0 &&
          stats.policy_cost_override_min == 0.0 &&
          stats.policy_cost_primary_valid && stats.policy_cost_cap_valid &&
          stats.policy_cost_gate_reason != POLICY_COST_GATE_NONE &&
          stats.confirm_worlds == 0 &&
          stats.policy_cost_selected >= 0 &&
          stats.policy_cost_selected < stats.n &&
          MOVE_PACK(selected) ==
              MOVE_PACK(stats.mv[stats.policy_cost_selected]),
          "production rollout root did not report policy-cost authority");
    actor.prune_dom = 1;
    rng_seed(&move_rng, UINT64_C(0x465245534850414e));
    selected = rollout_move(&actor, &state, &move_rng, NULL, &stats);
    CHECK(stats.worlds == 0 && !stats.policy_cost_active &&
          MOVE_PACK(selected) == MOVE_PACK(policy_move[literal_top]),
          "programmatic incompatible policy-cost actor did not fail closed");
    actor.prune_dom = 0;
    spec_release(&actor);
    CHECK(!actor.policy_cost && !actor.owns_policy_cost && !actor.net,
          "rolloutu5 release left an owned artifact or checkpoint live");

    /* Objective 3 shifts only the unchanged tail's optional field 41.  Both
     * the policy-cost artifact and actor must bind the exact checked table. */
    char match_path[128], objective_path[128];
    temporary_path(match_path, "match");
    temporary_path(objective_path, "objective3");
    MatchValueTable *match = match_fixture(
        match_value_net_fingerprint(net));
    CHECK(match && match_value_validate(match) &&
          match_value_save(match, match_path) == 0,
          "cannot save objective-3 match-value fixture");
    int match_error = 0;
    MatchValueTable *checked_match = match_value_load(match_path, &match_error);
    PolicyCostTable objective = table;
    objective.controller.objective = 3;
    objective.controller.match_value_fingerprint = checked_match
        ? checked_match->payload_fingerprint : 0;
    CHECK(checked_match && match_error == 0 &&
          policy_cost_save(&objective, objective_path) == 0,
          "cannot bind objective-3 policy-cost fixture");
    const char *objective_tail =
        "800:5:0.01:0:1:0:0:0:3:0:3.5:0:4:20:0:0:20:1:0:800:1:"
        "0:0:0:0:0:0:0:1:0:0:0:0:0:3:1:0:0:0:1:0";
    snprintf(spec, sizeof spec,
             "rolloutu5:data/champion.bin:data/champion.bin:%s:%s:%s",
             objective_path, objective_tail, match_path);
    CHECK(strlen(spec) < 511,
          "objective-3 rollout5 spec exceeds parser ABI");
    spec_parse(spec, &actor);
    CHECK(actor.policy_cost && actor.match_value && actor.win_q == 3 &&
          actor.policy_cost->controller.match_value_fingerprint ==
              actor.match_value->payload_fingerprint,
          "rollout5 did not bind optional objective-3 match-value table");
    MatchValueTable *mutable_match = (MatchValueTable *)actor.match_value;
    uint64_t bound_net = mutable_match->controller.net_fingerprint;
    mutable_match->controller.net_fingerprint ^= UINT64_C(1);
    CHECK(!policy_cost_matches_agent(&actor),
          "programmatic rollout5 accepted an objective-3 table for a "
          "different continuation controller");
    mutable_match->controller.net_fingerprint = bound_net;
    CHECK(policy_cost_matches_agent(&actor),
          "restoring objective-3 controller binding did not restore actor");
    spec_release(&actor);
    match_value_free(checked_match);
    match_value_free(match);
    (void)remove(match_path);
    (void)remove(objective_path);

    /* Field 27 enables the legacy trusted-prefix selector.  rollout5 must
     * reject it rather than composing two authorities. */
    const char *incompatible_tail =
        "800:5:0.01:0:1:0:0:0:0:0:3.5:0:4:20:0:0:20:1:0:800:1:"
        "0:0:0:0:0:0:3:0:0:0:0:0:0:3:1";
    snprintf(spec, sizeof spec,
             "rolloutu5:data/champion.bin:data/champion.bin:%s:%s",
             path, incompatible_tail);
    CHECK(parse_fails(spec),
          "rolloutu5 accepted an incompatible legacy prefix controller");
    snprintf(spec, sizeof spec,
             "rollout5:data/champion.bin:data/champion.bin:%s:%s",
             path, tail);
    CHECK(parse_fails(spec),
          "rollout5 accepted a policy-cost artifact without uniform belief");
    const char *legacy_floor_tail =
        "800:5:0.01:0:1:0:0:0:0:0:3.5:2:4:20:0:0:20:1:0:800:1:"
        "0:0:0:0:0:0:0:0:0:0:0:0:0:3:1";
    snprintf(spec, sizeof spec,
             "rolloutu5:data/champion.bin:data/champion.bin:%s:%s",
             path, legacy_floor_tail);
    CHECK(parse_fails(spec),
          "rolloutu5 accepted the legacy low-prior practical floor");
    (void)remove(path);
    free(net);
}

int main(int argc, char **argv)
{
    if (argc == 3 && !strcmp(argv[1], "--emit")) {
        PolicyCostTable table = fixture();
        return policy_cost_save(&table, argv[2]) == 0 ? 0 : 1;
    }
    if (argc == 3 && !strcmp(argv[1], "--emit-legacy")) {
        PolicyCostTable table = fixture();
        table.version = POLICY_COST_LEGACY_VERSION;
        table.source_seed = POLICY_COST_LEGACY_SOURCE_SEED;
        return policy_cost_save(&table, argv[2]) == 0 ? 0 : 1;
    }
    if (argc != 1) return 2;
    test_schedule_and_format();
    test_wager_semantic_complete_mass();
    test_full_policy_mass_and_all_pair_gates();
    test_directed_raw_guard_and_strict_floor();
    test_crossed_core_and_complete_mass_guard();
    test_policy_cost_master_and_no_refill_mask();
    test_rolloutu5_header_and_guards();
    if (failures) {
        printf("%d policy-cost test(s) failed\n", failures);
        return 1;
    }
    printf("policy-cost tests passed\n");
    return 0;
}
