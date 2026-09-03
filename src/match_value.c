#include "match_value.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char MATCH_VALUE_MAGIC[8] = {
    'L', 'C', 'M', 'V', 'A', 'L', '1', '\0'
};

enum {
    MATCH_VALUE_HEADER_BYTES = 128,
    MATCH_VALUE_CONTROLLER_WORDS = 11
};

_Static_assert(sizeof(double) == 8,
               "match-value artifacts require IEEE-style binary64 doubles");
_Static_assert(DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "match-value artifacts require binary64 semantics");

static uint64_t fingerprint_update(uint64_t hash,
                                   const unsigned char *data, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t match_value_net_fingerprint(const Net *net)
{
    if (!net) return 0;
    return fingerprint_update(UINT64_C(1469598103934665603),
                              (const unsigned char *)net, sizeof *net);
}

uint32_t match_value_build_profile(void)
{
    /* Controller trajectories can change at an argmax when floating-point
     * code generation changes.  Bind the artifact to the relevant compiler,
     * math, contraction/SIMD proxy, and evaluation-width profile instead of
     * pretending the network bytes alone define the controller. */
    uint32_t profile = 0;
#ifdef __FAST_MATH__
    profile |= UINT32_C(1) << 0;
#endif
#ifdef __FMA__
    profile |= UINT32_C(1) << 1;
#endif
#ifdef __AVX512F__
    profile |= UINT32_C(1) << 2;
#endif
#ifdef __AVX2__
    profile |= UINT32_C(1) << 3;
#endif
#ifdef __AVX__
    profile |= UINT32_C(1) << 4;
#endif
#ifdef __SSE4_2__
    profile |= UINT32_C(1) << 5;
#endif
#ifdef __ARM_NEON
    profile |= UINT32_C(1) << 6;
#endif
#ifdef __clang__
    profile |= UINT32_C(1) << 8;
    profile |= ((uint32_t)__clang_major__ & UINT32_C(0xff)) << 12;
    profile |= ((uint32_t)__clang_minor__ & UINT32_C(0xff)) << 20;
#elif defined(__GNUC__)
    profile |= UINT32_C(2) << 8;
    profile |= ((uint32_t)__GNUC__ & UINT32_C(0xff)) << 12;
    profile |= ((uint32_t)__GNUC_MINOR__ & UINT32_C(0xff)) << 20;
#else
    profile |= UINT32_C(15) << 8;
#endif
#ifdef __FLT_EVAL_METHOD__
    profile |= ((uint32_t)__FLT_EVAL_METHOD__ & UINT32_C(0x3)) << 28;
#endif
    return profile;
}

int match_value_controller_supported(const MatchValueController *c)
{
    if (!c || c->net_fingerprint == 0 ||
        c->controller_abi != MATCH_VALUE_CONTROLLER_ABI ||
        c->build_profile != match_value_build_profile() ||
        c->objective > 2 ||
        c->playout_sample != 4 ||
        (c->playout_symmetries != 5 && c->playout_symmetries != 10 &&
         c->playout_symmetries != 20 && c->playout_symmetries != 120) ||
        c->playout_prune > 1 || c->exact_terminal != 1 ||
        c->plan_deck_max > NCARD || c->draw_playout_deck_max > NCARD ||
        c->plan_block_gap > 1000000U ||
        c->deck2_replan_worlds != 0 || c->deck2_replan_cores != 0 ||
        c->max_plies != LC_MAX_PLIES)
        return 0;
    return 1;
}

int match_value_controller_equal(const MatchValueController *a,
                                 const MatchValueController *b)
{
    return a && b &&
           a->net_fingerprint == b->net_fingerprint &&
           a->controller_abi == b->controller_abi &&
           a->build_profile == b->build_profile &&
           a->objective == b->objective &&
           a->playout_symmetries == b->playout_symmetries &&
           a->playout_sample == b->playout_sample &&
           a->playout_prune == b->playout_prune &&
           a->exact_terminal == b->exact_terminal &&
           a->plan_deck_max == b->plan_deck_max &&
           a->plan_block_gap == b->plan_block_gap &&
           a->draw_playout_deck_max == b->draw_playout_deck_max &&
           a->deck2_replan_worlds == b->deck2_replan_worlds &&
           a->deck2_replan_cores == b->deck2_replan_cores &&
           a->max_plies == b->max_plies;
}

static int finite_double(double x)
{
    return lc_double_isfinite(x);
}

static int validate_values(const double *not_starting,
                           const double *starting, int limit,
                           int require_monotone)
{
    const double tolerance = 1e-10;
    int count = 2 * limit + 1;
    for (int i = 0; i < count; i++) {
        if (!finite_double(not_starting[i]) ||
            !finite_double(starting[i]) ||
            fabs(not_starting[i]) > MATCH_VALUE_MAX_ABS_UTILITY ||
            fabs(starting[i]) > MATCH_VALUE_MAX_ABS_UTILITY)
            return 0;
        if (require_monotone && i > 0 &&
            (not_starting[i] + tolerance < not_starting[i - 1] ||
             starting[i] + tolerance < starting[i - 1]))
            return 0;
        int mirror = count - 1 - i;
        if (fabs(not_starting[i] + starting[mirror]) > tolerance)
            return 0;
    }
    return 1;
}

int match_value_balanced_roles(const MatchValueTable *table)
{
    if (!table || !match_value_controller_supported(&table->controller))
        return 0;
    uint32_t symmetries = table->controller.playout_symmetries;
    uint32_t cycle = symmetries * symmetries;
    return table->role_cycle_size == cycle &&
           table->role_balance_complete == 1U &&
           table->samples_per_policy_lead % cycle == 0;
}

int match_value_validate(const MatchValueTable *table)
{
    if (!table || table->version != MATCH_VALUE_VERSION ||
        table->samples_per_policy_lead == 0 ||
        !match_value_controller_supported(&table->controller))
        return 0;
    uint32_t symmetries = table->controller.playout_symmetries;
    uint32_t expected_cycle = symmetries * symmetries;
    int expected_complete =
        table->samples_per_policy_lead % expected_cycle == 0;
    return
           table->role_cycle_size == expected_cycle &&
           table->role_balance_complete == (uint32_t)expected_complete &&
           table->isotonic_projected <= 1U &&
           finite_double(table->max_isotonic_adjustment[0]) &&
           finite_double(table->max_isotonic_adjustment[1]) &&
           table->max_isotonic_adjustment[0] >= 0.0 &&
           table->max_isotonic_adjustment[1] >= 0.0 &&
           table->max_isotonic_adjustment[0] <=
               2.0 * MATCH_VALUE_MAX_ABS_UTILITY &&
           table->max_isotonic_adjustment[1] <=
               2.0 * MATCH_VALUE_MAX_ABS_UTILITY &&
           validate_values(table->before_round1[0],
                           table->before_round1[1],
                           MATCH_VALUE_R1_LEAD_LIMIT,
                           table->isotonic_projected != 0) &&
           validate_values(table->before_round2[0],
                           table->before_round2[1],
                           MATCH_VALUE_R2_LEAD_LIMIT,
                           table->isotonic_projected != 0);
}

static void encode_u32(unsigned char out[4], uint32_t x)
{
    out[0] = (unsigned char)x;
    out[1] = (unsigned char)(x >> 8);
    out[2] = (unsigned char)(x >> 16);
    out[3] = (unsigned char)(x >> 24);
}

static void encode_u64(unsigned char out[8], uint64_t x)
{
    for (int i = 0; i < 8; i++) out[i] = (unsigned char)(x >> (8 * i));
}

static uint32_t decode_u32(const unsigned char in[4])
{
    return (uint32_t)in[0] | (uint32_t)in[1] << 8 |
           (uint32_t)in[2] << 16 | (uint32_t)in[3] << 24;
}

static uint64_t decode_u64(const unsigned char in[8])
{
    uint64_t x = 0;
    for (int i = 0; i < 8; i++) x |= (uint64_t)in[i] << (8 * i);
    return x;
}

static uint64_t double_bits(double x)
{
    uint64_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits;
}

static double bits_double(uint64_t bits)
{
    double x;
    memcpy(&x, &bits, sizeof x);
    return x;
}

static void header_store_controller(unsigned char *h,
                                    const MatchValueController *c)
{
    encode_u64(h + 32, c->net_fingerprint);
    const uint32_t word[MATCH_VALUE_CONTROLLER_WORDS] = {
        c->objective, c->playout_symmetries, c->playout_sample,
        c->playout_prune, c->exact_terminal, c->plan_deck_max,
        c->plan_block_gap, c->draw_playout_deck_max,
        c->deck2_replan_worlds, c->deck2_replan_cores, c->max_plies
    };
    for (int i = 0; i < MATCH_VALUE_CONTROLLER_WORDS; i++)
        encode_u32(h + 40 + 4 * i, word[i]);
}

static void header_load_controller(const unsigned char *h,
                                   MatchValueController *c)
{
    memset(c, 0, sizeof *c);
    c->net_fingerprint = decode_u64(h + 32);
    uint32_t *word[MATCH_VALUE_CONTROLLER_WORDS] = {
        &c->objective, &c->playout_symmetries, &c->playout_sample,
        &c->playout_prune, &c->exact_terminal, &c->plan_deck_max,
        &c->plan_block_gap, &c->draw_playout_deck_max,
        &c->deck2_replan_worlds, &c->deck2_replan_cores, &c->max_plies
    };
    for (int i = 0; i < MATCH_VALUE_CONTROLLER_WORDS; i++)
        *word[i] = decode_u32(h + 40 + 4 * i);
}

static int write_values(FILE *f, const double *value, size_t count,
                        uint64_t *fingerprint)
{
    unsigned char bytes[8];
    for (size_t i = 0; i < count; i++) {
        encode_u64(bytes, double_bits(value[i]));
        *fingerprint = fingerprint_update(*fingerprint, bytes, sizeof bytes);
        if (fwrite(bytes, 1, sizeof bytes, f) != sizeof bytes) return 0;
    }
    return 1;
}

static int read_values(FILE *f, double *value, size_t count,
                       uint64_t *fingerprint)
{
    unsigned char bytes[8];
    for (size_t i = 0; i < count; i++) {
        if (fread(bytes, 1, sizeof bytes, f) != sizeof bytes) return 0;
        *fingerprint = fingerprint_update(*fingerprint, bytes, sizeof bytes);
        value[i] = bits_double(decode_u64(bytes));
    }
    return 1;
}

int match_value_save(const MatchValueTable *table, const char *path)
{
    if (!path || !*path || !match_value_validate(table)) return -1;
    FILE *f = fopen(path, "wbx");
    if (!f) return -2;

    unsigned char h[MATCH_VALUE_HEADER_BYTES];
    memset(h, 0, sizeof h);
    memcpy(h, MATCH_VALUE_MAGIC, sizeof MATCH_VALUE_MAGIC);
    encode_u32(h + 8, MATCH_VALUE_VERSION);
    encode_u32(h + 12, MATCH_VALUE_HEADER_BYTES);
    encode_u32(h + 16, table->samples_per_policy_lead);
    encode_u32(h + 20, MATCH_VALUE_R1_COUNT);
    encode_u32(h + 24, MATCH_VALUE_R2_COUNT);
    encode_u32(h + 28, MATCH_VALUE_POLICY_LEAD_LIMIT);
    header_store_controller(h, &table->controller);
    encode_u64(h + 84, table->source_seed);
    encode_u64(h + 92, double_bits(table->max_isotonic_adjustment[0]));
    encode_u64(h + 100, double_bits(table->max_isotonic_adjustment[1]));
    encode_u32(h + 108, table->role_cycle_size);
    encode_u32(h + 112, table->role_balance_complete);
    encode_u32(h + 116, table->isotonic_projected);
    encode_u32(h + 120, table->controller.controller_abi);
    encode_u32(h + 124, table->controller.build_profile);

    uint64_t fingerprint = fingerprint_update(
        UINT64_C(1469598103934665603), h, sizeof h);
    int ok = fwrite(h, 1, sizeof h, f) == sizeof h &&
             write_values(f, &table->before_round1[0][0],
                          2U * MATCH_VALUE_R1_COUNT, &fingerprint) &&
             write_values(f, &table->before_round2[0][0],
                          2U * MATCH_VALUE_R2_COUNT, &fingerprint);
    unsigned char footer[8];
    encode_u64(footer, fingerprint);
    ok = ok && fwrite(footer, 1, sizeof footer, f) == sizeof footer;
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        (void)remove(path);
        return -3;
    }
    return 0;
}

MatchValueTable *match_value_load(const char *path, int *error)
{
    if (error) *error = -1;
    if (!path || !*path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char h[MATCH_VALUE_HEADER_BYTES];
    if (fread(h, 1, sizeof h, f) != sizeof h ||
        memcmp(h, MATCH_VALUE_MAGIC, sizeof MATCH_VALUE_MAGIC) != 0 ||
        decode_u32(h + 8) != MATCH_VALUE_VERSION ||
        decode_u32(h + 12) != MATCH_VALUE_HEADER_BYTES ||
        decode_u32(h + 20) != MATCH_VALUE_R1_COUNT ||
        decode_u32(h + 24) != MATCH_VALUE_R2_COUNT ||
        decode_u32(h + 28) != MATCH_VALUE_POLICY_LEAD_LIMIT) {
        fclose(f);
        if (error) *error = -2;
        return NULL;
    }

    MatchValueTable *table = (MatchValueTable *)calloc(1, sizeof *table);
    if (!table) {
        fclose(f);
        if (error) *error = -3;
        return NULL;
    }
    table->version = decode_u32(h + 8);
    table->samples_per_policy_lead = decode_u32(h + 16);
    table->source_seed = decode_u64(h + 84);
    table->role_cycle_size = decode_u32(h + 108);
    table->role_balance_complete = decode_u32(h + 112);
    table->isotonic_projected = decode_u32(h + 116);
    table->max_isotonic_adjustment[0] =
        bits_double(decode_u64(h + 92));
    table->max_isotonic_adjustment[1] =
        bits_double(decode_u64(h + 100));
    header_load_controller(h, &table->controller);
    table->controller.controller_abi = decode_u32(h + 120);
    table->controller.build_profile = decode_u32(h + 124);

    uint64_t fingerprint = fingerprint_update(
        UINT64_C(1469598103934665603), h, sizeof h);
    unsigned char footer[8];
    int ok = read_values(f, &table->before_round1[0][0],
                         2U * MATCH_VALUE_R1_COUNT, &fingerprint) &&
             read_values(f, &table->before_round2[0][0],
                         2U * MATCH_VALUE_R2_COUNT, &fingerprint) &&
             fread(footer, 1, sizeof footer, f) == sizeof footer &&
             decode_u64(footer) == fingerprint && fgetc(f) == EOF;
    if (fclose(f) != 0) ok = 0;
    table->payload_fingerprint = fingerprint;
    if (!ok || !match_value_validate(table)) {
        free(table);
        if (error) *error = -4;
        return NULL;
    }
    if (error) *error = 0;
    return table;
}

void match_value_free(MatchValueTable *table)
{
    free(table);
}

static double final_utility(int lead)
{
    int result = (lead > 0) - (lead < 0);
    return 0.05 * (double)lead + 50.0 * (double)result;
}

int match_value_terminal(const MatchValueTable *table, const State *terminal,
                         int p, double *value)
{
    /* Loaded tables are fully validated once.  Re-walking fourteen thousand
     * entries at every rollout leaf would make the supposedly constant-time
     * Bellman lookup more expensive than the policy continuation itself. */
    if (!table || !terminal || !value ||
        table->version != MATCH_VALUE_VERSION ||
        table->samples_per_policy_lead == 0 ||
        !match_value_controller_supported(&table->controller) ||
        (p != 0 && p != 1) || !terminal->over || terminal->deck_left != 0 ||
        terminal->round >= MATCH_ROUNDS)
        return 0;
    int o = p ^ 1;
    int lead = (int)terminal->cum[p] - (int)terminal->cum[o]
             + lc_score(terminal, p) - lc_score(terminal, o);
    if (terminal->round == MATCH_ROUNDS - 1) {
        if (lead < -MATCH_VALUE_MAX_MATCH_MARGIN ||
            lead > MATCH_VALUE_MAX_MATCH_MARGIN)
            return 0;
        *value = final_utility(lead);
        return 1;
    }
    int next_round = terminal->round + 1;
    int starts = p == (next_round & 1);
    if (next_round == 1) {
        if (lead < -MATCH_VALUE_R1_LEAD_LIMIT ||
            lead > MATCH_VALUE_R1_LEAD_LIMIT)
            return 0;
        double selected = table->before_round1[starts]
            [lead + MATCH_VALUE_R1_LEAD_LIMIT];
        if (!finite_double(selected) ||
            fabs(selected) > MATCH_VALUE_MAX_ABS_UTILITY)
            return 0;
        *value = selected;
        return 1;
    }
    if (lead < -MATCH_VALUE_R2_LEAD_LIMIT ||
        lead > MATCH_VALUE_R2_LEAD_LIMIT)
        return 0;
    double selected = table->before_round2[starts]
        [lead + MATCH_VALUE_R2_LEAD_LIMIT];
    if (!finite_double(selected) ||
        fabs(selected) > MATCH_VALUE_MAX_ABS_UTILITY)
        return 0;
    *value = selected;
    return 1;
}
