#include "policy_cost_v20.h"
#include "agent.h"
#include "match_value.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char POLICY_COST_MAGIC[8] = {
    'L', 'C', 'P', 'C', 'O', 'S', '1', '\0'
};

enum {
    POLICY_COST_HEADER_BYTES = 256,
    POLICY_COST_CONTROLLER_WORDS = 18,
    POLICY_COST_LEGACY_PAYLOAD_DOUBLES = POLICY_COST_ANCHORS * 2,
    POLICY_COST_PAYLOAD_DOUBLES = POLICY_COST_ANCHORS * 3
};

_Static_assert(sizeof(double) == 8,
               "policy-cost artifacts require binary64 doubles");
_Static_assert(DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "policy-cost artifacts require binary64 semantics");

static int finite_double(double x)
{
    return lc_double_isfinite(x);
}

static uint64_t fingerprint_update(uint64_t hash,
                                   const unsigned char *data, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
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

static int bytes_are_zero(const unsigned char *data, size_t size)
{
    for (size_t i = 0; i < size; i++)
        if (data[i] != 0) return 0;
    return 1;
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

static uint32_t float_bits(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits;
}

static float bits_float(uint32_t bits)
{
    float x;
    memcpy(&x, &bits, sizeof x);
    return x;
}

static int same_semantic_action(Move a, Move b)
{
    if (a.discard != b.discard) return 0;
    if (CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card))
        return CARD_SUIT(a.card) == CARD_SUIT(b.card);
    return a.card == b.card;
}

static int match_value_matches_policy_cost_agent(const struct Agent *a)
{
    if (!a || !a->match_value || !a->continuation_net ||
        !match_value_validate(a->match_value) ||
        !match_value_balanced_roles(a->match_value))
        return 0;
    const MatchValueController *c = &a->match_value->controller;
    int playout_prune = a->playout_prune < 0
        ? a->prune_dom : a->playout_prune != 0;
    return c->net_fingerprint ==
               match_value_net_fingerprint(a->continuation_net) &&
           c->playout_symmetries == (uint32_t)a->playout_symmetries &&
           c->playout_sample == (uint32_t)a->playout_sample &&
           c->playout_prune == (uint32_t)playout_prune &&
           c->exact_terminal == (uint32_t)a->exact_terminal &&
           c->plan_deck_max == (uint32_t)a->plan_deck_max &&
           c->plan_block_gap == (uint32_t)a->plan_block_gap &&
           c->draw_playout_deck_max ==
               (uint32_t)a->draw_playout_deck_max &&
           c->deck2_replan_worlds ==
               (uint32_t)a->deck2_replan_worlds &&
           c->deck2_replan_cores == (uint32_t)a->deck2_replan_cores &&
           c->max_plies == LC_MAX_PLIES;
}

int policy_cost_controller_valid(const PolicyCostController *c)
{
    int frozen_onset = c &&
        (c->ply_lo == 0 || c->ply_lo == 4 || c->ply_lo == 8 ||
         c->ply_lo == 10 || c->ply_lo == 12 || c->ply_lo == 14);
    if (!c || c->root_net_fingerprint == 0 ||
        c->continuation_net_fingerprint == 0 ||
        c->root_net_fingerprint != c->continuation_net_fingerprint ||
        c->controller_abi != POLICY_COST_CONTROLLER_ABI ||
        c->build_profile != match_value_build_profile() ||
        (c->objective != 0 && c->objective != 3) ||
        (c->objective == 3) != (c->match_value_fingerprint != 0) ||
        c->root_symmetries != 20 || c->playout_symmetries != 20 ||
        c->playout_sample != 4 || c->playout_prune != 1 ||
        c->exact_terminal != 1 || c->no_belief != 1 ||
        c->dets != 800 || c->confirm_dets != 800 ||
        c->root_width != 5 || c->action_core_count != 3 ||
        c->min_cand != 1 || !frozen_onset || c->ply_hi != 0 ||
        c->discard_guard != 1 || c->root_prune != 0 ||
        !lc_float_isfinite(c->cand_floor) ||
        (float_bits(c->cand_floor) != float_bits(0.01f) &&
         float_bits(c->cand_floor) != float_bits(0.02f)) ||
        !lc_float_isfinite(c->override_k) ||
        c->override_k != (float)POLICY_COST_PRIMARY_Z ||
        !lc_float_isfinite(c->override_min) ||
        float_bits(c->override_min) != 0)
        return 0;
    return 1;
}

int policy_cost_validate(const PolicyCostTable *table)
{
    static const uint32_t required_anchor[POLICY_COST_ANCHORS] = {
        0, 4, 8, 12, 16, 24, 32, 40, 48, 64
    };
    int predictive = table &&
        (table->version == POLICY_COST_VERSION ||
         table->version == POLICY_COST_V20_VERSION ||
         table->version == POLICY_COST_V3_VERSION);
    int identity_valid = table && (
        (table->version == POLICY_COST_VERSION &&
         table->source_seed == POLICY_COST_SOURCE_SEED) ||
        (table->version == POLICY_COST_V20_VERSION &&
         table->source_seed == POLICY_COST_V20_SOURCE_SEED) ||
        (table->version == POLICY_COST_V3_VERSION &&
         (table->source_seed == POLICY_COST_V3_SOURCE_SEED ||
          table->source_seed == POLICY_COST_V2_SOURCE_SEED)) ||
        (table->version == POLICY_COST_LEGACY_VERSION &&
         table->source_seed == POLICY_COST_LEGACY_SOURCE_SEED));
    if (!identity_valid ||
        !policy_cost_controller_valid(&table->controller) ||
        (predictive && table->controller.ply_lo != 0) ||
        !finite_double(table->epsilon) ||
        double_bits(table->epsilon) != double_bits(POLICY_COST_EPSILON) ||
        table->primary_z != POLICY_COST_PRIMARY_Z ||
        table->fresh_z != POLICY_COST_FRESH_Z)
        return 0;
    for (int a = 0; a < POLICY_COST_ANCHORS; a++) {
        if (table->ply_anchor[a] != required_anchor[a]) return 0;
        double beta = table->beta[a];
        double aa = table->alpha_action[a];
        double ad = table->alpha_draw[a];
        if (!finite_double(beta) || !finite_double(aa) ||
            !finite_double(ad) || beta <= 0.0 || aa < 0.0 || ad < 0.0 ||
            !finite_double(aa / beta) || !finite_double(ad / beta))
            return 0;
        /* Legacy payloads persisted the two normalized lambdas.  Mapping
         * them to alpha at beta=1 preserves their bytes and exact runtime
         * behavior while making the in-memory representation unambiguous. */
        if (table->version == POLICY_COST_LEGACY_VERSION &&
            (beta != 1.0 || aa > 1000.0 || ad > 1000.0))
            return 0;
    }
    if (table->version == POLICY_COST_VERSION &&
        table->source_seed == POLICY_COST_SOURCE_SEED) {
        for (int a = 4; a <= 6; a++)
            if (double_bits(table->alpha_action[a]) != UINT64_C(0) ||
                double_bits(table->alpha_draw[a]) != UINT64_C(0))
                return 0;
    }
    return 1;
}

int policy_cost_matches_agent(const struct Agent *a)
{
    if (!a) return 0;
    if (!a->policy_cost) return !a->owns_policy_cost;
    const PolicyCostTable *table = a->policy_cost;
    if (a->kind != AG_ROLLOUT || !policy_cost_validate(table) ||
        table->payload_fingerprint == 0 || !a->net ||
        !a->continuation_net || a->veto_continuation_net ||
        a->action_ranker_net || a->bounded_late_root ||
        a->deck2_replan_worlds != 0 || a->deck2_replan_cores != 0 ||
        a->plan_deck_max != 0 || a->plan_block_gap != 0 ||
        a->draw_root_deck_max != 0 || a->draw_playout_deck_max != 0 ||
        a->semantic_cand != 0 || a->draw_variant_cores != 0 ||
        a->draw_variant_deck_max != 0 || a->eval_cand != 0 ||
        a->cand_mass != 0.0f || a->batch_dets != 0 || a->gate != 0.0f ||
        a->deck_max != 0 || a->policy_prefix_mode != 0 ||
        a->prefix_confirm_k != 0.0f || a->prefix_confirm_min != 0.0f ||
        a->confirm_temp != 0.0f || a->confirm_exact5 != 0 ||
        !a->no_belief)
        return 0;
    if (a->win_q == 3) {
        if (!match_value_matches_policy_cost_agent(a))
            return 0;
    } else if (a->match_value || a->owns_match_value) {
        return 0;
    }
    const PolicyCostController *c = &table->controller;
    int playout_prune = a->playout_prune < 0
        ? a->prune_dom : a->playout_prune != 0;
    uint64_t match_fingerprint = a->match_value
        ? a->match_value->payload_fingerprint : 0;
    return c->root_net_fingerprint ==
               match_value_net_fingerprint(a->net) &&
           c->continuation_net_fingerprint ==
               match_value_net_fingerprint(a->continuation_net) &&
           c->match_value_fingerprint == match_fingerprint &&
           c->objective == (uint32_t)a->win_q &&
           c->root_symmetries == (uint32_t)a->symmetries &&
           c->playout_symmetries == (uint32_t)a->playout_symmetries &&
           c->playout_sample == (uint32_t)a->playout_sample &&
           c->playout_prune == (uint32_t)playout_prune &&
           c->exact_terminal == (uint32_t)a->exact_terminal &&
           c->no_belief == (uint32_t)(a->no_belief != 0) &&
           c->dets == (uint32_t)a->dets &&
           c->confirm_dets == (uint32_t)a->confirm_dets &&
           c->root_width == (uint32_t)a->root_width &&
           c->action_core_count == (uint32_t)a->action_core_count &&
           c->min_cand == (uint32_t)a->min_cand &&
           c->ply_lo == (uint32_t)a->ply_lo &&
           c->ply_hi == (uint32_t)a->ply_hi &&
           c->discard_guard == (uint32_t)a->discard_guard &&
           c->root_prune == (uint32_t)a->prune_dom &&
           c->cand_floor == a->cand_floor &&
           c->override_k == a->override_k &&
           c->override_min == a->override_min;
}

int policy_cost_coefficients(const PolicyCostTable *table, int nply,
                             double *beta, double *alpha_action,
                             double *alpha_draw, int *anchor_interval)
{
    if (!policy_cost_validate(table) || nply < 0 || nply >= LC_MAX_PLIES ||
        !beta || !alpha_action || !alpha_draw)
        return 0;
    int interval = POLICY_COST_ANCHORS - 1;
    for (int a = 0; a + 1 < POLICY_COST_ANCHORS; a++)
        if ((uint32_t)nply < table->ply_anchor[a + 1]) {
            interval = a;
            break;
        }
    if (interval == POLICY_COST_ANCHORS - 1) {
        /* There is deliberately no unsupported post-64 tail slope. */
        *beta = table->beta[interval];
        *alpha_action = table->alpha_action[interval];
        *alpha_draw = table->alpha_draw[interval];
    } else {
        double lo = table->ply_anchor[interval];
        double hi = table->ply_anchor[interval + 1];
        double t = ((double)nply - lo) / (hi - lo);
        *beta = table->beta[interval] + t *
            (table->beta[interval + 1] - table->beta[interval]);
        *alpha_action = table->alpha_action[interval] + t *
            (table->alpha_action[interval + 1] -
             table->alpha_action[interval]);
        *alpha_draw = table->alpha_draw[interval] + t *
            (table->alpha_draw[interval + 1] -
             table->alpha_draw[interval]);
    }
    if (table->version == POLICY_COST_VERSION &&
        table->source_seed == POLICY_COST_SOURCE_SEED &&
        nply >= 16 && nply < 40) {
        *alpha_action = 0.0;
        *alpha_draw = 0.0;
    }
    if (anchor_interval) *anchor_interval = interval;
    return finite_double(*beta) && *beta > 0.0 &&
           finite_double(*alpha_action) && *alpha_action >= 0.0 &&
           finite_double(*alpha_draw) && *alpha_draw >= 0.0;
}

int policy_cost_schedule(const PolicyCostTable *table, int nply,
                         double *lambda_action, double *lambda_draw,
                         int *anchor_interval)
{
    double beta = 0.0, alpha_action = 0.0, alpha_draw = 0.0;
    if (!lambda_action || !lambda_draw ||
        !policy_cost_coefficients(table, nply, &beta, &alpha_action,
                                  &alpha_draw, anchor_interval))
        return 0;
    /* Divide only after all three schedules have been interpolated.  Doing
     * this at anchors first would define a different (and incorrect) curve. */
    *lambda_action = alpha_action / beta;
    *lambda_draw = alpha_draw / beta;
    return finite_double(*lambda_action) && finite_double(*lambda_draw);
}

static void store_controller(unsigned char *h,
                             const PolicyCostController *c)
{
    encode_u64(h + 40, c->root_net_fingerprint);
    encode_u64(h + 48, c->continuation_net_fingerprint);
    encode_u64(h + 56, c->match_value_fingerprint);
    const uint32_t word[POLICY_COST_CONTROLLER_WORDS] = {
        c->controller_abi, c->build_profile, c->objective,
        c->root_symmetries, c->playout_symmetries, c->playout_sample,
        c->playout_prune, c->exact_terminal, c->no_belief, c->dets,
        c->confirm_dets, c->root_width, c->action_core_count, c->min_cand,
        c->ply_lo, c->ply_hi, c->discard_guard, c->root_prune
    };
    for (int i = 0; i < POLICY_COST_CONTROLLER_WORDS; i++)
        encode_u32(h + 64 + 4 * i, word[i]);
    encode_u32(h + 136, float_bits(c->cand_floor));
    encode_u32(h + 140, float_bits(c->override_k));
    encode_u32(h + 144, float_bits(c->override_min));
}

static void load_controller(const unsigned char *h, PolicyCostController *c)
{
    memset(c, 0, sizeof *c);
    c->root_net_fingerprint = decode_u64(h + 40);
    c->continuation_net_fingerprint = decode_u64(h + 48);
    c->match_value_fingerprint = decode_u64(h + 56);
    uint32_t *word[POLICY_COST_CONTROLLER_WORDS] = {
        &c->controller_abi, &c->build_profile, &c->objective,
        &c->root_symmetries, &c->playout_symmetries, &c->playout_sample,
        &c->playout_prune, &c->exact_terminal, &c->no_belief, &c->dets,
        &c->confirm_dets, &c->root_width, &c->action_core_count,
        &c->min_cand, &c->ply_lo, &c->ply_hi, &c->discard_guard,
        &c->root_prune
    };
    for (int i = 0; i < POLICY_COST_CONTROLLER_WORDS; i++)
        *word[i] = decode_u32(h + 64 + 4 * i);
    c->cand_floor = bits_float(decode_u32(h + 136));
    c->override_k = bits_float(decode_u32(h + 140));
    c->override_min = bits_float(decode_u32(h + 144));
}

static void store_header(const PolicyCostTable *table,
                         unsigned char h[POLICY_COST_HEADER_BYTES])
{
    memset(h, 0, POLICY_COST_HEADER_BYTES);
    memcpy(h, POLICY_COST_MAGIC, sizeof POLICY_COST_MAGIC);
    encode_u32(h + 8, table->version);
    encode_u32(h + 12, POLICY_COST_HEADER_BYTES);
    encode_u32(h + 16, MATCH_ROUNDS);
    encode_u32(h + 20, POLICY_COST_ANCHORS);
    encode_u32(h + 24, table->version == POLICY_COST_LEGACY_VERSION
                           ? POLICY_COST_LEGACY_PAYLOAD_DOUBLES
                           : POLICY_COST_PAYLOAD_DOUBLES);
    encode_u64(h + 32, table->source_seed);
    store_controller(h, &table->controller);
    encode_u64(h + 160, double_bits(table->epsilon));
    encode_u64(h + 168, double_bits(table->primary_z));
    encode_u64(h + 176, double_bits(table->fresh_z));
    for (int a = 0; a < POLICY_COST_ANCHORS; a++)
        encode_u32(h + 184 + 4 * a, table->ply_anchor[a]);
}

static int write_double(FILE *file, double value, uint64_t *fingerprint)
{
    unsigned char bytes[8];
    encode_u64(bytes, double_bits(value));
    *fingerprint = fingerprint_update(*fingerprint, bytes, sizeof bytes);
    return fwrite(bytes, 1, sizeof bytes, file) == sizeof bytes;
}

static int read_double(FILE *file, double *value, uint64_t *fingerprint)
{
    unsigned char bytes[8];
    if (fread(bytes, 1, sizeof bytes, file) != sizeof bytes) return 0;
    *fingerprint = fingerprint_update(*fingerprint, bytes, sizeof bytes);
    *value = bits_double(decode_u64(bytes));
    return 1;
}

int policy_cost_save(const PolicyCostTable *table, const char *path)
{
    if (!path || !*path || !policy_cost_validate(table)) return -1;
    FILE *file = fopen(path, "wbx");
    if (!file) return -2;
    unsigned char h[POLICY_COST_HEADER_BYTES];
    store_header(table, h);
    uint64_t fingerprint = fingerprint_update(
        UINT64_C(1469598103934665603), h, sizeof h);
    int ok = fwrite(h, 1, sizeof h, file) == sizeof h;
    for (int a = 0; a < POLICY_COST_ANCHORS && ok; a++) {
        if (table->version != POLICY_COST_LEGACY_VERSION)
            ok = write_double(file, table->beta[a], &fingerprint);
        ok = ok &&
             write_double(file, table->alpha_action[a], &fingerprint) &&
             write_double(file, table->alpha_draw[a], &fingerprint);
    }
    unsigned char footer[8];
    encode_u64(footer, fingerprint);
    ok = ok && fwrite(footer, 1, sizeof footer, file) == sizeof footer;
    if (fclose(file) != 0) ok = 0;
    if (!ok) {
        (void)remove(path);
        return -3;
    }
    return 0;
}

PolicyCostTable *policy_cost_load(const char *path, int *error)
{
    if (error) *error = -1;
    if (!path || !*path) return NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    unsigned char h[POLICY_COST_HEADER_BYTES];
    int header_read = fread(h, 1, sizeof h, file) == sizeof h;
    uint32_t version = header_read ? decode_u32(h + 8) : 0;
    int predictive = version == POLICY_COST_VERSION ||
                     version == POLICY_COST_V20_VERSION ||
                     version == POLICY_COST_V3_VERSION;
    uint32_t payload_doubles = predictive
        ? POLICY_COST_PAYLOAD_DOUBLES
        : POLICY_COST_LEGACY_PAYLOAD_DOUBLES;
    if (!header_read ||
        memcmp(h, POLICY_COST_MAGIC, sizeof POLICY_COST_MAGIC) != 0 ||
        !((version == POLICY_COST_VERSION &&
           decode_u64(h + 32) == POLICY_COST_SOURCE_SEED) ||
          (version == POLICY_COST_V20_VERSION &&
           decode_u64(h + 32) == POLICY_COST_V20_SOURCE_SEED) ||
          (version == POLICY_COST_V3_VERSION &&
           (decode_u64(h + 32) == POLICY_COST_V3_SOURCE_SEED ||
            decode_u64(h + 32) == POLICY_COST_V2_SOURCE_SEED)) ||
          (version == POLICY_COST_LEGACY_VERSION &&
           decode_u64(h + 32) == POLICY_COST_LEGACY_SOURCE_SEED)) ||
        decode_u32(h + 12) != POLICY_COST_HEADER_BYTES ||
        decode_u32(h + 16) != MATCH_ROUNDS ||
        decode_u32(h + 20) != POLICY_COST_ANCHORS ||
        decode_u32(h + 24) != payload_doubles ||
        !bytes_are_zero(h + 28, 4) ||
        !bytes_are_zero(h + 148, 12) ||
        !bytes_are_zero(
            h + 184 + 4 * POLICY_COST_ANCHORS,
            POLICY_COST_HEADER_BYTES -
                (184 + 4 * POLICY_COST_ANCHORS))) {
        fclose(file);
        if (error) *error = -2;
        return NULL;
    }
    PolicyCostTable *table = calloc(1, sizeof *table);
    if (!table) {
        fclose(file);
        if (error) *error = -3;
        return NULL;
    }
    table->version = version;
    table->source_seed = decode_u64(h + 32);
    load_controller(h, &table->controller);
    table->epsilon = bits_double(decode_u64(h + 160));
    table->primary_z = bits_double(decode_u64(h + 168));
    table->fresh_z = bits_double(decode_u64(h + 176));
    for (int a = 0; a < POLICY_COST_ANCHORS; a++)
        table->ply_anchor[a] = decode_u32(h + 184 + 4 * a);

    uint64_t fingerprint = fingerprint_update(
        UINT64_C(1469598103934665603), h, sizeof h);
    int ok = 1;
    for (int a = 0; a < POLICY_COST_ANCHORS && ok; a++) {
        if (table->version != POLICY_COST_LEGACY_VERSION)
            ok = read_double(file, &table->beta[a], &fingerprint);
        else
            table->beta[a] = 1.0;
        ok = ok &&
             read_double(file, &table->alpha_action[a], &fingerprint) &&
             read_double(file, &table->alpha_draw[a], &fingerprint);
    }
    unsigned char footer[8];
    ok = ok && fread(footer, 1, sizeof footer, file) == sizeof footer &&
         decode_u64(footer) == fingerprint && fgetc(file) == EOF;
    if (fclose(file) != 0) ok = 0;
    table->payload_fingerprint = fingerprint;
    if (!ok || !policy_cost_validate(table)) {
        free(table);
        if (error) *error = -4;
        return NULL;
    }
    if (error) *error = 0;
    return table;
}

void policy_cost_free(PolicyCostTable *table)
{
    free(table);
}

static int valid_probability_vector(const float *prob, int n)
{
    if (!prob || n < 1 || n > MAX_MOVES) return 0;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        if (!lc_float_isfinite(prob[i]) || prob[i] < 0.0f) return 0;
        total += prob[i];
    }
    /* policy_probs_sym() normalizes in binary32.  This tolerance is frozen,
     * tight enough to reject conditional/subset vectors while admitting the
     * bounded accumulation error of at most MAX_MOVES legal probabilities. */
    return fabs(total - 1.0) <= 1e-5;
}

static int move_terms_validated(
    const PolicyCostTable *table, double lambda_action, double lambda_draw,
    const Move *move, const float *prob, int n, int index,
    double *semantic_prior, double *conditional_draw_prior, double *cost)
{
    double action = 0.0, joint = 0.0;
    for (int i = 0; i < n; i++)
        if (same_semantic_action(move[i], move[index])) {
            action += prob[i];
            /* Physical copies of one same-suit wager are strategically
             * indistinguishable.  Aggregate them before factorizing the
             * conditional draw mass, just as the action core does. */
            if (move[i].draw == move[index].draw) joint += prob[i];
        }
    if (!(joint > table->epsilon) || !(action > table->epsilon) ||
        joint > action * (1.0 + 1e-6))
        return 0;
    double draw = joint / action;
    if (!(draw > table->epsilon)) return 0;
    /* The artifact floor is a domain assertion, never a clamp: the exact
     * post-symmetry probabilities are what enter the logarithms. */
    double result = -lambda_action * log(action) - lambda_draw * log(draw);
    if (!finite_double(result)) return 0;
    *semantic_prior = action;
    *conditional_draw_prior = draw;
    *cost = result;
    return 1;
}

int policy_cost_move_terms(const PolicyCostTable *table, int round, int nply,
                           const Move *move, const float *prob, int n,
                           int index, double *semantic_prior,
                           double *conditional_draw_prior, double *cost)
{
    double lambda_action = 0.0, lambda_draw = 0.0;
    if (round < 0 || round >= MATCH_ROUNDS ||
        !policy_cost_schedule(
            table, nply, &lambda_action, &lambda_draw, NULL) ||
        !move || !valid_probability_vector(prob, n) ||
        index < 0 || index >= n || !semantic_prior ||
        !conditional_draw_prior || !cost)
        return 0;
    return move_terms_validated(
        table, lambda_action, lambda_draw, move, prob, n, index,
        semantic_prior, conditional_draw_prior, cost);
}

int policy_cost_decide_summary(
    const PolicyCostTable *table, int round, int nply,
    const Move *move, const float *prob, int nlegal,
    const int *candidate_index, int n, const double *q,
    const double *paired_se, int pair_stride, double z,
    PolicyCostDecision *decision)
{
    if (decision) memset(decision, 0, sizeof *decision);
    double beta = 0.0, alpha_action = 0.0, alpha_draw = 0.0;
    double lambda_action = 0.0, lambda_draw = 0.0;
    int interval = -1;
    if (!decision || round < 0 || round >= MATCH_ROUNDS ||
        !policy_cost_coefficients(table, nply, &beta, &alpha_action,
                                  &alpha_draw, &interval) ||
        !move || !valid_probability_vector(prob, nlegal) ||
        !candidate_index || !q || !paired_se ||
        n < 1 || n > POLICY_COST_MAX_CANDIDATES ||
        pair_stride < n || !finite_double(z) || z <= 0.0)
        return 0;
    lambda_action = alpha_action / beta;
    lambda_draw = alpha_draw / beta;
    if (!finite_double(lambda_action) || !finite_double(lambda_draw))
        return 0;
    decision->anchor_interval = interval;
    decision->beta = beta;
    decision->alpha_action = alpha_action;
    decision->alpha_draw = alpha_draw;
    decision->lambda_action = lambda_action;
    decision->lambda_draw = lambda_draw;
    for (int c = 0; c < n; c++) {
        if (candidate_index[c] < 0 || candidate_index[c] >= nlegal)
            return 0;
        for (int j = 0; j < c; j++)
            if (candidate_index[j] == candidate_index[c] ||
                (same_semantic_action(move[candidate_index[j]],
                                      move[candidate_index[c]]) &&
                 move[candidate_index[j]].draw ==
                     move[candidate_index[c]].draw))
                return 0;
    }
    for (int c = 0; c < n; c++) {
        if (!finite_double(q[c])) return 0;
        decision->q[c] = q[c];
        if (!move_terms_validated(
                table, lambda_action, lambda_draw, move, prob, nlegal,
                candidate_index[c], &decision->semantic_prior[c],
                &decision->conditional_draw_prior[c],
                &decision->cost[c]))
            return 0;
        decision->adjusted_q[c] = decision->q[c] - decision->cost[c];
    }
    for (int c = 0; c < n; c++)
        for (int rival = 0; rival < n; rival++) {
            double se = paired_se[(size_t)c * (size_t)pair_stride + rival];
            double reverse =
                paired_se[(size_t)rival * (size_t)pair_stride + c];
            if (!finite_double(se) || se < 0.0 ||
                (c == rival && se != 0.0) || se != reverse)
                return 0;
        }
    int hierarchical_draw_only = alpha_action == 0.0;
    int leader = 0;
    if (hierarchical_draw_only) {
        /* Choose the strategic action core with raw search Q only.  Policy
         * confidence may then arbitrate among draw sources for that exact
         * action, but can never alter which card/action core search chose. */
        int raw_leader = 0;
        for (int c = 1; c < n; c++)
            if (decision->q[c] > decision->q[raw_leader]) raw_leader = c;
        leader = raw_leader;
        for (int c = 0; c < n; c++)
            if (same_semantic_action(move[candidate_index[c]],
                                     move[candidate_index[raw_leader]]) &&
                decision->adjusted_q[c] > decision->adjusted_q[leader])
                leader = c;
    } else {
        for (int c = 1; c < n; c++)
            if (decision->adjusted_q[c] > decision->adjusted_q[leader])
                leader = c;
    }
    decision->leader = leader;
    decision->selected = 0;
    if (leader == 0) {
        decision->all_pair_passed = 1;
        return 1;
    }

    int passed = 1;
    for (int rival = 0; rival < n; rival++) {
        if (rival == leader) continue;
        double mean = decision->q[leader] - decision->q[rival];
        double se = paired_se[
            (size_t)leader * (size_t)pair_stride + rival];
        int same_core = same_semantic_action(
            move[candidate_index[rival]], move[candidate_index[leader]]);
        double evidence_delta = hierarchical_draw_only && !same_core
            ? mean
            : decision->adjusted_q[leader] - decision->adjusted_q[rival];
        decision->pair_delta[rival] = evidence_delta;
        decision->pair_se[rival] = se;
        int prior_protected = hierarchical_draw_only
            ? (same_core && decision->conditional_draw_prior[rival] >
                              decision->conditional_draw_prior[leader])
            : (rival == 0 ||
               decision->semantic_prior[rival] >
                   decision->semantic_prior[leader] ||
               decision->semantic_prior[rival] *
                       decision->conditional_draw_prior[rival] >
                   decision->semantic_prior[leader] *
                       decision->conditional_draw_prior[leader] ||
               (same_core &&
                decision->conditional_draw_prior[rival] >
                    decision->conditional_draw_prior[leader]));
        if (prior_protected) decision->prior_protected_rivals++;
        /* Adjusted evidence is all-pair: the proposed scalar leader must be
         * unique at the requested confidence level.  Raw positivity is
         * directed by semantic policy frequency rather than joint-move rank. */
        if (!(evidence_delta > z * se) ||
            (prior_protected && !(mean > 0.0)))
            passed = 0;
    }
    decision->all_pair_passed = passed;
    decision->selected = passed ? leader : 0;
    return 1;
}

int policy_cost_decide(const PolicyCostTable *table, int round, int nply,
                       const Move *move, const float *prob, int nlegal,
                       const int *candidate_index, int n,
                       const double *values, int stride, int worlds, double z,
                       PolicyCostDecision *decision)
{
    if (decision) memset(decision, 0, sizeof *decision);
    if (!decision || !values || n < 1 ||
        n > POLICY_COST_MAX_CANDIDATES || stride < worlds || worlds < 2)
        return 0;
    double q[POLICY_COST_MAX_CANDIDATES] = { 0 };
    double paired_se[POLICY_COST_MAX_CANDIDATES]
                    [POLICY_COST_MAX_CANDIDATES] = { { 0 } };
    for (int c = 0; c < n; c++) {
        double total = 0.0;
        for (int d = 0; d < worlds; d++) {
            double x = values[(size_t)c * (size_t)stride + (size_t)d];
            if (!finite_double(x)) return 0;
            total += x;
        }
        q[c] = total / worlds;
    }
    for (int c = 0; c < n; c++)
        for (int rival = c + 1; rival < n; rival++) {
            double mean = q[c] - q[rival];
            double centered = 0.0;
            for (int d = 0; d < worlds; d++) {
                double diff =
                    values[(size_t)c * (size_t)stride + (size_t)d] -
                    values[(size_t)rival * (size_t)stride + (size_t)d];
                double residual = diff - mean;
                centered += residual * residual;
            }
            double se = sqrt(centered / (worlds - 1) / worlds);
            paired_se[c][rival] = se;
            paired_se[rival][c] = se;
        }
    return policy_cost_decide_summary(
        table, round, nply, move, prob, nlegal, candidate_index, n,
        q, &paired_se[0][0], POLICY_COST_MAX_CANDIDATES, z, decision);
}
