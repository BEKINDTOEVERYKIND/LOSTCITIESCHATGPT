/* build_policy_cost -- materialize one calibrated rollout5 spline artifact.
 * This performs no fitting or selection.  Frozen coefficients and controller
 * fields are validated, bound to the bytes loaded by production, and saved
 * through the runtime's canonical no-clobber serializer. */
#include "../src/match_value.h"
#include "../src/policy_cost.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *root_path, *continuation_path, *match_value_path, *out_path;
    const char *lambda_action_text, *lambda_draw_text;
    uint64_t source_seed;
    double epsilon;
    PolicyCostController controller;
    uint32_t seen;
} Config;

enum {
    S_SEED = 1u << 0, S_EPSILON = 1u << 1, S_OBJECTIVE = 1u << 2,
    S_ROOT_SYM = 1u << 3, S_PLAYOUT_SYM = 1u << 4,
    S_SAMPLE = 1u << 5, S_PRUNE = 1u << 6, S_EXACT = 1u << 7,
    S_UNIFORM = 1u << 8, S_DETS = 1u << 9, S_CONFIRM = 1u << 10,
    S_WIDTH = 1u << 11, S_CORES = 1u << 12, S_MIN = 1u << 13,
    S_PLY_LO = 1u << 14, S_PLY_HI = 1u << 15, S_DISCARD = 1u << 16,
    S_FLOOR = 1u << 17, S_OVERRIDE_K = 1u << 18,
    S_OVERRIDE_MIN = 1u << 19, S_ROOT_PRUNE = 1u << 20,
    S_ALL = (1u << 21) - 1
};

static void usage(const char *program)
{
    fprintf(stderr,
        "usage: %s --root-model PATH --continuation-model PATH --out PATH "
        "--source-seed N --epsilon X --objective N [--match-value PATH] "
        "--root-symmetries N --playout-symmetries N --playout-sample N "
        "--playout-prune N --exact-terminal N --no-belief 1 --dets N "
        "--confirm-dets N --root-width N --action-core-count N --min-cand N "
        "--ply-lo N --ply-hi N --discard-guard N --root-prune N --cand-floor X "
        "--override-k 3.5 --override-min 0 --lambda-action a0,...,a9 "
        "--lambda-draw d0,...,d9\n", program);
}

static int parse_u64(const char *text, uint64_t *out)
{
    if (!text || !*text || *text == '-') return 0;
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)value;
    return 1;
}

static int parse_u32(const char *text, uint32_t *out)
{
    uint64_t value = 0;
    if (!parse_u64(text, &value) || value > UINT32_MAX) return 0;
    *out = (uint32_t)value;
    return 1;
}

static int parse_double_exact(const char *text, double *out)
{
    errno = 0;
    char *end = NULL;
    double value = text && *text ? strtod(text, &end) : 0.0;
    if (!text || !*text || errno || !end || *end ||
        !lc_double_isfinite(value)) return 0;
    *out = value;
    return 1;
}

static int parse_float_exact(const char *text, float *out)
{
    errno = 0;
    char *end = NULL;
    float value = text && *text ? strtof(text, &end) : 0.0f;
    if (!text || !*text || errno || !end || *end ||
        !lc_float_isfinite(value)) return 0;
    *out = value;
    return 1;
}

static int parse_schedule(const char *text,
                          double value[POLICY_COST_ANCHORS])
{
    if (!text || !*text) return 0;
    const char *cursor = text;
    for (int i = 0; i < POLICY_COST_ANCHORS; i++) {
        errno = 0;
        char *end = NULL;
        double parsed = strtod(cursor, &end);
        if (errno || !end || end == cursor || !lc_double_isfinite(parsed) ||
            parsed < 0.0 || parsed > 1000.0) return 0;
        value[i] = parsed;
        if (i + 1 == POLICY_COST_ANCHORS) {
            if (*end) return 0;
        } else {
            if (*end != ',' || !end[1]) return 0;
            cursor = end + 1;
        }
    }
    return 1;
}

static int next_value(int argc, char **argv, int *index, const char **out)
{
    if (*index + 1 >= argc) return 0;
    *out = argv[++*index];
    return 1;
}

static int parse_args(int argc, char **argv, Config *c)
{
    memset(c, 0, sizeof *c);
    c->controller.controller_abi = POLICY_COST_CONTROLLER_ABI;
    c->controller.build_profile = match_value_build_profile();
    for (int i = 1; i < argc; i++) {
        const char *v = NULL;
#define PATH_ARG(flag, field) \
        else if (!strcmp(argv[i], flag)) { \
            if (!next_value(argc, argv, &i, &v) || c->field) return 0; \
            c->field = v; \
        }
#define U32_ARG(flag, field, bit) \
        else if (!strcmp(argv[i], flag)) { \
            if (!next_value(argc, argv, &i, &v) || (c->seen & (bit)) || \
                !parse_u32(v, &c->controller.field)) return 0; \
            c->seen |= (bit); \
        }
        if (0) {}
        PATH_ARG("--root-model", root_path)
        PATH_ARG("--continuation-model", continuation_path)
        PATH_ARG("--match-value", match_value_path)
        PATH_ARG("--out", out_path)
        PATH_ARG("--lambda-action", lambda_action_text)
        PATH_ARG("--lambda-draw", lambda_draw_text)
        else if (!strcmp(argv[i], "--source-seed")) {
            if (!next_value(argc, argv, &i, &v) || (c->seen & S_SEED) ||
                !parse_u64(v, &c->source_seed)) return 0;
            c->seen |= S_SEED;
        } else if (!strcmp(argv[i], "--epsilon")) {
            if (!next_value(argc, argv, &i, &v) || (c->seen & S_EPSILON) ||
                !parse_double_exact(v, &c->epsilon)) return 0;
            c->seen |= S_EPSILON;
        }
        U32_ARG("--objective", objective, S_OBJECTIVE)
        U32_ARG("--root-symmetries", root_symmetries, S_ROOT_SYM)
        U32_ARG("--playout-symmetries", playout_symmetries, S_PLAYOUT_SYM)
        U32_ARG("--playout-sample", playout_sample, S_SAMPLE)
        U32_ARG("--playout-prune", playout_prune, S_PRUNE)
        U32_ARG("--exact-terminal", exact_terminal, S_EXACT)
        U32_ARG("--no-belief", no_belief, S_UNIFORM)
        U32_ARG("--dets", dets, S_DETS)
        U32_ARG("--confirm-dets", confirm_dets, S_CONFIRM)
        U32_ARG("--root-width", root_width, S_WIDTH)
        U32_ARG("--action-core-count", action_core_count, S_CORES)
        U32_ARG("--min-cand", min_cand, S_MIN)
        U32_ARG("--ply-lo", ply_lo, S_PLY_LO)
        U32_ARG("--ply-hi", ply_hi, S_PLY_HI)
        U32_ARG("--discard-guard", discard_guard, S_DISCARD)
        U32_ARG("--root-prune", root_prune, S_ROOT_PRUNE)
        else if (!strcmp(argv[i], "--cand-floor")) {
            if (!next_value(argc, argv, &i, &v) || (c->seen & S_FLOOR) ||
                !parse_float_exact(v, &c->controller.cand_floor)) return 0;
            c->seen |= S_FLOOR;
        } else if (!strcmp(argv[i], "--override-k")) {
            if (!next_value(argc, argv, &i, &v) ||
                (c->seen & S_OVERRIDE_K) ||
                !parse_float_exact(v, &c->controller.override_k)) return 0;
            c->seen |= S_OVERRIDE_K;
        } else if (!strcmp(argv[i], "--override-min")) {
            if (!next_value(argc, argv, &i, &v) ||
                (c->seen & S_OVERRIDE_MIN) ||
                !parse_float_exact(v, &c->controller.override_min)) return 0;
            c->seen |= S_OVERRIDE_MIN;
        } else if (!strcmp(argv[i], "--help")) {
            return -1;
        } else return 0;
#undef U32_ARG
#undef PATH_ARG
    }
    return c->root_path && c->continuation_path && c->out_path &&
           c->lambda_action_text && c->lambda_draw_text && c->seen == S_ALL;
}

static void json_string(const char *text)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p == '"' || *p == '\\') putchar('\\');
        if (*p >= 0x20) putchar(*p);
    }
    putchar('"');
}

int main(int argc, char **argv)
{
    Config config;
    int parsed = parse_args(argc, argv, &config);
    if (parsed < 0) { usage(argv[0]); return 0; }
    if (!parsed) { usage(argv[0]); return 2; }
    Net *root = malloc(sizeof *root), *continuation = malloc(sizeof *continuation);
    if (!root || !continuation) {
        fprintf(stderr, "out of memory\n"); free(root); free(continuation);
        return 1;
    }
    if (net_load(root, config.root_path) != 0 ||
        net_load(continuation, config.continuation_path) != 0) {
        fprintf(stderr, "cannot load policy-cost checkpoint binding\n");
        free(root); free(continuation); return 1;
    }
    config.controller.root_net_fingerprint = match_value_net_fingerprint(root);
    config.controller.continuation_net_fingerprint =
        match_value_net_fingerprint(continuation);
    MatchValueTable *match_value = NULL;
    if (config.controller.objective == 3) {
        int error = 0;
        if (!config.match_value_path ||
            !(match_value = match_value_load(config.match_value_path, &error))) {
            fprintf(stderr, "objective 3 requires a valid match-value table "
                            "(error %d)\n", error);
            free(root); free(continuation); return 1;
        }
        config.controller.match_value_fingerprint =
            match_value->payload_fingerprint;
    } else if (config.match_value_path) {
        fprintf(stderr, "--match-value is permitted only with objective 3\n");
        free(root); free(continuation); return 2;
    }
    static const uint32_t anchor[POLICY_COST_ANCHORS] = {
        0, 4, 8, 12, 16, 24, 32, 40, 48, 64
    };
    PolicyCostTable table;
    memset(&table, 0, sizeof table);
    table.version = POLICY_COST_VERSION;
    table.source_seed = config.source_seed;
    table.epsilon = config.epsilon;
    table.primary_z = POLICY_COST_PRIMARY_Z;
    table.fresh_z = POLICY_COST_FRESH_Z;
    table.controller = config.controller;
    memcpy(table.ply_anchor, anchor, sizeof anchor);
    if (!parse_schedule(config.lambda_action_text, table.lambda_action) ||
        !parse_schedule(config.lambda_draw_text, table.lambda_draw) ||
        !policy_cost_validate(&table)) {
        fprintf(stderr, "invalid policy-cost controller or coefficient input\n");
        match_value_free(match_value); free(root); free(continuation); return 2;
    }
    int saved = policy_cost_save(&table, config.out_path);
    if (saved != 0) {
        fprintf(stderr, "cannot create policy-cost artifact '%s' (error %d)\n",
                config.out_path, saved);
        match_value_free(match_value); free(root); free(continuation); return 1;
    }
    int error = 0;
    PolicyCostTable *verified = policy_cost_load(config.out_path, &error);
    if (!verified) {
        fprintf(stderr, "cannot reopen policy-cost artifact (error %d)\n", error);
        (void)remove(config.out_path);
        match_value_free(match_value); free(root); free(continuation); return 1;
    }
    printf("{\"schema\":\"lc-policy-cost-build-v1\",\"output\":");
    json_string(config.out_path);
    printf(",\"source_seed\":%" PRIu64
           ",\"payload_fingerprint\":\"%016" PRIx64
           "\",\"root_net_fingerprint\":\"%016" PRIx64
           "\",\"continuation_net_fingerprint\":\"%016" PRIx64
           "\",\"match_value_fingerprint\":\"%016" PRIx64
           "\",\"primary_z\":3.5,\"fresh_z\":2.58,"
           "\"no_belief\":1,\"legacy_override_min\":0}\n",
           verified->source_seed, verified->payload_fingerprint,
           verified->controller.root_net_fingerprint,
           verified->controller.continuation_net_fingerprint,
           verified->controller.match_value_fingerprint);
    policy_cost_free(verified); match_value_free(match_value);
    free(root); free(continuation); return 0;
}
