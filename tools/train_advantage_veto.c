#define _POSIX_C_SOURCE 200809L
/* Fit a direct action-advantage ranker while freezing trunk/value/belief.
 *
 * For a proposal, the target pair log-odds is
 *
 *   signed_full_match_hybrid_delta / label_scale.
 *
 * The prediction is (ranker pair log-odds - champion pair log-odds), exactly
 * the direct signed action-advantage residual consumed by rolloutu4.  A Huber
 * loss on that raw residual cannot saturate behind a strong champion
 * preference.  Every record also
 * applies a full-legal-action champion KL anchor.  Generated source matches
 * are split as indivisible groups; no state from one match can cross the
 * train/validation boundary.
 */
#include "action_advantage_format.h"
#include "../src/agent.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    const char *records_path;
    const char *champion_path;
    const char *out_path;
    const char *metrics_path;
    uint64_t split_seed;
    int validation_permille;
    int epochs;
    int batch;
    float lr;
    float anchor_kl;
    float pair_scale;
    float label_scale;
    float huber_delta;
    float max_pair_weight;
    float weight_decay;
    float max_validation_kl;
    float max_state_kl;
    int force;
    int dry_run;
} Config;

typedef struct {
    Net *m, *v;
    uint64_t t;
} HeadAdam;

typedef struct {
    double kl_sum;
    double max_kl;
    double pair_loss_sum;
    double pair_weight_sum;
    double residual_alignment;
    uint64_t states;
    uint64_t proposals;
} Metrics;

#define THRESHOLD_GRID_N 5
typedef struct {
    double threshold;
    double runtime_threshold;
    uint64_t retained;
    uint64_t mistakes;
    uint64_t invalid_scores;
    double signed_hybrid_sum;
    double signed_hybrid_mean;
    double oracle_regret;
} ThresholdMetrics;

static int parse_i(const char *s, int lo, int hi, int *out)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return 0;
    *out = (int)v;
    return 1;
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)v;
    return 1;
}

static int parse_f(const char *s, float lo, float hi, float *out)
{
    char *end = NULL;
    errno = 0;
    float v = strtof(s, &end);
    if (errno || !end || *end || !lc_float_isfinite(v) || v < lo || v > hi)
        return 0;
    *out = v;
    return 1;
}

static void usage(FILE *f, const char *argv0)
{
    fprintf(f,
        "usage: %s --records FILE --champion NET --out RANKER [options]\n\n"
        "  --epochs N              deterministic passes (default 24)\n"
        "  --batch N               records per head-only Adam step (default 32)\n"
        "  --lr X                  learning rate (default 2e-5)\n"
        "  --anchor-kl X           full-action KL coefficient (default 1)\n"
        "  --pair-scale X          signed pairwise coefficient (default 1)\n"
        "  --label-scale X         hybrid points per residual log-odds "
        "(default 12.5)\n"
        "  --huber-delta X         raw-residual Huber transition (default 1)\n"
        "  --max-pair-weight X     confidence cap (default 4)\n"
        "  --weight-decay X        policy-head-only decay (default 0)\n"
        "  --split-seed N          grouped split seed (default 20260823)\n"
        "  --validation-permille N grouped validation share (default 200)\n"
        "  --max-validation-kl X   output gate, at most .02 (default .01)\n"
        "  --max-state-kl X        output gate, at most .10 (default .05)\n"
        "  --dry-run               run all training/gates without writing\n"
        "  --metrics-json FILE     atomically write heldout threshold metrics\n"
        "  --force                 atomically replace an existing output\n\n"
        "Only wplay/bplay, wdraw/bdraw, and wcomb/bcomb are updated.  The "
        "trainer rejects a validation pair-loss regression, any KL gate "
        "failure, mixed/empty source-match partitions, or one changed "
        "trunk/value/belief byte.\n", argv0);
}

static int parse_args(int argc, char **argv, Config *c)
{
    memset(c, 0, sizeof(*c));
    c->split_seed = UINT64_C(20260823);
    c->validation_permille = 200;
    c->epochs = 24;
    c->batch = 32;
    c->lr = 2e-5f;
    c->anchor_kl = 1.0f;
    c->pair_scale = 1.0f;
    c->label_scale = 12.5f;
    c->huber_delta = 1.0f;
    c->max_pair_weight = 4.0f;
    c->max_validation_kl = 0.01f;
    c->max_state_kl = 0.05f;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--records") && ++i < argc) c->records_path = argv[i];
        else if (!strcmp(a, "--champion") && ++i < argc)
            c->champion_path = argv[i];
        else if (!strcmp(a, "--out") && ++i < argc) c->out_path = argv[i];
        else if (!strcmp(a, "--epochs") && ++i < argc &&
                 parse_i(argv[i], 1, 10000, &c->epochs)) {}
        else if (!strcmp(a, "--batch") && ++i < argc &&
                 parse_i(argv[i], 1, 1000000, &c->batch)) {}
        else if (!strcmp(a, "--lr") && ++i < argc &&
                 parse_f(argv[i], 1e-8f, 1e-2f, &c->lr)) {}
        else if (!strcmp(a, "--anchor-kl") && ++i < argc &&
                 parse_f(argv[i], 0.0f, 100.0f, &c->anchor_kl)) {}
        else if (!strcmp(a, "--pair-scale") && ++i < argc &&
                 parse_f(argv[i], 1e-6f, 100.0f, &c->pair_scale)) {}
        else if (!strcmp(a, "--label-scale") && ++i < argc &&
                 parse_f(argv[i], 1.0f, 1000.0f, &c->label_scale)) {}
        else if (!strcmp(a, "--huber-delta") && ++i < argc &&
                 parse_f(argv[i], 0.01f, 10.0f, &c->huber_delta)) {}
        else if (!strcmp(a, "--max-pair-weight") && ++i < argc &&
                 parse_f(argv[i], 0.25f, 100.0f, &c->max_pair_weight)) {}
        else if (!strcmp(a, "--weight-decay") && ++i < argc &&
                 parse_f(argv[i], 0.0f, 1.0f, &c->weight_decay)) {}
        else if (!strcmp(a, "--split-seed") && ++i < argc &&
                 parse_u64(argv[i], &c->split_seed)) {}
        else if (!strcmp(a, "--validation-permille") && ++i < argc &&
                 parse_i(argv[i], 1, 999, &c->validation_permille)) {}
        else if (!strcmp(a, "--max-validation-kl") && ++i < argc &&
                 parse_f(argv[i], 0.0f, 0.02f, &c->max_validation_kl)) {}
        else if (!strcmp(a, "--max-state-kl") && ++i < argc &&
                 parse_f(argv[i], 0.0f, 0.10f, &c->max_state_kl)) {}
        else if (!strcmp(a, "--force")) c->force = 1;
        else if (!strcmp(a, "--dry-run")) c->dry_run = 1;
        else if (!strcmp(a, "--metrics-json") && ++i < argc)
            c->metrics_path = argv[i];
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "invalid or incomplete option: %s\n", a);
            return 0;
        }
    }
    if (!c->records_path || !c->champion_path ||
        (!c->dry_run && !c->out_path)) {
        fprintf(stderr, "--records and --champion are required; --out is "
                        "required unless --dry-run\n");
        return 0;
    }
    return 1;
}

static void softmax(const float *logit, int n, float *p)
{
    float hi = logit[0];
    for (int i = 1; i < n; i++) if (logit[i] > hi) hi = logit[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        p[i] = expf(logit[i] - hi);
        sum += p[i];
    }
    for (int i = 0; i < n; i++) p[i] = (float)(p[i] / sum);
}

static int move_index(const ActionAdvantageRecord *r, uint16_t packed)
{
    for (int i = 0; i < r->nlegal; i++)
        if (r->legal[i] == packed) return i;
    return -1;
}

static float clipped_signed_residual(const Config *c,
                                     const ActionAdvantageRecord *r)
{
    float x = r->hybrid_mean / c->label_scale;
    if (x > 6.0f) x = 6.0f;
    if (x < -6.0f) x = -6.0f;
    return x;
}

static float pair_weight(const Config *c, const ActionAdvantageRecord *r)
{
    float floor = c->label_scale /
        sqrtf((float)(r->label_worlds ? r->label_worlds : 1));
    float z = fabsf(r->hybrid_mean) / (r->hybrid_se + floor);
    float w = 0.25f + z;
    return w < c->max_pair_weight ? w : c->max_pair_weight;
}

/* Adds one anchored, signed pairwise sample.  net_backward also computes
 * discarded trunk gradients in g; the updater below never reads those bytes. */
static int accumulate_record(const Config *c, const Net *net,
                             const ActionAdvantageRecord *r, Net *g,
                             double *loss_weight)
{
    NetAct act;
    float logits[MAX_MOVES], teacher[MAX_MOVES], current[MAX_MOVES];
    float dlogit[MAX_MOVES];
    net_trunk(net, &r->features, &act);
    net_policy_act(net, &act, r->legal, r->nlegal, logits);
    softmax(r->champion_logits, r->nlegal, teacher);
    softmax(logits, r->nlegal, current);
    for (int i = 0; i < r->nlegal; i++)
        dlogit[i] = c->anchor_kl * (current[i] - teacher[i]);
    *loss_weight = c->anchor_kl;

    if (r->kind == AA_KIND_PROPOSAL) {
        int bi = move_index(r, r->baseline);
        int pi = move_index(r, r->proposal);
        if (bi < 0 || pi < 0 || bi == pi) return 0;
        float champion_gap = r->champion_logits[pi] -
                             r->champion_logits[bi];
        float predicted = (logits[pi] - logits[bi]) - champion_gap;
        float target = clipped_signed_residual(c, r);
        float error = predicted - target;
        float huber_grad = fabsf(error) <= c->huber_delta
            ? error : copysignf(c->huber_delta, error);
        float weight = c->pair_scale * pair_weight(c, r);
        float d = weight * huber_grad;
        dlogit[pi] += d;
        dlogit[bi] -= d;
        *loss_weight += weight;
    }
    net_backward(net, &r->features, &act, 0.0f, r->legal, dlogit,
                 r->nlegal, NULL, NULL, 0, g);
    return 1;
}

static void adam_region(float *w, const float *g, float *m, float *v,
                        size_t n, float step, float wd)
{
    for (size_t i = 0; i < n; i++) {
        float grad = g[i] + wd * w[i];
        m[i] = 0.9f * m[i] + 0.1f * grad;
        v[i] = 0.999f * v[i] + 0.001f * grad * grad;
        w[i] -= step * m[i] / (sqrtf(v[i]) + 1e-8f);
    }
}

static void head_adam_step(Net *n, Net *g, HeadAdam *a,
                           float lr, float scale, float wd)
{
    /* Physical wager copies are exactly indistinguishable.  Generated legal
     * support names one concrete ID, so sum and mirror those gradients before
     * updating to prevent arbitrary card-ID artifacts. */
    net_tie_wager_gradients(g);
    a->t++;
    float bc1 = 1.0f - powf(0.9f, (float)a->t);
    float bc2 = 1.0f - powf(0.999f, (float)a->t);
    float step = lr * scale * sqrtf(bc2) / bc1;
#define UPDATE(field) adam_region((float *)n->field, (const float *)g->field, \
                                  (float *)a->m->field, (float *)a->v->field, \
                                  sizeof(n->field) / sizeof(float), step, wd)
    UPDATE(wplay);
    UPDATE(bplay);
    UPDATE(wdraw);
    UPDATE(bdraw);
    UPDATE(wcomb);
    UPDATE(bcomb);
#undef UPDATE
}

static double huber_loss(double error, double delta)
{
    double a = fabs(error);
    return a <= delta ? 0.5 * error * error
                      : delta * (a - 0.5 * delta);
}

static Metrics evaluate(const Config *c, const Net *net,
                        const ActionAdvantageRecord *records,
                        const uint8_t *validation, uint64_t n,
                        int want_validation)
{
    Metrics m;
    memset(&m, 0, sizeof(m));
    for (uint64_t k = 0; k < n; k++) {
        if ((int)validation[k] != want_validation) continue;
        const ActionAdvantageRecord *r = &records[k];
        NetAct act;
        float logits[MAX_MOVES], teacher[MAX_MOVES], current[MAX_MOVES];
        net_trunk(net, &r->features, &act);
        net_policy_act(net, &act, r->legal, r->nlegal, logits);
        softmax(r->champion_logits, r->nlegal, teacher);
        softmax(logits, r->nlegal, current);
        double kl = 0.0;
        for (int i = 0; i < r->nlegal; i++)
            if (teacher[i] > 0.0f)
                kl += teacher[i] * log((double)teacher[i] / current[i]);
        if (kl < 0.0 && kl > -1e-10) kl = 0.0;
        m.kl_sum += kl;
        if (kl > m.max_kl) m.max_kl = kl;
        m.states++;
        if (r->kind == AA_KIND_PROPOSAL) {
            int bi = move_index(r, r->baseline);
            int pi = move_index(r, r->proposal);
            float gap0 = r->champion_logits[pi] - r->champion_logits[bi];
            float signed_target = clipped_signed_residual(c, r);
            float gap = logits[pi] - logits[bi];
            float w = pair_weight(c, r);
            float residual = gap - gap0;
            m.pair_loss_sum += w * huber_loss(
                (double)residual - signed_target, c->huber_delta);
            m.pair_weight_sum += w;
            m.residual_alignment +=
                (signed_target > 0.0f && residual > 0.0f) ||
                (signed_target < 0.0f && residual < 0.0f) ||
                (signed_target == 0.0f && fabsf(residual) < 1e-5f);
            m.proposals++;
        }
    }
    return m;
}

static double mean_kl(const Metrics *m)
{
    return m->states ? m->kl_sum / (double)m->states : INFINITY;
}

static double mean_pair_loss(const Metrics *m)
{
    return m->pair_weight_sum ? m->pair_loss_sum / m->pair_weight_sum
                              : INFINITY;
}

static int finite_region(const float *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (!lc_float_isfinite(x[i])) return 0;
    return 1;
}

static int policy_heads_finite(const Net *n)
{
#define FINITE_FIELD(field) \
    finite_region((const float *)n->field, sizeof(n->field) / sizeof(float))
    return FINITE_FIELD(wplay) && FINITE_FIELD(bplay) &&
           FINITE_FIELD(wdraw) && FINITE_FIELD(bdraw) &&
           FINITE_FIELD(wcomb) && FINITE_FIELD(bcomb);
#undef FINITE_FIELD
}

static int proposal_residual(const Net *champion, const Net *net,
                             const ActionAdvantageRecord *r, int symmetries,
                             double *score)
{
    Move baseline = {
        MOVE_CARD(r->baseline), MOVE_DISC(r->baseline), MOVE_DRAW(r->baseline)
    };
    Move proposal = {
        MOVE_CARD(r->proposal), MOVE_DISC(r->proposal), MOVE_DRAW(r->proposal)
    };
    return policy_residual_log_odds_sym(
        champion, net, &r->information_view, baseline, proposal,
        symmetries, score);
}

static double deployed_threshold(double canonical)
{
    return (double)(float)canonical;
}

static int threshold_grid(const Net *champion, const Net *net, int symmetries,
                           const ActionAdvantageRecord *records,
                           const uint8_t *validation, uint64_t n,
                           ThresholdMetrics out[THRESHOLD_GRID_N])
{
    const double threshold[THRESHOLD_GRID_N] = { 0, .1, .25, .5, 1 };
    memset(out, 0, THRESHOLD_GRID_N * sizeof(*out));
    for (int g = 0; g < THRESHOLD_GRID_N; g++)
    {
        out[g].threshold = threshold[g];
        out[g].runtime_threshold = deployed_threshold(threshold[g]);
    }
    for (uint64_t i = 0; i < n; i++) {
        const ActionAdvantageRecord *r = &records[i];
        if (!validation[i] || r->kind != AA_KIND_PROPOSAL) continue;
        double score = 0.0;
        int valid = proposal_residual(champion, net, r, symmetries, &score);
        double advantage = r->hybrid_mean;
        for (int g = 0; g < THRESHOLD_GRID_N; g++) {
            if (!valid) {
                out[g].invalid_scores++;
                continue;
            }
            int retain = score >= out[g].runtime_threshold;
            if (retain) {
                out[g].retained++;
                out[g].signed_hybrid_sum += advantage;
                out[g].mistakes += advantage < 0.0;
                if (advantage < 0.0) out[g].oracle_regret -= advantage;
            } else if (advantage > 0.0) {
                out[g].oracle_regret += advantage;
            }
        }
    }
    for (int g = 0; g < THRESHOLD_GRID_N; g++)
        out[g].signed_hybrid_mean = out[g].retained
            ? out[g].signed_hybrid_sum / (double)out[g].retained : 0.0;
    return out[0].invalid_scores == 0;
}

static void print_metrics_json(FILE *f, const ActionAdvantageHeader *h,
                               const Config *c, const Metrics *initial,
                               const Metrics *final,
                               const ThresholdMetrics grid[THRESHOLD_GRID_N],
                               int frozen, int passed)
{
    fprintf(f,
        "{\"schema\":\"lc-action-advantage-validation-v1\","
        "\"record_chain\":\"%016llx\","
        "\"champion_hash\":\"%016llx\","
        "\"record_header\":{"
        "\"generator_seed\":\"%llu\","
        "\"record_count\":%llu,\"anchor_count\":%llu,"
        "\"proposal_count\":%llu,"
        "\"label_worlds\":%u,\"ply_lo\":%u,"
        "\"match_rounds\":%u,\"label_threads\":%u,"
        "\"scoring_symmetries\":%u,"
        "\"source_matches_requested\":%u,"
        "\"source_matches_completed\":%u,"
        "\"proposal_cap\":%u,\"collection_stop_reason\":%u,"
        "\"maintained_actor_spec_hash\":\"%016llx\","
        "\"maintained_root_net_hash\":\"%016llx\","
        "\"maintained_continuation_net_hash\":\"%016llx\","
        "\"maintained_controller_net_hash\":\"%016llx\","
        "\"reroot_actor_spec_hash\":\"%016llx\","
        "\"reroot_root_net_hash\":\"%016llx\","
        "\"reroot_continuation_net_hash\":\"%016llx\","
        "\"reroot_controller_net_hash\":\"%016llx\"},"
        "\"split_seed\":%llu,\"validation_permille\":%d,"
        "\"validation_states\":%llu,\"validation_proposals\":%llu,"
        "\"initial_pair_huber\":%.17g,\"final_pair_huber\":%.17g,"
        "\"mean_policy_kl\":%.17g,\"max_policy_kl\":%.17g,"
        "\"frozen_nonpolicy\":%s,\"training_gate_passed\":%s,"
        "\"threshold_grid\":[",
        (unsigned long long)h->record_chain_hash,
        (unsigned long long)h->champion_net_hash,
        (unsigned long long)h->generator_seed,
        (unsigned long long)h->record_count,
        (unsigned long long)h->anchor_count,
        (unsigned long long)h->proposal_count,
        h->label_worlds, h->ply_lo, h->match_rounds, h->label_threads,
        h->scoring_symmetries, h->source_matches_requested,
        h->source_matches_completed, h->proposal_cap,
        h->collection_stop_reason,
        (unsigned long long)h->maintained_actor_spec_hash,
        (unsigned long long)h->maintained_root_net_hash,
        (unsigned long long)h->maintained_continuation_net_hash,
        (unsigned long long)h->maintained_controller_net_hash,
        (unsigned long long)h->reroot_actor_spec_hash,
        (unsigned long long)h->reroot_root_net_hash,
        (unsigned long long)h->reroot_continuation_net_hash,
        (unsigned long long)h->reroot_controller_net_hash,
        (unsigned long long)c->split_seed, c->validation_permille,
        (unsigned long long)final->states,
        (unsigned long long)final->proposals,
        mean_pair_loss(initial), mean_pair_loss(final), mean_kl(final),
        final->max_kl, frozen ? "true" : "false",
        passed ? "true" : "false");
    for (int g = 0; g < THRESHOLD_GRID_N; g++) {
        if (g) fputc(',', f);
        fprintf(f,
            "{\"threshold\":%.17g,\"runtime_threshold\":%.17g,"
            "\"retained\":%llu,"
            "\"signed_hybrid_sum\":%.17g,"
            "\"signed_hybrid_mean\":%.17g,\"mistakes\":%llu,"
            "\"invalid_scores\":%llu,"
            "\"oracle_regret\":%.17g}",
            grid[g].threshold, grid[g].runtime_threshold,
            (unsigned long long)grid[g].retained,
            grid[g].signed_hybrid_sum, grid[g].signed_hybrid_mean,
            (unsigned long long)grid[g].mistakes,
            (unsigned long long)grid[g].invalid_scores,
            grid[g].oracle_regret);
    }
    fputs("]}\n", f);
}

static int write_metrics_json(const char *path,
                              const ActionAdvantageHeader *h,
                              const Config *c, const Metrics *initial,
                              const Metrics *final,
                              const ThresholdMetrics grid[THRESHOLD_GRID_N],
                              int frozen, int passed)
{
    if (!path) return 1;
    struct stat st;
    if (!c->force && stat(path, &st) == 0) {
        fprintf(stderr, "%s exists (use --force)\n", path);
        return 0;
    }
    char *tmp = (char *)malloc(strlen(path) + 64);
    if (!tmp) return 0;
    snprintf(tmp, strlen(path) + 64, "%s.tmp.%ld", path, (long)getpid());
    FILE *f = fopen(tmp, "wb");
    int ok = f != NULL;
    if (ok) {
        print_metrics_json(f, h, c, initial, final, grid, frozen, passed);
        ok = fflush(f) == 0 && !ferror(f);
    }
    if (f && fclose(f) != 0) ok = 0;
    if (ok && rename(tmp, path) != 0) ok = 0;
    if (!ok) unlink(tmp);
    free(tmp);
    return ok;
}

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static void deterministic_shuffle(uint64_t *order, uint64_t n,
                                  uint64_t seed)
{
    Rng rng;
    rng_seed(&rng, mix64(seed));
    for (uint64_t i = n; i > 1; i--) {
        uint64_t j = rng_below(&rng, (uint32_t)i);
        uint64_t t = order[i - 1]; order[i - 1] = order[j]; order[j] = t;
    }
}

static int stored_champion_logits_match(
    const Net *champion, const ActionAdvantageRecord *r)
{
    NetAct act;
    float expected[MAX_MOVES];
    net_trunk(champion, &r->features, &act);
    net_policy_act(champion, &act, r->legal, r->nlegal, expected);
    return memcmp(expected, r->champion_logits,
                  (size_t)r->nlegal * sizeof(expected[0])) == 0;
}

static int save_verified(const Net *n, const char *path, int force)
{
    struct stat st;
    if (!force && stat(path, &st) == 0) {
        fprintf(stderr, "%s exists (use --force)\n", path);
        return 0;
    }
    char *tmp = (char *)malloc(strlen(path) + 64);
    Net *check = (Net *)malloc(sizeof(*check));
    if (!tmp || !check) { free(tmp); free(check); return 0; }
    snprintf(tmp, strlen(path) + 64, "%s.tmp.%ld", path, (long)getpid());
    int ok = net_save(n, tmp) == 0 && net_load(check, tmp) == 0 &&
             memcmp(n, check, sizeof(*n)) == 0 && rename(tmp, path) == 0;
    if (!ok) unlink(tmp);
    free(check);
    free(tmp);
    return ok;
}

int main(int argc, char **argv)
{
    Config c;
    if (!parse_args(argc, argv, &c)) {
        usage(stderr, argv[0]);
        return 2;
    }
    ActionAdvantageHeader header;
    char error[200];
    ActionAdvantageRecord *records = aa_read_file(
        c.records_path, &header, error, sizeof(error));
    if (!records) {
        fprintf(stderr, "cannot load records: %s\n", error);
        return 1;
    }
    Net *champion = (Net *)malloc(sizeof(*champion));
    Net *ranker = (Net *)malloc(sizeof(*ranker));
    Net *gradient = (Net *)calloc(1, sizeof(*gradient));
    HeadAdam adam = {
        (Net *)calloc(1, sizeof(Net)), (Net *)calloc(1, sizeof(Net)), 0
    };
    if (!champion || !ranker || !gradient || !adam.m || !adam.v ||
        net_load(champion, c.champion_path) != 0) {
        fprintf(stderr, "cannot allocate trainer or load champion\n");
        free(records); free(champion); free(ranker); free(gradient);
        free(adam.m); free(adam.v);
        return 1;
    }
    if (aa_hash_bytes(champion, sizeof(*champion)) !=
        header.champion_net_hash) {
        fprintf(stderr, "record champion hash does not match %s\n",
                c.champion_path);
        goto fail;
    }
    for (uint64_t i = 0; i < header.record_count; i++)
        if (!stored_champion_logits_match(champion, &records[i])) {
            fprintf(stderr, "stored champion logits do not match bound net "
                            "at record %llu\n", (unsigned long long)i);
            goto fail;
        }
    memcpy(ranker, champion, sizeof(*ranker));

    uint8_t *validation = (uint8_t *)malloc((size_t)header.record_count);
    uint64_t *order = (uint64_t *)malloc(
        (size_t)header.record_count * sizeof(*order));
    if (!validation || !order) {
        fprintf(stderr, "out of memory building grouped split\n");
        free(validation); free(order);
        goto fail;
    }
    uint64_t train_n = 0, valid_n = 0, train_groups = 0, valid_groups = 0;
    uint64_t train_proposals = 0, valid_proposals = 0;
    for (uint64_t i = 0; i < header.record_count; i++) {
        int is_valid = aa_group_is_validation(
            records[i].source_match_id, c.split_seed,
            (unsigned)c.validation_permille);
        validation[i] = (uint8_t)is_valid;
        if (is_valid) {
            valid_n++;
            valid_proposals += records[i].kind == AA_KIND_PROPOSAL;
        } else {
            order[train_n++] = i;
            train_proposals += records[i].kind == AA_KIND_PROPOSAL;
        }
        int first = 1;
        for (uint64_t j = 0; j < i; j++)
            if (records[j].source_match_id == records[i].source_match_id) {
                first = 0;
                if (validation[j] != validation[i]) {
                    fprintf(stderr, "source match crossed grouped split\n");
                    free(validation); free(order);
                    goto fail;
                }
                break;
            }
        if (first) {
            if (is_valid) valid_groups++;
            else train_groups++;
        }
    }
    if (!train_n || !valid_n || !train_groups || !valid_groups ||
        !train_proposals || !valid_proposals) {
        fprintf(stderr, "grouped split must contain states and signed "
                        "proposals from at least one complete source match in "
                        "both partitions\n");
        free(validation); free(order);
        goto fail;
    }
    printf("grouped split: train %llu records/%llu matches/%llu proposals; "
           "validation %llu records/%llu matches/%llu proposals\n",
           (unsigned long long)train_n, (unsigned long long)train_groups,
           (unsigned long long)train_proposals, (unsigned long long)valid_n,
           (unsigned long long)valid_groups,
           (unsigned long long)valid_proposals);

    Metrics initial_valid = evaluate(&c, champion, records, validation,
                                     header.record_count, 1);
    for (int epoch = 0; epoch < c.epochs; epoch++) {
        deterministic_shuffle(order, train_n,
                              c.split_seed ^ (uint64_t)(unsigned)epoch ^
                              UINT64_C(0x45504f4348534855));
        uint64_t pos = 0;
        while (pos < train_n) {
            memset(gradient, 0, sizeof(*gradient));
            uint64_t end = pos + (uint64_t)c.batch;
            if (end > train_n) end = train_n;
            double loss_weight = 0.0;
            for (uint64_t k = pos; k < end; k++) {
                double sample_weight = 0.0;
                if (!accumulate_record(&c, ranker, &records[order[k]],
                                       gradient, &sample_weight)) {
                    fprintf(stderr, "invalid training record after load\n");
                    free(validation); free(order);
                    goto fail;
                }
                loss_weight += sample_weight;
            }
            float scale = loss_weight > 0.0 ? (float)(1.0 / loss_weight) : 1.0f;
            head_adam_step(ranker, gradient, &adam, c.lr, scale,
                           c.weight_decay);
            if (!aa_nonpolicy_equal(champion, ranker) ||
                !policy_heads_finite(ranker)) {
                fprintf(stderr, "non-finite policy or changed frozen "
                                "trunk/value/belief during training\n");
                free(validation); free(order);
                goto fail;
            }
            pos = end;
        }
        Metrics train = evaluate(&c, ranker, records, validation,
                                 header.record_count, 0);
        Metrics valid = evaluate(&c, ranker, records, validation,
                                 header.record_count, 1);
        printf("epoch %d: train pair %.6f KL %.6g; valid pair %.6f KL "
               "%.6g max %.6g sign %.1f%%\n",
               epoch + 1, mean_pair_loss(&train), mean_kl(&train),
               mean_pair_loss(&valid), mean_kl(&valid), valid.max_kl,
               100.0 * valid.residual_alignment / valid.proposals);
    }
    Metrics final_valid = evaluate(&c, ranker, records, validation,
                                   header.record_count, 1);
    double initial_loss = mean_pair_loss(&initial_valid);
    double final_loss = mean_pair_loss(&final_valid);
    int frozen = aa_nonpolicy_equal(champion, ranker);
    double final_mean_kl = mean_kl(&final_valid);
    int gate_passed = lc_double_isfinite(final_loss) &&
        lc_double_isfinite(final_mean_kl) &&
        lc_double_isfinite(final_valid.max_kl) &&
        final_loss < initial_loss &&
        final_mean_kl <= c.max_validation_kl &&
        final_valid.max_kl <= c.max_state_kl && frozen;
    ThresholdMetrics grid[THRESHOLD_GRID_N];
    int grid_valid = threshold_grid(
        champion, ranker, (int)header.scoring_symmetries,
        records, validation, header.record_count, grid);
    gate_passed = gate_passed && grid_valid;
    for (int g = 0; g < THRESHOLD_GRID_N; g++)
        printf("heldout threshold %.2f: retained %llu, signed hybrid "
               "sum %+.3f mean %+.3f, mistakes %llu, invalid %llu, "
               "oracle regret %.3f\n",
               grid[g].threshold, (unsigned long long)grid[g].retained,
               grid[g].signed_hybrid_sum, grid[g].signed_hybrid_mean,
               (unsigned long long)grid[g].mistakes,
               (unsigned long long)grid[g].invalid_scores,
               grid[g].oracle_regret);
    fputs("METRICS_JSON ", stdout);
    print_metrics_json(stdout, &header, &c, &initial_valid, &final_valid,
                       grid, frozen, gate_passed);
    if (!write_metrics_json(c.metrics_path, &header, &c, &initial_valid,
                            &final_valid, grid, frozen, gate_passed)) {
        fprintf(stderr, "cannot atomically write validation metrics %s\n",
                c.metrics_path);
        gate_passed = 0;
    }
    if (!gate_passed) {
        fprintf(stderr, "training gate failed: validation pair %.6f -> "
                        "%.6f, KL %.6g (max %.6g), frozen %s\n",
                initial_loss, final_loss, mean_kl(&final_valid),
                final_valid.max_kl,
                frozen ? "exact" : "CHANGED");
        free(validation); free(order);
        goto fail;
    }
    printf("training gate passed: validation pair %.6f -> %.6f, KL %.6g "
           "(max %.6g); trunk/value/belief byte-exact\n",
           initial_loss, final_loss, mean_kl(&final_valid),
           final_valid.max_kl);
    int rc = 0;
    if (!c.dry_run && !save_verified(ranker, c.out_path, c.force)) {
        fprintf(stderr, "cannot atomically write and verify %s\n", c.out_path);
        rc = 1;
    } else if (!c.dry_run) {
        printf("wrote residual-log-odds ranker to %s\n", c.out_path);
    }
    free(validation); free(order);
    free(records); free(champion); free(ranker); free(gradient);
    free(adam.m); free(adam.v);
    return rc;

fail:
    free(records); free(champion); free(ranker); free(gradient);
    free(adam.m); free(adam.v);
    return 1;
}
