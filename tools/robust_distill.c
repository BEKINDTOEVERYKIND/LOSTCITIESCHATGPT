#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L

/* robust_distill -- conservative distillation of independently confirmed
 * rollout corrections into the otherwise-unused full-move residual head.
 *
 * This tool is intentionally separate from train.c and rl.c.  It has two
 * safety boundaries:
 *
 *   1. Generation records a preference only when rollout's primary screen and
 *      a fresh random-symmetry/greedy confirmation both support a
 *      challenger to the frozen policy leader.
 *   2. Training updates only Net.wcomb and Net.bcomb.  Every older parameter
 *      remains byte-identical to the frozen network, and a full-legal-action
 *      KL loss anchors the residual policy at every recorded state.
 *
 * The correction file is tied to an exact hash of the frozen Net.  This
 * prevents accidentally applying labels produced by one policy to another.
 *
 * Typical workflow (the defaults use 512, never 64, paired worlds):
 *
 *   ./bin/robust_distill --generate /tmp/lc-corrections.rdc \
 *       --net data/champion.bin --games 8 --max-searches 128
 *
 *   ./bin/robust_distill --inspect /tmp/lc-corrections.rdc
 *
 *   ./bin/robust_distill --train /tmp/lc-corrections.rdc \
 *       --net data/champion.bin --out /tmp/lc-residual-candidate.bin
 *
 * Reviewed information states can be fed through the same statistical gates
 * without hand-authored labels:
 *
 *   ./bin/robust_distill --generate /tmp/reviewed.rdc \
 *       --net data/champion.bin --state /tmp/ply25.state \
 *       --worlds 2048 --confirm-worlds 2048 --semantic-candidates
 *
 * It never replaces or promotes a champion checkpoint.
 */
#include "../src/agent.h"
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/search.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#define RD_MAGIC UINT32_C(0x4C435244) /* "LCRD" */
#define RD_VERSION UINT32_C(2)
#define RD_KIND_ANCHOR UINT32_C(0)
#define RD_KIND_CORRECTION UINT32_C(1)

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t state_size;
    uint32_t record_size;
    uint64_t count;
    uint64_t anchor_count;
    uint64_t correction_count;
    uint64_t anchor_hash;
    uint64_t seed;
    uint32_t worlds;
    uint32_t confirm_worlds;
    uint32_t actor_symmetries;
    uint32_t playout_symmetries;
    uint32_t ply_lo;
    uint32_t ply_hi;
    uint32_t deck_max;
    uint32_t header_size;
    float override_k;
    float override_min;
    float confirm_z;
    uint32_t playout_sample;
    uint32_t root_width;
    uint32_t discard_guard;
    uint32_t playout_prune;
    float policy_mass;
    uint32_t semantic_cand;
    uint32_t plan_deck_max;
    uint32_t plan_block_gap;
} RecordHeader;

typedef struct {
    State st;
    uint32_t kind;
    uint16_t baseline;
    uint16_t challenger;
    float primary_delta;
    float primary_se;
    float confirm_delta;
    float confirm_se;
    float lcb;
} DistillRecord;

typedef struct {
    float *mw;
    float *vw;
    float *mb;
    float *vb;
    uint64_t t;
} ResidualAdam;

typedef struct {
    double kl;
    double pair_loss;
    double pair_target;
    double pair_probability;
    double lcb;
    uint64_t states;
    uint64_t corrections;
} Metrics;

typedef enum {
    MODE_NONE = 0,
    MODE_GENERATE,
    MODE_TRAIN,
    MODE_INSPECT
} Mode;

typedef struct {
    Mode mode;
    const char *records_path;
    const char *net_path;
    const char *out_path;
    uint64_t seed;

    int games;
    int max_searches;
    int search_stride;
    int anchor_stride;
    int worlds;
    int confirm_worlds;
    int root_width;
    int actor_symmetries;
    int playout_symmetries;
    int ply_lo;
    int ply_hi;
    int deck_max;
    float policy_mass;
    float override_k;
    float override_min;
    int discard_guard;
    int playout_prune;
    int semantic_cand;
    int plan_deck_max;
    int plan_block_gap;

    int epochs;
    int batch;
    float lr;
    float kl_weight;
    float pair_scale;
    float lcb_scale;
    float margin_cap;
    float max_pair_weight;
    float max_kl;
    float weight_decay;
    int suit_augment;
    int random_suit_augment;
    int force;
    int dry_run;
    int verbose;
    const char *state_path[128];
    int state_count;
    const char *extra_records[64];
    int extra_record_count;
} Config;

static void usage(FILE *f, const char *argv0)
{
    fprintf(f,
        "usage:\n"
        "  %s --generate RECORDS --net NET [generation options]\n"
        "  %s --train RECORDS --net NET --out CANDIDATE [training options]\n"
        "  %s --inspect RECORDS [--net NET]\n"
        "\n"
        "generation options:\n"
        "  --games N                 champion self-play matches (default 4)\n"
        "  --max-searches N          hard rollout-call cap (default 64)\n"
        "  --state FILE              evaluate this saved text state instead of\n"
        "                            generated self-play (repeatable, max 128)\n"
        "  --search-stride N         search every Nth eligible state (default 4)\n"
        "  --anchor-stride N         retain a KL-only state every N plies (default 8)\n"
        "  --worlds N                primary paired worlds (default 512, min 256)\n"
        "  --confirm-worlds N        fresh confirmation worlds (default 512, min 256)\n"
        "  --root-width N            maximum top-policy moves (default 4)\n"
        "  --policy-mass X           shortest-prefix mass target (default .995)\n"
        "  --actor-symmetries N      root/actor suit ensemble (default 20)\n"
        "  --playout-symmetries N    random-sym greedy continuation group (default 20)\n"
        "  --ply-lo N                first searched round ply (default 14)\n"
        "  --ply-hi N                exclusive last searched ply, 0=off\n"
        "  --deck-max N              require deck_left <= N, 0=off\n"
        "  --override-k X            primary paired-SE gate (default 3.5)\n"
        "  --override-min X          primary practical gap (default 2 points)\n"
        "  --no-discard-guard        permit questionable discard overrides\n"
        "  --no-playout-prune        disable safe-dead-discard continuation focus\n"
        "  --semantic-candidates     add one public-wager pickup and one\n"
        "                            one-sided wager-discard challenger\n"
        "  --plan-deck-max N         use exact visible-hand scheduling with at\n"
        "                            most N deck cards (0=off, default 0)\n"
        "  --plan-block-gap N        scheduler information-preservation threshold\n"
        "                            (0=off, default 0)\n"
        "\n"
        "training options:\n"
        "  --extra-records FILE      add another validated record set tied to\n"
        "                            the same frozen network (repeatable)\n"
        "  --epochs N                passes over records (default 20)\n"
        "  --batch N                 residual-Adam batch size (default 32)\n"
        "  --lr X                    residual learning rate (default 1e-4)\n"
        "  --kl X                    KL(anchor||candidate) coefficient (default 1)\n"
        "  --pair-scale X            correction-loss multiplier (default 1)\n"
        "  --lcb-scale X             points per target-logit unit (default 4)\n"
        "  --margin-cap X            maximum target pair logit gap (default 1)\n"
        "  --max-pair-weight X       cap on LCB-derived example weight (default 2)\n"
        "  --max-kl X                reject an epoch above mean KL (default .01)\n"
        "  --weight-decay X          residual-only weight decay (default 0)\n"
        "  --no-suit-augment         do not deterministically permute suits\n"
        "  --random-suit-augment     sample permutations with replacement\n"
        "                            (default; explicit reproducibility flag)\n"
        "  --exhaustive-suit-augment tour all 120 permutations without replacement\n"
        "  --dry-run                 train and report without writing a model\n"
        "\n"
        "common options:\n"
        "  --seed N                  deterministic seed (default 20260730)\n"
        "  --force                   allow replacing RECORDS/CANDIDATE\n"
        "  --verbose                 list each confirmed correction on inspect\n"
        "  -h, --help                show this help\n"
        "\n"
        "Safety notes: early search (--ply-lo below 14) requires at least 1024\n"
        "primary and confirmation worlds.  Training refuses a nonzero residual\n"
        "head and verifies that every pre-residual byte remains unchanged.\n",
        argv0, argv0, argv0);
}

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static uint64_t hash_bytes(const void *ptr, size_t n)
{
    const unsigned char *p = (const unsigned char *)ptr;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int parse_long(const char *s, long lo, long hi, long *out)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return 0;
    *out = v;
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

static int parse_float(const char *s, float lo, float hi, float *out)
{
    char *end = NULL;
    errno = 0;
    float v = strtof(s, &end);
    if (errno || !end || *end || !lc_float_isfinite(v) || v < lo || v > hi)
        return 0;
    *out = v;
    return 1;
}

static int valid_symmetries(int n)
{
    return n == 1 || n == 5 || n == 10 || n == 20 || n == 120;
}

static int next_value(int argc, char **argv, int *i, const char **value)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", argv[*i]);
        return 0;
    }
    *value = argv[++*i];
    return 1;
}

static int parse_args(int argc, char **argv, Config *c)
{
    memset(c, 0, sizeof *c);
    c->seed = UINT64_C(20260730);
    c->games = 4;
    c->max_searches = 64;
    c->search_stride = 4;
    c->anchor_stride = 8;
    c->worlds = 512;
    c->confirm_worlds = 512;
    c->root_width = 4;
    c->actor_symmetries = 20;
    c->playout_symmetries = 20;
    c->ply_lo = 14;
    c->policy_mass = 0.995f;
    c->override_k = 3.5f;
    c->override_min = 2.0f;
    c->discard_guard = 1;
    c->playout_prune = 1;
    c->epochs = 20;
    c->batch = 32;
    c->lr = 1e-4f;
    c->kl_weight = 1.0f;
    c->pair_scale = 1.0f;
    c->lcb_scale = 4.0f;
    c->margin_cap = 1.0f;
    c->max_pair_weight = 2.0f;
    c->max_kl = 0.01f;
    c->weight_decay = 0.0f;
    c->suit_augment = 1;
    c->random_suit_augment = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = NULL;
        long iv = 0;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(a, "--generate")) {
            if (!next_value(argc, argv, &i, &v)) return 0;
            if (c->mode != MODE_NONE) goto one_mode;
            c->mode = MODE_GENERATE;
            c->records_path = v;
        } else if (!strcmp(a, "--train")) {
            if (!next_value(argc, argv, &i, &v)) return 0;
            if (c->mode != MODE_NONE) goto one_mode;
            c->mode = MODE_TRAIN;
            c->records_path = v;
        } else if (!strcmp(a, "--inspect")) {
            if (!next_value(argc, argv, &i, &v)) return 0;
            if (c->mode != MODE_NONE) goto one_mode;
            c->mode = MODE_INSPECT;
            c->records_path = v;
        } else if (!strcmp(a, "--net")) {
            if (!next_value(argc, argv, &i, &c->net_path)) return 0;
        } else if (!strcmp(a, "--out")) {
            if (!next_value(argc, argv, &i, &c->out_path)) return 0;
        } else if (!strcmp(a, "--seed")) {
            if (!next_value(argc, argv, &i, &v) || !parse_u64(v, &c->seed))
                goto bad_value;
        } else if (!strcmp(a, "--games")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, INT_MAX, &iv)) goto bad_value;
            c->games = (int)iv;
        } else if (!strcmp(a, "--max-searches")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, INT_MAX, &iv)) goto bad_value;
            c->max_searches = (int)iv;
        } else if (!strcmp(a, "--state")) {
            if (!next_value(argc, argv, &i, &v)) return 0;
            if (c->state_count >=
                (int)(sizeof c->state_path / sizeof c->state_path[0])) {
                fprintf(stderr, "too many --state inputs\n");
                return 0;
            }
            c->state_path[c->state_count++] = v;
        } else if (!strcmp(a, "--extra-records")) {
            if (!next_value(argc, argv, &i, &v)) return 0;
            if (c->extra_record_count >=
                (int)(sizeof c->extra_records /
                      sizeof c->extra_records[0])) {
                fprintf(stderr, "too many --extra-records inputs\n");
                return 0;
            }
            c->extra_records[c->extra_record_count++] = v;
        } else if (!strcmp(a, "--search-stride")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, INT_MAX, &iv)) goto bad_value;
            c->search_stride = (int)iv;
        } else if (!strcmp(a, "--anchor-stride")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, INT_MAX, &iv)) goto bad_value;
            c->anchor_stride = (int)iv;
        } else if (!strcmp(a, "--worlds")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, INT_MAX, &iv)) goto bad_value;
            c->worlds = (int)iv;
        } else if (!strcmp(a, "--confirm-worlds")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, INT_MAX, &iv)) goto bad_value;
            c->confirm_worlds = (int)iv;
        } else if (!strcmp(a, "--root-width")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 2, 8, &iv)) goto bad_value;
            c->root_width = (int)iv;
        } else if (!strcmp(a, "--actor-symmetries")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, 120, &iv)) goto bad_value;
            c->actor_symmetries = (int)iv;
        } else if (!strcmp(a, "--playout-symmetries")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, 120, &iv)) goto bad_value;
            c->playout_symmetries = (int)iv;
        } else if (!strcmp(a, "--ply-lo")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 0, LC_MAX_PLIES, &iv)) goto bad_value;
            c->ply_lo = (int)iv;
        } else if (!strcmp(a, "--ply-hi")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 0, LC_MAX_PLIES + 1, &iv)) goto bad_value;
            c->ply_hi = (int)iv;
        } else if (!strcmp(a, "--deck-max")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 0, NCARD, &iv)) goto bad_value;
            c->deck_max = (int)iv;
        } else if (!strcmp(a, "--policy-mass")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.5f, 1.0f, &c->policy_mass)) goto bad_value;
        } else if (!strcmp(a, "--override-k")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 3.5f, 20.0f, &c->override_k)) goto bad_value;
        } else if (!strcmp(a, "--override-min")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.1f, 100.0f, &c->override_min)) goto bad_value;
        } else if (!strcmp(a, "--epochs")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, 100000, &iv)) goto bad_value;
            c->epochs = (int)iv;
        } else if (!strcmp(a, "--batch")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 1, 1000000, &iv)) goto bad_value;
            c->batch = (int)iv;
        } else if (!strcmp(a, "--lr")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 1e-9f, 0.1f, &c->lr)) goto bad_value;
        } else if (!strcmp(a, "--kl")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.0f, 1000.0f, &c->kl_weight)) goto bad_value;
        } else if (!strcmp(a, "--pair-scale")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.0f, 1000.0f, &c->pair_scale)) goto bad_value;
        } else if (!strcmp(a, "--lcb-scale")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.01f, 1000.0f, &c->lcb_scale)) goto bad_value;
        } else if (!strcmp(a, "--margin-cap")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.0f, 20.0f, &c->margin_cap)) goto bad_value;
        } else if (!strcmp(a, "--max-pair-weight")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.01f, 100.0f, &c->max_pair_weight)) goto bad_value;
        } else if (!strcmp(a, "--max-kl")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.0f, 10.0f, &c->max_kl)) goto bad_value;
        } else if (!strcmp(a, "--weight-decay")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_float(v, 0.0f, 1.0f, &c->weight_decay)) goto bad_value;
        } else if (!strcmp(a, "--no-suit-augment")) {
            c->suit_augment = 0;
        } else if (!strcmp(a, "--suit-augment")) {
            c->suit_augment = 1;
        } else if (!strcmp(a, "--random-suit-augment")) {
            c->suit_augment = 1;
            c->random_suit_augment = 1;
        } else if (!strcmp(a, "--exhaustive-suit-augment")) {
            c->suit_augment = 1;
            c->random_suit_augment = 0;
        } else if (!strcmp(a, "--no-discard-guard")) {
            c->discard_guard = 0;
        } else if (!strcmp(a, "--no-playout-prune")) {
            c->playout_prune = 0;
        } else if (!strcmp(a, "--semantic-candidates")) {
            c->semantic_cand = 1;
        } else if (!strcmp(a, "--plan-deck-max")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 0, NCARD, &iv)) goto bad_value;
            c->plan_deck_max = (int)iv;
        } else if (!strcmp(a, "--plan-block-gap")) {
            if (!next_value(argc, argv, &i, &v) ||
                !parse_long(v, 0, 1000, &iv)) goto bad_value;
            c->plan_block_gap = (int)iv;
        } else if (!strcmp(a, "--force")) {
            c->force = 1;
        } else if (!strcmp(a, "--dry-run")) {
            c->dry_run = 1;
        } else if (!strcmp(a, "--verbose")) {
            c->verbose = 1;
        } else {
            fprintf(stderr, "unknown option: %s\n", a);
            return 0;
        }
        continue;

    one_mode:
        fprintf(stderr, "choose exactly one of --generate, --train, --inspect\n");
        return 0;
    bad_value:
        fprintf(stderr, "invalid value for %s\n", a);
        return 0;
    }

    if (c->mode == MODE_NONE || !c->records_path) {
        fprintf(stderr, "choose --generate, --train, or --inspect\n");
        return 0;
    }
    if ((c->mode == MODE_GENERATE || c->mode == MODE_TRAIN) && !c->net_path) {
        fprintf(stderr, "--net is required for generation and training\n");
        return 0;
    }
    if (c->mode == MODE_TRAIN && !c->out_path && !c->dry_run) {
        fprintf(stderr, "--out is required for training unless --dry-run is used\n");
        return 0;
    }
    if (c->mode == MODE_TRAIN && c->out_path &&
        !strcmp(c->out_path, c->net_path)) {
        fprintf(stderr, "--out must not be the frozen --net path\n");
        return 0;
    }
    if (c->mode != MODE_TRAIN && c->extra_record_count > 0) {
        fprintf(stderr, "--extra-records is only valid with --train\n");
        return 0;
    }
    if (c->mode == MODE_GENERATE) {
        if (!valid_symmetries(c->actor_symmetries) ||
            !valid_symmetries(c->playout_symmetries)) {
            fprintf(stderr, "suit symmetry count must be 1, 5, 10, 20, or 120\n");
            return 0;
        }
        if (c->worlds < 256 || c->confirm_worlds < 256) {
            fprintf(stderr, "generation requires at least 256 primary and "
                            "confirmation worlds\n");
            return 0;
        }
        if (c->ply_lo < 14 &&
            (c->worlds < 1024 || c->confirm_worlds < 1024)) {
            fprintf(stderr, "early-round generation (--ply-lo < 14) requires "
                            "at least 1024 primary and confirmation worlds\n");
            return 0;
        }
        if (c->ply_hi > 0 && c->ply_hi <= c->ply_lo) {
            fprintf(stderr, "--ply-hi must be greater than --ply-lo\n");
            return 0;
        }
        if ((c->plan_deck_max > 0) != (c->plan_block_gap > 0)) {
            fprintf(stderr, "--plan-deck-max and --plan-block-gap must both "
                            "be positive, or both be zero\n");
            return 0;
        }
    }
    if (c->mode != MODE_GENERATE && c->state_count > 0) {
        fprintf(stderr, "--state is only valid with --generate\n");
        return 0;
    }
    return 1;
}

static int path_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

/* Resolve an output path even before the final file exists.  realpath(3)
 * handles an existing path directly; otherwise resolve its parent and append
 * the final component.  This catches aliases such as ./x, ../dir/x and
 * symlinked parent directories before an atomic rename can replace an input. */
static int canonical_output_path(const char *path, char out[PATH_MAX])
{
    if (realpath(path, out)) return 1;

    size_t len = strlen(path);
    if (len == 0 || len >= PATH_MAX) return 0;
    char copy[PATH_MAX];
    memcpy(copy, path, len + 1);
    char *slash = strrchr(copy, '/');
    const char *base;
    const char *dir;
    if (!slash) {
        dir = ".";
        base = copy;
    } else {
        base = slash + 1;
        if (!*base) return 0;
        if (slash == copy) {
            dir = "/";
        } else {
            *slash = '\0';
            dir = copy;
        }
    }

    char parent[PATH_MAX];
    if (!realpath(dir, parent)) return 0;
    int n = snprintf(out, PATH_MAX, "%s%s%s", parent,
                     !strcmp(parent, "/") ? "" : "/", base);
    return n > 0 && n < PATH_MAX;
}

static int paths_alias(const char *a, const char *b)
{
    if (!a || !b) return 0;
    if (!strcmp(a, b)) return 1;

    struct stat sa, sb;
    if (stat(a, &sa) == 0 && stat(b, &sb) == 0 &&
        sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino)
        return 1;

    char ca[PATH_MAX], cb[PATH_MAX];
    return canonical_output_path(a, ca) &&
           canonical_output_path(b, cb) &&
           !strcmp(ca, cb);
}

static FILE *open_atomic_output(const char *dest, int force,
                                char temp_path[PATH_MAX])
{
    if (!force && path_exists(dest)) {
        fprintf(stderr, "%s already exists; use --force to replace it\n",
                dest);
        return NULL;
    }
    int n = snprintf(temp_path, PATH_MAX, "%s.tmp.XXXXXX", dest);
    if (n < 0 || n >= PATH_MAX) {
        fprintf(stderr, "output path is too long: %s\n", dest);
        return NULL;
    }
    int fd = mkstemp(temp_path);
    if (fd < 0) {
        fprintf(stderr, "cannot create temporary output for %s: %s\n",
                dest, strerror(errno));
        return NULL;
    }
    FILE *f = fdopen(fd, "w+b");
    if (!f) {
        fprintf(stderr, "cannot open temporary output for %s: %s\n",
                dest, strerror(errno));
        close(fd);
        unlink(temp_path);
        return NULL;
    }
    return f;
}

static int commit_atomic_output(FILE *f, const char *temp_path,
                                const char *dest, int force)
{
    int ok = fflush(f) == 0;
    if (ok && fsync(fileno(f)) != 0) ok = 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "cannot flush %s: %s\n", temp_path, strerror(errno));
        unlink(temp_path);
        return 0;
    }
    int installed;
    if (force) {
        installed = rename(temp_path, dest) == 0;
    } else {
        /* link() is an atomic no-clobber install on the same filesystem.
         * The temporary file lives beside dest specifically for this. */
        installed = link(temp_path, dest) == 0;
        if (installed && unlink(temp_path) != 0) {
            fprintf(stderr, "installed %s but could not remove temporary %s: "
                            "%s\n", dest, temp_path, strerror(errno));
            return 0;
        }
    }
    if (!installed) {
        fprintf(stderr, "cannot install %s: %s\n", dest, strerror(errno));
        unlink(temp_path);
        return 0;
    }
    return 1;
}

static int save_net_atomic(const Net *net, const char *dest, int force)
{
    if (!force && path_exists(dest)) {
        fprintf(stderr, "%s already exists; use --force to replace it\n",
                dest);
        return 0;
    }
    char temp_path[PATH_MAX];
    int n = snprintf(temp_path, sizeof temp_path, "%s.tmp.XXXXXX", dest);
    if (n < 0 || n >= (int)sizeof temp_path) {
        fprintf(stderr, "output path is too long: %s\n", dest);
        return 0;
    }
    int fd = mkstemp(temp_path);
    if (fd < 0) {
        fprintf(stderr, "cannot create temporary model for %s: %s\n",
                dest, strerror(errno));
        return 0;
    }
    if (close(fd) != 0 || net_save(net, temp_path) != 0) {
        fprintf(stderr, "cannot save temporary model for %s\n", dest);
        unlink(temp_path);
        return 0;
    }
    fd = open(temp_path, O_RDONLY);
    int ok = fd >= 0 && fsync(fd) == 0;
    if (fd >= 0 && close(fd) != 0) ok = 0;
    int installed = 0;
    if (ok) {
        if (force) installed = rename(temp_path, dest) == 0;
        else {
            installed = link(temp_path, dest) == 0;
            if (installed && unlink(temp_path) != 0) {
                fprintf(stderr, "installed %s but could not remove temporary "
                                "%s: %s\n", dest, temp_path, strerror(errno));
                return 0;
            }
        }
    }
    if (!installed) {
        fprintf(stderr, "cannot install %s: %s\n", dest, strerror(errno));
        unlink(temp_path);
        return 0;
    }
    return 1;
}

static Net *load_net(const char *path)
{
    Net *n = (Net *)malloc(sizeof *n);
    if (!n) {
        fprintf(stderr, "out of memory loading network\n");
        return NULL;
    }
    int rc = net_load(n, path);
    if (rc != 0) {
        fprintf(stderr, "cannot load %s (error %d)\n", path, rc);
        free(n);
        return NULL;
    }
    return n;
}

static int residual_is_zero(const Net *n)
{
    for (int i = 0; i < NET_NCOMB; i++) {
        if (n->bcomb[i] != 0.0f) return 0;
        for (int h = 0; h < NET_H2; h++)
            if (n->wcomb[i][h] != 0.0f) return 0;
    }
    return 1;
}

static Move unpack_move(uint16_t packed)
{
    Move m = { MOVE_CARD(packed), MOVE_DISC(packed), MOVE_DRAW(packed) };
    return m;
}

static uint16_t semantic_key(Move m)
{
    if (CARD_IS_WAGER(m.card))
        m.card = (uint8_t)CARD_MAKE(CARD_SUIT(m.card), 0);
    return MOVE_PACK(m);
}

static int same_move(Move a, Move b)
{
    return semantic_key(a) == semantic_key(b);
}

static int validate_state_payload(const State *st);

static int text_card_id(const char *name, uint64_t used)
{
    char shown[8];
    for (int card = 0; card < NCARD; card++) {
        lc_card_name(card, shown);
        if (!strcasecmp(shown, name) &&
            !((used >> card) & UINT64_C(1)))
            return card;
    }
    return -1;
}

/* Saved review states intentionally omit deck order.  Policy features and
 * root determinization do not consume that hidden order, but State remains a
 * complete engine object, so fill the inactive prefix from occupied cards and
 * the active suffix from the unseen complement. */
static int complete_text_state_deck(State *st)
{
    const uint64_t valid = (UINT64_C(1) << NCARD) - UINT64_C(1);
    uint64_t occupied = st->hand[0] | st->hand[1] |
                        st->played[0] | st->played[1] | st->discarded;
    uint64_t unseen = valid & ~occupied;
    if (__builtin_popcountll(unseen) != st->deck_left ||
        __builtin_popcountll(occupied) != NCARD - st->deck_left)
        return 0;
    st->deck_pos = (uint8_t)(NCARD - st->deck_left);
    int before = 0, active = st->deck_pos;
    for (int card = 0; card < NCARD; card++) {
        uint64_t bit = UINT64_C(1) << card;
        if (occupied & bit) st->deck[before++] = (uint8_t)card;
        else st->deck[active++] = (uint8_t)card;
    }
    return before == st->deck_pos && active == NCARD;
}

static int load_text_state(const char *path, State *st)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open saved state %s: %s\n",
                path, strerror(errno));
        return 0;
    }
    memset(st, 0, sizeof *st);
    uint64_t used = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *tok = strtok(line, " \t\r\n");
        if (!tok) continue;
        if (!strcmp(tok, "turn")) {
            char *v = strtok(NULL, " \t\r\n");
            long value;
            if (!v || !parse_long(v, 0, 1, &value)) goto bad;
            st->turn = (uint8_t)value;
        } else if (!strcmp(tok, "round")) {
            char *v = strtok(NULL, " \t\r\n");
            long value;
            if (!v || !parse_long(v, 0, MATCH_ROUNDS - 1, &value)) goto bad;
            st->round = (uint8_t)value;
        } else if (!strcmp(tok, "nply")) {
            char *v = strtok(NULL, " \t\r\n");
            long value;
            if (!v || !parse_long(v, 0, LC_MAX_PLIES - 1, &value)) goto bad;
            st->nply = (uint16_t)value;
        } else if (!strcmp(tok, "deck_left")) {
            char *v = strtok(NULL, " \t\r\n");
            long value;
            if (!v || !parse_long(v, 1, NCARD, &value)) goto bad;
            st->deck_left = (uint8_t)value;
        } else if (!strcmp(tok, "cum")) {
            char *a = strtok(NULL, " \t\r\n");
            char *b = strtok(NULL, " \t\r\n");
            long av, bv;
            if (!a || !b ||
                !parse_long(a, INT16_MIN, INT16_MAX, &av) ||
                !parse_long(b, INT16_MIN, INT16_MAX, &bv))
                goto bad;
            st->cum[0] = (int16_t)av;
            st->cum[1] = (int16_t)bv;
        } else if (!strncmp(tok, "hand", 4)) {
            int p = tok[4] - '0';
            if (p < 0 || p > 1) goto bad;
            char *word;
            while ((word = strtok(NULL, " \t\r\n"))) {
                int card = text_card_id(word, used);
                if (card < 0) goto bad;
                used |= UINT64_C(1) << card;
                st->hand[p] |= UINT64_C(1) << card;
                st->hand_n[p]++;
            }
        } else if (!strncmp(tok, "known", 5)) {
            int p = tok[5] - '0';
            if (p < 0 || p > 1) goto bad;
            char *word;
            while ((word = strtok(NULL, " \t\r\n"))) {
                char shown[8];
                int found = 0;
                for (int card = 0; card < NCARD; card++) {
                    lc_card_name(card, shown);
                    if (!strcasecmp(shown, word) &&
                        ((st->hand[p] >> card) & UINT64_C(1)) &&
                        !((st->known[p] >> card) & UINT64_C(1))) {
                        st->known[p] |= UINT64_C(1) << card;
                        found = 1;
                        break;
                    }
                }
                if (!found) goto bad;
            }
        } else if (!strcmp(tok, "exp")) {
            char *ps = strtok(NULL, " \t\r\n");
            char *ss = strtok(NULL, " \t\r\n");
            long pv, sv;
            if (!ps || !ss || !parse_long(ps, 0, 1, &pv) ||
                !parse_long(ss, 0, NSUIT - 1, &sv))
                goto bad;
            int p = (int)pv, suit = (int)sv;
            char *word;
            while ((word = strtok(NULL, " \t\r\n"))) {
                int card = text_card_id(word, used);
                if (card < 0 || CARD_SUIT(card) != suit) goto bad;
                used |= UINT64_C(1) << card;
                st->played[p] |= UINT64_C(1) << card;
                st->exp_n[p][suit]++;
                if (CARD_IS_WAGER(card)) {
                    st->exp_wager[p][suit]++;
                } else {
                    int value = CARD_VALUE(card);
                    st->exp_top[p][suit] = (uint8_t)value;
                    st->exp_sum[p][suit] += (uint8_t)value;
                }
            }
        } else if (!strcmp(tok, "pile")) {
            char *ss = strtok(NULL, " \t\r\n");
            long sv;
            if (!ss || !parse_long(ss, 0, NSUIT - 1, &sv)) goto bad;
            int suit = (int)sv;
            char *word;
            while ((word = strtok(NULL, " \t\r\n"))) {
                int card = text_card_id(word, used);
                if (card < 0 || CARD_SUIT(card) != suit ||
                    st->pile_n[suit] >= NRANK)
                    goto bad;
                used |= UINT64_C(1) << card;
                st->pile[suit][st->pile_n[suit]++] = (uint8_t)card;
                st->discarded |= UINT64_C(1) << card;
            }
        } else goto bad;
    }
    int read_error = ferror(f);
    int close_error = fclose(f);
    if (read_error || close_error != 0 || !complete_text_state_deck(st)) {
        fprintf(stderr, "invalid saved state %s\n", path);
        return 0;
    }
    return 1;

bad:
    fclose(f);
    fprintf(stderr, "invalid saved state %s\n", path);
    return 0;
}

static int write_record(FILE *f, const DistillRecord *r)
{
    return fwrite(r, sizeof *r, 1, f) == 1;
}

static int generate_records(const Config *c)
{
    Net *anchor = load_net(c->net_path);
    if (!anchor) return 1;
    if (!residual_is_zero(anchor)) {
        fprintf(stderr, "%s has a nonzero full-move residual; this conservative "
                        "stage requires an unused residual head\n", c->net_path);
        free(anchor);
        return 1;
    }

    char temp_path[PATH_MAX];
    FILE *out =
        open_atomic_output(c->records_path, c->force, temp_path);
    if (!out) {
        free(anchor);
        return 1;
    }

    RecordHeader hdr;
    memset(&hdr, 0, sizeof hdr);
    hdr.magic = RD_MAGIC;
    hdr.version = RD_VERSION;
    hdr.state_size = (uint32_t)sizeof(State);
    hdr.record_size = (uint32_t)sizeof(DistillRecord);
    hdr.anchor_hash = hash_bytes(anchor, sizeof *anchor);
    hdr.seed = c->seed;
    hdr.worlds = (uint32_t)c->worlds;
    hdr.confirm_worlds = (uint32_t)c->confirm_worlds;
    hdr.actor_symmetries = (uint32_t)c->actor_symmetries;
    hdr.playout_symmetries = (uint32_t)c->playout_symmetries;
    hdr.ply_lo = (uint32_t)c->ply_lo;
    hdr.ply_hi = (uint32_t)c->ply_hi;
    hdr.deck_max = (uint32_t)c->deck_max;
    hdr.header_size = (uint32_t)sizeof hdr;
    hdr.override_k = c->override_k;
    hdr.override_min = c->override_min;
    hdr.confirm_z = 2.58f;
    hdr.playout_sample = 2;
    hdr.root_width = (uint32_t)c->root_width;
    hdr.discard_guard = (uint32_t)c->discard_guard;
    hdr.playout_prune = (uint32_t)c->playout_prune;
    hdr.policy_mass = c->policy_mass;
    hdr.semantic_cand = (uint32_t)c->semantic_cand;
    hdr.plan_deck_max = (uint32_t)c->plan_deck_max;
    hdr.plan_block_gap = (uint32_t)c->plan_block_gap;
    if (fwrite(&hdr, sizeof hdr, 1, out) != 1) {
        fprintf(stderr, "cannot write %s\n", c->records_path);
        fclose(out);
        unlink(temp_path);
        free(anchor);
        return 1;
    }

    Agent evaluator;
    agent_default(&evaluator, AG_ROLLOUT, anchor);
    evaluator.dets = c->worlds;
    evaluator.root_width = c->root_width;
    evaluator.cand_mass = c->policy_mass;
    evaluator.min_cand = 2;
    evaluator.cand_floor = 0.0f;
    evaluator.gate = 0.0f;
    evaluator.ply_lo = c->ply_lo;
    evaluator.ply_hi = c->ply_hi;
    evaluator.deck_max = c->deck_max;
    evaluator.eval_cand = 0;
    evaluator.batch_dets = 0; /* always pay for the configured primary worlds */
    evaluator.symmetries = c->actor_symmetries;
    evaluator.playout_symmetries = c->playout_symmetries;
    evaluator.confirm_dets = c->confirm_worlds;
    evaluator.win_q = 2;
    evaluator.no_belief = 1;
    evaluator.prune_dom = 0;
    evaluator.playout_prune = c->playout_prune;
    evaluator.discard_guard = c->discard_guard;
    evaluator.semantic_cand = c->semantic_cand;
    evaluator.plan_deck_max = c->plan_deck_max;
    evaluator.plan_block_gap = c->plan_block_gap;
    evaluator.override_k = c->override_k;
    evaluator.override_min = c->override_min;
    /* One random member of the exact suit group per continuation decision,
     * followed by a greedy action.  Repeated policy probabilities average to
     * the exact ensemble, although argmax is nonlinear, so this is a cheap
     * stochastic approximation rather than an exact one.  Crucially it is
     * not mode 1's high-entropy policy-action sampling. */
    evaluator.playout_sample = 2;

    uint64_t count = 0, anchors = 0, corrections = 0;
    int searches = 0;
    uint64_t eligible_seen = 0;

    if (c->state_count > 0) {
        for (int si = 0; si < c->state_count; si++) {
            State st;
            if (!load_text_state(c->state_path[si], &st) ||
                !validate_state_payload(&st)) {
                fprintf(stderr, "saved state failed engine validation: %s\n",
                        c->state_path[si]);
                goto generation_error;
            }
            if (st.nply < c->ply_lo ||
                (c->ply_hi > 0 && st.nply >= c->ply_hi) ||
                (c->deck_max > 0 && st.deck_left > c->deck_max)) {
                fprintf(stderr, "saved state falls outside configured phase: "
                                "%s\n", c->state_path[si]);
                goto generation_error;
            }
            Rng search_rng;
            rng_seed(&search_rng, mix64(c->seed ^
                     ((uint64_t)(unsigned)si << 32) ^
                     ((uint64_t)st.nply << 8) ^
                     UINT64_C(0x535441544553)));
            SearchStats ss;
            Move selected =
                rollout_move(&evaluator, &st, &search_rng, NULL, &ss);
            searches++;
            int ci = -1;
            for (int i = 1; i < ss.n; i++)
                if (same_move(ss.mv[i], selected)) {
                    ci = i;
                    break;
                }
            int wrote = 0;
            if (ss.confirmed && ci > 0 && ss.csupported[ci] &&
                !ss.guard_rejected[ci]) {
                double plcb =
                    ss.delta[ci] -
                    (double)c->override_k * ss.dse[ci];
                double clcb =
                    ss.cdelta[ci] - 2.58 * ss.cdse[ci];
                double lcb = plcb < clcb ? plcb : clcb;
                if (lc_double_isfinite(lcb) && lcb > 0.0) {
                    DistillRecord r;
                    memset(&r, 0, sizeof r);
                    r.st = st;
                    r.kind = RD_KIND_CORRECTION;
                    r.baseline = MOVE_PACK(ss.mv[0]);
                    r.challenger = MOVE_PACK(ss.mv[ci]);
                    r.primary_delta = (float)ss.delta[ci];
                    r.primary_se = (float)ss.dse[ci];
                    r.confirm_delta = (float)ss.cdelta[ci];
                    r.confirm_se = (float)ss.cdse[ci];
                    r.lcb = (float)lcb;
                    if (!write_record(out, &r)) goto write_error;
                    count++;
                    corrections++;
                    wrote = 1;
                }
            }
            if (!wrote) {
                DistillRecord r;
                memset(&r, 0, sizeof r);
                r.st = st;
                r.kind = RD_KIND_ANCHOR;
                if (!write_record(out, &r)) goto write_error;
                count++;
                anchors++;
            }
            printf("evaluated saved state %d/%d: %s (%s)\n",
                   si + 1, c->state_count, c->state_path[si],
                   wrote ? "confirmed correction" : "anchor only");
            fflush(stdout);
        }
        goto finalize_records;
    }

    for (int game = 0; game < c->games; game++) {
        int cumulative[2] = { 0, 0 };
        for (int round = 0; round < MATCH_ROUNDS; round++) {
            int64_t units = (int64_t)c->games * MATCH_ROUNDS;
            int64_t unit = (int64_t)game * MATCH_ROUNDS + round;
            int round_quota =
                (int)((int64_t)c->max_searches / units) +
                (unit < c->max_searches % units ? 1 : 0);
            int round_searches = 0;
            Rng deal_rng;
            rng_seed(&deal_rng, mix64(c->seed ^
                     ((uint64_t)(unsigned)game << 32) ^
                     ((uint64_t)(unsigned)round << 24) ^
                     UINT64_C(0x4445414C)));
            State st;
            lc_deal(&st, &deal_rng);
            st.round = (uint8_t)round;
            st.turn = (uint8_t)(round & 1);
            st.cum[0] = (int16_t)cumulative[0];
            st.cum[1] = (int16_t)cumulative[1];

            while (!st.over) {
                int in_window =
                    st.nply >= c->ply_lo &&
                    (c->ply_hi <= 0 || st.nply < c->ply_hi) &&
                    (c->deck_max <= 0 || st.deck_left <= c->deck_max);
                int wrote = 0;
                if (in_window) {
                    uint64_t ordinal = eligible_seen++;
                    int should_search =
                        round_searches < round_quota &&
                        ordinal % (uint64_t)c->search_stride == 0;
                    if (should_search) {
                        Rng search_rng;
                        rng_seed(&search_rng, mix64(c->seed ^
                                 ((uint64_t)(unsigned)game << 40) ^
                                 ((uint64_t)(unsigned)round << 32) ^
                                 ((uint64_t)st.nply << 8) ^
                                 UINT64_C(0x534541524348)));
                        SearchStats ss;
                        Move selected =
                            rollout_move(&evaluator, &st, &search_rng, NULL, &ss);
                        searches++;
                        round_searches++;
                        int ci = -1;
                        for (int i = 1; i < ss.n; i++)
                            if (same_move(ss.mv[i], selected)) {
                                ci = i;
                                break;
                            }
                        if (ss.confirmed && ci > 0 && ss.csupported[ci] &&
                            !ss.guard_rejected[ci]) {
                            double plcb =
                                ss.delta[ci] -
                                (double)c->override_k * ss.dse[ci];
                            double clcb =
                                ss.cdelta[ci] - 2.58 * ss.cdse[ci];
                            double lcb = plcb < clcb ? plcb : clcb;
                            if (lc_double_isfinite(lcb) && lcb > 0.0) {
                                DistillRecord r;
                                memset(&r, 0, sizeof r);
                                r.st = st;
                                r.kind = RD_KIND_CORRECTION;
                                r.baseline = MOVE_PACK(ss.mv[0]);
                                r.challenger = MOVE_PACK(ss.mv[ci]);
                                r.primary_delta = (float)ss.delta[ci];
                                r.primary_se = (float)ss.dse[ci];
                                r.confirm_delta = (float)ss.cdelta[ci];
                                r.confirm_se = (float)ss.cdse[ci];
                                r.lcb = (float)lcb;
                                if (!write_record(out, &r)) goto write_error;
                                count++;
                                corrections++;
                                wrote = 1;
                            }
                        }
                    }
                    if (!wrote &&
                        (int)st.nply % c->anchor_stride == 0) {
                        DistillRecord r;
                        memset(&r, 0, sizeof r);
                        r.st = st;
                        r.kind = RD_KIND_ANCHOR;
                        if (!write_record(out, &r)) goto write_error;
                        count++;
                        anchors++;
                    }
                }

                Move mv[MAX_MOVES];
                float prob[MAX_MOVES];
                int n = policy_probs_sym(anchor, &st, mv, prob, NULL,
                                         c->actor_symmetries);
                if (n <= 0) {
                    fprintf(stderr, "no legal policy move in a live state\n");
                    goto generation_error;
                }
                int best = 0;
                for (int i = 1; i < n; i++)
                    if (prob[i] > prob[best]) best = i;
                lc_apply(&st, mv[best]);
            }
            cumulative[0] += lc_score(&st, 0);
            cumulative[1] += lc_score(&st, 1);
        }
        printf("generated game %d/%d: searches %d/%d, corrections %llu, "
               "anchors %llu\n", game + 1, c->games, searches,
               c->max_searches, (unsigned long long)corrections,
               (unsigned long long)anchors);
        fflush(stdout);
    }

finalize_records:
    hdr.count = count;
    hdr.anchor_count = anchors;
    hdr.correction_count = corrections;
    if (fseek(out, 0, SEEK_SET) != 0 ||
        fwrite(&hdr, sizeof hdr, 1, out) != 1) {
        fprintf(stderr, "cannot finalize %s\n", c->records_path);
        fclose(out);
        unlink(temp_path);
        free(anchor);
        return 1;
    }
    if (!commit_atomic_output(out, temp_path, c->records_path, c->force)) {
        free(anchor);
        return 1;
    }
    printf("wrote %llu records (%llu confirmed corrections, %llu KL anchors) "
           "to %s\n", (unsigned long long)count,
           (unsigned long long)corrections,
           (unsigned long long)anchors, c->records_path);
    printf("frozen net hash: %016llx\n",
           (unsigned long long)hdr.anchor_hash);
    free(anchor);
    return 0;

write_error:
    fprintf(stderr, "write failed for %s\n", c->records_path);
generation_error:
    fclose(out);
    unlink(temp_path);
    free(anchor);
    return 1;
}

static int read_header(FILE *f, RecordHeader *h, const char *path)
{
    if (fread(h, sizeof *h, 1, f) != 1) {
        fprintf(stderr, "%s: short record header\n", path);
        return 0;
    }
    if (h->magic != RD_MAGIC ||
        (h->version != 1U && h->version != RD_VERSION) ||
        h->header_size != sizeof(RecordHeader) ||
        h->state_size != sizeof(State) ||
        h->record_size != sizeof(DistillRecord)) {
        fprintf(stderr, "%s: incompatible robust-distill record format\n", path);
        return 0;
    }
    /* Version 1 reserved these final three words and writers zeroed them.
     * Interpret them as disabled even if a hostile legacy file filled the
     * reserved bytes with junk. */
    if (h->version == 1U) {
        h->semantic_cand = 0;
        h->plan_deck_max = 0;
        h->plan_block_gap = 0;
    }
    if (!lc_float_isfinite(h->override_k) ||
        !lc_float_isfinite(h->override_min) ||
        !lc_float_isfinite(h->confirm_z) || h->override_k < 3.5f ||
        h->override_min <= 0.0f || h->confirm_z <= 0.0f) {
        fprintf(stderr, "%s: invalid confirmation gates in header\n", path);
        return 0;
    }
    if (h->playout_sample != 2 || h->root_width < 2 ||
        h->root_width > 8 || h->discard_guard > 1 ||
        h->playout_prune > 1 || !lc_float_isfinite(h->policy_mass) ||
        h->policy_mass < 0.5f || h->policy_mass > 1.0f ||
        h->semantic_cand > 1 || h->plan_deck_max > NCARD ||
        h->plan_block_gap > 1000 ||
        ((h->plan_deck_max > 0) != (h->plan_block_gap > 0))) {
        fprintf(stderr, "%s: invalid rollout configuration in header\n", path);
        return 0;
    }
    if (h->worlds < 256 || h->confirm_worlds < 256 ||
        !valid_symmetries((int)h->actor_symmetries) ||
        !valid_symmetries((int)h->playout_symmetries) ||
        h->ply_lo > LC_MAX_PLIES ||
        h->ply_hi > LC_MAX_PLIES + 1u ||
        (h->ply_hi > 0 && h->ply_hi <= h->ply_lo) ||
        h->deck_max > NCARD ||
        (h->ply_lo < 14 &&
         (h->worlds < 1024 || h->confirm_worlds < 1024))) {
        fprintf(stderr, "%s: unsafe world count, phase, or symmetry metadata\n",
                path);
        return 0;
    }
    if (h->count > UINT32_MAX ||
        h->count > SIZE_MAX / sizeof(DistillRecord)) {
        fprintf(stderr, "%s: record count is too large\n", path);
        return 0;
    }
    return 1;
}

static DistillRecord *load_records(const char *path, RecordHeader *h)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (!read_header(f, h, path)) {
        fclose(f);
        return NULL;
    }
    DistillRecord *r =
        (DistillRecord *)malloc((size_t)(h->count ? h->count : 1) * sizeof *r);
    if (!r) {
        fprintf(stderr, "out of memory loading %s\n", path);
        fclose(f);
        return NULL;
    }
    if (h->count &&
        fread(r, sizeof *r, (size_t)h->count, f) != (size_t)h->count) {
        fprintf(stderr, "%s: short record payload\n", path);
        free(r);
        fclose(f);
        return NULL;
    }
    int extra = fgetc(f);
    if (extra != EOF || ferror(f)) {
        fprintf(stderr, "%s: trailing or unreadable record data\n", path);
        free(r);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return r;
}

/* Records are an input format, not trusted in-memory engine states.  Validate
 * every field that can become an array index, bit shift, or cached rule value
 * before calling lc_moves(), feature extraction, or suit augmentation. */
static int validate_state_payload(const State *st)
{
    const uint64_t valid_mask = (UINT64_C(1) << NCARD) - 1;

    if (st->over || st->turn > 1 || st->round >= MATCH_ROUNDS ||
        st->nply >= LC_MAX_PLIES)
        return 0;
    if (st->turn != (uint8_t)((st->round & 1u) ^ (st->nply & 1u)))
        return 0;
    if (st->deck_pos < 2 * HAND_SIZE || st->deck_pos >= NCARD ||
        st->deck_left == 0 ||
        (int)st->deck_left != NCARD - (int)st->deck_pos ||
        (int)st->deck_pos - 2 * HAND_SIZE > (int)st->nply)
        return 0;

    uint64_t deck_permutation = 0;
    for (int i = 0; i < NCARD; i++) {
        uint8_t card = st->deck[i];
        if (card >= NCARD) return 0;
        uint64_t bit = UINT64_C(1) << card;
        if (deck_permutation & bit) return 0;
        deck_permutation |= bit;
    }
    if (deck_permutation != valid_mask) return 0;

    for (int p = 0; p < 2; p++) {
        if ((st->hand[p] | st->played[p] | st->known[p]) & ~valid_mask)
            return 0;
        if (st->hand_n[p] != HAND_SIZE ||
            __builtin_popcountll(st->hand[p]) != st->hand_n[p] ||
            (st->known[p] & ~st->hand[p]))
            return 0;
    }
    if (st->discarded & ~valid_mask) return 0;

    uint64_t pile_mask = 0;
    for (int suit = 0; suit < NSUIT; suit++) {
        if (st->pile_n[suit] > NRANK) return 0;
        /* Suit augmentation permutes the entire fixed-size pile array,
         * including inactive slots retained after a pile draw. */
        for (int i = 0; i < NRANK; i++)
            if (st->pile[suit][i] >= NCARD) return 0;
        for (int i = 0; i < st->pile_n[suit]; i++) {
            uint8_t card = st->pile[suit][i];
            if (card >= NCARD || CARD_SUIT(card) != suit) return 0;
            uint64_t bit = UINT64_C(1) << card;
            if (pile_mask & bit) return 0;
            pile_mask |= bit;
        }
    }
    if (pile_mask != st->discarded) return 0;

    uint64_t locations = 0;
    const uint64_t public_parts[5] = {
        st->hand[0], st->hand[1], st->played[0], st->played[1],
        st->discarded
    };
    for (int i = 0; i < 5; i++) {
        if (locations & public_parts[i]) return 0;
        locations |= public_parts[i];
    }
    for (int i = st->deck_pos; i < NCARD; i++) {
        uint64_t bit = UINT64_C(1) << st->deck[i];
        if (locations & bit) return 0;
        locations |= bit;
    }
    if (locations != valid_mask) return 0;

    for (int p = 0; p < 2; p++) {
        for (int suit = 0; suit < NSUIT; suit++) {
            int n = 0, wagers = 0, sum = 0, top = 0;
            int lo = suit * NRANK, hi = lo + NRANK;
            for (int card = lo; card < hi; card++) {
                if (!((st->played[p] >> card) & UINT64_C(1))) continue;
                n++;
                if (CARD_IS_WAGER(card)) {
                    wagers++;
                } else {
                    int value = CARD_VALUE(card);
                    sum += value;
                    if (value > top) top = value;
                }
            }
            if (st->exp_n[p][suit] != n ||
                st->exp_wager[p][suit] != wagers ||
                st->exp_sum[p][suit] != sum ||
                st->exp_top[p][suit] != top)
                return 0;
        }
    }
    return 1;
}

static int validate_record(const DistillRecord *r, const RecordHeader *h,
                           uint64_t index)
{
    if (r->kind != RD_KIND_ANCHOR && r->kind != RD_KIND_CORRECTION) {
        fprintf(stderr, "record %llu has invalid kind %u\n",
                (unsigned long long)index, r->kind);
        return 0;
    }
    if (!validate_state_payload(&r->st)) {
        fprintf(stderr, "record %llu has an invalid or inconsistent state\n",
                (unsigned long long)index);
        return 0;
    }
    if (r->st.nply < h->ply_lo ||
        (h->ply_hi > 0 && r->st.nply >= h->ply_hi) ||
        (h->deck_max > 0 && r->st.deck_left > h->deck_max)) {
        fprintf(stderr, "record %llu falls outside its declared phase window\n",
                (unsigned long long)index);
        return 0;
    }
    Move legal[MAX_MOVES];
    int n = lc_moves(&r->st, legal);
    if (n <= 0) {
        fprintf(stderr, "record %llu has no legal moves\n",
                (unsigned long long)index);
        return 0;
    }
    if (r->kind == RD_KIND_CORRECTION) {
        Move b = unpack_move(r->baseline), c = unpack_move(r->challenger);
        int bi = -1, ci = -1;
        for (int i = 0; i < n; i++) {
            if (same_move(legal[i], b)) bi = i;
            if (same_move(legal[i], c)) ci = i;
        }
        double primary_lcb =
            (double)r->primary_delta -
            (double)h->override_k * r->primary_se;
        double confirm_lcb =
            (double)r->confirm_delta -
            (double)h->confirm_z * r->confirm_se;
        double expected_lcb =
            primary_lcb < confirm_lcb ? primary_lcb : confirm_lcb;
        double tolerance = 1e-4 * (1.0 + fabs(expected_lcb));
        if (bi < 0 || ci < 0 || bi == ci ||
            !lc_float_isfinite(r->primary_delta) ||
            !lc_float_isfinite(r->primary_se) ||
            !lc_float_isfinite(r->confirm_delta) ||
            !lc_float_isfinite(r->confirm_se) ||
            !lc_float_isfinite(r->lcb) || r->primary_se < 0.0f ||
            r->confirm_se < 0.0f || !(r->lcb > 0.0f) ||
            !(r->primary_delta > h->override_min) ||
            !(r->confirm_delta > 0.5f * h->override_min) ||
            !(expected_lcb > 0.0) ||
            fabs((double)r->lcb - expected_lcb) > tolerance) {
            fprintf(stderr, "record %llu has an invalid correction\n",
                    (unsigned long long)index);
            return 0;
        }
    }
    return 1;
}

static int inspect_records(const Config *c)
{
    RecordHeader h;
    DistillRecord *r = load_records(c->records_path, &h);
    if (!r) return 1;
    uint64_t anchors = 0, corrections = 0;
    double lcb_sum = 0.0, lcb_min = HUGE_VAL, lcb_max = -HUGE_VAL;
    for (uint64_t i = 0; i < h.count; i++) {
        if (!validate_record(&r[i], &h, i)) {
            free(r);
            return 1;
        }
        if (r[i].kind == RD_KIND_CORRECTION) {
            corrections++;
            lcb_sum += r[i].lcb;
            if (r[i].lcb < lcb_min) lcb_min = r[i].lcb;
            if (r[i].lcb > lcb_max) lcb_max = r[i].lcb;
        } else {
            anchors++;
        }
    }
    if (anchors != h.anchor_count || corrections != h.correction_count ||
        anchors + corrections != h.count) {
        fprintf(stderr, "%s: header counts do not match the payload\n",
                c->records_path);
        free(r);
        return 1;
    }
    printf("%s\n", c->records_path);
    printf("  records: %llu (%llu confirmed corrections, %llu KL anchors)\n",
           (unsigned long long)h.count,
           (unsigned long long)corrections,
           (unsigned long long)anchors);
    printf("  frozen net hash: %016llx\n",
           (unsigned long long)h.anchor_hash);
    printf("  generation: seed %llu, primary %u, confirmation %u, "
           "actor sym %u, playout sym %u, ply [%u,%s), deck max %u\n",
           (unsigned long long)h.seed, h.worlds, h.confirm_worlds,
           h.actor_symmetries, h.playout_symmetries, h.ply_lo,
           h.ply_hi ? "configured" : "off", h.deck_max);
    printf("  shortlist: width %u, mass %.4f; continuation mode %u "
           "(random-sym greedy), discard guard %u, playout prune %u\n",
           h.root_width, h.policy_mass, h.playout_sample,
           h.discard_guard, h.playout_prune);
    printf("  focused challengers: semantic %s, scheduler deck <= %u "
           "with block gap %u\n",
           h.semantic_cand ? "on" : "off",
           h.plan_deck_max, h.plan_block_gap);
    printf("  gates: primary %.2f SE and %.2f points; confirmation %.2f SE "
           "and %.2f points\n", h.override_k, h.override_min, h.confirm_z,
           0.5f * h.override_min);
    if (corrections)
        printf("  correction LCB: mean %.3f, min %.3f, max %.3f points\n",
               lcb_sum / (double)corrections, lcb_min, lcb_max);
    if (c->verbose) {
        for (uint64_t i = 0; i < h.count; i++) {
            if (r[i].kind != RD_KIND_CORRECTION) continue;
            Move baseline = unpack_move(r[i].baseline);
            Move challenger = unpack_move(r[i].challenger);
            char bname[64], cname[64];
            lc_move_name(&r[i].st, baseline, bname);
            lc_move_name(&r[i].st, challenger, cname);
            printf("  correction %llu: round %u ply %u deck %u: %s -> %s; "
                   "primary %+.2f +/- %.2f, confirm %+.2f +/- %.2f, "
                   "LCB %.2f\n",
                   (unsigned long long)i, r[i].st.round, r[i].st.nply,
                   r[i].st.deck_left, bname, cname,
                   r[i].primary_delta, r[i].primary_se,
                   r[i].confirm_delta, r[i].confirm_se, r[i].lcb);
        }
    }
    if (c->net_path) {
        Net *n = load_net(c->net_path);
        if (!n) {
            free(r);
            return 1;
        }
        uint64_t actual = hash_bytes(n, sizeof *n);
        printf("  supplied net hash: %016llx (%s)\n",
               (unsigned long long)actual,
               actual == h.anchor_hash ? "match" : "MISMATCH");
        free(n);
        if (actual != h.anchor_hash) {
            free(r);
            return 1;
        }
    }
    free(r);
    return 0;
}

static void sample_suit_permutation(uint64_t key, uint8_t perm[NSUIT])
{
    uint8_t left[NSUIT];
    for (int i = 0; i < NSUIT; i++) left[i] = (uint8_t)i;
    uint64_t code = key % UINT64_C(120);
    for (int i = 0; i < NSUIT; i++) {
        int nleft = NSUIT - i;
        int pick = (int)(code % (uint64_t)nleft);
        code /= (uint64_t)nleft;
        perm[i] = left[pick];
        for (int j = pick; j + 1 < nleft; j++)
            left[j] = left[j + 1];
    }
}

static void softmax(const float *logit, int n, float *prob)
{
    float mx = logit[0];
    for (int i = 1; i < n; i++)
        if (logit[i] > mx) mx = logit[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = exp((double)logit[i] - mx);
        prob[i] = (float)e;
        sum += e;
    }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) prob[i] *= inv;
}

static float logistic(float x)
{
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static void metrics_add(Metrics *dst, const Metrics *src)
{
    dst->kl += src->kl;
    dst->pair_loss += src->pair_loss;
    dst->pair_target += src->pair_target;
    dst->pair_probability += src->pair_probability;
    dst->lcb += src->lcb;
    dst->states += src->states;
    dst->corrections += src->corrections;
}

static int evaluate_record(const Config *c, const Net *anchor,
                           const Net *learner, const DistillRecord *input,
                           uint64_t augment_key, int augment, Net *grad,
                           Metrics *metric)
{
    DistillRecord transformed;
    const DistillRecord *r = input;
    if (augment) {
        uint8_t perm[NSUIT];
        sample_suit_permutation(augment_key, perm);
        transformed = *input;
        lc_permute_suits(&input->st, &transformed.st, perm);
        if (input->kind == RD_KIND_CORRECTION) {
            Move b = lc_permute_move(unpack_move(input->baseline), perm);
            Move ch = lc_permute_move(unpack_move(input->challenger), perm);
            transformed.baseline = MOVE_PACK(b);
            transformed.challenger = MOVE_PACK(ch);
        }
        r = &transformed;
    }

    Move move[MAX_MOVES];
    uint16_t packed[MAX_MOVES];
    float alogit[MAX_MOVES], llogit[MAX_MOVES];
    float aprob[MAX_MOVES], lprob[MAX_MOVES], dlog[MAX_MOVES];
    int n = lc_moves(&r->st, move);
    if (n <= 0) return 0;
    for (int i = 0; i < n; i++) packed[i] = MOVE_PACK(move[i]);

    Features feat;
    NetAct aa, la;
    feat_extract(&r->st, r->st.turn, &feat);
    net_trunk(anchor, &feat, &aa);
    net_trunk(learner, &feat, &la);
    net_policy_act(anchor, &aa, packed, n, alogit);
    net_policy_act(learner, &la, packed, n, llogit);
    softmax(alogit, n, aprob);
    softmax(llogit, n, lprob);

    Metrics m;
    memset(&m, 0, sizeof m);
    m.states = 1;
    for (int i = 0; i < n; i++) {
        double ap = aprob[i] > 1e-12f ? aprob[i] : 1e-12;
        double lp = lprob[i] > 1e-12f ? lprob[i] : 1e-12;
        m.kl += ap * log(ap / lp);
        dlog[i] = c->kl_weight * (lprob[i] - aprob[i]);
    }

    if (r->kind == RD_KIND_CORRECTION) {
        Move baseline = unpack_move(r->baseline);
        Move challenger = unpack_move(r->challenger);
        int bi = -1, ci = -1;
        for (int i = 0; i < n; i++) {
            if (same_move(move[i], baseline)) bi = i;
            if (same_move(move[i], challenger)) ci = i;
        }
        if (bi < 0 || ci < 0 || bi == ci) return 0;

        float scaled = r->lcb / c->lcb_scale;
        float target_gap =
            scaled < c->margin_cap ? scaled : c->margin_cap;
        if (target_gap < 0.0f) target_gap = 0.0f;
        float target = logistic(target_gap);
        float pair_prob = logistic(llogit[ci] - llogit[bi]);
        float example_weight =
            scaled < c->max_pair_weight ? scaled : c->max_pair_weight;
        if (example_weight < 0.0f) example_weight = 0.0f;
        example_weight *= c->pair_scale;
        float g = example_weight * (pair_prob - target);
        dlog[ci] += g;
        dlog[bi] -= g;

        double q = pair_prob;
        if (q < 1e-12) q = 1e-12;
        if (q > 1.0 - 1e-12) q = 1.0 - 1e-12;
        m.pair_loss =
            example_weight *
            (-(double)target * log(q) -
             (1.0 - (double)target) * log(1.0 - q));
        m.pair_target = target;
        m.pair_probability = pair_prob;
        m.lcb = r->lcb;
        m.corrections = 1;
    }

    if (grad)
        net_backward(learner, &feat, &la, 0.0f, packed, dlog, n,
                     NULL, NULL, 0, grad);
    if (metric) *metric = m;
    return 1;
}

static int residual_adam_init(ResidualAdam *a)
{
    memset(a, 0, sizeof *a);
    size_t nw = (size_t)NET_NCOMB * NET_H2;
    a->mw = (float *)calloc(nw, sizeof(float));
    a->vw = (float *)calloc(nw, sizeof(float));
    a->mb = (float *)calloc(NET_NCOMB, sizeof(float));
    a->vb = (float *)calloc(NET_NCOMB, sizeof(float));
    return a->mw && a->vw && a->mb && a->vb;
}

static int residual_wagers_are_tied(const Net *n)
{
    for (int suit = 0; suit < NSUIT; suit++) {
        int card = suit * NRANK;
        for (int discard = 0; discard < 2; discard++) {
            int p0 = card * 2 + discard;
            int p1 = (card + 1) * 2 + discard;
            int p2 = (card + 2) * 2 + discard;
            for (int draw = 0; draw < NET_NDRAW; draw++) {
                int c0 = p0 * NET_NDRAW + draw;
                int c1 = p1 * NET_NDRAW + draw;
                int c2 = p2 * NET_NDRAW + draw;
                if (n->bcomb[c0] != n->bcomb[c1] ||
                    n->bcomb[c0] != n->bcomb[c2])
                    return 0;
                for (int h = 0; h < NET_H2; h++)
                    if (n->wcomb[c0][h] != n->wcomb[c1][h] ||
                        n->wcomb[c0][h] != n->wcomb[c2][h])
                        return 0;
            }
        }
    }
    return 1;
}

static void residual_adam_free(ResidualAdam *a)
{
    free(a->mw);
    free(a->vw);
    free(a->mb);
    free(a->vb);
    memset(a, 0, sizeof *a);
}

static void residual_adam_step(Net *net, const Net *grad, ResidualAdam *a,
                               float lr, float scale, float wd)
{
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    a->t++;
    float bc1 = 1.0f - powf(beta1, (float)a->t);
    float bc2 = 1.0f - powf(beta2, (float)a->t);
    float step = lr * sqrtf(bc2) / bc1;

    size_t nw = (size_t)NET_NCOMB * NET_H2;
    float *w = &net->wcomb[0][0];
    const float *g = &grad->wcomb[0][0];
    for (size_t i = 0; i < nw; i++) {
        float gi = g[i] * scale + wd * w[i];
        a->mw[i] = beta1 * a->mw[i] + (1.0f - beta1) * gi;
        a->vw[i] = beta2 * a->vw[i] + (1.0f - beta2) * gi * gi;
        w[i] -= step * a->mw[i] / (sqrtf(a->vw[i]) + eps);
    }
    for (int i = 0; i < NET_NCOMB; i++) {
        float gi = grad->bcomb[i] * scale + wd * net->bcomb[i];
        a->mb[i] = beta1 * a->mb[i] + (1.0f - beta1) * gi;
        a->vb[i] = beta2 * a->vb[i] + (1.0f - beta2) * gi * gi;
        net->bcomb[i] -= step * a->mb[i] / (sqrtf(a->vb[i]) + eps);
    }
}

static Metrics evaluate_dataset(const Config *c, const Net *anchor,
                                const Net *learner,
                                const DistillRecord *record, uint64_t n)
{
    Metrics total;
    memset(&total, 0, sizeof total);
    for (uint64_t i = 0; i < n; i++) {
        Metrics m;
        if (evaluate_record(c, anchor, learner, &record[i], 0, 0, NULL, &m))
            metrics_add(&total, &m);
    }
    return total;
}

static void print_metrics(int epoch, const Metrics *m)
{
    double kl = m->states ? m->kl / (double)m->states : 0.0;
    double pair = m->corrections
                    ? m->pair_loss / (double)m->corrections : 0.0;
    double target = m->corrections
                    ? m->pair_target / (double)m->corrections : 0.0;
    double prob = m->corrections
                    ? m->pair_probability / (double)m->corrections : 0.0;
    double lcb = m->corrections
                    ? m->lcb / (double)m->corrections : 0.0;
    printf("epoch %3d: KL %.6f, pair CE %.5f, challenger %.1f%% -> "
           "target %.1f%%, mean LCB %.2f (%llu corrections)\n",
           epoch, kl, pair, 100.0 * prob, 100.0 * target, lcb,
           (unsigned long long)m->corrections);
}

static int train_records(const Config *c)
{
    RecordHeader h;
    DistillRecord *record = load_records(c->records_path, &h);
    if (!record) return 1;
    uint64_t anchors = 0, corrections = 0;
    for (uint64_t i = 0; i < h.count; i++) {
        if (!validate_record(&record[i], &h, i)) {
            free(record);
            return 1;
        }
        if (record[i].kind == RD_KIND_CORRECTION) corrections++;
        else anchors++;
    }
    if (anchors != h.anchor_count || corrections != h.correction_count ||
        anchors + corrections != h.count) {
        fprintf(stderr, "%s: header counts do not match the payload\n",
                c->records_path);
        free(record);
        return 1;
    }
    for (int set = 0; set < c->extra_record_count; set++) {
        RecordHeader eh;
        DistillRecord *extra =
            load_records(c->extra_records[set], &eh);
        if (!extra) {
            free(record);
            return 1;
        }
        uint64_t ea = 0, ec = 0;
        for (uint64_t i = 0; i < eh.count; i++) {
            if (!validate_record(&extra[i], &eh, i)) {
                free(extra);
                free(record);
                return 1;
            }
            if (extra[i].kind == RD_KIND_CORRECTION) ec++;
            else ea++;
        }
        if (ea != eh.anchor_count || ec != eh.correction_count ||
            ea + ec != eh.count) {
            fprintf(stderr, "%s: header counts do not match the payload\n",
                    c->extra_records[set]);
            free(extra);
            free(record);
            return 1;
        }
        if (eh.anchor_hash != h.anchor_hash) {
            fprintf(stderr, "%s is tied to a different frozen network\n",
                    c->extra_records[set]);
            free(extra);
            free(record);
            return 1;
        }
        if (eh.count > UINT64_MAX - h.count ||
            h.count + eh.count > SIZE_MAX / sizeof *record) {
            fprintf(stderr, "combined record sets are too large\n");
            free(extra);
            free(record);
            return 1;
        }
        uint64_t old_count = h.count;
        DistillRecord *combined =
            (DistillRecord *)realloc(
                record, (size_t)(h.count + eh.count) * sizeof *record);
        if (!combined) {
            fprintf(stderr, "out of memory combining record sets\n");
            free(extra);
            free(record);
            return 1;
        }
        record = combined;
        memcpy(record + old_count, extra,
               (size_t)eh.count * sizeof *extra);
        free(extra);
        h.count += eh.count;
        anchors += ea;
        corrections += ec;
        h.anchor_count = anchors;
        h.correction_count = corrections;
    }
    if (corrections == 0) {
        fprintf(stderr, "%s contains no confirmed corrections; refusing a "
                        "no-signal training run\n", c->records_path);
        free(record);
        return 1;
    }

    Net *anchor = load_net(c->net_path);
    Net *learner = (Net *)malloc(sizeof *learner);
    Net *epoch_start = (Net *)malloc(sizeof *epoch_start);
    Net *grad = (Net *)malloc(sizeof *grad);
    if (!anchor || !learner || !epoch_start || !grad) {
        fprintf(stderr, "out of memory allocating training networks\n");
        free(record);
        free(anchor);
        free(learner);
        free(epoch_start);
        free(grad);
        return 1;
    }
    uint64_t actual_hash = hash_bytes(anchor, sizeof *anchor);
    if (actual_hash != h.anchor_hash) {
        fprintf(stderr, "record net hash %016llx does not match %s "
                        "(%016llx)\n",
                (unsigned long long)h.anchor_hash, c->net_path,
                (unsigned long long)actual_hash);
        goto train_error;
    }
    if (!residual_is_zero(anchor)) {
        fprintf(stderr, "%s has a nonzero full-move residual; refusing to "
                        "reuse this first-stage trainer\n", c->net_path);
        goto train_error;
    }
    memcpy(learner, anchor, sizeof *learner);

    size_t *order = (size_t *)malloc((size_t)h.count * sizeof *order);
    if (!order) {
        fprintf(stderr, "out of memory allocating training order\n");
        goto train_error;
    }
    ResidualAdam adam;
    if (!residual_adam_init(&adam)) {
        fprintf(stderr, "out of memory allocating residual Adam state\n");
        free(order);
        goto train_error;
    }

    printf("robust residual distillation: %llu records, %llu corrections, "
           "%llu anchors\n", (unsigned long long)h.count,
           (unsigned long long)h.correction_count,
           (unsigned long long)h.anchor_count);
    printf("updates are restricted to %d x %d wcomb weights + %d biases; "
           "suit augmentation %s\n", NET_NCOMB, NET_H2, NET_NCOMB,
           !c->suit_augment ? "off" :
           (c->random_suit_augment ? "random" : "exhaustive"));
    Metrics initial =
        evaluate_dataset(c, anchor, learner, record, h.count);
    print_metrics(0, &initial);

    int accepted_epochs = 0;
    for (int epoch = 1; epoch <= c->epochs; epoch++) {
        memcpy(epoch_start, learner, sizeof *learner);
        for (uint64_t i = 0; i < h.count; i++) order[i] = (size_t)i;
        Rng shuffle;
        rng_seed(&shuffle, mix64(c->seed ^ (uint64_t)epoch));
        for (uint64_t i = h.count; i > 1; i--) {
            uint32_t j = rng_below(&shuffle, (uint32_t)i);
            size_t t = order[i - 1];
            order[i - 1] = order[j];
            order[j] = t;
        }

        for (uint64_t off = 0; off < h.count; off += (uint64_t)c->batch) {
            uint64_t remaining = h.count - off;
            int nb = remaining < (uint64_t)c->batch
                       ? (int)remaining : c->batch;
            net_zero(grad);
            int used = 0;
            for (int k = 0; k < nb; k++) {
                size_t ri = order[off + (uint64_t)k];
                /* The default random branch preserves the published trainer
                 * and its reproducible seeds.  Exhaustive mode gives every
                 * record a no-replacement tour of all 120 permutations over
                 * 120 epochs; it is a useful variance-control ablation, but
                 * still has to pass independent match play. */
                uint64_t aug_key;
                if (c->random_suit_augment) {
                    aug_key =
                        mix64(c->seed ^
                              ((uint64_t)(unsigned)epoch << 48) ^
                              ((uint64_t)ri *
                               UINT64_C(0xD1B54A32D192ED03))) %
                        UINT64_C(120);
                } else {
                    uint64_t offset =
                        mix64(c->seed ^
                              ((uint64_t)ri *
                               UINT64_C(0xD1B54A32D192ED03))) %
                        UINT64_C(120);
                    aug_key =
                        offset + (uint64_t)(unsigned)(epoch - 1);
                }
                if (evaluate_record(c, anchor, learner, &record[ri],
                                    aug_key, c->suit_augment, grad, NULL))
                    used++;
            }
            if (used > 0) {
                net_tie_wager_gradients(grad);
                residual_adam_step(learner, grad, &adam, c->lr,
                                   1.0f / (float)used, c->weight_decay);
            }
        }

        Metrics current =
            evaluate_dataset(c, anchor, learner, record, h.count);
        double mean_kl =
            current.states ? current.kl / (double)current.states : 0.0;
        if (!lc_double_isfinite(mean_kl) || mean_kl > c->max_kl) {
            memcpy(learner, epoch_start, sizeof *learner);
            printf("epoch %d rejected: mean KL %.6f exceeds trust limit %.6f; "
                   "restored epoch %d\n", epoch, mean_kl, c->max_kl,
                   accepted_epochs);
            break;
        }
        accepted_epochs = epoch;
        print_metrics(epoch, &current);
    }

    if (memcmp(anchor, learner, offsetof(Net, wcomb)) != 0) {
        fprintf(stderr, "internal error: a frozen pre-residual parameter changed\n");
        residual_adam_free(&adam);
        free(order);
        goto train_error;
    }
    if (!residual_wagers_are_tied(learner)) {
        fprintf(stderr, "internal error: semantic wager residuals diverged\n");
        residual_adam_free(&adam);
        free(order);
        goto train_error;
    }
    printf("verified: every pre-wcomb byte is frozen and semantic wager "
           "residuals remain tied\n");
    if (accepted_epochs == 0) {
        fprintf(stderr, "no training epoch passed the KL trust region; "
                        "no candidate written\n");
        residual_adam_free(&adam);
        free(order);
        goto train_error;
    }

    if (c->dry_run) {
        printf("dry run complete; no model written\n");
    } else {
        if (!save_net_atomic(learner, c->out_path, c->force)) {
            residual_adam_free(&adam);
            free(order);
            goto train_error;
        }
        printf("wrote unpromoted residual candidate to %s\n", c->out_path);
    }

    residual_adam_free(&adam);
    free(order);
    free(record);
    free(anchor);
    free(learner);
    free(epoch_start);
    free(grad);
    return 0;

train_error:
    free(record);
    free(anchor);
    free(learner);
    free(epoch_start);
    free(grad);
    return 1;
}

int main(int argc, char **argv)
{
    Config c;
    if (!parse_args(argc, argv, &c)) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (c.mode == MODE_GENERATE &&
        paths_alias(c.records_path, c.net_path)) {
        fprintf(stderr, "record output must not alias the frozen --net\n");
        return 2;
    }
    if (c.mode == MODE_TRAIN && c.out_path &&
        (paths_alias(c.out_path, c.net_path) ||
         paths_alias(c.out_path, c.records_path))) {
        fprintf(stderr, "candidate --out must not alias --net or the "
                        "training records\n");
        return 2;
    }
    if (c.mode == MODE_TRAIN) {
        for (int i = 0; i < c.extra_record_count; i++) {
            if ((c.out_path &&
                 paths_alias(c.out_path, c.extra_records[i])) ||
                paths_alias(c.records_path, c.extra_records[i])) {
                fprintf(stderr, "extra record inputs must be distinct from "
                                "the primary records and candidate output\n");
                return 2;
            }
            for (int j = 0; j < i; j++)
                if (paths_alias(c.extra_records[i], c.extra_records[j])) {
                    fprintf(stderr, "duplicate extra record input\n");
                    return 2;
                }
        }
    }
    switch (c.mode) {
    case MODE_GENERATE: return generate_records(&c);
    case MODE_TRAIN: return train_records(&c);
    case MODE_INSPECT: return inspect_records(&c);
    case MODE_NONE: break;
    }
    return 2;
}
