/* arena -- play two agents against each other over paired (mirrored) deals.
 *
 * Every deal is played twice, once with each agent seated first, so the
 * comparison is not polluted by deal luck.  Reported figures are the mean
 * score margin per game and the win rate, with standard errors over pairs.
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/match.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

static int parse_u64(const char *s, uint64_t *out)
{
    if (!s || !*s || !out) return 0;
    uint64_t value = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p < '0' || *p > '9') return 0;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / UINT64_C(10)) return 0;
        value = value * UINT64_C(10) + digit;
    }
    *out = value;
    return 1;
}

static int parse_int_range(const char *s, int low, int high, int *out)
{
    uint64_t value;
    if (low < 0 || high < low || !parse_u64(s, &value) ||
        value < (uint64_t)low || value > (uint64_t)high)
        return 0;
    *out = (int)value;
    return 1;
}

static int raw_rows_complete(const MatchPairResult *row, int pairs,
                             uint64_t pair_start, int rounds,
                             const MatchResult *result)
{
    if (!row || !result || pairs < 1 || pairs > MATCH_MAX_PAIRS ||
        rounds < 1 || rounds > MATCH_ROUNDS || result->pairs != pairs ||
        result->games != 2 * pairs)
        return 0;
    uint64_t caps = 0, wins = 0, losses = 0, draws = 0;
    for (int i = 0; i < pairs; i++) {
        uint64_t expected = pair_start + (uint64_t)i;
        if (row[i].index != expected) return 0;
        for (int leg = 0; leg < 2; leg++) {
            if (row[i].plies[leg] < rounds ||
                row[i].plies[leg] > rounds * LC_MAX_PLIES ||
                row[i].capped_rounds[leg] < 0 ||
                row[i].capped_rounds[leg] > rounds)
                return 0;
            caps += (uint64_t)row[i].capped_rounds[leg];
            if (row[i].score_a[leg] > row[i].score_b[leg]) wins++;
            else if (row[i].score_a[leg] < row[i].score_b[leg]) losses++;
            else draws++;
        }
    }
    return caps == result->capped_rounds &&
        (double)wins == result->wins &&
        (double)losses == result->losses &&
        (double)draws == result->draws;
}

static void json_string(FILE *f, const char *s)
{
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"': fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\b': fputs("\\b", f); break;
        case '\f': fputs("\\f", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (*p < 0x20) fprintf(f, "\\u%04x", *p);
            else fputc(*p, f);
        }
    }
    fputc('"', f);
}

static int write_raw_pairs(const char *path, const MatchPairResult *row,
                           int pairs, uint64_t pair_start, uint64_t seed,
                           int rounds, const char *spec0, const char *spec1,
                           const char *provenance)
{
    if (!path || !*path || !row || pairs < 1 || pairs > MATCH_MAX_PAIRS ||
        rounds < 1 ||
        rounds > MATCH_ROUNDS || !spec0 || !spec1 || !provenance ||
        !*provenance ||
        pair_start > UINT64_MAX - ((uint64_t)pairs - UINT64_C(1))) {
        fprintf(stderr, "refusing to write incomplete raw results\n");
        return -1;
    }
    size_t npath = strlen(path);
    if (npath > SIZE_MAX - 64) return -1;
    char *tmp = malloc(npath + 64);
    if (!tmp) return -1;
    snprintf(tmp, npath + 64, "%s.tmp.%ld", path, (long)getpid());
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0) {
        fprintf(stderr, "%s: cannot create temporary raw file: %s\n",
                tmp, strerror(errno));
        free(tmp);
        return -1;
    }
    FILE *f = fdopen(fd, "w");
    if (!f) {
        fprintf(stderr, "%s: fdopen failed: %s\n", tmp, strerror(errno));
        close(fd);
        unlink(tmp);
        free(tmp);
        return -1;
    }
    fputs("{\"record\":\"meta\",\"schema\":1,\"seed\":", f);
    fprintf(f, "\"%llu\",\"pair_start\":\"%llu\",\"pair_count\":%d,"
               "\"rounds\":%d,\"agent_a\":",
            (unsigned long long)seed, (unsigned long long)pair_start,
            pairs, rounds);
    json_string(f, spec0);
    fputs(",\"agent_b\":", f);
    json_string(f, spec1);
    fputs(",\"provenance\":", f);
    json_string(f, provenance ? provenance : "");
    fputs("}\n", f);
    for (int i = 0; i < pairs; i++) {
        fprintf(f,
            "{\"record\":\"pair\",\"index\":\"%llu\","
            "\"score_a\":[%d,%d],\"score_b\":[%d,%d],"
            "\"plies\":[%d,%d],\"capped_rounds\":[%d,%d]}\n",
            (unsigned long long)row[i].index,
            row[i].score_a[0], row[i].score_a[1],
            row[i].score_b[0], row[i].score_b[1],
            row[i].plies[0], row[i].plies[1],
            row[i].capped_rounds[0], row[i].capped_rounds[1]);
    }
    fprintf(f, "{\"record\":\"complete\",\"pairs\":%d}\n", pairs);
    int failed = ferror(f) || fflush(f) != 0 || fsync(fd) != 0;
    if (fclose(f) != 0) failed = 1;
    if (failed) {
        fprintf(stderr, "%s: failed while writing raw results\n", tmp);
        unlink(tmp);
        free(tmp);
        return -1;
    }
    /* link() is the portable no-clobber publish step: an accidental rerun
     * cannot silently replace completed evidence. */
    if (link(tmp, path) != 0) {
        fprintf(stderr, "%s: refusing to replace raw result: %s\n",
                path, strerror(errno));
        unlink(tmp);
        free(tmp);
        return -1;
    }
    unlink(tmp);
    free(tmp);
    return 0;
}

int main(int argc, char **argv)
{
    const char *spec0 = "heur", *spec1 = "random";
    int pairs = 500, nthread = 4, rounds = 1;
    uint64_t seed = 20260727, pair_start = 0;
    int quiet = 0, raw_only = 0;
    const char *raw_path = NULL, *provenance = "";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-a") && i + 1 < argc) spec0 = argv[++i];
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) spec1 = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            if (!parse_int_range(argv[++i], 1, MATCH_MAX_PAIRS, &pairs)) {
                fprintf(stderr, "invalid pair count\n");
                return 1;
            }
        }
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            if (!parse_int_range(argv[++i], 1, 1024, &nthread)) {
                fprintf(stderr, "invalid thread count\n");
                return 1;
            }
        }
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &seed)) {
                fprintf(stderr, "invalid seed\n");
                return 1;
            }
        }
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            if (!parse_int_range(argv[++i], 1, MATCH_ROUNDS, &rounds)) {
                fprintf(stderr, "invalid round count\n");
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--pair-start") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &pair_start)) {
                fprintf(stderr, "invalid pair start\n");
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--raw-pairs") && i + 1 < argc)
            raw_path = argv[++i];
        else if (!strcmp(argv[i], "--raw-only")) raw_only = 1;
        else if (!strcmp(argv[i], "--provenance") && i + 1 < argc)
            provenance = argv[++i];
        else if (!strcmp(argv[i], "-q")) quiet = 1;
        else {
            fprintf(stderr, "usage: %s -a SPEC -b SPEC [-n pairs] [-t threads] [-s seed] [-r rounds] [-q]\n"
                            "          [--pair-start N] [--raw-pairs FILE [--raw-only] [--provenance ID]]\n"
                            "  SPEC = random | heur | policy:... | rolloutu:... | rolloutu2:ROOT:CONT:... | net:... | mcts:...\n"
                            "  See src/spec.h for the complete positional tails.\n",
                    argv[0]);
            return 1;
        }
    }
    if (pairs < 1 || pairs > MATCH_MAX_PAIRS || nthread < 1 ||
        rounds < 1 || rounds > MATCH_ROUNDS ||
        pair_start > UINT64_MAX - ((uint64_t)pairs - UINT64_C(1))) {
        fprintf(stderr, "invalid pair/thread/round range\n");
        return 1;
    }
    if (raw_only && !raw_path) {
        fprintf(stderr, "--raw-only requires --raw-pairs\n");
        return 1;
    }
    if (raw_path && (!*raw_path || !provenance || !*provenance)) {
        fprintf(stderr, "--raw-pairs requires a nonempty path and provenance\n");
        return 1;
    }

    Agent a, b;
    spec_parse(spec0, &a);
    spec_parse(spec1, &b);
    MatchResult r;
    MatchPairResult *rows = raw_path
        ? calloc((size_t)pairs, sizeof(*rows)) : NULL;
    if (raw_path && !rows) {
        fprintf(stderr, "cannot allocate raw pair results\n");
        return 1;
    }
    struct timespec t0, t1;
    if (clock_gettime(CLOCK_MONOTONIC, &t0) != 0) {
        fprintf(stderr, "cannot start arena timer: %s\n", strerror(errno));
        free(rows);
        return 1;
    }
    if (match_run_range_r(&a, &b, pair_start, pairs, nthread, seed,
                          rounds, rows, &r) != 0) {
        fprintf(stderr, "match execution failed; no results were published\n");
        free(rows);
        return 1;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0) {
        fprintf(stderr, "cannot stop arena timer; no results were published: "
                        "%s\n", strerror(errno));
        free(rows);
        return 1;
    }
    double secs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

    if (raw_path) {
        if (!raw_rows_complete(rows, pairs, pair_start, rounds, &r)) {
            fprintf(stderr, "match returned incomplete raw rows; no results "
                            "were published\n");
            free(rows);
            return 1;
        }
        if (write_raw_pairs(raw_path, rows, pairs, pair_start, seed,
                            rounds, spec0, spec1, provenance) != 0) {
            free(rows);
            return 1;
        }
    }
    free(rows);

    if (raw_only) return 0;

    if (quiet) {
        printf("%.3f %.3f %.4f %.4f\n", r.margin, r.margin_se, r.winrate, r.winrate_se);
    } else {
        printf("%s  vs  %s", a.name ? a.name : spec0, b.name ? b.name : spec1);
        if (rounds > 1) printf("  (%d-round matches)", rounds);
        putchar('\n');
        printf("  %d games (%d paired deals) in %.1fs (%.1f games/s)\n",
               r.games, r.pairs, secs, r.games / secs);
        printf("  margin/game %+.2f  (pair-clustered SE %.2f)\n",
               r.margin, r.margin_se);
        printf("  W/L/D %.0f/%.0f/%.0f   score %.1f%%  (pair-clustered SE %.1f%%)\n",
               r.wins, r.losses, r.draws, 100 * r.winrate, 100 * r.winrate_se);
        printf("  points/game %.1f vs %.1f   plies/game %.1f\n", r.points_a, r.points_b, r.plies);
        printf("  cap-terminated rounds %llu\n",
               (unsigned long long)r.capped_rounds);
    }
    return 0;
}
