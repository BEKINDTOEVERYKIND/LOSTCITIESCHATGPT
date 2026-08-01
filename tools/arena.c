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

int main(int argc, char **argv)
{
    const char *spec0 = "heur", *spec1 = "random";
    int pairs = 500, nthread = 4, rounds = 1;
    uint64_t seed = 20260727;
    int quiet = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-a") && i + 1 < argc) spec0 = argv[++i];
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) spec1 = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) pairs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) nthread = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q")) quiet = 1;
        else {
            fprintf(stderr, "usage: %s -a SPEC -b SPEC [-n pairs] [-t threads] [-s seed] [-r rounds] [-q]\n"
                            "  SPEC = random | heur | policy:... | rolloutu:... | net:... | mcts:...\n"
                            "  See src/spec.h for the complete positional tails.\n",
                    argv[0]);
            return 1;
        }
    }

    Agent a, b;
    spec_parse(spec0, &a);
    spec_parse(spec1, &b);
    MatchResult r;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    match_run_r(&a, &b, pairs, nthread, seed, rounds, &r);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

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
    }
    return 0;
}
