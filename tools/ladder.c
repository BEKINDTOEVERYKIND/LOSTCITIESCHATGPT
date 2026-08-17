/* ladder -- round robin between agents, with Elo fitted from the results.
 *
 * Elo here is derived from win rates over paired deals, so it measures how
 * often an agent finishes ahead, not by how much.  The mean margin table below
 * it carries the size of the edge.
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/match.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXA 12

int main(int argc, char **argv)
{
    const char *specs[MAXA];
    int na = 0, pairs = 200, nthread = 4, rounds = 1;
    uint64_t seed = 12345;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) pairs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) nthread = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (na < MAXA) specs[na++] = argv[i];
    }
    if (na < 2) { fprintf(stderr, "usage: %s [-n pairs] [-t threads] SPEC SPEC [SPEC ...]\n", argv[0]); return 1; }

    Agent ag[MAXA];
    for (int i = 0; i < na; i++) spec_parse(specs[i], &ag[i]);

    static double margin[MAXA][MAXA], score[MAXA][MAXA], games[MAXA][MAXA];
    for (int i = 0; i < na; i++)
        for (int j = i + 1; j < na; j++) {
            MatchResult r;
            if (match_run_r(&ag[i], &ag[j], pairs, nthread,
                            seed + (uint64_t)(i * 31 + j), rounds, &r) != 0) {
                fprintf(stderr, "match execution failed for %s vs %s\n",
                        specs[i], specs[j]);
                return 1;
            }
            margin[i][j] = r.margin;   margin[j][i] = -r.margin;
            score[i][j] = r.winrate * r.games; score[j][i] = (1.0 - r.winrate) * r.games;
            games[i][j] = games[j][i] = r.games;
            printf("%-28s vs %-28s  margin %+7.2f +- %.2f   score %5.1f%%\n",
                   specs[i], specs[j], r.margin, r.margin_se, 100 * r.winrate);
            fflush(stdout);
        }

    /* Bradley-Terry / minorization-maximization fit of Elo ratings */
    double elo[MAXA];
    for (int i = 0; i < na; i++) elo[i] = 0.0;
    for (int iter = 0; iter < 4000; iter++) {
        for (int i = 0; i < na; i++) {
            double num = 0, den = 0;
            for (int j = 0; j < na; j++) {
                if (i == j || games[i][j] == 0) continue;
                double gi = pow(10.0, elo[i] / 400.0), gj = pow(10.0, elo[j] / 400.0);
                num += score[i][j];
                den += games[i][j] / (gi + gj);
            }
            if (den > 0 && num > 0) elo[i] = 400.0 * log10(num / den);
        }
        double mean = 0;
        for (int i = 0; i < na; i++) mean += elo[i];
        mean /= na;
        for (int i = 0; i < na; i++) elo[i] -= mean;
    }
    double base = elo[0];
    printf("\n%-34s %8s %10s\n", "agent", "elo", "avg margin");
    int order[MAXA];
    for (int i = 0; i < na; i++) order[i] = i;
    for (int i = 0; i < na; i++)
        for (int j = i + 1; j < na; j++)
            if (elo[order[j]] > elo[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    for (int k = 0; k < na; k++) {
        int i = order[k];
        double m = 0; int c = 0;
        for (int j = 0; j < na; j++) if (i != j && games[i][j] > 0) { m += margin[i][j]; c++; }
        printf("%-34s %+8.0f %+10.2f\n", specs[i], elo[i] - base, c ? m / c : 0.0);
    }
    return 0;
}
