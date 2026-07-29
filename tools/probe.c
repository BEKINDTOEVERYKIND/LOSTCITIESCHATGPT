/* probe -- diagnostics for a trained value network.
 *
 * Collects states from real games and reports how well the network's values
 * behave: calibration against realized outcomes, zero-sum consistency between
 * the two points of view, and how its move ranking compares with the
 * hand-crafted one.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/heuristic.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    State st;
    uint8_t persp;
    float target;
} Rec;

int main(int argc, char **argv)
{
    const char *net_path = "data/champion.bin";
    const char *play_spec = "heur";
    int games = 200;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) net_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) play_spec = argv[++i];
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) games = atoi(argv[++i]);
    }
    Net *net = (Net *)malloc(sizeof(Net));
    if (net_load(net, net_path) != 0) { fprintf(stderr, "cannot load %s\n", net_path); return 1; }

    Agent player;
    spec_parse(play_spec, &player);

    Rng rng; rng_seed(&rng, 4242);
    size_t cap = (size_t)games * 400;
    Rec *recs = (Rec *)malloc(sizeof(Rec) * cap);
    size_t nrec = 0;
    double plies = 0;
    int capped = 0;

    for (int g = 0; g < games; g++) {
        State st; lc_deal(&st, &rng);
        size_t start = nrec;
        while (!st.over) {
            if (nrec + 2 < cap) {
                for (int q = 0; q < 2; q++) {
                    recs[nrec].st = st;
                    recs[nrec].persp = (uint8_t)((st.turn + q) & 1);
                    nrec++;
                }
            }
            lc_apply(&st, agent_move(&player, &st, &rng));
        }
        int s0 = lc_score(&st, 0), s1 = lc_score(&st, 1);
        for (size_t i = start; i < nrec; i++)
            recs[i].target = (float)(recs[i].persp == 0 ? s0 - s1 : s1 - s0);
        plies += st.nply;
        if (st.nply >= LC_MAX_PLIES) capped++;
    }
    printf("collected %zu states from %d games played by %s (%.1f plies/game, %d hit the ply cap)\n",
           nrec, games, play_spec, plies / games, capped);

    /* --- calibration and error --- */
    double se = 0, sy = 0, syy = 0, sp = 0, spp = 0, spy = 0;
    Features f;
    for (size_t i = 0; i < nrec; i++) {
        feat_extract(&recs[i].st, recs[i].persp, &f);
        double v = net_value(net, &f) * VAL_SCALE;
        double y = recs[i].target;
        se += (v - y) * (v - y);
        sy += y; syy += y * y;
        sp += v; spp += v * v; spy += v * y;
    }
    double n = (double)nrec;
    double vary = syy / n - (sy / n) * (sy / n);
    double varp = spp / n - (sp / n) * (sp / n);
    double cov = spy / n - (sp / n) * (sy / n);
    printf("value: rmse %.1f pts, target sd %.1f, prediction sd %.1f, corr %.3f, R2 %.3f\n",
           sqrt(se / n), sqrt(vary), sqrt(varp), cov / sqrt(varp * vary), 1.0 - (se / n) / vary);

    /* --- zero-sum consistency: V(s,0) + V(s,1) should be near zero --- */
    double zs = 0, zsa = 0;
    int nz = 0;
    for (size_t i = 0; i + 1 < nrec; i += 2) {
        feat_extract(&recs[i].st, 0, &f);
        double v0 = net_value(net, &f) * VAL_SCALE;
        feat_extract(&recs[i].st, 1, &f);
        double v1 = net_value(net, &f) * VAL_SCALE;
        zs += v0 + v1; zsa += fabs(v0 + v1); nz++;
    }
    printf("zero-sum: mean V(s,0)+V(s,1) = %+.2f, mean |sum| = %.2f\n", zs / nz, zsa / nz);

    /* --- move ranking versus the heuristic --- */
    int agree = 0, tot = 0;
    double sxx = 0, sxy2 = 0, syy2 = 0;
    long deck_pref_net = 0, deck_pref_heur = 0, npick = 0;
    for (size_t i = 0; i < nrec && tot < 4000; i += 7) {
        const State *st = &recs[i].st;
        if (st->over) continue;
        Move mv[MAX_MOVES];
        float vh[MAX_MOVES], vn[MAX_MOVES];
        int nm = lc_moves(st, mv);
        if (nm < 2) continue;
        DrawSamples ds;
        draw_samples_init(st, st->turn, &rng, 12, &ds);
        int bh = 0, bn = 0;
        for (int k = 0; k < nm; k++) {
            vh[k] = move_value_heur(st, mv[k], &ds);
            vn[k] = move_value_net(net, st, mv[k], &ds);
            if (vh[k] > vh[bh]) bh = k;
            if (vn[k] > vn[bn]) bn = k;
        }
        if (bh == bn) agree++;
        tot++;
        /* correlation of the two rankings on this state */
        double mh = 0, mn = 0;
        for (int k = 0; k < nm; k++) { mh += vh[k]; mn += vn[k]; }
        mh /= nm; mn /= nm;
        for (int k = 0; k < nm; k++) {
            double a = vh[k] - mh, b = vn[k] - mn;
            sxx += a * a; syy2 += b * b; sxy2 += a * b;
        }
        if (mv[bn].draw == 0) deck_pref_net++;
        if (mv[bh].draw == 0) deck_pref_heur++;
        npick++;
    }
    printf("move ranking: top-1 agreement with heuristic %.1f%% over %d states, value corr %.3f\n",
           100.0 * agree / tot, tot, sxy2 / sqrt(sxx * syy2));
    printf("draws from deck: network picks %.1f%%, heuristic picks %.1f%%\n",
           100.0 * deck_pref_net / npick, 100.0 * deck_pref_heur / npick);

    /* --- calibration buckets --- */
    printf("calibration (predicted -> realized):\n");
    for (int b = -3; b <= 3; b++) {
        double lo = b * 25 - 12.5, hi = b * 25 + 12.5;
        double s = 0; int c = 0;
        for (size_t i = 0; i < nrec; i++) {
            feat_extract(&recs[i].st, recs[i].persp, &f);
            double v = net_value(net, &f) * VAL_SCALE;
            if (v >= lo && v < hi) { s += recs[i].target; c++; }
        }
        if (c > 20) printf("   %+4d..%+4d : n=%6d realized %+7.1f\n", (int)lo, (int)hi, c, s / c);
    }
    return 0;
}
