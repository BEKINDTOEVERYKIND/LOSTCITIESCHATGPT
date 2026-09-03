/* dumpfeat -- reference dumps for verifying an external (python) port.
 *
 * Default mode: deal a game from -s SEED, play 30 uniformly random legal
 * moves; for the first 20 states print the mover's full feature vector, the
 * (scaled) value head output and the policy probabilities over the legal
 * moves, plus the chosen move at every step so the state sequence can be
 * replayed exactly from the printed deal order.
 *
 * -g mode: play a full argmax-policy self-play game from the seed, printing
 * the deal order, every move, and the final score.
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *netpath = NULL;
    uint64_t seed = 1;
    int game = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) netpath = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-g")) game = 1;
        else { fprintf(stderr, "usage: %s -n NET [-s seed] [-g]\n", argv[0]); return 1; }
    }
    if (!netpath) { fprintf(stderr, "need -n NET\n"); return 1; }

    Net *net = (Net *)malloc(sizeof(Net));
    if (!net || net_load(net, netpath) != 0) {
        fprintf(stderr, "cannot load net %s (wrong dims for this build?)\n", netpath);
        return 1;
    }

    Rng rng; rng_seed(&rng, seed);
    State st;
    lc_deal(&st, &rng);

    printf("DIMS %d %d %d %d\n", FEAT_DIM, NET_H1, NET_H2, NET_NPLAY);
    printf("DEAL");
    for (int i = 0; i < NCARD; i++) printf(" %d", st.deck[i]);
    printf("\n");
    printf("HANDS %016llx %016llx\n",
           (unsigned long long)st.hand[0], (unsigned long long)st.hand[1]);

    if (game) {
        while (!st.over) {
            Move mv[MAX_MOVES];
            float prob[MAX_MOVES];
            int n = policy_probs(net, &st, mv, prob, NULL);
            int best = 0;
            for (int i = 1; i < n; i++) if (prob[i] > prob[best]) best = i;
            printf("MOVE %d %d %d\n", mv[best].card, mv[best].discard, mv[best].draw);
            lc_apply(&st, mv[best]);
        }
        printf("SCORE %d %d\n", lc_score(&st, 0), lc_score(&st, 1));
        printf("NPLY %d\n", st.nply);
        free(net);
        return 0;
    }

    for (int step = 0; step < 30 && !st.over; step++) {
        Move mv[MAX_MOVES];
        int n;
        if (step < 20) {
            printf("STATE %d PERSP %d\n", step, st.turn);
            Features f;
            feat_extract(&st, st.turn, &f);
            float full[FEAT_DIM];
            memset(full, 0, sizeof(full));
            for (int k = 0; k < f.nidx; k++) full[f.idx[k]] = 1.0f;
            memcpy(full + FEAT_BIN, f.dense, sizeof(f.dense));
            printf("FEAT");
            for (int i = 0; i < FEAT_DIM; i++) printf(" %.6f", full[i]);
            printf("\n");
            float prob[MAX_MOVES], value;
            n = policy_probs(net, &st, mv, prob, &value);
            printf("VALUE %.6f\n", value);
            printf("NMOVES %d\n", n);
            for (int i = 0; i < n; i++)
                printf("MV %d %d %d %.6f\n", mv[i].card, mv[i].discard, mv[i].draw, prob[i]);
        } else {
            n = lc_moves(&st, mv);
        }
        int pick = (int)rng_below(&rng, (uint32_t)n);
        printf("CHOSEN %d %d %d\n", mv[pick].card, mv[pick].discard, mv[pick].draw);
        lc_apply(&st, mv[pick]);
    }
    free(net);
    return 0;
}
