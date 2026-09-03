/* bench -- throughput of the pieces that matter for training and search. */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/heuristic.h"
#include "../src/search.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define P(...) do { printf(__VA_ARGS__); fflush(stdout); } while (0)

static double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(void)
{
    Rng rng; rng_seed(&rng, 42);
    Net *net = (Net *)malloc(sizeof(Net));
    net_init(net, 1);

    /* collect a set of mid-game states */
    enum { NS = 512 };
    State states[NS];
    for (int i = 0; i < NS; i++) {
        State st; lc_deal(&st, &rng);
        int steps = (int)rng_below(&rng, 40);
        for (int k = 0; k < steps && !st.over; k++) {
            Move mv[MAX_MOVES];
            int n = lc_moves(&st, mv);
            lc_apply(&st, mv[rng_below(&rng, (uint32_t)n)]);
        }
        states[i] = st;
    }

    /* feature extraction */
    Features f;
    double t0 = now();
    long iters = 300000;
    for (long i = 0; i < iters; i++) feat_extract(&states[i % NS], (int)(i & 1), &f);
    double dt = now() - t0;
    P("feat_extract   %8.0f /s\n", iters / dt);

    /* network forward */
    feat_extract(&states[0], 0, &f);
    t0 = now();
    iters = 300000;
    volatile float acc = 0;
    for (long i = 0; i < iters; i++) {
        feat_extract(&states[i % NS], (int)(i & 1), &f);
        acc += net_value(net, &f);
    }
    dt = now() - t0;
    P("feat+forward   %8.0f /s\n", iters / dt);

    /* heuristic evaluation */
    t0 = now();
    iters = 300000;
    for (long i = 0; i < iters; i++) acc += heur_eval(&states[i % NS], (int)(i & 1));
    dt = now() - t0;
    P("heur_eval      %8.0f /s\n", iters / dt);

    /* one-ply agents: moves per second */
    Agent a;
    agent_default(&a, AG_NET, net);
    t0 = now();
    iters = 20000;
    for (long i = 0; i < iters; i++) {
        State s = states[i % NS];
        if (s.over) continue;
        Move m = agent_move(&a, &s, &rng);
        acc += m.card;
    }
    dt = now() - t0;
    P("net one-ply    %8.0f moves/s\n", iters / dt);

    agent_default(&a, AG_HEUR, NULL);
    t0 = now();
    for (long i = 0; i < iters; i++) {
        State s = states[i % NS];
        if (s.over) continue;
        Move m = agent_move(&a, &s, &rng);
        acc += m.card;
    }
    dt = now() - t0;
    P("heur one-ply   %8.0f moves/s\n", iters / dt);

    /* full self-play games with the one-ply net policy */
    agent_default(&a, AG_NET, net);
    t0 = now();
    int games = 200;
    long plies = 0;
    for (int g = 0; g < games; g++) {
        State st; lc_deal(&st, &rng);
        while (!st.over) { lc_apply(&st, agent_move(&a, &st, &rng)); plies++; }
    }
    dt = now() - t0;
    P("net self-play  %8.1f games/s (%.0f plies/s, %.1f plies/game)\n",
           games / dt, plies / dt, (double)plies / games);

    /* search */
    agent_default(&a, AG_MCTS, net);
    for (int cfg = 0; cfg < 3; cfg++) {
        if (cfg == 0) { a.dets = 8;  a.sims = 80; }
        if (cfg == 1) { a.dets = 16; a.sims = 160; }
        if (cfg == 2) { a.dets = 32; a.sims = 400; }
        t0 = now();
        int nm = 60;
        for (int i = 0; i < nm; i++) {
            State s = states[i % NS];
            if (s.over) continue;
            Move m = search_move(&a, &s, &rng, NULL, NULL);
            acc += m.card;
        }
        dt = now() - t0;
        P("mcts d%-3d s%-4d %8.1f moves/s (%.0f ms/move)\n", a.dets, a.sims, nm / dt, 1000 * dt / nm);
    }
    return 0;
}
