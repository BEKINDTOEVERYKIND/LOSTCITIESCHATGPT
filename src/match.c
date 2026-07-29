#include "match.h"
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    const Agent *a, *b;
    int pairs, thread, nthread, rounds;
    uint64_t seed;
    double sum, sumsq, win_sum, win_sumsq;
    double wins, losses, draws, points_a, points_b, plies;
    int done;
} Job;

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static void pair_rng(Rng *rng, uint64_t seed, int pair, int stream)
{
    uint64_t key = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(pair + 1))
                        ^ (0xD1B54A32D192ED03ULL * (uint64_t)(stream + 1));
    rng_seed(rng, mix64(key));
}

/* One full match: `rounds` deals with cumulative context, the first player
 * alternating by round.  decks[r] supplies the deal for round r so a paired
 * rematch sees identical cards.  Scores are totals over all rounds for the
 * seats passed as first/second. */
static void play_one(const Agent *first, const Agent *second, int rounds,
                     uint8_t decks[][NCARD], Rng rng[2],
                     int *score0, int *score1, double *plies)
{
    const Agent *ag[2] = { first, second };
    int cum[2] = { 0, 0 };
    for (int r = 0; r < rounds; r++) {
        State st;
        lc_deal_from_deck(&st, decks[r]);
        st.round = (uint8_t)r;
        st.cum[0] = (int16_t)cum[0];
        st.cum[1] = (int16_t)cum[1];
        st.turn = (uint8_t)(r & 1);
        while (!st.over) {
            /* Separate per-agent streams make results independent of thread
             * scheduling and prevent one agent's search budget from consuming
             * randomness that would otherwise have gone to its opponent. */
            Move m = agent_move(ag[st.turn], &st, &rng[st.turn]);
            lc_apply(&st, m);
        }
        cum[0] += lc_score(&st, 0);
        cum[1] += lc_score(&st, 1);
        *plies += st.nply;
    }
    *score0 = cum[0];
    *score1 = cum[1];
}

static void *worker(void *arg)
{
    Job *j = (Job *)arg;
    for (int g = j->thread; g < j->pairs; g += j->nthread) {
        Rng deal_rng;
        pair_rng(&deal_rng, j->seed, g, 2);
        uint8_t decks[MATCH_ROUNDS][NCARD];
        for (int r = 0; r < j->rounds; r++) {
            for (int i = 0; i < NCARD; i++) decks[r][i] = (uint8_t)i;
            for (int i = NCARD - 1; i > 0; i--) {
                uint32_t k = rng_below(&deal_rng, (uint32_t)i + 1);
                uint8_t t = decks[r][i]; decks[r][i] = decks[r][k]; decks[r][k] = t;
            }
        }
        int s0, s1;
        double pair = 0.0, pair_win = 0.0;
        Rng arng, brng, first_rng[2], second_rng[2];
        pair_rng(&arng, j->seed, g, 0);
        pair_rng(&brng, j->seed, g, 1);
        first_rng[0] = arng; first_rng[1] = brng;
        second_rng[0] = brng; second_rng[1] = arng;

        play_one(j->a, j->b, j->rounds, decks, first_rng, &s0, &s1, &j->plies);
        j->points_a += s0; j->points_b += s1;
        pair += s0 - s1;
        if (s0 > s1) { j->wins++; pair_win += 1.0; }
        else if (s0 < s1) j->losses++;
        else { j->draws++; pair_win += 0.5; }

        play_one(j->b, j->a, j->rounds, decks, second_rng, &s0, &s1, &j->plies);
        j->points_a += s1; j->points_b += s0;
        pair += s1 - s0;
        if (s1 > s0) { j->wins++; pair_win += 1.0; }
        else if (s1 < s0) j->losses++;
        else { j->draws++; pair_win += 0.5; }
        j->sum += pair;
        j->sumsq += pair * pair;
        j->win_sum += pair_win;
        j->win_sumsq += pair_win * pair_win;
        j->done++;
    }
    return NULL;
}

void match_run_r(const Agent *a, const Agent *b, int pairs, int nthread,
                 uint64_t seed, int rounds, MatchResult *out)
{
    if (nthread < 1) nthread = 1;
    if (rounds < 1) rounds = 1;
    if (rounds > MATCH_ROUNDS) rounds = MATCH_ROUNDS;
    Job *jobs = (Job *)calloc((size_t)nthread, sizeof(Job));
    pthread_t *th = (pthread_t *)calloc((size_t)nthread, sizeof(pthread_t));
    for (int i = 0; i < nthread; i++) {
        jobs[i].a = a; jobs[i].b = b; jobs[i].pairs = pairs;
        jobs[i].thread = i; jobs[i].nthread = nthread; jobs[i].seed = seed;
        jobs[i].rounds = rounds;
    }
    for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, worker, &jobs[i]);
    for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);

    double sum = 0, sumsq = 0, win_sum = 0, win_sumsq = 0;
    double w = 0, l = 0, d = 0, pa = 0, pb = 0, pl = 0;
    int done = 0;
    for (int i = 0; i < nthread; i++) {
        sum += jobs[i].sum; sumsq += jobs[i].sumsq;
        win_sum += jobs[i].win_sum; win_sumsq += jobs[i].win_sumsq;
        w += jobs[i].wins; l += jobs[i].losses; d += jobs[i].draws;
        pa += jobs[i].points_a; pb += jobs[i].points_b; pl += jobs[i].plies;
        done += jobs[i].done;
    }
    free(jobs); free(th);
    if (done == 0) { memset(out, 0, sizeof(*out)); return; }
    double ngames = 2.0 * done;
    double margin_var = 0.0, win_var = 0.0;
    if (done > 1) {
        margin_var = (sumsq - sum * sum / done) / (done - 1);
        win_var = (win_sumsq - win_sum * win_sum / done) / (done - 1);
        if (margin_var < 0.0) margin_var = 0.0;
        if (win_var < 0.0) win_var = 0.0;
    }
    out->pairs = done;
    out->games = (int)ngames;
    out->margin = sum / done / 2.0;
    out->margin_se = sqrt(margin_var / done) / 2.0;
    out->winrate = win_sum / done / 2.0;
    out->winrate_se = sqrt(win_var / done) / 2.0;
    out->points_a = pa / ngames;
    out->points_b = pb / ngames;
    out->plies = pl / ngames;
    out->wins = w; out->losses = l; out->draws = d;
}

void match_run(const Agent *a, const Agent *b, int pairs, int nthread,
               uint64_t seed, MatchResult *out)
{
    match_run_r(a, b, pairs, nthread, seed, 1, out);
}
