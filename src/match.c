#include "match.h"
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>

typedef struct {
    const Agent *a, *b;
    int pairs, thread, nthread, rounds;
    uint64_t seed, pair_start;
    MatchPairResult *pair_out;
    unsigned char *pair_complete;
    double sum, sumsq, win_sum, win_sumsq;
    double wins, losses, draws, points_a, points_b, plies;
    uint64_t capped_rounds;
    int done;
    atomic_int finished;
    unsigned char join_failed;
} Job;

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static void pair_rng(Rng *rng, uint64_t seed, uint64_t pair, int stream)
{
    uint64_t key = seed ^ (0x9E3779B97F4A7C15ULL * (pair + UINT64_C(1)))
                        ^ (0xD1B54A32D192ED03ULL * (uint64_t)(stream + 1));
    rng_seed(rng, mix64(key));
}

/* One full match: `rounds` deals with cumulative context, the first player
 * alternating by round.  decks[r] supplies the deal for round r so a paired
 * rematch sees identical cards.  Scores are totals over all rounds for the
 * seats passed as first/second. */
static void play_one(const Agent *first, const Agent *second, int rounds,
                     uint8_t decks[][NCARD], Rng rng[2],
                     int *score0, int *score1, int *plies,
                     int *capped_rounds)
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
        if (st.deck_left > 0) (*capped_rounds)++;
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
        uint64_t absolute_pair = j->pair_start + (uint64_t)g;
        Rng deal_rng;
        pair_rng(&deal_rng, j->seed, absolute_pair, 2);
        uint8_t decks[MATCH_ROUNDS][NCARD];
        for (int r = 0; r < j->rounds; r++) {
            for (int i = 0; i < NCARD; i++) decks[r][i] = (uint8_t)i;
            for (int i = NCARD - 1; i > 0; i--) {
                uint32_t k = rng_below(&deal_rng, (uint32_t)i + 1);
                uint8_t t = decks[r][i]; decks[r][i] = decks[r][k]; decks[r][k] = t;
            }
        }
        int s0, s1, first_plies = 0, second_plies = 0;
        int first_caps = 0, second_caps = 0;
        double pair = 0.0, pair_win = 0.0;
        Rng arng, brng, first_rng[2], second_rng[2];
        pair_rng(&arng, j->seed, absolute_pair, 0);
        pair_rng(&brng, j->seed, absolute_pair, 1);
        first_rng[0] = arng; first_rng[1] = brng;
        second_rng[0] = brng; second_rng[1] = arng;

        play_one(j->a, j->b, j->rounds, decks, first_rng, &s0, &s1,
                 &first_plies, &first_caps);
        int first_a = s0, first_b = s1;
        j->points_a += s0; j->points_b += s1;
        j->plies += first_plies;
        j->capped_rounds += (uint64_t)first_caps;
        pair += s0 - s1;
        if (s0 > s1) { j->wins++; pair_win += 1.0; }
        else if (s0 < s1) j->losses++;
        else { j->draws++; pair_win += 0.5; }

        play_one(j->b, j->a, j->rounds, decks, second_rng, &s0, &s1,
                 &second_plies, &second_caps);
        int second_a = s1, second_b = s0;
        j->points_a += s1; j->points_b += s0;
        j->plies += second_plies;
        j->capped_rounds += (uint64_t)second_caps;
        pair += s1 - s0;
        if (s1 > s0) { j->wins++; pair_win += 1.0; }
        else if (s1 < s0) j->losses++;
        else { j->draws++; pair_win += 0.5; }
        j->sum += pair;
        j->sumsq += pair * pair;
        j->win_sum += pair_win;
        j->win_sumsq += pair_win * pair_win;
        if (j->pair_out) {
            MatchPairResult *row = &j->pair_out[g];
            row->index = absolute_pair;
            row->score_a[0] = first_a;
            row->score_b[0] = first_b;
            row->score_a[1] = second_a;
            row->score_b[1] = second_b;
            row->plies[0] = first_plies;
            row->plies[1] = second_plies;
            row->capped_rounds[0] = first_caps;
            row->capped_rounds[1] = second_caps;
            /* Publish completion only after every field in the row.  Each
             * worker owns a disjoint set of g values and the joins below
             * provide the synchronization before this array is inspected. */
            j->pair_complete[g] = 1;
        }
        j->done++;
    }
    atomic_store_explicit(&j->finished, 1, memory_order_release);
    return NULL;
}

int match_run_range_r(const Agent *a, const Agent *b, uint64_t pair_start,
                      int pairs, int nthread, uint64_t seed, int rounds,
                      MatchPairResult *pair_out, MatchResult *out)
{
    if (out) memset(out, 0, sizeof(*out));
    if (!a || !b || !out || pairs < 0 || pairs > MATCH_MAX_PAIRS ||
        nthread < 1 || rounds < 1 || rounds > MATCH_ROUNDS ||
        (pairs > 0 &&
         pair_start > UINT64_MAX - ((uint64_t)pairs - UINT64_C(1))))
        return -1;
    if (pairs == 0) return 0;
    if (nthread > pairs) nthread = pairs;
    Job *jobs = (Job *)calloc((size_t)nthread, sizeof(Job));
    pthread_t *th = (pthread_t *)calloc((size_t)nthread, sizeof(pthread_t));
    unsigned char *pair_complete = pair_out
        ? (unsigned char *)calloc((size_t)pairs, 1) : NULL;
    if (!jobs || !th || (pair_out && !pair_complete)) {
        free(pair_complete); free(jobs); free(th);
        return -1;
    }
    for (int i = 0; i < nthread; i++) {
        jobs[i].a = a; jobs[i].b = b; jobs[i].pairs = pairs;
        jobs[i].thread = i; jobs[i].nthread = nthread; jobs[i].seed = seed;
        jobs[i].rounds = rounds; jobs[i].pair_start = pair_start;
        jobs[i].pair_out = pair_out;
        jobs[i].pair_complete = pair_complete;
        atomic_init(&jobs[i].finished, 0);
    }
    int created = 0, create_error = 0, join_error = 0;
    for (int i = 0; i < nthread; i++) {
        if (pthread_create(&th[i], NULL, worker, &jobs[i]) != 0) {
            create_error = 1;
            break;
        }
        created++;
    }
    for (int i = 0; i < created; i++) {
        if (pthread_join(th[i], NULL) != 0) {
            jobs[i].join_failed = 1;
            join_error = 1;
        }
    }
    if (join_error) {
        /* A valid join of one of our joinable thread IDs should not fail.  If
         * the platform nevertheless reports an error, do not free memory a
         * worker could still access: wait for its final release-store and
         * detach the unjoinable handle before failing closed. */
        for (int i = 0; i < created; i++) {
            if (!jobs[i].join_failed) continue;
            while (!atomic_load_explicit(&jobs[i].finished,
                                         memory_order_acquire))
                sched_yield();
            (void)pthread_detach(th[i]);
        }
    }
    if (create_error || join_error || created != nthread) {
        free(pair_complete); free(jobs); free(th);
        return -1;
    }

    double sum = 0, sumsq = 0, win_sum = 0, win_sumsq = 0;
    double w = 0, l = 0, d = 0, pa = 0, pb = 0, pl = 0;
    uint64_t capped_rounds = 0;
    int done = 0;
    for (int i = 0; i < nthread; i++) {
        int expected = i < pairs ? 1 + (pairs - 1 - i) / nthread : 0;
        if (jobs[i].done != expected) {
            free(pair_complete); free(jobs); free(th);
            return -1;
        }
        sum += jobs[i].sum; sumsq += jobs[i].sumsq;
        win_sum += jobs[i].win_sum; win_sumsq += jobs[i].win_sumsq;
        w += jobs[i].wins; l += jobs[i].losses; d += jobs[i].draws;
        pa += jobs[i].points_a; pb += jobs[i].points_b; pl += jobs[i].plies;
        capped_rounds += jobs[i].capped_rounds;
        done += jobs[i].done;
    }
    if (done != pairs) {
        free(pair_complete); free(jobs); free(th);
        return -1;
    }
    if (pair_complete) {
        for (int i = 0; i < pairs; i++) {
            if (pair_complete[i] != 1) {
                free(pair_complete); free(jobs); free(th);
                return -1;
            }
        }
    }
    free(pair_complete); free(jobs); free(th);
    if (done == 0) { memset(out, 0, sizeof(*out)); return 0; }
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
    out->capped_rounds = capped_rounds;
    return 0;
}

int match_run_r(const Agent *a, const Agent *b, int pairs, int nthread,
                uint64_t seed, int rounds, MatchResult *out)
{
    return match_run_range_r(a, b, 0, pairs, nthread, seed, rounds,
                             NULL, out);
}

int match_run(const Agent *a, const Agent *b, int pairs, int nthread,
              uint64_t seed, MatchResult *out)
{
    return match_run_r(a, b, pairs, nthread, seed, 1, out);
}
