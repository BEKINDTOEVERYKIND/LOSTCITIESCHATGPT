/* searchcmp -- where does rollout search actually disagree with the policy?
 *
 * Plays policy-vs-policy self-play matches, and at every ply ALSO runs the
 * rollout search (its result is recorded, not played, so the trajectory stays
 * pure policy).  Each decision contributes one record:
 *
 *   confidence  the policy's probability on its own top move
 *   phase       deck cards left (early > 30, mid 11..30, late <= 10)
 *   clutch      round 3 with the match within 25 points
 *   disagree    rollout picked a different move than the policy's top
 *   dq          Q(rollout's pick) - Q(policy's top), in rollout's own units
 *               (>= 0 by construction when the top move is a candidate; an
 *               estimate of what switching buys, inflated by search noise)
 *
 * The aggregate table is the evidence for a confidence-gated hybrid: search
 * only where the policy is unsure enough that searching can change anything.
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define NCONF 5
#define NPHASE 3

static const float CONF_LO[NCONF] = { 0.00f, 0.40f, 0.60f, 0.80f, 0.95f };
static const char *CONF_NAME[NCONF] = { "<0.40", "0.40-0.60", "0.60-0.80", "0.80-0.95", ">=0.95" };
static const char *PHASE_NAME[NPHASE] = { "early", "mid", "late" };

typedef struct {
    long n, disagree;
    double dq_sum;
} Cell;

typedef struct {
    const Agent *actor;
    Agent srch;
    int matches, thread, nthread;
    uint64_t seed;
    Cell cell[NCONF][NPHASE];
    Cell clutch[NCONF];       /* round 3, |match diff| <= 25 */
    Cell rounds[MATCH_ROUNDS][NCONF];
    long plies;
} Job;

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static int conf_bucket(float p)
{
    for (int i = NCONF - 1; i >= 0; i--)
        if (p >= CONF_LO[i]) return i;
    return 0;
}

static void *worker(void *arg)
{
    Job *j = (Job *)arg;
    Move mv[MAX_MOVES];
    float pr[MAX_MOVES];

    for (int g = j->thread; g < j->matches; g += j->nthread) {
        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < MATCH_ROUNDS; rd++) {
            Rng deal_rng;
            rng_seed(&deal_rng, mix64(j->seed ^
                     ((uint64_t)g << 16) ^ (uint64_t)rd));
            State st;
            lc_deal(&st, &deal_rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)cum[0];
            st.cum[1] = (int16_t)cum[1];
            st.turn = (uint8_t)(rd & 1);
            while (!st.over) {
                int n = policy_probs_sym(j->actor->net, &st, mv, pr, NULL,
                                         j->actor->symmetries);
                int top = 0;
                for (int i = 1; i < n; i++) if (pr[i] > pr[top]) top = i;

                if (n > 1) {
                    SearchStats ss;
                    Rng eval_rng;
                    rng_seed(&eval_rng, mix64(j->seed ^
                             ((uint64_t)g << 32) ^
                             ((uint64_t)rd << 24) ^ st.nply));
                    Move rm =
                        rollout_move(&j->srch, &st, &eval_rng, NULL, &ss);
                    /* Q of the selected and policy moves among candidates. */
                    double qtop = -1e30, qselected = -1e30;
                    for (int i = 0; i < ss.n; i++) {
                        if (ss.mv[i].card == mv[top].card &&
                            ss.mv[i].discard == mv[top].discard &&
                            ss.mv[i].draw == mv[top].draw)
                            qtop = ss.q[i];
                        if (ss.mv[i].card == rm.card &&
                            ss.mv[i].discard == rm.discard &&
                            ss.mv[i].draw == rm.draw)
                            qselected = ss.q[i];
                    }
                    int chosen_differs = !(rm.card == mv[top].card &&
                                           rm.discard == mv[top].discard &&
                                           rm.draw == mv[top].draw);
                    double dq =
                        (qtop > -1e29 && qselected > -1e29)
                            ? (qselected - qtop) : 0.0;

                    int cb = conf_bucket(pr[top]);
                    int ph = st.deck_left > 30 ? 0 : (st.deck_left > 10 ? 1 : 2);
                    Cell *c = &j->cell[cb][ph];
                    c->n++; c->disagree += chosen_differs; c->dq_sum += dq;
                    Cell *r = &j->rounds[rd][cb];
                    r->n++; r->disagree += chosen_differs; r->dq_sum += dq;
                    if (rd == MATCH_ROUNDS - 1 && abs(cum[0] - cum[1]) <= 25) {
                        Cell *cl = &j->clutch[cb];
                        cl->n++; cl->disagree += chosen_differs; cl->dq_sum += dq;
                    }
                }
                /* the trajectory itself follows the raw policy */
                lc_apply(&st, mv[top]);
                j->plies++;
            }
            cum[0] += lc_score(&st, 0);
            cum[1] += lc_score(&st, 1);
        }
    }
    return NULL;
}

static void print_cells(const char *title, Cell *row, int stride, int nrow,
                        const char **names)
{
    printf("\n%s\n", title);
    printf("  %-11s %8s %10s %10s\n", "confidence", "plies", "disagree", "mean dQ");
    for (int i = 0; i < nrow; i++) {
        Cell *c = (Cell *)((char *)row + (size_t)i * stride);
        if (c->n == 0) { printf("  %-11s %8s\n", names[i], "-"); continue; }
        printf("  %-11s %8ld %9.1f%% %9.2f\n", names[i], c->n,
               100.0 * c->disagree / c->n, c->dq_sum / c->n);
    }
}

int main(int argc, char **argv)
{
    const char *net_path = "data/champion.bin";
    const char *actor_spec = NULL, *eval_spec = NULL;
    int matches = 40, nthread = 4, worlds = 96, cands = 5;
    uint64_t seed = 424242;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) net_path = argv[++i];
        else if (!strcmp(argv[i], "-a") && i + 1 < argc) actor_spec = argv[++i];
        else if (!strcmp(argv[i], "-e") && i + 1 < argc) eval_spec = argv[++i];
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) matches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) worlds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) cands = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) nthread = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else {
            fprintf(stderr, "usage: %s [-a actor_spec] [-e rollout_spec] "
                    "[-n NET] [-m matches] [-w worlds] [-c cands]\n",
                    argv[0]);
            return 1;
        }
    }

    char actor_buf[512];
    if (!actor_spec) {
        snprintf(actor_buf, sizeof actor_buf, "policy:%s:0:20", net_path);
        actor_spec = actor_buf;
    }
    Agent actor;
    spec_parse(actor_spec, &actor);
    if (actor.kind != AG_POLICY || !actor.net) {
        fprintf(stderr, "searchcmp: actor must be a network policy\n");
        return 1;
    }

    Agent evaluator;
    if (eval_spec) {
        spec_parse(eval_spec, &evaluator);
        if (evaluator.kind != AG_ROLLOUT || !evaluator.net) {
            fprintf(stderr, "searchcmp: evaluator must be a network rollout\n");
            return 1;
        }
    } else {
        agent_default(&evaluator, AG_ROLLOUT, actor.net);
        evaluator.dets = worlds;
        evaluator.root_width = cands;
        evaluator.symmetries = actor.symmetries;
    }

    Job *jobs = calloc((size_t)nthread, sizeof(Job));
    pthread_t th[64];
    for (int i = 0; i < nthread; i++) {
        jobs[i].actor = &actor;
        jobs[i].srch = evaluator;
        jobs[i].matches = matches;
        jobs[i].thread = i; jobs[i].nthread = nthread;
        jobs[i].seed = seed;
    }
    for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, worker, &jobs[i]);
    for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);

    /* merge */
    Job tot;
    memset(&tot, 0, sizeof tot);
    for (int i = 0; i < nthread; i++) {
        for (int a = 0; a < NCONF; a++) {
            for (int b = 0; b < NPHASE; b++) {
                tot.cell[a][b].n += jobs[i].cell[a][b].n;
                tot.cell[a][b].disagree += jobs[i].cell[a][b].disagree;
                tot.cell[a][b].dq_sum += jobs[i].cell[a][b].dq_sum;
            }
            tot.clutch[a].n += jobs[i].clutch[a].n;
            tot.clutch[a].disagree += jobs[i].clutch[a].disagree;
            tot.clutch[a].dq_sum += jobs[i].clutch[a].dq_sum;
            for (int r = 0; r < MATCH_ROUNDS; r++) {
                tot.rounds[r][a].n += jobs[i].rounds[r][a].n;
                tot.rounds[r][a].disagree += jobs[i].rounds[r][a].disagree;
                tot.rounds[r][a].dq_sum += jobs[i].rounds[r][a].dq_sum;
            }
        }
        tot.plies += jobs[i].plies;
    }

    printf("searchcmp: %d matches, actor %s, rollout %d worlds x %d candidates, "
           "%ld plies\n", matches, actor_spec, evaluator.dets,
           evaluator.root_width, tot.plies);

    long alln = 0, alld = 0;
    double alldq = 0;
    for (int a = 0; a < NCONF; a++)
        for (int b = 0; b < NPHASE; b++) {
            alln += tot.cell[a][b].n; alld += tot.cell[a][b].disagree;
            alldq += tot.cell[a][b].dq_sum;
        }
    printf("overall: %ld decisions, disagree %.1f%%, mean dQ %.2f pts\n",
           alln, 100.0 * alld / alln, alldq / alln);

    for (int b = 0; b < NPHASE; b++) {
        char t[64];
        snprintf(t, sizeof t, "phase: %s deck", PHASE_NAME[b]);
        Cell col[NCONF];
        for (int a = 0; a < NCONF; a++) col[a] = tot.cell[a][b];
        print_cells(t, col, sizeof(Cell), NCONF, CONF_NAME);
    }
    for (int r = 0; r < MATCH_ROUNDS; r++) {
        char t[64];
        snprintf(t, sizeof t, "round %d", r + 1);
        print_cells(t, tot.rounds[r], sizeof(Cell), NCONF, CONF_NAME);
    }
    print_cells("clutch (round 3, match within 25)", tot.clutch, sizeof(Cell), NCONF, CONF_NAME);
    return 0;
}
