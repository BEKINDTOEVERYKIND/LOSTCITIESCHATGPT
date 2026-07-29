/* mine -- class-conditioned error mining with trustworthy labels.
 *
 * Expert review of analysed games found the policy's mistakes cluster into
 * mechanically detectable classes: playing a card that forecloses obtainable
 * value while a zero-skip play exists, refusing a pile top it could profitably
 * play, stalling with plenty of its own plays left, discarding onto a pile
 * top it wants, and hedging onto dominated discards.  Blanket expert
 * iteration paid more in blurred sharpness than the fixed leaks returned, so
 * this tool builds a *corrections-only* corpus instead: cheap sampled
 * self-play visits states, the detectors flag class instances, and only
 * flagged states are labeled -- by the estimator configuration the review
 * validated (sampled playouts against determinism bias, dominated-discard
 * pruning, draw-variant expansion, the points-and-significance override
 * gates).  A state becomes a training sample only when that search actually
 * disagrees with the policy's argmax; training mixes these with plain
 * self-policy anchor games so the rest of the policy stays put.
 *
 *   ./bin/mine --net data/c8.bin --games 400 --threads 4 \
 *              --out data/corr.smp [--dets 256] [--dup 4]
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/net.h"
#include "../src/search.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI_K 12
#define SMP_MAGIC 0x4C435344u

typedef struct {
    State st;
    float target;
    uint8_t persp;
    uint8_t npi;
    uint16_t pmv[PI_K];
    float ppr[PI_K];
} Sample;

/* can player p legally play card c right now? */
static int playable(const State *st, int p, int c)
{
    int s = CARD_SUIT(c);
    return CARD_IS_WAGER(c) ? (st->exp_top[p][s] == 0)
                            : (CARD_VALUE(c) > st->exp_top[p][s]);
}

/* values a play of c would skip that the mover cannot rule out obtaining:
 * neither played nor discarded (own-hand skips count -- playing over your
 * own lower card is exactly the misorder being mined) */
static int skip_cost(const State *st, int c)
{
    if (CARD_IS_WAGER(c)) return 0;
    int p = st->turn, s = CARD_SUIT(c);
    int n = 0;
    uint64_t gone = st->played[0] | st->played[1] | st->discarded;
    for (int v = st->exp_top[p][s] + 1; v < CARD_VALUE(c); v++) {
        int id = s * NRANK + v + 1;
        if (!((gone >> id) & 1ULL)) n++;
    }
    return n;
}

/* class detectors over the policy distribution; returns a bitmask */
static int detect(const State *st, const Move *mv, const float *pr, int n)
{
    const int p = st->turn;
    int cls = 0;
    int top = 0;
    for (int i = 1; i < n; i++) if (pr[i] > pr[top]) top = i;

    /* playable-now cards in hand, and the best zero-skip play's prob mass */
    uint8_t hc[HAND_SIZE];
    int nh = lc_hand_cards(st, p, hc);
    int nplay = 0, zero_skip_exists = 0;
    for (int i = 0; i < nh; i++)
        if (playable(st, p, hc[i])) {
            nplay++;
            if (!CARD_IS_WAGER(hc[i]) && CARD_VALUE(hc[i]) >= 2 && skip_cost(st, hc[i]) == 0)
                zero_skip_exists = 1;
        }

    /* 1: top move plays over obtainable value while a zero-skip play exists */
    if (!mv[top].discard && skip_cost(st, mv[top].card) >= 1 && zero_skip_exists)
        cls |= 1;

    /* 2: a pile top is a playable number WORTH TAKING for the mover (decent
     * value or a wagered expedition), yet almost no policy mass draws it */
    for (int s = 0; s < NSUIT; s++) {
        if (st->pile_n[s] == 0) continue;
        int t = st->pile[s][st->pile_n[s] - 1];
        if (CARD_IS_WAGER(t) || !playable(st, p, t)) continue;
        if (CARD_VALUE(t) < 4 && st->exp_wager[p][s] == 0) continue;
        float mass = 0.0f;
        for (int i = 0; i < n; i++) if (mv[i].draw == s + 1) mass += pr[i];
        if (mass < 0.05f) { cls |= 2; break; }
    }

    /* 3: top move stalls (pile-draws a card the mover cannot play) with
     * plenty of own plays and deck left */
    if (mv[top].draw != 0 && st->deck_left >= 8 && nplay >= 3) {
        int t = st->pile[mv[top].draw - 1][st->pile_n[mv[top].draw - 1] - 1];
        if (!playable(st, p, t)) cls |= 4;
    }

    /* 4: top move discards onto a pile whose current top the mover can play
     * -- burying its own draw option */
    if (mv[top].discard) {
        int s = CARD_SUIT(mv[top].card);
        if (st->pile_n[s] > 0 && playable(st, p, st->pile[s][st->pile_n[s] - 1]))
            cls |= 8;
    }

    /* 5: visible probability on a dominated discard */
    uint64_t dead = lc_dead_cards(st);
    if (dead & st->hand[p]) {
        for (int i = 0; i < n; i++)
            if (pr[i] >= 0.01f && lc_discard_dominated(st, mv[i], dead)) { cls |= 16; break; }
    }
    return cls;
}

typedef struct {
    const Net *net;
    int games, thread, nthread, dets, dup;
    uint64_t seed;
    FILE *out;
    pthread_mutex_t *lk;
    long flagged, corrected, plies, written;
    long bycls[5];
} Job;

static void *worker(void *arg)
{
    Job *j = (Job *)arg;
    Rng rng;
    rng_seed(&rng, j->seed + 77777ULL * (uint64_t)(j->thread + 1));

    Agent lab;
    agent_default(&lab, AG_ROLLOUT, j->net);
    lab.dets = j->dets;
    lab.root_width = 5;
    lab.gate = 0.0f;
    lab.eval_cand = 4;
    lab.override_k = 3.0f;      /* override_min 4 and prune_dom on by default */
    lab.playout_sample = 1;

    Features f;
    for (int g = j->thread; g < j->games; g += j->nthread) {
        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < MATCH_ROUNDS; rd++) {
            State st;
            lc_deal(&st, &rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)cum[0];
            st.cum[1] = (int16_t)cum[1];
            st.turn = (uint8_t)(rd & 1);
            while (!st.over) {
                Move mv[MAX_MOVES];
                float pr[MAX_MOVES];
                int n = policy_probs(j->net, &st, mv, pr, NULL);
                if (n <= 0) break;
                j->plies++;

                int cls = detect(&st, mv, pr, n);
                /* the dominant class must not monopolize the corpus: a
                 * corpus that only ever says "take the pile" teaches the
                 * draw head a direction, not a judgment (measured: ply
                 * inflation 143->161 and a 15-point h2h collapse) */
                if (cls == 2 && (rng_next(&rng) & 1)) cls = 0;
                if (cls) {
                    j->flagged++;
                    SearchStats ss;
                    memset(&ss, 0, sizeof ss);
                    float sval = 0.0f;
                    Move lm = rollout_move(&lab, &st, &rng, &sval, &ss);
                    int top = 0;
                    for (int i = 1; i < n; i++) if (pr[i] > pr[top]) top = i;
                    int agree = lm.card == mv[top].card && lm.discard == mv[top].discard &&
                                lm.draw == mv[top].draw;
                    /* corrections AND confirmations: the same flagged state
                     * where the search agrees with the policy is the
                     * counterweight that keeps a class from becoming a
                     * direction */
                    {
                        if (!agree) {
                            j->corrected++;
                            for (int b = 0; b < 5; b++) if (cls & (1 << b)) j->bycls[b]++;
                        }
                        Sample s;
                        memset(&s, 0, sizeof s);
                        s.st = st;
                        s.persp = st.turn;
                        feat_extract(&st, st.turn, &f);
                        s.target = net_value(j->net, &f) * VAL_SCALE; /* zero value grad */
                        /* the target is the GATED choice -- the move the
                         * validated selection rule actually plays.  The
                         * first corpus generation softmaxed raw Q over all
                         * evaluated candidates, whose argmax is the ungated
                         * Q-maximum: exactly the estimator measured harmful
                         * (min_cand, 42.8%), and the reason five fine-tune
                         * variants collapsed while self-target and no-data
                         * controls sat at parity. */
                        s.pmv[0] = MOVE_PACK(lm);
                        s.ppr[0] = 1.0f;
                        s.npi = 1;
                        pthread_mutex_lock(j->lk);
                        int reps = agree ? 1 : j->dup;
                        for (int r = 0; r < reps; r++)
                            fwrite(&s, sizeof s, 1, j->out);
                        j->written += reps;
                        pthread_mutex_unlock(j->lk);
                    }
                }

                /* diverse traversal: sample the policy early, argmax later */
                int pick;
                if (st.nply < 30) pick = sample_index(pr, n, &rng);
                else { pick = 0; for (int i = 1; i < n; i++) if (pr[i] > pr[pick]) pick = i; }
                lc_apply(&st, mv[pick]);
            }
            cum[0] += lc_score(&st, 0);
            cum[1] += lc_score(&st, 1);
        }
    }
    return NULL;
}

/* --filter IN OUT: keep only samples whose labeled action (card + play/
 * discard) differs from the net's argmax action.  Draw-only corrections are
 * dropped: the 6-logit factored draw head turns any net draw-direction
 * pressure into a global reweighting (measured as ply inflation and a
 * 17-point collapse), while action corrections are state-specific. */
static int filter_mode(const Net *net, const char *inp, const char *outp)
{
    FILE *fi = fopen(inp, "rb");
    if (!fi) { fprintf(stderr, "mine: cannot open %s\n", inp); return 1; }
    uint32_t h[4];
    uint64_t cnt;
    if (fread(h, sizeof h, 1, fi) != 1 || fread(&cnt, sizeof cnt, 1, fi) != 1 ||
        h[0] != SMP_MAGIC || h[1] != sizeof(Sample)) { fprintf(stderr, "mine: bad file\n"); return 1; }
    FILE *fo = fopen(outp, "wb");
    uint64_t kept = 0;
    fwrite(h, sizeof h, 1, fo);
    fwrite(&kept, sizeof kept, 1, fo);
    Sample s;
    while (fread(&s, sizeof s, 1, fi) == 1) {
        Move mv[MAX_MOVES];
        float pr[MAX_MOVES];
        int n = policy_probs(net, &s.st, mv, pr, NULL);
        int top = 0;
        for (int i = 1; i < n; i++) if (pr[i] > pr[top]) top = i;
        int tb = 0;
        for (int i = 1; i < s.npi; i++) if (s.ppr[i] > s.ppr[tb]) tb = i;
        int tcard = s.pmv[tb] % 60, tdisc = (s.pmv[tb] / 60) % 2;
        if (tcard != mv[top].card || tdisc != mv[top].discard) {
            fwrite(&s, sizeof s, 1, fo);
            kept++;
        }
    }
    fseek(fo, sizeof h, SEEK_SET);
    fwrite(&kept, sizeof kept, 1, fo);
    fclose(fi); fclose(fo);
    printf("mine --filter: kept %llu action-level corrections of %llu\n",
           (unsigned long long)kept, (unsigned long long)cnt);
    return 0;
}

/* --selftarget IN OUT: same states, targets replaced by the net's own
 * policy top-k.  A structurally identical but semantically null corpus:
 * if training on THIS collapses, the sample pipeline is broken; if it is
 * inert, the labels are the poison. */
static int selftarget_mode(const Net *net, const char *inp, const char *outp)
{
    FILE *fi = fopen(inp, "rb");
    FILE *fo = fopen(outp, "wb");
    if (!fi || !fo) { fprintf(stderr, "mine: open failed\n"); return 1; }
    uint32_t h[4]; uint64_t cnt;
    if (fread(h, sizeof h, 1, fi) != 1 || fread(&cnt, sizeof cnt, 1, fi) != 1) return 1;
    fwrite(h, sizeof h, 1, fo); fwrite(&cnt, sizeof cnt, 1, fo);
    Sample s;
    while (fread(&s, sizeof s, 1, fi) == 1) {
        Move mv[MAX_MOVES]; float pr[MAX_MOVES];
        int n = policy_probs(net, &s.st, mv, pr, NULL);
        int idx[MAX_MOVES];
        for (int i = 0; i < n; i++) idx[i] = i;
        int k = n < PI_K ? n : PI_K;
        for (int i = 0; i < k; i++) {
            int b = i;
            for (int j2 = i + 1; j2 < n; j2++) if (pr[idx[j2]] > pr[idx[b]]) b = j2;
            int t = idx[i]; idx[i] = idx[b]; idx[b] = t;
        }
        float tot = 0.0f;
        for (int i = 0; i < k; i++) tot += pr[idx[i]];
        if (tot <= 0.0f) tot = 1.0f;
        for (int i = 0; i < k; i++) { s.pmv[i] = MOVE_PACK(mv[idx[i]]); s.ppr[i] = pr[idx[i]] / tot; }
        for (int i = k; i < PI_K; i++) { s.pmv[i] = 0; s.ppr[i] = 0.0f; }
        s.npi = (uint8_t)k;
        fwrite(&s, sizeof s, 1, fo);
    }
    fclose(fi); fclose(fo);
    printf("mine --selftarget: rewrote %llu samples\n", (unsigned long long)cnt);
    return 0;
}

int main(int argc, char **argv)
{
    const char *netpath = "data/c8.bin", *outpath = "data/corr.smp";
    const char *filter_in = NULL, *filter_out = NULL, *self_in = NULL, *self_out = NULL;
    int games = 200, nthread = 4, dets = 256, dup = 4;
    uint64_t seed = 20260729;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--net") && i + 1 < argc) netpath = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outpath = argv[++i];
        else if (!strcmp(argv[i], "--games") && i + 1 < argc) games = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthread = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dets") && i + 1 < argc) dets = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dup") && i + 1 < argc) dup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--filter") && i + 2 < argc) { filter_in = argv[++i]; filter_out = argv[++i]; }
        else if (!strcmp(argv[i], "--selftarget") && i + 2 < argc) { self_in = argv[++i]; self_out = argv[++i]; }
        else { fprintf(stderr, "usage: %s [--net N] [--out F] [--games G] [--threads T] [--dets D] [--dup K]\n", argv[0]); return 1; }
    }
    Net *net = (Net *)malloc(sizeof(Net));
    if (!net || net_load(net, netpath)) { fprintf(stderr, "mine: cannot load %s\n", netpath); return 1; }
    if (filter_in) return filter_mode(net, filter_in, filter_out);
    if (self_in) return selftarget_mode(net, self_in, self_out);

    FILE *out = fopen(outpath, "wb");
    if (!out) { fprintf(stderr, "mine: cannot open %s\n", outpath); return 1; }
    uint32_t h[4] = { SMP_MAGIC, sizeof(Sample), PI_K, 0 };
    uint64_t count = 0;
    fwrite(h, sizeof h, 1, out);
    fwrite(&count, sizeof count, 1, out);

    pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
    Job jobs[64];
    pthread_t th[64];
    if (nthread > 64) nthread = 64;
    for (int i = 0; i < nthread; i++) {
        memset(&jobs[i], 0, sizeof(Job));
        jobs[i].net = net; jobs[i].games = games; jobs[i].thread = i;
        jobs[i].nthread = nthread; jobs[i].dets = dets; jobs[i].dup = dup;
        jobs[i].seed = seed; jobs[i].out = out; jobs[i].lk = &lk;
        pthread_create(&th[i], NULL, worker, &jobs[i]);
    }
    long fl = 0, co = 0, pl = 0, bc[5] = { 0 };
    for (int i = 0; i < nthread; i++) {
        pthread_join(th[i], NULL);
        fl += jobs[i].flagged; co += jobs[i].corrected; pl += jobs[i].plies;
        for (int b = 0; b < 5; b++) bc[b] += jobs[i].bycls[b];
    }
    long wr = 0;
    for (int i = 0; i < nthread; i++) wr += jobs[i].written;
    count = (uint64_t)wr;
    fseek(out, sizeof h, SEEK_SET);
    fwrite(&count, sizeof count, 1, out);
    fclose(out);
    printf("mine: %d games, %ld plies, %ld flagged (%.1f%%), %ld corrected (%.1f%% of flagged)\n",
           games, pl, fl, 100.0 * fl / (pl ? pl : 1), co, 100.0 * co / (fl ? fl : 1));
    printf("      by class: skip %ld, pile-refusal %ld, stall %ld, burial %ld, hedge %ld\n",
           bc[0], bc[1], bc[2], bc[3], bc[4]);
    printf("      wrote %llu samples (dup %d) to %s\n",
           (unsigned long long)count, dup, outpath);
    return 0;
}
