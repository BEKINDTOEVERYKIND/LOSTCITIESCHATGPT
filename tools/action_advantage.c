#define _POSIX_C_SOURCE 200809L
/* Generate information-set-only, signed full-match action-advantage labels.
 *
 * Inputs are generated self-play matches only: there is deliberately no
 * --state/--probe option, so human-commented positions cannot silently become
 * training examples.  Each label clones one uniformly sanitized hidden world
 * and one set of future deals into candidate-zero and proposal branches.  All
 * later decisions are freshly re-rooted through the explicitly supplied
 * actor, with branch-neutral deterministic RNG domains.
 */
#include "action_advantage_format.h"
#include "../src/agent.h"
#include "../src/search.h"
#include "../src/spec.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *out_path;
    const char *champion_path;
    const char *actor_spec;
    const char *reroot_spec;
    uint64_t seed;
    int matches;
    int worlds;
    int ply_lo;
    int anchor_stride;
    int label_threads;
    int max_proposals;
    int force;
} Config;

typedef struct {
    double mean, m2;
    uint32_t n;
} OnlineStat;

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint64_t domain_seed(uint64_t seed, uint64_t match_id,
                            uint64_t state_id, uint64_t world,
                            uint64_t tag)
{
    return mix64(seed ^ mix64(match_id) ^ mix64(state_id) ^
                 mix64(world + UINT64_C(0x100000001b3)) ^ tag);
}

static void stat_add(OnlineStat *s, double x)
{
    s->n++;
    double d = x - s->mean;
    s->mean += d / (double)s->n;
    s->m2 += d * (x - s->mean);
}

static float stat_se(const OnlineStat *s)
{
    if (s->n < 2) return 0.0f;
    double variance = s->m2 / (double)(s->n - 1);
    if (variance < 0.0) variance = 0.0;
    return (float)sqrt(variance / (double)s->n);
}

static int parse_i(const char *s, int lo, int hi, int *out)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return 0;
    *out = (int)v;
    return 1;
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)v;
    return 1;
}

static void usage(FILE *f, const char *argv0)
{
    fprintf(f,
        "usage: %s --out RECORDS --champion NET --actor SPEC "
        "--reroot-actor SPEC [options]\n\n"
        "  --matches N       generated maintained-actor self-play matches "
        "(default 4)\n"
        "  --worlds N        paired full-remaining-match labels, min 256 "
        "(default 512)\n"
        "  --ply-lo N        first eligible round ply, min 14 (default 14)\n"
        "  --anchor-stride N keep a full-policy anchor every N eligible plies "
        "(default 8)\n"
        "  --label-threads N parallel exact re-root worlds (default 1)\n"
        "  --max-proposals N stop after labeling N observed proposals "
        "(default unlimited)\n"
        "  --seed N          deterministic campaign seed (default 20260823)\n"
        "  --force           replace an existing output atomically\n\n"
        "Both actor specs are mandatory.  The first must be a two-panel "
        "rollout actor; the second is re-rooted at every continuation node. "
        "Only generated matches are accepted; saved/probe states are not a "
        "supported input.\n", argv0);
}

static int parse_args(int argc, char **argv, Config *c)
{
    memset(c, 0, sizeof(*c));
    c->seed = UINT64_C(20260823);
    c->matches = 4;
    c->worlds = 512;
    c->ply_lo = 14;
    c->anchor_stride = 8;
    c->label_threads = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--out") && ++i < argc) c->out_path = argv[i];
        else if (!strcmp(a, "--champion") && ++i < argc)
            c->champion_path = argv[i];
        else if (!strcmp(a, "--actor") && ++i < argc) c->actor_spec = argv[i];
        else if (!strcmp(a, "--reroot-actor") && ++i < argc)
            c->reroot_spec = argv[i];
        else if (!strcmp(a, "--matches") && ++i < argc &&
                 parse_i(argv[i], 1, 100000, &c->matches)) {}
        else if (!strcmp(a, "--worlds") && ++i < argc &&
                 parse_i(argv[i], AA_MIN_LABEL_WORLDS, 1000000, &c->worlds)) {}
        else if (!strcmp(a, "--ply-lo") && ++i < argc &&
                 parse_i(argv[i], AA_MIN_SAFE_PLY, LC_MAX_PLIES,
                         &c->ply_lo)) {}
        else if (!strcmp(a, "--anchor-stride") && ++i < argc &&
                 parse_i(argv[i], 1, LC_MAX_PLIES, &c->anchor_stride)) {}
        else if (!strcmp(a, "--label-threads") && ++i < argc &&
                 parse_i(argv[i], 1, 128, &c->label_threads)) {}
        else if (!strcmp(a, "--max-proposals") && ++i < argc &&
                 parse_i(argv[i], 1, 1000000, &c->max_proposals)) {}
        else if (!strcmp(a, "--seed") && ++i < argc &&
                 parse_u64(argv[i], &c->seed)) {}
        else if (!strcmp(a, "--force")) c->force = 1;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "invalid or incomplete option: %s\n", a);
            return 0;
        }
    }
    if (!c->out_path || !c->champion_path || !c->actor_spec ||
        !c->reroot_spec) {
        fprintf(stderr, "--out, --champion, --actor, and --reroot-actor "
                        "are all required\n");
        return 0;
    }
    return 1;
}

static uint64_t net_or_absent_hash(const Net *n)
{
    static const char absent[] = "ABSENT";
    return n ? aa_hash_bytes(n, sizeof(*n))
             : aa_hash_bytes(absent, sizeof(absent));
}

static void shuffle_deck(uint8_t deck[NCARD], Rng *rng)
{
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    for (int i = NCARD - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = deck[i]; deck[i] = deck[j]; deck[j] = t;
    }
}

static Move reroot_move(const Agent *actor, const State *complete,
                        uint64_t rng_domain)
{
    State view;
    agent_information_view(complete, complete->turn, &view);
    Rng rng;
    rng_seed(&rng, rng_domain);
    return agent_move(actor, &view, &rng);
}

/* Complete the current round and every later deal.  future[r] is defined for
 * r > root.round and is byte-identical in the two branches. */
static int finish_remaining_match(State *root, int perspective,
                                  const uint8_t future[MATCH_ROUNDS][NCARD],
                                  const Agent *reroot, uint64_t rng_domain,
                                  int *margin_out, int *match_out,
                                  double *hybrid_out)
{
    int cumulative[2] = { root->cum[0], root->cum[1] };
    for (int round = root->round; round < MATCH_ROUNDS; round++) {
        State st;
        if (round == root->round) {
            st = *root;
        } else {
            lc_deal_from_deck(&st, future[round]);
            st.round = (uint8_t)round;
            st.turn = (uint8_t)(round & 1);
            st.cum[0] = (int16_t)cumulative[0];
            st.cum[1] = (int16_t)cumulative[1];
        }
        while (!st.over) {
            uint64_t decision_domain = mix64(
                rng_domain ^ ((uint64_t)(unsigned)round << 48) ^
                ((uint64_t)st.nply << 8) ^ (uint64_t)st.turn);
            Move m = reroot_move(reroot, &st, decision_domain);
            lc_apply(&st, m);
        }
        /* A nonempty deck means LC_MAX_PLIES, not the rules, ended the round. */
        if (st.deck_left > 0) return 0;
        cumulative[0] += lc_score(&st, 0);
        cumulative[1] += lc_score(&st, 1);
    }
    int margin = cumulative[perspective] - cumulative[perspective ^ 1];
    int outcome = (margin > 0) - (margin < 0);
    *margin_out = margin;
    *match_out = outcome;
    *hybrid_out = 0.05 * (double)margin + 50.0 * (double)outcome;
    return 1;
}

static int fill_policy_anchor(const Net *champion, const State *complete,
                              ActionAdvantageRecord *r)
{
    State view;
    agent_information_view(complete, complete->turn, &view);
    r->information_view = view;
    feat_extract(&view, view.turn, &r->features);
    r->information_view_hash = aa_hash_bytes(&view, sizeof(view));
    Move legal[MAX_MOVES];
    int n = lc_moves(&view, legal);
    if (n <= 0 || n > MAX_MOVES) return 0;
    r->nlegal = (uint16_t)n;
    for (int i = 0; i < n; i++) r->legal[i] = MOVE_PACK(legal[i]);
    NetAct act;
    net_trunk(champion, &r->features, &act);
    net_policy_act(champion, &act, r->legal, n, r->champion_logits);
    return 1;
}

typedef struct {
    int ok;
    int margin_delta;
    int match_delta;
    double hybrid_delta;
    uint64_t world_hash;
    uint64_t future_hash;
    uint64_t branch_domain;
} LabelOutcome;

typedef struct {
    const Config *config;
    const State *complete;
    const Agent *reroot;
    const ActionAdvantageRecord *record;
    LabelOutcome *outcome;
    int thread;
    int nthread;
} LabelJob;

static int label_one_world(const LabelJob *j, int w, LabelOutcome *out)
{
    const Config *c = j->config;
    const State *complete = j->complete;
    const ActionAdvantageRecord *r = j->record;
    const int p = complete->turn;
    State view;
    agent_information_view(complete, p, &view);
    uint64_t world_seed = domain_seed(c->seed, r->source_match_id,
                                      r->source_state_id, (uint64_t)w,
                                      UINT64_C(0x574f524c44534554));
    Rng world_rng;
    rng_seed(&world_rng, world_seed);
    State sampled;
    /* Uniform information-set sampling is intentional: labels never use the
     * referee's opponent hand or future deck. */
    determinize(&view, p, &world_rng, &sampled);
    out->world_hash = aa_hash_bytes(&sampled, sizeof(sampled));

    uint8_t future[MATCH_ROUNDS][NCARD];
    memset(future, 0, sizeof(future));
    uint64_t future_hash = aa_hash_init();
    for (int round = complete->round + 1; round < MATCH_ROUNDS; round++) {
        uint64_t deal_seed = domain_seed(
            c->seed, r->source_match_id, r->source_state_id,
            ((uint64_t)(unsigned)w << 8) | (unsigned)round,
            UINT64_C(0x465554555245444c));
        Rng deal_rng;
        rng_seed(&deal_rng, deal_seed);
        shuffle_deck(future[round], &deal_rng);
        future_hash = aa_hash_extend(future_hash, future[round], NCARD);
    }
    out->future_hash = future_hash;
    out->branch_domain = domain_seed(
        c->seed, r->source_match_id, r->source_state_id, (uint64_t)w,
        UINT64_C(0x4252414e4348524e));

    Move baseline = {
        MOVE_CARD(r->baseline), MOVE_DISC(r->baseline), MOVE_DRAW(r->baseline)
    };
    Move proposal = {
        MOVE_CARD(r->proposal), MOVE_DISC(r->proposal), MOVE_DRAW(r->proposal)
    };
    /* The same sampled world, future deals, and branch-neutral decision
     * domain are cloned into both counterfactuals. */
    State b = sampled, q = sampled;
    lc_apply(&b, baseline);
    lc_apply(&q, proposal);
    int bm, qm, bo, qo;
    double bh, qh;
    if (!finish_remaining_match(&b, p, future, j->reroot,
                                out->branch_domain, &bm, &bo, &bh) ||
        !finish_remaining_match(&q, p, future, j->reroot,
                                out->branch_domain, &qm, &qo, &qh))
        return 0;
    out->margin_delta = qm - bm;
    out->match_delta = qo - bo;
    out->hybrid_delta = qh - bh;
    out->ok = 1;
    return 1;
}

static void *label_worker(void *arg)
{
    LabelJob *j = (LabelJob *)arg;
    for (int w = j->thread; w < j->config->worlds; w += j->nthread)
        (void)label_one_world(j, w, &j->outcome[w]);
    return NULL;
}

static int label_proposal(const Config *c, const State *complete,
                          const Agent *reroot, ActionAdvantageRecord *r)
{
    LabelOutcome *out = (LabelOutcome *)calloc((size_t)c->worlds,
                                                sizeof(*out));
    int nthread = c->label_threads < c->worlds
        ? c->label_threads : c->worlds;
    pthread_t *thread = (pthread_t *)calloc((size_t)nthread, sizeof(*thread));
    LabelJob *job = (LabelJob *)calloc((size_t)nthread, sizeof(*job));
    if (!out || !thread || !job) {
        free(out); free(thread); free(job);
        return 0;
    }
    int created = 0;
    for (int t = 0; t < nthread; t++) {
        job[t].config = c;
        job[t].complete = complete;
        job[t].reroot = reroot;
        job[t].record = r;
        job[t].outcome = out;
        job[t].thread = t;
        job[t].nthread = nthread;
        if (pthread_create(&thread[t], NULL, label_worker, &job[t]) != 0)
            break;
        created++;
    }
    int ok = created == nthread;
    for (int t = 0; t < created; t++)
        if (pthread_join(thread[t], NULL) != 0) ok = 0;

    OnlineStat margin = {0}, match = {0}, hybrid = {0};
    uint64_t world_hash = aa_hash_init();
    uint64_t future_hash = aa_hash_init();
    uint64_t rng_hash = aa_hash_init();
    for (int w = 0; w < c->worlds && ok; w++) {
        if (!out[w].ok) { ok = 0; break; }
        world_hash = aa_hash_extend(world_hash, &out[w].world_hash,
                                    sizeof(out[w].world_hash));
        future_hash = aa_hash_extend(future_hash, &out[w].future_hash,
                                     sizeof(out[w].future_hash));
        rng_hash = aa_hash_extend(rng_hash, &out[w].branch_domain,
                                  sizeof(out[w].branch_domain));
        stat_add(&margin, out[w].margin_delta);
        stat_add(&match, out[w].match_delta);
        stat_add(&hybrid, out[w].hybrid_delta);
    }
    free(out); free(thread); free(job);
    if (!ok) {
        fprintf(stderr, "counterfactual worker failed or hit LC_MAX_PLIES at "
                        "source state %016llx; refusing a partial label\n",
                (unsigned long long)r->source_state_id);
        return 0;
    }
    r->hidden_world_set_hash = world_hash;
    r->future_deal_set_hash = future_hash;
    r->branch_rng_domain_hash = rng_hash;
    r->label_worlds = (uint32_t)c->worlds;
    r->margin_mean = (float)margin.mean;
    r->margin_se = stat_se(&margin);
    r->match_mean = (float)match.mean;
    r->match_se = stat_se(&match);
    r->hybrid_mean = (float)hybrid.mean;
    r->hybrid_se = stat_se(&hybrid);
    return 1;
}

static int append_record(ActionAdvantageRecord **records, uint64_t *count,
                         uint64_t *capacity,
                         const ActionAdvantageRecord *record)
{
    if (*count == *capacity) {
        uint64_t next = *capacity ? *capacity * 2 : 64;
        if (next < *capacity || next > SIZE_MAX / sizeof(**records)) return 0;
        void *p = realloc(*records, (size_t)next * sizeof(**records));
        if (!p) return 0;
        *records = (ActionAdvantageRecord *)p;
        *capacity = next;
    }
    (*records)[(*count)++] = *record;
    return 1;
}

int main(int argc, char **argv)
{
    Config c;
    if (!parse_args(argc, argv, &c)) {
        usage(stderr, argv[0]);
        return 2;
    }
    Net *champion = (Net *)malloc(sizeof(*champion));
    if (!champion || net_load(champion, c.champion_path) != 0) {
        fprintf(stderr, "cannot load champion %s\n", c.champion_path);
        free(champion);
        return 1;
    }
    Agent maintained, reroot;
    spec_parse(c.actor_spec, &maintained);
    spec_parse(c.reroot_spec, &reroot);
    if (maintained.kind != AG_ROLLOUT || !maintained.net ||
        maintained.policy_prefix_mode < 2 || maintained.ply_lo < 14 ||
        !reroot.net) {
        fprintf(stderr, "maintained actor must be a ply>=14 two-panel rollout "
                        "actor and the explicit re-rooting actor needs a net\n");
        spec_release(&maintained);
        spec_release(&reroot);
        free(champion);
        return 1;
    }
    if (memcmp(champion, maintained.net, sizeof(*champion)) != 0) {
        fprintf(stderr, "--champion is not byte-identical to the maintained "
                        "actor root net\n");
        spec_release(&maintained);
        spec_release(&reroot);
        free(champion);
        return 1;
    }

    ActionAdvantageHeader header;
    aa_header_init(&header);
    header.generator_seed = c.seed;
    header.label_worlds = (uint32_t)c.worlds;
    header.ply_lo = (uint32_t)c.ply_lo;
    header.label_threads = (uint32_t)c.label_threads;
    header.scoring_symmetries = (uint32_t)maintained.symmetries;
    header.source_matches_requested = (uint32_t)c.matches;
    header.source_matches_completed = 0;
    header.proposal_cap = (uint32_t)c.max_proposals;
    header.champion_net_hash = aa_hash_bytes(champion, sizeof(*champion));
    header.maintained_actor_spec_hash =
        aa_hash_bytes(c.actor_spec, strlen(c.actor_spec));
    header.maintained_root_net_hash = net_or_absent_hash(maintained.net);
    header.maintained_continuation_net_hash =
        net_or_absent_hash(maintained.continuation_net);
    header.maintained_controller_net_hash =
        net_or_absent_hash(maintained.veto_continuation_net);
    header.reroot_actor_spec_hash =
        aa_hash_bytes(c.reroot_spec, strlen(c.reroot_spec));
    header.reroot_root_net_hash = net_or_absent_hash(reroot.net);
    header.reroot_continuation_net_hash =
        net_or_absent_hash(reroot.continuation_net);
    header.reroot_controller_net_hash =
        net_or_absent_hash(reroot.veto_continuation_net);

    ActionAdvantageRecord *records = NULL;
    uint64_t count = 0, capacity = 0, anchors = 0, proposals = 0;
    int failed = 0, cap_reached = 0, matches_completed = 0;
    for (int game = 0; game < c.matches && !failed && !cap_reached; game++) {
        uint64_t match_id = mix64(c.seed ^ (uint64_t)(unsigned)game ^
                                  UINT64_C(0x534f555243454d54));
        if (!match_id) match_id = 1;
        int cumulative[2] = {0, 0};
        for (int round = 0; round < MATCH_ROUNDS && !failed && !cap_reached;
             round++) {
            Rng deal_rng;
            rng_seed(&deal_rng, mix64(match_id ^
                     ((uint64_t)(unsigned)round << 32) ^
                     UINT64_C(0x534f55524345444c)));
            State st;
            lc_deal(&st, &deal_rng);
            st.round = (uint8_t)round;
            st.turn = (uint8_t)(round & 1);
            st.cum[0] = (int16_t)cumulative[0];
            st.cum[1] = (int16_t)cumulative[1];
            while (!st.over && !cap_reached) {
                State view;
                agent_information_view(&st, st.turn, &view);
                uint64_t action_seed = mix64(
                    match_id ^ ((uint64_t)(unsigned)round << 48) ^
                    ((uint64_t)st.nply << 8) ^ st.turn ^
                    UINT64_C(0x534f555243454143));
                Rng action_rng;
                rng_seed(&action_rng, action_seed);
                SearchStats ss;
                memset(&ss, 0, sizeof(ss));
                Move selected;
                int eligible = st.nply >= c.ply_lo;
                if (eligible)
                    selected = rollout_move(&maintained, &view, &action_rng,
                                            NULL, &ss);
                else
                    selected = agent_move(&maintained, &view, &action_rng);

                int pi = eligible ? ss.prefix_proposed : 0;
                int is_proposal = pi > 0 && pi < ss.n &&
                    ss.prefix_confirm_worlds > 0 && ss.prefix_confirmed &&
                    ss.unfinished_cap_leaves == 0;
                int keep_anchor = eligible && !is_proposal &&
                    st.nply % c.anchor_stride == 0;
                if (is_proposal || keep_anchor) {
                    ActionAdvantageRecord r;
                    memset(&r, 0, sizeof(r));
                    r.source_match_id = match_id;
                    r.round = (uint16_t)round;
                    r.ply = st.nply;
                    r.root_player = st.turn;
                    r.kind = is_proposal ? AA_KIND_PROPOSAL : AA_KIND_ANCHOR;
                    if (!fill_policy_anchor(champion, &st, &r)) {
                        fprintf(stderr, "cannot encode source state\n");
                        failed = 1;
                        break;
                    }
                    uint64_t sid_parts[4] = {
                        match_id, (uint64_t)(unsigned)round, st.nply,
                        r.information_view_hash
                    };
                    r.source_state_id = aa_hash_bytes(sid_parts,
                                                      sizeof(sid_parts));
                    if (!r.source_state_id) r.source_state_id = 1;
                    if (ss.n > 0) r.baseline = MOVE_PACK(ss.mv[0]);
                    else {
                        /* An eligible forced state can skip rollout stats. */
                        Move pmv[MAX_MOVES]; float prob[MAX_MOVES];
                        int pn = policy_probs(champion, &view, pmv, prob, NULL);
                        int best = 0;
                        for (int i = 1; i < pn; i++)
                            if (prob[i] > prob[best]) best = i;
                        r.baseline = MOVE_PACK(pmv[best]);
                    }
                    r.proposal = is_proposal ? MOVE_PACK(ss.mv[pi])
                                             : r.baseline;
                    if (is_proposal) {
                        r.panel_mask = AA_PANEL_PRIMARY | AA_PANEL_FRESH;
                        if (ss.prefix_veto_attempted)
                            r.panel_mask |= AA_PANEL_CONTROLLER;
                        r.actor_primary_worlds = (uint32_t)ss.worlds;
                        r.actor_fresh_worlds =
                            (uint32_t)ss.prefix_confirm_worlds;
                        r.actor_primary_delta = (float)ss.delta[pi];
                        r.actor_primary_se = (float)ss.dse[pi];
                        r.actor_fresh_delta = (float)ss.prefix_delta[pi];
                        r.actor_fresh_se = (float)ss.prefix_dse[pi];
                        if (!label_proposal(&c, &st, &reroot, &r)) {
                            failed = 1;
                            break;
                        }
                    }
                    char error[160];
                    if (!aa_validate_record(&header, &r, error,
                                            sizeof(error))) {
                        fprintf(stderr, "generated record rejected: %s\n",
                                error);
                        failed = 1;
                        break;
                    }
                    if (!append_record(&records, &count, &capacity, &r)) {
                        fprintf(stderr, "out of memory storing records\n");
                        failed = 1;
                        break;
                    }
                    proposals += is_proposal;
                    anchors += !is_proposal;
                    if (is_proposal) {
                        printf("proposal match %016llx round %d ply %u: "
                               "hybrid %+.3f +/- %.3f, margin %+.3f, "
                               "match %+.4f\n",
                               (unsigned long long)match_id, round, st.nply,
                               r.hybrid_mean, r.hybrid_se, r.margin_mean,
                               r.match_mean);
                        fflush(stdout);
                    }
                    if (is_proposal && c.max_proposals > 0 &&
                        proposals >= (uint64_t)c.max_proposals)
                        cap_reached = 1;
                }
                if (!cap_reached) lc_apply(&st, selected);
            }
            if (!failed && !cap_reached) {
                if (st.deck_left > 0) {
                    fprintf(stderr, "source match hit LC_MAX_PLIES\n");
                    failed = 1;
                } else {
                    cumulative[0] += lc_score(&st, 0);
                    cumulative[1] += lc_score(&st, 1);
                }
            }
        }
        if (!failed && !cap_reached) matches_completed++;
    }

    int rc = 1;
    if (!failed) {
        header.record_count = count;
        header.anchor_count = anchors;
        header.proposal_count = proposals;
        header.source_matches_completed = (uint32_t)matches_completed;
        header.collection_stop_reason = cap_reached
            ? AA_COLLECTION_PROPOSAL_CAP : AA_COLLECTION_COMPLETE;
        header.record_chain_hash = aa_record_chain_hash(records, count);
        char error[200];
        if (!aa_write_file(c.out_path, &header, records, c.force,
                           error, sizeof(error))) {
            fprintf(stderr, "cannot write %s: %s\n", c.out_path, error);
        } else {
            printf("wrote %llu records (%llu signed proposals, %llu anchors) "
                   "from %d/%d complete generated matches to %s%s\n",
                   (unsigned long long)count,
                   (unsigned long long)proposals,
                   (unsigned long long)anchors, matches_completed, c.matches,
                   c.out_path, cap_reached ? " (explicit proposal cap)" : "");
            printf("record chain %016llx; champion %016llx; reroot spec "
                   "%016llx\n",
                   (unsigned long long)header.record_chain_hash,
                   (unsigned long long)header.champion_net_hash,
                   (unsigned long long)header.reroot_actor_spec_hash);
            rc = 0;
        }
    }
    free(records);
    spec_release(&maintained);
    spec_release(&reroot);
    free(champion);
    return rc;
}
