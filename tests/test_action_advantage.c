#define _POSIX_C_SOURCE 200809L
#define main train_advantage_veto_cli_main
#include "../tools/train_advantage_veto.c"
#undef main

#include <assert.h>
#include <stdio.h>

static ActionAdvantageHeader test_header(void)
{
    ActionAdvantageHeader h;
    aa_header_init(&h);
    h.champion_net_hash = 1;
    h.maintained_actor_spec_hash = 2;
    h.maintained_root_net_hash = 3;
    h.maintained_continuation_net_hash = 4;
    h.maintained_controller_net_hash = 5;
    h.reroot_actor_spec_hash = 6;
    h.reroot_root_net_hash = 7;
    h.reroot_continuation_net_hash = 8;
    h.reroot_controller_net_hash = 9;
    return h;
}

static ActionAdvantageRecord proposal_record(const Net *champion)
{
    ActionAdvantageRecord r;
    memset(&r, 0, sizeof(r));
    r.source_match_id = 101;
    r.source_state_id = 202;
    r.hidden_world_set_hash = 404;
    r.future_deal_set_hash = 505;
    r.branch_rng_domain_hash = 606;
    r.kind = AA_KIND_PROPOSAL;
    r.panel_mask = AA_PANEL_PRIMARY | AA_PANEL_FRESH;
    r.label_worlds = 512;
    r.actor_primary_worlds = 512;
    r.actor_fresh_worlds = 512;
    Rng rng;
    rng_seed(&rng, 4242);
    State complete;
    lc_deal(&complete, &rng);
    for (int ply = 0; ply < 14; ply++) {
        Move legal[MAX_MOVES];
        int n = lc_moves(&complete, legal);
        assert(n > 0);
        /* The first enumerated draw is from deck, guaranteeing progress. */
        lc_apply(&complete, legal[0]);
    }
    assert(!complete.over && complete.nply == 14);
    agent_information_view(&complete, complete.turn, &r.information_view);
    r.information_view_hash = aa_hash_bytes(
        &r.information_view, sizeof(r.information_view));
    r.round = complete.round;
    r.ply = complete.nply;
    r.root_player = complete.turn;
    feat_extract(&r.information_view, r.root_player, &r.features);
    Move legal[MAX_MOVES];
    int nlegal = lc_moves(&r.information_view, legal);
    assert(nlegal >= 2);
    r.nlegal = (uint16_t)nlegal;
    for (int i = 0; i < nlegal; i++) r.legal[i] = MOVE_PACK(legal[i]);
    r.baseline = r.legal[0];
    r.proposal = r.legal[nlegal - 1];
    NetAct act;
    net_trunk(champion, &r.features, &act);
    net_policy_act(champion, &act, r.legal, r.nlegal,
                   r.champion_logits);
    r.margin_mean = -3.0f;
    r.margin_se = 1.0f;
    r.match_mean = -0.05f;
    r.match_se = 0.02f;
    r.hybrid_mean = -2.65f;
    r.hybrid_se = 1.2f;
    r.actor_primary_delta = 1.0f;
    r.actor_primary_se = 0.2f;
    r.actor_fresh_delta = 0.8f;
    r.actor_fresh_se = 0.25f;
    return r;
}

static int policy_wager_symmetric(const Net *n)
{
    for (int suit = 0; suit < NSUIT; suit++) {
        int card = suit * NRANK;
        for (int discard = 0; discard < 2; discard++) {
            int p0 = card * 2 + discard;
            for (int copy = 1; copy < 3; copy++) {
                int p = (card + copy) * 2 + discard;
                if (memcmp(n->wplay[p0], n->wplay[p], sizeof(n->wplay[p0])) ||
                    memcmp(&n->bplay[p0], &n->bplay[p], sizeof(n->bplay[p0])))
                    return 0;
                for (int draw = 0; draw < NET_NDRAW; draw++) {
                    int c0 = p0 * NET_NDRAW + draw;
                    int c = p * NET_NDRAW + draw;
                    if (memcmp(n->wcomb[c0], n->wcomb[c],
                               sizeof(n->wcomb[c0])) ||
                        memcmp(&n->bcomb[c0], &n->bcomb[c],
                               sizeof(n->bcomb[c0])))
                        return 0;
                }
            }
        }
    }
    return 1;
}

static void test_format_and_grouping(void)
{
    assert(aa_hash_init() == UINT64_C(14695981039346656037));
    assert(aa_hash_bytes("a", 1) == UINT64_C(0xaf63dc4c8601ec8c));
    assert(deployed_threshold(0.1) == (double)(float)0.1);
    assert(deployed_threshold(0.1) > 0.1);
    Net *n = (Net *)malloc(sizeof(*n));
    assert(n);
    net_init(n, 77);
    ActionAdvantageHeader h = test_header();
    ActionAdvantageRecord r = proposal_record(n);
    assert(stored_champion_logits_match(n, &r));
    r.champion_logits[0] = nextafterf(r.champion_logits[0], INFINITY);
    assert(!stored_champion_logits_match(n, &r));
    r = proposal_record(n);
    h.record_count = h.proposal_count = 1;
    h.record_chain_hash = aa_record_chain_hash(&r, 1);
    char error[160];
    assert(aa_validate_header(&h, error, sizeof(error)));
    assert(aa_validate_record(&h, &r, error, sizeof(error)));

    /* Signed negative labels are first-class records, not discarded. */
    assert(r.hybrid_mean < 0.0f);
    ActionAdvantageRecord bad = r;
    bad.panel_mask = AA_PANEL_PRIMARY;
    assert(!aa_validate_record(&h, &bad, error, sizeof(error)));

    char path[128];
    snprintf(path, sizeof(path), "/tmp/lc-action-advantage-%ld.bin",
             (long)getpid());
    unlink(path);
    assert(aa_write_file(path, &h, &r, 0, error, sizeof(error)));
    ActionAdvantageHeader loaded_h;
    ActionAdvantageRecord *loaded =
        aa_read_file(path, &loaded_h, error, sizeof(error));
    assert(loaded);
    assert(memcmp(&r, loaded, sizeof(r)) == 0);
    free(loaded);
    unlink(path);

    int a = aa_group_is_validation(1234, 9, 200);
    int bsplit = aa_group_is_validation(1234, 9, 200);
    assert(a == bsplit && (a == 0 || a == 1));
    assert(aa_group_is_validation(1, 2, 1001) == -1);
    free(n);
}

static void test_signed_head_only_update(void)
{
    Config c;
    memset(&c, 0, sizeof(c));
    c.anchor_kl = 1.0f;
    c.pair_scale = 1.0f;
    c.label_scale = 12.5f;
    c.huber_delta = 1.0f;
    c.max_pair_weight = 4.0f;
    c.lr = 1e-3f;

    Net *champion = (Net *)malloc(sizeof(*champion));
    Net *ranker = (Net *)malloc(sizeof(*ranker));
    Net *g = (Net *)calloc(1, sizeof(*g));
    HeadAdam adam = {
        (Net *)calloc(1, sizeof(Net)), (Net *)calloc(1, sizeof(Net)), 0
    };
    assert(champion && ranker && g && adam.m && adam.v);
    net_init(champion, 88);
    memcpy(ranker, champion, sizeof(*ranker));
    ActionAdvantageRecord r = proposal_record(champion);
    NetAct act;
    float before[MAX_MOVES], after[MAX_MOVES];
    net_trunk(ranker, &r.features, &act);
    net_policy_act(ranker, &act, r.legal, r.nlegal, before);
    double weight;
    assert(accumulate_record(&c, ranker, &r, g, &weight));
    assert(weight > 0.0);
    head_adam_step(ranker, g, &adam, c.lr, (float)(1.0 / weight), 0.0f);
    assert(aa_nonpolicy_equal(champion, ranker));
    assert(policy_wager_symmetric(ranker));
    net_trunk(ranker, &r.features, &act);
    net_policy_act(ranker, &act, r.legal, r.nlegal, after);
    /* A negative signed label must lower proposal-vs-baseline residual even
     * if the champion's absolute pair gap is already extreme. */
    int bi = move_index(&r, r.baseline);
    int pi = move_index(&r, r.proposal);
    assert((after[pi] - after[bi]) < (before[pi] - before[bi]));

    ranker->w1[0][0] += 1.0f;
    assert(!aa_nonpolicy_equal(champion, ranker));
    free(champion); free(ranker); free(g); free(adam.m); free(adam.v);
}

int main(void)
{
    test_format_and_grouping();
    test_signed_head_only_update();
    puts("action-advantage tests passed");
    return 0;
}
