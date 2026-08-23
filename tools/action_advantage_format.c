#define _POSIX_C_SOURCE 200809L
#include "action_advantage_format.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void set_error(char *dst, size_t n, const char *message)
{
    if (dst && n) snprintf(dst, n, "%s", message);
}

uint64_t aa_hash_init(void)
{
    return UINT64_C(14695981039346656037);
}

uint64_t aa_hash_extend(uint64_t h, const void *data, size_t size)
{
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < size; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

uint64_t aa_hash_bytes(const void *data, size_t size)
{
    return aa_hash_extend(aa_hash_init(), data, size);
}

uint64_t aa_record_chain_hash(const ActionAdvantageRecord *records,
                              uint64_t count)
{
    uint64_t h = aa_hash_init();
    for (uint64_t i = 0; i < count; i++)
        h = aa_hash_extend(h, &records[i], sizeof(records[i]));
    return h;
}

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

uint64_t aa_group_key(uint64_t source_match_id, uint64_t split_seed)
{
    return mix64(source_match_id ^ split_seed ^
                 UINT64_C(0x41445647524f5550));
}

int aa_group_is_validation(uint64_t source_match_id, uint64_t split_seed,
                           unsigned validation_permille)
{
    if (validation_permille > 1000u) return -1;
    return (unsigned)(aa_group_key(source_match_id, split_seed) % 1000u)
           < validation_permille;
}

int aa_nonpolicy_equal(const Net *a, const Net *b)
{
    return a && b &&
        memcmp(a->w1, b->w1, sizeof(a->w1)) == 0 &&
        memcmp(a->b1, b->b1, sizeof(a->b1)) == 0 &&
        memcmp(a->w2, b->w2, sizeof(a->w2)) == 0 &&
        memcmp(a->b2, b->b2, sizeof(a->b2)) == 0 &&
        memcmp(a->w3, b->w3, sizeof(a->w3)) == 0 &&
        memcmp(&a->b3, &b->b3, sizeof(a->b3)) == 0 &&
        memcmp(a->wbel, b->wbel, sizeof(a->wbel)) == 0 &&
        memcmp(a->bbel, b->bbel, sizeof(a->bbel)) == 0;
}

void aa_header_init(ActionAdvantageHeader *h)
{
    memset(h, 0, sizeof(*h));
    h->magic = AA_MAGIC;
    h->version = AA_VERSION;
    h->endian_tag = AA_ENDIAN_TAG;
    h->header_size = (uint32_t)sizeof(*h);
    h->record_size = (uint32_t)sizeof(ActionAdvantageRecord);
    h->feature_size = (uint32_t)sizeof(Features);
    h->hash_kind = AA_HASH_FNV1A64;
    h->source_kind = AA_SOURCE_GENERATED_SELFPLAY;
    h->label_worlds = 512;
    h->ply_lo = AA_MIN_SAFE_PLY;
    h->match_rounds = MATCH_ROUNDS;
    h->label_threads = 1;
    h->scoring_symmetries = 20;
    h->source_matches_requested = 1;
    h->source_matches_completed = 1;
    h->collection_stop_reason = AA_COLLECTION_COMPLETE;
}

static int finite_float(float x)
{
    return lc_float_isfinite(x);
}

int aa_validate_header(const ActionAdvantageHeader *h,
                       char *error, size_t error_size)
{
    if (!h || h->magic != AA_MAGIC || h->version != AA_VERSION ||
        h->endian_tag != AA_ENDIAN_TAG || h->header_size != sizeof(*h) ||
        h->record_size != sizeof(ActionAdvantageRecord) ||
        h->feature_size != sizeof(Features)) {
        set_error(error, error_size, "incompatible action-advantage format");
        return 0;
    }
    if (h->hash_kind != AA_HASH_FNV1A64 ||
        h->source_kind != AA_SOURCE_GENERATED_SELFPLAY) {
        set_error(error, error_size,
                  "records are not generated-selfplay FNV1a provenance");
        return 0;
    }
    int valid_symmetries = h->scoring_symmetries == 1 ||
        h->scoring_symmetries == 5 || h->scoring_symmetries == 10 ||
        h->scoring_symmetries == 20 || h->scoring_symmetries == 120;
    if (h->ply_lo < AA_MIN_SAFE_PLY ||
        h->label_worlds < AA_MIN_LABEL_WORLDS ||
        h->match_rounds != MATCH_ROUNDS || !h->label_threads ||
        !valid_symmetries ||
        !h->source_matches_requested ||
        h->source_matches_completed > h->source_matches_requested) {
        set_error(error, error_size,
                  "unsafe phase, world count, or match horizon");
        return 0;
    }
    if (!h->champion_net_hash || !h->maintained_actor_spec_hash ||
        !h->maintained_root_net_hash ||
        !h->maintained_continuation_net_hash ||
        !h->maintained_controller_net_hash ||
        !h->reroot_actor_spec_hash || !h->reroot_root_net_hash ||
        !h->reroot_continuation_net_hash ||
        !h->reroot_controller_net_hash) {
        set_error(error, error_size, "missing model/spec provenance hash");
        return 0;
    }
    if (h->anchor_count + h->proposal_count != h->record_count) {
        set_error(error, error_size, "header record counts disagree");
        return 0;
    }
    if ((h->collection_stop_reason == AA_COLLECTION_COMPLETE &&
         h->source_matches_completed != h->source_matches_requested) ||
        (h->collection_stop_reason == AA_COLLECTION_PROPOSAL_CAP &&
         (!h->proposal_cap || h->proposal_count != h->proposal_cap)) ||
        h->collection_stop_reason > AA_COLLECTION_PROPOSAL_CAP) {
        set_error(error, error_size, "invalid collection completion metadata");
        return 0;
    }
    if (h->record_count > UINT32_MAX ||
        h->record_count > SIZE_MAX / sizeof(ActionAdvantageRecord)) {
        set_error(error, error_size, "record count is too large");
        return 0;
    }
    return 1;
}

int aa_validate_record(const ActionAdvantageHeader *h,
                       const ActionAdvantageRecord *r,
                       char *error, size_t error_size)
{
    if (!h || !r || (r->kind != AA_KIND_ANCHOR &&
                     r->kind != AA_KIND_PROPOSAL)) {
        set_error(error, error_size, "invalid record kind");
        return 0;
    }
    if (r->round >= MATCH_ROUNDS || r->root_player > 1 ||
        r->ply < h->ply_lo || r->nlegal == 0 || r->nlegal > MAX_MOVES ||
        r->baseline >= MOVE_NPACK || r->proposal >= MOVE_NPACK ||
        !r->source_match_id || !r->source_state_id ||
        !r->information_view_hash) {
        set_error(error, error_size, "invalid record identity or dimensions");
        return 0;
    }
    if (r->features.nidx < 0 || r->features.nidx > FEAT_MAX_ACTIVE) {
        set_error(error, error_size, "invalid sparse feature count");
        return 0;
    }
    for (int i = 0; i < r->features.nidx; i++) {
        if (r->features.idx[i] >= FEAT_BIN ||
            (i && r->features.idx[i] <= r->features.idx[i - 1])) {
            set_error(error, error_size, "invalid sparse feature index");
            return 0;
        }
    }
    for (int i = 0; i < FEAT_DENSE; i++)
        if (!finite_float(r->features.dense[i])) {
            set_error(error, error_size, "non-finite dense feature");
            return 0;
        }
    const State *view = &r->information_view;
    const uint64_t valid_mask = (UINT64_C(1) << NCARD) - 1;
    if (view->over || view->turn != r->root_player ||
        view->round != r->round || view->nply != r->ply ||
        view->deck_pos != 0 || view->deck_left == 0 ||
        view->hand[r->root_player ^ 1] != view->known[r->root_player ^ 1] ||
        (view->hand[0] | view->hand[1] | view->played[0] |
         view->played[1] | view->discarded | view->known[0] |
         view->known[1]) & ~valid_mask ||
        __builtin_popcountll(view->hand[r->root_player]) !=
            view->hand_n[r->root_player] ||
        __builtin_popcountll(view->hand[r->root_player ^ 1]) >
            view->hand_n[r->root_player ^ 1] ||
        (view->known[r->root_player] & ~view->hand[r->root_player])) {
        set_error(error, error_size, "information view contains hidden data");
        return 0;
    }
    for (int i = 0; i < NCARD; i++)
        if (view->deck[i] != 0) {
            set_error(error, error_size, "information view retains deck order");
            return 0;
        }
    Features expected;
    memset(&expected, 0, sizeof(expected));
    feat_extract(view, r->root_player, &expected);
    if (memcmp(&expected, &r->features, sizeof(expected)) != 0 ||
        aa_hash_bytes(view, sizeof(*view)) != r->information_view_hash) {
        set_error(error, error_size, "information view/feature hash mismatch");
        return 0;
    }
    Move engine_legal[MAX_MOVES];
    int engine_nlegal = lc_moves(view, engine_legal);
    if (engine_nlegal != r->nlegal) {
        set_error(error, error_size, "record legal support is incomplete");
        return 0;
    }
    int have_baseline = 0, have_proposal = 0;
    for (unsigned i = 0; i < r->nlegal; i++) {
        if (r->legal[i] != MOVE_PACK(engine_legal[i]) ||
            r->legal[i] >= MOVE_NPACK ||
            !finite_float(r->champion_logits[i])) {
            set_error(error, error_size, "invalid legal action or anchor logit");
            return 0;
        }
        for (unsigned j = 0; j < i; j++)
            if (r->legal[j] == r->legal[i]) {
                set_error(error, error_size, "duplicate legal action");
                return 0;
            }
        have_baseline |= r->legal[i] == r->baseline;
        have_proposal |= r->legal[i] == r->proposal;
    }
    if (!have_baseline || !have_proposal) {
        set_error(error, error_size, "baseline/proposal is not legal");
        return 0;
    }
    const float stats[] = {
        r->margin_mean, r->margin_se, r->match_mean, r->match_se,
        r->hybrid_mean, r->hybrid_se, r->actor_primary_delta,
        r->actor_primary_se, r->actor_fresh_delta, r->actor_fresh_se
    };
    for (unsigned i = 0; i < sizeof(stats) / sizeof(stats[0]); i++)
        if (!finite_float(stats[i])) {
            set_error(error, error_size, "non-finite signed statistic");
            return 0;
        }
    if (r->margin_se < 0 || r->match_se < 0 || r->hybrid_se < 0 ||
        r->actor_primary_se < 0 || r->actor_fresh_se < 0) {
        set_error(error, error_size, "negative standard error");
        return 0;
    }
    if (r->kind == AA_KIND_PROPOSAL) {
        if (r->proposal == r->baseline || r->label_worlds != h->label_worlds ||
            !r->hidden_world_set_hash || !r->future_deal_set_hash ||
            !r->branch_rng_domain_hash ||
            (r->panel_mask & (AA_PANEL_PRIMARY | AA_PANEL_FRESH)) !=
             (AA_PANEL_PRIMARY | AA_PANEL_FRESH) ||
            !r->actor_primary_worlds || !r->actor_fresh_worlds) {
            set_error(error, error_size,
                      "proposal lacks paired labels or two-panel provenance");
            return 0;
        }
    } else {
        if (r->proposal != r->baseline || r->label_worlds != 0 ||
            r->panel_mask != 0 || r->hidden_world_set_hash ||
            r->future_deal_set_hash || r->branch_rng_domain_hash) {
            set_error(error, error_size, "anchor contains proposal label data");
            return 0;
        }
    }
    return 1;
}

int aa_write_file(const char *path, const ActionAdvantageHeader *h,
                  const ActionAdvantageRecord *records, int force,
                  char *error, size_t error_size)
{
    if (!path || !aa_validate_header(h, error, error_size)) return 0;
    for (uint64_t i = 0; i < h->record_count; i++)
        if (!aa_validate_record(h, &records[i], error, error_size)) return 0;
    if (aa_record_chain_hash(records, h->record_count) !=
        h->record_chain_hash) {
        set_error(error, error_size, "record-chain hash mismatch before write");
        return 0;
    }
    struct stat st;
    if (!force && stat(path, &st) == 0) {
        set_error(error, error_size, "output exists (use --force)");
        return 0;
    }
    char *tmp = (char *)malloc(strlen(path) + 64);
    if (!tmp) {
        set_error(error, error_size, "out of memory");
        return 0;
    }
    snprintf(tmp, strlen(path) + 64, "%s.tmp.%ld", path, (long)getpid());
    FILE *f = fopen(tmp, "wb");
    int ok = f && fwrite(h, sizeof(*h), 1, f) == 1 &&
        (!h->record_count ||
         fwrite(records, sizeof(*records), (size_t)h->record_count, f) ==
             (size_t)h->record_count) &&
        fflush(f) == 0;
    if (f && fclose(f) != 0) ok = 0;
    if (ok && rename(tmp, path) != 0) ok = 0;
    if (!ok) {
        unlink(tmp);
        set_error(error, error_size, "cannot atomically write record file");
    }
    free(tmp);
    return ok;
}

ActionAdvantageRecord *aa_read_file(const char *path,
                                    ActionAdvantageHeader *h,
                                    char *error, size_t error_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        set_error(error, error_size, strerror(errno));
        return NULL;
    }
    if (fread(h, sizeof(*h), 1, f) != 1 ||
        !aa_validate_header(h, error, error_size)) {
        fclose(f);
        return NULL;
    }
    ActionAdvantageRecord *r = (ActionAdvantageRecord *)calloc(
        (size_t)(h->record_count ? h->record_count : 1), sizeof(*r));
    if (!r) {
        fclose(f);
        set_error(error, error_size, "out of memory");
        return NULL;
    }
    if ((h->record_count &&
         fread(r, sizeof(*r), (size_t)h->record_count, f) !=
             (size_t)h->record_count) ||
        fgetc(f) != EOF || ferror(f)) {
        free(r);
        fclose(f);
        set_error(error, error_size, "short or trailing record payload");
        return NULL;
    }
    fclose(f);
    uint64_t anchors = 0, proposals = 0;
    for (uint64_t i = 0; i < h->record_count; i++) {
        if (!aa_validate_record(h, &r[i], error, error_size)) {
            free(r);
            return NULL;
        }
        anchors += r[i].kind == AA_KIND_ANCHOR;
        proposals += r[i].kind == AA_KIND_PROPOSAL;
    }
    if (anchors != h->anchor_count || proposals != h->proposal_count ||
        aa_record_chain_hash(r, h->record_count) != h->record_chain_hash) {
        free(r);
        set_error(error, error_size, "payload count or provenance hash mismatch");
        return NULL;
    }
    return r;
}
