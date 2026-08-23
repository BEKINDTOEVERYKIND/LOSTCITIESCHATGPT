/* action_advantage_format.h -- fail-closed, information-set-only records for
 * the direct action-advantage veto.
 *
 * A record contains the frozen trunk input, legal policy logits, and a
 * sanitized information-view State, never the referee State.  In particular
 * it contains neither the hidden opponent hand nor future deck order.
 * Proposal labels are signed paired differences from candidate zero after
 * both branches finish the complete remaining three-round match.
 */
#ifndef ACTION_ADVANTAGE_FORMAT_H
#define ACTION_ADVANTAGE_FORMAT_H

#include "../src/features.h"
#include "../src/net.h"
#include <stddef.h>
#include <stdint.h>

#define AA_MAGIC UINT32_C(0x4c434141) /* "LCAA" */
#define AA_VERSION UINT32_C(1)
#define AA_ENDIAN_TAG UINT32_C(0x01020304)
#define AA_HASH_FNV1A64 UINT32_C(1)
#define AA_SOURCE_GENERATED_SELFPLAY UINT32_C(1)
#define AA_MIN_SAFE_PLY UINT32_C(14)
#define AA_MIN_LABEL_WORLDS UINT32_C(256)

enum {
    AA_COLLECTION_COMPLETE = 0,
    AA_COLLECTION_PROPOSAL_CAP = 1
};

enum {
    AA_KIND_ANCHOR = 0,
    AA_KIND_PROPOSAL = 1
};

enum {
    AA_PANEL_PRIMARY = 1u << 0,
    AA_PANEL_FRESH = 1u << 1,
    AA_PANEL_CONTROLLER = 1u << 2
};

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t endian_tag;
    uint32_t header_size;
    uint32_t record_size;
    uint32_t feature_size;
    uint32_t hash_kind;
    uint32_t source_kind;
    uint64_t record_count;
    uint64_t anchor_count;
    uint64_t proposal_count;
    uint64_t generator_seed;
    uint64_t record_chain_hash;
    uint64_t champion_net_hash;
    uint64_t maintained_actor_spec_hash;
    uint64_t maintained_root_net_hash;
    uint64_t maintained_continuation_net_hash;
    uint64_t maintained_controller_net_hash;
    uint64_t reroot_actor_spec_hash;
    uint64_t reroot_root_net_hash;
    uint64_t reroot_continuation_net_hash;
    uint64_t reroot_controller_net_hash;
    uint32_t label_worlds;
    uint32_t ply_lo;
    uint32_t match_rounds;
    uint32_t label_threads;
    uint32_t scoring_symmetries;
    uint32_t source_matches_requested;
    uint32_t source_matches_completed;
    uint32_t proposal_cap;
    uint32_t collection_stop_reason;
} ActionAdvantageHeader;

typedef struct {
    /* Grouping/provenance.  source_match_id is the indivisible train/valid
     * split key.  All hashes use the algorithm declared by the header. */
    uint64_t source_match_id;
    uint64_t source_state_id;
    uint64_t information_view_hash;
    uint64_t hidden_world_set_hash;
    uint64_t future_deal_set_hash;
    uint64_t branch_rng_domain_hash;

    uint32_t kind;
    uint32_t panel_mask;
    uint32_t label_worlds;
    uint32_t actor_primary_worlds;
    uint32_t actor_fresh_worlds;
    uint16_t round;
    uint16_t ply;
    uint16_t root_player;
    uint16_t nlegal;
    uint16_t baseline;
    uint16_t proposal;
    uint16_t reserved16[2];

    /* Signed proposal-minus-baseline counterfactual statistics. */
    float margin_mean;
    float margin_se;
    float match_mean;
    float match_se;
    float hybrid_mean;
    float hybrid_se;

    /* Evidence that the maintained actor really produced a two-panel
     * nonbaseline proposal.  These are diagnostic only, never the label. */
    float actor_primary_delta;
    float actor_primary_se;
    float actor_fresh_delta;
    float actor_fresh_se;

    /* Sanitized actor view: own hand and public state only; opponent hand is
     * reduced to known face-up cards and the complete deck array is zero.
     * Keeping this (never the referee State) permits the trainer's heldout
     * threshold grid to use the exact runtime suit orbit. */
    State information_view;
    /* Frozen-trunk input plus the champion policy distribution used by the
     * full-legal-action anchor loss. */
    Features features;
    uint16_t legal[MAX_MOVES];
    float champion_logits[MAX_MOVES];
} ActionAdvantageRecord;

uint64_t aa_hash_init(void);
uint64_t aa_hash_extend(uint64_t h, const void *data, size_t size);
uint64_t aa_hash_bytes(const void *data, size_t size);
uint64_t aa_record_chain_hash(const ActionAdvantageRecord *records,
                              uint64_t count);
uint64_t aa_group_key(uint64_t source_match_id, uint64_t split_seed);
int aa_group_is_validation(uint64_t source_match_id, uint64_t split_seed,
                           unsigned validation_permille);
/* Exact byte check for every frozen trunk/value/belief parameter.  Policy
 * heads (additive and full-move residual) are intentionally excluded. */
int aa_nonpolicy_equal(const Net *a, const Net *b);

void aa_header_init(ActionAdvantageHeader *header);
int aa_validate_header(const ActionAdvantageHeader *header,
                       char *error, size_t error_size);
int aa_validate_record(const ActionAdvantageHeader *header,
                       const ActionAdvantageRecord *record,
                       char *error, size_t error_size);
int aa_write_file(const char *path, const ActionAdvantageHeader *header,
                  const ActionAdvantageRecord *records, int force,
                  char *error, size_t error_size);
ActionAdvantageRecord *aa_read_file(const char *path,
                                    ActionAdvantageHeader *header,
                                    char *error, size_t error_size);

#endif
