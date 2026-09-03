/* history_belief_exclusion.h -- fail-closed reviewed-ply firewall.
 *
 * The canonical 17 reviewed positions are represented by SHA-256 digests of
 * their mover information views after collapsing physical wager identities
 * and taking the lexicographically least of all 120 suit relabellings.  The
 * public API deliberately accepts a complete referee state and performs the
 * information-view projection itself, so callers can reject a position
 * before reading its private hand as a truth label.
 */
#ifndef HISTORY_BELIEF_EXCLUSION_H
#define HISTORY_BELIEF_EXCLUSION_H

#include "lc.h"

#include <stdint.h>

#define HISTORY_BELIEF_EXCLUSION_COUNT 17
#define HISTORY_BELIEF_EXCLUSION_DIGEST_BYTES 32
#define HISTORY_BELIEF_EXCLUSION_HEX_BYTES 64

/* This is the immutable canonical 17-orbit text payload used by the locked
 * commented-ply audit and policy-cost v1 through v7 campaigns. */
#define HISTORY_BELIEF_EXACT17_CANONICAL_SHA256 \
    "10034cf8b83aadf24fa0775e4dad2712573e1d84cbf364568ce6136682ac254c"

typedef struct {
    unsigned char orbit[HISTORY_BELIEF_EXCLUSION_COUNT]
                       [HISTORY_BELIEF_EXCLUSION_DIGEST_BYTES];
    unsigned char manifest_sha256[HISTORY_BELIEF_EXCLUSION_DIGEST_BYTES];
    char manifest_sha256_hex[HISTORY_BELIEF_EXCLUSION_HEX_BYTES + 1];
    int count;
} HistoryBeliefExclusions;

/* Load exactly 17 unique lowercase hashes under the canonical schema.  The
 * complete file bytes must match expected_sha256_hex.  Returns 1 on success;
 * on failure, returns 0 and clears out. */
int history_belief_exclusions_load(
    const char *path,
    const char expected_sha256_hex[HISTORY_BELIEF_EXCLUSION_HEX_BYTES + 1],
    HistoryBeliefExclusions *out);

/* Compute the canonical suit-orbit digest from a complete nonterminal mover
 * state.  observer must equal complete->turn.  A firewall-local fieldwise
 * projector reads only public fields and the current observer's information;
 * hidden deck bytes and the opponent private hand are never read. */
int history_belief_exclusion_orbit(
    const State *complete, int observer,
    unsigned char out[HISTORY_BELIEF_EXCLUSION_DIGEST_BYTES]);

int history_belief_exclusions_contains(
    const HistoryBeliefExclusions *set,
    const unsigned char orbit[HISTORY_BELIEF_EXCLUSION_DIGEST_BYTES]);

/* Tri-state helper intended for the very first line of a scoring path:
 *   -1  invalid state, manifest, or arguments (fail the run)
 *    0  position is not reviewed (labels may now be materialized)
 *    1  reviewed-ply suit orbit (skip before reading labels)
 * If orbit_out is non-NULL, it receives the computed digest on 0 or 1. */
int history_belief_exclusions_check(
    const HistoryBeliefExclusions *set, const State *complete, int observer,
    unsigned char orbit_out[HISTORY_BELIEF_EXCLUSION_DIGEST_BYTES]);

#endif /* HISTORY_BELIEF_EXCLUSION_H */
