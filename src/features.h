/* features.h -- information-set encoding for the value network.
 *
 * A feature vector describes the game from the point of view of one player:
 * that player's own hand plus everything public.  Nothing about the
 * opponent's hidden cards leaks in.  The vector is split into a sparse
 * binary part (card planes, encoded as a list of active indices) and a dense
 * part (engineered per-suit and global scalars), which lets the first network
 * layer skip almost all of its multiplies.
 */
#ifndef FEATURES_H
#define FEATURES_H

#include "lc.h"

/* card planes: my hand / my expeditions / their expeditions / discarded /
 * pile tops / cards I know they hold / my cards they know about */
#define FEAT_PLANES 7
#define FEAT_BIN (FEAT_PLANES * NCARD) /* 420 */
#define SUIT_FEATS 24
#define GLOBAL_FEATS 16
#define FEAT_LEGACY_DENSE (NSUIT * SUIT_FEATS + GLOBAL_FEATS) /* 136 */
#define FEAT_LEGACY_DIM (FEAT_BIN + FEAT_LEGACY_DENSE)        /* 556 */

/* The original feature layout represented each discard pile as an unordered
 * set plus its top card and size.  The order below the top is public and can
 * matter after one or more face-up draws, so append (rather than interleave)
 * an exact semantic encoding for every buried depth.  A card is represented
 * by normalized number value plus a wager flag; the three physical wager
 * copies are intentionally identical because their identity is unobservable.
 *
 * Depth 1 (the current top) remains in suit features 13/14.  These appended
 * pairs cover top-relative depths 2..12, preserving all public pile order.
 * Appending keeps feature rows 0..555 identical for legacy model upgrades. */
#define FEAT_PILE_BURIED_DEPTHS (NRANK - 1)
#define FEAT_PILE_CARD_FEATS 2
#define FEAT_PILE_ORDER (NSUIT * FEAT_PILE_BURIED_DEPTHS * FEAT_PILE_CARD_FEATS)

#define FEAT_DENSE (FEAT_LEGACY_DENSE + FEAT_PILE_ORDER) /* 246 */
#define FEAT_DIM (FEAT_BIN + FEAT_DENSE)                 /* 666 */
#define FEAT_MAX_ACTIVE 184

typedef struct {
    int nidx;
    uint16_t idx[FEAT_MAX_ACTIVE];
    float dense[FEAT_DENSE];
} Features;

/* Encode st from player p's point of view. */
void feat_extract(const State *st, int p, Features *f);
/* Legacy-prefix variant for a checkpoint whose appended input rows have been
 * proven to be exact +0.  Only callers holding that per-network proof may use
 * it.  The appended slots are zeroed but their ordered-pile construction is
 * skipped, so the resulting Features remains safe for accidental full reads. */
void feat_extract_legacy(const State *st, int p, Features *f);

#endif
