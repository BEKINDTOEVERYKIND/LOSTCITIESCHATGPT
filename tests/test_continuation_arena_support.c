/* White-box contracts for continuation-arena role scheduling. */
#define main continuation_arena_cli_main_for_test
#include "../tools/continuation_arena.c"
#undef main

#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) {                         \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);          \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; \
} } while (0)

static void test_independent_ordered_product(void)
{
    uint8_t group[120][NSUIT];
    CHECK(suit_permutations(ROOT_SYMMETRIES, group) == ROOT_SYMMETRIES,
          "could not construct arena role group");
    int count[ROOT_SYMMETRIES][ROOT_SYMMETRIES] = { { 0 } };
    for (uint64_t d = 0;
         d < (uint64_t)ROOT_SYMMETRIES * ROOT_SYMMETRIES; d++) {
        int root_player = (int)(d & UINT64_C(1));
        int mapping[2], repeated_mapping[2];
        uint8_t perm[2][NSUIT], repeated[2][NSUIT];
        CHECK(assign_player_mappings(
                  UINT64_C(20260825), d, root_player,
                  ROLE_MAPPING_INDEPENDENT, group, mapping, perm),
              "independent mapping rejected trajectory %llu",
              (unsigned long long)d);
        CHECK(assign_player_mappings(
                  UINT64_C(20260825), d, root_player,
                  ROLE_MAPPING_INDEPENDENT, group,
                  repeated_mapping, repeated) &&
              memcmp(mapping, repeated_mapping, sizeof mapping) == 0 &&
              memcmp(perm, repeated, sizeof perm) == 0,
              "independent mapping was not deterministic at %llu",
              (unsigned long long)d);
        int fixed = mapping[root_player];
        int other = mapping[root_player ^ 1];
        CHECK(fixed >= 0 && fixed < ROOT_SYMMETRIES &&
              other >= 0 && other < ROOT_SYMMETRIES,
              "independent mapping left role group at %llu",
              (unsigned long long)d);
        if (fixed >= 0 && fixed < ROOT_SYMMETRIES &&
            other >= 0 && other < ROOT_SYMMETRIES)
            count[fixed][other]++;
    }
    for (int fixed = 0; fixed < ROOT_SYMMETRIES; fixed++)
        for (int other = 0; other < ROOT_SYMMETRIES; other++)
            CHECK(count[fixed][other] == 1,
                  "ordered arena role pair %d/%d occurred %d times",
                  fixed, other, count[fixed][other]);
}

static void test_legacy_and_shared_contracts(void)
{
    uint8_t group[120][NSUIT];
    CHECK(suit_permutations(ROOT_SYMMETRIES, group) == ROOT_SYMMETRIES,
          "could not construct arena role group");
    int legacy_seen[ROOT_SYMMETRIES] = { 0 };
    for (uint64_t d = 0; d < 10; d++) {
        int mapping[2];
        uint8_t perm[2][NSUIT];
        CHECK(assign_player_mappings(
                  UINT64_C(20260825), d, 0, ROLE_MAPPING_LEGACY,
                  group, mapping, perm),
              "legacy mapping rejected trajectory %llu",
              (unsigned long long)d);
        legacy_seen[mapping[0]]++;
        legacy_seen[mapping[1]]++;
    }
    for (int k = 0; k < ROOT_SYMMETRIES; k++)
        CHECK(legacy_seen[k] == 1,
              "legacy ten-pair block covered mapping %d %d times",
              k, legacy_seen[k]);

    for (uint64_t d = 0; d < ROOT_SYMMETRIES; d++) {
        int mapping[2];
        uint8_t perm[2][NSUIT];
        CHECK(assign_player_mappings(
                  UINT64_C(20260825), d, (int)(d & UINT64_C(1)),
                  ROLE_MAPPING_SHARED, group, mapping, perm),
              "shared mapping rejected trajectory %llu",
              (unsigned long long)d);
        CHECK(mapping[0] == mapping[1] &&
              memcmp(perm[0], perm[1], NSUIT) == 0,
              "shared mapping split player roles at %llu",
              (unsigned long long)d);
    }
}

int main(void)
{
    test_independent_ordered_product();
    test_legacy_and_shared_contracts();
    if (failures == 0) {
        puts("continuation arena support tests passed");
        return 0;
    }
    fprintf(stderr, "%d continuation arena support failures\n", failures);
    return 1;
}
