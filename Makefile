CC      ?= gcc
CFLAGS  ?= -O3 -march=native -ffast-math -funroll-loops -Wall -Wextra -std=c11
LDFLAGS ?= -lm -pthread

BIN     := bin
SRC     := src
DATA    := data

HDRS    := $(wildcard $(SRC)/*.h)
CORE    := $(SRC)/lc.c $(SRC)/features.c $(SRC)/net.c $(SRC)/heuristic.c \
           $(SRC)/planner.c $(SRC)/search.c $(SRC)/rollout.c \
           $(SRC)/late_resolver.c $(SRC)/match_value.c $(SRC)/policy_cost.c \
           $(SRC)/agent.c \
           $(SRC)/match.c $(SRC)/spec.c

all: $(BIN)/test_engine $(BIN)/test_runtime $(BIN)/test_role_coherence \
	$(BIN)/test_late_resolver $(BIN)/test_rl_support \
	$(BIN)/test_action_ranker $(BIN)/test_action_advantage \
	$(BIN)/test_continuation_arena_support $(BIN)/test_match_value \
	$(BIN)/test_policy_cost \
	$(BIN)/arena $(BIN)/train \
	$(BIN)/bench $(BIN)/probe $(BIN)/rl $(BIN)/ladder $(BIN)/play \
	$(BIN)/showgame $(BIN)/dumpfeat $(BIN)/analyze $(BIN)/searchcmp \
	$(BIN)/qpair $(BIN)/commented_ply_eval $(BIN)/flagged_ply_probe \
	$(BIN)/mine $(BIN)/robust_distill $(BIN)/action_advantage \
	$(BIN)/train_advantage_veto $(BIN)/symmetrize \
	$(BIN)/net_average \
	$(BIN)/history_belief $(BIN)/planprobe $(BIN)/planarena \
	$(BIN)/belief_eval $(BIN)/test_belief_eval $(BIN)/continuation_arena \
	$(BIN)/build_match_value $(BIN)/build_policy_cost \
	$(DATA)/champion.bin

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_engine: tests/test_engine.c $(SRC)/lc.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/test_runtime: tests/test_runtime.c tools/train_target.h $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/test_action_ranker: tests/test_action_ranker.c $(CORE) $(HDRS) \
	$(DATA)/champion.bin $(DATA)/c8.bin $(DATA)/best.bin | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

# White-box signed-loss test includes the head-only trainer so it can prove
# both residual direction and byte-exact freezing without widening its CLI ABI.
$(BIN)/test_action_advantage: tests/test_action_advantage.c \
	tools/train_advantage_veto.c tools/action_advantage_format.c \
	$(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ \
		$(filter-out tools/train_advantage_veto.c,$(filter %.c,$^)) $(LDFLAGS)

# White-box role-coherence tests include rollout.c directly so they can trace
# its fixed-permutation policy calls without adding hooks to the gameplay ABI.
ROLE_TEST_CORE := $(filter-out $(SRC)/rollout.c,$(CORE))
$(BIN)/test_role_coherence: tests/test_role_coherence.c $(ROLE_TEST_CORE) \
	$(HDRS) $(DATA)/champion.bin | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/test_late_resolver: tests/test_late_resolver.c $(CORE) $(HDRS) \
	$(DATA)/champion.bin | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

# White-box continuation-PPO tests include rl.c so they can verify the exact
# conditional support and per-logit gradients without widening the trainer ABI.
$(BIN)/test_rl_support: tests/test_rl_support.c tools/rl.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ \
		$(filter-out tools/rl.c,$(filter %.c,$^)) $(LDFLAGS)

# White-box continuation-screen tests verify the exact legacy/shared/product
# role schedules without spending games on a 400-tail integration fixture.
$(BIN)/test_continuation_arena_support: \
	tests/test_continuation_arena_support.c tools/continuation_arena.c \
	$(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ \
		$(filter-out tools/continuation_arena.c,$(filter %.c,$^)) $(LDFLAGS)

$(BIN)/arena: tools/arena.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/continuation_arena: tools/continuation_arena.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/train: tools/train.c tools/train_target.h $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/bench: tools/bench.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/build_match_value: tools/build_match_value.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/build_policy_cost: tools/build_policy_cost.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/test_match_value: tests/test_match_value.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off \
		-o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/test_policy_cost: tests/test_policy_cost.c $(CORE) $(HDRS) \
	$(DATA)/champion.bin | $(BIN)
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off \
		-o $@ $(filter %.c,$^) $(LDFLAGS)

test: $(BIN)/test_engine $(BIN)/test_runtime $(BIN)/test_role_coherence \
	$(BIN)/test_late_resolver $(BIN)/test_rl_support \
	$(BIN)/test_action_ranker $(BIN)/test_action_advantage \
	$(BIN)/test_continuation_arena_support $(BIN)/test_match_value \
	$(BIN)/test_policy_cost $(BIN)/test_belief_eval \
	$(BIN)/belief_eval $(BIN)/rl $(BIN)/arena $(BIN)/qpair \
	$(BIN)/commented_ply_eval \
	$(BIN)/continuation_arena \
	$(BIN)/net_average $(BIN)/flagged_ply_probe $(BIN)/history_belief \
	$(BIN)/build_match_value $(BIN)/build_policy_cost \
	$(BIN)/train $(BIN)/showgame $(BIN)/analyze \
	$(BIN)/play $(BIN)/probe \
	$(DATA)/champion.bin
	./$(BIN)/test_engine
	./$(BIN)/test_runtime
	./$(BIN)/test_role_coherence
	./$(BIN)/test_late_resolver
	./$(BIN)/test_rl_support
	./$(BIN)/test_action_ranker
	./$(BIN)/test_action_advantage
	./$(BIN)/test_continuation_arena_support
	./$(BIN)/test_match_value
	./$(BIN)/test_policy_cost
	./$(BIN)/test_belief_eval
	python3 -m unittest tests/test_rl_population.py
	python3 -m unittest tests/test_belief_eval.py
	python3 -m unittest tests/test_make_showcase.py
	python3 -m unittest tests/test_merge_arena.py
	python3 -m unittest tests/test_actor_panel.py
	python3 -m unittest tests/test_controller_veto_v3.py
	python3 -m unittest tests/test_world800_campaign.py
	python3 -m unittest tests/test_archive_world800.py
	python3 -m unittest tests/test_action_advantage_campaign.py
	python3 -m unittest tests/test_action_ranker_veto.py
	python3 -m unittest tests/test_commented_ply_audit.py
	python3 -m unittest tests/test_commented_ply_execution.py
	python3 -m unittest tests/test_continuation_arena.py
	python3 -m unittest tests/test_select_continuation_v2.py
	python3 -m unittest tests/test_net_average.py
	python3 -m unittest tests/test_flagged_ply_audit.py
	python3 -m unittest tests/test_match_value.py
	python3 -m unittest tests/test_match_value_campaign.py
	python3 -m unittest tests/test_policy_cost.py
	python3 -m unittest tests/test_policy_cost_dataset.py
	python3 -m unittest tests/test_policy_cost_calibration.py
	python3 -m unittest tests/test_policy_cost_selection.py
	python3 -m unittest tests/test_policy_cost_campaign.py
	python3 -m unittest tests/test_policy_cost_exact17.py
	python3 -m unittest tests/test_action_core_campaign.py

audit-test: $(BIN)/qpair $(DATA)/champion.bin
	python3 tools/audit_regression.py

history-belief-test: $(BIN)/history_belief $(DATA)/champion.bin
	python3 -m unittest tests/test_history_belief.py

controller-veto-v3-test:
	python3 -m unittest tests/test_controller_veto_v3.py

world800-test: $(DATA)/champion.bin
	python3 -m unittest tests/test_world800_campaign.py \
		tests/test_archive_world800.py

flagged-ply-test: $(BIN)/flagged_ply_probe $(BIN)/history_belief \
	$(DATA)/champion.bin
	python3 -m unittest tests/test_flagged_ply_audit.py

match-value-test: $(BIN)/test_match_value $(BIN)/build_match_value \
	$(BIN)/arena $(BIN)/train $(BIN)/rl $(BIN)/showgame $(BIN)/analyze \
	$(BIN)/play $(BIN)/probe $(BIN)/flagged_ply_probe $(DATA)/champion.bin
	./$(BIN)/test_match_value
	python3 -m unittest tests/test_match_value.py \
		tests/test_match_value_campaign.py

action-core-campaign-test:
	python3 -m unittest tests/test_action_core_campaign.py

belief-eval-test: $(BIN)/belief_eval $(BIN)/test_belief_eval \
	$(DATA)/champion.bin
	./$(BIN)/test_belief_eval
	python3 -m unittest tests/test_belief_eval.py

clean:
	rm -rf $(BIN)
	rm -f $(DATA)/champion.bin

.PHONY: all test audit-test history-belief-test controller-veto-v3-test \
	world800-test flagged-ply-test match-value-test action-core-campaign-test \
	belief-eval-test clean

$(BIN)/probe: tools/probe.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/rl: tools/rl.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/ladder: tools/ladder.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/play: tools/play.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/showgame: tools/showgame.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/dumpfeat: tools/dumpfeat.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/analyze: tools/analyze.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/searchcmp: tools/searchcmp.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/qpair: tools/qpair.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/commented_ply_eval: tools/commented_ply_eval.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/flagged_ply_probe: tools/flagged_ply_probe.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/mine: tools/mine.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/robust_distill: tools/robust_distill.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/action_advantage: tools/action_advantage.c \
	tools/action_advantage_format.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/train_advantage_veto: tools/train_advantage_veto.c \
	tools/action_advantage_format.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/history_belief: tools/history_belief.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/planprobe: tools/planprobe.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/planarena: tools/planarena.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/belief_eval: tools/belief_eval.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/test_belief_eval: tests/test_belief_eval.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/symmetrize: tools/symmetrize.c $(SRC)/net.c $(SRC)/features.c \
	$(SRC)/lc.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/net_average: tools/net_average.c $(SRC)/net.c $(SRC)/features.c \
	$(SRC)/lc.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off \
		-o $@ $(filter %.c,$^) $(LDFLAGS)

$(DATA)/champion.bin: $(DATA)/c8.bin $(BIN)/symmetrize
	./$(BIN)/symmetrize $< $@
