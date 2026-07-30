CC      ?= gcc
CFLAGS  ?= -O3 -march=native -ffast-math -funroll-loops -Wall -Wextra -std=c11
LDFLAGS ?= -lm -pthread

BIN     := bin
SRC     := src
DATA    := data

HDRS    := $(wildcard $(SRC)/*.h)
CORE    := $(SRC)/lc.c $(SRC)/features.c $(SRC)/net.c $(SRC)/heuristic.c \
           $(SRC)/search.c $(SRC)/rollout.c $(SRC)/agent.c $(SRC)/match.c $(SRC)/spec.c

all: $(BIN)/test_engine $(BIN)/test_runtime $(BIN)/arena $(BIN)/train \
	$(BIN)/bench $(BIN)/probe $(BIN)/rl $(BIN)/ladder $(BIN)/play \
	$(BIN)/showgame $(BIN)/dumpfeat $(BIN)/analyze $(BIN)/searchcmp \
	$(BIN)/qpair $(BIN)/mine $(BIN)/robust_distill $(BIN)/symmetrize \
	$(DATA)/champion.bin

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_engine: tests/test_engine.c $(SRC)/lc.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/test_runtime: tests/test_runtime.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/arena: tools/arena.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/train: tools/train.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/bench: tools/bench.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

test: $(BIN)/test_engine $(BIN)/test_runtime $(DATA)/champion.bin
	./$(BIN)/test_engine
	./$(BIN)/test_runtime

audit-test: $(BIN)/qpair $(DATA)/champion.bin
	python3 tools/audit_regression.py

clean:
	rm -rf $(BIN)
	rm -f $(DATA)/champion.bin

.PHONY: all test audit-test clean

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

$(BIN)/mine: tools/mine.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/robust_distill: tools/robust_distill.c $(CORE) $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(BIN)/symmetrize: tools/symmetrize.c $(SRC)/net.c $(SRC)/features.c \
	$(SRC)/lc.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS)

$(DATA)/champion.bin: $(DATA)/c8.bin $(BIN)/symmetrize
	./$(BIN)/symmetrize $< $@
