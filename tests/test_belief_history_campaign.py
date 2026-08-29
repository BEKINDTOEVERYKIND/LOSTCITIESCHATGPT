"""Locked-definition contracts for the accuracy-only history campaign."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import struct
import unittest
from unittest import mock

from tools import belief_history_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_COMMIT = "43dcf5e33b293392fce81d2f478cc109a77b7dd5"
LAUNCH_PARENT = "7b31cf7564b7230b2b965dd002fd70a509003187"
PLAN = ROOT / campaign.PLAN_PATH
EXCLUSIONS = ROOT / campaign.EXCLUSIONS_PATH
EXECUTION = ROOT / campaign.EXECUTION_PATH
WORKFLOW = ROOT / campaign.WORKFLOW_PATH
DEFINITION_WORKFLOW = ROOT / campaign.DEFINITION_WORKFLOW_PATH
DEFINITION_REQUIREMENTS = ROOT / campaign.DEFINITION_REQUIREMENTS_PATH


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BeliefHistoryCampaignTests(unittest.TestCase):
    def test_definition_is_inert_accuracy_only_and_large(self) -> None:
        plan = load(PLAN)
        self.assertEqual(plan["schema"], "lc-belief-history-v1-plan-v1")
        self.assertEqual(plan["artifact_schemas"]["control_run"],
                         "lc-history-belief-control-run-v1")
        self.assertEqual(plan["status"],
                         "definition_complete_inert_execution_addendum_absent")
        if EXECUTION.exists():
            parent = subprocess.check_output(
                ["git", "rev-parse", f"{LAUNCH_COMMIT}^"], cwd=ROOT,
                text=True,
            ).strip()
            changed = subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--name-status", "-r",
                 LAUNCH_COMMIT], cwd=ROOT, text=True,
            ).strip()
            self.assertEqual(parent, LAUNCH_PARENT)
            self.assertEqual(
                changed,
                "A\tdata/experiments/locked_belief_history_v1_execution.json",
            )
        else:
            self.assertFalse(EXECUTION.exists())
        self.assertIn("match playing-strength evaluation", plan["non_goals"])
        self.assertFalse(
            plan["models"]["candidate"]["playing_actor_bytes_changed"])
        self.assertEqual(plan["models"]["candidate"]["base_alpha"], 1.0)
        self.assertEqual(plan["models"]["candidate"]
                         ["base_reconstruction_marginal_tolerance"], 2e-6)
        self.assertEqual(plan["models"]["candidate"]["learning_rate"], 0.01)
        self.assertEqual(plan["models"]["candidate"]["l2"], 1e-7)
        self.assertFalse(
            plan["models"]["base_head_only_control"]
                ["playing_actor_bytes_changed"])
        self.assertFalse(
            plan["models"]["matched_head_only_control"]
                ["playing_actor_bytes_changed"])
        self.assertTrue(
            plan["models"]["matched_head_only_control"]
                ["primary_test_comparator"])
        self.assertEqual(plan["models"]["incumbent_head"]["alpha"], 1.15)
        expected = {
            "TEST": (65536, 16),
        }
        for split, (matches, shards) in expected.items():
            self.assertEqual(plan["data"]["splits"][split]["matches"], matches)
            self.assertEqual(plan["data"]["splits"][split]["shards"], shards)
        train = plan["data"]["splits"]["TRAIN"]
        self.assertEqual(train["history_matches"], 65536)
        self.assertEqual(train["matched_control_additional_matches"], 65536)
        self.assertEqual(train["base_control_matches"], 262144)
        self.assertEqual(train["history_shards"], 1)
        self.assertEqual(train["matched_control_shards"], 1)
        self.assertEqual(train["base_control_shards"], 4)
        self.assertEqual(train["history_root"], train["matched_control_root"])
        self.assertIn("no fabricated fixed states-per-match count",
                      plan["data"]["state_count_contract"])
        throughput = plan["execution_protocol"][
            "pre_efficacy_throughput_guard"]
        self.assertEqual(throughput["history_matches"], 100)
        self.assertEqual(throughput["control_matches"], 100)
        self.assertEqual(throughput["minimum_history_matches_per_second"],
                         3.5)
        self.assertEqual(throughput["minimum_control_matches_per_second"],
                         3.5)
        self.assertEqual(throughput["candidate_job_timeout_minutes"], 360)
        self.assertEqual(
            plan["models"]["base_head_only_control"]["control_batch_states"],
            256)
        self.assertEqual(plan["models"]["base_head_only_control"]
                         ["learning_rate"], 0.0001)
        self.assertEqual(plan["models"]["matched_head_only_control"]
                         ["learning_rate"], 0.00015)
        for model in ("base_head_only_control",
                      "matched_head_only_control"):
            self.assertEqual(plan["models"][model]["l2"], 1e-7)
        for model in ("candidate", "base_head_only_control",
                      "matched_head_only_control"):
            commands = " ".join(
                str(value) for key, value in plan["models"][model].items()
                if key.endswith("command_contract"))
            self.assertIn("--base-alpha 1.0", commands)
            self.assertIn("--exclusions-sha256", commands)
            self.assertNotIn("bin/rl", commands)

    def test_information_boundary_and_probe_firewall_are_explicit(self) -> None:
        plan, exclusions = load(PLAN), load(EXCLUSIONS)
        forbidden = " ".join(plan["data"]["information_boundary"]["forbidden"])
        for term in (
            "opponent hidden cards", "opponent deck-draw identities",
            "future actions", "hidden deck order", "truth labels",
            "source actor identity",
        ):
            self.assertIn(term, forbidden)
        self.assertEqual(exclusions["exact17"]["case_count"], 17)
        self.assertEqual(exclusions["exact17"]["training_use"], "forbidden")
        self.assertEqual(exclusions["exact17"]["selection_use"],
                         "not_applicable_no_select_split")
        self.assertIn("reject any of the 17 bound hashes",
                      exclusions["exact17"]["order"])
        self.assertTrue(exclusions["exact17"]
                        ["natural_state_orbit_rejection_claimed"])
        self.assertEqual(plan["probe_firewall"]["selection_use"],
                         "not_applicable_no_select_split")
        self.assertTrue(plan["probe_firewall"]
                        ["natural_state_orbit_rejection_claimed"])
        for item in exclusions["exact17"]["canonical_bindings"]:
            path = ROOT / item["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(item["sha256"], sha(path))

    def test_fresh_roots_are_disjoint_from_every_policy_cost_plan(self) -> None:
        plan = load(PLAN)
        active = set(plan["seeds"]["active_production_roots"])
        smoke = set(plan["seeds"]["definition_smoke_roots"])
        self.assertEqual(len(active), 4)
        self.assertEqual(len(smoke), 3)
        self.assertFalse(active & smoke)
        self.assertTrue(all(value.startswith("202706")
                            for value in active | smoke))
        prior: set[str] = set()
        for path in sorted((ROOT / "data/experiments").glob(
                "locked_policy_cost_v*_plan.json")):
            prior.update(campaign.seed_values(load(path)))
        self.assertFalse((active | smoke) & prior)
        exclusions = load(EXCLUSIONS)
        development = exclusions["burned_seed_contract"][
            "belief_accuracy_development"]
        self.assertEqual(
            set(development["burned_roots"]),
            set(load(ROOT / development["path"])["seed_disposition"]),
        )
        self.assertEqual(development["sha256"],
                         sha(ROOT / development["path"]))
        self.assertFalse((active | smoke) & set(development["burned_roots"]))
        self.assertEqual(exclusions["burned_seed_contract"]
                         ["reserved_namespace_prefixes"], ["20260829"])
        self.assertTrue(all(root.startswith("20260829") for root in
                            exclusions["burned_seed_contract"]
                            ["known_belief_development_roots"]))
        self.assertFalse(any(value.startswith("20260829")
                             for value in active | smoke))
        self.assertFalse((active | smoke) & set(
            exclusions["burned_seed_contract"]["ad_hoc_roots"]))
        self.assertEqual(
            set(exclusions["burned_seed_contract"]
                ["retired_prelaunch_roots"]),
            {"202706100401"},
        )
        self.assertFalse((active | smoke) & set(
            exclusions["burned_seed_contract"]
            ["retired_prelaunch_roots"]))

    def test_native_structural_test_uses_declared_fresh_smoke_root(self) -> None:
        plan = load(PLAN)
        self.assertEqual(
            plan["seeds"]["definition_smoke_roots"],
            ["202706290101", "202706290102",
             campaign.NATIVE_STRUCTURAL_SMOKE_ROOT],
        )
        source = ROOT / campaign.NATIVE_STRUCTURAL_TEST_PATH
        expected = (
            "rng_seed(&rng, "
            f"UINT64_C({campaign.NATIVE_STRUCTURAL_SMOKE_ROOT}));"
        )
        self.assertEqual(
            [line.strip() for line in source.read_text(encoding="utf-8")
             .splitlines() if "rng_seed(&rng" in line],
            [expected],
        )

        with tempfile.TemporaryDirectory(
                prefix="lc-belief-structural-seed-") as tmp:
            root = Path(tmp)
            for relative in campaign.BOUND_PATHS:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            structural = root / campaign.NATIVE_STRUCTURAL_TEST_PATH
            structural.write_text(
                structural.read_text(encoding="utf-8").replace(
                    campaign.NATIVE_STRUCTURAL_SMOKE_ROOT,
                    "202608290141",
                    1,
                ),
                encoding="utf-8",
            )
            with mock.patch.object(campaign, "_verify_external_bindings"):
                with self.assertRaisesRegex(
                        campaign.DefinitionError,
                        "native structural test seed drifted"):
                    campaign.validate_plan(root)

    def test_metrics_are_paired_clustered_and_test_is_one_look(self) -> None:
        evaluation = load(PLAN)["evaluation"]
        self.assertEqual(evaluation["bootstrap"]["replicates"], 20000)
        self.assertEqual(evaluation["bootstrap"]["unit"], "source_match")
        self.assertEqual(evaluation["bootstrap"]["simultaneous_familywise"], {
            "components": 9,
            "confidence": 0.99,
            "coverage_claim": "nominal_asymptotic",
            "exact_finite_sample_coverage_claimed": False,
            "method": "single_step_max_standardized_error",
            "studentization": "fixed_original_source_match_cluster_se",
            "zero_standard_error_policy": "report null simultaneous LCB, mark inferentially ineligible, and fail the affected replacement bundle",
        })
        self.assertEqual(evaluation["primary_gate"]
                         ["relative_nll_improvement_at_least"], 0.0)
        self.assertEqual(evaluation["history_gate"]
                         ["relative_nll_improvement_at_least"], 0.0)
        for name in ("primary_gate", "history_gate", "brier_gate"):
            self.assertEqual(evaluation[name]
                             ["point_gain_strictly_above"], 0.0)
        self.assertEqual(evaluation["history_gate"]
                         ["min_opponent_actions"], 1)
        self.assertEqual(evaluation["primary_gate"]["confidence"], 0.99)
        self.assertEqual(evaluation["brier_gate"]["confidence"], 0.99)
        self.assertEqual(set(evaluation["stages"]), {"TRAIN", "TEST"})
        self.assertTrue(evaluation["stages"]["TEST"]["one_look"])
        self.assertFalse(evaluation["stages"]["TEST"]
                         ["second_test_or_top_up"])
        self.assertEqual(evaluation["terminal_artifact_selection"], {
            "comparison_bundle": "For each ordered replacement comparison, require the frozen all-state joint-NLL, post-opponent-action joint-NLL, and all-state per-card-Brier point/simultaneous-one-sided-nominal-99%-LCB gates directly. One max-standardized-error critical value controls the nine-component selection family asymptotically; exact finite-sample coverage is not claimed and marginal percentile bounds are report-only. A zero original cluster SE is inferentially ineligible and fails its bundle. Never infer a pair by transitivity.",
            "rule": [
                "retain history only if history passes directly against both matched_head_control and incumbent_head",
                "otherwise retain matched_head_control only if it passes directly against incumbent_head",
                "otherwise retain incumbent_head",
            ],
            "selected_artifact_is_playing_actor": False,
        })

    def test_head_control_verifier_rejects_nonbelief_mutation(self) -> None:
        source = (ROOT / "data/champion.bin").read_bytes()
        _, feature_dim, hidden1, hidden2, nplay, _ = struct.unpack(
            "=6I", source[:24])
        prefix_floats = (
            feature_dim * hidden1 + hidden1 + hidden1 * hidden2 + hidden2
            + hidden2 + 1 + nplay * hidden2 + nplay + 6 * hidden2 + 6
        )
        belief_start = 24 + prefix_floats * 4
        belief_end = belief_start + (60 * hidden2 + 60) * 4
        with tempfile.TemporaryDirectory(prefix="lc-head-proof-") as tmp:
            control = bytearray(source)
            value = struct.unpack_from("=f", control, belief_start + 3 * hidden2 * 4)[0]
            struct.pack_into("=f", control, belief_start + 3 * hidden2 * 4,
                             value + 0.001)
            control_path = Path(tmp) / "control.bin"
            control_path.write_bytes(control)
            proof = campaign.verify_head_control(
                ROOT / "data/champion.bin", control_path)
            self.assertTrue(proof["nonbelief_bytes_identical"])
            self.assertEqual(proof["belief_byte_start"], belief_start)
            self.assertEqual(proof["belief_byte_end"], belief_end)

            invalid = bytearray(control)
            invalid[24] ^= 1
            invalid_path = Path(tmp) / "invalid.bin"
            invalid_path.write_bytes(invalid)
            with self.assertRaisesRegex(campaign.DefinitionError,
                                        "trunk/policy/value"):
                campaign.verify_head_control(
                    ROOT / "data/champion.bin", invalid_path)

    def test_workflow_is_push_only_and_execution_path_only(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("push:", text)
        self.assertNotIn("pull_request:", text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertIn(
            "data/experiments/locked_belief_history_v1_execution.json", text)
        self.assertNotIn("locked_policy_cost_v7_execution.json", text)
        self.assertNotIn("policy-cost-v7.yml", text)
        for fixture in campaign.exact17_fixture_paths(ROOT):
            self.assertNotIn(fixture.as_posix(), text)
        self.assertIn("-fno-fast-math -ffp-contract=off", text)
        self.assertNotIn("Python 3.12.3", text)
        self.assertNotIn("= 13.3.0", text)
        self.assertIn("sys.version_info[:2] == (3, 12)", text)
        self.assertIn("gcc -dumpversion | cut -d. -f1", text)
        self.assertIn("--no-deps --target python-runtime \"$NUMPY_WHEEL\"",
                      text)
        self.assertIn(
            "export PYTHONNOUSERSITE=1 PYTHONPATH=\"$PY_RUNTIME\"", text)
        self.assertIn("actual.is_relative_to(expected)", text)
        self.assertIn("python-dependency-proof.json", text)
        self.assertIn("python_dependency_proof_sha256", text)
        heredocs = list(re.finditer(
            r"(?ms)^( +)python3 - <<'PY'\n(.*?)^\1PY$", text))
        self.assertEqual(len(heredocs), 5)
        for index, match in enumerate(heredocs):
            indent = match.group(1)
            lines = match.group(2).splitlines()
            self.assertTrue(all(
                not line or line.startswith(indent) for line in lines))
            source = "\n".join(
                line[len(indent):] if line else "" for line in lines
            ) + "\n"
            compile(source, f"workflow-heredoc-{index}", "exec")
        names = (
            "base_control_1", "base_control_2", "base_control_3",
            "base_control_4", "history_train",
            "matched_control_train", "test_evaluate",
        )
        sections = {}
        for name in names:
            match = re.search(
                rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_]+:\n|\Z)",
                text,
            )
            self.assertIsNotNone(match, name)
            sections[name] = match.group(1)
        for name in ("base_control_1", "base_control_2", "base_control_3"):
            self.assertNotIn("--control-finalize", sections[name])
            self.assertIn("--lr 0.0001 --l2 0.0000001", sections[name])
        self.assertIn("--control-finalize", sections["base_control_4"])
        self.assertIn("--lr 0.0001 --l2 0.0000001",
                      sections["base_control_4"])
        self.assertIn("--control-batch-states 256",
                      sections["base_control_4"])
        self.assertIn("--control-finalize",
                      sections["matched_control_train"])
        self.assertNotIn("--control-state-in",
                         sections["matched_control_train"])
        for fragment in (
            "train-control --out out/matched-control.bin",
            "--control-state-out out/matched-control.state",
            "--actor-net runtime/data/champion.bin --base-net base/control.bin",
            "--matches 65536 --rounds 3 --seed 202706100101 --match-start 0",
            "--max-ply 300 --symmetries 20 --temperature 0.03 --base-alpha 1.0",
            "--epochs 1 --lr 0.00015 --l2 0.0000001",
            "--control-batch-states 256 --control-finalize",
            "--exclusions runtime/bindings/exact17/exclusions.txt",
            "--exclusions-sha256 \"$EXCLUSIONS_SHA256\"",
        ):
            self.assertIn(fragment, sections["matched_control_train"])
        self.assertNotIn("--control-finalize", sections["history_train"])
        self.assertNotIn("--control-batch-states", sections["history_train"])
        self.assertIn("--lr 0.01 --l2 0.0000001",
                      sections["history_train"])
        self.assertIn("--incumbent-alpha 1.15", sections["test_evaluate"])
        self.assertGreaterEqual(text.count("--incumbent-alpha 1.15"), 3)
        self.assertIn("'incumbent_alpha':1.15", text)
        self.assertIn(
            "'incumbent_net_fingerprint':h['actor_fingerprint']", text)
        self.assertIn("control-chain-manifest.json", text)
        self.assertIn("cp -a runtime complete/runtime", text)
        self.assertIn("cp -a shards complete/test-shards", text)
        self.assertIn("mkdir -p python-runtime complete/raw", text)
        self.assertIn(
            "summary['trained_state_count'] - previous_trained == summary['source_state_count']",
            text)
        self.assertIn(
            "summary['optimizer_steps'] - previous_steps == expected_step_delta",
            text)
        self.assertIn("m['trained_state_count'] == m['source_state_count']",
                      text)
        self.assertIn("h['model_train_states'] == h['source_state_count']",
                      text)
        self.assertIn(
            "h['output_sha256'] == sha('history/history-model.bin')", text)
        self.assertIn("f32(h['training_learning_rate']) == f32(0.01)", text)
        self.assertIn("h['training_l2']", text)
        self.assertIn("f32(m['lr']) == f32(0.00015)", text)
        self.assertIn("f32(m['l2']) == f32(0.0000001)", text)
        for fragment in (
            "m['schema'] == 'lc-history-belief-control-run-v1'",
            "m['mode'] == 'train-control'",
            "m['next_match_start'] == 65536",
            "h['rounds'] == m['rounds'] == 3",
            "h['max_scored_ply'] == m['max_scored_ply'] == 300",
            "h['symmetries'] == m['symmetries'] == 20",
            "f32(h['temperature']) == f32(m['temperature']) == f32(0.03)",
            "m['control_state_source_manifest_scope'] == 'current_invocation'",
            "m['playing_actor_changed'] is False",
            "m['control_changed_only_belief_head'] is True",
            "m['training_augmentation'] == 'one deterministic scheduled member per state from the declared suit group'",
            "summary['input_checkpoint_sha256'] == sha('runtime/data/champion.bin')",
            "summary['input_checkpoint_sha256'] == previous_output_sha256",
            "summary['control_state_checkpoint_sha256'] == summary['output_sha256']",
            "m['input_checkpoint_sha256'] == sha('base/control.bin')",
            "m['control_state_checkpoint_sha256'] == m['output_sha256']",
        ):
            self.assertIn(fragment, text)

    def test_definition_workflow_is_clean_inert_and_complete(self) -> None:
        text = DEFINITION_WORKFLOW.read_text(encoding="utf-8")
        trigger = text.split("\npermissions:\n", 1)[0]
        execution = campaign.EXECUTION_PATH.as_posix()
        self.assertIn("  push:\n", trigger)
        self.assertNotIn("pull_request:", trigger)
        self.assertNotIn("workflow_dispatch", trigger)
        self.assertNotIn(execution, trigger)
        self.assertIn(
            "branches: [agent/correctness-and-policy-upgrade]", trigger)
        self.assertEqual(len(campaign.BOUND_PATHS),
                         len(set(campaign.BOUND_PATHS)))
        expected_paths = {
            relative.as_posix() for relative in campaign.BOUND_PATHS
        } | {
            relative.as_posix()
            for relative in campaign.exact17_fixture_paths(ROOT)
        } | set(campaign.DEFINITION_TRIGGER_GUARD_PATTERNS)
        push = trigger.split("  push:\n", 1)[1]
        paths = re.findall(r"(?m)^      - (.+)$", push)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), expected_paths)
        self.assertEqual(
            text.count(
                "git -C campaign worktree add --detach ../source "
                '"$SOURCE_COMMIT"'
            ),
            2,
        )
        self.assertEqual(
            text.count('test -z "$(git -C source status --porcelain)"'),
            2,
        )
        self.assertNotIn("git -C campaign archive HEAD", text)
        self.assertEqual(text.count('if test -e "$EXECUTION"; then'), 2)
        self.assertGreaterEqual(
            text.count("belief_history_v1_result.json"), 2)
        self.assertGreaterEqual(
            text.count("belief_history_v1_run_33253283912_failure.json"), 2)
        self.assertIn("belief_history_campaign.py validate-plan", text)
        self.assertIn("belief_history_campaign.py prepare-execution", text)
        self.assertIn("belief_history_campaign.py guard-execution", text)
        self.assertIn("rm \"$EXECUTION\"", text)
        self.assertIn("make -j2 all CC=gcc", text)
        self.assertIn("make test CC=gcc", text)
        self.assertIn("find tests -maxdepth 1 -type f -name 'test_*.py'", text)
        self.assertIn("'tests[/.]test_[A-Za-z0-9_]+'", text)
        self.assertIn("in_test { print }' Makefile | grep -Eo", text)
        self.assertIn("/^test:/ { in_test = 1; next }", text)
        self.assertIn("in_test && /^[^[:space:]]/ { exit }", text)
        self.assertIn("mapfile -t ALL_TEST_MODULES", text)
        self.assertIn("mapfile -t MAKE_TEST_MODULES", text)
        self.assertIn("mapfile -t TEST_MODULES < <(comm -23", text)
        self.assertIn("test -z \"$(comm -13", text)
        self.assertIn("${#MAKE_TEST_MODULES[@]} + ${#TEST_MODULES[@]}", text)
        self.assertIn('for module in "${TEST_MODULES[@]}"; do', text)
        self.assertIn('python3 -m unittest "$module"', text)
        self.assertNotIn(
            'python3 -m unittest "${TEST_MODULES[@]}"', text)
        for recovery in range(3, 8):
            self.assertIn(
                f"! -name 'test_policy_cost_v{recovery}_recovery.py'",
                text,
            )
        self.assertIn("make test CC=clang", text)
        self.assertIn("-fsanitize=address,undefined", text)
        self.assertIn("ASAN_OPTIONS: detect_leaks=1:halt_on_error=1", text)
        self.assertIn("UBSAN_OPTIONS: halt_on_error=1:print_stacktrace=1", text)
        self.assertIn("--only-binary=:all: --require-hashes", text)
        self.assertIn("numpy.__version__==\"2.3.5\"", text)
        self.assertIn("yaml.__version__==\"6.0.3\"", text)
        heredocs = list(re.finditer(
            r"(?ms)^( +)python3 - <<'PY'\n(.*?)^\1PY$", text))
        self.assertEqual(len(heredocs), 1)
        for index, match in enumerate(heredocs):
            indent = match.group(1)
            lines = match.group(2).splitlines()
            self.assertTrue(all(
                not line or line.startswith(indent) for line in lines))
            source = "\n".join(
                line[len(indent):] if line else "" for line in lines
            ) + "\n"
            compile(source, f"definition-workflow-heredoc-{index}", "exec")

    def test_bound_transitive_and_prior_plan_inventories_are_exact(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        core = re.search(r"(?ms)^CORE\s*:=\s*(.*?)(?=\n\n)", makefile)
        self.assertIsNotNone(core)
        core_paths = tuple(
            Path("src") / name
            for name in re.findall(
                r"\$\(SRC\)/([A-Za-z0-9_]+\.c)", core.group(1))
        )
        header_paths = tuple(
            path.relative_to(ROOT) for path in sorted((ROOT / "src").glob("*.h"))
        )
        bound_transitive = campaign.HISTORY_BELIEF_TRANSITIVE_PATHS
        current_transitive = core_paths + header_paths
        self.assertTrue(set(bound_transitive) <= set(current_transitive))
        future_headers = set(current_transitive) - set(bound_transitive)
        self.assertTrue(all(re.fullmatch(
            r"src/policy_cost_v(?:[89]|[1-9][0-9]+)\.h", str(path)
        ) for path in future_headers))
        current_plans = tuple(path.relative_to(ROOT) for path in sorted(
            (ROOT / "data/experiments").glob("locked_policy_cost_v*_plan.json")
        ))
        self.assertEqual(
            tuple(path for path in current_plans if int(re.search(
                r"_v(\d+)_plan", str(path)
            ).group(1)) < 8), campaign.PRIOR_POLICY_COST_PLAN_PATHS,
        )
        for paths in (
                campaign.INTEGRATION_PATHS,
                campaign.HISTORY_BELIEF_TRANSITIVE_PATHS,
                campaign.PRIOR_POLICY_COST_PLAN_PATHS):
            self.assertTrue(set(paths) <= set(campaign.BOUND_PATHS))

        exact17 = load(
            ROOT / "data/experiments/policy_cost_v7_exact17_exclusions.json")
        self.assertEqual(
            tuple(Path(case["state_path"]) for case in exact17["cases"]),
            campaign.exact17_fixture_paths(ROOT),
        )

    def test_exclusion_projection_has_a_closed_hidden_field_allowlist(self) -> None:
        source = (ROOT / "src/history_belief_exclusion.c").read_text(
            encoding="utf-8")
        projection = source.split(
            "static int project_information_view", 1)[1].split(
                "/* This validation duplicates", 1)[0]
        for forbidden in (
            "agent_information_view",
            "complete->deck[",
            "complete->deck_pos",
            "complete->hand[opponent]",
            "*view = *complete",
            "memcpy(view, complete",
        ):
            self.assertNotIn(forbidden, projection)
        for required in (
            "projected.deck_left = complete->deck_left",
            "projected.hand[observer] = complete->hand[observer]",
            "projected.hand[opponent] = complete->known[opponent]",
            "projected.discarded = complete->discarded",
            "projected.turn = complete->turn",
            "projected.round = complete->round",
            "*view = projected",
        ):
            self.assertIn(required, projection)

    def test_validator_rejects_exact17_fixture_or_trigger_drift(self) -> None:
        with tempfile.TemporaryDirectory(
                prefix="lc-belief-fixture-trigger-") as tmp:
            root = Path(tmp)
            for relative in campaign.BOUND_PATHS:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            for relative in campaign.exact17_fixture_paths(ROOT):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            exclusions = load(root / campaign.EXCLUSIONS_PATH)
            campaign._verify_external_bindings(root, exclusions)

            fixture = root / campaign.exact17_fixture_paths(root)[0]
            original_fixture = fixture.read_bytes()
            fixture.write_bytes(original_fixture + b"\n")
            with self.assertRaisesRegex(
                    campaign.DefinitionError,
                    "exact17 fixture binding mismatch"):
                campaign._verify_external_bindings(root, exclusions)
            fixture.write_bytes(original_fixture)

            workflow = root / campaign.DEFINITION_WORKFLOW_PATH
            original_workflow = workflow.read_text(encoding="utf-8")
            workflow.write_text(
                original_workflow.replace(
                    "on:\n  push:\n",
                    "on:\n  pull_request:\n  push:\n",
                    1,
                ),
                encoding="utf-8",
            )
            with mock.patch.object(campaign, "_verify_external_bindings"):
                with self.assertRaisesRegex(
                        campaign.DefinitionError,
                        "definition workflow is not exact-branch push-only"):
                    campaign.validate_plan(root)

    def test_execution_template_requires_exact_definition_success(self) -> None:
        value = load(ROOT / campaign.TEMPLATE_PATH)
        self.assertEqual(value["status"], "inert_template_not_a_launch_binding")
        self.assertIsNone(value["results"])
        self.assertIn(
            "push-only .github/workflows/belief-history-v1-definition.yml "
            "concluded success on attempt 1 of the push event whose head SHA "
            "and tree are the exact inert definition parent",
            value["instructions"][1],
        )
        self.assertIn("do not substitute a pull-request check",
                      value["instructions"][1])
        self.assertIn("Git history alone cannot attest this Actions result",
                      value["instructions"][1])
        self.assertIn(campaign.DEFINITION_WORKFLOW_PATH,
                      campaign.BOUND_PATHS)
        self.assertIn(campaign.DEFINITION_REQUIREMENTS_PATH,
                      campaign.BOUND_PATHS)
        requirements = DEFINITION_REQUIREMENTS.read_text(encoding="ascii")
        self.assertIn("numpy==2.3.5 --hash=sha256:", requirements)
        self.assertIn("PyYAML==6.0.3 --hash=sha256:", requirements)

    def test_make_test_runs_belief_campaign_and_reducer_once(self) -> None:
        text = (ROOT / "Makefile").read_text(encoding="utf-8")
        test_recipe = text.split("\ntest:", 1)[1].split("\naudit-test:", 1)[0]
        for module in (
            "tests.test_belief_history_campaign",
            "tests.test_belief_history_reduce",
        ):
            self.assertEqual(test_recipe.count(module), 1, module)

    def test_validator_rejects_missing_matched_receipt_assertion(self) -> None:
        required = (
            "m['schema'] == 'lc-history-belief-control-run-v1'",
            "m['next_match_start'] == 65536",
            "h['rounds'] == m['rounds'] == 3",
            "h['max_scored_ply'] == m['max_scored_ply'] == 300",
            "h['symmetries'] == m['symmetries'] == 20",
            "f32(h['temperature']) == f32(m['temperature']) == f32(0.03)",
            "m['control_state_source_manifest_scope'] == 'current_invocation'",
            "m['input_checkpoint_sha256'] == sha('base/control.bin')",
            "m['control_state_checkpoint_sha256'] == m['output_sha256']",
        )
        with tempfile.TemporaryDirectory(
                prefix="lc-belief-matched-receipt-") as tmp:
            root = Path(tmp)
            for relative in campaign.BOUND_PATHS:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            workflow = root / campaign.WORKFLOW_PATH
            original = workflow.read_text(encoding="utf-8")
            with mock.patch.object(campaign, "_verify_external_bindings"):
                campaign.validate_plan(root)
                for fragment in required:
                    with self.subTest(fragment=fragment):
                        self.assertIn(fragment, original)
                        workflow.write_text(
                            original.replace(fragment, "removed", 1),
                            encoding="utf-8",
                        )
                        with self.assertRaisesRegex(
                                campaign.DefinitionError,
                                "matched-control receipt is not fully frozen"):
                            campaign.validate_plan(root)
                        workflow.write_text(original, encoding="utf-8")

    def test_validator_rejects_terminal_comparison_bundle_drift(self) -> None:
        with tempfile.TemporaryDirectory(
                prefix="lc-belief-terminal-bundle-") as tmp:
            root = Path(tmp)
            for relative in campaign.BOUND_PATHS:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            plan = load(root / campaign.PLAN_PATH)
            plan["evaluation"]["terminal_artifact_selection"] \
                ["comparison_bundle"] += " Drift."
            (root / campaign.PLAN_PATH).write_bytes(
                campaign.canonical_bytes(plan))
            with mock.patch.object(campaign, "_verify_external_bindings"):
                with self.assertRaisesRegex(
                        campaign.DefinitionError,
                        "terminal artifact comparison bundle changed"):
                    campaign.validate_plan(root)

    def test_prepare_and_guard_are_canonical_and_complete(self) -> None:
        # Binding generation is tested without weakening the production path
        # list: make a temporary complete mirror of every bound file.
        with tempfile.TemporaryDirectory(prefix="lc-belief-definition-") as tmp:
            root = Path(tmp)
            for relative in campaign.BOUND_PATHS:
                source = ROOT / relative
                self.assertTrue(source.is_file(), relative)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            output = root / campaign.EXECUTION_PATH
            commit = "1" * 40
            tree = "2" * 40
            with mock.patch.object(campaign, "validate_plan",
                                   return_value=load(PLAN)):
                campaign.prepare_execution(root, output, commit, tree)
                value = campaign.guard_execution(
                    root, output, commit, tree, check_git_child=False)
            self.assertEqual(value["schema"],
                             "lc-belief-history-v1-execution-v1")
            self.assertIsNone(value["results"])
            prerequisite = value["definition_validation_prerequisite"]
            self.assertEqual(prerequisite["conclusion"], "success")
            self.assertEqual(prerequisite["event"], "push")
            self.assertEqual(prerequisite["required_attempt"], 1)
            self.assertEqual(prerequisite["head_sha"], commit)
            self.assertEqual(prerequisite["head_tree"], tree)
            self.assertEqual(
                prerequisite["workflow"]["path"],
                campaign.DEFINITION_WORKFLOW_PATH.as_posix(),
            )
            self.assertEqual(
                prerequisite["workflow"]["sha256"],
                sha(root / campaign.DEFINITION_WORKFLOW_PATH),
            )
            self.assertEqual(len(value["bindings"]),
                             len(campaign.BOUND_PATHS))
            self.assertEqual(output.read_bytes(),
                             campaign.canonical_bytes(value))
            mutated = copy.deepcopy(value)
            mutated["fixed_roots"][0] = "202705100101"
            output.write_bytes(campaign.canonical_bytes(mutated))
            with mock.patch.object(campaign, "validate_plan",
                                   return_value=load(PLAN)):
                with self.assertRaisesRegex(campaign.DefinitionError,
                                            "not canonical or current"):
                    campaign.guard_execution(
                        root, output, commit, tree, check_git_child=False)


if __name__ == "__main__":
    unittest.main()
