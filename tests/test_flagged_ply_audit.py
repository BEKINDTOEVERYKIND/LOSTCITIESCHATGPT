from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/flagged_ply_audit.py"
SPEC = importlib.util.spec_from_file_location("flagged_ply_audit", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

EXECUTION_MODULE_PATH = ROOT / "tools/flagged_ply_execution.py"
EXECUTION_SPEC = importlib.util.spec_from_file_location(
    "flagged_ply_execution", EXECUTION_MODULE_PATH
)
assert EXECUTION_SPEC and EXECUTION_SPEC.loader
execution = importlib.util.module_from_spec(EXECUTION_SPEC)
EXECUTION_SPEC.loader.exec_module(execution)

ACTOR = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1"
)
CHALLENGER = ACTOR.replace(":512:", ":800:")

EXPECTED = {
    2214615196: {3, 4, 8, 10, 12, 13, 16, 20},
    5726968372613385: {4, 7, 14, 15, 17, 25, 32},
    725402798: {
        1, 2, 3, 7, 14, 21, 22, 23, 25, 29,
        30, 31, 36, 40, 46, 47, 55, 62, 63, 64,
    },
    95647345759839: {44},
}


def assert_workflow_mapping_keys_are_unique(
        test: unittest.TestCase, text: str) -> None:
    pattern = re.compile(
        r"^(?P<indent> *)(?P<list>- )?"
        r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*):(?P<value>.*)$"
    )
    stack: list[tuple[int, object]] = []
    seen: dict[object, set[str]] = {"root": set()}
    block_indent: int | None = None
    for number, raw in enumerate(text.splitlines(), 1):
        test.assertNotIn("\t", raw, f"workflow line {number} contains a tab")
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if block_indent is not None:
            if indent > block_indent:
                continue
            block_indent = None
        match = pattern.fullmatch(raw)
        if match is None and raw.lstrip().startswith("- "):
            while stack and stack[-1][0] >= indent:
                stack.pop()
            continue
        test.assertIsNotNone(match, f"unsupported YAML line {number}: {raw!r}")
        assert match is not None
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else "root"
        if match.group("list"):
            parent = ("item", number)
            seen[parent] = set()
            stack.append((indent, parent))
        key = match.group("key")
        test.assertNotIn(key, seen.setdefault(parent, set()))
        seen[parent].add(key)
        value = match.group("value").strip()
        if value in {"", "|", ">"}:
            node = ("mapping", number, key)
            seen[node] = set()
            stack.append((indent, node))
        if value in {"|", ">"}:
            block_indent = indent


def assert_workflow_shell_blocks_parse(
        test: unittest.TestCase, text: str) -> None:
    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)run:\s*\|\s*$", lines[index])
        if match is None:
            index += 1
            continue
        indent = len(match.group(1))
        index += 1
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line[indent + 2:] if len(line) >= indent + 2 else "")
            index += 1
        blocks.append("\n".join(body) + "\n")
    test.assertGreaterEqual(len(blocks), 6)
    for ordinal, block in enumerate(blocks):
        completed = subprocess.run(
            ["bash", "-n"], input=block, text=True,
            capture_output=True, check=False,
        )
        test.assertEqual(
            completed.returncode, 0,
            f"shell block {ordinal}: {completed.stderr}",
        )


class CorpusTests(unittest.TestCase):
    def test_manifest_is_exact_literal_inventory_and_hashes_verify(self) -> None:
        manifest, digest = audit.load_manifest(
            ROOT / "data/user_reviewed_plies.json"
        )
        self.assertEqual(len(digest), 64)
        actual: dict[int, set[int]] = {}
        for case in manifest["cases"]:
            actual.setdefault(int(case["seed"]), set()).add(int(case["ply"]))
        self.assertEqual(actual, EXPECTED)
        self.assertEqual(
            {case["id"] for case in manifest["cases"] if case["kind"] == "belief"},
            {"ui221-p13", "showcase572-p4"},
        )
        p14 = next(
            case for case in manifest["cases"]
            if case["id"] == "showcase572-p14"
        )
        self.assertEqual(p14["preferred"], ["G7 p deck", "B3 d deck"])
        self.assertEqual(p14["criticized"], ["R4 d deck"])

    def test_failure_classification_separates_policy_omission_and_rollout(self) -> None:
        case = {"preferred": ["B10 p deck"], "criticized": ["Y10 p deck"]}
        self.assertEqual(
            audit.classify_move(case, "Y10 p deck", {"Y10 p deck"}),
            "preferred_move_missing_from_top_policy_union",
        )
        self.assertEqual(
            audit.classify_move(
                case, "Y10 p deck", {"Y10 p deck", "B10 p deck"}
            ),
            "flagged_move_selected_by_rollout_panel",
        )
        self.assertEqual(
            audit.classify_move(
                case, "B10 p deck", {"Y10 p deck", "B10 p deck"}
            ),
            "review_aligned",
        )

    def test_locked_plan_and_worker_preserve_top_three_union_cap_five(self) -> None:
        plan = json.loads((ROOT / "data/flagged_ply_audit_plan.json").read_text())
        self.assertEqual(plan["cases"], 36)
        self.assertEqual(plan["decision_cases"], 34)
        self.assertEqual(plan["belief_only_cases"], 2)
        self.assertEqual(plan["decision_worlds_per_actor_per_case"], 16384)
        self.assertEqual(plan["history_worlds"], 20000)
        self.assertEqual(plan["belief_alpha"], 1.15)
        self.assertEqual(plan["base_seed"], 202608231701)
        self.assertEqual(plan["shards"]["count"], 12)
        worker = (ROOT / "tools/flagged_ply_probe.c").read_text()
        self.assertIn("#define AUDIT_TOP_MOVES 3", worker)
        self.assertIn("#define AUDIT_CANDIDATES 5", worker)
        self.assertIn("if (ncandidate > AUDIT_CANDIDATES)", worker)

    def test_match_value_table_is_hashed_as_actor_provenance(self) -> None:
        tail = ["0"] * 41 + ["data/champion.bin"]
        provenance = audit.actor_provenance(
            ":".join(["rolloutu", "data/champion.bin", *tail])
        )
        self.assertEqual(
            provenance["match_value_table"]["sha256"],
            provenance["checkpoints"][0]["sha256"],
        )

        ranker_provenance = audit.actor_provenance(
            ":".join([
                "rolloutu4",
                "data/champion.bin",
                "data/champion.bin",
                "data/champion.bin",
                *tail,
            ])
        )
        self.assertEqual(len(ranker_provenance["checkpoints"]), 3)
        self.assertEqual(
            ranker_provenance["match_value_table"]["sha256"],
            ranker_provenance["checkpoints"][2]["sha256"],
        )

    def test_merge_rejects_mixed_source_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable = {
                "manifest_sha256": "a" * 64,
                "reference": {}, "candidate": {}, "decision_worlds": 2,
                "belief_alpha": 1.15, "history_worlds": 1,
                "base_seed": 1, "shard_count": 2,
                "candidate_rule": "top three", "world_model": "uniform",
                "selection": "all", "execution_sha256": None,
                "source_tree": "3" * 40,
                "evaluator_manifest_sha256": None,
                "authoritative_result_sha256": None,
                "launch_mode": "local_unbound",
            }
            inputs = []
            for shard, commit in enumerate(("1" * 40, "2" * 40)):
                path = root / f"shard-{shard}.json"
                path.write_text(json.dumps({
                    "schema": "lc-flagged-ply-audit-v1",
                    "provenance": {
                        **stable, "source_commit": commit,
                        "shard_index": shard,
                    },
                    "errors": [], "cases": [],
                }))
                inputs.append(path)
            completed = subprocess.run(
                [
                    "python3", "tools/merge_flagged_ply_audit.py",
                    *(str(path) for path in inputs),
                    "--allow-partial", "--output", str(root / "merged.json"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("source_commit", completed.stderr)

    def _final_fixture(self, root: Path, passed: bool = True,
                       same_actor: bool = False,
                       mode: str = "component_final",
                       decision_schema: str = "standard") -> Path:
        (root / "data/experiments").mkdir(parents=True, exist_ok=True)
        (root / "data").mkdir(exist_ok=True)
        (root / "data/champion.bin").write_bytes(b"fixture-model")
        source = "2" * 40
        tree = "3" * 40
        challenger = ACTOR if same_actor else CHALLENGER
        reciprocal_path = root / "data/experiments/final-reciprocal.json"
        reciprocal_path.write_text('{"raw":"fixture"}\n', encoding="utf-8")
        reciprocal_sha = execution.sha256(reciprocal_path)
        if decision_schema == "standard":
            requirements = {
                "raw_inputs_validated": True,
                "zero_capped_rounds": True,
                "match_score_one_sided_lower_bound_above_half": passed,
                "margin_one_sided_lower_bound_strictly_positive": passed,
                "each_orientation_match_score_strictly_above_half": passed,
            }
            artifact_kind = "locked_reciprocal_actor_gate_decision"
            decision_status = {}
            gate_field = "passed"
        else:
            requirements = {
                "complete_equal_reciprocal_blocks": True,
                "raw_inputs_validated": True,
                "zero_capped_rounds": True,
                "pair_clustered_orientation_stratified_score_lcb_above_half": passed,
                "combined_match_score_point_estimate_above_half": passed,
                "each_reciprocal_orientation_strictly_above_half": passed,
                "combined_margin_strictly_positive": passed,
            }
            artifact_kind = "match_value_reserved_final_gate"
            decision_status = {"status": "complete_reserved_final_test"}
            gate_field = "promotion_gate_passed"
        decision = {
            "schema_version": 1,
            "artifact_kind": artifact_kind,
            "candidate": challenger,
            "baseline": ACTOR,
            "provenance": f"campaign=fixture;source={source};tree={tree}",
            "reciprocal_path": "runtime/final-reciprocal.json",
            "reciprocal_sha256": reciprocal_sha,
            "requirements": requirements,
            gate_field: passed,
            **decision_status,
        }
        if decision_schema == "standard":
            decision["mode"] = "final"
        decision_path = root / "data/experiments/final-decision.json"
        decision_path.write_text(
            json.dumps(decision, sort_keys=True) + "\n", encoding="utf-8"
        )
        digest = execution.sha256(decision_path)
        authoritative = [
            {
                "path": "data/experiments/final-decision.json",
                "sha256": digest,
                "role": "final_decision",
            },
            {
                "path": "data/experiments/final-reciprocal.json",
                "sha256": reciprocal_sha,
                "role": "final_reciprocal",
            },
        ]
        if mode == "composition_final":
            source_binding_path = root / "data/experiments/source-binding.json"
            source_binding_path.write_text(json.dumps({
                "artifact_kind": "locked_composition_pre_efficacy_manifest",
                "status": "frozen_before_composition_efficacy",
                "source": {"commit": source, "tree": tree},
                "actors": {
                    "reference": {"spec": ACTOR},
                    "challengers": [{"spec": challenger}],
                },
            }) + "\n")
            authoritative.append({
                "path": "data/experiments/source-binding.json",
                "sha256": execution.sha256(source_binding_path),
                "role": "composition_source_binding",
            })
        if decision_schema == "match_value":
            source_binding_path = root / "data/experiments/match-value-source.json"
            source_binding_path.write_text(json.dumps({
                "artifact_kind": "match_value_pre_efficacy_build_manifest",
                "source": {"commit": source, "tree": tree},
                "actors": {"actors": {
                    "legacy": ACTOR, "selected": challenger,
                }},
            }) + "\n")
            authoritative.append({
                "path": "data/experiments/match-value-source.json",
                "sha256": execution.sha256(source_binding_path),
                "role": "match_value_source_binding",
            })
        reference_provenance = execution._actor_provenance(root, ACTOR, "reference")
        challenger_provenance = execution._actor_provenance(
            root, challenger, "challenger")
        winner = challenger if passed else ACTOR
        winner_provenance = challenger_provenance if passed else reference_provenance
        final = {
            "schema": execution.FINAL_SCHEMA,
            "status": "complete",
            "selection_mode": mode,
            "source_commit": source,
            "source_tree": tree,
            "reference_actor": ACTOR,
            "challenger_actor": challenger,
            "winner_actor": winner,
            "actor_assets": {
                "reference": execution._actor_assets(reference_provenance),
                "challenger": execution._actor_assets(challenger_provenance),
                "winner": execution._actor_assets(winner_provenance),
            },
            "decisive_result": {
                "path": "data/experiments/final-decision.json",
                "sha256": digest,
            },
            "authoritative_results": authoritative,
        }
        final_path = root / execution.FINAL_RESULT_PATH
        final_path.write_text(
            json.dumps(final, sort_keys=True) + "\n", encoding="utf-8"
        )
        return final_path

    def test_authoritative_result_mechanically_selects_pass_or_no_change(self) -> None:
        for mode in ("component_final", "composition_final"):
            for passed, expected in ((True, CHALLENGER), (False, ACTOR)):
                with self.subTest(mode=mode, passed=passed), \
                        tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._final_fixture(root, passed=passed, mode=mode)
                    context = mock.patch.dict(sys.modules, {
                        "tools.composition_campaign": types.SimpleNamespace(
                            validate_frozen_composition_manifest=lambda _, path:
                            execution.strict_json(path)
                        )
                    }) if mode == "composition_final" else mock.patch.dict(
                        sys.modules, {})
                    with context, mock.patch.object(
                            execution, "_revalidate_standard_gate",
                            return_value=passed):
                        bound = execution.authoritative_final_result(root)
                    self.assertEqual(bound["winner"]["spec"], expected)
                    self.assertEqual(bound["no_change"], not passed)
                    self.assertEqual(bound["promotion_gate_passed"], passed)

    def test_match_value_component_final_uses_its_distinct_gate_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._final_fixture(
                root, mode="component_final", decision_schema="match_value")
            with mock.patch.object(
                    execution, "_revalidate_match_value_gate",
                    return_value=True):
                bound = execution.authoritative_final_result(root)
            self.assertEqual(bound["selection_mode"], "component_final")
            self.assertEqual(bound["winner"]["spec"], CHALLENGER)

    def test_standard_gate_is_rebuilt_and_identity_schedule_checked(self) -> None:
        provenance = "stage=x;source=" + "2" * 40 + ";tree=" + "3" * 40
        seeds = {"candidate_first": "11", "baseline_first": "12"}
        reciprocal = {
            "candidate": CHALLENGER,
            "baseline": ACTOR,
            "provenance": provenance,
            "blocks": [
                {"pair_start": "0", "pair_count": 2500, "metadata": {
                    "agent_a": CHALLENGER, "agent_b": ACTOR,
                    "seed": "11", "rounds": 3, "provenance": provenance,
                }},
                {"pair_start": "0", "pair_count": 2500, "metadata": {
                    "agent_a": ACTOR, "agent_b": CHALLENGER,
                    "seed": "12", "rounds": 3, "provenance": provenance,
                }},
            ],
        }
        expected = {
            "artifact_kind": "locked_reciprocal_actor_gate_decision",
            "mode": "final", "passed": True,
        }
        decision = {
            **expected,
            "candidate": CHALLENGER, "baseline": ACTOR,
            "provenance": provenance,
            "reciprocal_path": "runtime/final.json",
            "reciprocal_sha256": "4" * 64,
            "pairs_per_orientation": 2500,
            "seeds": seeds,
        }
        with mock.patch(
                "tools.gate_actor_panel._rebuild_reciprocal",
                return_value=(reciprocal, "4" * 64)), mock.patch(
                "tools.gate_actor_panel.evaluate_gate",
                return_value=expected):
            self.assertTrue(execution._revalidate_standard_gate(
                ROOT, decision, ROOT / "unused.json", "4" * 64))
            decision["pairs_per_orientation"] = 2499
            with self.assertRaises(execution.ExecutionError):
                execution._revalidate_standard_gate(
                    ROOT, decision, ROOT / "unused.json", "4" * 64)

    def test_match_value_gate_is_rebuilt_exactly(self) -> None:
        expected = {
            "artifact_kind": "match_value_reserved_final_gate",
            "status": "complete_reserved_final_test",
            "promotion_gate_passed": True,
        }
        decision = {
            **expected,
            "candidate": CHALLENGER,
            "baseline": ACTOR,
            "provenance": "fixture",
            "reciprocal_path": "runtime/final.json",
            "reciprocal_sha256": "5" * 64,
            "seeds": {"candidate_first": "21", "baseline_first": "22"},
        }
        with mock.patch(
                "tools.match_value_campaign.load_verified_panel",
                return_value=({}, "5" * 64)), mock.patch(
                "tools.match_value_campaign.final_gate",
                return_value=expected):
            self.assertTrue(execution._revalidate_match_value_gate(
                ROOT, decision, ROOT / "unused.json", "5" * 64))
            decision["unexpected"] = True
            with self.assertRaises(execution.ExecutionError):
                execution._revalidate_match_value_gate(
                    ROOT, decision, ROOT / "unused.json", "5" * 64)

    def test_no_challenge_mechanically_carries_forward_world_winner(self) -> None:
        from tools.match_value_campaign import (
            BASELINE_512, WORLD800_SOURCE_COMMIT, WORLD800_SOURCE_TREE,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/experiments").mkdir(parents=True)
            (root / "data/champion.bin").write_bytes(b"fixture-model")
            world = root / "data/experiments/world-result.json"
            world.write_text('{"fixture":true}\n')
            reference = execution._actor_provenance(
                root, BASELINE_512, "reference")
            final = {
                "schema": execution.FINAL_SCHEMA,
                "status": "complete",
                "selection_mode": "no_challenge",
                "source_commit": WORLD800_SOURCE_COMMIT,
                "source_tree": WORLD800_SOURCE_TREE,
                "reference_actor": BASELINE_512,
                "challenger_actor": None,
                "winner_actor": BASELINE_512,
                "actor_assets": {
                    "reference": execution._actor_assets(reference),
                    "challenger": [],
                    "winner": execution._actor_assets(reference),
                },
                "decisive_result": None,
                "authoritative_results": [{
                    "path": "data/experiments/world-result.json",
                    "sha256": execution.sha256(world),
                    "role": "world_winner",
                }],
            }
            final_path = root / execution.FINAL_RESULT_PATH
            final_path.write_text(json.dumps(final) + "\n")
            with mock.patch(
                    "tools.match_value_campaign._world_result",
                    return_value=({}, False, BASELINE_512, 512)):
                bound = execution.authoritative_final_result(root)
            self.assertEqual(bound["selection_mode"], "no_challenge")
            self.assertEqual(bound["winner"]["spec"], BASELINE_512)
            self.assertTrue(bound["no_change"])

    def test_authoritative_result_rejects_same_actor_or_gate_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._final_fixture(root, same_actor=True)
            with self.assertRaises(execution.ExecutionError):
                execution.authoritative_final_result(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = self._final_fixture(root, passed=True)
            final = json.loads(final_path.read_text())
            decision_path = root / final["decisive_result"]["path"]
            decision = json.loads(decision_path.read_text())
            decision["passed"] = False
            decision_path.write_text(json.dumps(decision) + "\n")
            digest = execution.sha256(decision_path)
            final["decisive_result"]["sha256"] = digest
            final["authoritative_results"][0]["sha256"] = digest
            final_path.write_text(json.dumps(final) + "\n")
            with self.assertRaises(execution.ExecutionError):
                execution.authoritative_final_result(root)

    def test_prepare_execution_is_atomic_canonical_and_exactly_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._final_fixture(root)
            for name in (execution.PLAN_PATH, execution.WORKFLOW_PATH,
                         *execution.TOOL_PATHS):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"fixture {name}\n", encoding="utf-8")
            output = root / execution.EXECUTION_PATH
            manifest = {
                "path": execution.MANIFEST_PATH,
                "sha256": "4" * 64,
                "cases": 36,
                "decision_cases": 34,
                "belief_cases": 2,
            }
            with mock.patch.object(
                    execution, "_manifest_binding", return_value=(manifest, [])), \
                    mock.patch.object(
                        execution, "_revalidate_standard_gate",
                        return_value=True):
                created = execution.prepare_execution(
                    root, output, "0" * 40, "1" * 40)
                self.assertEqual(execution.strict_json(output), created)
                self.assertEqual(created["audit"], {
                    "decision_worlds_per_actor_per_case": 16384,
                    "history_worlds": 20000,
                    "belief_alpha": 1.15,
                    "base_seed": 202608231701,
                    "shard_count": 12,
                    "candidate_rule": (
                        "top three complete semantic policy moves per actor; "
                        "deterministic union capped at five; never all legal moves"
                    ),
                })
                self.assertEqual(created["build"]["runner"], "ubuntu-24.04")
                self.assertEqual(
                    created["build"]["required_compiler_semantic_version"],
                    "13.3.0",
                )
                snapshot = output.read_bytes()
                with self.assertRaises(execution.ExecutionError):
                    execution.prepare_execution(
                        root, output, "0" * 40, "1" * 40)
                self.assertEqual(output.read_bytes(), snapshot)
                with self.assertRaises(execution.ExecutionError):
                    execution.prepare_execution(
                        root, root / "execution.json", "0" * 40, "1" * 40)

    def test_execution_template_is_inert_and_real_addendum_absent(self) -> None:
        template = json.loads((
            ROOT / "data/flagged_ply_audit_execution.template.json"
        ).read_text())
        self.assertEqual(
            template["status"], "inert_example_only_do_not_copy_or_edit"
        )
        self.assertNotEqual(template["schema"], execution.SCHEMA)
        self.assertFalse(
            (ROOT / "data/flagged_ply_audit_execution.json").exists(),
            "the real one-shot launch addendum must not be created early",
        )

    def test_workflow_is_addendum_only_compile_once_and_fully_bound(self) -> None:
        workflow = (ROOT / ".github/workflows/flagged-ply-audit.yml").read_text()
        assert_workflow_mapping_keys_are_unique(self, workflow)
        assert_workflow_shell_blocks_parse(self, workflow)
        self.assertIn("push:", workflow)
        self.assertNotIn("workflow_dispatch", workflow)
        self.assertIn("data/flagged_ply_audit_execution.json", workflow)
        self.assertIn("test \"$GITHUB_RUN_ATTEMPT\" = 1", workflow)
        self.assertIn("git diff-tree --no-commit-id --name-status", workflow)
        self.assertIn("ubuntu-24.04", workflow)
        self.assertIn("REQUIRED_COMPILER_SEMANTIC_VERSION: '13.3.0'", workflow)
        self.assertIn("compiler_banner=$(gcc --version | head -1)", workflow)
        self.assertEqual(workflow.count("make -C source"), 1)
        self.assertEqual(workflow.count("actions/checkout@"), 1)
        self.assertNotIn("actions/checkout@v", workflow)
        self.assertNotIn("actions/upload-artifact@v", workflow)
        self.assertNotIn("actions/download-artifact@v", workflow)
        self.assertIn("AUDIT_WORLDS: '16384'", workflow)
        self.assertIn("HISTORY_WORLDS: '20000'", workflow)
        self.assertIn("BELIEF_ALPHA: '1.15'", workflow)
        self.assertIn("BASE_SEED: '202608231701'", workflow)
        self.assertIn("SHARD_COUNT: '12'", workflow)
        self.assertIn("--launch-mode addendum_push", workflow)
        self.assertIn("EVALUATOR_SHA256SUMS.txt", workflow)
        self.assertIn("raw-shards", workflow)
        self.assertNotIn("pull_request:", workflow)


@unittest.skipUnless(
    (ROOT / "bin/flagged_ply_probe").is_file()
    and (ROOT / "bin/history_belief").is_file(),
    "build bin/flagged_ply_probe and bin/history_belief for integration tests",
)
class ProbeIntegrationTests(unittest.TestCase):
    def probe(self, state: str, worlds: int) -> dict:
        completed = subprocess.run(
            [
                str(ROOT / "bin/flagged_ply_probe"),
                "-S", str(ROOT / state),
                "-a", ACTOR,
                "-b", ACTOR,
                "-w", str(worlds),
                "-s", "202608231799",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(completed.stdout)

    def test_identical_actors_get_identical_common_panel(self) -> None:
        result = self.probe(
            "data/probes/ui_seed725402798_p36.state", 32
        )
        self.assertLessEqual(result["evaluated_moves"], 5)
        self.assertLess(result["evaluated_moves"], result["legal_moves"])
        self.assertEqual(
            result["actors"][0]["rows"], result["actors"][1]["rows"]
        )
        self.assertEqual(
            result["actors"][0]["panel_selected"],
            result["actors"][1]["panel_selected"],
        )
        self.assertEqual(
            result["actors"][0]["deployed_selected"],
            result["actors"][1]["deployed_selected"],
        )
        self.assertEqual(result["actors"][0]["unfinished_cap_leaves"], 0)
        self.assertEqual(result["actors"][0]["objective_label"], "round_margin")
        self.assertEqual(result["actors"][0]["objective_units"], "round_points")

    def test_probe_distinguishes_final_hybrid_from_round_points(self) -> None:
        fields = ACTOR.split(":")
        # One-network rollout tail field 8 is the selection objective.
        fields[2 + 8] = "2"
        hybrid_actor = ":".join(fields)
        completed = subprocess.run(
            [
                str(ROOT / "bin/flagged_ply_probe"),
                "-S", str(ROOT / "data/probes/g424_p111.state"),
                "-a", ACTOR,
                "-b", hybrid_actor,
                "-w", "2",
                "-s", "202608231798",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        actors = json.loads(completed.stdout)["actors"]
        self.assertEqual(
            (actors[0]["objective_label"], actors[0]["objective_units"]),
            ("round_margin", "round_points"),
        )
        self.assertEqual(
            (actors[1]["objective_label"], actors[1]["objective_units"]),
            ("final_hybrid", "hybrid_match_utility_points"),
        )

    def test_decision_report_renders_objective_label_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            markdown = Path(directory) / "audit.md"
            subprocess.run(
                [
                    "python3", "tools/flagged_ply_audit.py",
                    "--reference", ACTOR,
                    "--candidate", ACTOR,
                    "--case", "ui725-p1",
                    "--worlds", "2",
                    "--output", str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(
                [
                    "python3", "tools/render_flagged_ply_audit.py",
                    str(output), "--output", str(markdown),
                ],
                cwd=ROOT,
                check=True,
            )
            report = markdown.read_text()
            self.assertIn("Panel objectives: reference", report)
            self.assertIn("`round_margin` in `round_points`", report)
            self.assertIn("only when both labels and units match", report)

    def test_deck_two_uses_complete_ordered_hidden_support(self) -> None:
        result = self.probe(
            "data/probes/ui_seed95647345759839_p43.state", 1000
        )
        for actor in result["actors"]:
            self.assertEqual(actor["hidden_support"], 90)
            self.assertEqual(actor["worlds"], 90)
            self.assertTrue(actor["exact_hidden_support"])
            self.assertEqual(actor["unfinished_cap_leaves"], 0)

    def test_deck_one_uses_complete_hidden_support(self) -> None:
        result = self.probe(
            "data/probes/ui_seed95647345759839_p44.state", 1000
        )
        admitted = {row["move"] for row in result["candidates"]}
        self.assertIn("G8 p deck", admitted)
        self.assertIn("G8 p W", admitted)
        self.assertIn("complete semantic policy moves", result["candidate_rule"])
        for actor in result["actors"]:
            self.assertEqual(actor["hidden_support"], 9)
            self.assertEqual(actor["worlds"], 9)
            self.assertTrue(actor["exact_hidden_support"])
            self.assertEqual(actor["unfinished_cap_leaves"], 0)
            belief = actor["belief"]
            cards = belief["cards"]
            self.assertEqual(
                len({row["card"] for row in cards}), len(cards),
                "indistinguishable wager copies must share one semantic row",
            )
            for row in cards:
                self.assertAlmostEqual(
                    row["head_minus_prior"],
                    row["estimate"] - row["prior"],
                    places=7,
                )
                if row["metric"] == "expected_count":
                    self.assertAlmostEqual(
                        row["prior"],
                        row["unseen_copies"]
                        * belief["unknown_hand"] / belief["unknown_pool"],
                        places=7,
                    )

    def test_runner_reports_belief_focus_without_large_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            subprocess.run(
                [
                    "python3", "tools/flagged_ply_audit.py",
                    "--reference", ACTOR,
                    "--candidate", ACTOR,
                    "--case", "ui221-p13",
                    "--worlds", "32",
                    "--history-worlds", "50",
                    "--output", str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            result = json.loads(output.read_text())
            case = result["cases"][0]
            self.assertEqual(case["kind"], "belief")
            self.assertTrue(case["probe"]["belief_only"])
            self.assertEqual(case["probe"]["evaluated_moves"], 0)
            self.assertEqual(case["probe"]["candidates"], [])
            self.assertFalse(case["probe"]["actors"][0]["action_panel"])
            self.assertNotIn("rows", case["probe"]["actors"][0])
            history = case["history_aware_belief"]
            self.assertEqual(
                history["provenance"]["view_sha256"],
                "a9ef8595235d5b1de3e168c13cbe57fe4943fd703cce665d1acee68b46944725",
            )
            self.assertEqual(
                history["status"], "insufficient_accepted_support"
            )
            focus = case["classifications"]["candidate"]["focus_cards"]
            self.assertEqual([row["card"] for row in focus], ["Y9"])
            belief = case["probe"]["actors"][1]["belief"]
            self.assertAlmostEqual(
                focus[0]["prior"],
                belief["unknown_hand"] / belief["unknown_pool"],
                places=7,
            )
            self.assertAlmostEqual(
                focus[0]["head_minus_prior"],
                focus[0]["probability"] - focus[0]["prior"],
                places=7,
            )
            markdown = Path(directory) / "audit.md"
            subprocess.run(
                [
                    "python3", "tools/render_flagged_ply_audit.py",
                    str(output), "--output", str(markdown),
                ],
                cwd=ROOT,
                check=True,
            )
            report = markdown.read_text()
            self.assertIn("## ui221-p13", report)
            self.assertIn("snapshot-only", report.lower())
            self.assertIn("insufficient_accepted_support", report)
            self.assertIn("No action or rollout-Q panel was run", report)
            self.assertNotIn("Admitted policy moves", report)
            self.assertNotIn("Panel selection", report)

    def test_history_aware_belief_consumes_frozen_public_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            subprocess.run(
                [
                    "python3", "tools/flagged_ply_audit.py",
                    "--reference", ACTOR,
                    "--candidate", ACTOR,
                    "--case", "showcase572-p4",
                    "--worlds", "2",
                    "--history-worlds", "1000",
                    "--output", str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            case = json.loads(output.read_text())["cases"][0]
            history = case["history_aware_belief"]
            self.assertEqual(history["status"], "ok")
            self.assertGreater(history["accepted"], 0)
            focus = case["classifications"]["history_aware"]["focus_cards"]
            self.assertEqual(
                [row["card"] for row in focus], ["Y4", "Y9", "Y10"]
            )
            self.assertEqual(
                history["provenance"]["checkpoint_sha256"],
                "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417",
            )


if __name__ == "__main__":
    unittest.main()
