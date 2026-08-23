"""Fail-closed contracts for the precommitted 800-world actor campaign.

The workflow is intentionally published before its execution addendum.  These
tests bind the frozen source, mechanically selected role-coherent actors,
reserved seeds, complete shard set, evidence tools, and exact promotion gate.
They also ensure the template cannot itself trigger the campaign.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/experiments/locked_world800_plan.json"
PARENT_RESULT = ROOT / "data/experiments/role_coherent_result.json"
WORKFLOW = ROOT / ".github/workflows/world800.yml"
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"
TEMPLATE = (
    ROOT / "data/experiments/locked_world800_execution.template.json"
)
EXECUTION = ROOT / "data/experiments/locked_world800_execution.json"
MERGER = ROOT / "tools/merge_arena.py"
VALIDATOR = ROOT / "tools/validate_actor_shards.py"
MODEL = ROOT / "data/champion.bin"

SOURCE_COMMIT = "08f9e1a5218e03c399b257b852efe20b0089c7b0"
SOURCE_TREE = "c70405a09b88919b228f96d19d84d83875d4fea4"
PLAN_SHA = "3f7d4e8b4be2c58268c9f85ade126a7f15357ab30bf146d71b3c6dc247e74e34"
PARENT_RESULT_SHA = (
    "9ae1caa83b9a2ffef715a6c90c3987e386795a00cd92bd19f000f8d2ca1811fb"
)
MODEL_SHA = "af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417"
MERGER_SHA = "9cad23c9e6550ea36d7721acf8e64144a44058083ad4aeb5bb5613a3a79139fb"
VALIDATOR_SHA = (
    "bca430a94af64180436c7fb60d29b2e86ec4b3567ab3aabb09984aabee054855"
)
CFLAGS = (
    "-O3 -march=x86-64-v3 -ffast-math -funroll-loops "
    "-Wall -Wextra -std=c11"
)
BASELINE = (
    "rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
CANDIDATE = (
    "rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:"
    "3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:0:"
    "0:0:0:0:1"
)
STARTS = list(range(0, 2500, 100))


def expected_execution(parent: str, workflow_sha: str) -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "locked_world800_execution",
        "status": "launch_bound_before_world800_efficacy",
        "source_parent_commit": parent,
        "workflow": {
            "path": ".github/workflows/world800.yml",
            "sha256": workflow_sha,
        },
        "plan": {
            "path": "data/experiments/locked_world800_plan.json",
            "sha256": PLAN_SHA,
        },
        "authoritative_parent_result": {
            "path": "data/experiments/role_coherent_result.json",
            "sha256": PARENT_RESULT_SHA,
            "promotion_gate_passed": True,
            "selected_family": "role_coherent_mode4_prefix3",
        },
        "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
        "build": {
            "runner": "ubuntu-24.04",
            "compiler": "gcc",
            "cflags": CFLAGS,
            "ldflags": "-lm -pthread",
            "single_evaluator_binary": True,
        },
        "model": {"path": "data/champion.bin", "sha256": MODEL_SHA},
        "evidence_tools": {
            "merge_arena_sha256": MERGER_SHA,
            "validate_actor_shards_sha256": VALIDATOR_SHA,
        },
        "actors": {
            "selection_rule": "role_coherent_parent_pass",
            "candidate": CANDIDATE,
            "baseline": BASELINE,
        },
        "final": {
            "rounds": 3,
            "mirrored_deals": True,
            "threads_per_shard": 4,
            "pairs_per_orientation": 2500,
            "shards_per_orientation": 25,
            "pairs_per_shard": 100,
            "starts": STARTS,
            "candidate_first_seed": "202608221501",
            "baseline_first_seed": "202608221502",
            "gate_z": 1.645,
            "gate": (
                "score-z*SE>0.5; each orientation score>0.5; "
                "combined margin>0; zero caps; exact raw validity"
            ),
        },
        "inspection_rule": (
            "No efficacy parse, merge, gate, or selection occurs until all "
            "50 immutable raw shards, hash sidecars, and timing sidecars have "
            "completed and structurally validated."
        ),
        "results": None,
    }


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token}")


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def strict_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workflow_env(text: str, name: str) -> str:
    match = re.search(
        rf"(?m)^  {re.escape(name)}:\s*['\"]?([^'\"\n]+?)['\"]?\s*$",
        text,
    )
    if match is None:
        raise AssertionError(f"workflow omits env {name}")
    return match.group(1)


def assert_workflow_mapping_keys_are_unique(test: unittest.TestCase,
                                            text: str) -> None:
    """Reject duplicate keys in the workflow without a PyYAML dependency.

    The workflow uses the deliberately small mapping/list/block-scalar subset
    of YAML supported here.  Each list item gets its own mapping identity, so
    repeated keys in different action steps remain valid while a duplicate
    ``env`` (or any other key) inside one step fails.
    """
    key_pattern = re.compile(
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
        match = key_pattern.fullmatch(raw)
        if match is None and raw.lstrip().startswith("- "):
            # Scalar sequence member (branch, path, or artifact path).
            while stack and stack[-1][0] >= indent:
                stack.pop()
            continue
        test.assertIsNotNone(match, f"unsupported YAML line {number}: {raw!r}")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else "root"
        if match.group("list"):
            parent = ("item", number)
            seen[parent] = set()
            stack.append((indent, parent))
        key = match.group("key")
        test.assertNotIn(
            key, seen.setdefault(parent, set()),
            f"duplicate YAML key {key!r} on line {number}",
        )
        seen[parent].add(key)
        value = match.group("value").strip()
        if value in {"", "|", ">"}:
            node = ("mapping", number, key)
            seen[node] = set()
            stack.append((indent, node))
        if value in {"|", ">"}:
            block_indent = indent


class World800CampaignTests(unittest.TestCase):
    maxDiff = None

    def test_plan_locks_exact_parent_selected_actor_and_only_world_counts(self) -> None:
        plan = strict_json(PLAN)
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(
            plan["artifact_kind"],
            "precommitted_conditional_world_count_strength_plan",
        )
        self.assertIsNone(plan["results"])
        dependency = plan["dependency_and_selection_rule"]
        self.assertTrue(dependency["no_other_result_dependent_choice"])
        self.assertEqual(dependency["role_pass_candidate"], CANDIDATE)
        self.assertEqual(dependency["role_pass_baseline"], BASELINE)
        self.assertIn("If and only if", dependency["rule"])
        self.assertIn("promotion_gate_passed=true", dependency["rule"])
        intended = plan["candidate_definition"]["only_intended_changes"]
        self.assertEqual(len(intended), 2)
        self.assertTrue(all("512 to 800" in change for change in intended))
        self.assertIn(
            "Only the existing top-policy prefix",
            plan["candidate_definition"]["candidate_width"],
        )

    def test_plan_locks_source_model_complete_panel_and_exact_gate(self) -> None:
        plan = strict_json(PLAN)
        source = plan["source"]
        self.assertEqual(source["remote_evaluation_commit"], SOURCE_COMMIT)
        self.assertEqual(source["remote_evaluation_tree"], SOURCE_TREE)
        self.assertEqual(source["cflags"], CFLAGS)
        self.assertEqual(plan["model"], {
            "path": "data/champion.bin",
            "sha256": MODEL_SHA,
        })
        panel = plan["locked_actor_test"]
        self.assertEqual(panel["rounds_per_game"], 3)
        self.assertTrue(panel["mirrored_deals"])
        self.assertEqual(panel["threads_per_shard"], 4)
        self.assertEqual(panel["pairs_per_orientation"], 2500)
        self.assertEqual(panel["shards_per_orientation"], 25)
        self.assertEqual(panel["pairs_per_shard"], 100)
        self.assertEqual(panel["pair_starts"], STARTS)
        self.assertEqual(panel["blocks"][0]["seed"], "202608221501")
        self.assertEqual(panel["blocks"][1]["seed"], "202608221502")
        self.assertEqual(panel["efficacy_looks"], 1)
        self.assertTrue(panel["no_optional_stopping_or_variant_changes"])
        self.assertEqual(panel["one_sided_alpha"], 0.05)
        self.assertEqual(panel["promotion_gate"], [
            "combined candidate match score minus 1.645 pair-clustered standard errors exceeds 0.5",
            "candidate match-score point estimate exceeds 0.5 in each reciprocal orientation",
            "combined candidate point margin exceeds 0",
            "all recorded raw shards reopen, hash, and exactly remerge to the submitted summaries",
            "zero cap-terminated rounds, incomplete shards, gaps, overlaps, malformed rows, provenance mismatches, or operational errors",
        ])

    def test_all_frozen_hashes_and_authoritative_parent_are_current(self) -> None:
        self.assertEqual(sha256(PLAN), PLAN_SHA)
        self.assertEqual(sha256(PARENT_RESULT), PARENT_RESULT_SHA)
        self.assertEqual(sha256(MODEL), MODEL_SHA)
        self.assertEqual(sha256(MERGER), MERGER_SHA)
        self.assertEqual(sha256(VALIDATOR), VALIDATOR_SHA)
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{SOURCE_COMMIT}^{{tree}}"],
            cwd=ROOT,
            text=True,
        ).strip()
        self.assertEqual(tree, SOURCE_TREE)
        parent = strict_json(PARENT_RESULT)
        self.assertEqual(parent["status"], "complete_valid_gate_passed")
        self.assertTrue(parent["decision"]["promotion_gate_passed"])
        self.assertEqual(parent["candidate"]["spec"], BASELINE)
        self.assertEqual(parent["evidence"]["capped_rounds"], 0)
        self.assertTrue(
            parent["evidence"]["all_raw_inputs_reopened_and_exactly_remerged"]
        )

    def test_template_is_exact_and_any_real_addendum_has_fail_closed_history(self) -> None:
        template = strict_json(TEMPLATE)
        self.assertEqual(
            template,
            expected_execution(
                "__COMMIT_CONTAINING_WORLD800_WORKFLOW__",
                "__SHA256_OF_COMMITTED_WORLD800_WORKFLOW__",
            ),
        )
        self.assertEqual(template["schema_version"], 1)
        self.assertEqual(template["artifact_kind"], "locked_world800_execution")
        self.assertEqual(
            template["status"], "launch_bound_before_world800_efficacy"
        )
        self.assertEqual(
            template["source_parent_commit"],
            "__COMMIT_CONTAINING_WORLD800_WORKFLOW__",
        )
        self.assertEqual(
            template["workflow"],
            {
                "path": ".github/workflows/world800.yml",
                "sha256": "__SHA256_OF_COMMITTED_WORLD800_WORKFLOW__",
            },
        )
        self.assertEqual(template["plan"]["sha256"], PLAN_SHA)
        self.assertEqual(
            template["authoritative_parent_result"]["sha256"],
            PARENT_RESULT_SHA,
        )
        self.assertTrue(
            template["authoritative_parent_result"]["promotion_gate_passed"]
        )
        self.assertEqual(template["source"], {
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
        })
        self.assertEqual(template["build"]["cflags"], CFLAGS)
        self.assertTrue(template["build"]["single_evaluator_binary"])
        self.assertEqual(template["model"]["sha256"], MODEL_SHA)
        self.assertEqual(template["actors"], {
            "selection_rule": "role_coherent_parent_pass",
            "candidate": CANDIDATE,
            "baseline": BASELINE,
        })
        self.assertEqual(template["final"]["starts"], STARTS)
        self.assertEqual(template["final"]["candidate_first_seed"], "202608221501")
        self.assertEqual(template["final"]["baseline_first_seed"], "202608221502")
        self.assertIsNone(template["results"])

        # Before launch, absence is the intended and CI-clean state.  After
        # launch, validate the immutable bytes from the unique add-only commit
        # instead of permanently failing the repository's test suite.
        tracked = subprocess.run(
            ["git", "cat-file", "-e", "HEAD:data/experiments/locked_world800_execution.json"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if not tracked:
            self.assertFalse(EXECUTION.exists())
            return

        launches = subprocess.check_output(
            [
                "git", "log", "--diff-filter=A", "--format=%H", "--",
                "data/experiments/locked_world800_execution.json",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines()
        self.assertEqual(len(launches), 1)
        launch = launches[0]
        parent_line = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", launch],
            cwd=ROOT,
            text=True,
        ).split()
        self.assertEqual(len(parent_line), 2)
        parent = parent_line[1]
        changed = subprocess.check_output(
            [
                "git", "diff-tree", "--no-commit-id", "--name-status", "-r",
                launch,
            ],
            cwd=ROOT,
            text=True,
        )
        self.assertEqual(
            changed,
            "A\tdata/experiments/locked_world800_execution.json\n",
        )
        workflow_bytes = subprocess.check_output(
            ["git", "show", f"{launch}:.github/workflows/world800.yml"],
            cwd=ROOT,
        )
        launch_bytes = subprocess.check_output(
            [
                "git", "show",
                f"{launch}:data/experiments/locked_world800_execution.json",
            ],
            cwd=ROOT,
        )
        launch_value = json.loads(
            launch_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        self.assertEqual(
            launch_value,
            expected_execution(parent, hashlib.sha256(workflow_bytes).hexdigest()),
        )
        # The execution addendum is evidence, not a mutable status document.
        # Completion belongs in a separate result artifact.
        self.assertEqual(EXECUTION.read_bytes(), launch_bytes)

    def test_workflow_is_valid_yaml_and_only_future_addendum_can_trigger(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert_workflow_mapping_keys_are_unique(self, text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("pull_request", text)
        self.assertNotIn("continue-on-error", text)
        self.assertIn("on:\n  push:", text)
        self.assertIn("agent/correctness-and-policy-upgrade", text)
        self.assertRegex(
            text,
            r"(?s)paths:\s*\n\s*- data/experiments/"
            r"locked_world800_execution\.json(?:\n|$)",
        )
        self.assertNotIn(
            "locked_world800_execution.template.json\n", text.split("paths:", 1)[1].split("permissions:", 1)[0]
        )
        self.assertIn('test "$GITHUB_RUN_ATTEMPT" = 1', text)
        self.assertIn('test "$EVENT_FORCED" = false', text)
        self.assertIn("git rev-list --parents -n 1 HEAD", text)
        self.assertIn(
            'test "$(git diff-tree --no-commit-id --name-only -r HEAD)"',
            text,
        )
        self.assertIn("execution addendum existed before the launch commit", text)
        self.assertIn('git rev-list --all --count -- "$EXECUTION_PATH"', text)
        self.assertEqual(text.count("if: ${{ github.run_attempt == 1 }}"), 3)
        self.assertGreaterEqual(
            CI_WORKFLOW.read_text(encoding="utf-8").count("fetch-depth: 0"),
            2,
            "CI needs full history to verify the unique addendum-only commit",
        )

    def test_workflow_binds_every_frozen_identity_and_strict_addendum(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        expected_env = {
            "SOURCE_COMMIT": SOURCE_COMMIT,
            "SOURCE_TREE": SOURCE_TREE,
            "PLAN_SHA": PLAN_SHA,
            "PARENT_RESULT_SHA": PARENT_RESULT_SHA,
            "MODEL_SHA": MODEL_SHA,
            "MERGER_SHA": MERGER_SHA,
            "SHARD_VALIDATOR_SHA": VALIDATOR_SHA,
            "EVAL_CFLAGS": CFLAGS,
            "CANDIDATE": CANDIDATE,
            "BASELINE": BASELINE,
            "CANDIDATE_SEED": "202608221501",
            "BASELINE_SEED": "202608221502",
        }
        for name, expected in expected_env.items():
            with self.subTest(name=name):
                self.assertEqual(workflow_env(text, name), expected)
        self.assertIn("object_pairs_hook=unique", text)
        self.assertIn("parse_constant=reject_constant", text)
        self.assertIn('if value != expected:', text)
        self.assertIn('candidate.get("spec") != os.environ["BASELINE"]', text)
        self.assertIn(
            'decision.get("mechanically_selected_tail") != expected_tail', text
        )

    def test_one_build_is_transported_to_source_free_evaluation_workers(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(text.count("make -C source CC=gcc"), 1)
        self.assertIn('test ! -e source/bin/arena', text)
        self.assertIn('test ! -e source/data/champion.bin', text)
        self.assertIn("bin/arena data/champion.bin", text)
        self.assertRegex(
            text,
            rf"(?s)actions/checkout@v4\s*\n\s*with:\s*\n"
            rf"\s*ref: \$\{{\{{ env\.SOURCE_COMMIT \}}\}}\s*\n"
            rf"\s*path: source",
        )
        evaluate = text.split("\n  evaluate:\n", 1)[1].split(
            "\n  merge:\n", 1
        )[0]
        self.assertNotIn("actions/checkout", evaluate)
        self.assertNotIn("make ", evaluate)
        self.assertIn("actions/download-artifact@v4", evaluate)
        self.assertIn('sha256sum evaluator/arena', evaluate)
        self.assertIn('sha256sum evaluator/champion.bin', evaluate)

    def test_exact_50_shards_are_validated_before_the_only_efficacy_look(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        matrix = re.search(
            r"(?s)matrix:\s*\n\s*orientation: "
            r"\[candidate-first, baseline-first\]\s*\n"
            r"\s*start: \[([^\]]+)\]",
            text,
        )
        self.assertIsNotNone(matrix)
        starts = [int(item.strip()) for item in matrix.group(1).split(",")]
        self.assertEqual(starts, STARTS)
        self.assertIn("-n 100 -t 4", text)
        self.assertIn("--pair-start \"$PAIR_START\"", text)
        self.assertIn("--raw-pairs \"${OUT}.jsonl\" --raw-only", text)
        self.assertIn("sha256sum \"${OUT}.jsonl\"", text)
        self.assertIn("wall_s=%e user_s=%U sys_s=%S max_rss_kb=%M exit=%x", text)
        validator = text.index("python3 campaign/tools/validate_actor_shards.py")
        first_merge = text.index("python3 campaign/tools/merge_arena.py block")
        reciprocal = text.index("python3 campaign/tools/merge_arena.py reciprocal")
        self.assertLess(validator, first_merge)
        self.assertLess(first_merge, reciprocal)
        self.assertIn("--pairs-per-shard 100 --rounds 3", text)
        self.assertIn("--expect-pairs 2500", text)
        self.assertEqual(text.count("merge_arena.py reciprocal"), 1)

    def test_raw_backed_merger_applies_exact_plan_gate_without_extra_margin_lcb(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        merger = MERGER.read_text(encoding="utf-8")
        self.assertIn("_remerge_recorded_block(first, \"first\")", merger)
        self.assertIn("_remerge_recorded_block(second, \"second\")", merger)
        self.assertIn("raw_validated and statistical_gate_passed", merger)
        self.assertIn("gate_lower > 0.5", merger)
        self.assertIn('ra["match_score"] > 0.5', merger)
        self.assertIn("score_b_as_candidate > 0.5", merger)
        self.assertIn("margin > 0.0", merger)
        self.assertIn("cap-terminated reciprocal evidence", merger)
        self.assertRegex(
            workflow,
            r"merge_arena\.py reciprocal[\s\\\n]+"
            r"--first merged/candidate-first\.json[\s\\\n]+"
            r"--second merged/baseline-first\.json[\s\\\n]+"
            r"--gate-z 1\.645 --require-positive-margin",
        )
        self.assertNotIn("gate_actor_panel.py", workflow)
        self.assertNotIn("margin_one_sided_lower_bound", workflow)

    def test_provenance_and_final_archive_bind_every_required_artifact(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        provenance = (
            "stage=world800_final;plan=$PLAN_SHA;execution=$EXECUTION_SHA;"
            "parent_result=$PARENT_RESULT_SHA;source=$SOURCE_COMMIT;"
            "tree=$SOURCE_TREE;arena=$ARENA_SHA;model=$MODEL_SHA;threads=4"
        )
        self.assertIn(provenance, text)
        for path in (
            "locked_world800_plan.json",
            "locked_world800_execution.json",
            "role_coherent_result.json",
            "world800.yml",
            "merge_arena.py",
            "validate_actor_shards.py",
            "BUILD_INFO.txt",
            "SHA256SUMS.txt",
        ):
            with self.subTest(path=path):
                self.assertIn(path, text)
        self.assertIn("find downloads merged bindings evaluator", text)
        self.assertIn("sha256sum -c merged/SHA256SUMS.txt", text)


if __name__ == "__main__":
    unittest.main()
