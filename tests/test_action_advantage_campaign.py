"""Fail-closed protocol tests for action-advantage-veto-v1.

The locked plan and workflow are committed before the one execution addendum.
These tests require the execution file to be absent in the source commit.  If
it is later present on the branch, they validate its unique historical
addendum-only launch topology.  They also check the campaign's data firewall,
one-candidate calibration, compile-once transport, and unchanged safety/final
promotion gates.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.action_advantage_campaign import (
    BASELINE_512,
    CANDIDATE_800,
    COMPILER,
    COMPILER_SEMANTIC_VERSION_COMMAND,
    EvidenceError,
    REQUIRED_COMPILER_SEMANTIC_VERSION,
    candidate_prefix,
    expected_execution,
    guard_execution,
    prepare_execution,
)

PLAN = ROOT / "data/experiments/locked_action_advantage_veto_v1_plan.json"
WORKFLOW = ROOT / ".github/workflows/action-advantage-veto-v1.yml"
EXECUTION = (
    ROOT / "data/experiments/locked_action_advantage_veto_v1_execution.json"
)
GENERATOR = ROOT / "tools/action_advantage.c"

BASELINE = BASELINE_512
FULL_TAIL = BASELINE.split(":", 2)[2] + ":0:0:0:1"
PREFIX = candidate_prefix(BASELINE)


def unique(items: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in items:
        if key in out:
            raise ValueError(f"duplicate JSON key {key}")
        out[key] = value
    return out


def strict_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    return value


def assert_workflow_mapping_keys_are_unique(test: unittest.TestCase,
                                            text: str) -> None:
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


def assert_workflow_shell_blocks_parse(test: unittest.TestCase,
                                       text: str) -> None:
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
            body.append(
                line[indent + 2:] if len(line) >= indent + 2 else ""
            )
            index += 1
        blocks.append("\n".join(body) + "\n")
    test.assertGreaterEqual(len(blocks), 9)
    for ordinal, block in enumerate(blocks):
        result = subprocess.run(
            ["bash", "-n"], input=block, text=True,
            capture_output=True, check=False,
        )
        test.assertEqual(
            result.returncode, 0,
            f"shell block {ordinal}: {result.stderr}",
        )


class ActionAdvantageCampaignTests(unittest.TestCase):
    maxDiff = None

    def test_prepare_is_canonical_atomic_and_no_clobber(self) -> None:
        bound = {
            "path": "data/experiments/world800_result.json",
            "sha256": "2" * 64,
            "candidate_actor": CANDIDATE_800,
            "baseline_actor": BASELINE,
            "promotion_gate_passed": False,
            "selected_world_cap": 512,
            "selected_actor": BASELINE,
            "candidate_prefix": candidate_prefix(BASELINE),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "data/experiments/locked_action_advantage_veto_v1_plan.json"
            workflow = root / ".github/workflows/action-advantage-veto-v1.yml"
            output = root / (
                "data/experiments/"
                "locked_action_advantage_veto_v1_execution.json"
            )
            plan.parent.mkdir(parents=True)
            workflow.parent.mkdir(parents=True)
            plan.write_text("{}\n", encoding="utf-8")
            workflow.write_text("name: fixture\n", encoding="utf-8")
            with mock.patch(
                    "tools.action_advantage_campaign.authoritative_inputs",
                    return_value=bound):
                created = prepare_execution(
                    root, output, "0" * 40, "1" * 40)
                snapshot = output.read_bytes()
                self.assertEqual(strict_json(output), created)
                self.assertEqual(output.stat().st_mode & 0o777, 0o644)
                self.assertEqual(list(output.parent.glob(".*.tmp")), [])
                with self.assertRaises(EvidenceError):
                    prepare_execution(root, output, "0" * 40, "1" * 40)
                self.assertEqual(output.read_bytes(), snapshot)
                with self.assertRaises(EvidenceError):
                    prepare_execution(
                        root, root / "execution.json", "0" * 40, "1" * 40)

    def test_world_result_winner_mechanically_constructs_both_actors(self) -> None:
        for passed, selected, world in (
            (False, BASELINE, 512),
            (True, CANDIDATE_800, 800),
        ):
            bound = {
                "path": "data/experiments/world800_result.json",
                "sha256": "2" * 64,
                "candidate_actor": CANDIDATE_800,
                "baseline_actor": BASELINE,
                "promotion_gate_passed": passed,
                "selected_world_cap": world,
                "selected_actor": selected,
                "candidate_prefix": candidate_prefix(selected),
            }
            execution = expected_execution(
                ROOT, "0" * 40, "1" * 40, authoritative=bound)
            self.assertEqual(execution["actors"]["baseline"], selected)
            self.assertEqual(
                execution["actors"]["candidate_prefix"],
                candidate_prefix(selected),
            )
            self.assertEqual(
                execution["authoritative_world800_result"]
                ["selected_world_cap"], world,
            )
            self.assertEqual(execution["build"]["compiler"], COMPILER)
            self.assertEqual(
                execution["build"]["compiler_semantic_version_command"],
                COMPILER_SEMANTIC_VERSION_COMMAND,
            )
        with self.assertRaises(EvidenceError):
            candidate_prefix(CANDIDATE_800.replace(":800:", ":799:", 1))

    def test_plan_locks_general_one_way_method(self) -> None:
        plan = strict_json(PLAN)
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(
            plan["artifact_kind"],
            "locked_action_advantage_veto_v1_actor_campaign",
        )
        self.assertEqual(
            plan["status"],
            "blocked_pending_add_only_authoritative_world800_binding",
        )
        self.assertIsNone(plan["results"])
        source = plan["source_binding"]
        self.assertEqual(
            source["authoritative_world_result"],
            "data/experiments/world800_result.json",
        )
        self.assertIn("exactly recomputes", source["actor_selection"])
        self.assertIn("no manual actor edit", source["actor_selection"])
        self.assertIn("prepare-execution", source["execution_preparation"])
        self.assertIn("guard-execution", source["execution_preparation"])
        method = plan["method"]
        self.assertIn("mechanically selected", method["baseline_actor"])
        self.assertEqual(method["authorized_world_actors"], {
            "512": BASELINE,
            "800": CANDIDATE_800,
        })
        self.assertIn("preserve all 36", method["candidate_template"])
        self.assertIn("{heldout_threshold}", method["candidate_template"])
        self.assertEqual(method["candidate_tail_fields_before_threshold"], 40)
        self.assertEqual(len(FULL_TAIL.split(":")), 40)
        self.assertEqual(PREFIX, (
            "rolloutu4:data/champion.bin:data/champion.bin:"
            "data/models/action_advantage_veto_v1.bin:" + FULL_TAIL + ":"
        ))
        prefix800 = candidate_prefix(CANDIDATE_800)
        self.assertEqual(len(prefix800.rstrip(":").split(":", 4)[4].split(":")), 40)
        self.assertEqual(prefix800.split(":", 5)[4], "800")
        joined = "\n".join(method["invariants"])
        self.assertIn("cannot add, replace, reorder, or widen", joined)
        self.assertIn("top-policy-moves-only", joined)
        self.assertIn("consumes no RNG", joined)
        self.assertIn("sanitized information view", joined)
        self.assertIn("every loaded checkpoint role", joined)
        self.assertIn("direct action ranker", joined)
        self.assertIn("field-41 match-value table", joined)
        self.assertIn("path text alone is insufficient", joined)
        self.assertIn("trunk, value head, and belief head", joined)
        self.assertIn("indistinguishable physical wager copies", joined)
        self.assertEqual(plan["multiplicity"], {
            "rankers": 1,
            "thresholds_entering_safety": 1,
            "candidate_actors": 1,
            "safety_looks": 1,
            "final_looks": 1,
            "unplanned_retries": 0,
            "optional_stopping": False,
        })

        compiler = plan["build"]["compiler"]
        self.assertEqual(compiler, {
            "executable": COMPILER,
            "semantic_version_command": COMPILER_SEMANTIC_VERSION_COMMAND,
            "required_semantic_version": REQUIRED_COMPILER_SEMANTIC_VERSION,
            "build_info_records": [
                "gcc --version first line", "uname -a", "ImageOS",
                "ImageVersion", "RUNNER_OS", "RUNNER_ARCH",
            ],
        })

    def test_plan_locks_generated_data_and_honest_teacher(self) -> None:
        plan = strict_json(PLAN)
        data = plan["development_data"]
        self.assertEqual(data["source"], "generated maintained-actor self-play only")
        self.assertIn("zero-cap", data["proposal_population"])
        self.assertIn("primary-plus-fresh", data["proposal_population"])
        self.assertEqual(data["source_matches"], 64)
        self.assertIn("finish all 64", data["completion_rule"])
        self.assertIn("no proposal cap", data["completion_rule"])
        self.assertEqual(data["label_worlds"], 256)
        self.assertGreaterEqual(data["label_worlds"], 256)
        self.assertEqual(data["label_threads"], 4)
        self.assertEqual(data["teacher_actor"], "policy:data/champion.bin:0:20")
        self.assertIn("exact 20-symmetry champion policy", data["teacher_declaration"])
        self.assertIn("not claimed to equal", data["teacher_declaration"])
        prohibited = "\n".join(data["prohibited_inputs"])
        self.assertIn("data/probes/*.state", prohibited)
        self.assertIn("user-commented", prohibited)
        self.assertIn("human labels", prohibited)
        self.assertIn("no repository checkout", data["enforcement"])

        generator = GENERATOR.read_text(encoding="utf-8")
        self.assertIn("Only generated matches are accepted", generator)
        self.assertNotRegex(generator, r'!strcmp\(a, "--(?:state|probe)"')
        self.assertIn("AA_MIN_LABEL_WORLDS", generator)
        self.assertIn("finish_remaining_match", generator)
        self.assertIn("ss.prefix_confirmed", generator)
        self.assertIn("ss.unfinished_cap_leaves == 0", generator)

    def test_grouped_split_and_heldout_threshold_are_frozen(self) -> None:
        plan = strict_json(PLAN)
        training = plan["training"]
        self.assertIn("source_match_id is an indivisible group", training["split"])
        self.assertEqual(training["validation_permille"], 250)
        self.assertEqual(training["max_validation_kl"], 0.01)
        self.assertEqual(training["max_state_kl"], 0.05)
        self.assertTrue(any("stored champion logit" in rule
                            for rule in training["fail_closed"]))
        self.assertTrue(any("every checkpoint role" in rule and
                            "match-value table" in rule
                            for rule in training["fail_closed"]))
        calibration = plan["threshold_calibration"]
        self.assertEqual(calibration["data"], "heldout development records only")
        self.assertEqual(calibration["predeclared_grid"], [0, 0.1, 0.25, 0.5, 1])
        self.assertIn("exact float32 threshold", calibration["runtime_boundary_rule"])
        self.assertEqual(
            calibration["selection"],
            "among eligible grid points, lexicographically minimize (oracle_regret, mistakes, negative signed_hybrid_sum, negative threshold)",
        )
        self.assertIn("at least 20 percent", calibration["eligibility"])
        self.assertIn("strictly positive", calibration["eligibility"])
        self.assertIn("before any safety game", calibration["freeze"])

        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("validation_proposals < 20", text)
        self.assertIn('"source_matches_completed": 64', text)
        self.assertIn('"proposal_cap": 0', text)
        self.assertIn('fnv1a64(b"ABSENT\\0")', text)
        self.assertIn('"format_version": 2', text)
        self.assertIn('"maintained_ranker_net_hash": absent_fnv', text)
        self.assertIn('"maintained_match_value_hash": absent_fnv', text)
        self.assertIn('"reroot_ranker_net_hash": absent_fnv', text)
        self.assertIn('"reroot_match_value_hash": absent_fnv', text)
        self.assertIn("champion_bytes[24:]", text)
        self.assertIn("generated-record header drift", text)
        self.assertIn('metrics.get("training_gate_passed") is not True', text)
        self.assertNotIn('metrics.get("promotion_gate_passed")', text)
        self.assertIn("not final_loss < initial_loss", text)
        self.assertIn('row["invalid_scores"] != 0', text)
        self.assertIn('row.get("runtime_threshold") != runtime_threshold', text)
        self.assertIn('row["retained"] * 5 >= validation_proposals', text)
        self.assertIn('row["signed_hybrid_sum"] > 0.0', text)
        self.assertIn('item[1]["oracle_regret"]', text)
        self.assertIn('-item[1]["threshold"]', text)
        self.assertIn("threshold_and_actor_frozen_before_safety", text)
        self.assertRegex(
            text,
            r"(?s)  safety_evaluate:.*?needs: train_and_freeze.*?"
            r"name: action-advantage-v1-evaluator",
        )

    def test_workflow_is_one_addendum_compile_once_transport(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert_workflow_mapping_keys_are_unique(self, text)
        assert_workflow_shell_blocks_parse(self, text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("continue-on-error", text)
        self.assertIn("on:\n  push:", text)
        self.assertIn(
            "data/experiments/locked_action_advantage_veto_v1_execution.json",
            text,
        )
        self.assertIn(
            "data/experiments/world800_result.json", text,
        )
        self.assertNotRegex(text, r"(?m)^  (?:BASELINE|CANDIDATE_PREFIX):")
        self.assertIn(
            "tools/action_advantage_campaign.py guard-execution", text,
        )
        self.assertIn(
            "baseline: ${{ steps.guard.outputs.baseline }}", text,
        )
        self.assertIn(
            "CANDIDATE_PREFIX: ${{ needs.preflight.outputs.candidate_prefix }}",
            text,
        )
        self.assertIn(
            'test "$(git diff-tree --no-commit-id --name-status -r HEAD)"',
            text,
        )
        self.assertIn(
            "concurrency:\n"
            "  group: locked-action-advantage-veto-v1-${{ github.ref }}\n"
            "  cancel-in-progress: false",
            text,
        )
        for token in (
            'test "$EVENT_FORCED" = false',
            'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"',
            'test "$EVENT_AFTER" = "$GITHUB_SHA"',
            'test "$(git rev-list --parents -n 1 HEAD | wc -w)" -eq 2',
            'test "$EVENT_BEFORE" = "$SOURCE_COMMIT"',
            'git cat-file -e "HEAD^:$WORLD_RESULT_PATH"',
        ):
            self.assertIn(token, text)
        pinned = {
            "actions/checkout":
                "11bd71901bbe5b1630ceea73d27597364c9af683",
            "actions/upload-artifact":
                "ea165f8d65b6e75b540449e92b4886f43607fa02",
            "actions/download-artifact":
                "d3f86a106a0bac45b974a628896c90dbdf5c8093",
        }
        uses = re.findall(
            r"(?m)^\s*- uses: (actions/[a-z-]+)@([0-9a-f]{40})$", text,
        )
        self.assertEqual(len(uses), text.count("uses: actions/"))
        self.assertEqual(
            {name for name, _ in uses}, set(pinned),
        )
        for name, revision in uses:
            self.assertEqual(revision, pinned[name])
        preflight_header = text.split("\n  preflight:", 1)[1].split(
            "\n    steps:", 1
        )[0]
        self.assertNotIn("if:", preflight_header)
        self.assertIn('test "$GITHUB_RUN_ATTEMPT" = 1', text)
        self.assertIn('git -C campaign archive HEAD^ | tar -x -C source', text)
        self.assertIn("Compile the evaluator and training tools exactly once", text)
        self.assertNotIn("COMPILER_ID", text)
        self.assertIn("COMPILER_EXECUTABLE: gcc", text)
        self.assertIn(
            "COMPILER_SEMANTIC_VERSION_COMMAND: "
            "'gcc -dumpfullversion -dumpversion'",
            text,
        )
        self.assertIn("REQUIRED_COMPILER_SEMANTIC_VERSION: '13.3.0'", text)
        self.assertIn(
            'test "$(gcc -dumpfullversion -dumpversion)" =', text,
        )
        self.assertNotIn('test "$(gcc --version | head -1)" =', text)
        for record in (
            'echo "compiler_banner=$(gcc --version | head -1)"',
            'echo "uname=$(uname -a)"',
            'echo "runner_image_os=$ImageOS"',
            'echo "runner_image_version=$ImageVersion"',
            'echo "runner_os=$RUNNER_OS"',
            'echo "runner_arch=$RUNNER_ARCH"',
        ):
            self.assertIn(record, text)
        self.assertIn("(cd source && ./bin/test_action_ranker", text)
        self.assertIn("./bin/test_action_advantage)", text)
        self.assertIn("(cd transport && sha256sum -c SHA256SUMS.txt)", text)
        self.assertIn("(cd evaluator && sha256sum -c SHA256SUMS.txt)", text)
        self.assertIn("selected.json PREFLIGHT_BUILD_INFO.txt", text)
        self.assertIn(
            "no_eligible_threshold_fail_closed_before_safety", text,
        )
        self.assertIn(
            'stream.write("eligible=false\\n")', text,
        )
        self.assertIn(
            "frozen_action_advantage_veto_v1_no_eligible_candidate", text,
        )
        self.assertIn("dev/selection-failure.json", text)
        self.assertIn("dev/DEVELOPMENT_SHA256SUMS.txt", text)
        self.assertIn(
            "Fail closed after preserving no-candidate development evidence",
            text,
        )
        self.assertRegex(
            text,
            r"(?s)- uses: actions/upload-artifact@[0-9a-f]{40}\n"
            r"\s+if: \$\{\{ always\(\) \}\}\n"
            r"\s+with:\n\s+name: action-advantage-v1-development-evidence",
        )
        self.assertRegex(
            text,
            r"(?s)name: Package the one candidate and immutable evaluator\n"
            r"\s+if: \$\{\{ success\(\) && "
            r"steps\.select\.outputs\.eligible == 'true' \}\}",
        )
        self.assertEqual(text.count("timeout-minutes: 360"), 3)
        self.assertEqual(text.count("/usr/bin/time"), 4)
        self.assertEqual(text.count("wall_s=%e user_s=%U sys_s=%S "), 4)
        self.assertIn('find downloads merged bindings -type f ! -name SHA256SUMS.txt', text)
        safety_eval = text.split("\n  safety_evaluate:", 1)[1].split(
            "\n  safety_merge:", 1
        )[0]
        self.assertEqual(
            safety_eval.count('sha256sum "${OUT}.jsonl" > "${OUT}.sha256"'),
            1,
        )

        after_preflight = text.split("\n  train_and_freeze:", 1)[1]
        self.assertNotRegex(after_preflight, r"(?m)^\s*(?:make|gcc|cc|clang)\b")
        self.assertNotIn("actions/checkout", after_preflight)
        self.assertIn("test ! -e data/probes", after_preflight)
        self.assertIn("find . -name '*.state'", after_preflight)
        self.assertIn("--matches 64 --worlds 256 --label-threads 4", after_preflight)
        self.assertNotIn("--max-proposals", after_preflight)
        self.assertIn('--reroot-actor "$TEACHER"', after_preflight)
        self.assertGreaterEqual(after_preflight.count('--actor "$BASELINE"'), 2)
        if EXECUTION.exists():
            execution = strict_json(EXECUTION)
            source = execution.get("source_parent_commit")
            self.assertRegex(source, r"^[0-9a-f]{40}$")
            self.assertEqual(
                execution["source_parent_tree"],
                subprocess.check_output(
                    ["git", "rev-parse", f"{source}^{{tree}}"],
                    cwd=ROOT, text=True,
                ).strip(),
            )
            self.assertEqual(
                execution["plan"]["sha256"],
                hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                execution["workflow"]["sha256"],
                hashlib.sha256(WORKFLOW.read_bytes()).hexdigest(),
            )
            path = EXECUTION.relative_to(ROOT).as_posix()
            additions = subprocess.check_output(
                ["git", "log", "--all", "--format=%H", "--diff-filter=A",
                 "--", path],
                cwd=ROOT, text=True,
            ).splitlines()
            self.assertEqual(len(additions), 1)
            launch = additions[0]
            self.assertEqual(
                subprocess.check_output(
                    ["git", "rev-parse", f"{launch}^"], cwd=ROOT, text=True,
                ).strip(),
                source,
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "diff-tree", "--no-commit-id", "--name-status",
                     "-r", launch],
                    cwd=ROOT, text=True,
                ).strip(),
                f"A\t{path}",
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "show", f"{launch}:{path}"], cwd=ROOT,
                ),
                EXECUTION.read_bytes(),
            )
            expected, bound = guard_execution(
                ROOT, EXECUTION,
                execution["source_parent_commit"],
                execution["source_parent_tree"],
            )
            self.assertEqual(expected, execution)
            self.assertEqual(
                bound["selected_actor"], execution["actors"]["baseline"]
            )

    def test_unchanged_actor_gates_and_fresh_namespaces(self) -> None:
        plan = strict_json(PLAN)
        firewall = plan["seed_firewall"]
        self.assertEqual(firewall["development_namespace"], "20260903")
        self.assertEqual(
            plan["development_data"]["generator_seed"], "202609030101"
        )
        self.assertEqual(plan["training"]["split_seed"], "202609030201")
        self.assertEqual(
            firewall["burned_development_seeds"],
            {
                "generator_seed": "202609010101",
                "split_seed_used_by_smoke": "1",
                "reason": (
                    "a two-match local implementation smoke generated and "
                    "inspected records in the original namespace before launch; "
                    "the entire original development namespace is excluded from "
                    "campaign evidence"
                ),
            },
        )
        self.assertEqual(firewall["safety_final_namespace"], "20260902")
        seeds = [
            firewall["safety"]["candidate_first"],
            firewall["safety"]["baseline_first"],
            firewall["final"]["candidate_first"],
            firewall["final"]["baseline_first"],
        ]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(seed.startswith("20260902") for seed in seeds))
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("DEVELOPMENT_SEED: '202609030101'", text)
        self.assertIn("SPLIT_SEED: '202609030201'", text)
        self.assertNotIn("DEVELOPMENT_SEED: '202609010101'", text)

        safety = plan["safety_screen"]
        self.assertEqual(safety["pairs_per_orientation"], 200)
        self.assertEqual(safety["matches_total"], 800)
        self.assertEqual(safety["pairs_per_shard"], 20)
        self.assertEqual(safety["pair_starts"], list(range(0, 200, 20)))
        self.assertEqual(safety["gate"], [
            "equal-weight reciprocal combined candidate match score >= 0.5",
            "equal-weight reciprocal combined candidate point margin > 0",
            "candidate match score in each reciprocal orientation after inversion >= 0.475",
            "zero capped rounds, gaps, overlaps, incomplete footers, malformed rows, hash failures, provenance drift, or operational errors",
        ])

        final = plan["final_promotion"]
        self.assertTrue(final["execute_only_if_safety_passes"])
        self.assertEqual(final["pairs_per_orientation"], 2500)
        self.assertEqual(final["matches_total"], 10000)
        self.assertEqual(final["pairs_per_shard"], 100)
        self.assertEqual(final["pair_starts"], list(range(0, 2500, 100)))
        self.assertEqual(final["confidence_z"], 1.645)
        self.assertEqual(final["promotion_gate"], [
            "combined candidate match score - 1.645 * orientation-stratified pair-clustered SE > 0.5",
            "combined candidate point margin - 1.645 * orientation-stratified pair-clustered SE > 0",
            "candidate match-score point estimate > 0.5 in each reciprocal orientation after inversion",
            "zero capped rounds, gaps, overlaps, incomplete footers, malformed rows, hash failures, provenance drift, or operational errors",
        ])

        self.assertIn("needs.safety_merge.outputs.passed == 'true'", text)
        self.assertIn("needs.safety_merge.result == 'success'", text)
        self.assertIn("needs.final_evaluate.result == 'success'", text)
        self.assertEqual(text.count("--mode safety"), 1)
        self.assertEqual(text.count("--mode final"), 1)
        self.assertIn("--pairs-per-orientation 200", text)
        self.assertIn("--pairs-per-orientation 2500", text)
        self.assertEqual(text.count("--gate-z 1.645"), 4)

    def test_complete_shards_precede_each_gate(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        safety = text.split("\n  safety_merge:", 1)[1].split(
            "\n  final_evaluate:", 1
        )[0]
        final = text.split("\n  final_merge:", 1)[1]
        for section, pairs, starts in (
            (safety, 200, "0,20,40,60,80,100,120,140,160,180"),
            (
                final,
                2500,
                "0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400",
            ),
        ):
            validate_at = section.index("validate_actor_shards.py")
            merge_at = section.index("merge_arena.py reciprocal")
            gate_at = section.index("gate_actor_panel.py")
            self.assertLess(validate_at, merge_at)
            self.assertLess(merge_at, gate_at)
            self.assertIn(f"--expect-pairs {pairs}", section)
            self.assertIn(f"--starts {starts}", section)


if __name__ == "__main__":
    unittest.main()
