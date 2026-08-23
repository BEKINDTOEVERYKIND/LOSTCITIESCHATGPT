"""Fail-closed contracts for the controller-bound match-value campaign."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import tempfile
import unittest

from tools.match_value_campaign import (
    BASELINE_512,
    BUILD_PROFILE_HEX,
    CANDIDATE_800,
    COMPILER,
    COMPILER_SEMANTIC_VERSION_COMMAND,
    MODEL_SHA256,
    REQUIRED_COMPILER_SEMANTIC_VERSION,
    SOURCE_FILES,
    TABLE_CONTROLLER_WORDS,
    TABLE_ROLE_CYCLE,
    TABLE_SAMPLES,
    TABLE_SEED,
    TIE_PRIORITY,
    VARIANTS,
    EvidenceError,
    build_actors,
    expected_execution,
    final_gate,
    guard_execution,
    inspect_table,
    stage1_selection,
    stage2_gate,
    strict_json,
    table_manifest,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/experiments/match_value_variant_plan.json"
WORKFLOW = ROOT / ".github/workflows/match-value-variant.yml"
TEMPLATE = (
    ROOT / "data/experiments/locked_match_value_variant_execution.template.json"
)
EXECUTION = (
    ROOT / "data/experiments/locked_match_value_variant_execution.json"
)


def workflow_mapping_keys_are_unique(test: unittest.TestCase, text: str) -> None:
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
        assert match is not None
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


def panel(
    pairs: int,
    first_quarters: int,
    second_quarters: int,
    first_margin: int,
    second_margin: int,
    *,
    score: float = 0.51,
    score_se: float = 0.005,
    margin: float = 0.5,
    margin_se: float = 0.2,
    orientations: tuple[float, float] = (0.51, 0.51),
    caps: int = 0,
) -> dict:
    def sufficient(quarters: int, margin_sum: int, block_caps: int) -> dict:
        return {
            "pairs": pairs,
            "score_quarters_sum": quarters,
            "margin_sum": margin_sum,
            "capped_rounds": block_caps,
        }

    return {
        "blocks": [
            {"sufficient_statistics": sufficient(
                first_quarters, first_margin, caps)},
            {"sufficient_statistics": sufficient(
                second_quarters, second_margin, 0)},
        ],
        "raw_input_validation": {"status": "validated"},
        "candidate_result": {
            "match_score": score,
            "match_score_pair_clustered_se": score_se,
            "margin_per_game": margin,
            "margin_pair_clustered_se": margin_se,
            "orientation_match_scores": list(orientations),
            "capped_rounds": caps,
        },
    }


def fnv1a(snapshot: bytes) -> int:
    value = 1469598103934665603
    for byte in snapshot:
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def table_bytes(projected: bool, *, corrupt_profile: bool = False) -> bytes:
    r1_count, r2_count = 2361, 4721
    header = bytearray(128)
    header[:8] = b"LCMVAL1\0"
    for offset, value in (
        (8, 1), (12, 128), (16, TABLE_SAMPLES), (20, r1_count),
        (24, r2_count), (28, 150),
    ):
        struct.pack_into("<I", header, offset, value)
    struct.pack_into("<Q", header, 32, 0x123456789ABCDEF0)
    for index, value in enumerate(TABLE_CONTROLLER_WORDS):
        struct.pack_into("<I", header, 40 + 4 * index, value)
    struct.pack_into("<Q", header, 84, int(TABLE_SEED))
    struct.pack_into("<d", header, 92, 0.25)
    struct.pack_into("<d", header, 100, 0.5)
    struct.pack_into("<I", header, 108, TABLE_ROLE_CYCLE)
    struct.pack_into("<I", header, 112, 1)
    struct.pack_into("<I", header, 116, int(projected))
    struct.pack_into("<I", header, 120, 1)
    struct.pack_into(
        "<I", header, 124,
        int(BUILD_PROFILE_HEX, 16) + int(corrupt_profile),
    )
    values: list[float] = []
    for length in (r1_count, r2_count):
        not_start = [20.0 * (index / (length - 1) - 0.5)
                     for index in range(length)]
        if not projected:
            not_start[length // 3] += 0.125
        starts = [-value for value in reversed(not_start)]
        values.extend(not_start)
        values.extend(starts)
    body = bytes(header) + struct.pack(f"<{len(values)}d", *values)
    return body + struct.pack("<Q", fnv1a(body))


class MatchValueCampaignTests(unittest.TestCase):
    maxDiff = None

    def test_plan_is_the_exact_two_by_two_staged_protocol(self) -> None:
        value = strict_json(PLAN)
        validate_plan(value)
        self.assertEqual(value["schema"], 3)
        self.assertEqual(
            value["status"], "blocked_pending_add_only_world800_binding"
        )
        self.assertEqual(
            value["scope"], "development_then_reserved_final_only"
        )
        self.assertEqual(
            set(value["factorial_candidates"]), set(VARIANTS)
        )
        self.assertEqual(
            value["stage_1_factorial_screen"]["total_mirrored_pairs"], 1600
        )
        self.assertEqual(
            value["stage_2_development_confirmation"]
            ["total_mirrored_pairs"], 1000
        )
        self.assertEqual(
            value["locked_final_test_reservation"]["total_mirrored_pairs"],
            5000,
        )
        self.assertEqual(value["artifact_build"]["compiler"], {
            "executable": COMPILER,
            "semantic_version_command": COMPILER_SEMANTIC_VERSION_COMMAND,
            "required_semantic_version": REQUIRED_COMPILER_SEMANTIC_VERSION,
            "build_info_records": [
                "gcc --version first line", "uname -a", "ImageOS",
                "ImageVersion", "RUNNER_OS", "RUNNER_ARCH",
            ],
        })
        self.assertIn(
            "95% one-sided lower confidence bound",
            value["locked_final_test_reservation"]["promotion_gate"],
        )
        self.assertIn(
            "equivalent 90% two-sided confidence interval",
            value["locked_final_test_reservation"]["promotion_gate"],
        )
        self.assertIn(
            "never edits a branch",
            value["execution_protocol"]["promotion_behavior"],
        )
        self.assertIn(
            "complete stage",
            value["execution_protocol"]["inspection_firewall"],
        )

    def test_execution_is_absent_and_template_only_until_winner_exists(self) -> None:
        template = strict_json(TEMPLATE)
        self.assertEqual(
            template["artifact_kind"],
            "locked_match_value_variant_execution",
        )
        self.assertEqual(template["source_parent_commit"], "__SOLE_PARENT_COMMIT__")
        self.assertEqual(
            set(template["source_files_sha256"]), set(SOURCE_FILES)
        )
        self.assertEqual(
            template["authoritative_world800_result"]["path"],
            "data/experiments/world800_result.json",
        )
        self.assertEqual(
            template["authoritative_world800_result"]["candidate_actor"],
            CANDIDATE_800,
        )
        self.assertEqual(
            template["authoritative_world800_result"]["baseline_actor"],
            BASELINE_512,
        )
        self.assertEqual(template["build"]["expected_build_profile_hex"],
                         BUILD_PROFILE_HEX)
        self.assertEqual(template["build"]["compiler"], COMPILER)
        self.assertEqual(
            template["build"]["compiler_semantic_version_command"],
            COMPILER_SEMANTIC_VERSION_COMMAND,
        )
        self.assertEqual(
            template["build"]["required_compiler_semantic_version"],
            REQUIRED_COMPILER_SEMANTIC_VERSION,
        )
        expected = expected_execution(
            ROOT, "0" * 40, "1" * 40, "2" * 64, False, BASELINE_512, 512,
        )
        self.assertEqual(expected["build"]["compiler"], COMPILER)
        self.assertEqual(
            expected["build"]["compiler_semantic_version_command"],
            COMPILER_SEMANTIC_VERSION_COMMAND,
        )
        self.assertEqual(
            expected["build"]["required_compiler_semantic_version"],
            REQUIRED_COMPILER_SEMANTIC_VERSION,
        )
        self.assertEqual(template["build"]["model"]["sha256"], MODEL_SHA256)
        self.assertIsNone(template["results"])

        tracked = subprocess.run(
            ["git", "cat-file", "-e", "HEAD:" + EXECUTION.relative_to(ROOT).as_posix()],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if not tracked:
            self.assertFalse(EXECUTION.exists())
            return

        relative = EXECUTION.relative_to(ROOT).as_posix()
        launches = subprocess.check_output(
            ["git", "log", "--all", "--diff-filter=A", "--format=%H", "--", relative],
            cwd=ROOT, text=True,
        ).splitlines()
        self.assertEqual(len(launches), 1)
        launch = launches[0]
        parents = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", launch],
            cwd=ROOT, text=True,
        ).split()
        self.assertEqual(len(parents), 2)
        parent = parents[1]
        self.assertEqual(
            subprocess.check_output(
                ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", launch],
                cwd=ROOT, text=True,
            ),
            f"A\t{relative}\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "parent.tar"
            with archive.open("wb") as stream:
                subprocess.run(
                    ["git", "archive", parent], cwd=ROOT,
                    check=True, stdout=stream,
                )
            extracted = Path(directory) / "source"
            extracted.mkdir()
            with tarfile.open(archive) as stream:
                stream.extractall(extracted, filter="data")
            execution_path = extracted / relative
            execution_path.parent.mkdir(parents=True, exist_ok=True)
            execution_path.write_bytes(subprocess.check_output(
                ["git", "show", f"{launch}:{relative}"], cwd=ROOT,
            ))
            tree = subprocess.check_output(
                ["git", "rev-parse", f"{parent}^{{tree}}"], cwd=ROOT,
                text=True,
            ).strip()
            guard_execution(extracted, execution_path, parent, tree)

    def test_workflow_only_addendum_triggers_and_compiles_builds_once(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        workflow_mapping_keys_are_unique(self, text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("pull_request", text)
        self.assertNotIn("continue-on-error", text)
        self.assertIn("permissions:\n  contents: read", text)
        trigger = text.split("paths:", 1)[1].split("permissions:", 1)[0]
        self.assertIn(
            "data/experiments/locked_match_value_variant_execution.json",
            trigger,
        )
        self.assertNotIn(".template.json", trigger)
        for token in (
            'test "$GITHUB_RUN_ATTEMPT" = 1',
            'test "$EVENT_FORCED" = false',
            "git rev-list --parents -n 1 HEAD",
            "git diff-tree --no-commit-id --name-status -r HEAD",
            'git rev-list --all --count -- "$EXECUTION_PATH"',
            "git -C campaign archive HEAD^ | tar -x -C source",
        ):
            self.assertIn(token, text)
        self.assertEqual(text.count("./bin/build_match_value \\"), 1)
        self.assertIn("--samples 16000 --threads 8 --seed 7331001", text)
        self.assertIn("--playout-symmetries 20", text)
        self.assertIn("variant=isotonic", text)
        self.assertIn("variant=raw", text)
        self.assertIn("role_cycle=400 role_balance=complete", text)
        self.assertIn("abi=1 build=0030d23b", text)
        self.assertNotIn("COMPILER_ID", text)
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
        after_preflight = text.split("\n  stage1_evaluate:\n", 1)[1]
        self.assertNotIn("actions/checkout", after_preflight)
        self.assertNotRegex(after_preflight, r"(?m)^\s*(?:make|gcc|cc|clang)\b")
        self.assertNotIn("git push", text)
        self.assertNotIn("contents: write", text)

    def test_workflow_has_complete_opaque_stages_and_reserved_gate(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        stage1_eval = text.split("\n  stage1_evaluate:\n", 1)[1].split(
            "\n  stage1_select:\n", 1
        )[0]
        self.assertIn("variant: [R14, P14, R0, P0]", stage1_eval)
        self.assertIn("orientation: [candidate-first, baseline-first]", stage1_eval)
        self.assertIn("start: [0, 100]", stage1_eval)
        self.assertIn("--raw-only", stage1_eval)
        self.assertIn('STEM="$ORIENTATION-$START"', stage1_eval)
        self.assertNotIn("merge_arena.py", stage1_eval)

        selection = text.split("\n  stage1_select:\n", 1)[1].split(
            "\n  stage2_evaluate:\n", 1
        )[0]
        self.assertLess(
            selection.index("validate_actor_shards.py"),
            selection.index("merge_arena.py block"),
        )
        self.assertEqual(selection.count("select-stage1"), 1)
        self.assertIn('SOURCE="downloads/match-value-stage1-$VARIANT-', selection)
        self.assertIn('--directory "panels/$VARIANT"', selection)
        self.assertIn("--starts 0,100 --pairs-per-shard 100", selection)
        self.assertIn("--expect-pairs 200", selection)

        stage2_eval = text.split("\n  stage2_evaluate:\n", 1)[1].split(
            "\n  stage2_gate:\n", 1
        )[0]
        self.assertIn("start: [0, 100, 200, 300, 400]", stage2_eval)
        self.assertIn("needs.stage1_select.outputs.selected_actor", stage2_eval)
        self.assertNotIn("R14, P14, R0, P0", stage2_eval)
        stage2 = text.split("\n  stage2_gate:\n", 1)[1].split(
            "\n  final_evaluate:\n", 1
        )[0]
        self.assertLess(
            stage2.index("validate_actor_shards.py"),
            stage2.index("merge_arena.py block"),
        )
        self.assertIn("--expect-pairs 500", stage2)
        self.assertEqual(stage2.count("gate-stage2"), 1)

        final_eval = text.split("\n  final_evaluate:\n", 1)[1].split(
            "\n  final_gate:\n", 1
        )[0]
        self.assertIn("needs.stage2_gate.outputs.passed == 'true'", final_eval)
        self.assertIn("start: [0, 100, 200, 300, 400, 500", final_eval)
        final = text.split("\n  final_gate:\n", 1)[1]
        self.assertLess(
            final.index("validate_actor_shards.py"),
            final.index("merge_arena.py block"),
        )
        self.assertIn("--expect-pairs 2500", final)
        self.assertEqual(final.count("gate-final"), 1)
        self.assertIn("No repository promotion was performed.", final)
        for evidence in (
            "table-manifest.json", "post-build-manifest.json", "actors.json",
            "stage1-selection.json", "stage2-decision.json",
            "final-decision.json", "SHA256SUMS.txt",
        ):
            self.assertIn(evidence, text)

    def test_actor_construction_changes_only_objective_phase_and_table(self) -> None:
        value = build_actors(
            BASELINE_512, 512,
            "tables/winner-o0-16000-raw.lcmv",
            "tables/winner-o0-16000-isotonic.lcmv",
        )
        self.assertEqual(value["actors"]["legacy"], BASELINE_512)
        self.assertEqual(set(value["actors"]), {"legacy", *VARIANTS})
        for name in VARIANTS:
            fields = value["actors"][name].split(":")
            self.assertEqual(fields[:3], ["rolloutu2", "data/champion.bin",
                                          "data/champion.bin"])
            tail = fields[3:]
            self.assertEqual(len(tail), 42)
            self.assertEqual(tail[0], "512")
            self.assertEqual(tail[19], "512")
            self.assertEqual(tail[5], "14" if name.endswith("14") else "0")
            self.assertEqual(tail[8], "3")
            self.assertEqual(tail[40], "0")
            self.assertIn("isotonic" if name.startswith("P") else "raw",
                          tail[41])
        with self.assertRaises(EvidenceError):
            build_actors(BASELINE_512, 800, "raw", "projected")

    def test_table_pair_parser_checks_profile_roles_pairing_and_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.lcmv"
            projected = root / "projected.lcmv"
            raw.write_bytes(table_bytes(False))
            projected.write_bytes(table_bytes(True))
            value = table_manifest(raw, projected)
            self.assertEqual(
                value["status"],
                "complete_valid_single_transition_generation",
            )
            self.assertFalse(value["raw"]["isotonic_projected"])
            self.assertTrue(value["projected"]["isotonic_projected"])
            self.assertEqual(
                value["raw"]["controller"], value["projected"]["controller"]
            )
            self.assertNotEqual(value["raw"]["sha256"],
                                value["projected"]["sha256"])

            broken = root / "broken.lcmv"
            snapshot = bytearray(raw.read_bytes())
            snapshot[-1] ^= 1
            broken.write_bytes(snapshot)
            with self.assertRaises(EvidenceError):
                inspect_table(broken, False)
            wrong_profile = root / "wrong-profile.lcmv"
            wrong_profile.write_bytes(table_bytes(False, corrupt_profile=True))
            with self.assertRaises(EvidenceError):
                inspect_table(wrong_profile, False)

    def test_stage1_selection_is_exact_complete_and_deterministic(self) -> None:
        panels = {
            "R14": panel(200, 410, 390, 20, 0),
            "P14": panel(200, 410, 390, 20, 0),
            "R0": panel(200, 405, 395, 40, 0),
            "P0": panel(200, 390, 410, 100, 0),
        }
        actors = {name: f"actor-{name}" for name in VARIANTS}
        digests = {name: hashlib.sha256(name.encode()).hexdigest()
                   for name in VARIANTS}
        value = stage1_selection(panels, actors, digests)
        self.assertEqual(value["selected_variant"], "P14")
        self.assertEqual(value["selected_actor"], "actor-P14")
        self.assertFalse(value["promotion_claim"])
        self.assertEqual(TIE_PRIORITY[0], "P14")
        with self.assertRaises(EvidenceError):
            stage1_selection({key: panels[key] for key in VARIANTS[:-1]},
                             actors, digests)
        capped = json.loads(json.dumps(panels))
        capped["R0"]["blocks"][0]["sufficient_statistics"]["capped_rounds"] = 1
        with self.assertRaises(EvidenceError):
            stage1_selection(capped, actors, digests)

    def test_stage2_gate_uses_strict_reciprocal_integer_boundaries(self) -> None:
        passing = panel(500, 1001, 999, 1, 0)
        decision = stage2_gate(passing)
        self.assertTrue(decision["passed"])
        self.assertFalse(decision["promotion_claim"])
        equality = panel(500, 1000, 1000, 1, 0)
        self.assertFalse(stage2_gate(equality)["passed"])
        weak_orientation = panel(500, 1001, 1000, 1, 0)
        self.assertFalse(stage2_gate(weak_orientation)["passed"])
        zero_margin = panel(500, 1001, 999, 0, 0)
        self.assertFalse(stage2_gate(zero_margin)["passed"])

    def test_final_gate_is_pair_clustered_orientation_stratified_and_exact(self) -> None:
        passing = panel(
            2500, 5100, 4900, 100, 0,
            score=0.51, score_se=0.005, margin=0.1,
            orientations=(0.51, 0.51),
        )
        decision = final_gate(passing)
        self.assertTrue(decision["promotion_gate_passed"])
        self.assertFalse(decision["repository_promotion_performed"])
        self.assertAlmostEqual(
            decision["candidate_result"]["match_score_lower_bound"],
            0.51 - 1.645 * 0.005,
        )
        self.assertIn("mirrored-pair clusters", decision["estimator"])
        self.assertEqual(
            decision["confidence_bound"],
            "95% one-sided lower bound; equivalently, the lower endpoint "
            "of a 90% two-sided confidence interval",
        )

        lcb_fail = panel(
            2500, 5100, 4900, 100, 0,
            score=0.505, score_se=0.004, margin=0.1,
            orientations=(0.51, 0.51),
        )
        self.assertFalse(final_gate(lcb_fail)["promotion_gate_passed"])
        orientation_fail = panel(
            2500, 5100, 4900, 100, 0,
            score=0.52, score_se=0.001, margin=0.1,
            orientations=(0.5, 0.54),
        )
        self.assertFalse(final_gate(orientation_fail)["promotion_gate_passed"])
        margin_fail = panel(
            2500, 5100, 4900, 100, 0,
            score=0.52, score_se=0.001, margin=0.0,
            orientations=(0.52, 0.52),
        )
        self.assertFalse(final_gate(margin_fail)["promotion_gate_passed"])
        with self.assertRaises(EvidenceError):
            final_gate(passing, math.nextafter(1.645, math.inf))


if __name__ == "__main__":
    unittest.main()
