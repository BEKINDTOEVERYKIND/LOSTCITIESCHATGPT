"""Focused contracts for the explicit commented-ply diagnostic."""

from __future__ import annotations

import importlib.util
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "audit_commented_plies.py"
SPEC = importlib.util.spec_from_file_location("audit_commented_plies", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class CommentedPlyAuditTests(unittest.TestCase):
    POLICY_ACTOR = "policy:data/champion.bin:0:20"
    SMOKE_BASE_SEED = 202610090001
    SMOKE_CHANGED_SEED = 202610090002
    SMOKE_P13_SEED = 202610090013
    SMOKE_P44_SEED = 202610090044
    SMOKE_CAP_SEED = 202610090099

    @staticmethod
    def with_complete_deck(text: str, *, reverse: bool = False) -> str:
        full = [
            f"{suit}{rank}"
            for suit in "YBWGR"
            for rank in ("x", "x", "x", "2", "3", "4", "5", "6",
                         "7", "8", "9", "10")
        ]
        remaining = Counter(full)
        for line in text.splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0].startswith("hand"):
                visible = fields[1:]
            elif fields[0] == "exp":
                visible = fields[3:]
            elif fields[0] == "pile":
                visible = fields[2:]
            else:
                continue
            for card in visible:
                remaining[card] -= 1
                if remaining[card] < 0:
                    raise AssertionError(f"invalid fixture card {card}")
        deck: list[str] = []
        for card in full:
            if remaining[card] > 0:
                deck.append(card)
                remaining[card] -= 1
        if reverse:
            deck.reverse()
        return text.rstrip() + "\ndeck " + " ".join(deck) + "\n"

    def helper(
        self, state: Path, *extra: str, seed: int = SMOKE_BASE_SEED,
        worlds: int = 2,
    ) -> dict:
        command = [
            str(ROOT / "bin" / "commented_ply_eval"),
            "--state", str(state),
            "--actor", self.POLICY_ACTOR,
            "--net", "data/champion.bin",
            "--seed", str(seed),
            "--worlds", str(worlds),
            "--symmetries", "20",
            *extra,
        ]
        result = subprocess.run(
            command, cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return json.loads(result.stdout)

    def test_inventory_is_exactly_the_17_explicit_comments(self) -> None:
        expected = {
            2214615196: {3, 4, 8, 10, 12, 13, 16, 20},
            5726968372613385: {14, 15, 17, 32},
            725402798: {21, 22, 23, 25},
            95647345759839: {44},
        }
        actual: dict[int, set[int]] = {}
        for case in audit.CASES:
            actual.setdefault(case.source_seed, set()).add(case.ply)
            self.assertTrue((ROOT / case.state).is_file())
            self.assertNotIn("train", case.state.lower())
        self.assertEqual(len(audit.CASES), 17)
        self.assertEqual(actual, expected)
        self.assertEqual(
            sum(case.min_worlds for case in audit.CASES), 18_432
        )
        production_seeds = {case.audit_seed for case in audit.CASES}
        self.assertEqual(len(production_seeds), 17)
        self.assertTrue(all(
            str(seed).startswith("20261001") for seed in production_seeds
        ))
        smoke_seeds = {
            self.SMOKE_BASE_SEED, self.SMOKE_CHANGED_SEED,
            self.SMOKE_P13_SEED, self.SMOKE_P44_SEED,
            self.SMOKE_CAP_SEED,
        }
        self.assertTrue(all(
            str(seed).startswith("20261009") for seed in smoke_seeds
        ))
        self.assertTrue(production_seeds.isdisjoint(smoke_seeds))

        p14 = next(case for case in audit.CASES if case.case_id == "showcase-572-p14")
        self.assertEqual(
            p14.candidates,
            ("R4 d deck", "G7 p deck", "B3 d deck"),
        )
        p10 = next(case for case in audit.CASES if case.case_id == "ui-221-p10")
        self.assertEqual(p10.min_worlds, 2048)
        self.assertTrue(all(
            case.min_worlds == 1024
            for case in audit.CASES if case is not p10
        ))
        p13 = next(case for case in audit.CASES if case.case_id == "ui-221-p13")
        self.assertEqual(p13.candidates, ())
        self.assertEqual(p13.belief_card, "Y9")

    def test_confidence_interval_that_spans_zero_is_inconclusive(self) -> None:
        self.assertEqual(audit.descriptive_signal(0.19, 1.17), "inconclusive")
        self.assertEqual(audit.descriptive_signal(2.0, 0.25), "alternative_ahead")
        self.assertEqual(audit.descriptive_signal(-2.0, 0.25), "reference_ahead")

    def test_helper_output_is_deterministic_and_information_safe(self) -> None:
        original = ROOT / "data" / "probes" / "ui_seed725402798_p21.state"
        text = original.read_text()
        self.assertIn("hand1 Yx Y3 Y10", text)
        # Y3 is hidden from the mover here. Swap it for the unseen Y2 while
        # preserving the opponent hand size; no decision-time result may move.
        mutated_text = text.replace("hand1 Yx Y3 Y10", "hand1 Yx Y2 Y10", 1)
        with tempfile.TemporaryDirectory(prefix="lc-commented-audit-") as tmp:
            completed = Path(tmp) / "complete.state"
            mutated = Path(tmp) / "mutated.state"
            completed.write_text(self.with_complete_deck(text))
            mutated.write_text(self.with_complete_deck(
                mutated_text, reverse=True
            ))
            args = (
                "--candidate", "Bx d deck",
                "--candidate", "G5 p deck",
            )
            first = self.helper(completed, *args)
            repeated = self.helper(completed, *args)
            changed_hidden_truth = self.helper(mutated, *args)
            changed_seed = self.helper(
                completed, *args, seed=self.SMOKE_CHANGED_SEED
            )
        self.assertEqual(first, repeated)
        self.assertEqual(first["state"]["input_deck_entries"], 24)
        self.assertEqual(changed_hidden_truth["state"]["input_deck_entries"], 24)
        first["state"]["path"] = "STATE"
        changed_hidden_truth["state"]["path"] = "STATE"
        self.assertEqual(first, changed_hidden_truth)
        self.assertEqual(first["schema"], audit.EVAL_SCHEMA)
        counterfactual = first["counterfactual"]
        self.assertTrue(counterfactual["shared_current_hidden_worlds"])
        self.assertTrue(counterfactual["shared_future_deals"])
        self.assertTrue(counterfactual["branch_neutral_rng_domains"])
        self.assertEqual(counterfactual["hash_algorithm"], "fnv1a64")
        self.assertEqual(counterfactual["cap_hits"], 0)
        for key in (
            "hidden_world_set_hash", "future_deal_set_hash",
            "branch_rng_domain_hash",
        ):
            self.assertRegex(counterfactual[key], r"^[0-9a-f]{16}$")
        continuation = counterfactual["continuation"]
        self.assertEqual(continuation["kind"], "exact_policy_argmax")
        self.assertEqual(
            continuation["scope"], "full_remaining_three_round_match"
        )
        self.assertEqual(continuation["checkpoint"], "data/champion.bin")
        self.assertTrue(continuation["exact_group_average"])
        self.assertTrue(continuation["fresh_information_view_each_node"])
        self.assertFalse(continuation["recursive_actor"])
        self.assertNotEqual(
            counterfactual["hidden_world_set_hash"],
            changed_seed["counterfactual"]["hidden_world_set_hash"],
        )
        self.assertNotEqual(
            counterfactual["future_deal_set_hash"],
            changed_seed["counterfactual"]["future_deal_set_hash"],
        )

    def test_p13_reports_exact_k_posterior_programmatically(self) -> None:
        state = ROOT / "data" / "probes" / "ui_seed2214615196_p13.state"
        case = next(case for case in audit.CASES if case.case_id == "ui-221-p13")
        smoke_case = replace(case, audit_seed=self.SMOKE_P13_SEED)
        result = self.helper(
            state,
            "--belief", "--belief-alpha", "1.15",
            "--belief-card", "Y9",
            seed=smoke_case.audit_seed,
        )
        result["state"]["path"] = case.state
        audit._validate_evaluation(
            smoke_case, result, actor_spec=self.POLICY_ACTOR,
            net_label="data/champion.bin", worlds=2, symmetries=20,
            belief_alpha=1.15,
        )
        panel = result["counterfactual"]
        self.assertEqual(panel["nominated_candidates"], 0)
        self.assertEqual(panel["policy_reference_candidates"], 2)
        self.assertIn(len(panel["candidates"]), (2, 3))
        self.assertTrue(panel["actor_selected_included"])
        self.assertIn(
            result["actor"]["selected"],
            [row["move"] for row in panel["candidates"]],
        )
        self.assertEqual(
            panel["candidates"][0]["move"], result["actor"]["selected"]
        )
        self.assertGreaterEqual(
            panel["candidates"][0]["policy_prior"],
            panel["candidates"][1]["policy_prior"],
        )
        belief = result["belief"]
        self.assertTrue(belief["valid"])
        self.assertEqual(belief["kind"], "fixed_k")
        self.assertTrue(belief["information_view"])
        self.assertTrue(belief["complete_state_used_only_as_truth_label"])
        self.assertEqual(belief["symmetries"], 20)
        self.assertAlmostEqual(belief["alpha"], 1.15, places=6)
        self.assertEqual(belief["need"], 8)
        self.assertAlmostEqual(belief["marginal_sum"], 8.0, places=5)
        self.assertAlmostEqual(belief["uniform_marginal"], 0.2, places=8)
        self.assertEqual(belief["target"]["card"], "Y9")
        self.assertGreater(belief["target"]["marginal"], 0.2)
        cards = {row["card"]: row for row in belief["cards"]}
        self.assertTrue(cards["B10"]["held"])
        self.assertTrue(cards["B9"]["held"])

    def test_public_driver_refuses_any_locked_contract_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal the locked base budget"):
            audit.build_audit(
                helper=ROOT / "bin" / "commented_ply_eval",
                actor_spec=self.POLICY_ACTOR,
                net_path=ROOT / "data" / "champion.bin",
                worlds=1023,
            )
        with self.assertRaisesRegex(ValueError, "equal the locked base budget"):
            audit.build_audit(
                helper=ROOT / "bin" / "commented_ply_eval",
                actor_spec=self.POLICY_ACTOR,
                net_path=ROOT / "data" / "champion.bin",
                worlds=1025,
            )
        with self.assertRaisesRegex(ValueError, "policy-20"):
            audit.build_audit(
                helper=ROOT / "bin" / "commented_ply_eval",
                actor_spec=self.POLICY_ACTOR,
                net_path=ROOT / "data" / "champion.bin",
                symmetries=5,
            )
        with self.assertRaisesRegex(ValueError, "belief alpha"):
            audit.build_audit(
                helper=ROOT / "bin" / "commented_ply_eval",
                actor_spec=self.POLICY_ACTOR,
                net_path=ROOT / "data" / "champion.bin",
                belief_alpha=1.0,
            )

    def test_actor_selected_move_is_always_counterfactually_graded(self) -> None:
        state = ROOT / "data" / "probes" / "ui_seed95647345759839_p44.state"
        result = self.helper(
            state,
            "--candidate", "W10 p deck",
            "--candidate", "W10 p G",
            seed=self.SMOKE_P44_SEED,
        )
        case = next(case for case in audit.CASES if case.case_id == "ui-956-p44")
        smoke_case = replace(case, audit_seed=self.SMOKE_P44_SEED)
        result["state"]["path"] = case.state
        audit._validate_evaluation(
            smoke_case, result, actor_spec=self.POLICY_ACTOR,
            net_label="data/champion.bin", worlds=2, symmetries=20,
            belief_alpha=1.15,
        )
        moves = [row["move"] for row in result["counterfactual"]["candidates"]]
        self.assertEqual(moves[0], "W10 p deck")
        self.assertNotEqual(result["actor"]["selected"], "W10 p deck")
        self.assertIn(result["actor"]["selected"], moves)
        self.assertTrue(result["counterfactual"]["actor_selected_included"])

        semantic_keys = [
            row["semantic_key"]
            for row in result["counterfactual"]["candidates"]
        ]
        self.assertEqual(len(set(semantic_keys)), len(semantic_keys))
        for row in result["counterfactual"]["candidates"]:
            metrics = row["metrics"]
            self.assertEqual(
                set(metrics), {"match_score", "final_margin", "hybrid"}
            )
            for metric in metrics.values():
                self.assertRegex(metric["samples_fnv1a64"], r"^[0-9a-f]{16}$")
            self.assertAlmostEqual(
                metrics["hybrid"]["mean"],
                0.05 * metrics["final_margin"]["mean"]
                + 100 * (metrics["match_score"]["mean"] - 0.5),
                places=7,
            )

    def test_duplicate_physical_wager_nominations_are_semantically_deduped(
        self,
    ) -> None:
        state = ROOT / "data" / "probes" / "ui_seed95647345759839_p44.state"
        self.assertIn("hand1 Y7 Bx Bx B5 B7 W10 G8 R4", state.read_text())
        result = self.helper(
            state,
            "--candidate", "Bx d deck",
            "--candidate", "Bx d deck",
            "--candidate", "W10 p deck",
        )
        counterfactual = result["counterfactual"]
        self.assertEqual(counterfactual["nominated_candidates"], 2)
        self.assertEqual(
            counterfactual["policy_probability_aggregation"],
            "sum_by_semantic_move",
        )
        moves = [row["move"] for row in counterfactual["candidates"]]
        self.assertEqual(moves.count("Bx d deck"), 1)
        self.assertEqual(
            len({row["semantic_key"] for row in counterfactual["candidates"]}),
            len(counterfactual["candidates"]),
        )

    def test_counterfactual_cap_is_fail_closed(self) -> None:
        source = ROOT / "data" / "probes" / "ui_seed725402798_p21.state"
        capped = source.read_text().replace("nply 20", "nply 299", 1)
        with tempfile.TemporaryDirectory(prefix="lc-commented-cap-") as tmp:
            path = Path(tmp) / "cap.state"
            path.write_text(capped)
            command = [
                str(ROOT / "bin" / "commented_ply_eval"),
                "--state", str(path),
                "--actor", self.POLICY_ACTOR,
                "--net", "data/champion.bin",
                "--seed", str(self.SMOKE_CAP_SEED),
                "--worlds", "2",
                "--symmetries", "20",
                "--candidate", "Bx d deck",
                "--candidate", "G5 p deck",
            ]
            result = subprocess.run(
                command, cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("LC_MAX_PLIES", result.stderr)

    def test_locked_schema_exposes_shards_and_exact_merge(self) -> None:
        contract = audit._contract()
        self.assertEqual(audit.ATTEMPT_ID, "v3")
        self.assertEqual(
            audit.RECOVERY_BINDING["rerun_previous_attempt"], "forbidden"
        )
        self.assertEqual(contract["case_count"], 17)
        self.assertEqual(contract["exact_policy_teacher"], self.POLICY_ACTOR)
        self.assertEqual(contract["exact_policy_symmetries"], 20)
        self.assertEqual(
            contract["paired_worlds_by_case"]["ui-221-p10"], 2048
        )
        self.assertTrue(contract["actor_selected_action_graded_on_all_17"])
        self.assertEqual(contract["counterfactual_caps"], "zero_fail_closed")
        args = audit.parse_args([
            "--actor", self.POLICY_ACTOR,
            "--case", "ui-221-p3",
            "--source-commit", "a" * 40,
        ])
        self.assertEqual(args.case_ids, ["ui-221-p3"])
        self.assertEqual(args.source_commit, "a" * 40)
        merge_args = audit.parse_args([
            "--merge", "shard-a.json", "shard-b.json",
            "--output-json", "merged.json",
            "--output-md", "merged.md",
        ])
        self.assertEqual(
            merge_args.merge, [Path("shard-a.json"), Path("shard-b.json")]
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            audit.merge_audits([])
        with self.assertRaisesRegex(RuntimeError, "unexpected audit schema"):
            audit.validate_audit_document(
                {"schema": "lc-commented-ply-audit-v1"},
                require_full=True,
            )

    def test_source_commit_binding_supports_sealed_gitless_transport(self) -> None:
        case = audit.CASES[0]
        with mock.patch.object(audit, "_git_head", return_value=None), \
                mock.patch.object(audit, "run_case", return_value={}), \
                mock.patch.object(
                    audit, "_envelope", return_value={"bound": True}
                ):
            value = audit.build_audit(
                helper=ROOT / "bin" / "commented_ply_eval",
                actor_spec=self.POLICY_ACTOR,
                net_path=ROOT / "data" / "champion.bin",
                selected_cases=(case,),
                source_commit="a" * 40,
            )
        self.assertEqual(value, {"bound": True})
        with mock.patch.object(audit, "_git_head", return_value="b" * 40), \
                mock.patch.object(audit, "run_case") as run:
            with self.assertRaisesRegex(ValueError, "checked-out HEAD"):
                audit.build_audit(
                    helper=ROOT / "bin" / "commented_ply_eval",
                    actor_spec=self.POLICY_ACTOR,
                    net_path=ROOT / "data" / "champion.bin",
                    selected_cases=(case,),
                    source_commit="a" * 40,
                )
            run.assert_not_called()

    def test_merge_rejects_duplicate_case_even_after_fragment_validation(
        self,
    ) -> None:
        metadata = {
            "schema": audit.AUDIT_SCHEMA,
            "attempt_id": audit.ATTEMPT_ID,
            "recovery": audit.RECOVERY_BINDING,
            "audit_definition_sha256": "definition",
            "repository_head": "a" * 40,
            "subject": {},
            "implementation": {},
            "contract": {},
            "selection": {"case_ids": ["ui-221-p3"]},
            "summary": {},
            "cases": [{"case_id": "ui-221-p3"}],
        }
        with mock.patch.object(audit, "validate_audit_document"):
            with self.assertRaisesRegex(RuntimeError, "duplicate audit case"):
                audit.merge_audits([
                    json.loads(json.dumps(metadata)),
                    json.loads(json.dumps(metadata)),
                ])


if __name__ == "__main__":
    unittest.main()
