from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "policy_cost_exact17_v9.py"
SPEC = importlib.util.spec_from_file_location("policy_cost_exact17_v9", MODULE_PATH)
assert SPEC and SPEC.loader
exact17 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exact17)

ALLOCATOR_PATH = ROOT / "tools" / "policy_cost_allocate_v9.py"
ALLOCATOR_SPEC = importlib.util.spec_from_file_location(
    "policy_cost_allocate_v9", ALLOCATOR_PATH
)
assert ALLOCATOR_SPEC and ALLOCATOR_SPEC.loader
allocator = importlib.util.module_from_spec(ALLOCATOR_SPEC)
ALLOCATOR_SPEC.loader.exec_module(allocator)


class PolicyCostExact17V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temp.name) / "policy_cost_dataset"
        core = [
            "src/lc.c", "src/features.c", "src/net.c", "src/heuristic.c",
            "src/planner.c", "src/search.c", "src/rollout.c",
            "src/late_resolver.c", "src/match_value.c", "src/policy_cost.c",
            "src/agent.c", "src/match.c", "src/spec.c",
        ]
        subprocess.run([
            "gcc", "-O0", "-Wall", "-Wextra", "-std=c11",
            "-fno-fast-math", "-ffp-contract=off", "-o", str(cls.binary),
            "tools/policy_cost_dataset_v9.c", *core[:-4],
            "src/policy_cost_v9.c", *core[-3:], "-lm", "-pthread",
        ], cwd=ROOT, check=True, capture_output=True, text=True)
        exclusions = (
            ROOT / "data" / "experiments" /
            "policy_cost_v9_exact17_exclusions.txt"
        )
        exclusion_sha = hashlib.sha256(exclusions.read_bytes()).hexdigest()
        cls.native_dir = tempfile.TemporaryDirectory()
        native_root = Path(cls.native_dir.name)
        discovery = native_root / "discovery.jsonl"
        reservoir = native_root / "reservoir.tsv"
        cls.native_discovery_path = discovery
        cls.native_reservoir_path = reservoir
        subprocess.run([
            str(cls.binary), "discover", "--out", str(discovery),
            "--reservoir-out", str(reservoir),
            "--net", "data/champion.bin", "--split", "TRAIN",
            "--matches", "1", "--match-start", "0",
            "--reservoir-per-cell", "1", "--exclusions", str(exclusions),
            "--exclusions-sha256", exclusion_sha,
            "--smoke-seed", "202708290999",
        ], cwd=ROOT, check=True, capture_output=True, text=True)
        cls.native_records = [
            json.loads(line)
            for line in discovery.read_text(encoding="ascii").splitlines()
        ]
        cls.native_reservoir = reservoir.read_text(encoding="ascii")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.native_dir.cleanup()
        cls.temp.cleanup()

    def test_exact_locked_case_set_exports_unique_orbits(self) -> None:
        text, evidence = exact17.build_outputs(ROOT, self.binary)
        lines = text.decode("ascii").splitlines()
        self.assertEqual(lines[0], exact17.TEXT_SCHEMA)
        self.assertEqual(len(lines), 18)
        self.assertEqual(len(set(lines[1:])), 17)
        self.assertTrue(all(len(digest) == 64 for digest in lines[1:]))
        self.assertEqual(evidence["case_count"], 17)
        self.assertEqual(evidence["orbit_count"], 17)
        self.assertEqual(evidence["training_use"], "forbidden")
        self.assertEqual(
            evidence["bindings"]["native_hash_probe"]["source_path"],
            "tools/policy_cost_dataset_v9.c",
        )
        native = evidence["bindings"]["native_hash_probe"]
        self.assertNotIn("binary_sha256", native)
        self.assertEqual(
            native["runtime_binary_binding"],
            "dynamically_sealed_in_build_identity_and_transport",
        )
        self.assertEqual(
            evidence["runtime_exclusion_text"]["sha256"],
            hashlib.sha256(text).hexdigest(),
        )
        payload_digest = evidence["canonical_payload_sha256"]
        digest_payload = dict(evidence)
        del digest_payload["canonical_payload_sha256"]
        self.assertEqual(
            payload_digest,
            hashlib.sha256(exact17._canonical(digest_payload)).hexdigest(),
        )

    def test_checked_in_pair_is_exactly_regenerated(self) -> None:
        text, evidence = exact17.build_outputs(ROOT, self.binary)
        checked_text = (
            ROOT / "data/experiments/policy_cost_v9_exact17_exclusions.txt"
        )
        checked_json = (
            ROOT / "data/experiments/policy_cost_v9_exact17_exclusions.json"
        )
        self.assertEqual(checked_text.read_bytes(), text)
        self.assertEqual(checked_json.read_bytes(), exact17._canonical(evidence))

    def test_publish_is_atomic_pair_and_no_clobber(self) -> None:
        text, evidence = exact17.build_outputs(ROOT, self.binary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "excluded.txt"
            json_path = root / "excluded.json"
            exact17._publish_pair(text_path, text, json_path, evidence)
            self.assertEqual(text_path.read_bytes(), text)
            self.assertEqual(
                json.loads(json_path.read_text(encoding="ascii")), evidence
            )
            with self.assertRaisesRegex(exact17.ExclusionError, "replace"):
                exact17._publish_pair(text_path, text, json_path, evidence)

    def _native_shaped_campaign_contract(self):
        native_header, census, native_footer = copy.deepcopy(
            self.native_records
        )
        self.assertEqual(
            [record["record_type"] for record in self.native_records],
            ["header", "census", "footer"],
        )
        self.assertEqual(native_header["schema"], "lc-policy-cost-discovery-v5")
        self.assertTrue(
            self.native_reservoir.startswith(
                "LCPOLICYCOST-TRAIN-RESERVOIR-V5\n"
            )
        )
        reservoir_header = {
            "split": "TRAIN",
            "seed": allocator.SEEDS["TRAIN"],
            "net": native_header["net_sha256"],
            "exclusion": native_header["exclusion_manifest_sha256"],
        }
        reservoir_footer = {
            "eligible": native_footer["eligible_units"],
            "retained": native_footer["retained_units"],
            "rejected": native_footer["units_rejected_by_bound"],
            "chain": census["state_commitment_chain_sha256"],
            "pooled": sum(census["pooled_ge64_by_round"]),
        }
        campaign_header = allocator.campaign_discovery_header(
            "TRAIN", reservoir_header
        )
        campaign_footer = native_footer
        campaign_footer["requested_matches"] = allocator.MATCHES["TRAIN"]
        campaign_footer["completed_matches"] = allocator.MATCHES["TRAIN"]
        return (
            campaign_header, census, campaign_footer,
            reservoir_header, reservoir_footer, "TRAIN",
        )

    def test_native_producer_and_allocator_share_five_bin_union_contract(
        self,
    ) -> None:
        values = self._native_shaped_campaign_contract()
        census = values[1]
        self.assertEqual(allocator.MASK_MAX, 5)
        self.assertEqual(allocator.UNION_MAX, 5)
        self.assertEqual(len(census["union_width_counts"]), 5)
        self.assertEqual(
            census["union_width_counts"], census["mask_width_counts"][0]
        )
        allocator.validate_campaign_discovery(*values)
        completed = subprocess.run([
            "python3", str(ALLOCATOR_PATH), "--validate-contract-only",
            "--discovery", str(self.native_discovery_path),
            "--discovery-sha256", hashlib.sha256(
                self.native_discovery_path.read_bytes()
            ).hexdigest(),
            "--reservoir", str(self.native_reservoir_path),
            "--reservoir-sha256", hashlib.sha256(
                self.native_reservoir_path.read_bytes()
            ).hexdigest(),
            "--split", "TRAIN", "--smoke-seed", "202708290999",
            "--matches", "1", "--reservoir-per-cell", "1",
        ], cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertRegex(
            completed.stdout, r"\Acontract_smoke_sha256=[0-9a-f]{64}\n\Z"
        )

    def test_allocator_rejects_four_six_or_mismatched_union_bins(self) -> None:
        values = self._native_shaped_campaign_contract()
        four = copy.deepcopy(values)
        four[1]["union_width_counts"] = four[1]["union_width_counts"][:-1]
        with self.assertRaisesRegex(
            allocator.AllocationError, "invalid discovery union_width_counts"
        ):
            allocator.validate_campaign_discovery(*four)

        six = copy.deepcopy(values)
        six[1]["union_width_counts"].append(0)
        with self.assertRaisesRegex(
            allocator.AllocationError, "invalid discovery union_width_counts"
        ):
            allocator.validate_campaign_discovery(*six)

        mismatch = copy.deepcopy(values)
        union = mismatch[1]["union_width_counts"]
        source = next(index for index, count in enumerate(union) if count)
        target = (source + 1) % len(union)
        union[source] -= 1
        union[target] += 1
        with self.assertRaisesRegex(
            allocator.AllocationError, "does not equal the 1pct master"
        ):
            allocator.validate_campaign_discovery(*mismatch)

    def test_contract_only_and_campaign_identities_are_disjoint(self) -> None:
        discovery_sha = hashlib.sha256(
            self.native_discovery_path.read_bytes()
        ).hexdigest()
        reservoir_sha = hashlib.sha256(
            self.native_reservoir_path.read_bytes()
        ).hexdigest()
        with self.assertRaisesRegex(
            allocator.AllocationError, "outside burned 20270829"
        ):
            allocator.validate_contract_smoke(
                self.native_discovery_path, discovery_sha,
                self.native_reservoir_path, reservoir_sha,
                split="TRAIN", seed=allocator.SEEDS["TRAIN"], matches=1,
                reservoir_per_cell=1,
            )
        forbidden_out = Path(self.native_dir.name) / "forbidden.tsv"
        common = [
            "python3", str(ALLOCATOR_PATH),
            "--discovery", str(self.native_discovery_path),
            "--discovery-sha256", discovery_sha,
            "--reservoir", str(self.native_reservoir_path),
            "--reservoir-sha256", reservoir_sha,
        ]
        validation_with_output = subprocess.run([
            *common, "--validate-contract-only", "--split", "TRAIN",
            "--smoke-seed", "202708290999", "--matches", "1",
            "--reservoir-per-cell", "1", "--out", str(forbidden_out),
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(validation_with_output.returncode, 0)
        self.assertFalse(forbidden_out.exists())

        production_on_smoke = subprocess.run([
            *common, "--out", str(forbidden_out),
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(production_on_smoke.returncode, 0)
        self.assertIn("campaign identity mismatch", production_on_smoke.stderr)
        self.assertFalse(forbidden_out.exists())


if __name__ == "__main__":
    unittest.main()
