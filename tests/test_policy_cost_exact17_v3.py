from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "policy_cost_exact17_v3.py"
SPEC = importlib.util.spec_from_file_location("policy_cost_exact17_v3", MODULE_PATH)
assert SPEC and SPEC.loader
exact17 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exact17)


class PolicyCostExact17Tests(unittest.TestCase):
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
            "tools/policy_cost_dataset_v3.c", *core[:-4],
            "src/policy_cost_v3.c", *core[-3:], "-lm", "-pthread",
        ], cwd=ROOT, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls) -> None:
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
            "tools/policy_cost_dataset_v3.c",
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


if __name__ == "__main__":
    unittest.main()
