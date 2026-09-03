from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import copy
import struct
import json
import hashlib
import os
import unittest
from unittest import mock

from tools.policy_cost_artifact_v2 import (
    ANCHORS,
    ArtifactError,
    fnv1a64,
    read_policy_cost,
)
from tools.policy_cost_artifact import read_policy_cost as read_legacy_policy_cost
from tools import policy_cost_calibration_v2 as policy_cost_calibration, policy_cost_campaign_v2 as policy_cost_campaign
from tools import flagged_ply_audit, flagged_ply_execution
from tools import history_belief, make_showcase


ROOT = Path(__file__).resolve().parents[1]


class PolicyCostArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            ["make", "bin/test_policy_cost", "bin/build_policy_cost",
             "bin/commented_ply_eval", "bin/analyze"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        # Ordinary CI builds under generic GCC and Clang profiles, while
        # canonical LCPC evidence accepts only the frozen GCC-13/x86-64-v3
        # profile.  Probe the existing writer and rebuild only the two
        # evidence writers when necessary.  Locked preflight already builds
        # them exactly, so this branch stays cold and does not mutate the
        # compile-once transport before sealing.
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory) / "profile-probe.lcpc"
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(probe)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            try:
                read_policy_cost(probe)
            except ArtifactError:
                if os.environ.get("POLICY_COST_COMPILE_ONCE") == "1":
                    raise
                subprocess.run(
                    ["make", "-B", "CC=gcc",
                     f"CFLAGS={policy_cost_campaign.CFLAGS}",
                     f"LDFLAGS={policy_cost_campaign.LDFLAGS}",
                     "bin/test_policy_cost", "bin/build_policy_cost",
                     "bin/commented_ply_eval", "bin/analyze"],
                    cwd=ROOT,
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )

    def test_independent_reader_matches_c_writer_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.lcpc"
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            evidence = read_policy_cost(path)
            self.assertEqual(evidence["schema"], "lc-policy-cost-v2")
            self.assertEqual(evidence["artifact_version"], 3)
            self.assertEqual(evidence["anchors"], list(ANCHORS))
            self.assertEqual(evidence["primary_z"], 3.5)
            self.assertEqual(evidence["fresh_z"], 2.58)
            self.assertEqual(evidence["beta"], [1.0] * len(ANCHORS))
            self.assertEqual(len(evidence["alpha_action"]), len(ANCHORS))
            self.assertEqual(len(evidence["alpha_draw"]), len(ANCHORS))
            self.assertEqual(
                evidence["controller"]["cand_floor"],
                struct.unpack("<f", struct.pack("<f", 0.01))[0],
            )
            self.assertEqual(len(evidence["sha256"]), 64)
            self.assertEqual(len(evidence["content_fingerprint"]), 16)

            reserved = bytearray(path.read_bytes())
            reserved[28] = 1
            struct.pack_into("<Q", reserved, len(reserved) - 8,
                             fnv1a64(reserved[:-8]))
            path.write_bytes(reserved)
            with self.assertRaisesRegex(ArtifactError, "reserved"):
                read_policy_cost(path)

            path.unlink()
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            wrong_sample = bytearray(path.read_bytes())
            struct.pack_into("<I", wrong_sample, 64 + 4 * 5, 3)
            struct.pack_into("<Q", wrong_sample, len(wrong_sample) - 8,
                             fnv1a64(wrong_sample[:-8]))
            path.write_bytes(wrong_sample)
            with self.assertRaisesRegex(ArtifactError, "sampling/pruning"):
                read_policy_cost(path)

            path.unlink()
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            learned_belief = bytearray(path.read_bytes())
            struct.pack_into("<I", learned_belief, 64 + 4 * 8, 0)
            struct.pack_into("<Q", learned_belief, len(learned_belief) - 8,
                             fnv1a64(learned_belief[:-8]))
            path.write_bytes(learned_belief)
            with self.assertRaisesRegex(ArtifactError, "uniform-belief"):
                read_policy_cost(path)

            path.unlink()
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            delayed_onset = bytearray(path.read_bytes())
            struct.pack_into("<I", delayed_onset, 64 + 4 * 14, 14)
            struct.pack_into("<Q", delayed_onset, len(delayed_onset) - 8,
                             fnv1a64(delayed_onset[:-8]))
            path.write_bytes(delayed_onset)
            with self.assertRaisesRegex(ArtifactError, "all-ply search"):
                read_policy_cost(path)

            path.unlink()
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            legacy_floor = bytearray(path.read_bytes())
            struct.pack_into("<f", legacy_floor, 144, 2.0)
            struct.pack_into("<Q", legacy_floor, len(legacy_floor) - 8,
                             fnv1a64(legacy_floor[:-8]))
            path.write_bytes(legacy_floor)
            with self.assertRaisesRegex(ArtifactError, "low-prior"):
                read_policy_cost(path)

            path.unlink()
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            zero_beta = bytearray(path.read_bytes())
            struct.pack_into("<d", zero_beta, 256, 0.0)
            struct.pack_into("<Q", zero_beta, len(zero_beta) - 8,
                             fnv1a64(zero_beta[:-8]))
            path.write_bytes(zero_beta)
            with self.assertRaisesRegex(ArtifactError, "spline coefficient"):
                read_policy_cost(path)

            path.unlink()
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            negative_alpha = bytearray(path.read_bytes())
            struct.pack_into("<d", negative_alpha, 264, -1.0)
            struct.pack_into("<Q", negative_alpha, len(negative_alpha) - 8,
                             fnv1a64(negative_alpha[:-8]))
            path.write_bytes(negative_alpha)
            with self.assertRaisesRegex(ArtifactError, "spline coefficient"):
                read_policy_cost(path)

            path.unlink()
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            corrupted = bytearray(path.read_bytes())
            corrupted[256] ^= 1
            path.write_bytes(corrupted)
            with self.assertRaisesRegex(ArtifactError, "fingerprint mismatch"):
                read_policy_cost(path)

    def test_legacy_artifact_remains_byte_identical_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.lcpc"
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit-legacy", str(path)],
                cwd=ROOT, check=True,
            )
            self.assertEqual(len(path.read_bytes()), 424)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                "2c2437b37f8072178f9befdc57507ed45338ca87a23efdb2072a6c1dc3997a69",
            )
            legacy = read_legacy_policy_cost(path)
            self.assertEqual(legacy["source_seed"], 202611140101)
            with self.assertRaisesRegex(
                    ArtifactError, "bytes, expected|dimensions or version"):
                read_policy_cost(path)

    def test_trailing_bytes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.lcpc"
            subprocess.run(
                [str(ROOT / "bin/test_policy_cost"), "--emit", str(path)],
                cwd=ROOT,
                check=True,
            )
            path.write_bytes(path.read_bytes() + b"unexpected")
            with self.assertRaisesRegex(ArtifactError, "bytes, expected"):
                read_policy_cost(path)

    def test_deterministic_builder_derives_bindings_and_will_not_clobber(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "built.lcpc"
            schedule = ",".join(["1"] * len(ANCHORS))
            draw = ",".join(["0"] * len(ANCHORS))
            command = [
                str(ROOT / "bin/build_policy_cost"),
                "--root-model", "data/champion.bin",
                "--continuation-model", "data/champion.bin",
                "--out", str(path),
                "--source-seed", "202612140101",
                "--epsilon", "0x1p-150",
                "--objective", "0",
                "--root-symmetries", "20",
                "--playout-symmetries", "20",
                "--playout-sample", "4",
                "--playout-prune", "1",
                "--exact-terminal", "1",
                "--no-belief", "1",
                "--dets", "800",
                "--confirm-dets", "800",
                "--root-width", "5",
                "--action-core-count", "3",
                "--min-cand", "1",
                "--ply-lo", "0",
                "--ply-hi", "0",
                "--discard-guard", "1",
                "--root-prune", "0",
                "--cand-floor", "0.01",
                "--override-k", "3.5",
                "--override-min", "0",
                "--beta", schedule,
                "--alpha-action", schedule,
                "--alpha-draw", draw,
            ]
            built = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            manifest = json.loads(built.stdout)
            evidence = read_policy_cost(path)
            self.assertEqual(manifest["schema"], "lc-policy-cost-build-v2")
            self.assertEqual(manifest["artifact_version"], 3)
            self.assertEqual(manifest["legacy_override_min"], 0)
            self.assertEqual(manifest["no_belief"], 1)
            self.assertEqual(
                manifest["payload_fingerprint"],
                evidence["content_fingerprint"],
            )
            self.assertEqual(
                manifest["root_net_fingerprint"],
                manifest["continuation_net_fingerprint"],
            )
            nonuniform_path = Path(directory) / "nonuniform.lcpc"
            nonuniform = command.copy()
            nonuniform[nonuniform.index("--out") + 1] = str(nonuniform_path)
            nonuniform[nonuniform.index("--no-belief") + 1] = "0"
            rejected_nonuniform = subprocess.run(
                nonuniform,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(rejected_nonuniform.returncode, 0)
            self.assertFalse(nonuniform_path.exists())
            self.assertIn(
                "invalid policy-cost controller",
                rejected_nonuniform.stderr,
            )
            legacy_path = Path(directory) / "legacy-floor.lcpc"
            legacy_floor = command.copy()
            legacy_floor[legacy_floor.index("--out") + 1] = str(legacy_path)
            legacy_floor[legacy_floor.index("--override-min") + 1] = "2"
            rejected_floor = subprocess.run(
                legacy_floor,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(rejected_floor.returncode, 0)
            self.assertFalse(legacy_path.exists())
            self.assertIn(
                "invalid policy-cost controller",
                rejected_floor.stderr,
            )
            tail = (
                "800:5:0.01:0:1:0:0:0:0:0:3.5:0:4:20:0:0:20:1:"
                "0:800:1:0:0:0:0:0:0:0:0:0:0:0:0:0:3:1"
            )
            actor = (
                "rolloutu5:data/champion.bin:data/champion.bin:"
                f"{path}:{tail}"
            )
            evaluated = subprocess.run(
                [
                    str(ROOT / "bin/commented_ply_eval"),
                    "--state", "data/probes/ui_seed2214615196_p8.state",
                    "--actor", actor,
                    "--net", "data/champion.bin",
                    "--seed", "99117",
                    "--worlds", "2",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            actor_search = json.loads(evaluated.stdout)["actor"]["search"]
            diagnostics = actor_search["policy_cost"]
            self.assertTrue(diagnostics["active"])
            self.assertEqual(diagnostics["legacy_override_min"], 0)
            self.assertEqual(actor_search["confirmation_worlds"], 0)
            self.assertEqual(
                actor_search["fresh_worlds"], diagnostics["fresh"]["worlds"]
            )
            self.assertEqual(
                diagnostics["artifact_fingerprint"],
                evidence["content_fingerprint"],
            )
            self.assertEqual(diagnostics["primary"]["worlds"], 800)
            self.assertIn(diagnostics["gate_reason"], {
                "adjusted_baseline", "primary_all_pair_gate",
                "invalid_fresh", "fresh_leader_mismatch",
                "fresh_all_pair_gate", "discard_guard",
                "continuation_cap", "selected",
            })
            self.assertGreaterEqual(len(diagnostics["candidates"]), 1)
            for row in diagnostics["candidates"]:
                self.assertIn("semantic_action_mass", row)
                self.assertIn("conditional_draw_mass", row)
                self.assertIn("cost", row)
                self.assertIn("primary_adjusted_q", row)
                self.assertIn("fresh_adjusted_q", row)
            rejected = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("cannot create policy-cost artifact", rejected.stderr)

    def test_actor_manifest_accepts_native_binary32_one_percent_floor(
        self,
    ) -> None:
        """The frozen decimal 1% floor is represented as binary32 in LCPC."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "selected.lcpc"
            schedule_values = [1.0] * len(ANCHORS)
            draw_values = [0.0] * len(ANCHORS)
            command = [
                str(ROOT / "bin/build_policy_cost"),
                "--root-model", "data/champion.bin",
                "--continuation-model", "data/champion.bin",
                "--out", str(artifact),
                "--source-seed", policy_cost_campaign.POLICY_COST_SOURCE_SEED,
                "--epsilon", policy_cost_campaign.POLICY_COST_EPSILON,
                "--objective", "0",
                "--root-symmetries", "20",
                "--playout-symmetries", "20",
                "--playout-sample", "4",
                "--playout-prune", "1",
                "--exact-terminal", "1",
                "--no-belief", "1",
                "--dets", "800",
                "--confirm-dets", "800",
                "--root-width", "5",
                "--action-core-count", "3",
                "--min-cand", "1",
                "--ply-lo", "0",
                "--ply-hi", "0",
                "--discard-guard", "1",
                "--root-prune", "0",
                "--cand-floor", "0.01",
                "--override-k", "3.5",
                "--override-min", "0",
                "--beta", ",".join("1" for _ in ANCHORS),
                "--alpha-action", ",".join(map(str, schedule_values)),
                "--alpha-draw", ",".join(map(str, draw_values)),
            ]
            subprocess.run(command, cwd=ROOT, check=True,
                           stdout=subprocess.PIPE, text=True)
            parsed = read_policy_cost(artifact)
            self.assertNotEqual(parsed["controller"]["cand_floor"], 0.01)
            self.assertEqual(
                parsed["controller"]["cand_floor"],
                struct.unpack("<f", struct.pack("<f", 0.01))[0],
            )

            maintained = (
                "rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:"
                "3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:0:"
                "0:0:0:0:1"
            )
            execution = root / "execution.json"
            execution.write_bytes(policy_cost_campaign.canonical_json({
                "subject": {"maintained_actor": maintained, "objective": 0},
            }))
            train_evidence = {
                "required": True,
                "validated": True,
                "stage": "TRAIN",
                "raw_verified": True,
                "execution_sha256": policy_cost_campaign.sha256(execution),
                "train_input_sha256": "1" * 64,
            }
            calibration_payload = {
                "schema": "lc-policy-cost-calibration-v2",
                "calibration_passed": True,
                "status": "passed",
                "deployment": {"permitted": True, "reason": None},
                "schedule": {
                    "ply_anchors": list(ANCHORS),
                    "beta_search": schedule_values,
                    "alpha_core": schedule_values,
                    "alpha_draw": draw_values,
                },
                "campaign_design": {
                    "allocation_binding": {
                        "required": True, "validated": True,
                    },
                    "evidence_binding": train_evidence,
                },
                "model_adequacy": {
                    "required": True,
                    "evaluated": True,
                    "authoritative_pre_select_gate": True,
                    "passed": True,
                },
            }
            calibration_payload["calibration_sha256"] = hashlib.sha256(
                policy_cost_calibration._canonical_json_bytes(
                    calibration_payload
                )
            ).hexdigest()
            calibration = root / "calibration.json"
            calibration.write_bytes(
                policy_cost_campaign.canonical_json(calibration_payload)
            )
            select_evidence = {
                "calibration_sha256":
                    policy_cost_campaign.sha256(calibration),
                "policy_cost_sha256": policy_cost_campaign.sha256(artifact),
            }
            selection_path = root / "selection.json"
            selection_path.write_bytes(policy_cost_campaign.canonical_json({
                "campaign_evidence_binding": select_evidence,
            }))
            with mock.patch.object(
                    policy_cost_campaign, "_sealed_campaign_selection",
                    return_value=("floor-0.01_ply-00", 0.01, 0)), \
                    mock.patch.object(
                        policy_cost_campaign, "_validated_holdout_evidence",
                        return_value=select_evidence):
                actor = policy_cost_campaign.actor_manifest(
                    execution, calibration, selection_path, artifact, artifact,
                    "data/models/policy_cost_v2.lcpc",
                )
            self.assertEqual(
                actor["selected_configuration"]["policy_floor"], 0.01
            )
            self.assertIn("rolloutu5:", actor["candidate_actor"])

            failed_payload = copy.deepcopy(calibration_payload)
            failed_payload["calibration_passed"] = False
            failed_payload["status"] = "failed_model_adequacy"
            failed_payload["deployment"] = {
                "permitted": False,
                "reason": "authoritative_predictive_model_adequacy_gate_failed",
            }
            failed_payload["model_adequacy"]["passed"] = False
            failed_payload.pop("schedule")
            failed_payload["calibration_sha256"] = hashlib.sha256(
                policy_cost_calibration._canonical_json_bytes(
                    {key: value for key, value in failed_payload.items()
                     if key != "calibration_sha256"}
                )
            ).hexdigest()
            calibration.write_bytes(
                policy_cost_campaign.canonical_json(failed_payload)
            )
            failed_select_evidence = dict(
                select_evidence,
                calibration_sha256=policy_cost_campaign.sha256(calibration),
            )
            with mock.patch.object(
                    policy_cost_campaign, "_sealed_campaign_selection",
                    return_value=("floor-0.01_ply-00", 0.01, 0)), \
                    mock.patch.object(
                        policy_cost_campaign, "_validated_holdout_evidence",
                        return_value=failed_select_evidence), \
                    self.assertRaisesRegex(
                        policy_cost_campaign.EvidenceError,
                        "authoritative pre-SELECT gate"):
                policy_cost_campaign.actor_manifest(
                    execution, calibration, selection_path, artifact,
                    artifact, "data/models/policy_cost_v2.lcpc",
                )

    def test_rollout5_provenance_has_two_models_and_shifted_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in (
                ("root.bin", b"root"),
                ("continuation.bin", b"continuation"),
                ("schedule.lcpc", b"policy-cost"),
                ("match.lcmv", b"match-value"),
            ):
                (root / name).write_bytes(payload)
            tail = [str(index) for index in range(41)] + ["match.lcmv"]
            spec = (
                "rolloutu5:root.bin:continuation.bin:schedule.lcpc:"
                + ":".join(tail)
            )
            fields = spec.split(":")
            with mock.patch.object(make_showcase, "ROOT", root):
                self.assertEqual(make_showcase.rollout_tail_start(fields), 4)
                self.assertEqual(
                    make_showcase.actor_model_paths(spec),
                    (root / "root.bin", root / "continuation.bin"),
                )
                self.assertEqual(
                    make_showcase.rollout_policy_cost_path(fields),
                    root / "schedule.lcpc",
                )
                self.assertEqual(
                    make_showcase.rollout_match_value_path(fields),
                    root / "match.lcmv",
                )
            with mock.patch.object(history_belief, "ROOT", root):
                provenance = history_belief.source_actor_provenance(spec)
            self.assertEqual(
                [row["role"] for row in provenance["checkpoints"]],
                ["root", "continuation"],
            )
            self.assertEqual(
                provenance["policy_cost_artifact"]["path"],
                "schedule.lcpc",
            )
            self.assertEqual(
                provenance["match_value_table"]["path"], "match.lcmv"
            )
            execution_provenance = flagged_ply_execution._actor_provenance(
                root, spec, "candidate"
            )
            self.assertEqual(len(execution_provenance["checkpoints"]), 2)
            assets = flagged_ply_execution._actor_assets(execution_provenance)
            self.assertEqual(
                {row["kind"] for row in assets},
                {"checkpoint", "policy_cost_artifact", "match_value_table"},
            )
            with mock.patch.object(flagged_ply_audit, "ROOT", root):
                _, checkpoint_count = flagged_ply_audit.actor_layout(spec)
                audit_provenance = flagged_ply_audit.actor_provenance(spec)
            self.assertEqual(checkpoint_count, 2)
            self.assertEqual(
                audit_provenance["policy_cost_artifact"]["path"],
                "schedule.lcpc",
            )
            self.assertEqual(
                audit_provenance["match_value_table"]["path"], "match.lcmv"
            )


if __name__ == "__main__":
    unittest.main()
