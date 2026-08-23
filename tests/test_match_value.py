"""End-to-end contracts for controller-bound match-value tables."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "bin" / "build_match_value"
ARENA = ROOT / "bin" / "arena"
TRAIN = ROOT / "bin" / "train"
SHOWGAME = ROOT / "bin" / "showgame"
ANALYZE = ROOT / "bin" / "analyze"
PLAY = ROOT / "bin" / "play"
PROBE = ROOT / "bin" / "probe"
FLAGGED_PROBE = ROOT / "bin" / "flagged_ply_probe"
RL = ROOT / "bin" / "rl"
CHAMPION = ROOT / "data" / "champion.bin"
OTHER_CONTROLLER = ROOT / "data" / "models" / "continuation_soup_v1.bin"


def build_table(
    path: Path,
    threads: int,
    *,
    samples: int = 2,
    symmetries: int = 20,
    raw_out: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
            str(BUILDER),
            "--model", str(CHAMPION),
            "--out", str(path),
            "--samples", str(samples),
            "--threads", str(threads),
            "--seed", "7331002",
            "--playout-symmetries", str(symmetries),
        ]
    if raw_out is not None:
        command.extend(["--raw-out", str(raw_out)])
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )


def table_actor(
    table: Path,
    continuation: Path = CHAMPION,
    *,
    playout_symmetries: int = 5,
) -> str:
    # Every field is explicit because the table path is positional field 41;
    # field 40 remains the action-ranker threshold and is zero for rollout2.
    # Search is restricted to deck=1 to keep this parser/runtime fixture tiny;
    # production experiments can use the same table at their measured phase.
    tail: list[object] = [
        2, 2, 0, 0, 1, 0, 0, 0, 3, 0,
        0, 0, 4, 20, 0, 0, playout_symmetries, 0, 1, 2,
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        table,
    ]
    assert len(tail) == 42
    return "rolloutu2:" + ":".join(
        [str(CHAMPION), str(continuation), *(str(value) for value in tail)]
    )


class MatchValueTableTest(unittest.TestCase):
    def test_builder_is_thread_invariant_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            one = Path(temporary) / "one.lcmv"
            many = Path(temporary) / "many.lcmv"
            first = build_table(one, 1)
            parallel = build_table(many, 4)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(parallel.returncode, 0, parallel.stderr)
            self.assertIn("samples=2 seed=7331002", first.stdout)
            self.assertIn("role_cycle=400 role_balance=incomplete", first.stdout)
            self.assertEqual(one.read_bytes(), many.read_bytes())
            self.assertEqual(one.read_bytes()[:8], b"LCMVAL1\0")

            before = one.read_bytes()
            repeated = build_table(one, 2)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("cannot create match-value table", repeated.stderr)
            self.assertEqual(one.read_bytes(), before)

            projected = Path(temporary) / "rolled-back.lcmv"
            occupied_raw = Path(temporary) / "occupied-raw.lcmv"
            sentinel = b"preexisting raw output\n"
            occupied_raw.write_bytes(sentinel)
            half_pair = build_table(
                projected,
                2,
                raw_out=occupied_raw,
            )
            self.assertNotEqual(half_pair.returncode, 0)
            self.assertFalse(projected.exists())
            self.assertEqual(occupied_raw.read_bytes(), sentinel)

    def test_table_actor_loads_and_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary) / "fixture.lcmv"
            raw_table = Path(temporary) / "fixture-raw.lcmv"
            built = build_table(
                table,
                3,
                samples=25,
                symmetries=5,
                raw_out=raw_table,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertIn("variant=isotonic", built.stdout)
            self.assertIn("variant=raw", built.stdout)
            self.assertIn("role_cycle=25 role_balance=complete", built.stdout)
            self.assertNotEqual(table.read_bytes(), raw_table.read_bytes())

            played = subprocess.run(
                [
                    str(ARENA),
                    "-a", table_actor(table, playout_symmetries=5),
                    "-b", f"policy:{CHAMPION}:0:20",
                    "-n", "1",
                    "-t", "1",
                    "-s", "7331009",
                    "-r", "3",
                    "-q",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(played.returncode, 0, played.stderr)
            self.assertEqual(len(played.stdout.split()), 4)

            played_raw = subprocess.run(
                [
                    str(ARENA),
                    "-a", table_actor(
                        raw_table,
                        playout_symmetries=5,
                    ),
                    "-b", f"policy:{CHAMPION}:0:20",
                    "-n", "1",
                    "-t", "1",
                    "-s", "7331015",
                    "-r", "3",
                    "-q",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(played_raw.returncode, 0, played_raw.stderr)

            tail = table_actor(
                table,
                playout_symmetries=5,
            ).split(":", 3)[3]
            unsafe_veto = ":".join([
                "rolloutu3",
                str(CHAMPION),
                str(CHAMPION),
                str(CHAMPION),
                tail,
            ])
            veto_rejected = subprocess.run(
                [
                    str(ARENA),
                    "-a", unsafe_veto,
                    "-b", "random",
                    "-n", "1",
                    "-t", "1",
                    "-s", "7331016",
                    "-r", "3",
                    "-q",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(veto_rejected.returncode, 0)
            self.assertIn("invalid rollout configuration",
                          veto_rejected.stderr)

            ranker_tail = tail.split(":")
            ranker_tail[27] = "2"
            ranked_table_actor = ":".join([
                "rolloutu4",
                str(CHAMPION),
                str(CHAMPION),
                str(CHAMPION),
                *ranker_tail,
            ])
            ranker_played = subprocess.run(
                [
                    str(ARENA),
                    "-a", ranked_table_actor,
                    "-b", "random",
                    "-n", "1",
                    "-t", "1",
                    "-s", "7331017",
                    "-r", "3",
                    "-q",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(ranker_played.returncode, 0,
                             ranker_played.stderr)

            for rounds in (1, 2):
                wrong_horizon = subprocess.run(
                    [
                        str(ARENA),
                        "-a", table_actor(
                            table,
                            playout_symmetries=5,
                        ),
                        "-b", "random",
                        "-n", "1",
                        "-t", "1",
                        "-s", f"733102{rounds}",
                        "-r", str(rounds),
                        "-q",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertNotEqual(wrong_horizon.returncode, 0)
                self.assertIn("match execution failed", wrong_horizon.stderr)

            actor = table_actor(table, playout_symmetries=5)
            policy = f"policy:{CHAMPION}:0:20"

            legacy_fields = actor.split(":")
            legacy_fields[3 + 8] = "0"
            # Tail fields 40 and 41 are respectively reserved for an action
            # ranker and a match-value table.  A legacy rollout2 actor has
            # neither role, so omit both optional fields.
            legacy_evaluator = ":".join(legacy_fields[:-2])
            objective_probe = subprocess.run(
                [
                    str(FLAGGED_PROBE),
                    "-S", str(
                        ROOT / "data/probes/ui_seed2214615196_p13.state"
                    ),
                    "-a", legacy_evaluator,
                    "-b", actor,
                    "-w", "2",
                    "--belief-only",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(
                objective_probe.returncode, 0, objective_probe.stderr
            )
            objective_actors = json.loads(objective_probe.stdout)["actors"]
            self.assertEqual(
                (
                    objective_actors[0]["objective_label"],
                    objective_actors[0]["objective_units"],
                ),
                ("round_margin", "round_points"),
            )
            self.assertEqual(
                (
                    objective_actors[1]["objective_label"],
                    objective_actors[1]["objective_units"],
                ),
                (
                    "controller_bound_full_match_value",
                    "expected_full_match_hybrid_utility_points",
                ),
            )

            analyzed = subprocess.run(
                [
                    str(ANALYZE),
                    "-a", policy,
                    "-e", actor,
                    "-s", "7331018",
                    "-r", "3",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(analyzed.returncode, 0, analyzed.stderr)
            report = json.loads(analyzed.stdout)
            objectives_by_round = {
                round_index: {
                    ply["analysis"]["objective"]
                    for ply in report["plies"]
                    if ply["round"] == round_index
                }
                for round_index in range(3)
            }
            self.assertEqual(
                objectives_by_round,
                {
                    0: {"controller_bound_full_match_value"},
                    1: {"controller_bound_full_match_value"},
                    2: {"final_hybrid"},
                },
            )

            # The new table-backed selection mode must not relabel the
            # established last-round-only objective used by legacy actors.
            legacy_fields[3 + 8] = "2"
            legacy_evaluator = ":".join(legacy_fields[:-2])
            legacy_analyzed = subprocess.run(
                [
                    str(ANALYZE),
                    "-a", policy,
                    "-e", legacy_evaluator,
                    "-s", "7331019",
                    "-r", "3",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(
                legacy_analyzed.returncode, 0, legacy_analyzed.stderr
            )
            legacy_report = json.loads(legacy_analyzed.stdout)
            legacy_objectives_by_round = {
                round_index: {
                    ply["analysis"]["objective"]
                    for ply in legacy_report["plies"]
                    if ply["round"] == round_index
                }
                for round_index in range(3)
            }
            self.assertEqual(
                legacy_objectives_by_round,
                {
                    0: {"round_margin"},
                    1: {"round_margin"},
                    2: {"final_hybrid"},
                },
            )

            direct_simulator_rejections = [
                (
                    [str(SHOWGAME), "-a", actor, "-r", "1"],
                    "showgame: a match-value actor requires exactly 3 rounds",
                ),
                (
                    [str(ANALYZE), "-a", actor, "-e", policy,
                     "-r", "2"],
                    "analyze: a match-value actor or evaluator requires "
                    "exactly 3 rounds",
                ),
                (
                    [str(ANALYZE), "-a", policy, "-e", actor,
                     "-r", "1"],
                    "analyze: a match-value actor or evaluator requires "
                    "exactly 3 rounds",
                ),
                (
                    [str(PLAY), "-a", actor],
                    "play: match-value actors require a complete "
                    "three-round match",
                ),
                (
                    [str(PROBE), "-n", str(CHAMPION), "-p", actor,
                     "-g", "1"],
                    "probe: match-value actors are incompatible with "
                    "independent-round margin targets",
                ),
                (
                    [
                        str(RL),
                        "--init", str(CHAMPION),
                        "--out", str(Path(temporary) / "unused-rl.bin"),
                        "--gen-opponent", actor,
                        "--rounds", "1",
                        "--iters", "1",
                        "--games", "1",
                        "--threads", "1",
                        "--eval", "0",
                    ],
                    "match-value reference/opponent actors require "
                    "--rounds 3",
                ),
            ]
            for command, diagnostic in direct_simulator_rejections:
                rejected = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    input="",
                    capture_output=True,
                    timeout=30,
                )
                self.assertNotEqual(rejected.returncode, 0, command)
                self.assertIn(diagnostic, rejected.stderr)

            wrong_controller = subprocess.run(
                [
                    str(ARENA),
                    "-a", table_actor(
                        table,
                        OTHER_CONTROLLER,
                        playout_symmetries=5,
                    ),
                    "-b", "random",
                    "-n", "1",
                    "-t", "1",
                    "-s", "7331010",
                    "-r", "1",
                    "-q",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(wrong_controller.returncode, 0)
            self.assertIn("invalid rollout configuration", wrong_controller.stderr)

            fields = table_actor(table, playout_symmetries=5).split(":")
            # Two-network prefixes occupy fields 0..2; tail objective is 8.
            fields[3 + 8] = "0"
            wrong_mode = subprocess.run(
                [
                    str(ARENA),
                    "-a", ":".join(fields),
                    "-b", "random",
                    "-n", "1",
                    "-t", "1",
                    "-s", "7331011",
                    "-r", "1",
                    "-q",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(wrong_mode.returncode, 0)
            self.assertIn("invalid rollout configuration", wrong_mode.stderr)

            incomplete = Path(temporary) / "incomplete.lcmv"
            incomplete_build = build_table(incomplete, 2)
            self.assertEqual(incomplete_build.returncode, 0,
                             incomplete_build.stderr)
            unbalanced_actor = subprocess.run(
                [
                    str(ARENA),
                    "-a", table_actor(
                        incomplete,
                        playout_symmetries=20,
                    ),
                    "-b", "random",
                    "-n", "1",
                    "-t", "1",
                    "-s", "7331012",
                    "-r", "1",
                    "-q",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(unbalanced_actor.returncode, 0)
            self.assertIn("invalid rollout configuration",
                          unbalanced_actor.stderr)

            selfrollout = "selfrollout:" + table_actor(
                table,
                playout_symmetries=5,
            ).split(":", 3)[3]
            training = subprocess.run(
                [
                    str(TRAIN),
                    "--init", str(CHAMPION),
                    "--out", str(Path(temporary) / "unused.bin"),
                    "--gen", selfrollout,
                    "--iters", "1",
                    "--games", "1",
                    "--threads", "1",
                    "--buffer", "1",
                    "--steps", "1",
                    "--eval", "0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(training.returncode, 0)
            self.assertIn("live selfrollout cannot use a match-value table",
                          training.stderr)

            loaded_training = subprocess.run(
                [
                    str(TRAIN),
                    "--init", str(CHAMPION),
                    "--out", str(Path(temporary) / "unused-loaded.bin"),
                    "--gen", table_actor(
                        table,
                        playout_symmetries=5,
                    ),
                    "--iters", "1",
                    "--games", "1",
                    "--threads", "1",
                    "--buffer", "1",
                    "--steps", "1",
                    "--eval", "0",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(loaded_training.returncode, 0)
            self.assertIn("--gen cannot use a match-value table",
                          loaded_training.stderr)


if __name__ == "__main__":
    unittest.main()
