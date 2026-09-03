import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
QPAIR = ROOT / "bin" / "qpair"
CHAMPION = ROOT / "data" / "champion.bin"
STATE = ROOT / "data" / "probes" / "g424_p54.state"

# Two-world fixture: enough to exercise the ordinary evaluator report without
# turning a telemetry regression into a campaign-sized test.  Field 41 is the
# direct action-ranker threshold; policy-prefix mode 2 is required by parser.
TAIL_41 = (
    "2:2:0.02:0:1:0:0:0:0:0:0:4:0:1:0:0:1:0:0:2:-1:0:0:0:0:0:0:2:"
    "0:0:0:0:0:0:0:1:0:0:0:1:0"
)


class ActionRankerVetoTelemetryTests(unittest.TestCase):
    def test_qpair_reports_separate_direct_ranker_role(self):
        self.assertEqual(len(TAIL_41.split(":")), 41)
        spec = (
            f"rolloutu4:{CHAMPION}:{CHAMPION}:{CHAMPION}:{TAIL_41}"
        )
        proc = subprocess.run(
            [
                str(QPAIR),
                "-n",
                str(CHAMPION),
                "-S",
                str(STATE),
                "-E",
                spec,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("controller veto: disabled", proc.stdout)
        self.assertIn("action ranker veto: configured", proc.stdout)
        self.assertIn("role: direct signed ranker veto only", proc.stdout)
        self.assertIn("attempted: yes; valid: yes; score: +0.000000", proc.stdout)
        self.assertIn("threshold: 0.000000", proc.stdout)
        self.assertIn("result: confirmed override retained", proc.stdout)


if __name__ == "__main__":
    unittest.main()
