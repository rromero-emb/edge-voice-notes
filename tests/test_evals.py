"""Evaluation set as a regression gate (fake engines, no models), plus WER unit tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

import evallib  # noqa: E402
from edge_voice_notes.engines.fake import FakeSummariser  # noqa: E402

TH = json.loads((ROOT / "evals" / "thresholds.json").read_text())


class TestWer(unittest.TestCase):
    def test_identical_after_normalisation(self):
        self.assertEqual(evallib.wer("Ask not, what your country...", "ask NOT what your country")["wer"], 0.0)

    def test_counts(self):
        r = evallib.wer("the cat sat on the mat", "the cat sat on mat now")
        self.assertEqual(r["sub"] + r["del"] + r["ins"], 2)
        self.assertAlmostEqual(r["wer"], 2 / 6, places=3)
        r = evallib.wer("one two three", "one three")
        self.assertEqual((r["sub"], r["del"], r["ins"]), (0, 1, 0))

    def test_brackets_dropped(self):
        self.assertEqual(evallib.normalize("[BLANK_AUDIO] Hello (laughs) world!"), "hello world")


class TestDataset(unittest.TestCase):
    def test_cases_are_consistent(self):
        cases = evallib.load_cases()
        self.assertGreaterEqual(len(cases), 4)
        for c in cases:
            lines = {t for t, _ in c["transcript"]}
            text = " ".join(x for _, x in c["transcript"]).lower()
            for item in c["actions"] + c["decisions"] + c["traps"]:
                self.assertIn(item["t"], lines, c["id"])
            for a in c["actions"]:
                self.assertTrue(a["owner"] or a["due"], f"{c['id']}: an action needs an owner or a deadline")
                for v in (a["owner"], a["due"]):
                    if v:
                        self.assertIn(v.lower().split()[-1], text, c["id"])
        self.assertTrue(any(c["traps"] for c in cases), "need at least one trap case")


class TestRegressionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = evallib.evaluate(FakeSummariser(), evallib.load_cases())

    def test_current_pipeline(self):
        s = self.res["current"]["summary"]
        self.assertLessEqual(s["invented_actions"], TH["max_invented_actions"])
        self.assertLessEqual(s["trap_leaks"], TH["max_trap_leaks"])
        self.assertGreaterEqual(s["action_precision"], TH["min_action_precision"])
        self.assertGreaterEqual(s["action_recall"], TH["min_action_recall"])
        self.assertGreaterEqual(s["decision_recall"], TH["min_decision_recall"])
        self.assertGreaterEqual(s["ts_valid"], TH["min_ts_valid"])

    def test_baseline_shows_the_problem(self):
        # If the naive pipeline stops leaking traps, the trap cases no longer test anything.
        if TH["baseline_must_leak_traps"]:
            self.assertGreater(self.res["baseline"]["summary"]["trap_leaks"], 0)


if __name__ == "__main__":
    unittest.main()
