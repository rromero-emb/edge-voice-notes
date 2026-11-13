import unittest

from edge_voice_notes import notes
from edge_voice_notes.engines.base import Segment


def segs():
    lines = [
        "Let's go through the battery numbers.",
        "Maria will send the current profile by Friday.",
        "Someone should check the microphone gain.",
        "We agreed to keep networking off by default.",
        "Javier will order enclosures.",
        "I will look into it.",
    ]
    return [Segment(i * 7000, i * 7000 + 6500, t) for i, t in enumerate(lines)]


class TimestampTests(unittest.TestCase):
    def test_roundtrip(self):
        for ms in (0, 59_000, 61_000, 3_599_000, 3_600_000, 7_384_000):
            self.assertEqual(notes.parse_ts(notes.fmt_ts(ms)), ms)

    def test_bad(self):
        for t in (None, "", "abc", "12", "1:75", 12):
            self.assertIsNone(notes.parse_ts(t))
        self.assertEqual(notes.parse_ts("[01:05]"), 65_000)
        self.assertEqual(notes.parse_ts("00:25-00:32"), 25_000)
        self.assertEqual(notes.parse_ts("1:02:03.5"), 3_723_000)


class SectionTests(unittest.TestCase):
    def test_split_keeps_order_and_bounds(self):
        s = segs() * 20
        s = [Segment(i * 1000, i * 1000 + 900, x.text) for i, x in enumerate(s)]
        secs = notes.split_sections(s, max_chars=500)
        self.assertGreater(len(secs), 3)
        self.assertEqual(secs[0].first, 0)
        self.assertEqual(secs[-1].last, len(s) - 1)
        for a, b in zip(secs, secs[1:]):
            self.assertEqual(a.last + 1, b.first)
            self.assertLessEqual(len(notes.section_lines(s, a)), 500 + 60)

    def test_one_huge_segment_still_gets_a_section(self):
        s = [Segment(0, 1000, "x" * 10000)]
        self.assertEqual(len(notes.split_sections(s, 100)), 1)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.s = segs()
        self.sec = notes.split_sections(self.s, 10_000)[0]

    def v(self, actions=(), points=(), decisions=()):
        return notes.validate_section({"actions": list(actions), "points": list(points),
                                       "decisions": list(decisions)}, self.s, self.sec)

    def test_keeps_explicit_owner_and_deadline(self):
        r = self.v([{"task": "send the current profile", "owner": "Maria", "due": "Friday", "t": "00:07"}])
        self.assertEqual(r["actions"], [{"task": "send the current profile", "owner": "Maria",
                                         "due": "Friday", "t_ms": 7000}])

    def test_owner_only_or_due_only_is_enough(self):
        r = self.v([{"task": "order enclosures", "owner": "Javier", "due": None, "t": "00:28"}])
        self.assertEqual(len(r["actions"]), 1)
        self.assertIsNone(r["actions"][0]["due"])

    def test_drops_vague_and_invented(self):
        r = self.v([
            {"task": "check mic gain", "owner": None, "due": None, "t": "00:14"},
            {"task": "check mic gain", "owner": "Pedro", "due": "Monday", "t": "00:14"},  # invented
            {"task": "look into it", "owner": "I", "due": None, "t": "00:35"},              # pronoun
            {"task": "send profile", "owner": "Maria", "due": "next week", "t": "00:07"},  # wrong due
        ])
        self.assertEqual(len(r["actions"]), 1)
        self.assertEqual(r["actions"][0]["owner"], "Maria")
        self.assertIsNone(r["actions"][0]["due"], "invented deadline must be removed")
        self.assertEqual(len(r["rejected_actions"]), 3)

    def test_reworded_deadline_needs_the_same_words(self):
        self.s[4] = Segment(28000, 34500, "Javier will order enclosures before the 30th of November.")
        ok = self.v([{"task": "order", "owner": "Javier", "due": "November 30th", "t": "00:28"}])
        self.assertEqual(ok["actions"][0]["due"], "November 30th")
        bad = self.v([{"task": "order", "owner": "Javier", "due": "November 15th", "t": "00:28"}])
        self.assertIsNone(bad["actions"][0]["due"])
        ts = self.v([{"task": "order", "owner": "Javier", "due": "00:28", "t": "00:28"}])
        self.assertIsNone(ts["actions"][0]["due"])

    def test_owner_must_be_whole_word(self):
        r = self.v([{"task": "x", "owner": "Mar", "due": None, "t": "00:07"}])
        self.assertEqual(r["actions"], [])

    def test_timestamps_snap_to_real_segments(self):
        r = self.v(points=[{"text": "a", "t": "00:09"}, {"text": "b", "t": "nonsense"},
                           {"text": "c", "t": "59:00"}])
        self.assertEqual([p["t_ms"] for p in r["points"]], [7000, 35000])

    def test_cited_neighbour_line_is_corrected(self):
        r = self.v(points=[{"text": "Networking stays off by default", "t": "00:14"},
                           {"text": "Maria sends the profile", "t": "garbage"},
                           {"text": "unrelated words entirely", "t": "garbage"}])
        self.assertEqual([p["t_ms"] for p in r["points"]], [21000, 7000])

    def test_duplicates_removed(self):
        r = self.v(points=[{"text": "Same.", "t": "00:00"}, {"text": "same", "t": "00:07"}])
        self.assertEqual(len(r["points"]), 1)

    def test_garbage_model_output(self):
        r = notes.validate_section({"points": "oops", "actions": [1, None, "x"]}, self.s, self.sec)
        self.assertEqual(r["points"], [])
        self.assertEqual(r["actions"], [])


if __name__ == "__main__":
    unittest.main()
