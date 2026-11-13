import struct
import unittest
import wave

from edge_voice_notes import audio
from helpers import TmpCase


class AudioTests(TmpCase):
    def test_repair_header_after_power_cut(self):
        p = self.wav("a.wav", 3.0)
        with open(p, "r+b") as f:
            f.seek(4)
            f.write(struct.pack("<I", 0))   # RIFF size never patched
            f.seek(40)
            f.write(struct.pack("<I", 0))   # data size never patched
            f.seek(0, 2)
            f.write(b"\x01")                # half a frame at the tail
        frames = audio.repair_header(p)
        self.assertEqual(frames, 3 * audio.SAMPLE_RATE)
        with wave.open(str(p)) as w:
            self.assertEqual(w.getnframes(), frames)

    def test_repair_rejects_garbage(self):
        p = self.tmp / "x.wav.part"
        p.write_bytes(b"\0" * 100)
        with self.assertRaises(audio.AudioError):
            audio.repair_header(p)

    def test_format_check(self):
        p = self.tmp / "s.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\0" * 400)
        with self.assertRaisesRegex(audio.AudioError, "16000 Hz mono"):
            audio.check_format(p)

    def test_chunks_cover_file_and_cut_in_silence(self):
        p = self.wav("long.wav", 95.0, seed=3)
        plan = audio.plan_chunks(p, 20_000)
        self.assertEqual(plan, audio.plan_chunks(p, 20_000))  # deterministic
        self.assertEqual(plan[0][0], 0)
        self.assertEqual(plan[-1][1], audio.info(p).frames)
        for (a, b), (c, _) in zip(plan, plan[1:]):
            self.assertEqual(b, c)
            win = audio._read_frames(p, b - 400, 800)
            self.assertLess(audio._rms(win), 100, "cut should fall in a silent gap")
        for a, b in plan:
            self.assertLessEqual((b - a) / audio.SAMPLE_RATE, 25.0)

    def test_short_file_single_chunk(self):
        p = self.wav("s.wav", 7.0)
        self.assertEqual(audio.plan_chunks(p, 300_000), [(0, 7 * audio.SAMPLE_RATE)])


if __name__ == "__main__":
    unittest.main()
