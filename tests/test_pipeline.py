import json
import shutil
import struct
import unittest

from edge_voice_notes import audio
from edge_voice_notes.pipeline import Pipeline
from helpers import TmpCase

OUTPUTS = ("audio.wav", "transcript.json", "transcript.txt", "summary.md", "actions.json", "status.json")


class PipelineTests(TmpCase):
    def test_end_to_end_outputs(self):
        p = Pipeline(self.config())
        rid = p.ingest(self.wav("m.wav", 95.0))
        self.assertEqual(p.run(once=True), 1)
        rec = p.paths.rec(rid)
        for f in OUTPUTS:
            self.assertTrue((rec / f).exists(), f)
        self.assertFalse((rec / ".work").exists())
        tr = json.loads((rec / "transcript.json").read_text())
        starts = [s["start_ms"] for s in tr["segments"]]
        self.assertEqual(starts, sorted(starts))
        self.assertGreater(tr["segments"][-1]["end_ms"], 85_000, "chunk offsets must be applied")
        self.assertLessEqual(tr["segments"][-1]["end_ms"], 95_000)
        st = json.loads((rec / "status.json").read_text())
        self.assertEqual(st["state"], "done")
        self.assertEqual(st["engines"]["transcriber"]["engine"], "fake")
        self.assertEqual(st["engines"]["summariser"]["engine"], "fake")
        self.assertGreater(st["metrics"]["sections"], 1)
        self.assertIn("transcribe_rtf", st["metrics"])
        acts = json.loads((rec / "actions.json").read_text())
        for a in acts["actions"]:
            self.assertTrue(a["owner"] or a["due"])
        md = (rec / "summary.md").read_text()
        self.assertIn("## Overview", md)
        self.assertIn("## Action items", md)

    def test_queue_order_and_status(self):
        p = Pipeline(self.config())
        ids = [p.ingest(self.wav(f"{i}.wav", 12.0, seed=i), rec_id=f"r{i}") for i in range(3)]
        self.assertEqual([j.state for j in p.queue.all()], ["pending"] * 3)
        p.run(once=True)
        self.assertEqual([j.id for j in p.queue.all()], ids)
        self.assertEqual({j.state for j in p.queue.all()}, {"done"})

    def test_bad_audio_is_parked_not_retried(self):
        p = Pipeline(self.config())
        rid = p.ingest(self.wav("ok.wav", 5.0))
        (p.paths.rec(rid) / "audio.wav").write_bytes(b"RIFF....garbage")
        p.run(once=True)
        j = p.queue.get(rid)
        self.assertEqual(j.state, "failed")
        self.assertEqual(json.loads((p.paths.rec(rid) / "status.json").read_text())["state"], "failed")

    def test_engine_error_retried_then_failed(self):
        from edge_voice_notes.engines import EngineError
        from edge_voice_notes.engines.fake import FakeTranscriber

        class Down(FakeTranscriber):
            calls = 0

            def transcribe(self, wav):
                Down.calls += 1
                raise EngineError("server down")

        p = Pipeline(self.config(max_attempts=2), transcriber=Down())
        rid = p.ingest(self.wav("a.wav", 5.0))
        p.run(once=True)
        self.assertEqual(Down.calls, 2)
        self.assertEqual(p.queue.get(rid).state, "failed")
        self.assertIn("server down", p.queue.get(rid).last_error)
        self.assertTrue(p.queue.retry(rid))
        self.assertEqual(p.queue.get(rid).state, "pending")

    def test_recover(self):
        p = Pipeline(self.config())
        # 1. recording interrupted by a power cut: header never finalised
        part = p.paths.incoming / "20261113T100000Z-aaaaaa.wav.part"
        shutil.copy(self.wav("x.wav", 8.0), part)
        with open(part, "r+b") as f:
            f.seek(40)
            f.write(struct.pack("<I", 0))
        # 2. empty .part (power cut before any audio)
        (p.paths.incoming / "e.wav.part").write_bytes(b"")
        # 3. audio ingested but job row lost
        orphan = p.paths.rec("orphan")
        orphan.mkdir()
        shutil.copy(self.wav("y.wav", 4.0), orphan / "audio.wav")
        did = p.recover()
        self.assertEqual(len(did), 3, did)
        self.assertEqual({j.id for j in p.queue.all()}, {"20261113T100000Z-aaaaaa", "orphan"})
        self.assertEqual(audio.info(p.paths.rec("20261113T100000Z-aaaaaa") / "audio.wav").frames,
                         8 * audio.SAMPLE_RATE)
        self.assertEqual(list(p.paths.incoming.iterdir()), [p.paths.incoming / "e.wav.bad"])
        p.run(once=True)
        self.assertEqual({j.state for j in p.queue.all()}, {"done"})
        self.assertEqual(p.recover(), [])

    def test_resume_reuses_checkpoints(self):
        from edge_voice_notes.engines.fake import FakeTranscriber

        class Counting(FakeTranscriber):
            n = 0

            def transcribe(self, wav):
                Counting.n += 1
                if Counting.n == 4:
                    raise KeyboardInterrupt  # stand-in for the process dying mid-job
                return super().transcribe(wav)

        cfg = self.config()
        p = Pipeline(cfg, transcriber=Counting())
        rid = p.ingest(self.wav("m.wav", 60.0))
        with self.assertRaises(KeyboardInterrupt):
            p.run(once=True)
        chunks = len(json.loads((p.paths.rec(rid) / ".work" / "chunks.json").read_text())["chunks"])
        p.run(once=True)
        self.assertEqual(Counting.n, chunks + 1, "finished chunks must not be transcribed again")
        self.assertEqual(p.queue.get(rid).state, "done")


if __name__ == "__main__":
    unittest.main()
