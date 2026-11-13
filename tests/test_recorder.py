"""Recorder against a stand-in for arecord that writes a WAV with an unfinalised
header (sizes = 0) and only patches it on SIGINT, like the real tool."""

import stat
import sys
import unittest

from edge_voice_notes import audio
from edge_voice_notes.recorder import record
from helpers import TmpCase

FAKE_ARECORD = r'''#!PYTHON
import signal, struct, sys, time
path = sys.argv[-1]
f = open(path, "wb")
f.write(b"RIFF" + struct.pack("<I", 0) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data" + struct.pack("<I", 0))
n = 0
def stop(*_):
    f.seek(4); f.write(struct.pack("<I", 36 + n)); f.seek(40); f.write(struct.pack("<I", n)); f.close()
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
while True:
    f.write(b"\x10\x00" * 1600); n += 3200   # 100 ms
    time.sleep(0.1)
'''


class RecorderTests(TmpCase):
    def fake(self):
        p = self.tmp / "arecord"
        p.write_text(FAKE_ARECORD.replace("PYTHON", sys.executable))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return str(p)

    def test_timed_recording_is_valid_wav(self):
        part = record(self.tmp, "r1", seconds=1.0, fsync_seconds=0.2, binary=self.fake())
        i = audio.check_format(part)
        self.assertGreater(i.duration_ms, 500)

    def test_unfinalised_file_is_repairable(self):
        import os
        import signal
        import subprocess
        import time
        part = self.tmp / "r2.wav.part"
        p = subprocess.Popen([self.fake(), str(part)])
        time.sleep(0.6)
        os.kill(p.pid, signal.SIGKILL)  # no chance to patch the header: like a power cut
        p.wait()
        self.assertRaises(audio.AudioError, audio.check_format, part)
        self.assertGreater(audio.repair_header(part), 0)
        audio.check_format(part)


if __name__ == "__main__":
    unittest.main()
