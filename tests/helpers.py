from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from edge_voice_notes import audio
from edge_voice_notes.config import Config

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def write_config(path: Path, data_dir: Path, *, chunk_seconds=10, section_max_chars=600, max_attempts=3,
                 t_extra="", s_extra="") -> Path:
    path.write_text(f"""
data_dir = "{data_dir}"
chunk_seconds = {chunk_seconds}
section_max_chars = {section_max_chars}
max_attempts = {max_attempts}

[transcriber]
engine = "fake"
{t_extra}

[summariser]
engine = "fake"
{s_extra}
""")
    return path


def run_cli(cfg: Path, *args: str, check=True, **kw) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(SRC))
    return subprocess.run([sys.executable, "-m", "edge_voice_notes", "-c", str(cfg), *args],
                          env=env, capture_output=True, text=True, check=check, **kw)


def popen_cli(cfg: Path, *args: str) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=str(SRC))
    return subprocess.Popen([sys.executable, "-m", "edge_voice_notes", "-c", str(cfg), *args],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory(prefix="evn-test-")
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def wav(self, name: str, seconds: float, seed: int = 1) -> Path:
        p = self.tmp / name
        audio.write_tone_bursts(p, seconds, seed=seed)
        return p

    def config(self, **kw) -> Config:
        c = Config()
        c.data_dir = self.tmp / "data"
        c.chunk_seconds = kw.pop("chunk_seconds", 10)
        c.section_max_chars = kw.pop("section_max_chars", 600)
        for k, v in kw.items():
            setattr(c, k, v)
        return c
