"""whisper.cpp through its `whisper-cli` binary.

A subprocess (not Python bindings) on purpose: if the model runs the board out
of memory, the OOM killer takes whisper-cli, not the queue worker, and the job
is retried or parked as failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from .base import EngineError, Segment, Transcriber
from .util import file_sha256, tool_version


class WhisperCpp(Transcriber):
    def __init__(self, binary: str = "whisper-cli", model: str = "", language: str = "auto",
                 threads: int = 0, extra_args: list[str] | None = None,
                 timeout_factor: float = 10.0) -> None:
        if not model:
            raise ValueError("whisper_cpp: 'model' (path to a ggml .bin) is required")
        self.binary = binary
        self.model = Path(model)
        self.language = language
        self.threads = int(threads) or max(1, (os.cpu_count() or 1))
        self.extra_args = list(extra_args or [])
        self.timeout_factor = float(timeout_factor)

    def transcribe(self, wav: Path) -> list[Segment]:
        with wave.open(str(wav), "rb") as w:
            dur_s = w.getnframes() / w.getframerate()
        with tempfile.TemporaryDirectory(prefix="evn-whisper-") as td:
            prefix = Path(td) / "out"
            cmd = [self.binary, "-m", str(self.model), "-f", str(wav), "-l", self.language,
                   "-t", str(self.threads), "-oj", "-of", str(prefix), "-np", *self.extra_args]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=60 + self.timeout_factor * dur_s)
            except subprocess.TimeoutExpired as e:
                raise EngineError(f"whisper-cli timed out after {e.timeout:.0f} s") from e
            except FileNotFoundError as e:
                raise EngineError(f"whisper-cli not found: {self.binary}") from e
            if p.returncode != 0:
                raise EngineError(f"whisper-cli exit {p.returncode}: {p.stderr.strip()[-400:]}")
            try:
                data = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError) as e:
                raise EngineError(f"whisper-cli produced no usable JSON: {e}") from e
        return parse_whisper_json(data)

    def info(self) -> dict[str, Any]:
        return {
            "engine": "whisper.cpp",
            "binary": self.binary,
            "binary_version": tool_version([self.binary, "--version"]),
            "model": self.model.name,
            "model_sha256": file_sha256(self.model),
            "language": self.language,
            "threads": self.threads,
        }


def parse_whisper_json(data: dict) -> list[Segment]:
    segs = []
    for item in data.get("transcription", []):
        text = item.get("text", "").strip()
        if not text or text in ("[BLANK_AUDIO]", "[ Silence ]"):
            continue
        off = item.get("offsets", {})
        segs.append(Segment(int(off.get("from", 0)), int(off.get("to", 0)), text))
    return segs
