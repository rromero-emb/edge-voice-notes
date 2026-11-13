from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(os.environ.get("EVN_CONFIG", "/etc/edge-voice-notes/config.toml"))

EXAMPLE = """\
# edge-voice-notes configuration
data_dir = "/var/lib/edge-voice-notes"
chunk_seconds = 300        # transcription checkpoint granularity
section_max_chars = 3000   # transcript characters per LLM call
max_attempts = 3           # crashes without progress before a recording is parked as failed

[recorder]
device = "default"         # ALSA device, e.g. "hw:1,0" for a USB microphone
fsync_seconds = 5          # at most this much audio is lost on a power cut

[transcriber]
engine = "whisper_cpp"     # or "fake"
binary = "/usr/local/bin/whisper-cli"
model = "/opt/models/ggml-base.en.bin"
language = "auto"
threads = 0                # 0 = all cores

[summariser]
engine = "llama_server"    # or "fake"
url = "http://127.0.0.1:8080"
temperature = 0.0
max_tokens = 1024
"""


@dataclass
class Config:
    data_dir: Path = Path("/var/lib/edge-voice-notes")
    chunk_seconds: int = 300
    section_max_chars: int = 3000
    max_attempts: int = 3
    recorder: dict[str, Any] = field(default_factory=lambda: {"device": "default", "fsync_seconds": 5})
    transcriber: dict[str, Any] = field(default_factory=lambda: {"engine": "fake"})
    summariser: dict[str, Any] = field(default_factory=lambda: {"engine": "fake"})

    @staticmethod
    def load(path: Path | None) -> "Config":
        path = Path(path) if path else DEFAULT_PATH
        with open(path, "rb") as f:
            d = tomllib.load(f)
        c = Config()
        for k, v in d.items():
            if not hasattr(c, k):
                raise ValueError(f"{path}: unknown key {k!r}")
            setattr(c, k, Path(v) if k == "data_dir" else v)
        if not Path(c.data_dir).is_absolute():
            c.data_dir = (path.parent / c.data_dir).resolve()
        return c
