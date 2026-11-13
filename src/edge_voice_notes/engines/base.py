"""Engine interfaces. The pipeline only talks to these two classes, so whisper.cpp
and llama.cpp can be swapped for an NPU backend, a different model server or the
fake engines used in CI without touching the queue or the output format."""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Segment:
    start_ms: int
    end_ms: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Segment":
        return Segment(int(d["start_ms"]), int(d["end_ms"]), str(d["text"]))


class EngineError(Exception):
    """Recoverable engine failure (bad output, server down). The job is retried."""


class Transcriber(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, wav: Path) -> list[Segment]:
        """Transcribe one 16 kHz mono chunk. Times are relative to the chunk start."""

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Engine and model identity, written to status.json."""


class Summariser(abc.ABC):
    @abc.abstractmethod
    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """Return a JSON object that follows `schema`."""

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Engine and model identity, written to status.json."""
