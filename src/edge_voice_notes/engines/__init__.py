"""Engine registry. Add a backend by writing a Transcriber/Summariser subclass and
registering it here; the config file selects it by name."""

from __future__ import annotations

from typing import Any, Callable

from .base import EngineError, Segment, Summariser, Transcriber
from .fake import FakeSummariser, FakeTranscriber
from .llama_server import LlamaServer
from .whisper_cpp import WhisperCpp

TRANSCRIBERS: dict[str, Callable[..., Transcriber]] = {
    "whisper_cpp": WhisperCpp,
    "fake": FakeTranscriber,
}
SUMMARISERS: dict[str, Callable[..., Summariser]] = {
    "llama_server": LlamaServer,
    "fake": FakeSummariser,
}


def _make(table: dict, kind: str, cfg: dict[str, Any]):
    cfg = dict(cfg)
    name = cfg.pop("engine", "fake")
    try:
        cls = table[name]
    except KeyError:
        raise ValueError(f"unknown {kind} engine {name!r}; known: {', '.join(sorted(table))}") from None
    return cls(**cfg)


def make_transcriber(cfg: dict[str, Any]) -> Transcriber:
    return _make(TRANSCRIBERS, "transcriber", cfg)


def make_summariser(cfg: dict[str, Any]) -> Summariser:
    return _make(SUMMARISERS, "summariser", cfg)


__all__ = ["EngineError", "Segment", "Summariser", "Transcriber", "make_transcriber",
           "make_summariser", "TRANSCRIBERS", "SUMMARISERS"]
