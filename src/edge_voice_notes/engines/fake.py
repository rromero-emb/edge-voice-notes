"""Deterministic fake engines for CI and for testing the queue on a board without
models. Same input, same output, so a resumed run can be compared byte for byte
with an uninterrupted one."""

from __future__ import annotations

import hashlib
import os
import re
import time
import wave
from pathlib import Path
from typing import Any

from .base import Segment, Summariser, Transcriber

CORPUS = [
    "Okay, let's start with the battery results from last week.",
    "The prototype ran for nine hours on the 3000 mAh cell.",
    "Transcription is the most expensive phase, roughly twice real time.",
    "We decided to keep the 16 GB board for the first batch.",
    "Maria will send the measured current profile by Friday.",
    "I think the USB gadget mode still needs checking on the new image.",
    "Someone should look at the microphone gain at some point.",
    "Javier will order ten more enclosures before the 30th of November.",
    "We agreed to keep networking disabled by default.",
    "The summary for a one hour meeting took about four minutes.",
    "Let's review the LUKS unlock flow next time.",
    "Ana will update the download page by next Tuesday.",
]

TS = re.compile(r"^\[(\d+):(\d\d)(?::(\d\d))?\]\s*(.*)$")
ACTION = re.compile(r"\b([A-Z][a-z]+) will (.+?)(?: (by|before) (.+?))?\.?$")


class FakeTranscriber(Transcriber):
    def __init__(self, segment_ms: int = 5000, delay_s: float = 0.0, crash_on: str = "", **_ignored) -> None:
        # **_ignored: flipping engine = "fake" in a real config just works
        self.segment_ms = int(segment_ms)
        self.delay_s = float(delay_s)
        self.crash_on = crash_on

    def transcribe(self, wav: Path) -> list[Segment]:
        if self.crash_on and self.crash_on in str(wav):
            os._exit(139)  # simulate the engine taking the whole device down
        if self.delay_s:
            time.sleep(self.delay_s)
        with wave.open(str(wav), "rb") as w:
            n = w.getnframes()
            dur = n * 1000 // w.getframerate()
            digest = hashlib.sha256(w.readframes(n)).digest()
        segs, t, k = [], 0, 0
        while t < dur:
            end = min(dur, t + self.segment_ms)
            idx = (digest[k % len(digest)] + k) % len(CORPUS)
            segs.append(Segment(t, end, CORPUS[idx]))
            t, k = end, k + 1
        return segs

    def info(self) -> dict[str, Any]:
        return {"engine": "fake", "model": "fake-transcriber", "version": "1"}


class FakeSummariser(Summariser):
    """Rule-based stand-in for the LLM. It reads the prompt like the model would."""

    def __init__(self, delay_s: float = 0.0, **_ignored) -> None:
        self.delay_s = float(delay_s)

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        if self.delay_s:
            time.sleep(self.delay_s)
        lines = []
        for raw in user.splitlines():
            m = TS.match(raw.strip())
            if m:
                ts = raw.strip()[1:].split("]")[0]
                lines.append((ts, m.group(4)))
        props = schema.get("properties", {})
        if "overview" in props:
            return {"overview": [{"text": text, "t": ts} for ts, text in lines[:3]]}
        out: dict[str, list] = {"points": [], "decisions": [], "actions": []}
        seen = set()
        for ts, text in lines:
            if text in seen:
                continue
            seen.add(text)
            low = text.lower()
            if "decided" in low or "agreed" in low:
                out["decisions"].append({"text": text, "t": ts})
            m = ACTION.search(text)
            if m:
                out["actions"].append({
                    "task": m.group(2), "owner": m.group(1),
                    "due": m.group(4) if m.group(4) else None, "t": ts,
                })
            elif "should" in low:
                # vague item with no owner and no date: a careless model would list it
                out["actions"].append({"task": text, "owner": None, "due": None, "t": ts})
            elif len(out["points"]) < 4:
                out["points"].append({"text": text, "t": ts})
        return out

    def info(self) -> dict[str, Any]:
        return {"engine": "fake", "model": "fake-summariser", "version": "1"}
