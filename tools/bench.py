#!/usr/bin/env python3
"""Measure the two expensive phases with the real engines on this machine.

  transcription: whisper.cpp on a WAV -> real-time factor, peak RSS of whisper-cli
  summary:       llama-server on a timestamped transcript -> seconds, kept/rejected actions

Usage:
  PYTHONPATH=src tools/bench.py --whisper-bin build/bin/whisper-cli \
      --whisper-model models/ggml-base.en.bin --audio rec.wav \
      --llama-url http://127.0.0.1:8080 --transcript bench/meeting_en.txt --out bench/out
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import time
from pathlib import Path

from edge_voice_notes import audio, notes
from edge_voice_notes.engines.base import Segment
from edge_voice_notes.engines.llama_server import LlamaServer
from edge_voice_notes.engines.whisper_cpp import WhisperCpp
from edge_voice_notes.fsutil import atomic_write_json, atomic_write_text


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("model name", "Model")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def proc_hwm_mb(name: str) -> float | None:
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if name in (d / "cmdline").read_bytes().split(b"\0")[0].decode(errors="ignore"):
                for line in (d / "status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            continue
    return None


def load_transcript(path: Path) -> list[Segment]:
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"\[(\d+:\d\d(?::\d\d)?)\]\s*(.+)", line)
        if m:
            rows.append((notes.parse_ts(m.group(1)), m.group(2)))
    return [Segment(t, rows[i + 1][0] if i + 1 < len(rows) else t + 5000, x) for i, (t, x) in enumerate(rows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-bin")
    ap.add_argument("--whisper-model")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--llama-url")
    ap.add_argument("--transcript", type=Path)
    ap.add_argument("--section-max-chars", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("bench/out"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res: dict = {"machine": cpu_name(), "cores": os.cpu_count(), "arch": platform.machine()}

    if a.audio:
        w = WhisperCpp(binary=a.whisper_bin, model=a.whisper_model, threads=a.threads, language="en")
        dur = audio.check_format(a.audio).duration_ms / 1000
        t0 = time.monotonic()
        segs = w.transcribe(a.audio)
        dt = time.monotonic() - t0
        res["transcribe"] = {"model": Path(a.whisper_model).name, "threads": w.threads,
                             "audio_s": round(dur, 1), "wall_s": round(dt, 2), "rtf": round(dt / dur, 3),
                             "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024),
                             "binary": w.info()["binary_version"]}
        atomic_write_text(a.out / "bench_transcript.txt", notes.render_transcript_txt(segs))

    if a.transcript:
        s = LlamaServer(url=a.llama_url)
        segs = load_transcript(a.transcript)
        secs = notes.split_sections(segs, a.section_max_chars)
        t0 = time.monotonic()
        results = [notes.validate_section(s.complete_json(notes.SYSTEM, notes.section_prompt(segs, sec),
                                                          notes.SECTION_SCHEMA), segs, sec) for sec in secs]
        overview = []
        if len(results) > 1:
            overview = notes.validate_overview(s.complete_json(notes.SYSTEM, notes.overview_prompt(results),
                                                               notes.OVERVIEW_SCHEMA), segs)
        dt = time.monotonic() - t0
        acts = notes.collect(results)
        atomic_write_text(a.out / "bench_summary.md",
                          notes.render_summary("bench", segs[-1].end_ms, overview, results, acts))
        atomic_write_json(a.out / "bench_actions.json", acts)
        info = s.info()
        res["summarise"] = {"model": info.get("model"), "build": info.get("build_info"),
                            "transcript_s": round(segs[-1].end_ms / 1000), "sections": len(secs),
                            "wall_s": round(dt, 1), "actions_kept": len(acts["actions"]),
                            "actions_rejected": len(acts["rejected_actions"]),
                            "llama_server_peak_rss_mb": round(proc_hwm_mb("llama-server") or 0)}

    atomic_write_json(a.out / "bench.json", res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
