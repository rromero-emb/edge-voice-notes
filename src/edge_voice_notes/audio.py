"""WAV handling: validation, header repair after power loss, silence-aware chunking.

whisper.cpp wants 16 kHz, mono, 16-bit PCM, so that is the only format the
pipeline accepts. Everything here is stdlib so it runs on a bare board image.
"""

from __future__ import annotations

import array
import os
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from .fsutil import fsync_dir

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPWIDTH = 2


class AudioError(Exception):
    pass


@dataclass(frozen=True)
class AudioInfo:
    frames: int
    rate: int
    channels: int
    sampwidth: int

    @property
    def duration_ms(self) -> int:
        return self.frames * 1000 // self.rate


def info(path: Path) -> AudioInfo:
    try:
        with wave.open(str(path), "rb") as w:
            return AudioInfo(w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth())
    except (wave.Error, EOFError) as e:
        raise AudioError(f"{path}: not a readable WAV ({e})") from e


def check_format(path: Path) -> AudioInfo:
    i = info(path)
    if (i.rate, i.channels, i.sampwidth) != (SAMPLE_RATE, CHANNELS, SAMPWIDTH):
        raise AudioError(
            f"{path}: need {SAMPLE_RATE} Hz mono 16-bit PCM, got {i.rate} Hz, "
            f"{i.channels} ch, {8 * i.sampwidth}-bit (convert with: "
            f"ffmpeg -i in -ar 16000 -ac 1 -c:a pcm_s16le out.wav)"
        )
    if i.frames == 0:
        raise AudioError(f"{path}: empty recording")
    return i


def repair_header(path: Path) -> int:
    """Fix RIFF/data sizes of a WAV whose writer died before finalising it.

    arecord (and most recorders) write the header first and patch the sizes on
    a clean stop. After a power cut the sizes are 0 or bogus while the samples
    are on disk up to the last flushed page. Returns the number of whole frames
    kept. Raises AudioError if there is no usable header at all.
    """
    with open(path, "r+b") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise AudioError(f"{path}: no RIFF/WAVE header")
        block_align = None
        pos = 12
        size = os.fstat(f.fileno()).st_size
        while pos + 8 <= size:
            f.seek(pos)
            cid, clen = struct.unpack("<4sI", f.read(8))
            if cid == b"fmt ":
                fmt = f.read(16)
                block_align = struct.unpack("<H", fmt[12:14])[0]
            if cid == b"data":
                if not block_align:
                    raise AudioError(f"{path}: data chunk before fmt chunk")
                data_start = pos + 8
                avail = size - data_start
                frames = avail // block_align
                data_len = frames * block_align
                f.truncate(data_start + data_len)
                f.seek(pos + 4)
                f.write(struct.pack("<I", data_len))
                f.seek(4)
                f.write(struct.pack("<I", data_start + data_len - 8))
                f.flush()
                os.fsync(f.fileno())
                return frames
            pos += 8 + clen + (clen & 1)
    raise AudioError(f"{path}: no data chunk")


def _read_frames(path: Path, start: int, count: int) -> bytes:
    with wave.open(str(path), "rb") as w:
        w.setpos(start)
        return w.readframes(count)


def _rms(pcm: bytes) -> float:
    a = array.array("h")
    a.frombytes(pcm)
    if sys.byteorder == "big":
        a.byteswap()
    if not a:
        return 0.0
    return (sum(x * x for x in a) / len(a)) ** 0.5


def plan_chunks(path: Path, chunk_ms: int, search_ms: int = 5000, win_ms: int = 100) -> list[tuple[int, int]]:
    """Split a recording into ~chunk_ms pieces, cutting in the quietest window
    within +/- search_ms of each nominal boundary so words are not split.

    Returns [(start_frame, end_frame), ...] covering the whole file. The plan is
    deterministic for a given file, which matters: it is persisted once and a
    resumed job must transcribe exactly the same chunks.
    """
    i = check_format(path)
    chunk = chunk_ms * i.rate // 1000
    search = search_ms * i.rate // 1000
    win = max(1, win_ms * i.rate // 1000)
    bounds = [0]
    while i.frames - bounds[-1] > chunk + search:
        nominal = bounds[-1] + chunk
        lo, hi = nominal - search, nominal + search
        pcm = _read_frames(path, lo, hi - lo)
        best, best_rms = nominal, float("inf")
        for off in range(0, hi - lo - win + 1, win):
            r = _rms(pcm[off * 2:(off + win) * 2])
            if r < best_rms:
                best, best_rms = lo + off + win // 2, r
        bounds.append(best)
    bounds.append(i.frames)
    return list(zip(bounds[:-1], bounds[1:]))


def extract(path: Path, start: int, end: int, dst: Path) -> None:
    pcm = _read_frames(path, start, end - start)
    tmp = dst.with_name(f".{dst.name}.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPWIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    fsync_dir(dst.parent)


def write_tone_bursts(dst: Path, seconds: float, burst_s: float = 4.0, gap_s: float = 0.6, seed: int = 1) -> None:
    """Synthetic 'speech-like' audio (tone bursts separated by silence) for tests."""
    import math
    import random

    rnd = random.Random(seed)
    n = int(seconds * SAMPLE_RATE)
    a = array.array("h", bytes(2 * n))
    t = 0
    while t < n:
        blen = int(rnd.uniform(0.5, 1.5) * burst_s * SAMPLE_RATE)
        f0 = rnd.uniform(120, 300)
        for k in range(t, min(n, t + blen)):
            a[k] = int(6000 * math.sin(2 * math.pi * f0 * k / SAMPLE_RATE))
        t += blen + int(gap_s * SAMPLE_RATE)
    if sys.byteorder == "big":
        a.byteswap()
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPWIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(a.tobytes())
