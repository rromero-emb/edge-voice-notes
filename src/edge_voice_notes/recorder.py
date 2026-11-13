"""Record from ALSA with arecord into incoming/<id>.wav.part.

The parent fsyncs the growing file every `fsync_seconds`, so a power cut loses
at most that much audio; recover() repairs the WAV header on the next boot.
Stop with SIGINT/SIGTERM (a button handler or `systemctl stop`) or --seconds.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from .fsutil import fsync_dir


def record(incoming: Path, rec_id: str, device: str = "default", seconds: float | None = None,
           fsync_seconds: float = 5.0, binary: str = "arecord") -> Path:
    part = incoming / f"{rec_id}.wav.part"
    cmd = [binary, "-q", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", str(part)]
    proc = subprocess.Popen(cmd)
    stopping = False

    def _stop(signum, frame):
        nonlocal stopping
        stopping = True

    old = {s: signal.signal(s, _stop) for s in (signal.SIGINT, signal.SIGTERM)}
    t_end = time.monotonic() + seconds if seconds else None
    try:
        fd, last_sync = None, time.monotonic()
        while proc.poll() is None:
            if stopping or (t_end and time.monotonic() >= t_end):
                proc.send_signal(signal.SIGINT)  # arecord finalises the header on SIGINT
                proc.wait(timeout=10)
                break
            if fd is None and part.exists():
                fd = os.open(part, os.O_RDONLY)
                fsync_dir(incoming)
            if fd is not None and time.monotonic() - last_sync >= fsync_seconds:
                os.fsync(fd)
                last_sync = time.monotonic()
            time.sleep(0.1)
        if fd is not None:
            os.fsync(fd)
            os.close(fd)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    if proc.returncode not in (0, None) and not stopping and not (t_end and time.monotonic() >= t_end):
        raise RuntimeError(f"arecord exited with {proc.returncode}")
    return part
