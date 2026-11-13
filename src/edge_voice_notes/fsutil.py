"""Crash-safe file helpers.

Every artefact the pipeline produces is written with write-to-temp, fsync,
rename, fsync-directory. After a power cut a file is either the old version,
the new version, or absent, never half written.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_copy(src: Path, dst: Path) -> None:
    dst = Path(dst)
    tmp = dst.with_name(f".{dst.name}.tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)
    fsync_dir(dst.parent)


def mkdir_durable(path: Path) -> None:
    path = Path(path)
    if path.is_dir():
        return
    mkdir_durable(path.parent)
    path.mkdir(exist_ok=True)
    fsync_dir(path.parent)
