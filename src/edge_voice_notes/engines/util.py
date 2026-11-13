from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_sha256(path: Path) -> str | None:
    """Model identity for status.json. Cached per (path, size, mtime)."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return _sha256(str(path), st.st_size, st.st_mtime_ns)


def tool_version(cmd: list[str]) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (p.stdout + p.stderr).strip().splitlines()
    for line in out:
        if re.search(r"version|build", line, re.I):
            return line.strip()[:200]
    return out[0][:200] if out and p.returncode == 0 else None


def extract_json_object(text: str) -> dict:
    """Parse the first JSON object in an LLM reply (tolerates code fences and chatter)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in model output")
