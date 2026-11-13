"""llama.cpp through `llama-server` (OpenAI-compatible HTTP on localhost).

The server keeps the model mapped between jobs, so a queue of recordings does
not pay the model load time on each one. Output is constrained to the JSON
schema by the server's grammar sampler, then validated again on our side.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import EngineError, Summariser
from .util import extract_json_object, file_sha256


class LlamaServer(Summariser):
    def __init__(self, url: str = "http://127.0.0.1:8080", model_path: str = "",
                 temperature: float = 0.0, max_tokens: int = 1024, timeout_s: float = 900,
                 seed: int = 42) -> None:
        self.url = url.rstrip("/")
        self.model_path = model_path
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.seed = int(seed)

    def _req(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            raise EngineError(f"llama-server {path}: {e}") from e

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object", "schema": schema},
        }
        r = self._req("/v1/chat/completions", body)
        try:
            content = r["choices"][0]["message"]["content"]
            return extract_json_object(content)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise EngineError(f"llama-server returned unusable output: {e}") from e

    def info(self) -> dict[str, Any]:
        d: dict[str, Any] = {"engine": "llama.cpp (llama-server)", "url": self.url}
        try:
            props = self._req("/props", timeout=5)
            d["model"] = Path(props.get("model_path", "")).name or None
            d["build_info"] = props.get("build_info")
            mp = props.get("model_path") or self.model_path
        except EngineError:
            mp = self.model_path
        if mp:
            d.setdefault("model", Path(mp).name)
            d["model_sha256"] = file_sha256(Path(mp))
        d["temperature"] = self.temperature
        d["seed"] = self.seed
        return d
