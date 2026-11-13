"""The real adapters, exercised against stand-ins for the binaries/servers so CI
covers the subprocess and HTTP code paths without downloading models."""

import json
import os
import stat
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from edge_voice_notes.config import EXAMPLE, Config
from edge_voice_notes.engines import EngineError, make_summariser, make_transcriber
from edge_voice_notes.engines.util import extract_json_object
from edge_voice_notes.engines.whisper_cpp import parse_whisper_json
from helpers import TmpCase

WHISPER_JSON = {
    "model": {"type": "base"},
    "transcription": [
        {"offsets": {"from": 0, "to": 2100}, "text": " Hello there."},
        {"offsets": {"from": 2100, "to": 3000}, "text": " [BLANK_AUDIO]"},
        {"offsets": {"from": 3000, "to": 5200}, "text": " Maria will send it by Friday."},
    ],
}


class WhisperTests(TmpCase):
    def fake_cli(self, body: str):
        p = self.tmp / "whisper-cli"
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return p

    def test_parse(self):
        segs = parse_whisper_json(WHISPER_JSON)
        self.assertEqual([(s.start_ms, s.text) for s in segs], [(0, "Hello there."), (3000, "Maria will send it by Friday.")])

    def test_subprocess_path(self):
        js = json.dumps(WHISPER_JSON).replace("'", "")
        # write <prefix>.json where prefix follows -of, like whisper-cli does
        cli = self.fake_cli(f"""
while [ $# -gt 0 ]; do [ "$1" = "-of" ] && OF="$2"; shift; done
[ -n "$OF" ] && printf '%s' '{js}' > "$OF.json"; exit 0
""")
        model = self.tmp / "m.bin"
        model.write_bytes(b"model")
        t = make_transcriber({"engine": "whisper_cpp", "binary": str(cli), "model": str(model)})
        segs = t.transcribe(self.wav("a.wav", 2.0))
        self.assertEqual(len(segs), 2)
        info = t.info()
        self.assertEqual(info["model"], "m.bin")
        self.assertEqual(len(info["model_sha256"]), 64)

    def test_failure_is_engine_error(self):
        cli = self.fake_cli("echo boom >&2; exit 3\n")
        t = make_transcriber({"engine": "whisper_cpp", "binary": str(cli), "model": "x.bin"})
        with self.assertRaisesRegex(EngineError, "exit 3"):
            t.transcribe(self.wav("a.wav", 1.0))
        t = make_transcriber({"engine": "whisper_cpp", "binary": "/nonexistent", "model": "x.bin"})
        with self.assertRaises(EngineError):
            t.transcribe(self.wav("a.wav", 1.0))


class _Handler(BaseHTTPRequestHandler):
    reply = ""
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"model_path": "/opt/models/qwen.gguf", "build_info": "b9999-test"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append(body)
        self._send({"choices": [{"message": {"content": _Handler.reply}}]})


class LlamaServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_json_with_chatter(self):
        _Handler.reply = 'Sure!\n```json\n{"points": [{"text": "a {b}", "t": "00:01"}]}\n```'
        s = make_summariser({"engine": "llama_server", "url": self.url})
        out = s.complete_json("sys", "user", {"type": "object"})
        self.assertEqual(out["points"][0]["text"], "a {b}")
        req = _Handler.seen[-1]
        self.assertEqual(req["temperature"], 0.0)
        self.assertEqual(req["response_format"]["schema"], {"type": "object"})
        self.assertEqual(s.info()["build_info"], "b9999-test")

    def test_unusable_output(self):
        _Handler.reply = "no json here"
        s = make_summariser({"engine": "llama_server", "url": self.url})
        with self.assertRaises(EngineError):
            s.complete_json("sys", "user", {})

    def test_server_down(self):
        s = make_summariser({"engine": "llama_server", "url": "http://127.0.0.1:9", "timeout_s": 2})
        with self.assertRaises(EngineError):
            s.complete_json("sys", "user", {})
        self.assertEqual(s.info()["engine"], "llama.cpp (llama-server)")


class MiscTests(TmpCase):
    def test_unknown_engine(self):
        with self.assertRaisesRegex(ValueError, "known: fake, whisper_cpp"):
            make_transcriber({"engine": "nope"})

    def test_extract_json(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json_object('x {"a": "}"} y'), {"a": "}"})
        with self.assertRaises(ValueError):
            extract_json_object("[1, 2]")

    def test_example_config_loads(self):
        p = self.tmp / "c.toml"
        p.write_text(EXAMPLE)
        c = Config.load(p)
        self.assertEqual(c.transcriber["engine"], "whisper_cpp")
        self.assertEqual(c.chunk_seconds, 300)
        p.write_text("bogus = 1\n" + EXAMPLE)
        with self.assertRaisesRegex(ValueError, "bogus"):
            Config.load(p)


if __name__ == "__main__":
    unittest.main()
