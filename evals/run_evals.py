#!/usr/bin/env python3
"""Run the evaluation set and write results.json + results.md.

  fake engines (CI, < 1 s):
    PYTHONPATH=src python3 evals/run_evals.py --engine fake --out evals/results/fake

  real engines (llama-server already running, whisper.cpp built):
    PYTHONPATH=src python3 evals/run_evals.py --engine real --llama-url http://127.0.0.1:8080 \
        --whisper-bin ~/src/whisper.cpp/build/bin/whisper-cli \
        --whisper-model ~/src/models/ggml-base.en.bin \
        --samples-dir ~/src/whisper.cpp/samples --out evals/results/x86-qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evallib  # noqa: E402
from edge_voice_notes.engines.fake import FakeSummariser  # noqa: E402
from edge_voice_notes.fsutil import atomic_write_json, atomic_write_text  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    cases = evallib.load_cases(args.cases)
    if args.engine == "fake":
        summariser = FakeSummariser()
    else:
        from edge_voice_notes.engines.llama_server import LlamaServer
        summariser = LlamaServer(url=args.llama_url)
    t0 = time.monotonic()
    res = evallib.evaluate(summariser, cases, args.section_max_chars)
    info = summariser.info()
    res.update({
        "label": args.label or args.engine,
        "summariser": {k: info.get(k) for k in ("engine", "model", "build_info", "model_sha256", "temperature", "seed")},
        "section_max_chars": args.section_max_chars,
        "cases": [c["id"] for c in cases],
        "machine": platform.machine(),
        "summarise_wall_s": round(time.monotonic() - t0, 1),
        "asr": [],
    })
    if args.whisper_bin and args.whisper_model:
        from edge_voice_notes.engines.whisper_cpp import WhisperCpp
        w = WhisperCpp(binary=args.whisper_bin, model=args.whisper_model, language="en")
        spec = json.loads((Path(__file__).resolve().parent / "asr_cases.json").read_text())
        for c in spec["cases"]:
            wav = Path(args.samples_dir).expanduser() / c["audio"]
            if not wav.exists():
                print(f"skip ASR case {c['id']}: {wav} not found", file=sys.stderr)
                continue
            hyp = " ".join(s.text for s in w.transcribe(wav))
            res["asr"].append({"id": c["id"], "model": Path(args.whisper_model).name, "hypothesis": hyp,
                               **evallib.wer(c["reference"], hyp)})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fake", "real"], default="fake")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--whisper-bin")
    ap.add_argument("--whisper-model")
    ap.add_argument("--samples-dir", default="~/src/whisper.cpp/samples")
    ap.add_argument("--cases", type=Path, default=evallib.HERE / "cases")
    ap.add_argument("--section-max-chars", type=int, default=3000)
    ap.add_argument("--label")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res = run(args)
    md = evallib.markdown(res)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out / "results.json", res)
        atomic_write_text(args.out / "results.md", md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
