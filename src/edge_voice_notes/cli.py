from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

from . import __version__
from .config import EXAMPLE, Config
from .notes import fmt_ts
from .pipeline import Pipeline, new_id


def _cfg(args) -> Config:
    cfg = Config.load(args.config)
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir).resolve()
    return cfg


def cmd_example_config(args) -> int:
    sys.stdout.write(EXAMPLE)
    return 0


def cmd_enqueue(args) -> int:
    p = Pipeline(_cfg(args))
    for f in args.files:
        print(p.ingest(Path(f)))
    return 0


def cmd_record(args) -> int:
    from .recorder import record

    cfg = _cfg(args)
    p = Pipeline(cfg)
    rec_id = new_id()
    part = record(p.paths.incoming, rec_id, device=args.device or cfg.recorder.get("device", "default"),
                  seconds=args.seconds, fsync_seconds=float(cfg.recorder.get("fsync_seconds", 5)))
    wav = part.with_name(rec_id + ".wav")
    part.replace(wav)
    print(p.ingest(wav, rec_id=rec_id, move=True))
    return 0


def cmd_recover(args) -> int:
    for line in Pipeline(_cfg(args)).recover():
        print(line)
    return 0


def cmd_run(args) -> int:
    p = Pipeline(_cfg(args))
    p.recover()
    stop = {"flag": False}

    def _term(signum, frame):
        stop["flag"] = True  # finish the current step; checkpoints make an abrupt stop safe too

    signal.signal(signal.SIGTERM, _term)
    n = p.run(once=args.once, stop=lambda: stop["flag"])
    logging.getLogger("evn").info("processed %d recording(s)", n)
    return 0


def cmd_status(args) -> int:
    p = Pipeline(_cfg(args))
    jobs = p.queue.all()
    if args.json:
        print(json.dumps([j.__dict__ for j in jobs], indent=2))
        return 0
    for j in jobs:
        st = p.paths.rec(j.id) / "status.json"
        extra = ""
        if st.exists():
            m = json.loads(st.read_text()).get("metrics") or {}
            if m:
                extra = (f"  {fmt_ts(m['audio_seconds'] * 1000)} audio, RTF {m['transcribe_rtf']}, "
                         f"summary {m['summarise_seconds']:.0f} s, {m['actions_kept']} actions")
        err = f"  ({j.last_error})" if j.last_error and j.state != "done" else ""
        print(f"{j.id}  {j.state:<7}  attempts={j.attempts}{extra}{err}")
    return 0


def cmd_retry(args) -> int:
    p = Pipeline(_cfg(args))
    ok = p.queue.retry(args.id)
    print("re-queued" if ok else "not a failed job")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evn", description="Offline voice notes: record, queue, transcribe, summarise.")
    ap.add_argument("-c", "--config", type=Path, help="config file (default $EVN_CONFIG or /etc/edge-voice-notes/config.toml)")
    ap.add_argument("-d", "--data-dir", help="override data_dir")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"edge-voice-notes {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("example-config", help="print an example config").set_defaults(fn=cmd_example_config)
    s = sub.add_parser("enqueue", help="queue existing 16 kHz mono WAV files")
    s.add_argument("files", nargs="+")
    s.set_defaults(fn=cmd_enqueue)
    s = sub.add_parser("record", help="record from ALSA until SIGINT/SIGTERM or --seconds, then queue")
    s.add_argument("--seconds", type=float)
    s.add_argument("--device")
    s.set_defaults(fn=cmd_record)
    sub.add_parser("recover", help="repair interrupted recordings and re-queue orphans").set_defaults(fn=cmd_recover)
    s = sub.add_parser("run", help="process the queue (recover first)")
    s.add_argument("--once", action="store_true", help="exit when the queue is empty")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("status", help="list recordings and their state")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("retry", help="re-queue a failed recording")
    s.add_argument("id")
    s.set_defaults(fn=cmd_retry)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd != "example-config" and args.config is None:
        from .config import DEFAULT_PATH
        args.config = DEFAULT_PATH
    return args.fn(args)
