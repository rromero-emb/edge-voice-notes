"""Ingest, recover, and process recordings.

On-disk layout under data_dir:

    queue.db                       job states (SQLite, WAL, synchronous=FULL)
    incoming/<id>.wav.part         recording in progress
    recordings/<id>/audio.wav      the recording (never modified after ingest)
    recordings/<id>/.work/         checkpoints: chunk plan, per-chunk and per-section results
    recordings/<id>/transcript.json, transcript.txt, summary.md, actions.json, status.json

Every step is idempotent: it looks for its checkpoint, and only does the work
if the checkpoint is missing. Kill the process at any instant, start it again,
and it carries on from the last checkpoint with identical final output.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Callable

from . import __version__, audio, notes
from .config import Config
from .engines import EngineError, Segment, Summariser, Transcriber, make_summariser, make_transcriber
from .fsutil import atomic_copy, atomic_write_json, atomic_write_text, fsync_dir, mkdir_durable, read_json
from .jobqueue import JobQueue

log = logging.getLogger("evn")


class Paths:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.db = self.root / "queue.db"
        self.incoming = self.root / "incoming"
        self.recordings = self.root / "recordings"

    def rec(self, rec_id: str) -> Path:
        return self.recordings / rec_id

    def ensure(self) -> None:
        for d in (self.incoming, self.recordings):
            mkdir_durable(d)


def new_id(src: Path | None = None) -> str:
    """UTC timestamp + 6 hex chars (from the audio, or random for a new recording)."""
    if src is None:
        suffix = os.urandom(3).hex()
    else:
        h = hashlib.sha256()
        with open(src, "rb") as f:
            h.update(f.read(1 << 20))
        suffix = h.hexdigest()[:6]
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + suffix


class Pipeline:
    def __init__(self, cfg: Config, transcriber: Transcriber | None = None,
                 summariser: Summariser | None = None) -> None:
        self.cfg = cfg
        self.paths = Paths(cfg.data_dir)
        self.paths.ensure()
        self.queue = JobQueue(self.paths.db)
        self._transcriber = transcriber
        self._summariser = summariser

    # Engines are created lazily: `evn status` must work without a model on disk.
    @property
    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = make_transcriber(self.cfg.transcriber)
        return self._transcriber

    @property
    def summariser(self) -> Summariser:
        if self._summariser is None:
            self._summariser = make_summariser(self.cfg.summariser)
        return self._summariser

    # ------------------------------------------------------------ ingest / recover

    def ingest(self, src: Path, rec_id: str | None = None, move: bool = False) -> str:
        """Put a finished 16 kHz mono WAV in the queue. Safe against power loss:
        the audio is durable before the job row exists, and recover() re-adds a
        recording directory that has audio but no job row."""
        src = Path(src)
        audio.check_format(src)
        rec_id = rec_id or new_id(src)
        rec = self.paths.rec(rec_id)
        mkdir_durable(rec)
        dst = rec / "audio.wav"
        if move and src.stat().st_dev == rec.stat().st_dev:
            os.replace(src, dst)
            fsync_dir(rec)
            fsync_dir(src.parent)
        else:
            atomic_copy(src, dst)
            if move:
                src.unlink()
        self.queue.add(rec_id)
        log.info("queued %s (%s)", rec_id, src.name)
        return rec_id

    def recover(self) -> list[str]:
        """Run at boot, before the worker. Returns what it did, for the log."""
        did = []
        for part in sorted(self.paths.incoming.glob("*.wav.part")):
            rec_id = part.name[:-len(".wav.part")]
            try:
                frames = audio.repair_header(part)
            except audio.AudioError as e:
                did.append(f"{part.name}: unrecoverable ({e}), kept for inspection")
                part.rename(part.with_suffix(".bad"))
                continue
            if frames == 0:
                part.unlink()
                did.append(f"{part.name}: empty, removed")
                continue
            fixed = part.with_name(rec_id + ".wav")
            os.replace(part, fixed)
            self.ingest(fixed, rec_id=rec_id, move=True)
            did.append(f"{rec_id}: interrupted recording repaired ({frames / audio.SAMPLE_RATE:.1f} s) and queued")
        for wav in sorted(self.paths.incoming.glob("*.wav")):
            rec_id = wav.stem
            self.ingest(wav, rec_id=rec_id, move=True)
            did.append(f"{rec_id}: finished recording queued")
        for rec in sorted(p for p in self.paths.recordings.iterdir() if p.is_dir()):
            job = self.queue.get(rec.name)
            if job is None and (rec / "audio.wav").exists():
                self.queue.add(rec.name)
                did.append(f"{rec.name}: re-queued (audio present, job row missing)")
            elif job is not None and job.state == "done" and (rec / ".work").exists():
                shutil.rmtree(rec / ".work")
        for line in did:
            log.info("recover: %s", line)
        return did

    # ------------------------------------------------------------ worker

    def run(self, once: bool = False, poll_s: float = 2.0, stop: Callable[[], bool] = lambda: False) -> int:
        """Process the queue. With once=True, return when nothing is pending."""
        processed = 0
        while not stop():
            job = self.queue.next_pending()
            if job is None:
                if once:
                    return processed
                time.sleep(poll_s)
                continue
            attempts = self.queue.begin_attempt(job.id)
            if attempts > self.cfg.max_attempts:
                msg = f"gave up after {self.cfg.max_attempts} attempts without progress: {job.last_error or 'process died (crash, OOM or power loss)'}"
                self.queue.mark_failed(job.id, msg)
                self._write_status(job.id, "failed", error=msg)
                log.error("%s: %s", job.id, msg)
                continue
            try:
                self.process(job.id)
                processed += 1
            except audio.AudioError as e:
                self.queue.mark_failed(job.id, str(e))
                self._write_status(job.id, "failed", error=str(e))
                log.error("%s: %s", job.id, e)
            except EngineError as e:
                self.queue.set_error(job.id, str(e))
                log.warning("%s: attempt %d failed: %s", job.id, attempts, e)
                if not once:
                    time.sleep(min(300, 10 * 2 ** attempts))
        return processed

    def process(self, rec_id: str) -> None:
        rec = self.paths.rec(rec_id)
        work = rec / ".work"
        mkdir_durable(work)
        self._write_status(rec_id, "transcribing")
        segs, info = self._transcribe(rec_id, rec, work)
        self._write_status(rec_id, "summarising")
        self._summarise(rec_id, rec, work, segs, info)
        self.queue.mark_done(rec_id)
        shutil.rmtree(work, ignore_errors=True)
        log.info("%s: done", rec_id)

    def _transcribe(self, rec_id: str, rec: Path, work: Path) -> tuple[list[Segment], audio.AudioInfo]:
        wav = rec / "audio.wav"
        ainfo = audio.check_format(wav)
        out = rec / "transcript.json"
        if out.exists():
            return [Segment.from_dict(d) for d in read_json(out)["segments"]], ainfo
        plan_file = work / "chunks.json"
        if not plan_file.exists():
            plan = audio.plan_chunks(wav, self.cfg.chunk_seconds * 1000)
            atomic_write_json(plan_file, {"chunks": plan})
        plan = read_json(plan_file)["chunks"]
        segs: list[Segment] = []
        for i, (a, b) in enumerate(plan):
            ck = work / f"chunk_{i:04d}.json"
            if not ck.exists():
                cw = work / f"chunk_{i:04d}.wav"
                audio.extract(wav, a, b, cw)
                t0 = time.monotonic()
                part = self.transcriber.transcribe(cw)
                elapsed = time.monotonic() - t0
                off = a * 1000 // audio.SAMPLE_RATE
                end = b * 1000 // audio.SAMPLE_RATE
                shifted = [Segment(min(end, s.start_ms + off), min(end, s.end_ms + off), s.text) for s in part]
                atomic_write_json(ck, {"start_ms": off, "end_ms": end, "elapsed_s": round(elapsed, 3),
                                       "segments": [s.to_dict() for s in shifted]})
                cw.unlink(missing_ok=True)
                self.queue.progress(rec_id)
                log.info("%s: chunk %d/%d transcribed (%.1f s audio in %.1f s)",
                         rec_id, i + 1, len(plan), (b - a) / audio.SAMPLE_RATE, elapsed)
            segs += [Segment.from_dict(d) for d in read_json(ck)["segments"]]
        elapsed = sum(read_json(work / f"chunk_{i:04d}.json")["elapsed_s"] for i in range(len(plan)))
        atomic_write_text(rec / "transcript.txt", notes.render_transcript_txt(segs))
        # Timing goes to status.json, not here: transcript.json is reproducible output.
        self._write_status(rec_id, "transcribing", metrics={
            "audio_seconds": round(ainfo.duration_ms / 1000, 1), "chunks": len(plan),
            "transcribe_seconds": round(elapsed, 3),
            "transcribe_rtf": round(elapsed * 1000 / ainfo.duration_ms, 3)})
        atomic_write_json(out, {"duration_ms": ainfo.duration_ms, "chunks": len(plan),
                                "segments": [s.to_dict() for s in segs]})
        return segs, ainfo

    def _summarise(self, rec_id: str, rec: Path, work: Path, segs: list[Segment], ainfo) -> None:
        sections = notes.split_sections(segs, self.cfg.section_max_chars)
        results, elapsed = [], 0.0
        for sec in sections:
            ck = work / f"section_{sec.index:04d}.json"
            if not ck.exists():
                t0 = time.monotonic()
                raw = self.summariser.complete_json(notes.SYSTEM, notes.section_prompt(segs, sec),
                                                    notes.SECTION_SCHEMA)
                res = notes.validate_section(raw, segs, sec)
                res["elapsed_s"] = round(time.monotonic() - t0, 3)
                atomic_write_json(ck, res)
                self.queue.progress(rec_id)
            r = read_json(ck)
            elapsed += r.pop("elapsed_s", 0.0)
            results.append(r)
        overview: list[dict] = []
        if len(results) > 1:
            ck = work / "overview.json"
            if not ck.exists():
                t0 = time.monotonic()
                raw = self.summariser.complete_json(notes.SYSTEM, notes.overview_prompt(results),
                                                    notes.OVERVIEW_SCHEMA)
                atomic_write_json(ck, {"overview": notes.validate_overview(raw, segs),
                                       "elapsed_s": round(time.monotonic() - t0, 3)})
            ov = read_json(ck)
            overview, elapsed = ov["overview"], elapsed + ov["elapsed_s"]
        acts = notes.collect(results)
        atomic_write_json(rec / "actions.json", acts)
        atomic_write_text(rec / "summary.md", notes.render_summary(rec_id, ainfo.duration_ms, overview, results, acts))
        self._write_status(rec_id, "done", metrics={
            "summarise_seconds": round(elapsed, 3), "sections": len(results),
            "actions_kept": len(acts["actions"]), "actions_rejected": len(acts["rejected_actions"]),
        })

    def _write_status(self, rec_id: str, state: str, error: str | None = None, metrics: dict | None = None) -> None:
        rec = self.paths.rec(rec_id)
        if not rec.exists():
            return
        f = rec / "status.json"
        st = read_json(f) if f.exists() else {"id": rec_id}
        job = self.queue.get(rec_id)
        st.update({"state": state, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "attempts": job.attempts if job else 0, "error": error,
                   "software": {"edge_voice_notes": __version__, "python": platform.python_version(),
                                "machine": platform.machine()}})
        if state in ("transcribing", "summarising", "done"):
            engines = st.setdefault("engines", {})
            try:
                if state == "transcribing":
                    engines["transcriber"] = self.transcriber.info()
                elif state == "summarising":
                    engines["summariser"] = self.summariser.info()
            except EngineError:
                pass
        if metrics is not None:
            st.setdefault("metrics", {}).update(metrics)
        atomic_write_json(f, st)
