"""Kill the worker with SIGKILL at random instants, restart it, and check the final
outputs are byte-identical to an uninterrupted run.

SIGKILL loses nothing that was written, only what was in flight, so this checks
the crash-consistency of the checkpoint logic. Durability against a real power
cut also depends on the fsync discipline in fsutil.py and on the filesystem;
that part is checked on hardware (see docs/POWER_LOSS.md).
"""

import json
import os
import random
import signal
import sys
import time
import unittest

from edge_voice_notes.pipeline import Pipeline
from helpers import TmpCase, popen_cli, run_cli, write_config

COMPARED = ("transcript.json", "transcript.txt", "summary.md", "actions.json")
SEEDS = [int(s) for s in os.environ.get("EVN_CRASH_SEEDS", "1,2,3").split(",")]


class CrashTests(TmpCase):
    def _setup(self, name, **kw):
        data = self.tmp / name
        cfg = write_config(self.tmp / f"{name}.toml", data,
                           t_extra="delay_s = 0.05", s_extra="delay_s = 0.05", **kw)
        p = Pipeline(_load(cfg))
        for i in range(3):
            p.ingest(self.wav(f"{name}{i}.wav", 70.0 + 20 * i, seed=i), rec_id=f"rec{i}")
        p.queue.close()
        return cfg, data

    def _outputs(self, data):
        out = {}
        for i in range(3):
            rec = data / "recordings" / f"rec{i}"
            for f in COMPARED:
                out[f"rec{i}/{f}"] = (rec / f).read_bytes()
        return out

    def test_random_kills_give_identical_output(self):
        ref_cfg, ref_data = self._setup("ref", max_attempts=1000)
        run_cli(ref_cfg, "run", "--once")
        ref = self._outputs(ref_data)

        for seed in SEEDS:
            with self.subTest(seed=seed):
                rnd = random.Random(seed)
                cfg, data = self._setup(f"k{seed}", max_attempts=1000)
                kills = 0
                for _ in range(300):
                    proc = popen_cli(cfg, "run", "--once")
                    time.sleep(rnd.uniform(0.0, 0.6))
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGKILL)
                        proc.wait()
                        kills += 1
                        continue
                    self.assertEqual(proc.returncode, 0)
                    break
                else:
                    self.fail("never finished")
                print(f"\n  seed {seed}: finished after {kills} SIGKILLs", file=sys.stderr)
                self.assertGreaterEqual(kills, 5, "test did not actually interrupt anything")
                self.assertEqual(self._outputs(data), ref)
                st = json.loads(run_cli(cfg, "status", "--json").stdout)
                self.assertEqual({j["state"] for j in st}, {"done"})
                leftovers = [p for p in data.rglob("*") if p.name.endswith(".tmp") or p.name == ".work"]
                self.assertEqual(leftovers, [])

    def test_poison_recording_is_parked_after_max_attempts(self):
        # rec1 kills the whole process every time it reaches chunk 2 (think OOM).
        cfg, data = self._setup("poison", max_attempts=3)
        cfg.write_text(cfg.read_text().replace(
            "delay_s = 0.05\n", 'delay_s = 0.0\ncrash_on = "rec1/.work/chunk_0002"\n', 1))
        rcs = []
        for _ in range(10):
            rcs.append(run_cli(cfg, "run", "--once", check=False).returncode)
            if rcs[-1] == 0:
                break
        self.assertEqual(rcs[-1], 0)
        # 1st crash comes after progress (chunks 0-1 done), so the counter restarts;
        # then 3 crashes without progress, and the 5th start parks it and goes on.
        self.assertEqual(rcs.count(139), 4)
        jobs = {j["id"]: j for j in json.loads(run_cli(cfg, "status", "--json").stdout)}
        self.assertEqual(jobs["rec0"]["state"], "done")
        self.assertEqual(jobs["rec1"]["state"], "failed")
        self.assertIn("without progress", jobs["rec1"]["last_error"])
        self.assertEqual(jobs["rec2"]["state"], "done")


def _load(path):
    from edge_voice_notes.config import Config
    return Config.load(path)


if __name__ == "__main__":
    unittest.main()
