# Power loss: design and hardware test procedure

## What can be lost, and what cannot

| Moment of the power cut | After reboot (`evn run` calls `recover` first) |
|---|---|
| While recording | Audio up to the last `fsync` (≤ `fsync_seconds`, default 5 s) is kept: the WAV header is repaired and the recording is queued. |
| Between end of recording and queue insert | The audio is durable before the job row is written; `recover` finds the WAV and queues it. |
| During transcription | Finished chunks are on disk; transcription resumes at the first unfinished chunk (≤ `chunk_seconds` of work lost). |
| During summarisation | Finished sections are on disk; resumes at the first unfinished section. |
| While writing any output file | Old version, new version, or no file. Never a partial file (temp + `fsync` + `rename` + directory `fsync`). |
| While SQLite commits | WAL with `synchronous=FULL`: the transaction is committed or it is not. |
| Same recording crashes the device every time | Counted attempts without progress; after `max_attempts` it is parked as `failed`, the queue continues. |

## Assumptions to verify on the target

1. **Storage honours flush.** eMMC and good SD cards do; some cheap SD cards and USB sticks acknowledge writes
   still in their cache. This is the one assumption software cannot fix. Test it (below) on the exact part used in
   production.
2. **Filesystem.** ext4 with the default `data=ordered` journal. Do not mount with `nobarrier`.
3. **Hold-up time.** If the board has a supercapacitor or a battery fuel gauge, a low-battery signal should stop
   the recorder cleanly (`systemctl stop evn-record`). The design does not depend on it.

## Hardware test procedure

Equipment: the target board, its production storage, a USB relay or programmable supply to cut power, a second
machine to drive the relay over a serial console.

1. Queue 3 recordings of 10, 30 and 60 minutes with known content; keep reference outputs from a run without cuts.
2. Loop 200 times: power on, wait a random 5-300 s, cut power hard (not a shutdown).
   Vary the moment: while recording (start `evn-record` first in some iterations), during whisper.cpp, during
   llama.cpp, during file writes (short `chunk_seconds` makes these more frequent).
3. After each boot, log `evn status --json`, `dmesg | grep -i -E 'ext4|mmc|I/O error'` and the journal.
4. Pass criteria:
   - every recording ends `done` (or `failed` only for a deliberate poison file);
   - transcripts are byte-identical to the reference (whisper.cpp is deterministic for the same build, model and
     threads);
   - no `.tmp` files and no `*.wav.bad` except those deliberately created;
   - no filesystem errors, `fsck` clean at the end;
   - interrupted recordings lose at most `fsync_seconds` of audio.

What CI already covers without hardware: `tests/test_crash.py` (random `SIGKILL`, byte-identical outputs, poison
recording) and `tests/test_recorder.py` (recorder killed before finalising its header, then repaired).
