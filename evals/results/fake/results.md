# Evaluation: fake engines (CI)

Summariser: `fake-summariser` · 5 cases · section_max_chars=3000

| Pipeline | Action P | Action R | Invented | Trap leaks | Owner ok | Due ok | Action ts ok | Decision P | Decision R | Ts on a real line |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.67 | 0.57 | 2 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| current | 1.00 | 0.57 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Per case (current pipeline): expected / found / invented actions

| Case | Expected | Found | Invented | Trap leaks | Decisions found |
|---|---|---|---|---|---|
| 01_firmware_standup | 2 | 2 | 0 | 0 | 1/1 |
| 02_supplier_call | 2 | 1 | 0 | 0 | 1/1 |
| 03_vague_planning | 1 | 1 | 0 | 0 | 0/0 |
| 04_dialogue_handoff | 2 | 0 | 0 | 0 | 1/1 |
| 05_decisions_only | 0 | 0 | 0 | 0 | 3/3 |

ASR (WER): skipped, no audio in this run.
