# Evaluation: x86 laptop, Qwen2.5-1.5B-Instruct Q4_K_M + whisper base.en

Summariser: `qwen2.5-1.5b-instruct-q4_k_m.gguf` · 5 cases · section_max_chars=3000

| Pipeline | Action P | Action R | Invented | Trap leaks | Owner ok | Due ok | Action ts ok | Decision P | Decision R | Ts on a real line |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.42 | 0.71 | 7 | 5 | 1.00 | 0.40 | 0.80 | 0.38 | 1.00 | 1.00 |
| current | 1.00 | 0.71 | 0 | 0 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 |

Per case (current pipeline): expected / found / invented actions

| Case | Expected | Found | Invented | Trap leaks | Decisions found |
|---|---|---|---|---|---|
| 01_firmware_standup | 2 | 2 | 0 | 0 | 1/1 |
| 02_supplier_call | 2 | 1 | 0 | 0 | 1/1 |
| 03_vague_planning | 1 | 1 | 0 | 0 | 0/0 |
| 04_dialogue_handoff | 2 | 1 | 0 | 0 | 1/1 |
| 05_decisions_only | 0 | 0 | 0 | 0 | 3/3 |

| ASR case | Model | WER | Ref words | S/D/I |
|---|---|---|---|---|
| jfk | ggml-base.en.bin | 0.000 | 22 | 0/0/0 |
