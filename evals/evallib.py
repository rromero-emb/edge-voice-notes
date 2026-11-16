"""Evaluation helpers: dataset loading, WER, matching and scoring.

Stdlib only, like the rest of the project. The scorer compares two pipelines on
the same model output:

  baseline  the model's action items taken as they come (what a naive "ask the
            LLM for action items" integration ships)
  current   the same output after notes.validate_section (owner/deadline must be
            said in the cited section, timestamps snapped to real lines)

so the difference between the two rows is the validator, not the model.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from edge_voice_notes import notes
from edge_voice_notes.engines.base import Segment, Summariser

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- dataset


def load_cases(directory: Path = HERE / "cases") -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def segments(case: dict) -> list[Segment]:
    rows = [(notes.parse_ts(t), text) for t, text in case["transcript"]]
    out = []
    for i, (t, text) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else t + 5000
        out.append(Segment(t, end, text))
    return out


# ---------------------------------------------------------------- WER

_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]|\(([^)]+?)\)")


def normalize(text: str) -> str:
    """Whisper's BasicTextNormalizer rules (lowercase, drop bracketed text, turn
    symbols and punctuation into spaces, collapse whitespace), reimplemented with
    the stdlib. No number or spelling normalisation, so "1.4" vs "one point four"
    counts as an error."""
    s = _BRACKETS.sub("", text.lower())
    s = "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))
    return re.sub(r"\s+", " ", s).strip()


def wer(reference: str, hypothesis: str) -> dict:
    r, h = normalize(reference).split(), normalize(hypothesis).split()
    # Levenshtein on words, keeping the S/D/I split
    prev = [(j, 0, 0, j) for j in range(len(h) + 1)]  # (cost, sub, del, ins)
    for i in range(1, len(r) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                cur.append(prev[j - 1])
                continue
            s, d, n = prev[j - 1], prev[j], cur[j - 1]
            best = min((s[0], 0), (d[0], 1), (n[0], 2))[1]
            c = (s, d, n)[best]
            cur.append((c[0] + 1, c[1] + (best == 0), c[2] + (best == 1), c[3] + (best == 2)))
        prev = cur
    cost, sub, dele, ins = prev[-1]
    return {"wer": round(cost / max(1, len(r)), 4), "ref_words": len(r), "sub": sub, "del": dele, "ins": ins}


# ---------------------------------------------------------------- running the summariser


def run_case(summariser: Summariser, case: dict, section_max_chars: int = 3000) -> dict:
    """One model call per section. Returns the raw outputs and the validated ones."""
    segs = segments(case)
    raw, validated = [], []
    for sec in notes.split_sections(segs, section_max_chars):
        r = summariser.complete_json(notes.SYSTEM, notes.section_prompt(segs, sec), notes.SECTION_SCHEMA)
        raw.append(r)
        validated.append(notes.validate_section(r, segs, sec))
    return {"segments": segs, "raw": raw, "validated": validated}


def baseline_items(run: dict) -> tuple[list[dict], list[dict]]:
    """Actions and decisions exactly as the model returned them."""
    acts, decs = [], []
    for r in run["raw"]:
        for a in r.get("actions") or []:
            if isinstance(a, dict) and str(a.get("task") or "").strip():
                acts.append({"task": str(a["task"]), "owner": a.get("owner") if isinstance(a.get("owner"), str) else None,
                             "due": a.get("due") if isinstance(a.get("due"), str) else None,
                             "t_ms": notes.parse_ts(a.get("t"))})
        for d in r.get("decisions") or []:
            if isinstance(d, dict) and str(d.get("text") or "").strip():
                decs.append({"text": str(d["text"]), "t_ms": notes.parse_ts(d.get("t"))})
    return acts, decs


def current_items(run: dict) -> tuple[list[dict], list[dict]]:
    c = notes.collect(run["validated"])
    return c["actions"], c["decisions"]


# ---------------------------------------------------------------- scoring


def _words(s: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", (s or "").lower().replace("'s", "")))


def _hits(keywords: list[str], text: str) -> float:
    kw = [k.lower() for k in keywords]
    have = _words(text)
    return sum(k in have for k in kw) / len(kw) if kw else 0.0


def _same_owner(pred: str | None, exp: str | None) -> bool:
    return (pred or "").strip().lower() == (exp or "").strip().lower()


def _same_due(pred: str | None, exp: str | None) -> bool:
    if exp is None or pred is None:
        return pred is None and exp is None
    filler = {"by", "before", "on", "the", "of"}
    return {w for w in _words(pred) if w not in filler} == {w for w in _words(exp) if w not in filler}


def _match(preds: list[dict], expected: list[dict], text_key: str) -> list[tuple[int, int]]:
    """Greedy one-to-one matching on keyword coverage (>= 0.5)."""
    pairs = sorted(((_hits(e["keywords"], p[text_key] + " " + (p.get("owner") or "")), i, j)
                    for i, p in enumerate(preds) for j, e in enumerate(expected)), reverse=True)
    used_p, used_e, out = set(), set(), []
    for cov, i, j in pairs:
        if cov >= 0.5 and i not in used_p and j not in used_e:
            used_p.add(i)
            used_e.add(j)
            out.append((i, j))
    return out


def _ts_ok(pred_ms, exp: dict) -> bool:
    allowed = [exp["t"], *exp.get("alt_t", [])]
    return pred_ms is not None and any(pred_ms == notes.parse_ts(t) for t in allowed)


def score_case(case: dict, segs: list[Segment], actions: list[dict], decisions: list[dict]) -> dict:
    starts = {s.start_ms for s in segs}
    am = _match(actions, case["actions"], "task")
    matched_p = {i for i, _ in am}
    leaks = 0
    for i, a in enumerate(actions):
        if i not in matched_p and any(_hits(t["keywords"], a["task"]) >= 0.5 for t in case["traps"]):
            leaks += 1
    dm = _match(decisions, case["decisions"], "text")
    items = actions + decisions
    return {
        "actions_expected": len(case["actions"]),
        "actions_predicted": len(actions),
        "actions_tp": len(am),
        "invented": len(actions) - len(am),
        "trap_leaks": leaks,
        "owner_ok": sum(_same_owner(actions[i]["owner"], case["actions"][j]["owner"]) for i, j in am),
        "due_ok": sum(_same_due(actions[i]["due"], case["actions"][j]["due"]) for i, j in am),
        "action_ts_ok": sum(_ts_ok(actions[i]["t_ms"], case["actions"][j]) for i, j in am),
        "decisions_expected": len(case["decisions"]),
        "decisions_predicted": len(decisions),
        "decisions_tp": len(dm),
        "items": len(items),
        "items_ts_on_line": sum(it.get("t_ms") in starts for it in items),
    }


def aggregate(per_case: list[dict]) -> dict:
    t = {k: sum(c[k] for c in per_case) for k in per_case[0]}

    def ratio(a, b):
        return round(a / b, 3) if b else None

    return {
        "action_precision": ratio(t["actions_tp"], t["actions_predicted"]),
        "action_recall": ratio(t["actions_tp"], t["actions_expected"]),
        "invented_actions": t["invented"],
        "trap_leaks": t["trap_leaks"],
        "owner_accuracy": ratio(t["owner_ok"], t["actions_tp"]),
        "due_accuracy": ratio(t["due_ok"], t["actions_tp"]),
        "action_ts_accuracy": ratio(t["action_ts_ok"], t["actions_tp"]),
        "decision_precision": ratio(t["decisions_tp"], t["decisions_predicted"]),
        "decision_recall": ratio(t["decisions_tp"], t["decisions_expected"]),
        "ts_valid": ratio(t["items_ts_on_line"], t["items"]),
        "totals": t,
    }


def evaluate(summariser: Summariser, cases: list[dict], section_max_chars: int = 3000) -> dict:
    out: dict = {"baseline": {"cases": {}}, "current": {"cases": {}}}
    for case in cases:
        run = run_case(summariser, case, section_max_chars)
        for name, fn in (("baseline", baseline_items), ("current", current_items)):
            acts, decs = fn(run)
            out[name]["cases"][case["id"]] = score_case(case, run["segments"], acts, decs)
            out[name]["cases"][case["id"]]["output"] = {"actions": acts, "decisions": decs}
    for name in ("baseline", "current"):
        scores = [{k: v for k, v in c.items() if k != "output"} for c in out[name]["cases"].values()]
        out[name]["summary"] = aggregate(scores)
    return out


# ---------------------------------------------------------------- report

COLUMNS = [("action_precision", "Action P"), ("action_recall", "Action R"), ("invented_actions", "Invented"),
           ("trap_leaks", "Trap leaks"), ("owner_accuracy", "Owner ok"), ("due_accuracy", "Due ok"),
           ("action_ts_accuracy", "Action ts ok"), ("decision_precision", "Decision P"),
           ("decision_recall", "Decision R"), ("ts_valid", "Ts on a real line")]


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def markdown(results: dict) -> str:
    L = [f"# Evaluation: {results['label']}", "",
         f"Summariser: `{results['summariser'].get('model')}` · {len(results['cases'])} cases · "
         f"section_max_chars={results['section_max_chars']}", ""]
    L.append("| Pipeline | " + " | ".join(h for _, h in COLUMNS) + " |")
    L.append("|---" * (len(COLUMNS) + 1) + "|")
    for name in ("baseline", "current"):
        s = results[name]["summary"]
        L.append(f"| {name} | " + " | ".join(_fmt(s[k]) for k, _ in COLUMNS) + " |")
    L += ["", "Per case (current pipeline): expected / found / invented actions", ""]
    L.append("| Case | Expected | Found | Invented | Trap leaks | Decisions found |")
    L.append("|---|---|---|---|---|---|")
    for cid, c in results["current"]["cases"].items():
        L.append(f"| {cid} | {c['actions_expected']} | {c['actions_tp']} | {c['invented']} | {c['trap_leaks']} | "
                 f"{c['decisions_tp']}/{c['decisions_expected']} |")
    if results.get("asr"):
        L += ["", "| ASR case | Model | WER | Ref words | S/D/I |", "|---|---|---|---|---|"]
        for a in results["asr"]:
            L.append(f"| {a['id']} | {a['model']} | {a['wer']:.3f} | {a['ref_words']} | {a['sub']}/{a['del']}/{a['ins']} |")
    else:
        L += ["", "ASR (WER): skipped, no audio in this run."]
    return "\n".join(L) + "\n"
