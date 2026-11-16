"""Transcript -> sections -> summary points, decisions and action items.

The LLM proposes; this module disposes. Every timestamp the model returns is
mapped back onto a real transcript segment, and an action item survives only if
its owner or its deadline literally appears in the transcript section it cites.
That check is plain code, so it holds whatever model is plugged in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .engines.base import Segment

# ---------------------------------------------------------------- timestamps


def fmt_ts(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# "[01:05]", "01:05", "1:01:05", and ranges like "00:25-00:32" (the start is used)
_TS = re.compile(r"^\s*\[?(?:(\d+):)?(\d{1,3}):(\d{2})(?:\.\d+)?\]?\s*(?:(?:-|–|to)\s*\[?[\d:.]+\]?\s*)?$")


def parse_ts(text) -> int | None:
    if not isinstance(text, str):
        return None
    m = _TS.match(text)
    if not m:
        return None
    h, mi, s = int(m.group(1) or 0), int(m.group(2)), int(m.group(3))
    if s >= 60:
        return None
    return ((h * 60 + mi) * 60 + s) * 1000


# ---------------------------------------------------------------- sections


@dataclass(frozen=True)
class Section:
    index: int
    first: int  # index of first segment (inclusive)
    last: int   # index of last segment (inclusive)
    start_ms: int
    end_ms: int


def split_sections(segs: list[Segment], max_chars: int = 3000) -> list[Section]:
    """Greedy split at segment boundaries so each section's prompt stays well within
    the model's context. 3000 characters is ~750 tokens of English: small (1-3B)
    models follow the instructions noticeably better on short sections."""
    out: list[Section] = []
    first, size = 0, 0
    for i, s in enumerate(segs):
        line = len(s.text) + 12
        if i > first and size + line > max_chars:
            out.append(Section(len(out), first, i - 1, segs[first].start_ms, segs[i - 1].end_ms))
            first, size = i, 0
        size += line
    if segs:
        out.append(Section(len(out), first, len(segs) - 1, segs[first].start_ms, segs[-1].end_ms))
    return out


def section_lines(segs: list[Segment], sec: Section) -> str:
    return "\n".join(f"[{fmt_ts(s.start_ms)}] {s.text}" for s in segs[sec.first:sec.last + 1])


# ---------------------------------------------------------------- prompts

SYSTEM = (
    "You take notes from a transcribed meeting or voice memo. Use only what is said in the "
    "transcript; never add facts, names or dates. Every item must cite the [mm:ss] timestamp "
    "of the transcript line it comes from. Write in the language of the transcript."
)

SECTION_INSTRUCTIONS = (
    "Transcript section {start}-{end}:\n\n{lines}\n\n"
    "Return JSON with:\n"
    "- points: the key points of this section (at most 6), each with text and t (timestamp).\n"
    "- decisions: things explicitly decided or agreed, each with text and t.\n"
    "- actions: tasks someone committed to, each with task, owner, due and t. owner is the name "
    "of the person who will do it, only if the transcript says it, otherwise null. due is the "
    "deadline in the words used in the transcript (for example \"by Friday\"), only if one is "
    "said, otherwise null; never a timestamp. Do not guess either."
)

OVERVIEW_INSTRUCTIONS = (
    "These are the key points of a recording, section by section, with timestamps:\n\n{points}\n\n"
    "Return JSON with overview: the 3 to 6 most important points of the whole recording, each "
    "with text and t (the timestamp of the point it is based on)."
)

_ITEM = {"type": "object", "properties": {"text": {"type": "string"}, "t": {"type": "string"}},
         "required": ["text", "t"]}
SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {"type": "array", "items": _ITEM, "maxItems": 6},
        "decisions": {"type": "array", "items": _ITEM},
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {"task": {"type": "string"},
                           "owner": {"type": ["string", "null"]},
                           "due": {"type": ["string", "null"]},
                           "t": {"type": "string"}},
            "required": ["task", "owner", "due", "t"]}},
    },
    "required": ["points", "decisions", "actions"],
}
OVERVIEW_SCHEMA = {"type": "object", "properties": {"overview": {"type": "array", "items": _ITEM, "maxItems": 6}},
                   "required": ["overview"]}

# ---------------------------------------------------------------- validation

_NOT_OWNERS = {"i", "we", "you", "he", "she", "they", "someone", "somebody", "everyone", "team",
               "the team", "us", "all", "anyone", "tbd", "unknown", "none", "n/a", "null"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" .,:;!?\"'")


_STOP = set("the and for that this with have from will was were are but not you our its it's they them "
            "then than about into what when where which would could should there their just also can "
            "let's okay yes".split())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in _STOP}


def _snap(segs: list[Segment], sec: Section, t, text: str = "") -> int | None:
    """Map a model-supplied timestamp onto the start of a real segment of the section.

    Small models often cite the neighbouring line. If another segment of the
    section shares clearly more words with the item, cite that one instead."""
    item = _words(text)

    def cover(i: int) -> float:
        return len(item & _words(segs[i].text)) / len(item) if item else 0.0

    ms = parse_ts(t)
    if ms is None:
        # no usable timestamp: accept the item only if one line clearly matches it
        best = max(range(sec.first, sec.last + 1), key=cover)
        return segs[best].start_ms if cover(best) >= 0.5 else None
    k = sec.first
    for i in range(sec.first, sec.last + 1):
        if segs[i].start_ms <= ms + 999:  # [mm:ss] has 1 s resolution
            k = i
    if item:
        best = max(range(sec.first, sec.last + 1), key=lambda i: (cover(i), -abs(i - k)))
        if cover(best) >= cover(k) + 0.25:
            k = best
    return segs[k].start_ms


def _clean_items(raw, segs, sec, key="text", limit=None) -> list[dict]:
    out, seen = [], set()
    for it in raw if isinstance(raw, list) else []:
        if not isinstance(it, dict):
            continue
        text = str(it.get(key) or "").strip()
        ms = _snap(segs, sec, it.get("t"), text)
        if not text or ms is None or _norm(text) in seen:
            continue
        seen.add(_norm(text))
        out.append({"text": text, "t_ms": ms})
        if limit and len(out) >= limit:
            break
    return out


def _in_text(needle: str, haystack: str) -> bool:
    n = _norm(needle)
    return bool(n) and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", haystack) is not None


_DUE_FILLER = {"by", "before", "on", "until", "till", "the", "of", "at", "in", "end", "no", "later", "than"}


def _due_in_text(due: str, haystack: str) -> bool:
    """A deadline is accepted if it is quoted verbatim, or if every meaningful word
    of it is said ("November 30th" vs "the 30th of November"). Numbers and day
    names must be there; the model cannot introduce a date of its own."""
    if parse_ts(due) is not None:  # a transcript timestamp is not a deadline
        return False
    if _in_text(due, haystack):
        return True
    need = {w for w in re.findall(r"[a-z0-9]+", due.lower()) if w not in _DUE_FILLER}
    have = set(re.findall(r"[a-z0-9]+", haystack))
    return bool(need) and need <= have


_PAST = re.compile(r"\b(last|yesterday|ago|previous|earlier)\b")


def _window(segs: list[Segment], sec: Section, ms: int, before: int, after: int) -> str:
    """Text of the cited line plus `before`/`after` neighbours inside the section.
    An owner is usually named in the line itself or in the question just before it
    ("Lucia, can you...?" / "Sure, by Monday."); a name further away belongs to
    another item."""
    k = next(i for i in range(sec.first, sec.last + 1) if segs[i].start_ms == ms)
    lo, hi = max(sec.first, k - before), min(sec.last, k + after)
    return _norm(" ".join(s.text for s in segs[lo:hi + 1]))


_DECISION_CUES = re.compile(r"\b(decid\w*|agree\w*|approv\w*|chose|choose|chosen|go with|going with|settled|"
                            r"confirm\w*|final|stays|will stay|won't|will not)\b")


def validate_section(raw: dict, segs: list[Segment], sec: Section) -> dict:
    actions, rejected = [], []
    for a in raw.get("actions", []) if isinstance(raw.get("actions"), list) else []:
        if not isinstance(a, dict):
            continue
        task = str(a.get("task") or "").strip()
        ms = _snap(segs, sec, a.get("t"), task)
        owner = a.get("owner") if isinstance(a.get("owner"), str) else None
        due = a.get("due") if isinstance(a.get("due"), str) else None
        if not task or ms is None:
            rejected.append({"task": task, "t": a.get("t"), "reason": "missing task or bad timestamp"})
            continue
        reasons = []
        if owner is not None and (_norm(owner) in _NOT_OWNERS or not _in_text(owner, _window(segs, sec, ms, 1, 0))):
            reasons.append(f"owner {owner!r} not named at the cited line")
            owner = None
        if due is not None and (not _due_in_text(due, _window(segs, sec, ms, 1, 0)) or _PAST.search(due.lower())):
            reasons.append(f"due {due!r} not stated as a deadline at the cited line")
            due = None
        if owner is None and due is None:
            rejected.append({"task": task, "t_ms": ms,
                             "reason": "; ".join(reasons) or "no explicit owner or deadline"})
            continue
        actions.append({"task": task, "owner": owner.strip() if owner else None,
                        "due": due.strip() if due else None, "t_ms": ms})
    return {
        "section": sec.index, "start_ms": sec.start_ms, "end_ms": sec.end_ms,
        "points": _clean_items(raw.get("points"), segs, sec, limit=6),
        "decisions": [d for d in _clean_items(raw.get("decisions"), segs, sec)
                      if _DECISION_CUES.search(_window(segs, sec, d["t_ms"], 0, 0))],
        "actions": actions,
        "rejected_actions": rejected,
    }


def validate_overview(raw: dict, segs: list[Segment]) -> list[dict]:
    whole = Section(0, 0, len(segs) - 1, segs[0].start_ms, segs[-1].end_ms)
    return _clean_items(raw.get("overview"), segs, whole, limit=6)


def overview_prompt(sections: list[dict]) -> str:
    pts = "\n".join(f"[{fmt_ts(p['t_ms'])}] {p['text']}" for s in sections for p in s["points"])
    return OVERVIEW_INSTRUCTIONS.format(points=pts)


def section_prompt(segs: list[Segment], sec: Section) -> str:
    return SECTION_INSTRUCTIONS.format(start=fmt_ts(sec.start_ms), end=fmt_ts(sec.end_ms),
                                       lines=section_lines(segs, sec))


# ---------------------------------------------------------------- rendering


def collect(sections: list[dict]) -> dict:
    seen_d, seen_a = set(), set()
    decisions, actions, rejected = [], [], []
    for s in sections:
        for d in s["decisions"]:
            if _norm(d["text"]) not in seen_d:
                seen_d.add(_norm(d["text"]))
                decisions.append(d)
        for a in s["actions"]:
            k = (_norm(a["task"]), _norm(a["owner"] or ""))
            if k not in seen_a:
                seen_a.add(k)
                actions.append(a)
        rejected.extend(s["rejected_actions"])
    return {"decisions": decisions, "actions": actions, "rejected_actions": rejected}


def render_summary(rec_id: str, duration_ms: int, overview: list[dict], sections: list[dict],
                   acts: dict) -> str:
    L = [f"# Recording {rec_id}", "", f"Duration: {fmt_ts(duration_ms)}", ""]
    if overview:
        L += ["## Overview", ""] + [f"- {p['text']} [{fmt_ts(p['t_ms'])}]" for p in overview] + [""]
    L += ["## Sections", ""]
    for s in sections:
        L.append(f"### {s['section'] + 1}. {fmt_ts(s['start_ms'])}-{fmt_ts(s['end_ms'])}")
        L.append("")
        L += [f"- {p['text']} [{fmt_ts(p['t_ms'])}]" for p in s["points"]] or ["- (no key points)"]
        L.append("")
    L += ["## Decisions", ""]
    L += [f"- {d['text']} [{fmt_ts(d['t_ms'])}]" for d in acts["decisions"]] or ["- None stated."]
    L += ["", "## Action items", ""]
    if acts["actions"]:
        for a in acts["actions"]:
            who = a["owner"] or "owner not stated"
            due = a["due"] or "no deadline stated"
            L.append(f"- {a['task']} ({who}; {due}) [{fmt_ts(a['t_ms'])}]")
    else:
        L.append("- None with an explicit owner or deadline.")
    return "\n".join(L) + "\n"


def render_transcript_txt(segs: list[Segment]) -> str:
    return "".join(f"[{fmt_ts(s.start_ms)}] {s.text}\n" for s in segs)
