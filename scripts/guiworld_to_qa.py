#!/usr/bin/env python
"""Convert the GUI-World dataset into the Screex accuracy-harness ``qa.jsonl`` format.

GUI-World (ICLR 2025, HF dataset ``shuaishuaicdp/GUI-World``) stores rows with a ``video_path``
(e.g. ``IOS/0.mov``), a ``system``/``app`` for the GUI domain, and an ``MCQA`` object holding a
``Question``, ``Options`` and a ``Correct Answer``. This script flattens those MCQs into the
one-MCQ-per-line schema ``scripts/eval.py --qa`` consumes:

    {"clip": ..., "question": ..., "choices": [...], "answer": "B", "type": "state"}

Notes:
- The published GUI-World JSON has a known schema inconsistency (``Options`` is sometimes a list,
  sometimes a ``{"A": ...}`` dict). Both are handled; malformed rows are skipped and counted.
- ``type`` is assigned by a transparent keyword heuristic (action/state/count/visual) — approximate,
  not GUI-World's native labels.
- GUI-World's license is unspecified upstream: do not commit the dataset or the generated qa.jsonl;
  verify licensing before redistributing any derived data.

Usage:
    python scripts/guiworld_to_qa.py guiworld.json -o qa.jsonl [--domain desktop] [--limit 100]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Domain inference from GUI-World's `system` field.
_MOBILE_SYSTEMS = {"ios", "android"}

_ACTION_KW = ("click", "type", "press", "open", "navigate", "select", "drag", "scroll",
              "tap", "enter", "submit", "do", "perform", "action", "next")
_COUNT_KW = ("how many", "number of", "count", "how much")
_VISUAL_KW = ("color", "colour", "icon", "where", "position", "located", "appear", "layout",
              "look", "visible", "shown", "displayed")


def classify_bucket(question: str) -> str:
    """Heuristically bucket a question into action / count / visual / state (default).
    Approximate — keyword-based, not GUI-World's native question types."""
    q = question.lower()
    if any(k in q for k in _COUNT_KW):
        return "count"
    if any(k in q for k in _ACTION_KW):
        return "action"
    if any(k in q for k in _VISUAL_KW):
        return "visual"
    return "state"


def domain_of(row: dict) -> str:
    """Return 'mobile' or 'desktop' for a GUI-World row, from its ``system`` field."""
    system = str(row.get("system", "")).strip().lower()
    return "mobile" if system in _MOBILE_SYSTEMS else "desktop"


def normalize_options(options) -> list[str] | None:
    """Coerce GUI-World ``Options`` (a list, or a ``{"A": ...}`` dict) into an ordered list of
    choice strings. Returns None if it cannot be interpreted as >=2 choices."""
    if isinstance(options, dict):
        # Order by label (A, B, C, ...) for stability.
        items = sorted(options.items(), key=lambda kv: str(kv[0]))
        choices = [f"{k}) {v}" for k, v in items]
    elif isinstance(options, list):
        choices = [str(o) for o in options]
    else:
        return None
    choices = [c for c in (c.strip() for c in choices) if c]
    return choices if len(choices) >= 2 else None


def iter_mcqs(row: dict):
    """Yield each MCQA dict on a row. GUI-World usually has a single ``MCQA`` object, but a list
    of them is tolerated too."""
    mcqa = row.get("MCQA")
    if isinstance(mcqa, dict):
        yield mcqa
    elif isinstance(mcqa, list):
        yield from (m for m in mcqa if isinstance(m, dict))


def convert(rows, domain="desktop", limit=None):
    """Convert GUI-World rows into ``(qa_items, stats)``.
    ``domain`` filters by inferred GUI domain ('desktop'/'mobile'/'all'); ``limit`` caps output."""
    qa, skipped = [], 0
    for row in rows:
        if not isinstance(row, dict):
            skipped += 1
            continue
        if domain != "all" and domain_of(row) != domain:
            continue
        clip = row.get("video_path")
        for mcqa in iter_mcqs(row):
            question = mcqa.get("Question")
            answer = mcqa.get("Correct Answer")
            choices = normalize_options(mcqa.get("Options"))
            if not (clip and question and answer and choices):
                skipped += 1
                continue
            qa.append({
                "clip": clip,
                "question": str(question).strip(),
                "choices": choices,
                "answer": str(answer).strip(),
                "type": classify_bucket(str(question)),
            })
            if limit is not None and len(qa) >= limit:
                return qa, {"emitted": len(qa), "skipped": skipped}
    return qa, {"emitted": len(qa), "skipped": skipped}


def load_rows(path) -> list:
    """Load GUI-World rows from a JSON file (a top-level list, or a dict wrapping one)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "rows", "train", "test"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("unrecognized GUI-World JSON: expected a list of rows or a dict wrapping one")


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert GUI-World JSON to Screex qa.jsonl")
    p.add_argument("guiworld_json", help="path to a GUI-World annotations JSON file")
    p.add_argument("-o", "--out", default=None, help="output qa.jsonl (default: stdout)")
    p.add_argument("--domain", choices=["desktop", "mobile", "all"], default="desktop",
                   help="keep only this GUI domain (default: desktop)")
    p.add_argument("--limit", type=int, default=None, help="cap the number of MCQs emitted")
    args = p.parse_args(argv)

    rows = load_rows(args.guiworld_json)
    qa, stats = convert(rows, domain=args.domain, limit=args.limit)
    lines = "\n".join(json.dumps(item, ensure_ascii=False) for item in qa)
    if args.out:
        Path(args.out).write_text(lines + ("\n" if lines else ""), encoding="utf-8")
        print(f"wrote {stats['emitted']} MCQs to {args.out} "
              f"(domain={args.domain}, skipped {stats['skipped']})", file=sys.stderr)
    else:
        print(lines)
        print(f"# {stats['emitted']} MCQs, skipped {stats['skipped']}", file=sys.stderr)


if __name__ == "__main__":
    main()
