from __future__ import annotations

import re

_engines: dict = {}


def _get_engine(lang: str | None = None):
    """Return a cached RapidOCR engine. ``lang`` is accepted for forward
    compatibility; RapidOCR's default models are multilingual (latin + digits),
    so the same engine is reused unless a caller wants distinct configs later."""
    key = lang or "default"
    engine = _engines.get(key)
    if engine is None:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        _engines[key] = engine
    return engine


def extract_text(bgr, lang: str | None = None) -> list:
    """Return on-screen text lines from a BGR frame, in reading order
    (top->bottom, left->right). The vertical bucket is scaled to the frame height so
    reading order is stable across resolutions."""
    engine = _get_engine(lang)
    result, _ = engine(bgr)
    if not result:
        return []

    height = bgr.shape[0] if hasattr(bgr, "shape") else 0
    # ~1.5% of frame height per line bucket (min 1px); groups words on the same line.
    bucket = max(1.0, height * 0.015) if height else 10.0

    def sort_key(item):
        box = item[0]  # 4 corner points [[x,y],...]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        return (round(min(ys) / bucket), min(xs))

    ordered = sorted(result, key=sort_key)
    return [str(item[1]).strip() for item in ordered if str(item[1]).strip()]


def normalize_line(line: str) -> str:
    """Canonical form of an OCR line for noise-tolerant comparison:
    lowercase, drop non-alphanumeric glyphs, collapse whitespace."""
    cleaned = re.sub(r"[^0-9a-z]+", " ", line.lower())
    return " ".join(cleaned.split())


def text_diff(prev_lines, cur_lines):
    """Return (added, removed): lines present in cur but not prev, and vice versa.

    Comparison is on a normalized form so minor OCR noise (stray glyphs, casing,
    punctuation) does not spuriously flip a line into both added and removed. Original
    line text and order are preserved in the output, and duplicate lines are matched by
    count rather than collapsed."""
    from collections import Counter

    prev_norm = Counter(normalize_line(line) for line in prev_lines if normalize_line(line))
    cur_norm = Counter(normalize_line(line) for line in cur_lines if normalize_line(line))

    added = []
    seen = Counter()
    for line in cur_lines:
        n = normalize_line(line)
        if not n:
            continue
        seen[n] += 1
        if seen[n] > prev_norm.get(n, 0):
            added.append(line)

    removed = []
    seen = Counter()
    for line in prev_lines:
        n = normalize_line(line)
        if not n:
            continue
        seen[n] += 1
        if seen[n] > cur_norm.get(n, 0):
            removed.append(line)

    return added, removed


def text_similarity(a_lines, b_lines) -> float:
    """Jaccard similarity (0..1) of two OCR line sets using normalized lines.
    Two empty texts are considered identical (1.0)."""
    a = {normalize_line(x) for x in a_lines if normalize_line(x)}
    b = {normalize_line(x) for x in b_lines if normalize_line(x)}
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0
