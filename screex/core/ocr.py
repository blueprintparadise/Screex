from __future__ import annotations

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def extract_text(bgr) -> list:
    """Return on-screen text lines from a BGR frame, in reading order (top->bottom, left->right)."""
    engine = _get_engine()
    result, _ = engine(bgr)
    if not result:
        return []

    def sort_key(item):
        box = item[0]  # 4 corner points [[x,y],...]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        return (round(min(ys) / 10.0), min(xs))

    ordered = sorted(result, key=sort_key)
    return [str(item[1]).strip() for item in ordered if str(item[1]).strip()]


def text_diff(prev_lines, cur_lines):
    """Return (added, removed): lines in cur not in prev, and lines in prev not in cur."""
    prev_set = set(prev_lines)
    cur_set = set(cur_lines)
    added = [line for line in cur_lines if line not in prev_set]
    removed = [line for line in prev_lines if line not in cur_set]
    return added, removed
