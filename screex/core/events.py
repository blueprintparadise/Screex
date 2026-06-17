from __future__ import annotations

import re

from screex.core.ocr import normalize_line

_ERROR_RE = re.compile(
    r"\b(error|errors|failed|failure|exception|invalid|denied|forbidden|"
    r"unauthorized|cannot|unable|traceback)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|[\w-]+\.(?:com|org|io|net|dev|app|gov|edu)\b\S*)",
    re.IGNORECASE,
)

# Fraction of the previous state's text that must disappear to read as a navigation.
_NAVIGATE_REMOVED_RATIO = 0.6


def _field_for(added_line, prev_state, cur_state):
    """Nearest persistent label (text present in prev_state) to the left/above the box of
    ``added_line`` in cur_state.boxes. Returns the label text, or None."""
    boxes = getattr(cur_state, "boxes", []) or []
    target = next((b for b in boxes
                   if normalize_line(b["text"]) == normalize_line(added_line)), None)
    if target is None:
        return None
    tx, ty, _tw, th = target["box"]
    persistent = {normalize_line(x) for x in (getattr(prev_state, "ocr_text", []) or [])}
    best, best_d = None, None
    for b in boxes:
        if b is target or normalize_line(b["text"]) not in persistent:
            continue
        x, y, _w, _h = b["box"]
        if y > ty + th:  # label sits below the value — not a field label
            continue
        d = (x - tx) ** 2 + (y - ty) ** 2
        if best_d is None or d < best_d:
            best_d, best = d, b["text"]
    return best


def classify_event(prev_state, cur_state, region, interaction=None, narration_text=""):
    """Classify the transition into cur_state as a typed action event dict, or {} when no
    confident classification is possible. First matching rule wins."""
    added = list(getattr(cur_state, "text_added", []) or [])
    removed = list(getattr(cur_state, "text_removed", []) or [])
    t = round(float(getattr(cur_state, "t_start", 0.0)), 3)
    shape = region.shape if region is not None else None
    box = region.box if region is not None else [0, 0, 0, 0]

    def ev(type_, confidence, **kw):
        out = {"type": type_, "t": t, "region": box, "confidence": round(confidence, 2)}
        out.update({k: v for k, v in kw.items() if v})
        return out

    # 1. error — any added line matches an error pattern.
    for line in added:
        if _ERROR_RE.search(line):
            return ev("error", 0.8, label=line)

    # 2. navigate — whole screen replaced, or most prior text gone.
    prev_n = len(getattr(prev_state, "ocr_text", []) or [])
    removed_ratio = (len(removed) / prev_n) if prev_n else 0.0
    if shape == "full" or removed_ratio >= _NAVIGATE_REMOVED_RATIO:
        url = None
        for line in added:
            m = _URL_RE.search(line)
            if m:
                url = m.group(0)
                break
        return ev("navigate", 0.7, value=url)

    # 3. open_dialog — centered overlay over persisting content.
    if shape == "overlay" and added:
        return ev("open_dialog", 0.6, label=added[0])

    # 4. scroll — large vertical band shifted.
    if shape == "band":
        return ev("scroll", 0.5)

    # 5. type — added value sits near a persistent field label.
    if added:
        field = _field_for(added[0], prev_state, cur_state)
        if field:
            return ev("type", 0.6, field=field, value=added[0])

    # 6. click — a cursor interaction with a small/local change.
    if interaction and shape in (None, "local"):
        return ev("click", 0.6, label=interaction.get("label"))

    # 7. edit — confident-but-unclassified local change.
    if added or removed or shape == "local":
        return ev("edit", 0.4)

    return {}
