"""Set-of-Mark annotation: draw numbered boxes (from Screex's OCR boxes) onto a keyframe.

Vision-language models reason about *where* things are far better when regions are explicitly
marked (Yang et al., 2023, "Set-of-Mark Prompting"). Screex already extracts per-line OCR boxes;
this overlays them as numbered rectangles so a curated keyframe carries explicit grounding. Dense
marks hurt, so tiny boxes are dropped and the count is capped. cv2-only — no new dependency.
"""
from __future__ import annotations

import shutil

_RED = (0, 0, 255)


def select_marks(boxes, min_size: int = 12, max_marks: int = 30) -> list:
    """Pick which OCR boxes to mark: drop boxes whose larger side is < ``min_size`` px
    (OCR noise / sub-glyph fragments), then keep the ``max_marks`` largest by area."""
    cand = [b for b in boxes if max(b["box"][2], b["box"][3]) >= min_size]
    cand.sort(key=lambda b: b["box"][2] * b["box"][3], reverse=True)
    return cand[:max_marks]


def annotate(keyframe_path, boxes, out_path, min_size: int = 12, max_marks: int = 30) -> str:
    """Draw numbered red rectangles for the selected OCR ``boxes`` onto the keyframe and
    write it to ``out_path``. Filtering/cap via ``select_marks``. If the keyframe can't be
    read as an image, copy it through unchanged. Returns ``out_path``."""
    import cv2

    img = cv2.imread(str(keyframe_path))
    if img is None:
        shutil.copyfile(str(keyframe_path), str(out_path))
        return str(out_path)
    for i, b in enumerate(select_marks(boxes, min_size, max_marks), start=1):
        x, y, w, h = b["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), _RED, 2)
        cv2.putText(img, str(i), (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _RED, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return str(out_path)
