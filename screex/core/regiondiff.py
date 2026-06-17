from __future__ import annotations

from dataclasses import dataclass

# Per-pixel intensity delta (0-255) that counts a pixel as "changed".
_DIFF_THRESH = 25
# Coverage (fraction of frame pixels changed) at/above which the change is "full".
_FULL_COVERAGE = 0.45
# Bounding-box height/width fractions that mark a large "band" region (scroll-like).
_BAND_H_FRAC = 0.5
_BAND_W_FRAC = 0.5
# Coverage at/below which the change is a small "local" edit/click.
_LOCAL_COVERAGE = 0.08


@dataclass
class RegionChange:
    box: list  # [x, y, w, h] bounding rect of the changed area (pixels)
    coverage: float  # fraction of frame pixels that changed (0..1)
    shape: str  # "full" | "overlay" | "band" | "local"


def changed_region(prev_bgr, cur_bgr):
    """Return a RegionChange describing where ``cur_bgr`` differs from ``prev_bgr``,
    or None when a frame is missing, shapes mismatch, or nothing changed."""
    import cv2
    import numpy as np

    if prev_bgr is None or cur_bgr is None:
        return None
    if getattr(prev_bgr, "shape", None) != getattr(cur_bgr, "shape", None):
        return None

    pg = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(cg, pg)
    _, mask = cv2.threshold(diff, _DIFF_THRESH, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None

    h, w = cg.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    coverage = float((mask > 0).sum()) / float(h * w)

    ar_h = box[3] / h
    ar_w = box[2] / w
    if coverage >= _FULL_COVERAGE:
        shape = "full"
    elif ar_h >= _BAND_H_FRAC and ar_w >= _BAND_W_FRAC:
        shape = "band"
    elif coverage <= _LOCAL_COVERAGE:
        shape = "local"
    else:
        shape = "overlay"

    return RegionChange(box=box, coverage=round(coverage, 4), shape=shape)
