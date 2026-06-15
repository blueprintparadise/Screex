"""Heuristic cursor / interaction tracking.

Screen recordings carry no input events, so Screex *estimates* where the user was acting
from the pixels alone: a moving mouse cursor produces a small, localized region of change
between consecutive frames. ``CursorTracker`` follows that region while frames stream past
the segmenter, and ``hotspot`` + ``nearest_label`` turn the dwell point of each UI state
into an "interacted near <text>" annotation.

This is a best-effort heuristic — it cannot detect true mouse-button clicks and will be
noisy on busy video. It is opt-in (``screex index --interactions``).
"""
from __future__ import annotations

from typing import Any, cast


def _estimate_cursor(prev_gray, cur_gray, min_area: int = 4, max_area_frac: float = 0.01):
    """Return (x, y) of the most likely cursor location between two grayscale frames,
    or None. The cursor is a small, compact region of change; large regions are UI
    repaints and are ignored."""
    import cv2

    diff = cv2.absdiff(cur_gray, prev_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8)
    stats = cast(Any, stats)
    centroids = cast(Any, centroids)
    if n <= 1:
        return None

    h, w = cur_gray.shape[:2]
    max_area = max_area_frac * h * w
    best = None
    best_area = -1.0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        # Among plausible cursor-sized blobs, prefer the largest (most motion energy).
        if area > best_area:
            best_area = area
            best = (float(centroids[i][0]), float(centroids[i][1]))
    if best is None:
        return None
    return (int(best[0]), int(best[1]))


class CursorTracker:
    """Wraps a frame iterator, passing frames through unchanged while recording cursor
    position estimates into ``self.positions`` as ``(t, x, y)`` tuples."""

    def __init__(self, frames):
        self._frames = frames
        self.positions: list[tuple[float, int, int]] = []

    def __iter__(self):
        import cv2

        prev = None
        for idx, t, bgr in self._frames:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                pt = _estimate_cursor(prev, gray)
                if pt is not None:
                    self.positions.append((t, pt[0], pt[1]))
            prev = gray
            yield idx, t, bgr


def hotspot(positions, t_start: float, t_end: float, dwell: int = 5):
    """Return the (x, y) dwell point for a UI state: the median of the last ``dwell``
    cursor positions within [t_start, t_end], or None if there were none."""
    pts = [(x, y) for (t, x, y) in positions if t_start <= t <= t_end]
    if not pts:
        return None
    last = pts[-dwell:]
    xs = sorted(p[0] for p in last)
    ys = sorted(p[1] for p in last)
    return (xs[len(xs) // 2], ys[len(ys) // 2])


def nearest_label(point, boxes, max_dist_frac: float = 0.15, frame_size=None):
    """Return the text of the OCR box nearest ``point``, or None. If ``frame_size``
    (w, h) is given, ignore labels farther than ``max_dist_frac`` of the diagonal."""
    if point is None or not boxes:
        return None
    px, py = point
    best = None
    best_d = None
    for b in boxes:
        x, y, w, h = b["box"]
        cx, cy = x + w / 2.0, y + h / 2.0
        d = (cx - px) ** 2 + (cy - py) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best = b["text"]
    if frame_size and best_d is not None:
        fw, fh = frame_size
        limit = (max_dist_frac * (fw ** 2 + fh ** 2) ** 0.5) ** 2
        if best_d > limit:
            return None
    return best
