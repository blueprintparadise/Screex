from __future__ import annotations

import numpy as np

from screex.core.manifest import EventRecord


def motion_score(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    a = np.asarray(prev_gray, dtype=np.int16)
    b = np.asarray(cur_gray, dtype=np.int16)
    return float(np.abs(b - a).mean()) / 255.0


def histogram_similarity(prev_gray, cur_gray, bins: int = 64) -> float:
    a = np.asarray(prev_gray)
    b = np.asarray(cur_gray)
    if a.shape == b.shape and np.array_equal(a, b):
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(0, 256))
    hb, _ = np.histogram(b, bins=bins, range=(0, 256))
    ha = ha.astype(np.float64)
    hb = hb.astype(np.float64)
    if ha.sum() > 0:
        ha /= ha.sum()
    if hb.sum() > 0:
        hb /= hb.sum()
    if ha.std() == 0 or hb.std() == 0:
        return 1.0 if np.array_equal(ha, hb) else 0.0
    corr = float(np.corrcoef(ha, hb)[0, 1])
    return max(0.0, corr)


def flag_events(scores, threshold: float):
    return [s >= threshold for s in scores]


def group_events(scores, times, flags):
    events = []
    n = len(flags)
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        peak = max(range(i, j + 1), key=lambda k: scores[k])
        events.append(
            EventRecord(
                t_start=times[i],
                t_end=times[j],
                peak_frame=peak,
                peak_score=scores[peak],
            )
        )
        i = j + 1
    return events


def classify_events(events, similarities, cut_threshold: float = 0.5):
    for e in events:
        s = similarities[e.peak_frame]
        if s < cut_threshold:
            e.type = "cut"
            e.confidence = round((cut_threshold - s) / cut_threshold, 3) if cut_threshold > 0 else 1.0
        else:
            denom = 1.0 - cut_threshold
            e.type = "motion"
            e.confidence = round((s - cut_threshold) / denom, 3) if denom > 0 else 1.0
    return events
