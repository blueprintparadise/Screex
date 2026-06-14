from __future__ import annotations

from dataclasses import dataclass

from screex.core.analyzer import motion_score


@dataclass
class Segment:
    idx: int
    t_start: float
    t_end: float
    frame_bgr: object  # numpy BGR array: the settled keyframe of this UI state


def segment_stream(frames, change_threshold: float = 0.04):
    """Yield one Segment per UI state.

    A new state begins when *either* the frame-to-frame motion crosses
    ``change_threshold`` (an abrupt change), *or* the cumulative drift away from the
    current state's anchor frame crosses it (a slow change — scrolling, fading, typing
    one character at a time — that never trips the per-frame check). The representative
    keyframe is the last (settled) frame of the state. Memory-bounded: holds only the
    current state's anchor frame and last frame."""
    import cv2

    prev_gray = None
    anchor_gray = None  # first settled frame of the current state
    seg_idx = 0
    cur_start_t = None
    last_t = None
    last_bgr = None

    for _idx, t, bgr in frames:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            cur_start_t = t
            anchor_gray = gray
        else:
            step = motion_score(prev_gray, gray)
            drift = motion_score(anchor_gray, gray)
            if step >= change_threshold or drift >= change_threshold:
                yield Segment(seg_idx, cur_start_t, last_t, last_bgr)
                seg_idx += 1
                cur_start_t = t
                anchor_gray = gray
        prev_gray = gray
        last_t = t
        last_bgr = bgr

    if last_bgr is not None:
        yield Segment(seg_idx, cur_start_t, last_t, last_bgr)
