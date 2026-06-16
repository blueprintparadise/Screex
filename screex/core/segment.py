from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from screex.core.analyzer import motion_score


@dataclass
class Segment:
    idx: int
    t_start: float
    t_end: float
    frame_bgr: Any  # numpy BGR array: the settled keyframe of this UI state
    ocr_text: list[str] | None = None


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
            assert anchor_gray is not None
            step = motion_score(prev_gray, gray)
            drift = motion_score(anchor_gray, gray)
            if step >= change_threshold or drift >= change_threshold:
                assert cur_start_t is not None
                assert last_t is not None
                assert last_bgr is not None
                yield Segment(seg_idx, cur_start_t, last_t, last_bgr)
                seg_idx += 1
                cur_start_t = t
                anchor_gray = gray
        prev_gray = gray
        last_t = t
        last_bgr = bgr

    if last_bgr is not None:
        assert cur_start_t is not None
        assert last_t is not None
        yield Segment(seg_idx, cur_start_t, last_t, last_bgr)


def segment_by_text(frames, text_fn, text_threshold: float = 0.80, motion_epsilon: float = 0.003):
    """Yield one Segment per text-distinct UI state.

    A cheap motion pre-filter skips OCR on frames essentially identical to the previous one
    (``motion < motion_epsilon``). Any frame that changed is OCR'd via ``text_fn(bgr) ->
    list[str]``; a new state begins when its text diverges from the current state's
    (``text_similarity < text_threshold``). The representative keyframe is the last (settled)
    frame of the state. Memory-bounded."""
    import cv2

    from screex.core.ocr import text_similarity

    prev_gray = None
    seg_idx = 0
    cur_start_t = None
    state_text = None
    last_t = None
    last_bgr = None

    for _idx, t, bgr in frames:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            cur_start_t = t
            state_text = text_fn(bgr)
        elif motion_score(prev_gray, gray) >= motion_epsilon:
            new_text = text_fn(bgr)
            if text_similarity(state_text, new_text) < text_threshold:
                assert cur_start_t is not None
                assert last_t is not None
                assert last_bgr is not None
                yield Segment(seg_idx, cur_start_t, last_t, last_bgr, state_text)
                seg_idx += 1
                cur_start_t = t
            state_text = new_text
        prev_gray = gray
        last_t = t
        last_bgr = bgr

    if last_bgr is not None:
        assert cur_start_t is not None
        assert last_t is not None
        yield Segment(seg_idx, cur_start_t, last_t, last_bgr, state_text)
