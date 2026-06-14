import numpy as np

from screex.core import segment


def _frames(seq):
    # seq: list of (t, fill_value); yields (idx, t, bgr 20x20)
    for idx, (t, val) in enumerate(seq):
        yield idx, t, np.full((20, 20, 3), val, dtype=np.uint8)


def test_segment_stream_two_states_on_big_change():
    seq = [(0.0, 0), (0.5, 0), (1.0, 0), (1.5, 255), (2.0, 255), (2.5, 255)]
    segs = list(segment.segment_stream(_frames(seq), change_threshold=0.5))
    assert len(segs) == 2
    assert segs[0].t_start == 0.0
    assert segs[0].t_end == 1.0
    assert segs[1].t_start == 1.5
    assert segs[1].t_end == 2.5
    assert int(segs[0].frame_bgr.mean()) == 0     # settled (black) keyframe
    assert int(segs[1].frame_bgr.mean()) == 255   # settled (white) keyframe


def test_segment_stream_single_state_when_static():
    seq = [(0.0, 100), (1.0, 100), (2.0, 100), (3.0, 100)]
    segs = list(segment.segment_stream(_frames(seq), change_threshold=0.5))
    assert len(segs) == 1
    assert segs[0].t_start == 0.0
    assert segs[0].t_end == 3.0


def test_segment_stream_splits_on_slow_drift():
    # A gradual ramp: each step is small (~10/255 ≈ 0.039 < threshold) so per-frame
    # motion never fires, but cumulative drift from the anchor must eventually split.
    seq = [(float(i), i * 10) for i in range(12)]  # 0,10,...,110
    threshold = 0.1  # ~25.5 gray levels; no single step reaches it
    segs = list(segment.segment_stream(_frames(seq), change_threshold=threshold))
    assert len(segs) >= 2  # drift detection split the slow change
