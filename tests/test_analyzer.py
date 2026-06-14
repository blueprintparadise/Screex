import numpy as np
from screex.core import analyzer
from screex.core.manifest import EventRecord


def test_motion_score_bounds():
    zero = np.zeros((4, 4), dtype=np.uint8)
    full = np.full((4, 4), 255, dtype=np.uint8)
    assert analyzer.motion_score(zero, zero) == 0.0
    assert analyzer.motion_score(zero, full) == 1.0


def test_flag_events_threshold():
    scores = [0.0, 0.5, 0.6, 0.1, 0.7]
    assert analyzer.flag_events(scores, 0.4) == [False, True, True, False, True]


def test_group_events_segments_and_peak():
    scores = [0.0, 0.5, 0.6, 0.1, 0.7]
    times = [0.0, 0.2, 0.4, 0.6, 0.8]
    flags = analyzer.flag_events(scores, 0.4)
    events = analyzer.group_events(scores, times, flags)
    assert events == [
        EventRecord(t_start=0.2, t_end=0.4, peak_frame=2, peak_score=0.6),
        EventRecord(t_start=0.8, t_end=0.8, peak_frame=4, peak_score=0.7),
    ]


def test_histogram_similarity_identical_is_one():
    f = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert analyzer.histogram_similarity(f, f) == 1.0


def test_histogram_similarity_black_vs_white_is_zero():
    black = np.zeros((8, 8), dtype=np.uint8)
    white = np.full((8, 8), 255, dtype=np.uint8)
    assert analyzer.histogram_similarity(black, white) == 0.0


def test_histogram_similarity_same_distribution_stays_high():
    # identical intensity distribution, different spatial layout -> high similarity
    # (orthogonal to motion: pixels moved, histogram unchanged)
    a = np.zeros((8, 8), dtype=np.uint8); a[:, :4] = 200
    b = np.zeros((8, 8), dtype=np.uint8); b[:, 4:] = 200
    assert analyzer.histogram_similarity(a, b) > 0.9


def test_classify_events_cut_and_motion():
    events = [
        EventRecord(t_start=1.0, t_end=1.0, peak_frame=5, peak_score=0.16),
        EventRecord(t_start=2.2, t_end=2.4, peak_frame=11, peak_score=0.15),
    ]
    sims = [1.0] * 12
    sims[5] = 0.1    # low similarity at peak -> cut
    sims[11] = 0.9   # high similarity at peak -> motion
    out = analyzer.classify_events(events, sims, cut_threshold=0.5)
    assert out[0].type == "cut"
    assert out[0].confidence == 0.8     # (0.5 - 0.1) / 0.5
    assert out[1].type == "motion"
    assert out[1].confidence == 0.8     # (0.9 - 0.5) / 0.5


def test_classify_events_returns_same_objects():
    e = EventRecord(t_start=0.0, t_end=0.0, peak_frame=0, peak_score=0.2)
    out = analyzer.classify_events([e], [0.2], cut_threshold=0.5)
    assert out[0] is e
    assert e.type == "cut"
