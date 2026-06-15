import numpy as np

from screex.core import cursor


def _moving_dot_frames(positions, w=80, h=60):
    """Yield (idx, t, bgr) frames with a small white dot at each given (x, y)."""
    for idx, (x, y) in enumerate(positions):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[max(0, y - 2):y + 2, max(0, x - 2):x + 2] = 255
        yield idx, float(idx), frame


def test_tracker_follows_moving_dot():
    positions = [(10, 10), (20, 12), (30, 14), (40, 16)]
    tracker = cursor.CursorTracker(_moving_dot_frames(positions))
    list(tracker)  # drive the generator
    assert tracker.positions, "expected at least one cursor estimate"
    # Frame differencing surfaces the dot's motion region; each estimate should land on
    # the path the dot travelled (not arbitrary noise far from it).
    for _, x, y in tracker.positions:
        assert 5 <= x <= 45, f"x={x} off the motion path"
        assert 5 <= y <= 21, f"y={y} off the motion path"


def test_hotspot_returns_dwell_point():
    pos = [(0.0, 10, 10), (1.0, 20, 20), (2.0, 30, 30)]
    pt = cursor.hotspot(pos, 0.0, 2.0)
    assert pt is not None
    assert pt[0] >= 10 and pt[1] >= 10


def test_hotspot_none_outside_window():
    pos = [(5.0, 10, 10)]
    assert cursor.hotspot(pos, 0.0, 1.0) is None


def test_nearest_label_picks_closest_box():
    boxes = [
        {"text": "Save", "box": [100, 100, 40, 20]},
        {"text": "Cancel", "box": [0, 0, 40, 20]},
    ]
    assert cursor.nearest_label((110, 105), boxes) == "Save"
    assert cursor.nearest_label((5, 5), boxes) == "Cancel"


def test_nearest_label_empty():
    assert cursor.nearest_label(None, []) is None
    assert cursor.nearest_label((1, 2), []) is None
