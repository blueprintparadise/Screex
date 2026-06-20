from screex.core.index import ScreenIndex, ScreenState
from screex.query import search_index, summarize_index


def _index():
    return ScreenIndex(
        video="cast.mp4", duration=9.0, sampled_fps=2.0,
        states=[
            ScreenState(0, 0.0, 3.0, "t0.png", "k0.png",
                        ocr_text=["Menu", "Open Settings"], salience=0.2),
            ScreenState(1, 3.0, 6.0, "t1.png", "k1.png",
                        ocr_text=["Menu", "Save Changes"],
                        event={"type": "click", "t": 3.0}, salience=0.9),
            ScreenState(2, 6.0, 9.0, "t2.png", "k2.png",
                        ocr_text=["Menu", "Error: invalid key"],
                        event={"type": "error", "t": 6.0}),
        ],
        warnings=["ocr failed on frame 4"],
    )


def test_summarize_index():
    s = summarize_index(_index())
    assert s["states"] == 3
    assert s["duration_s"] == 9.0
    assert s["events"] == {"click": 1, "error": 1}
    assert s["warnings"] == 1
    assert s["persistent_ui_lines"] == 1          # "Menu" in every state
    assert s["has_curated_keyframes"] is True     # some salience > 0


def test_search_by_text_pattern():
    hits = search_index(_index(), pattern="error")
    assert len(hits) == 1
    assert hits[0]["idx"] == 2
    assert hits[0]["matches"] == ["Error: invalid key"]


def test_search_by_event_type():
    hits = search_index(_index(), event_type="click")
    assert [h["idx"] for h in hits] == [1]


def test_search_by_time_window():
    # since=6.5 falls inside state 2 only (state 1 ends at 6.0).
    assert [h["idx"] for h in search_index(_index(), since=6.5)] == [2]
    # until=2.0 falls inside state 0 only (state 1 starts at 3.0).
    assert [h["idx"] for h in search_index(_index(), until=2.0)] == [0]


def test_search_no_filters_returns_all():
    assert len(search_index(_index())) == 3
