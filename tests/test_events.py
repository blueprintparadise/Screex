from screex.core.events import classify_event
from screex.core.index import ScreenState
from screex.core.regiondiff import RegionChange


def _state(idx=1, t=1.0, ocr=None, added=None, removed=None, boxes=None):
    return ScreenState(idx=idx, t_start=t, t_end=t + 1, thumbnail="t.png", keyframe="k.png",
                       ocr_text=ocr or [], text_added=added or [], text_removed=removed or [],
                       boxes=boxes or [])


def test_error_event():
    prev = _state(0, 0.0, ocr=["Dashboard"])
    cur = _state(1, 1.0, ocr=["Dashboard", "Error: invalid API key"],
                 added=["Error: invalid API key"])
    ev = classify_event(prev, cur, None, None)
    assert ev["type"] == "error"
    assert "invalid" in ev["label"]


def test_navigate_event_with_url():
    prev = _state(0, 0.0, ocr=["Home", "About", "Contact"])
    cur = _state(1, 1.0, ocr=["acme.io/settings", "Profile"],
                 added=["acme.io/settings", "Profile"], removed=["Home", "About", "Contact"])
    rc = RegionChange(box=[0, 0, 640, 480], coverage=0.9, shape="full")
    ev = classify_event(prev, cur, rc, None)
    assert ev["type"] == "navigate"
    assert ev["value"] == "acme.io/settings"


def test_open_dialog_event():
    prev = _state(0, 0.0, ocr=["Projects"])
    cur = _state(1, 1.0, ocr=["Projects", "Add project"], added=["Add project"])
    rc = RegionChange(box=[170, 140, 300, 200], coverage=0.2, shape="overlay")
    ev = classify_event(prev, cur, rc, None)
    assert ev["type"] == "open_dialog"
    assert ev["label"] == "Add project"


def test_scroll_event():
    prev = _state(0, 0.0, ocr=["row1", "row2", "row3"])
    cur = _state(1, 1.0, ocr=["row2", "row3", "row4"], added=["row4"], removed=["row1"])
    rc = RegionChange(box=[0, 0, 640, 360], coverage=0.3, shape="band")
    ev = classify_event(prev, cur, rc, None)
    assert ev["type"] == "scroll"


def test_type_event_near_label():
    prev = _state(0, 0.0, ocr=["Email"])
    cur = _state(1, 1.0, ocr=["Email", "rushi@acme.io"], added=["rushi@acme.io"],
                 boxes=[{"text": "Email", "box": [10, 10, 40, 20]},
                        {"text": "rushi@acme.io", "box": [60, 10, 120, 20]}])
    rc = RegionChange(box=[60, 10, 120, 20], coverage=0.02, shape="local")
    ev = classify_event(prev, cur, rc, None)
    assert ev["type"] == "type"
    assert ev["field"] == "Email"
    assert ev["value"] == "rushi@acme.io"


def test_click_event_from_interaction():
    prev = _state(0, 0.0, ocr=["Settings"])
    cur = _state(1, 1.0, ocr=["Settings"])
    rc = RegionChange(box=[100, 100, 10, 10], coverage=0.001, shape="local")
    ev = classify_event(prev, cur, rc, {"t": 1.0, "x": 105, "y": 105, "label": "Save"})
    assert ev["type"] == "click"
    assert ev["label"] == "Save"


def test_no_event_when_no_signal():
    prev = _state(0, 0.0, ocr=["Same"])
    cur = _state(1, 1.0, ocr=["Same"])
    assert classify_event(prev, cur, None, None) == {}
