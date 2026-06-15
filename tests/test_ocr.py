import cv2
import numpy as np

from screex.core import ocr


def test_extract_text_reads_rendered_text():
    img = np.full((140, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Open Settings", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    lines = ocr.extract_text(img)
    joined = " ".join(lines).lower()
    assert "open" in joined
    assert "settings" in joined


def test_extract_text_boxes_returns_boxes():
    img = np.full((140, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Open Settings", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    items = ocr.extract_text_boxes(img)
    assert items, "expected at least one detected text box"
    for it in items:
        assert "text" in it and "box" in it
        x, y, w, h = it["box"]
        assert w > 0 and h > 0
        assert all(isinstance(v, int) for v in it["box"])
    # text from boxes matches the flat extractor
    assert [it["text"] for it in items] == ocr.extract_text(img)


def test_text_diff_added_and_removed():
    added, removed = ocr.text_diff(["a", "b"], ["b", "c"])
    assert added == ["c"]
    assert removed == ["a"]


def test_text_diff_empty_prev():
    added, removed = ocr.text_diff([], ["x", "y"])
    assert added == ["x", "y"]
    assert removed == []


def test_text_diff_ignores_ocr_noise():
    # Same line, minor OCR noise (casing, stray glyph, punctuation) -> no spurious diff.
    added, removed = ocr.text_diff(["Open Settings"], ["open  settings!"])
    assert added == []
    assert removed == []


def test_text_diff_preserves_order_and_duplicates():
    added, removed = ocr.text_diff(["a"], ["b", "c", "b"])
    assert added == ["b", "c", "b"]
    assert removed == ["a"]


def test_text_similarity():
    assert ocr.text_similarity([], []) == 1.0
    assert ocr.text_similarity(["a"], []) == 0.0
    assert ocr.text_similarity(["Open Settings"], ["open settings"]) == 1.0
    assert 0.0 < ocr.text_similarity(["a", "b"], ["b", "c"]) < 1.0


def test_extract_text_accepts_threads_and_reads_text():
    import cv2
    import numpy as np
    img = np.full((140, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Save", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    lines = ocr.extract_text(img, threads=2)
    assert any("save" in line.lower() for line in lines)


def test_extract_text_tolerates_frame_failure(monkeypatch):
    import numpy as np

    class Boom:
        def __call__(self, img):
            raise RuntimeError("bad frame")

    monkeypatch.setattr(ocr, "_get_engine", lambda *a, **k: Boom())
    assert ocr.extract_text(np.zeros((4, 4, 3), dtype="uint8")) == []


def test_get_engine_caches_per_threads(monkeypatch):
    import sys
    import types
    created = []

    class Fake:
        def __init__(self, **kw):
            created.append(kw)

        def __call__(self, img):
            return ([], None)

    monkeypatch.setattr(ocr, "_engines", {})
    fake_mod = types.ModuleType("rapidocr_onnxruntime")
    fake_mod.RapidOCR = Fake
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_mod)

    e1 = ocr._get_engine(None, 2)
    e2 = ocr._get_engine(None, 2)
    e3 = ocr._get_engine(None, 1)
    assert e1 is e2          # same (lang, threads) -> cached
    assert e1 is not e3      # different threads -> different engine
    assert {"intra_op_num_threads": 2} in created
