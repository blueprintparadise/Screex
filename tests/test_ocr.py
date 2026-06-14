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
