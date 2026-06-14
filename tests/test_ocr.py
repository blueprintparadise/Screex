import numpy as np
import cv2
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
