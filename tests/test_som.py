import cv2
import numpy as np

from screex.core.som import annotate, select_marks


def _boxes(n, size=40):
    return [{"text": f"t{i}", "box": [i * 5, i * 5, size, size]} for i in range(n)]


def test_select_marks_filters_tiny():
    boxes = [{"text": "big", "box": [0, 0, 40, 40]},
             {"text": "tiny", "box": [50, 50, 5, 5]}]
    marks = select_marks(boxes, min_size=12, max_marks=30)
    assert [m["text"] for m in marks] == ["big"]


def test_select_marks_caps():
    marks = select_marks(_boxes(50), min_size=12, max_marks=30)
    assert len(marks) == 30


def test_annotate_writes_image(tmp_path):
    src = tmp_path / "kf.png"
    cv2.imwrite(str(src), np.zeros((100, 100, 3), dtype=np.uint8))
    out = tmp_path / "marked.png"
    res = annotate(str(src), _boxes(3), str(out))
    assert res == str(out)
    img = cv2.imread(str(out))
    assert img is not None
    assert int(img.sum()) > 0  # something was drawn


def test_annotate_unreadable_copies(tmp_path):
    src = tmp_path / "notimage.txt"
    src.write_text("not an image", encoding="utf-8")
    out = tmp_path / "out.png"
    res = annotate(str(src), _boxes(2), str(out))
    assert res == str(out)
    assert out.exists()
