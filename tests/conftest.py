import numpy as np
import pytest


@pytest.fixture
def moving_square_video(tmp_path):
    """A 30-frame AVI (MJPG) where a white square slides across a black frame."""
    import cv2

    path = tmp_path / "clip.avi"
    w, h, fps, n = 64, 48, 15, 30
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert vw.isOpened(), "MJPG VideoWriter failed to open"
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i * 2) % (w - 10)
        frame[10:20, x:x + 10] = 255
        vw.write(frame)
    vw.release()
    return path


@pytest.fixture
def screencast_video(tmp_path):
    """A synthetic screencast: two visually-distinct UI states (different bg tint + text),
    so the change detector fires deterministically and OCR can read the text."""
    import cv2
    import numpy as np

    path = tmp_path / "cast.avi"
    w, h, fps = 360, 240, 4
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert vw.isOpened(), "MJPG VideoWriter failed to open"
    states = [((255, 255, 255), "Open Settings"), ((225, 235, 255), "Save Changes")]
    for bg, text in states:
        for _ in range(6):
            frame = np.full((h, w, 3), bg, dtype=np.uint8)
            cv2.putText(frame, text, (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
            vw.write(frame)
    vw.release()
    return path


@pytest.fixture
def dup_text_video(tmp_path):
    """Two visually-distinct segments (different bg tint) that show the SAME text, so the
    change detector fires but the on-screen text is identical -> dedup should merge them."""
    import cv2
    import numpy as np

    path = tmp_path / "dup.avi"
    w, h, fps = 360, 240, 4
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert vw.isOpened(), "MJPG VideoWriter failed to open"
    for bg in ((255, 255, 255), (200, 210, 255)):
        for _ in range(6):
            frame = np.full((h, w, 3), bg, dtype=np.uint8)
            cv2.putText(frame, "Open Settings", (15, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
            vw.write(frame)
    vw.release()
    return path
