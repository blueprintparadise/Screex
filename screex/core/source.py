from __future__ import annotations


def _open(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    return cap


def video_info(path: str) -> dict:
    import cv2

    cap = _open(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = count / fps if fps else 0.0
    return {"fps": fps, "count": count, "width": w, "height": h, "duration": duration}


def iter_frames(path: str, sample_fps: float):
    cap = _open(path)
    import cv2

    native = cap.get(cv2.CAP_PROP_FPS) or sample_fps or 1.0
    step = max(1, round(native / sample_fps)) if sample_fps else 1
    raw_idx = 0
    out_idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if raw_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield out_idx, raw_idx / native, frame
                out_idx += 1
            raw_idx += 1
    finally:
        cap.release()


def capture_webcam(out_path: str, seconds: float, fps: float = 15.0, device: int = 0) -> str:
    """Record a short clip from the default webcam into out_path. Manual/hardware path."""
    import cv2

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError("cannot open webcam")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for _ in range(int(seconds * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    return str(out_path)
