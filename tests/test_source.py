import pytest

from screex.core import source


def test_open_missing_file_raises_filenotfound(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError, match="no such video file"):
        source.video_info(str(missing))


def test_open_corrupt_file_raises_valueerror(tmp_path):
    bogus = tmp_path / "corrupt.mp4"
    bogus.write_bytes(b"not a video at all")
    with pytest.raises(ValueError, match="cannot decode video"):
        source.video_info(str(bogus))


def test_iter_frames_truncated_video_is_graceful(moving_square_video):
    # A half-written file is either rejected at open (clear ValueError) or read
    # partially without crashing; in the partial case any diagnostics are strings.
    data = moving_square_video.read_bytes()
    truncated = moving_square_video.parent / "trunc.avi"
    truncated.write_bytes(data[: len(data) // 2])

    diagnostics: list[str] = []
    try:
        frames = list(source.iter_frames(str(truncated), sample_fps=5.0,
                                          diagnostics=diagnostics))
    except ValueError:
        return  # opening a truncated container failed outright — the clear-error path
    assert isinstance(frames, list)
    assert all(isinstance(d, str) for d in diagnostics)


def test_video_info(moving_square_video):
    info = source.video_info(str(moving_square_video))
    assert info["width"] == 64
    assert info["height"] == 48
    assert info["count"] >= 1
    assert info["duration"] > 0


def test_iter_frames_samples(moving_square_video):
    frames = list(source.iter_frames(str(moving_square_video), sample_fps=5.0))
    assert len(frames) >= 1
    idx0, t0, bgr0 = frames[0]
    assert idx0 == 0
    assert bgr0.shape == (48, 64, 3)
    assert [f[0] for f in frames] == list(range(len(frames)))


def test_iter_frames_handles_zero_sample_fps(moving_square_video):
    # sample_fps=0 must not crash with ZeroDivisionError; it should still yield frames
    frames = list(source.iter_frames(str(moving_square_video), sample_fps=0))
    assert len(frames) >= 1
    # timestamps are finite numbers
    assert all(isinstance(f[1], float) for f in frames)
