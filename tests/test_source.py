from screex.core import source


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
