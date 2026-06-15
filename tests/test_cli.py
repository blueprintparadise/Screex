from screex.cli import analyze
from screex.core.manifest import Manifest


def test_capture_rejects_multiple_sources():
    import pytest

    from screex.cli import main

    with pytest.raises(SystemExit):
        main(["capture", "--screen", "--webcam"])


def test_capture_screen_passes_controls(monkeypatch, capsys):
    from screex.cli import main
    from screex.core import source

    calls = []
    monkeypatch.setattr(
        source,
        "capture_screen",
        lambda out, seconds, fps=10.0, monitor=1: calls.append(
            (out, seconds, fps, monitor)
        ) or out,
    )

    main(["capture", "--screen", "--seconds", "2.5", "--fps", "12", "--monitor", "2",
          "--out", "screen.mp4"])

    assert calls == [("screen.mp4", 2.5, 12.0, 2)]
    assert "captured: screen.mp4" in capsys.readouterr().out


def test_capture_webcam_passes_controls_by_default(monkeypatch):
    from screex.cli import main
    from screex.core import source

    calls = []
    monkeypatch.setattr(
        source,
        "capture_webcam",
        lambda out, seconds, fps=15.0, device=0: calls.append(
            (out, seconds, fps, device)
        ) or out,
    )

    main(["capture", "--seconds", "1", "--device", "3", "--out", "cam.mp4"])

    assert calls == [("cam.mp4", 1.0, 15.0, 3)]


def test_analyze_produces_manifest_and_files(moving_square_video, tmp_path):
    out = tmp_path / "work"
    manifest_path = analyze(
        str(moving_square_video), fps=5.0, cols=40,
        sensitivity=0.01, edge=False, out=str(out),
    )
    assert manifest_path.exists()

    m = Manifest.load(manifest_path)
    assert m.video == moving_square_video.name
    assert len(m.frames) >= 1
    # the moving square creates motion -> at least one event
    assert len(m.events) >= 1

    # every referenced artifact exists on disk
    for fr in m.frames:
        assert (out / fr.png).exists()
        assert (out / fr.ascii).exists()
    # first frame has zero motion score by construction
    assert m.frames[0].score == 0.0


def test_analyze_events_have_type(moving_square_video, tmp_path):
    out = tmp_path / "work_typed"
    manifest_path = analyze(
        str(moving_square_video), fps=5.0, cols=40,
        sensitivity=0.01, edge=False, out=str(out),
    )
    m = Manifest.load(manifest_path)
    assert len(m.events) >= 1
    for e in m.events:
        assert e.type in {"cut", "motion"}
        assert 0.0 <= e.confidence <= 1.0


def test_index_builds_states_with_ocr(screencast_video, tmp_path):
    from screex.cli import index
    from screex.core.index import ScreenIndex

    out = tmp_path / "work_index"
    index_path = index(str(screencast_video), fps=4.0, change_threshold=0.03, out=str(out))
    assert index_path.exists()

    si = ScreenIndex.load(index_path)
    assert si.video == screencast_video.name
    assert len(si.states) >= 2

    for s in si.states:
        assert (out / s.thumbnail).exists()
        assert (out / s.keyframe).exists()

    all_text = " ".join(" ".join(s.ocr_text) for s in si.states).lower()
    assert any(word in all_text for word in ("settings", "save", "open", "changes"))
    # a later state registers newly-appeared text
    assert any(s.text_added for s in si.states[1:])
    # schema version is written
    assert si.schema_version >= 1


def test_index_merges_duplicate_text_states(dup_text_video, tmp_path):
    from screex.cli import index
    from screex.core.index import ScreenIndex

    out = tmp_path / "work_dup"
    index_path = index(str(dup_text_video), fps=4.0, change_threshold=0.03, out=str(out))
    si = ScreenIndex.load(index_path)
    # two visual segments, identical text -> merged into a single state
    assert len(si.states) == 1
    assert si.states[0].t_end >= si.states[0].t_start


def test_index_jpg_keyframes_and_clean_rerun(screencast_video, tmp_path):
    from screex.cli import index

    out = tmp_path / "work_jpg"
    index(str(screencast_video), fps=4.0, change_threshold=0.03, out=str(out),
          keyframe_format="jpg")
    jpgs = list((out / "frames").glob("*.jpg"))
    assert jpgs, "expected jpg keyframes"
    assert not list((out / "frames").glob("*.png"))

    # re-run must not leave stale files from the previous run behind
    index(str(screencast_video), fps=4.0, change_threshold=0.03, out=str(out),
          keyframe_format="png")
    assert not list((out / "frames").glob("*.jpg"))


def test_index_empty_video_raises(tmp_path):
    import pytest

    from screex.cli import index

    bogus = tmp_path / "empty.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises((ValueError, FileNotFoundError)):
        index(str(bogus), out=str(tmp_path / "o"))


def test_index_with_boxes(screencast_video, tmp_path):
    from screex.cli import index
    from screex.core.index import ScreenIndex

    out = tmp_path / "work_boxes"
    index_path = index(str(screencast_video), fps=4.0, out=str(out), boxes=True, audio=False)
    si = ScreenIndex.load(index_path)
    assert any(s.boxes for s in si.states), "expected at least one state with boxes"
    for s in si.states:
        for b in s.boxes:
            assert "text" in b and "box" in b
            assert len(b["box"]) == 4


def test_index_redacts_secret(secret_video, tmp_path):
    from screex.cli import index
    from screex.core.index import ScreenIndex

    out = tmp_path / "work_redact"
    index_path = index(str(secret_video), fps=4.0, out=str(out), redact=True, audio=False)
    si = ScreenIndex.load(index_path)
    all_text = " ".join(" ".join(s.ocr_text) for s in si.states)
    assert "rushi@acme.io" not in all_text
    assert "REDACTED" in all_text


def test_index_interactions_label(screencast_video, tmp_path):
    from screex.cli import index
    from screex.core.index import ScreenIndex

    out = tmp_path / "work_interactions"
    index_path = index(str(screencast_video), fps=4.0, out=str(out),
                       interactions=True, audio=False)
    si = ScreenIndex.load(index_path)
    # interactions are heuristic; assert the field is well-formed when present
    for s in si.states:
        for it in s.interactions:
            assert {"t", "x", "y", "label"} <= set(it.keys())


def test_transcript_cli_writes_markdown(screencast_video, tmp_path):
    from screex.cli import main

    out_md = tmp_path / "steps.md"
    main(["transcript", str(screencast_video), "--fps", "4", "-o", str(out_md)])
    text = out_md.read_text(encoding="utf-8")
    assert text.startswith("# Transcript")
    assert "State 1" in text


def test_transcript_cli_from_index_does_not_need_recording(tmp_path):
    from screex.cli import main
    from screex.core.index import ScreenIndex, ScreenState

    idx = tmp_path / "index.json"
    out_md = tmp_path / "steps.md"
    ScreenIndex(
        video="cast.mp4",
        duration=1.0,
        sampled_fps=2.0,
        states=[ScreenState(0, 0.0, 1.0, "t.png", "k.png", ocr_text=["Ready"])],
    ).save(idx)

    main(["transcript", "--from-index", str(idx), "-o", str(out_md)])

    assert "Ready" in out_md.read_text(encoding="utf-8")


def test_transcript_cli_requires_recording_without_index():
    import pytest

    from screex.cli import main

    with pytest.raises(SystemExit):
        main(["transcript"])


def test_slow_warning_logic():
    from screex.cli import _slow_warning

    assert _slow_warning(10, 2, fast=False, max_frames=None) is None      # ~20 frames: fine
    assert _slow_warning(200, 2, fast=False, max_frames=None) is not None  # ~400 frames: warn
    assert _slow_warning(200, 2, fast=True, max_frames=None) is None       # fast mode: never warn
    assert _slow_warning(200, 2, fast=False, max_frames=60) is None        # capped to 60: fine


def test_index_text_mode_catches_subtle_change(subtle_screencast_video, tmp_path):
    from screex.cli import index
    from screex.core.index import ScreenIndex

    # default = text mode -> the "Loading" -> "Ready" change becomes a 2nd state
    p_text = index(str(subtle_screencast_video), fps=4.0, out=str(tmp_path / "t"), quiet=True)
    assert len(ScreenIndex.load(p_text).states) >= 2

    # --fast = motion-only -> the subtle change is below --change-threshold -> 1 state
    p_fast = index(str(subtle_screencast_video), fps=4.0, fast=True,
                   out=str(tmp_path / "f"), quiet=True)
    assert len(ScreenIndex.load(p_fast).states) == 1


def test_index_has_text_helper():
    from screex.cli import _index_has_text
    from screex.core.index import ScreenState
    empty = ScreenState(0, 0.0, 1.0, "t.png", "k.png", ocr_text=[], text_added=[], text_removed=[])
    has = ScreenState(1, 1.0, 2.0, "t.png", "k.png", ocr_text=["Hi"], text_added=["Hi"], text_removed=[])
    assert _index_has_text([has]) is True
    assert _index_has_text([empty]) is False
    assert _index_has_text([]) is False


def test_index_skips_audio_when_unavailable(screencast_video, tmp_path, monkeypatch):
    import screex.core.audio as audio_mod
    from screex.cli import index
    from screex.core.index import ScreenIndex
    monkeypatch.setattr(audio_mod, "is_available", lambda: False)
    p = index(str(screencast_video), fps=4.0, out=str(tmp_path / "o"), quiet=True)  # audio default True
    assert ScreenIndex.load(p).narration == []


def test_index_populates_narration_when_available(screencast_video, tmp_path, monkeypatch):
    import screex.core.audio as audio_mod
    from screex.cli import index
    from screex.core.index import NarrationSegment, ScreenIndex
    monkeypatch.setattr(audio_mod, "is_available", lambda: True)
    monkeypatch.setattr(audio_mod, "transcribe",
                        lambda path, model="base": [NarrationSegment(0.0, 1.0, "hello there")])
    p = index(str(screencast_video), fps=4.0, out=str(tmp_path / "o2"), quiet=True)
    assert any(n.text == "hello there" for n in ScreenIndex.load(p).narration)
