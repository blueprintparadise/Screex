from screex.core.index import ScreenIndex, ScreenState


def test_screen_index_roundtrip(tmp_path):
    si = ScreenIndex(
        video="cast.mp4", duration=12.0, sampled_fps=2.0,
        states=[
            ScreenState(idx=0, t_start=0.0, t_end=3.0,
                        thumbnail="frames/00000_thumb.png", keyframe="frames/00000.png",
                        ocr_text=["Open Settings"], text_added=["Open Settings"], text_removed=[]),
            ScreenState(idx=1, t_start=3.0, t_end=6.0,
                        thumbnail="frames/00001_thumb.png", keyframe="frames/00001.png",
                        ocr_text=["Save Changes"], text_added=["Save Changes"], text_removed=["Open Settings"]),
        ],
    )
    path = tmp_path / "index.json"
    si.save(path)
    loaded = ScreenIndex.load(path)
    assert loaded == si
    assert loaded.states[1].text_removed == ["Open Settings"]


def test_screen_index_narration_roundtrip(tmp_path):
    from screex.core.index import NarrationSegment
    si = ScreenIndex(
        video="cast.mp4", duration=6.0, sampled_fps=2.0,
        states=[ScreenState(0, 0.0, 3.0, "frames/0_thumb.png", "frames/0.png",
                            ocr_text=["Hi"], text_added=["Hi"], text_removed=[])],
        narration=[NarrationSegment(0.5, 2.0, "click save")],
    )
    path = tmp_path / "index.json"
    si.save(path)
    loaded = ScreenIndex.load(path)
    assert loaded == si
    assert loaded.narration[0].text == "click save"


def test_screen_index_loads_without_narration_key():
    d = {"video": "x.mp4", "duration": 1.0, "sampled_fps": 2.0, "states": []}
    si = ScreenIndex.from_dict(d)
    assert si.narration == []
    assert si.warnings == []


def test_screen_index_rejects_future_schema():
    import pytest

    d = {
        "schema_version": 999,
        "video": "x.mp4",
        "duration": 1.0,
        "sampled_fps": 2.0,
        "states": [],
    }
    with pytest.raises(ValueError, match="unsupported ScreenIndex schema_version"):
        ScreenIndex.from_dict(d)


def test_screen_index_reports_missing_required_fields():
    import pytest

    with pytest.raises(ValueError, match="missing video"):
        ScreenIndex.from_dict({"duration": 1.0, "sampled_fps": 2.0, "states": []})


def test_screen_index_loads_older_state_without_text_fields():
    d = {
        "video": "x.mp4",
        "duration": 1.0,
        "sampled_fps": 2.0,
        "states": [{
            "idx": 0,
            "t_start": 0.0,
            "t_end": 1.0,
            "thumbnail": "frames/0_thumb.png",
            "keyframe": "frames/0.png",
        }],
    }
    si = ScreenIndex.from_dict(d)
    assert si.states[0].ocr_text == []
    assert si.states[0].text_added == []
    assert si.states[0].text_removed == []
