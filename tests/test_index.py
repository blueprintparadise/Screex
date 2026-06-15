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


def _compact_index():
    return ScreenIndex(
        video="cast.mp4", duration=9.0, sampled_fps=2.0,
        states=[
            ScreenState(idx=0, t_start=0.0, t_end=3.0, thumbnail="t0.png", keyframe="k0.png",
                        ocr_text=["File", "Edit", "Hello"], text_added=[], text_removed=[],
                        boxes=[{"text": "File", "box": [0, 0, 10, 10]}],
                        interactions=[{"t": 1.0, "x": 5, "y": 5, "label": "click"}]),
            ScreenState(idx=1, t_start=3.0, t_end=6.0, thumbnail="t1.png", keyframe="k1.png",
                        ocr_text=["File", "Edit", "World"], text_added=["World"],
                        text_removed=["Hello"]),
            ScreenState(idx=2, t_start=6.0, t_end=9.0, thumbnail="t2.png", keyframe="k2.png",
                        ocr_text=["File", "Edit", "World", "Extra"], text_added=["Extra"],
                        text_removed=[]),
        ],
    )


def test_compact_dict_drops_diffs_boxes_interactions():
    d = _compact_index().compact_dict()
    for s in d["states"]:
        assert "text_added" not in s and "text_removed" not in s
        assert "boxes" not in s and "interactions" not in s


def test_compact_dict_factors_persistent_ui():
    d = _compact_index().compact_dict()
    assert set(d["persistent_ui"]) == {"File", "Edit"}     # present in every state
    for s in d["states"]:
        assert "File" not in s["ocr_text"] and "Edit" not in s["ocr_text"]
    assert d["states"][0]["ocr_text"] == ["Hello"]


def test_compact_dict_preserves_per_state_text_line_sets():
    si = _compact_index()
    original = [set(s.ocr_text) for s in si.states]
    d = si.compact_dict()
    persistent = set(d["persistent_ui"])
    for orig, s in zip(original, d["states"]):
        assert persistent | set(s["ocr_text"]) == orig   # lossless at the text level


def test_compact_dict_can_keep_diffs():
    d = _compact_index().compact_dict(drop_diffs=False)
    assert d["states"][1]["text_added"] == ["World"]


def test_compact_dict_no_universal_lines_omits_persistent_ui():
    si = ScreenIndex(video="x.mp4", duration=2.0, sampled_fps=1.0, states=[
        ScreenState(0, 0.0, 1.0, "t.png", "k.png", ocr_text=["a"]),
        ScreenState(1, 1.0, 2.0, "t.png", "k.png", ocr_text=["b"]),
    ])
    d = si.compact_dict()
    assert "persistent_ui" not in d
    assert d["states"][0]["ocr_text"] == ["a"]
