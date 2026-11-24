import importlib.util
import json
from pathlib import Path

from screex.core.index import ScreenIndex, ScreenState


def _load_eval_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("screex_eval_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_estimate_reuses_existing_index(screencast_video, tmp_path, monkeypatch):
    eval_script = _load_eval_module()
    index_path = tmp_path / "index.json"
    ScreenIndex(
        video=screencast_video.name,
        duration=2.0,
        sampled_fps=2.0,
        states=[
            ScreenState(0, 0.0, 1.0, "frames/0_thumb.png", "frames/0.png",
                        ocr_text=["Open Settings"], text_added=["Open Settings"]),
            ScreenState(1, 1.0, 2.0, "frames/1_thumb.png", "frames/1.png",
                        ocr_text=["Save"], text_added=["Save"]),
        ],
    ).save(index_path)

    def fail_index(*args, **kwargs):
        raise AssertionError("estimate should reuse --from-index")

    monkeypatch.setattr("screex.cli.index", fail_index)
    result = eval_script.estimate(
        str(screencast_video),
        fps=2.0,
        escalate=1,
        tokens_per_image=1000,
        from_index=str(index_path),
    )

    assert result["states"] == 2
    assert result["escalated_images"] == 1
    assert result["screex_tokens"] > 1000


def test_mock_answerer_picks_overlapping_choice_deterministically():
    eval_script = _load_eval_module()
    ans = eval_script.MockAnswerer()
    view = "Settings > API Keys\nError: invalid API key format"
    choices = ["A) 404 not found", "B) invalid API key format", "C) timed out"]
    pick1 = ans.answer("what error?", choices, view)
    pick2 = ans.answer("what error?", choices, view)
    assert pick1 == pick2 == 1           # deterministic, overlaps choice B
    # No readable text view (the raw-frames arm) -> deterministic fallback to first choice.
    assert ans.answer("what error?", choices, []) == 0


def test_accuracy_harness_buckets_and_scores(screencast_video, tmp_path, monkeypatch):
    eval_script = _load_eval_module()

    # A prebuilt index whose OCR text contains the correct answer for the "state" question.
    index_path = tmp_path / "index.json"
    ScreenIndex(
        video=screencast_video.name, duration=2.0, sampled_fps=2.0,
        states=[
            ScreenState(0, 0.0, 1.0, "frames/0_thumb.png", "frames/0.png",
                        ocr_text=["Open Settings"], text_added=["Open Settings"]),
            ScreenState(1, 1.0, 2.0, "frames/1_thumb.png", "frames/1.png",
                        ocr_text=["Save Changes"], text_added=["Save Changes"]),
        ],
    ).save(index_path)

    monkeypatch.setattr("screex.cli.index", lambda *a, **k: index_path)
    monkeypatch.setattr("screex.core.source.video_info", lambda path: {"duration": 2.0})

    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        json.dumps({"clip": screencast_video.name, "type": "state",
                    "question": "What did the screen show?",
                    "choices": ["A) Delete account", "B) Save Changes"], "answer": "B"}) + "\n"
        + json.dumps({"clip": screencast_video.name, "type": "action",
                      "question": "What happened?",
                      "choices": ["A) nothing", "B) error"], "answer": "A"}) + "\n",
        encoding="utf-8",
    )

    result = eval_script.score_accuracy(
        str(qa_path), fps=2.0, frames=8, tokens_per_image=1000, view="compact",
        answerer=eval_script.MockAnswerer(), clips_dir=str(tmp_path),
    )

    buckets = result["buckets"]
    assert set(buckets) == {"state", "action"}
    assert buckets["state"]["n"] == 1 and buckets["action"]["n"] == 1
    # "Save Changes" is in the index -> mock picks B -> correct on the state bucket.
    assert buckets["state"]["idx_ok"] == 1
    # Both arms report token counts; frames arm uses the flat per-image cost.
    # sampled frames = max(states=2, round(duration*fps)=round(2.0*2.0)=4) = 4, capped at --frames=8.
    expected_frames = min(8, max(2, round(2.0 * 2.0)))
    for b in buckets.values():
        assert b["idx_tok"] > 0
        assert b["fr_tok"] == expected_frames * 1000
    assert result["answerer"] == "MockAnswerer"
    assert result["skipped"] == []


def test_estimate_can_disable_audio_when_building(monkeypatch, tmp_path):
    eval_script = _load_eval_module()
    calls = []
    index_path = tmp_path / "index.json"
    ScreenIndex(
        video="clip.mp4",
        duration=1.0,
        sampled_fps=1.0,
        states=[ScreenState(0, 0.0, 1.0, "t.png", "k.png", ocr_text=["Hi"])],
    ).save(index_path)

    def fake_index(recording, fps, out=None, quiet=True, audio=True):
        calls.append({"recording": recording, "fps": fps, "audio": audio})
        return index_path

    monkeypatch.setattr("screex.cli.index", fake_index)
    monkeypatch.setattr("screex.core.source.video_info", lambda path: {"duration": 1.0})

    eval_script.estimate(
        "clip.mp4",
        fps=1.0,
        escalate=1,
        tokens_per_image=1000,
        audio=False,
    )

    assert calls == [{"recording": "clip.mp4", "fps": 1.0, "audio": False}]


def test_hybrid_accuracy_uses_curated_keyframe_and_charges_its_tokens(tmp_path, monkeypatch):
    eval_script = _load_eval_module()
    index_dir = tmp_path / "index"
    frames_dir = index_dir / "frames"
    frames_dir.mkdir(parents=True)
    curated = frames_dir / "curated.png"
    curated.write_bytes(b"curated frame")
    uniform = tmp_path / "uniform.png"
    uniform.write_bytes(b"uniform frame")
    index_path = index_dir / "index.json"
    ScreenIndex(
        video="clip.mp4", duration=2.0, sampled_fps=2.0,
        states=[
            ScreenState(0, 0.0, 1.0, "t0.png", "frames/first.png", salience=0.1),
            ScreenState(1, 1.0, 2.0, "t1.png", "frames/curated.png",
                        ocr_text=["Settings"], salience=0.9),
        ],
    ).save(index_path)
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(json.dumps({
        "clip": "clip.mp4", "type": "state", "question": "What was shown?",
        "choices": ["Search", "Settings"], "answer": "B",
    }), encoding="utf-8")

    def build_index(path, *, fps, quiet, audio, keyframe_budget):
        assert keyframe_budget == 1
        return index_path

    monkeypatch.setattr("screex.cli.index", build_index)
    monkeypatch.setattr("screex.core.source.video_info", lambda path: {"duration": 2.0})
    monkeypatch.setattr(eval_script, "_materialize_frames", lambda path, n: [str(uniform)])

    class ImageAnswerer:
        needs_images = True

        def __init__(self):
            self.calls = []

        def answer(self, question, choices, text=None, images=None):
            self.calls.append((text, images))
            return int(text is not None)

    answerer = ImageAnswerer()
    result = eval_script.score_accuracy(
        qa_path, fps=2.0, frames=1, tokens_per_image=1000, view="compact",
        answerer=answerer, keyframe_budget=1,
    )

    (index_text, index_images), (frame_text, frame_images) = answerer.calls
    assert "Settings" in index_text
    assert index_images == [str(curated)]
    assert frame_text is None
    assert frame_images == [str(uniform)]
    assert result["skipped"] == []
    assert result["buckets"]["state"] == {
        "n": 1, "idx_ok": 1, "fr_ok": 0,
        "idx_tok": len(index_text) // 4 + 1000, "fr_tok": 1000,
    }
