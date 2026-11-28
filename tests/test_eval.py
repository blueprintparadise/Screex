import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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


@pytest.mark.parametrize(("pick", "answer", "expected"), [
    (0, "Save", True),
    (1, "Save", False),
    (1, " B) Save as ", True),
    (1, " b ", True),
    (0, "b", False),
    (0, "", False),
    (1, "B)", False),
    (-1, "Save", False),
])
def test_accuracy_requires_a_matching_choice(pick, answer, expected):
    eval_script = _load_eval_module()
    choices = ["A) Save", "B) Save as"]

    assert eval_script._is_correct(pick, choices, answer) is expected


@pytest.mark.parametrize(("text", "expected"), [
    (" B. ", 1), ("a", 0), ("", -1), ("Answer: B", -1), ("Z", -1),
])
def test_claude_answerer_requires_a_valid_reply(text, expected):
    eval_script = _load_eval_module()
    messages = Mock()
    messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
    )
    answerer = eval_script.ClaudeAnswerer.__new__(eval_script.ClaudeAnswerer)
    answerer._anthropic = SimpleNamespace(Anthropic=lambda: SimpleNamespace(messages=messages))

    assert answerer.answer("What appeared?", ["Save", "Search"], "Screen text") == expected


@pytest.mark.parametrize(("sdk", "message"), [
    (None, "pip install 'screex[eval]'"),
    (SimpleNamespace(), "ANTHROPIC_API_KEY is not set"),
])
def test_eval_cli_reports_missing_claude_setup(sdk, message, monkeypatch, capsys):
    eval_script = _load_eval_module()
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        eval_script.main(["--qa", "qa.jsonl", "--answerer", "claude", "--json"])

    output = capsys.readouterr()
    assert exc.value.code == 2
    assert message in output.err
    assert output.out == ""


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
    def materialize_short_clip(path, n):
        assert n == 2
        return nullcontext([str(uniform)])

    monkeypatch.setattr(eval_script, "_materialize_frames", materialize_short_clip)

    class ImageAnswerer:
        needs_images = True

        def __init__(self):
            self.calls = []

        def answer(self, question, choices, text=None, images=None):
            self.calls.append((text, images))
            return int(text is not None)

    answerer = ImageAnswerer()
    result = eval_script.score_accuracy(
        qa_path, fps=2.0, frames=2, tokens_per_image=1000, view="compact",
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


@pytest.mark.parametrize(("args", "message"), [
    (["clip.mp4", "--keyframe-budget", "1"], "--keyframe-budget requires --qa"),
    (["--qa", "qa.jsonl", "--keyframe-budget", "0"], "--keyframe-budget must be greater than 0"),
    (["--qa", "qa.jsonl", "--keyframe-budget", "-1"], "--keyframe-budget must be greater than 0"),
])
def test_eval_cli_rejects_invalid_keyframe_budget(args, message, capsys):
    eval_script = _load_eval_module()

    with pytest.raises(SystemExit) as exc:
        eval_script.main(args)

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("interrupted", [False, True])
def test_uniform_images_are_scoped_to_the_consumer(moving_square_video, interrupted):
    module = _load_eval_module()
    with pytest.raises(RuntimeError, match="answer failed") if interrupted else nullcontext():
        with module._materialize_frames(moving_square_video, 2) as images:
            paths = [Path(image) for image in images]
            assert len(paths) == 2
            assert all(path.is_file() for path in paths)
            if interrupted:
                raise RuntimeError("answer failed")
    assert not paths[0].parent.exists()


@pytest.mark.parametrize("requested", [1, 16, 40])
def test_uniform_sampling_covers_the_decoded_clip(moving_square_video, requested):
    import cv2
    import numpy as np

    from screex.core.source import iter_frames

    reference = [frame for _, _, frame in iter_frames(str(moving_square_video), sample_fps=0)]
    count = min(requested, len(reference))
    targets = np.linspace(0, len(reference) - 1, count).round().astype(int)
    with _load_eval_module()._materialize_frames(moving_square_video, requested) as paths:
        assert len(paths) == count
        for path, target in zip(paths, targets):
            np.testing.assert_array_equal(cv2.imread(path), reference[target])
