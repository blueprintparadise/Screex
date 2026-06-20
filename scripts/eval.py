#!/usr/bin/env python
"""Runnable eval for Screex.

Two modes:

* **Cost** (default): compares the cost of the Screex path (skim on-screen text across states,
  escalate a few keyframes) against the baseline (read every sampled frame as an image), in
  estimated input tokens, so the README's "cheap by design" claim is measurable.

      python scripts/eval.py path/to/recording.mp4 --fps 2 [--escalate 3]

* **Accuracy** (opt-in, ``--qa qa.jsonl``): scores two arms — the Screex index vs uniform-N raw
  frames — on a small bucketed multiple-choice set, reporting **accuracy AND tokens per question
  type** (action / state / count / visual). This is the template for proving an index beats raw
  frames. The answerer is pluggable: a deterministic, offline ``mock`` (the default, for CI and
  plumbing) or an optional ``claude`` answerer (needs ``anthropic`` + ``ANTHROPIC_API_KEY``).

      python scripts/eval.py --qa qa.jsonl --frames 8 [--view compact] [--answerer mock]

Token estimates use coarse public-ish rules of thumb and are meant for *relative* comparison,
not billing accuracy:
  - text:  ~1 token per 4 characters
  - image: a flat per-image tile cost (see --tokens-per-image)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Allow running as `python scripts/eval.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTION_TYPES = ("action", "state", "count", "visual")


def estimate(recording, fps, escalate, tokens_per_image, out=None, from_index=None, audio=True):
    from screex.cli import index
    from screex.core import source
    from screex.core.index import ScreenIndex

    index_path = from_index or index(recording, fps=fps, out=out, quiet=True, audio=audio)
    si = ScreenIndex.load(index_path)

    text_chars = sum(len("\n".join(s.ocr_text)) for s in si.states)
    text_tokens = round(text_chars / 4)

    # Screex path: read all text, escalate to `escalate` keyframes.
    escalated = min(escalate, len(si.states))
    screex_tokens = text_tokens + escalated * tokens_per_image

    # Baseline: read every sampled frame as an image.
    info = source.video_info(str(recording))
    sampled_frames = max(len(si.states), round(info["duration"] * fps))
    baseline_tokens = sampled_frames * tokens_per_image

    ratio = screex_tokens / baseline_tokens if baseline_tokens else 0.0
    return {
        "recording": Path(recording).name,
        "duration_s": round(info["duration"], 2),
        "states": len(si.states),
        "sampled_frames": sampled_frames,
        "text_chars": text_chars,
        "text_tokens": text_tokens,
        "escalated_images": escalated,
        "screex_tokens": screex_tokens,
        "baseline_tokens": baseline_tokens,
        "cost_ratio": round(ratio, 4),
    }


# --------------------------------------------------------------------------- #
# Accuracy mode
# --------------------------------------------------------------------------- #

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _strip_choice_label(choice: str) -> str:
    """Drop a leading enumeration label like ``A)`` / ``B.`` / ``c:`` from a choice."""
    return re.sub(r"^\s*[A-Za-z][).:]\s*", "", choice)


class MockAnswerer:
    """Deterministic, offline answerer for CI and pipeline validation.

    Given a *text* view it picks the choice whose words overlap the view most (ties → first
    choice). Given a non-text view (the raw-frames arm has only images) it has nothing to read,
    so it falls back to the first choice. Mock results validate the harness mechanics; they are
    NOT a real accuracy signal — use ``--answerer claude`` for that."""

    needs_images = False

    def answer(self, question: str, choices: list[str], view) -> int:
        if not isinstance(view, str) or not view.strip():
            return 0
        view_tokens = _tokens(view)
        best_i, best_score = 0, -1
        for i, choice in enumerate(choices):
            score = len(_tokens(_strip_choice_label(choice)) & view_tokens)
            if score > best_score:
                best_i, best_score = i, score
        return best_i


class ClaudeAnswerer:
    """Optional answerer backed by the Anthropic API. Never a required dependency — imported
    lazily and only constructed when ``anthropic`` and ``ANTHROPIC_API_KEY`` are available."""

    needs_images = True
    model = "claude-opus-4-8"

    def __init__(self):
        import anthropic  # noqa: F401  (presence check; raises ImportError if missing)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._anthropic = anthropic

    def _letter_prompt(self, question: str, choices: list[str]) -> str:
        lines = [f"{chr(65 + i)}. {_strip_choice_label(c)}" for i, c in enumerate(choices)]
        return (f"{question}\n\n" + "\n".join(lines) +
                "\n\nAnswer with only the single letter of the best choice.")

    def answer(self, question: str, choices: list[str], view) -> int:
        import base64

        client = self._anthropic.Anthropic()
        content: list[dict] = []
        if isinstance(view, str):
            content.append({"type": "text", "text": f"Context:\n{view}"})
        else:  # list of image paths
            for path in view:
                ext = Path(path).suffix.lstrip(".").lower()
                media = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
                data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": media, "data": data}})
        content.append({"type": "text", "text": self._letter_prompt(question, choices)})
        resp = client.messages.create(
            model=self.model, max_tokens=8,
            messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        m = re.search(r"[A-Za-z]", text)
        return (ord(m.group().upper()) - 65) if m else 0


def get_answerer(name: str):
    """Return an answerer instance. Falls back to the mock (with a notice) if ``claude`` is
    requested but unavailable, so the harness never hard-fails on a missing optional dep."""
    if name == "claude":
        try:
            return ClaudeAnswerer()
        except Exception as exc:  # ImportError or missing key
            print(f"note: claude answerer unavailable ({exc}); falling back to mock",
                  file=sys.stderr)
    return MockAnswerer()


def _load_qa(qa_path) -> list[dict]:
    items = []
    for line in Path(qa_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _render_view(si, view: str) -> str:
    if view == "transcript":
        from screex.transcript import format_transcript
        return format_transcript(si)
    return json.dumps(si.compact_dict())


def _is_correct(pick: int, choices: list[str], answer: str) -> bool:
    if not 0 <= pick < len(choices):
        return False
    answer = str(answer).strip()
    if len(answer) == 1 and answer.isalpha():
        return chr(65 + pick).upper() == answer.upper()
    chosen = _strip_choice_label(choices[pick]).strip()
    return chosen == answer or answer in choices[pick]


def _materialize_frames(recording, n: int) -> list[str]:
    """Write ``n`` uniformly-spaced frames to temp PNGs and return their paths (for the
    image-capable answerer). Returns [] if the video can't be read."""
    import tempfile

    import cv2

    from screex.core import source

    frames = list(source.iter_frames(str(recording), sample_fps=0))
    if not frames or n <= 0:
        return []
    step = max(1, len(frames) // n)
    picks = frames[::step][:n]
    out = []
    tmp = Path(tempfile.mkdtemp(prefix="screex_eval_"))
    for i, (_idx, _t, bgr) in enumerate(picks):
        p = tmp / f"frame_{i:03d}.png"
        cv2.imwrite(str(p), bgr)
        out.append(str(p))
    return out


def score_accuracy(qa_path, fps, frames, tokens_per_image, view, answerer,
                   clips_dir=None, audio=True):
    """Score the index arm vs the uniform-N frames arm on a bucketed MCQ set.
    Returns ``{"buckets": {type: {...}}, "skipped": [...], "answerer": str}``."""
    from screex.cli import index as build_index
    from screex.core import source
    from screex.core.index import ScreenIndex

    qa = _load_qa(qa_path)
    base = Path(clips_dir) if clips_dir else Path(qa_path).resolve().parent
    cache: dict[str, tuple | None] = {}
    buckets: dict[str, dict] = {}
    skipped: list[dict] = []

    for item in qa:
        clip = item["clip"]
        qtype = item.get("type", "other")
        if clip not in cache:
            clip_path = base / clip
            try:
                ip = build_index(str(clip_path), fps=fps, quiet=True, audio=audio)
                si = ScreenIndex.load(ip)
                duration = source.video_info(str(clip_path))["duration"]
                total = max(len(si.states), round(duration * fps))
                cache[clip] = (si, _render_view(si, view), total)
            except Exception as exc:  # missing/corrupt clip or unbuildable index
                skipped.append({"clip": clip, "reason": str(exc)})
                cache[clip] = None
        cached = cache[clip]
        if cached is None:
            continue
        si, view_text, total = cached
        choices, answer = item["choices"], item["answer"]

        idx_correct = _is_correct(answerer.answer(item["question"], choices, view_text),
                                  choices, answer)
        idx_tokens = len(view_text) // 4

        n_frames = min(frames, total)
        frame_view = (_materialize_frames(base / clip, n_frames)
                      if answerer.needs_images else [])
        fr_correct = _is_correct(answerer.answer(item["question"], choices, frame_view),
                                 choices, answer)
        fr_tokens = n_frames * tokens_per_image

        b = buckets.setdefault(qtype, {"n": 0, "idx_ok": 0, "fr_ok": 0,
                                       "idx_tok": 0, "fr_tok": 0})
        b["n"] += 1
        b["idx_ok"] += int(idx_correct)
        b["fr_ok"] += int(fr_correct)
        b["idx_tok"] += idx_tokens
        b["fr_tok"] += fr_tokens

    return {"buckets": buckets, "skipped": skipped,
            "answerer": type(answerer).__name__}


def _accuracy_table(result: dict) -> str:
    buckets = result["buckets"]
    rows = ["| bucket | n | index_acc | frames_acc | index_tokens | frames_tokens |",
            "|--------|---|-----------|------------|--------------|---------------|"]
    tot = {"n": 0, "idx_ok": 0, "fr_ok": 0, "idx_tok": 0, "fr_tok": 0}
    order = [t for t in QUESTION_TYPES if t in buckets] + \
            [t for t in buckets if t not in QUESTION_TYPES]
    for t in order:
        b = buckets[t]
        for k in tot:
            tot[k] += b[k]
        rows.append(f"| {t} | {b['n']} | {b['idx_ok'] / b['n']:.0%} | "
                    f"{b['fr_ok'] / b['n']:.0%} | {b['idx_tok']} | {b['fr_tok']} |")
    if tot["n"]:
        rows.append(f"| **TOTAL** | {tot['n']} | {tot['idx_ok'] / tot['n']:.0%} | "
                    f"{tot['fr_ok'] / tot['n']:.0%} | {tot['idx_tok']} | {tot['fr_tok']} |")
    return "\n".join(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Screex eval (cost + accuracy)")
    p.add_argument("recording", nargs="?",
                   help="recording to cost-eval; omit in accuracy mode (--qa)")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--escalate", type=int, default=3,
                   help="number of keyframes the agent escalates to as images (cost mode)")
    p.add_argument("--tokens-per-image", type=int, default=1500,
                   help="flat estimated input tokens per image read")
    p.add_argument("--out", default=None)
    p.add_argument("--from-index", default=None,
                   help="reuse an existing index.json instead of rebuilding it (cost mode)")
    p.add_argument("--no-audio", action="store_true",
                   help="skip speech-to-text narration when building an index")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    # Accuracy mode
    p.add_argument("--qa", default=None,
                   help="JSONL of MCQs ({clip,question,choices,answer,type}); enables accuracy mode")
    p.add_argument("--clips-dir", default=None,
                   help="directory holding the clips named in --qa (default: the qa file's dir)")
    p.add_argument("--frames", type=int, default=8,
                   help="uniform frames for the raw-frames arm (accuracy mode)")
    p.add_argument("--view", choices=["compact", "transcript"], default="compact",
                   help="index view shown to the answerer (accuracy mode)")
    p.add_argument("--answerer", choices=["mock", "claude"], default="mock",
                   help="who answers the MCQs; mock is deterministic/offline (accuracy mode)")
    args = p.parse_args(argv)

    if args.qa:
        answerer = get_answerer(args.answerer)
        result = score_accuracy(
            args.qa, args.fps, args.frames, args.tokens_per_image, args.view, answerer,
            clips_dir=args.clips_dir, audio=not args.no_audio)
        if args.json:
            print(json.dumps(result, indent=2))
            return
        print(_accuracy_table(result))
        print(f"\nanswerer: {result['answerer']}"
              + ("  (mock = pipeline check, not a real accuracy signal)"
                 if result["answerer"] == "MockAnswerer" else ""))
        for s in result["skipped"]:
            print(f"skipped {s['clip']}: {s['reason']}", file=sys.stderr)
        return

    if not args.recording:
        p.error("a recording is required in cost mode (or pass --qa for accuracy mode)")

    r = estimate(
        args.recording,
        args.fps,
        args.escalate,
        args.tokens_per_image,
        args.out,
        from_index=args.from_index,
        audio=not args.no_audio,
    )
    if args.json:
        print(json.dumps(r, indent=2))
        return

    print("| metric | value |")
    print("|--------|-------|")
    for k, v in r.items():
        print(f"| {k} | {v} |")
    if r["cost_ratio"]:
        print(f"\nScreex uses ~{r['cost_ratio'] * 100:.1f}% of the baseline token cost "
              f"(~{1 / r['cost_ratio']:.1f}x cheaper) at the same recording.")


if __name__ == "__main__":
    main()
