#!/usr/bin/env python
"""Runnable cost eval for Screex.

Compares the cost of the Screex path (skim on-screen text across states, escalate a few
keyframes) against the baseline (read every sampled frame as an image). It reports a
cost proxy in estimated input tokens so the README's "cheap by design" claim is
measurable.

Usage:
    python scripts/eval.py path/to/recording.mp4 --fps 2 [--escalate 3]

Token estimates use coarse public-ish rules of thumb and are meant for *relative*
comparison, not billing accuracy:
  - text:  ~1 token per 4 characters
  - image: a flat per-image tile cost (see --tokens-per-image)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/eval.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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


def main(argv=None):
    p = argparse.ArgumentParser(description="Screex cost eval")
    p.add_argument("recording")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--escalate", type=int, default=3,
                   help="number of keyframes the agent escalates to as images")
    p.add_argument("--tokens-per-image", type=int, default=1500,
                   help="flat estimated input tokens per image read")
    p.add_argument("--out", default=None)
    p.add_argument("--from-index", default=None,
                   help="reuse an existing index.json instead of rebuilding it")
    p.add_argument("--no-audio", action="store_true",
                   help="skip speech-to-text narration when building an index")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    args = p.parse_args(argv)

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
