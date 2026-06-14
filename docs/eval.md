# Squint evaluation

The thesis: Claude can answer "what happened?" accurately by reading motion-selected ASCII
+ a few escalated images, far cheaper than reading every sampled frame as an image.

## Test clips
Use 3–5 short clips (5–15s) with a known ground-truth event:
- a person walking into frame,
- a screen showing a specific word/error,
- an object moving across a static scene.

## Accuracy
For each clip, ask Claude (via the skill) the event question and record whether it:
1. detects the event, and
2. timestamps it within ±1s.

## Cost (the number that matters)
Run two conditions on the same clips and compare total input tokens:
- **Screex path:** read on-screen text across states + a few escalated keyframes.
- **Baseline:** `Read` every sampled frame as an image.

A runnable harness estimates this for any recording:

```bash
python scripts/eval.py path/to/recording.mp4 --fps 2 --escalate 3
```

It prints states, on-screen-text tokens, escalated-image tokens, the baseline
(every-frame-as-image) token cost, and the **cost ratio** (Screex / baseline). The token
numbers are coarse estimates for *relative* comparison, not billing accuracy. The win is a
low ratio at equal or better accuracy. Log results in a table here as you run them.
