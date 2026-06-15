# Screex evaluation

Screex should prove two claims on real screen recordings:

1. It preserves the UI states a person needs to understand what happened.
2. It is cheaper for an agent to read than a naive every-frame-as-image baseline.

The unit tests cover deterministic synthetic clips. This page is for repeatable real-world
checks: browser flows, terminal sessions, forms, modals, scrolling pages, narrated demos,
and low-contrast UI.

## Test Clips

Use 3-5 short clips at first, then grow the table as bugs appear:

| clip | duration | scenario | expected signal |
|------|----------|----------|-----------------|
| browser-login.mp4 | 5-15s | browser login or settings flow | important labels/errors appear in `text_added` |
| terminal-task.mp4 | 5-15s | command output changes | command and result text are preserved |
| scroll-page.mp4 | 5-15s | slow scroll through text | enough states to reconstruct movement |
| narrated-demo.mp4 | 10-30s | spoken walkthrough | `narration` aligns with visible states |
| low-contrast-ui.mp4 | 5-15s | subtle status/error text | text mode catches changes missed by `--fast` |

Do not commit private recordings. Store local fixtures outside the repo or use synthetic
fixtures in `tests/` when a regression can be made shareable.

## Accuracy

For each clip, run:

```bash
screex index path/to/recording.mp4 --fps 2
screex transcript path/to/recording.mp4 -o steps.md
```

Record whether Screex:

1. emits a state for each meaningful UI change,
2. captures the important on-screen text,
3. timestamps the change within about 1 second,
4. includes useful narration when audio is installed, and
5. avoids duplicate states for visually different but text-identical screens.

## Cost

Run two conditions on the same clips and compare estimated input tokens:

- **Screex path:** read on-screen text across states plus a few escalated keyframes.
- **Baseline:** read every sampled frame as an image.

The harness estimates this for any recording:

```bash
python scripts/eval.py path/to/recording.mp4 --fps 2 --escalate 3
```

It prints state count, on-screen-text tokens, escalated-image tokens, the baseline
every-frame-as-image cost, and the cost ratio (`Screex / baseline`). These are coarse
relative estimates, not billing promises.

## Results Log

| date | clip | fps | states | escalated images | screex tokens | baseline tokens | cost ratio | accuracy notes |
|------|------|-----|--------|------------------|---------------|-----------------|------------|----------------|
| TBD | browser-login.mp4 | 2 | TBD | 3 | TBD | TBD | TBD | add notes |

## OCR Speed

OCR dominates indexing. onnxruntime's default threading is slow for the detection and
recognition models; `--ocr-threads 2` is the current default. Measured on a 640x360 clip:

| intra-op threads | ms/frame | speedup |
|------------------|----------|---------|
| default | ~1996 | 1x |
| 1 | ~796 | 2.5x |
| 2 (default) | ~519 | 3.85x |
| 4 | ~529 | 3.8x |
