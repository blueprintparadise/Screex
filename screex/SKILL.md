---
name: screex
description: Use when the user wants Claude to understand a screen recording / screencast / demo / bug-repro video — e.g. "what are the steps in this recording?", "turn this into a how-to doc", "write a bug report from this repro", "what URL did they open?". Screex builds a queryable index of UI states (with on-screen text) and Claude reads it to produce a transcript, answer questions, or generate docs.
---

# Screex — screen-recording understanding

## When to use
The user points you at a screen recording (a screencast, demo, tutorial, or bug repro) and
wants a step transcript, a how-to doc, a bug report, or answers to questions about it.

## Build the index
Run:
`python -m screex.cli index <recording> --fps 2`
(raise `--fps` for fast-moving recordings; lower `--change-threshold` to split states more
eagerly.) This writes `<recording>.screex/index.json` plus per-state `frames/NNNNN.png`
(full-res keyframe) and `frames/NNNNN_thumb.png` (thumbnail).

## Read the index
`Read` `index.json`. It is an ordered list of UI `states`, each with `t_start`/`t_end`,
`ocr_text` (the on-screen text), `text_added` / `text_removed` (what text appeared or
disappeared vs the previous state — the strongest signal of what the user did), and paths to
a `thumbnail` and full-res `keyframe`. The on-screen text is plain text — reading it across
states is cheap.

## Produce one of three views

- **Action transcript:** walk the states in order; use `text_added`/`text_removed` plus the
  thumbnail to narrate timestamped steps, e.g. "0:04 opened Settings; 0:09 entered an API
  key; 0:14 an 'invalid key' error appeared."
- **Q&A:** answer the user's question by scanning `ocr_text` across states (cheap). `Read`
  the full-res `keyframe` PNG for a state only when the text is insufficient (small icons,
  layout, colour).
- **Doc / bug report:** format the transcript into a how-to guide, or a structured
  reproduction report (steps to reproduce, expected vs actual).

## Cost discipline
The `ocr_text` and `text_*` fields are text and nearly free to read. Escalate to a
`keyframe` image only for the few states where the text doesn't answer the question.

## Caveats
`ocr_text` can contain minor OCR noise (stray glyphs), and a busy recording can produce many
near-duplicate consecutive states — collapse states whose `ocr_text` is essentially identical
when you narrate. Tune `--change-threshold` up to merge states, down to split them.
