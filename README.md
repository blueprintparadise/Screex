# Screex

Screen-recording understanding for Claude. Screex turns a screencast into a queryable
**index** of UI states — each with the on-screen text (OCR), what text changed since the
previous state, a thumbnail, and a full-resolution keyframe — so Claude can produce an
action transcript, answer questions, or generate a how-to / bug report from a recording.

Training-free, model-agnostic, and `pip install`-only (OCR via `rapidocr-onnxruntime`, no
system binaries).

## Install
```
pip install -r requirements.txt
```

## Use (standalone)
```
python -m screex.cli index path/to/recording.mp4 --fps 2
```
Produces `path/to/recording.screex/index.json` + `frames/` (keyframes + thumbnails).

Options: `--fps`, `--change-threshold` (0..1; lower = more states), `--thumb-width`, `--out`.

## Use (as a Claude skill)
`SKILL.md` drives Claude through: build index → read `index.json` → produce a transcript,
answer questions, or generate a how-to / bug report.

## Architecture
`screex/core/`: `source` (decode/sample), `segment` (group frames into settled UI states),
`ocr` (RapidOCR text + text-diff), `index` (ScreenState/ScreenIndex schema). The `index` CLI
wires them into `index.json`.
