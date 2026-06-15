<p align="center">
  <img src="docs/assets/logo.jpg" alt="Screex — screen recordings to LLM-ready" width="320">
</p>

# Screex

**Screen-recording understanding for agents.** Screex turns a screencast into a queryable
**index** of UI states — each with the on-screen text (OCR), what text changed since the
previous state, a thumbnail, and a full-resolution keyframe — so an LLM/agent can produce an
action transcript, answer questions, or generate a how-to guide / bug report from a recording.

- **Training-free & model-agnostic** — no fine-tuned UI model; any LLM can read the index.
- **`pip install`-only** — OCR via [`rapidocr-onnxruntime`](https://pypi.org/project/rapidocr-onnxruntime/), no system binaries.
- **Cheap by design** — the on-screen text is plain text (nearly free to read); full-res
  keyframes are escalated to only when the text is insufficient.

---

## Install

### From PyPI
```bash
pip install screex
```

### From source
```bash
git clone https://github.com/blueprintparadise/Screex.git
cd Screex
pip install -e .          # add ".[test]" to also install pytest
```

Both give you a `screex` command (entry point `screex.cli:main`). Requires Python ≥ 3.9.
First run downloads the small RapidOCR ONNX models automatically.

---

## Quickstart (CLI)

```bash
# Build the index for a screen recording
screex index path/to/recording.mp4 --fps 2
#   (or, without installing the package:)
python -m screex.cli index path/to/recording.mp4 --fps 2
```

This writes:
```
path/to/recording.screex/
  index.json            # the ScreenIndex (ordered UI states)
  frames/00000.png      # full-res keyframe per state
  frames/00000_thumb.png# thumbnail per state
  ...
```

### `index` options
| Flag | Default | Meaning |
|------|---------|---------|
| `--fps` | `2` | frames sampled per second (raise for fast-moving recordings) |
| `--change-threshold` | `0.04` | mean frame-to-frame intensity change (0–1) that starts a new UI state; also fires on cumulative drift from the state's anchor frame (catches slow scrolls/fades). Lower = more states, higher = fewer |
| `--text-threshold` | `0.80` | **(default text mode)** start a new state when on-screen text similarity vs the current state drops below this (0–1) |
| `--motion-epsilon` | `0.003` | skip OCR on frames essentially identical to the previous one (performance only) |
| `--fast` | off | motion-only segmentation (no per-frame OCR) — faster, but misses subtle local changes |
| `--dedupe-threshold` | `0.95` | merge consecutive states whose on-screen text is at least this similar (0–1); set `>1` to disable |
| `--thumb-width` | `320` | thumbnail width in px |
| `--keyframe-format` | `png` | `png` (lossless) or `jpg` (much smaller) for keyframes/thumbnails |
| `--keyframe-quality` | `90` | JPEG quality (only used with `jpg`) |
| `--max-frames` | _none_ | cap sampled frames (guardrail for long/high-res recordings) |
| `--lang` | _auto_ | OCR language hint |
| `--out` | `<recording>.screex` | output directory |
| `-q, --quiet` | off | suppress progress output (place before the subcommand) |

### Transcript (no LLM needed)

Turn a recording straight into a timestamped markdown step list:

```bash
screex transcript path/to/recording.mp4 -o steps.md    # omit -o to print to stdout
screex transcript path/to/recording.mp4 --from-index path/to/recording.screex/index.json
```

By default `index`/`transcript` segment by **on-screen text change**, so a dialog or a status
line appearing becomes its own step. Use `--fast` for motion-only segmentation on simple clips.

### What `index.json` contains
A `schema_version`, the source `video`/`duration`/`sampled_fps`, and an ordered list of
`states`, each with:
`t_start` / `t_end`, `ocr_text` (on-screen text lines), `text_added` / `text_removed`
(text that appeared/disappeared vs the previous state — the strongest signal of what the user
did), and `thumbnail` / `keyframe` paths.

---

## Use as a Claude skill

Screex ships a `SKILL.md` that teaches Claude to build the index and turn it into one of three
views: an **action transcript**, **Q&A** over the recording, or a **how-to / bug report**.

1. **Install the package** so `python -m screex.cli` is available in the environment Claude
   uses (`pip install -e .`).
2. **Install the skill** — the package bundles `SKILL.md`, so one command installs it where
   Claude Code discovers skills:
   ```bash
   screex skill --install                                          # ~/.claude/skills/screex/
   screex skill --install --dir <project>/.claude/skills/screex    # per-project
   screex skill --path                                             # just print the target path
   ```
3. **Use it** — in Claude Code, just ask in natural language, e.g.:
   - *"Use screex to turn `~/Downloads/bug-repro.mp4` into a bug report."*
   - *"What steps does this screen recording show?"*
   - *"From this demo, write a how-to doc."*

   Claude runs `screex index`, reads `index.json`, skims the on-screen text across states, and
   escalates to a full-res keyframe only when the text isn't enough — then produces the
   transcript / answer / document.

> The skill is model-agnostic: the same `index.json` can be read by any LLM/agent, not only
> Claude.

---

## How it works

```
recording → sample frames → segment into UI states → per state: OCR text + text-diff
          → write thumbnail + full-res keyframe → index.json
                                                      ↓
            views (agent-driven): transcript · Q&A · how-to / bug report
```

`screex/core/`:
- `source` — decode & sample frames (OpenCV)
- `segment` — group frames into settled UI states by visual change
- `ocr` — RapidOCR text extraction + text-diff between states
- `index` — the `ScreenState` / `ScreenIndex` schema (JSON)

`screex/cli.py` wires them into the `screex index` command.

---

## Development

```bash
pip install -e ".[test]"
python -m pytest -q
```

---

## License

[MIT](LICENSE) © 2026 Rushikesh Hiray
