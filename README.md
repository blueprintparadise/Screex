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

### From source (recommended while pre-release)
```bash
git clone <your-repo-url> screex
cd screex
pip install -e .
```
`pip install -e .` installs the package and a `screex` command (entry point
`screex.cli:main`). To run tests too: `pip install -e ".[test]"`.

### Just the dependencies
```bash
pip install -r requirements.txt
```
Then call the module directly: `python -m screex.cli ...`.

Requires Python ≥ 3.9. First run downloads the small RapidOCR ONNX models automatically.

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
| `--change-threshold` | `0.04` | visual-change fraction (0–1) that starts a new UI state — lower = more states, higher = fewer |
| `--thumb-width` | `320` | thumbnail width in px |
| `--out` | `<recording>.screex` | output directory |

### What `index.json` contains
An ordered list of `states`, each with:
`t_start` / `t_end`, `ocr_text` (on-screen text lines), `text_added` / `text_removed`
(text that appeared/disappeared vs the previous state — the strongest signal of what the user
did), and `thumbnail` / `keyframe` paths.

---

## Use as a Claude skill

Screex ships a `SKILL.md` that teaches Claude to build the index and turn it into one of three
views: an **action transcript**, **Q&A** over the recording, or a **how-to / bug report**.

1. **Install the package** so `python -m screex.cli` is available in the environment Claude
   uses (`pip install -e .`).
2. **Install the skill** by placing `SKILL.md` where Claude Code discovers skills:
   - **User-wide:** `~/.claude/skills/screex/SKILL.md`
   - **Per project:** `<project>/.claude/skills/screex/SKILL.md`
   ```bash
   mkdir -p ~/.claude/skills/screex
   cp SKILL.md ~/.claude/skills/screex/SKILL.md
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
