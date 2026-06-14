# Squint — Design Spec

**Date:** 2026-06-14
**Status:** Approved (pending spec review)
**Working dir:** `C:\Users\RushiHiray\Pictures\asciivideo`

## 1. Vision

Squint gives a language model **cheap, continuous, private vision** by treating ASCII
as a *perception substrate*. A video frame becomes compact text a model can skim almost
for free; the model spends real perception budget (reading a full-resolution image) only
on the handful of moments that earn it.

This is the "north star" use case distilled from a broader platform idea: turning the
ASCILINE ASCII-video engine (https://github.com/YusufB5/ASCILINE) into a **core engine +
plugins**. Squint is v1 — shipped as a **Claude skill** so we can test the core hypothesis
locally, at zero external cost, with Claude itself as the model.

### The hypothesis v1 proves

Claude can understand "what happened in this video" by reading
**motion-selected ASCII frames + a few escalated full-res images**, at a tiny fraction of
the cost of reading every frame as an image — and accurately enough to be useful.

### Honest framing (where the value comes from)

- **Motion-triage / frame-selection is the biggest win.** A 10-second clip is ~300 frames
  but maybe ~5 events. Selecting events is what slashes cost.
- **ASCII is the cheap skim layer.** It lets Claude scan a strip of candidate frames as
  *text* to pinpoint the moment.
- **Full-res image is the confirm.** ASCII does not replace vision — it indexes it. Claude
  escalates to the real PNG only for moments that matter.

## 2. How it works (the workflow Claude follows)

When the user asks Claude to watch/analyze a clip with a rule or question
(*"did anyone approach the door?"*, *"summarize what happens"*, *"tell me when the screen
shows an error"*), the skill drives this loop:

1. **Analyze** — run `squint analyze <video>`; produces a working dir with sampled frames
   (PNG), their ASCII (TXT), and `manifest.json`.
2. **Read the manifest** — Claude gets a motion timeline + a list of event frames instead
   of all frames.
3. **Skim ASCII** — Claude reads the *text* of event frames (cheap) to localize what is
   happening relative to the user's rule.
4. **Squint → lean in** — for any moment that matters, Claude `Read`s the full-resolution
   PNG of that exact frame to confirm fine detail (a face, screen text, an object).
5. **Answer** — Claude reports back in natural language with timestamps.

Steps 2–3 are nearly free; step 4 spends perception budget only on frames that earned it.

## 3. Architecture (the core-engine seed)

Even though v1 ships as a skill, the scripts underneath are organized as the **plugin seams**
of the larger platform design, so this grows into the platform later. Each `core/` module is
a clean function with one job and a defined I/O contract — the same Source / Mapper / Analyzer
/ Sink stages from the platform design, with a single implementation each in v1.

```
squint/
  core/
    source.py     # frames in  → v1: video file  (future: webcam, screen, URL)
    mapper.py     # px → ASCII  → v1: grayscale LUT (+ optional edge mode)
    analyzer.py   # motion/delta change-scoring → selects "event" frames
    manifest.py   # the AsciiFrame index (JSON) everything reads/writes
  cli.py          # `squint analyze <video>` and `squint capture ...`
  SKILL.md        # tells Claude the workflow (the "sink" in v1 is Claude itself)
```

The **manifest** is the canonical `AsciiFrame` stream made concrete on disk — it is what
makes "skim text → escalate to image" trivial for Claude.

### Stage contracts (v1 implementations)

- **Source** (`source.py`): given a video file path + target sample FPS, yield `(idx, t,
  bgr_frame)`. Uses OpenCV (`cv2.VideoCapture`), sampling via timestamp/decimation rather
  than decoding every frame.
- **Mapper** (`mapper.py`): given a BGR/grayscale frame + `cols`, return an ASCII string.
  Grayscale intensity → character via a NumPy lookup table (LUT), rows auto-derived from
  aspect ratio with character-cell correction. Optional `--edge` mode emphasizes structure
  (e.g. gradient/DoG before mapping).
- **Analyzer** (`analyzer.py`): given consecutive frames, compute a per-frame motion
  `score` (normalized frame-difference magnitude), flag `event` frames above a sensitivity
  threshold, and group contiguous high-motion frames into `events`.
- **Manifest** (`manifest.py`): assemble/serialize the JSON index below; provide a loader
  for tests and for the skill.

## 4. Data: the manifest format

```json
{
  "video": "door.mp4", "duration": 12.4, "sampled_fps": 5, "cols": 120,
  "frames": [
    {"idx": 0,  "t": 0.0, "score": 0.00, "event": false, "ascii": "frames/0000.txt", "png": "frames/0000.png"},
    {"idx": 12, "t": 2.4, "score": 0.78, "event": true,  "ascii": "frames/0012.txt", "png": "frames/0012.png"}
  ],
  "events": [ {"t_start": 2.4, "t_end": 3.1, "peak_frame": 12, "peak_score": 0.78} ]
}
```

- `score`: motion magnitude vs the previous sampled frame (0..1, from the analyzer).
- `event`: whether the frame crossed the sensitivity threshold.
- `events`: contiguous high-motion segments grouped for Claude (start/end time, peak frame,
  peak score).
- `ascii` / `png`: relative paths into the working dir.

## 5. CLI surface

- `squint analyze <video> [--fps N] [--cols N] [--sensitivity F] [--edge] [--out DIR]`
  Runs source → analyzer → mapper, writes frames + ASCII + `manifest.json` to the work dir.
- `squint capture --webcam [--seconds N] [--out DIR]`
  Optional helper that records a short live clip into the same pipeline (so the demo can be
  live). Webcam via OpenCV.

## 6. Scope

### In v1
- Input: video file (primary) + optional `squint capture --webcam` helper.
- Mapper: grayscale LUT ASCII, optional `--edge` mode.
- Analyzer: frame-difference motion scoring + event grouping.
- Output: `manifest.json` + on-disk frames (PNG + ASCII TXT).
- `SKILL.md` encoding the workflow in Section 2.

### Out (deferred to later platform stages)
- 24/7 daemon / continuous monitoring / push notifications (the later "Sentinel" CLI).
- External local/cloud models (Claude is the model in v1).
- WebSocket streaming, web component, hardware sinks, MCP server.
- Audio, color ASCII (grayscale is sufficient for triage; color is bandwidth/aesthetic).
- Screen capture (cross-platform capture deferred; webcam only in v1).

### Reuse policy
Reuse ASCILINE's *ideas* (NumPy gray→LUT mapping, delta change-detection) but reimplement
lean. Do **not** import the ASCILINE monolith (it couples decode + map + codec + WebSocket +
FastAPI + audio).

## 7. Testing & success criteria

Sample clips with known events (person walks in, screen shows text, object moves).

- **Accuracy:** Claude correctly identifies the events and their timestamps.
- **Cost:** compare tokens for the Squint path vs a naive "Read every sampled frame as an
  image" baseline. This number is the proof of the thesis.
- **Unit tests:**
  - `analyzer`: synthetic frame sequences with known motion → expected scores/events.
  - `mapper`: deterministic ASCII output for a fixed input frame + cols.
  - `manifest`: schema round-trip (write → load → equality).

## 8. Platform / environment notes

- OS: Windows 11 (win32). Shell: PowerShell.
- Python with `opencv-python` and `numpy`. No FFmpeg needed in v1 (no audio).
- Skill is tested locally inside Claude Code; Claude is the perceiving model.

## 9. Future direction (context, not v1 work)

Squint's `core/` stages are the seed of the unified platform: additional **Source** plugins
(webcam/screen/URL/RTSP), **Sink** plugins (notifier, exporter, MCP server, hardware/serial,
transport), and **Analyzer** plugins (LLM-watch, recorder/indexer) extend the same pipeline.
Each future use case (terminal player, web component, video-over-impossible-networks,
searchable/diffable video, universal display layer) is a recipe over these stages, not a new
project.
