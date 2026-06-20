# Changelog

All notable changes to Screex are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0]

### Added
- `screex info <index.json>` — summarize an index (states, duration, event histogram, warnings,
  persistent-UI lines, whether curated keyframes are present).
- `screex search <index.json> <pattern>` — query states by on-screen text, time range
  (`--since`/`--until`), or event type (`--event`).
- Structured transcript export: `screex transcript --format {md,json,srt,vtt}` (markdown remains the
  default).
- Optional **MCP server** (`pip install 'screex[mcp]'`, then `screex mcp`) exposing index build,
  info, search, and transcript as Model Context Protocol tools.
- `py.typed` marker — Screex now ships its type hints (PEP 561).
- `CHANGELOG.md` and a tag-triggered PyPI publish workflow.

## [0.6.0]

### Added
- `--events`: classify each state transition into a typed action event
  (`navigate`/`type`/`click`/`open_dialog`/`error`/`scroll`/`edit`), region-grounded (schema v2).
- `--keyframe-budget N`: score per-state `salience` and surface the N most informative, temporally
  spread **curated keyframes** via `compact_dict(keyframe_budget=N)`.
- Accuracy eval harness (`scripts/eval.py --qa`) reporting accuracy and tokens per question bucket,
  with a pluggable mock/Claude answerer and an optional hybrid (text + curated keyframes) index arm.
- GUI-World → `qa.jsonl` converter and benchmark runbook.

### Fixed
- Windows native crash from a duplicate OpenMP runtime (PyTorch + ONNX Runtime).
- Clearer errors and recoverable diagnostics for corrupt/truncated video.
- `compact_dict` no longer empties the only state's text for single-state indexes.

## [0.5.0]

### Added
- `compact_dict()` — token-efficient, LLM-oriented index view.
- Capture controls, transcript-from-existing-index, and eval-harness index reuse.
