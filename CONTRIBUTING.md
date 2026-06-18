# Contributing to Screex

Thanks for your interest in improving Screex! This guide covers local setup, the checks CI
runs, and the conventions that make a PR easy to review and merge.

## Development setup

Requires Python ≥ 3.9.

```bash
git clone https://github.com/blueprintparadise/Screex.git
cd Screex
python -m pip install -e ".[dev]"   # ruff + mypy + pytest + mss
```

The first OCR run downloads the small RapidOCR ONNX models automatically.

For the optional features, also install the relevant extra:

```bash
python -m pip install -e ".[audio]"     # faster-whisper narration
python -m pip install -e ".[capture]"   # mss screen capture
```

## The checks CI runs

A PR must pass the same three checks the CI workflow runs (`.github/workflows/ci.yml`), on
Python 3.9–3.12:

```bash
ruff check screex tests        # lint
python -m mypy screex          # type check
python -m pytest -q            # tests
```

Please run all three locally before opening a PR. The tests use small synthetic video fixtures
(see `tests/conftest.py`) and need no external media.

## Windows note (OpenMP runtime conflict)

If you install the **audio** extra, `faster-whisper` pulls in PyTorch and `rapidocr-onnxruntime`
pulls in ONNX Runtime. On Windows these can load two copies of the OpenMP runtime in one process,
which makes `pytest` (and `screex index --audio`) abort with a native fault. If you hit that, set:

```bash
# bash / Git Bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q
```

```powershell
# PowerShell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"; python -m pytest -q
```

## Conventions

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) — e.g. `fix(source):`,
  `feat(eval):`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`, `chore:`. Keep each commit a single
  logical change.
- **Scope:** one focused change per PR. It is much easier to merge a small PR than a large one.
- **Style:** ruff handles linting/imports (line length 100; `E501` ignored). Match the surrounding
  code — docstrings on public functions, lazy `import cv2` inside functions, validate inputs.
- **Types:** keep `mypy screex` clean.
- **Tests:** add a regression test for every bug fix and a test for every new behaviour. Don't lower
  coverage. Keep fixtures synthetic and shareable — **do not commit private recordings**.
- **Backward compatibility:** the `index.json` schema is versioned (`schema_version`). Prefer
  additive, opt-in changes; flag any breaking change and include a migration note.

## Proposing larger changes

For anything beyond a small fix or doc change, please **open an issue first** describing the
problem and a brief design, and link any relevant section of the roadmap
(`docs/superpowers/specs/`). This keeps work aligned and avoids wasted effort.

## Reporting bugs

Open an issue using the bug-report template and include the command you ran, the full output
(stack traces help), your OS and Python version, and whether the `audio`/`capture` extras are
installed.
