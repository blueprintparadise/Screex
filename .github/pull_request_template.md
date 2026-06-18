<!-- Thanks for contributing to Screex! Keep PRs focused on one logical change. -->

## What

Briefly describe the change.

## Why

The user value or problem this addresses. Link any related issue (`Closes #NN`).

## How

Key implementation notes. Call out anything touching the `index.json` schema or pipeline.

## Testing

```bash
ruff check screex tests
python -m mypy screex
python -m pytest -q
```

<!-- Paste the results, and describe any manual testing performed. -->

## Risk / backward compatibility

Risk level and whether this changes any existing behaviour or output format.

## Checklist

- [ ] `ruff`, `mypy`, and `pytest` all pass locally
- [ ] Added/updated tests for the change
- [ ] Updated docs (README / docstrings) where relevant
- [ ] Single, focused change with a clear commit history
