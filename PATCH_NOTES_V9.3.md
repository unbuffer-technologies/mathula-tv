# Mathula TV v9.3 — Concise translation CLI output

## Behavior

`mathula-tv translate` now prints one concise success line to stdout:

    Translation complete for job JOB_ID.

Batch and provider progress remains on stderr. The full installed multivariant
translation is still written to the normal job artifacts and is not removed.

Use `--json-output` only when a calling script explicitly needs the full JSON.
Combine `--json-output --no-progress` for machine-readable JSON-only output.

## Validation

- `python -m py_compile src/mathula_tv/cli.py`
- 24 focused translation tests passed.
