# Native batch adaptive timing repair v3.3

## What changed

- Native dub now synthesizes/measures the complete canonical phrase set before any timing-repair AI calls.
- All measured failures are classified from timing geometry, not phrase identity:
  - `soft_compress`, `hard_compress`, `extreme_compress`
  - `soft_expand`, `hard_expand`, `extreme_expand`
- Mixed failure classes are sent together in bounded Grok batches (default 12, configurable 1-24).
- Round 2 contains residual failures only and recalculates required movement from the newly measured Azure TTS audio.
- If the prior accepted/rejected recast achieved less than 50% of the requested duration movement, the residual item escalates one severity tier.
- Hard/extreme compression prompts explicitly require structural isiZulu recasting rather than synonym-only edits.
- Each request carries exact measured milliseconds, required percentage movement, maximum/minimum usable raw duration, and deterministic high-risk name/date/number surfaces.
- Existing English + isiZulu + timing diagnostics remain active for initial misses, residual misses, and unresolved failures.
- Unresolved early finishes remain non-fatal: keep the closest measured candidate and leave the source window remainder silent.
- Unresolved late finishes remain fatal: no silence borrowing, later-block shift, >configured speed, or video retiming.

## Example geometry

For the observed 6.720 s source window / 10.598 s raw candidate at +12% maximum speed and +40 ms late tolerance:

- maximum rescuable raw duration: 7.571 s
- required removal: 3.027 s / 28.562%
- class: `extreme_compress`

If round 1 only moves 10.598 s -> 9.762 s, round 2 recalculates 2.191 s / 22.444% remaining. Because round 1 achieved under half of the requested movement, it escalates to `extreme_compress` again rather than issuing another timid repair.

## Validation performed here

- `py_compile` / `compileall` pass for patched source and included tests.
- Mixed-batch provider simulator passed with one extreme-compress + one extreme-expand item in round 1.
- Residual-only round simulator passed: only the still-failing compress item entered round 2 and escalated based on measured progress.
- Deterministic protected-literal extraction test passed for November/December/January and 2024/2025.

A complete persisted-job replay cannot be run in this environment because the full production repository, Azure backend, and job state are not present in this overlay.
