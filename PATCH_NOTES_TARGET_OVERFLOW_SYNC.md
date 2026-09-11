# Native target-overflow hard-sync v3.1 — asymmetric mouth-close tolerance

## Fix

A zero-length source gap previously reduced the configured mouth-close tolerance from
`±40 ms` to `±0 ms`. That was correct only for **late** speech: a late finish can collide
with the next source block. It was incorrect for **early** speech: finishing 27 ms early
cannot create overlap and should remain acceptable inside the configured 40 ms target.

The hard-sync bounds are now asymmetric:

- early tolerance = configured mouth-close tolerance (default `40 ms`)
- late tolerance = `min(configured tolerance, free room before next source onset)`
- therefore a zero-gap boundary uses `-40/+0 ms`, not `±0 ms`

This preserves the no-gap-borrowing/no-overlap contract while accepting harmless small
early landings.

## FFmpeg atempo refinement

The adaptive speed fitter now refines in both directions when FFmpeg `atempo` quantization
misses the target:

- late output -> slightly more acceleration
- early output -> slightly less acceleration
- acceleration never falls below `1.0x`; speech is still never slowed
- `+12%` remains the default natural speed ceiling

## Timing contract retained

- Azure TTS remains `rate_percent=0`.
- Default raw target remains source mouth-window `+6%`.
- No source-gap borrowing.
- No later-block shifting.
- No video retiming.
- Canonical Pass-2 translation remains immutable.

## Validation performed for this delta

- `native_dub.py` and `cli.py` pass Python bytecode compilation.
- Isolated zero-gap regression: `-27 ms` is accepted with `early=40 ms`, `late=0 ms`.
- Real FFmpeg `atempo` regression on a 1070 ms synthetic WAV targeting 1000 ms with
  `early=40 ms`, `late=0 ms` converged to `997 ms` (`-3 ms`) and passed sync validation.
- The full Mathula pytest suite was not runnable from this overlay alone because the
  overlay intentionally does not contain every imported project module.

## v3.2 unresolved-early containment (2026-08-21)

- A timing-repair `expand` recast that still finishes earlier than the configured mouth-close tolerance is no longer fatal.
- Mathula keeps the best measured candidate, preserves natural speed, leaves the unused remainder of the source mouth window silent, and continues rendering.
- A red console warning records the unresolved early mouth-close miss.
- `unresolved_early_mouth_close_ms` is written into the measured group artifact for audit/QA.
- Unresolved *late* speech remains fatal when it cannot fit inside the protected late tolerance at the configured maximum natural speed; this patch does not permit overlap, gap borrowing, speech slowing, or later-block shifting.

## Console timing-failure diagnostics (2026-08-21)

Every hard-sync phrase that falls outside the mouth-close contract now prints a red diagnostic containing:

- phrase/group id, speaker and constituent segment ids;
- restored English source text;
- current isiZulu spoken candidate (including timing recasts);
- source mouth window, measured TTS duration and signed mouth-close error;
- preferred raw target and asymmetric early/late tolerance;
- source end, next protected onset and free gap;
- repair direction, exact required speed and configured maximum natural speed.

Diagnostics are emitted for the initial measured failure, for repaired candidates that still miss sync, and for unresolved contained fallbacks.
