# Native batch timing repair — safety-first candidate selection

## Problem fixed

The batch repair controller could synthesize a recast that moved a phrase from an
unrescuable late state into a safe early state, but `_timing_target_score()` could
still rank the original late candidate as "better" because the early recast was
farther from the preferred +6% raw-duration target.

That produced a destructive loop:

1. canonical phrase is too long and cannot fit at the +12% natural speed bound;
2. Grok compresses it enough that it now finishes early;
3. selector rejects that safe early recast and keeps the original unsafe late audio;
4. round 2 sees the original late candidate and asks Grok to compress again;
5. final validation reports the original late candidate as fatal.

## New selection hierarchy

Measured candidates are now ordered by safety class first:

1. `fit/speed-fit` — already fits or can fit within the configured natural speed cap;
2. `early-contained` — ends early; can be contained by leaving the remainder silent;
3. `late-unresolved` — remains late even at maximum natural speed and is unsafe.

Within the same class, Mathula prefers the candidate closest to the intended timing target.

This means a safe early recast always beats an unrescuable late candidate, but a closer
early candidate still beats a more extreme early candidate.

## Production-log regressions

The regression tests include the observed geometry for:

- `native_phrase_0001`: 49.250 s canonical late -> 29.125 s round-1 early;
- `native_phrase_0002`: 42.969 s canonical late -> 34.493 s round-1 early with zero late tolerance.

Both now select the safe early recast. For phrase 0001, a later 23.886 s over-compressed
candidate does not replace the closer 29.125 s early candidate.

No late-overlap rule, maximum speed bound, source-gap policy, or semantic identity validation
is relaxed by this patch.
