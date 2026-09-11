# Native causal rolling-syllable translation

This cumulative patch adds an opt-in Pass-2 translation mode that moves timing intent into the initial isiZulu wording decision instead of relying primarily on post-TTS repair.

## Causal rolling context

`native-translate --rolling-syllable` processes bounded ordered batches. For each current segment the model is instructed to use only:

- the two most recent locked/approved isiZulu predecessor segments (`n-2`, `n-1`),
- the current English source (`n`), and
- the immediate next English source as read-only lookahead (`n+1`).

Within one provider request the model processes the requested batch sequentially and treats each newly generated isiZulu segment as locked context for the next segment. Across provider calls the final two accepted outputs are carried forward. Previous output is never rewritten by a later block.

The default maximum is 8 segments per Grok call. This preserves rolling dependency without requiring one network call per source segment.

## Duration-derived isiZulu syllable budget

The target is **not the English syllable count**. Mathula derives an isiZulu pronunciation budget from the source mouth window and a simple affine Azure duration proxy:

```
target_raw_ms = source_ms * 1.06
content_ms = max(0, target_raw_ms - azure_utterance_overhead_ms)
target_syllables = round(content_ms / 1000 * content_syllables_per_second)
```

Defaults:

- preferred raw overflow: +6%
- Azure fixed utterance overhead: 900 ms
- isiZulu content rate: 5.3 syllables/s
- preferred syllable range: +/-12%

The fixed overhead matters especially for short turns. If even one modeled syllable cannot fit after the utterance overhead, the segment is marked `budget_impossible`; Grok is told to produce the shortest semantically faithful natural isiZulu rather than delete meaning to hit an impossible count.

The syllable counter is deterministic and deliberately approximate: vowel nuclei for ordinary words plus explicit duration proxies for numeric and acronym tokens. It is a planning signal only. Azure's measured waveform duration remains authoritative in `native-dub`.

## One bounded syllable refinement

After a rolling batch is returned, Mathula counts the resulting isiZulu syllables itself. Only severe deterministic range misses receive one refinement call. The refiner is given the exact measured count and target range. In-range items are immutable. A revised candidate is not accepted if it moves farther away from its deterministic range than the first candidate.

This is still pre-TTS wording planning. Existing measured Azure hard-sync and batched timing repair remain the final closed loop.

## CLI

Run Pass 2 only, reusing the existing Pass-1 restored transcript:

```powershell
python -m mathula_tv.cli native-translate "$JOB_ID" `
  --pass 2 `
  --rolling-syllable `
  --rolling-syllable-batch-size 8 `
  --force `
  --live-operation
```

Optional calibration controls:

- `--rolling-syllables-per-second` (default `5.3`)
- `--rolling-azure-overhead-ms` (default `900`)
- `--rolling-syllable-tolerance-percent` (default `12`)

Then run `native-dub` normally. Because Pass 2 wording changes, use `--force` on the first dub run so deterministic Azure TTS phrase artifacts are regenerated for the new text.

## Compatibility

The existing Pass-2 translator remains the default. `--rolling-syllable` is opt-in. The rolling mode writes the same `native_dub/pass2_natural_translation.json` location with a distinct schema/prompt/input hash, so a legacy checkpoint cannot be silently reused as a rolling checkpoint.
