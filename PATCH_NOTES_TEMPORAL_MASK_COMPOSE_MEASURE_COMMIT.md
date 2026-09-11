# Temporal Mask Compose → Measure → Commit

Version: `mathula-native-dub-v3.7-temporal-mask-compose-measure-commit`

This patch adds a new **opt-in** `native-translate --temporal-mask` Pass-2 architecture. It does not replace the standard or rolling-syllable modes.

## Core model

Pass 2 no longer has to translate a source block first and repair timing later. It builds the exact phrase groups that `native-dub` will render, then treats each phrase as a temporal mask.

For mask `n`, Grok receives:

- English `n-2`, `n-1`, `n`, `n+1` context;
- up to two **committed** prior isiZulu phrase groups;
- current speaker/discourse relationship;
- source start/end, next protected onset, bounded protected gap, raw-duration envelope, and syllable planning range;
- protected high-risk source literals.

Only `n` is mutable. `n+1` is read-only comprehension context.

## Candidate composition

Each mask requests three natural isiZulu candidates by default. Candidate `a` is the primary/provisional wording. The model may use:

- tight mask: ellipsis, brachylogy, asyndeton, compact contextual reference, context-grounded epitome;
- roomy mask: pleonasm, tautology, contextual restatement, periphrasis, source-supported auxesis, macrology as last resort.

The objective is one faithful natural isiZulu utterance that fills the acoustic mask, not a literal translation plus later rescue.

## Azure is authoritative

Every candidate is synthesized with the resolved speaker's Azure isiZulu voice at `rate_percent=0`. Measurements run concurrently (`--temporal-mask-tts-workers`, default 6).

A candidate is committable only when:

- it is not materially shorter than the source mouth window (within configured early tolerance);
- it can end inside the protected late envelope at no more than the native +12% pitch-preserving speed bound;
- deterministic high-risk numeric/acronym guards pass.

Syllables remain a planning proxy only.

## Measured residuals

If none of the initial candidates can fill the mask, Grok receives the actual Azure durations and generates a new candidate set. Default residual budget is two rounds.

If no candidate fits after the bounded residual budget, Pass 2 fails visibly. It does not commit dead air, exceed the natural speed bound, or move the next block.

## Causal commit + speculative batching

To avoid one Grok call per phrase, up to six masks are composed speculatively in one request. Candidate `a` of an earlier mask is provisional context for the next mask in that request.

Mathula commits in chronological order only after Azure measurement:

- if candidate `a` is accepted, the speculative suffix remains valid;
- if candidate `b`/`c` wins or a residual rewrite wins, the current mask is committed and the dependent speculative suffix is discarded/regenerated using the actually accepted isiZulu.

This prevents future context from depending on rejected wording while retaining most batching latency savings.

## Exact render grouping

Temporal-mask Pass 2 builds the same acoustic phrase grouping used by `native-dub`, including same-speaker overlap grouping. The artifact persists `temporal_mask_groups`, and the renderer reuses those exact groups rather than regrouping segment-level translations.

This removes the earlier mismatch where Pass 2 planned 76 segments but `native-dub` later synthesized 57 phrase groups.

## CLI

```powershell
python -m mathula_tv.cli native-translate "$JOB_ID" `
  --pass 2 `
  --temporal-mask `
  --temporal-mask-batch-size 6 `
  --temporal-mask-candidates 3 `
  --temporal-mask-residual-rounds 2 `
  --temporal-mask-tts-workers 6 `
  --force `
  --live-operation
```

`--temporal-mask` and `--rolling-syllable` are mutually exclusive.

Temporal-mask Pass 2 requires the normal Azure Speech configuration because it performs real TTS candidate measurement during translation.
