# Context-aware comprehension expansion

## What changed

Native timing repair v5 replaces generic under-duration expansion with causal, context-grounded expansion.

For every measured `expand` repair item Mathula now sends Grok:

- the two previous phrase groups with English source and the latest accepted isiZulu;
- the current English + isiZulu phrase (the only mutable scope);
- the immediate next English phrase as read-only lookahead;
- actual Azure measured duration and timing geometry;
- a deterministic extra-syllable target/range derived from the missing milliseconds and the phrase's measured Azure syllable rate.

The model is instructed to improve comprehension and conversational flow rather than add arbitrary filler. Preferred expansion moves are discourse connectors, ellipsis completion, grounded reference clarification, source-licensed restatement, and question/answer flow. Minimal semantic padding is explicitly last resort.

The following remain forbidden: new facts, names/identities, numbers/dates, causal claims, certainty/allegation changes, speaker intentions, moving the next block's fact early, or meaningless repetition solely to consume duration.

## Causal batch behavior

Repair items are processed chronologically. If a previous context group also appears earlier in the same Grok repair batch, the prompt tells Grok to use its newly returned wording as the effective previous context for later items. Across repair rounds, Mathula rebuilds each context window from the latest accepted candidates.

## Timing behavior unchanged

- Azure TTS remains rate 0%.
- Final speed-up remains pitch-preserving and bounded by the configured maximum.
- Protected source-gap overflow remains bounded and cannot shift the next block.
- Safe unresolved early endings remain containable; unsafe unresolved late endings remain strict.
