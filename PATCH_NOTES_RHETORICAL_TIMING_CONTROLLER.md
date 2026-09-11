# Rhetorical timing controller

This cumulative patch replaces generic `compress` / `expand` wording instructions with a measured rhetorical timing controller.

## Compression

The controller chooses a technique ladder from the measured shortening severity:

1. ellipsis
2. brachylogy
3. asyndeton
4. epitome without loss of unique current-turn propositions
5. aposiopesis only when the English source itself is interrupted/trailing/unfinished

Hard and extreme compression require structural recasting rather than synonym substitution. Rolling `n-2`, `n-1`, `n`, `n+1` context remains available; only `n` is mutable and `n+1` is interpretation-only.

## Expansion

An early/undersized isiZulu block is no longer a successful terminal state. Expansion uses, in increasing strength:

1. pleonasm
2. tautology
3. context-grounded restatement
4. periphrasis / circumlocution
5. auxesis only when source intensity already licenses it
6. macrology as a final natural-language expansion tool

Current-block semantic redundancy is preferred. `n-1` may be reused only under an attribution guard: same-speaker previous content can be naturally restated as continuation; different-speaker content can only be referenced/acknowledged/answered/denied/confirmed when the current source licenses that move.

After the configured normal timing-repair rounds, residual early phrases receive up to two dedicated rhetorical expansion rescue passes. If they remain materially early, Mathula fails visibly rather than accepting dead air as success.

## Unchanged hard constraints

- Azure TTS remains at `rate_percent=0`.
- Final speed-up remains bounded by the configured natural speed limit.
- Speech is never slowed.
- Protected source-gap overflow remains bounded and never shifts a later block.
- Video is never retimed.
- Names/identities, numbers/dates, negation, uncertainty, attribution, references, causal/temporal logic, legal/evidential qualification, question/answer intent, and unique current-turn propositions remain protected.
- `n+1` is read-only lookahead and can never supply facts to `n`.

## Validation performed in this environment

- Python compilation of changed source files.
- Selective AST execution of the real rhetorical planner functions.
- Severe expansion strategy / syllable-budget simulation.
- Severe compression strategy / syllable-budget simulation.
- Aposiopesis complete-vs-unfinished source guard.
- Cross-speaker previous-context attribution guard.
- Static check that the former "leave remainder silent" success path is absent and bounded rhetorical expansion rescue is present.

A complete production repository / persisted job / Azure TTS environment is not available in this container, so a full end-to-end production replay was not claimed.
