# Temporal Mask Occurrence-Level Semantic Contract

This patch fixes repeated high-risk literals (numbers/acronyms) in temporal-mask composition and residual repair.

## Changes

- High-risk literals now carry exact `required_count` and per-occurrence source-context windows.
- Initial temporal-mask composition receives `high_risk_occurrence_requirements`.
- Residual repair receives both candidate-specific `semantic_guard_requirements` and the authoritative source-side `required_high_risk_occurrences` checklist.
- A repeated literal must preserve the distinct source claim around every occurrence; duplicating a token in one clause does not satisfy two source occurrences.
- Timing-valid semantic repairs are told to reclaim any added duration by compressing non-protected wording elsewhere in the current utterance.
- Residual candidate search prefers fewer semantic deficits before acoustic closeness.
- Numeric parsing no longer absorbs sentence punctuation (`24,` -> `24`) while retaining grouped/decimal values (`3,978`, `3.14`).

## Regression geometry

The production failure with two separate `less than 24 hours` source claims and only one target `24` now yields an occurrence-level requirement with `required_count=2`, `observed_count=1`, `missing_count=1`, and both source clause contexts.
