# Temporal-mask semantic residual feedback fix

A temporal-mask candidate can satisfy the acoustic envelope but still be rejected by deterministic semantic guards. Previously the residual Grok request only received `direction`, so a timing-valid candidate (`direction=None`) could be repeatedly rewritten for timing without being told that it had dropped a protected numeric/acronym occurrence.

This patch:

- sends `timing_fit`, `semantic_guard_pass`, `semantic_guard_missing`, and `rejection_reason` for every measured residual candidate;
- instructs residual composition to restore every missing protected literal occurrence first when semantic validation fails;
- explicitly tells Grok that `direction=None` + semantic failure means timing already fits and fidelity must be repaired without moving away from the envelope;
- surfaces semantic guard failures in progress and final error diagnostics;
- tightens number tokenization so trailing sentence punctuation is not absorbed into numeric literals (`24,` is normalized as `24`, while `3,978` and `3.14` remain intact);
- bumps the temporal-mask prompt version to invalidate stale checkpoints.

The live failure that motivated the patch had a 41.475 s isiZulu candidate for a 38.640 s source mask, which is speed-fit inside the +12% bound. It was rejected because the English contains `24` twice and the candidate retained it only once. The residual model now receives `semantic_guard_missing=["number:24 x1"]` directly.
