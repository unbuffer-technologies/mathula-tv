# Temporal-mask residual candidate ID fix

Fixes a live residual-stage failure where Grok returned three valid fresh candidate wordings labeled `d`, `e`, `f` and Mathula rejected them because the temporal-mask controller expected positional labels `a`, `b`, `c`.

## Changes

- Residual prompt now explicitly says candidate IDs are positional wire labels and must restart at `a` every residual round.
- Temporal-mask JSON schema constrains `candidate_id` to the exact requested label set (`a`, `b`, `c`, ...).
- Response normalization remains robust to an already-returned alternate unique label set such as `d/e/f`: when cardinality is exact and there is no overlap with the canonical label set, wordings are canonicalized by response position to `a/b/c`.
- Canonical responses such as reordered `c/a/b` continue to be reordered by identity.
- Duplicate labels, mixed canonical/unknown labels, missing groups, unknown groups, wrong cardinality, and empty text remain hard failures.

## Live failure covered

```text
Temporal mask native_phrase_0001 candidate identity mismatch:
missing=['a', 'b', 'c'], unknown=['d', 'e', 'f']
```

The `d/e/f` response is now accepted as three new residual wordings and normalized to `a/b/c` for measurement and selection.
