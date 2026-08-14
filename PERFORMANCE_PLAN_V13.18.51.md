# Performance-plan v13.18.51

This cumulative release makes Claude semantic-review responses recoverable at
the provider boundary. It includes every v13.18.50 measured-region controller
change.

## Failure corrected

Claude can return a logically contradictory review such as:

```json
{
  "accepted": true,
  "reason_codes": ["minor_wording_concern"]
}
```

Earlier code treated that contradiction as invalid structured output and
terminated the entire dub. The new application normalizer interprets it
conservatively as a rejected candidate, checkpoints the reason, and lets the
bounded measured controller continue.

## Defensive normalization

The review boundary now canonicalizes:

- non-empty rejection reasons with `accepted: true`;
- incomplete required-fact confirmations;
- incomplete protected-entity confirmations;
- false or malformed semantic gate booleans;
- low, malformed, or string-encoded scores;
- scalar fact/entity values that should be arrays;
- a rejection that omitted its reason.

Only logically safe conversions are made. Contradictions are never converted
into acceptance. The renderer independently refuses any accepted review that
still contains a rejection reason.

The review prompt also states the contract explicitly: `reason_codes` must be
empty when `accepted` is true.

## Resume behavior

The strategy version is advanced, so a stopped v13.18.50 job replays its paid
timing-fit candidates and receives a fresh bounded performance budget. The
failed provider response itself was never selected.

## Verification

The cumulative source passes 77 regressions, including the exact contradictory
review reported by job `d724a4c947404afeae1d748363b41827` and an adversarial
matrix covering every acceptance gate above.
