# Batch timing repair: deterministic changed derivation

## Failure fixed

A valid `native_natural_timing_repair_batch` response could be rejected after schema validation with:

`Natural timing repair changed flag does not match returned text`

The model-generated `changed` boolean is bookkeeping and can disagree with its own returned `spoken_text`.

## Production behavior

- The provider wire schema remains unchanged: `changed` is still required for structured-output compatibility.
- Mathula no longer trusts the provider boolean as application state.
- `changed` is derived deterministically by comparing each returned `spoken_text` with the exact current phrase text.
- Provider `changed=false` + genuinely different text is treated as changed and proceeds to TTS measurement.
- Provider `changed=true` + identical text is treated as a no-op and does not waste a TTS call.
- Exact group cardinality/order, segment identity, and non-empty returned text remain strict.
- No timing boundary, semantic preservation, speed bound, or overlap policy is relaxed.
