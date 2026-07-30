# Mathula TV v13.17.2 — Claude Request and Translation Resilience

This is a cumulative patch. It includes the AI-consumption telemetry, unified
AI/FFmpeg progress, phase timing, targeted validation repair, Azure autocorrect
schema resilience, and checkpoint-resume work from v13.16.4 through v13.17.1.

## Claude failures handled

- Azure Foundry native Structured Outputs are attempted in `auto` mode. If a
  deployment rejects `output_config.format`, the request is rebuilt once using
  the prompt-schema plus local-validation path. The capability result is
  remembered for the rest of the process.
- Unsupported Foundry `effort`, `thinking`, and prompt-cache controls are
  disabled independently. A deployment rejecting one feature no longer causes
  the application to discard all optional controls.
- HTTP 529 overload responses are retried. `Retry-After` values up to 15 minutes
  are honored instead of being capped at 60 seconds.
- A complete JSON object received before an SSE disconnect, timeout, or stream
  error is validated and salvaged. Incomplete content is never accepted.
- `max_tokens`, context-window, and HTTP 413 failures can split only the failed
  semantic chunk into smaller serial requests. Completed chunks remain reusable,
  split children are checkpointed, and their responses are merged back in the
  original unit order.
- Saved parent transport failures are recognized on resume, so the paid failed
  parent request is not repeated before split recovery.
- Paid token usage from the rejected parent request is retained in the
  consumption aggregate instead of disappearing during recovery.
- Unexpected `pause_turn`, `tool_use`, `stop_sequence`, and context-window stop
  reasons now produce explicit diagnostic failures.

## Schema safeguards

- Provider-only Claude schemas remove unsupported numeric, string, array, and
  object constraints while preserving those rules for local validation.
- Schema complexity is measured before the HTTP request. Schemas exceeding
  Claude's optional-field or union limits fail locally without consuming tokens.
- Fully open object schemas remain on the prompt-schema path instead of being
  converted into empty closed objects.
- Unambiguous case-only enum/const differences are normalized before local
  validation.
- Application schemas remain authoritative. The recovery code does not accept
  missing facts, identities, required translations, or malformed partial JSON.

## Runtime visibility

The cumulative patch retains:

- request bytes, schema bytes, estimated input tokens, configured output limit,
  attempts, cache reads, actual token usage, and cumulative AI consumption;
- chunk progress, validation/identity repair progress, checkpoint reuse, and
  split-child recovery progress;
- FFmpeg frame/time/speed/percentage progress;
- an end-of-command phase-duration table and `TOTAL WALL TIME`.

## Validation

- Python compilation passed.
- 45 focused Claude, Foundry, schema, identity, and translation tests passed.
- A 107-test cumulative regression run had 104 passes and the same three
  pre-existing v13.17.1 failures. A clean v13.17.1 baseline reproduces those
  failures:
  - protected wordplay punctuation;
  - two legacy Khumalo identity-candidate recovery tests.
- Ruff still reports three pre-existing `F402` loop-variable shadow warnings in
  `ai_provider.py`; this patch does not introduce an additional Ruff finding.

## Intentional limits

The process remains fail-closed when recovery cannot prove a valid result.
Semantic omissions, identity loss, or malformed JSON must not be silently
published. The upgrade prevents avoidable termination by retrying, salvaging,
falling back, or splitting only where correctness can still be verified.
