# Temporal-mask exact mask identity fix

## Problem
A residual request for `native_phrase_0005` could receive a structurally valid provider response for a neighbouring mask such as `native_phrase_0007`. The generic temporal-mask JSON schema only required a non-empty `group_id`, so the response passed provider schema validation and failed later with:

`Temporal-mask response identity mismatch: missing=native_phrase_0005; unknown=native_phrase_0007`

Mask IDs are semantic identities and must never be remapped by position.

## Fix
- `_temporal_mask_schema()` now accepts `expected_group_ids`.
- The schema constrains `group_id` to the exact requested identity set.
- The schema constrains `masks` cardinality to the exact number requested.
- Speculative compose calls pass the exact batch group IDs.
- Residual calls pass exactly the current `group_id`.
- Residual payload now includes `immutable_group_id` and an explicit identity contract.
- The residual system prompt states that neighbour/previous/future/speculative mask IDs are never valid substitutes.
- Existing application normalization remains strict for missing, duplicate and unknown group IDs.
- Candidate IDs remain positional and may be normalized as before; **mask IDs never are**.

## Safety behavior
A wrong mask ID is rejected by structured-output validation and can use the provider's normal validation retry path. It is never attached to the requested English block.

## Targeted validation
- Wrong residual mask (`0007` returned for requested `0005`) is rejected by JSON Schema `enum`.
- Correct residual mask is accepted.
- Requested compose IDs are accepted in arbitrary response order.
- Exact mask cardinality is enforced.
- Changed Python files compile successfully.
