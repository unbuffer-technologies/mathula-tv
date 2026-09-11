# Native batch timing repair — group identity matching fix

## Problem

`native_natural_timing_repair_batch` required the provider to return `repairs[]` in the exact same list order as the request. Structured-output providers can reorder independent batch items while still returning a complete and valid result, causing Mathula to abort with:

`Native timing repair batch must preserve exact group order and cardinality`

## Fix

- Treat `group_id` as the stable batch identity.
- Accept provider responses in any order.
- Reconstruct the application result in the original requested order.
- Continue deriving `changed` deterministically from returned text.
- Reject true identity failures:
  - missing requested `group_id`
  - duplicate returned `group_id`
  - unknown/extra returned `group_id`
  - unique cardinality mismatch

This does not weaken segment-level validation inside each phrase repair.
