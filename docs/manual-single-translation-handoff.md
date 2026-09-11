# Manual AI single translation handoff

When `--provider manual-chatgpt` is selected, Mathula TV now exports the complete
multivariant translation pass as one manual request instead of reusing the
normal cloud semantic chunk count.

- Every active translation unit appears once in the top-level `units` array.
- `translation_batch.chunk_count` is `1`.
- `translation_batch.manual_single_handoff` is `true`.
- `translation_batch.context_units` is empty because all active units are
  already present at the top level.
- The bounded context spine is retained for global story/context continuity.
- The manual provider may use up to its larger translation output budget rather
  than the normal 12k compact cloud-chunk ceiling.
- Azure/OpenAI cloud translation chunking is unchanged.

Set `MATHULA_TV_MANUAL_TRANSLATION_SINGLE_HANDOFF=0` to restore chunked manual
translation deliberately. `MATHULA_TV_MANUAL_TRANSLATION_MAX_OUTPUT_TOKENS`
controls the manual handoff ceiling and defaults to at least 100000.
