# One-call temporal-mask self-QA

## Behavioral change
- Python no longer performs temporal-mask residual wording rounds.
- Exactly one logical Grok provider call is made for each forward window of the current block plus up to three following blocks.
- The final window may contain three or two remaining blocks; earlier blocks are never prepended to manufacture four.
- Grok internally performs draft -> whole-window Zulu-to-English back-translation -> semantic comparison -> revision, up to four internal cycles, before returning.
- Azure TTS is not invoked by Pass 2 temporal-mask composition. Timing measurement/fitting is deferred to native-dub.
- A semantically incomplete or undersized provider response is a provider-contract failure; Mathula does not launch a Python repair call.
- The +12% threshold remains downstream warning-only.

## Token/performance goal
This removes repeated large `native_temporal_mask_residual` requests and repeated provider context reconstruction.
