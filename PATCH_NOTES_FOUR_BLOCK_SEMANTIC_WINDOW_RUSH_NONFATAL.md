# Four-block semantic-window + non-fatal rush timing

## Behavioral contract

- Temporal-mask Pass 2 composes fixed chronological four-block semantic/comprehension windows.
- The four English blocks are sent once per provider call.
- Up to two previously committed Zulu blocks are sent once as read-only continuity context.
- Per-mask payloads no longer duplicate n-2/n-1/n/n+1 text.
- Candidate IDs are joint columns across the whole window: candidate `a` across all four masks is one coherent four-block Zulu translation. Mathula never mixes candidate columns.
- Default candidate count is 1. Grok is instructed to internally self-review/revise the four-block combination up to four times before returning it.
- The linguistic timing target is syllabic. Grok does not receive Azure raw-duration milliseconds.
- Each block targets at least one syllable above the source-equivalent syllable estimate. Slightly long is preferred to short.
- Azure TTS still runs at `rate_percent=0` once the wording is returned and remains the physical duration measurement.

## +12% policy

`--max-natural-speed-percent 12` is now a red console quality-warning threshold, not a hard speed ceiling.

- A semantically complete overlong phrase is committable even if it requires >+12% speed.
- Overlong temporal-mask phrases do not enter residual compression merely because they exceed +12%.
- Native-dub does not call Grok to compress an overlong phrase solely for speed.
- Final FFmpeg `atempo` is allowed to exceed the threshold by whatever amount is required to land on the source mouth-close.
- Exceeding the threshold prints a red console warning and never terminates the pipeline.
- Timing candidate ordering now prefers a mechanically rush-fit overlong phrase to an undersized phrase that would require slowing/dead air.

## Residual policy

Residual repair is one call for the entire four-block window, not one call per failing block. It is used only when the joint window has a semantic-guard failure or a materially undersized block. Overlong speech is not a residual failure.

## Token efficiency

The previous temporal-mask wire repeated overlapping English context inside every mask. This build sends:

1. four English blocks once;
2. two committed Zulu predecessors once;
3. four compact mask records containing IDs, speaker, syllable target, and protected literals.

Default `--temporal-mask-candidates` is now 1 instead of 3.
