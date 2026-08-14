# Mathula TV direct Azure timing repair v4

This patch supersedes the v3 duration-evidence patch and includes the required CLI provider wiring.

## Behaviour

- One Claude timing-repair call returns exactly three distinct, validated candidates:
  - `balanced`
  - `safe`
  - `maximum_compression`
- Each candidate includes Claude's duration estimate.
- The server independently predicts duration from exact same-block Azure measurements.
- Candidate ranking blends the server's evidence-derived prediction with Claude's estimate.
- The candidate expected to fit closest to the preferred duration is synthesized first.
- If Azure disproves the estimate, that measured result becomes new evidence and the remaining candidates are re-ranked without another Claude call.
- A second Claude round occurs only when all three candidates from the first round remain too long.
- The shortest measured failed candidate becomes the starting point for the next Claude round.
- Original approved translation artifacts remain immutable and all selected repairs require human review.

## Prompt and artifact versions

- Claude prompt: `claude-turn-repair-zu-v8`
- Claude options response: `claude-turn-timing-repair-options-v1`
- Selected direct repair summary: `mathula-direct-ai-timing-repair-v2`

## Files

- `src/mathula_tv/ai_provider.py`
- `src/mathula_tv/direct_azure_dub.py`
- `src/mathula_tv/cli.py`
- `tests/test_direct_azure_timing_repair.py`
- `tests/test_claude_turn_repair_contract.py`
