# Mathula TV v5: batched timing repair and authoritative final output

Base: apply on top of the v4 three-candidate duration patch.

## Timing repair

- Runs the baseline Azure synthesis pass across all speech blocks first.
- Collects every measured overflow before calling Claude.
- Sends up to 12 problematic blocks in one Claude request by default.
- Requests balanced, safe, and maximum-compression candidates for every block.
- Keeps protected placeholders isolated per block.
- Validates every block independently; one malformed block cannot select output for another.
- Ranks with exact same-voice Azure duration evidence and synthesizes only the best candidate first.
- Re-ranks the two remaining candidates after every failed Azure measurement.
- Sends only unresolved blocks in the next bounded Claude round.
- Persists batch requests/responses and per-block selected artifacts.

## Final output naming

- Clean rendered master: direct_dub/dubbed_master.mp4
- User-facing master copy: output/dubbed_master_<JOB_ID>.mp4
- Publication final with persistent hook panel: output/final_dubbed_<JOB_ID>.mp4
- The direct renderer removes the misleading legacy direct_dub/final_dubbed.mp4.
- A newly rendered master invalidates an old publication final before the panel pass.
- report.json records publication_final_video only after the TikTok editor succeeds.
- The CLI reloads report.json after the panel render, so its final_video is authoritative.

## Verification

Focused offline regression set: 64 passed.
The repository-wide suite still has the pre-existing collection failure in tests/test_webm_ingestion.py because mathula_tv.media does not export normalize_webm_to_mp4.
