# Dubbing cadence architecture

Facts, numbers, protected entities, attribution, uncertainty and editorial
intent are immutable. Natural South African code-switching is unrestricted:
English may comprise half or more of a turn when that best matches the original
speaker. Translation purity is never a quality target; source speed, tone,
energy and fluency take priority. Approved wording is versioned and may be revised through
an auditable quality loop; the renderer may never rewrite it silently.

```text
versioned approved segments
  -> adjacent same-speaker grouping
  -> phrase/breath-group cadence plan
  -> source phrase timestamps and pause map
  -> natural persona-rate Azure synthesis (one request per group)
  -> align each translated breath group to its source phrase window
  -> preserve the speaker's original pause positions and approximate durations
  -> meaning-preserving shorter wording revision for material overflow
  -> bounded natural-rate correction only
  -> cross-speaker collision check
  -> perceptual cadence QC
  -> semantic/editorial invariant verification
  -> audited revised translation version
  -> dialogue loudness normalization
  -> constant-frame-rate TikTok render
```

Each block report contains:

- `cadence_plan`: exact phrase text spans and allocated timeline windows;
- `timeline_fill_action`: how its audio was fitted;
- `cadence_qc`: rate delta, whole-voice stretch, pause evidence, limits and
  review issues.

Perceptual QC flags a block when:

- its whole-turn Azure rate differs from the speaker persona by more than 5%;
- whole-voice post-fit correction changes duration by more than 3%;
- any internal pause has been inserted or expanded.
- continuous speech leaves more than 750 ms of unexplained room at the end of
  a segment instead of preserving source phrase pauses.

Short target speech is never post-stretched and internal pauses are never
invented from waveform silence. Translated breath groups are synthesized
naturally and placed against the source phrase/pause map. Small genuine room
tone can remain; accumulated missing rhythm may not be dumped at the end of a
continuous speech segment. When speech would overlap the next speaker, wording
is shortened and resynthesized while preserving facts, numbers, protected
entities, attribution, uncertainty and editorial intent.

Cadence failure does not make poor audio silently production-ready. The renderer
writes `direct_dub/translation_revision_required.json` with exact block and
phrase targets. A translator or language model may propose revised wording, but
that revision must preserve the hard invariants and become a new approved
version with its previous and revised hashes recorded.
