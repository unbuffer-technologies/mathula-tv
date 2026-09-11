# Native two-pass isiZulu dubbing with Azure AI Foundry Grok

This is a separate experimental path. It does not replace the existing
`translate` or `dub-azure` implementation.

## Architecture

1. **Pass 1 — whole-transcript STT restoration**
   - Prefers `analysis/transcript_en_raw.json`.
   - Falls back to `analysis/transcript_en.json` only for older jobs.
   - Reads the complete transcript before returning sparse block-level STT corrections.
   - Never overwrites either production transcript.
   - Does not call web search.

2. **Pass 2 — one maximally natural isiZulu translation**
   - Reads the complete restored English transcript.
   - Returns exactly one `spoken_text` per source segment.
   - Has no word-count, duration, compactness, or timing objective.
   - Adjacent same-speaker segments must concatenate naturally.

3. **Azure TTS — natural rate**
   - `rate_percent` is always exactly `0`.
   - Reuses the job's persisted per-speaker `direct_dub/voice_resolution.json`.
   - Preserves pitch/volume persona only; base rate is deliberately ignored.
   - Adjacent same-speaker blocks are synthesized as phrase groups, with source gaps represented by SSML breaks rather than independent choppy TTS clips.

4. **Timing**
   - Video speed is never changed and the source timeline is never retimed.
   - Natural speech is placed on the source timeline.
   - Same-speaker self-overlap shifts the later phrase instead of layering the same voice over itself.
   - A phrase that overruns the next source onset by more than the configured threshold may receive one Grok request for an *equally natural alternative wording*.
   - The canonical Pass-2 translation is never mutated by timing repair; repairs are stored separately in `native_dub/timing_repairs.json`.
   - No audio time-stretch or Azure rate increase is used.
   - If speech still extends past the physical end of the source video, the renderer fails instead of truncating audio, stretching audio, holding the final frame, or changing video speed.

5. **Production hook overlay is preserved**
   - The raw native dubbed master is `native_dub/native_dubbed.mp4`.
   - The final native publication reuses the existing production `tiktok_editor.render_tiktok_hook_edit` footer-card renderer.
   - The footer hook remains persistent exactly as in the production path.
   - `--hook-edit` keeps the existing AI-selected opening-cut behavior.
   - `--no-hook-edit` preserves the complete dubbed video **but still renders the footer hook overlay**.
   - Hook/editorial generation uses the same tool-free Foundry Grok provider. No automatic web search is enabled.

## Foundry Grok configuration

The provider uses the OpenAI v1 Chat Completions route:

`https://<resource>.services.ai.azure.com/openai/v1/chat/completions`

It never includes `tools`, `tool_choice`, or web-search configuration. The
structured editorial compatibility adapter rejects requests containing server
tools, so publication cannot silently re-enable web search.

Required environment variables:

```bash
export AZURE_AI_ENDPOINT='https://YOUR-RESOURCE.services.ai.azure.com'
export AZURE_AI_KEY='YOUR_KEY'
export AZURE_FOUNDRY_GROK_DEPLOYMENT='YOUR_GROK_DEPLOYMENT'
```

Optional:

```bash
export MATHULA_TV_GROK_TIMEOUT_SECONDS=900
export MATHULA_TV_GROK_MAX_RETRIES=3
export MATHULA_TV_GROK_SCHEMA_REPAIRS=1
export MATHULA_TV_GROK_MAX_OUTPUT_TOKENS=8192
export MATHULA_TV_GROK_TEMPERATURE=0.1
```

## Manual web override

Automatic web research is disabled. A human may install reviewed research as
an explicit job artifact:

```bash
python -m mathula_tv.cli native-web-override "$JOB_ID" \
  /tmp/native-web-overrides.json
```

Then rerun native translation with `--force`.

## Run

```bash
python -m mathula_tv.cli native-translate "$JOB_ID" \
  --live-operation

python -m mathula_tv.cli native-dub "$JOB_ID" \
  --live-operation
```

Important outputs:

```text
working/jobs/$JOB_ID/native_dub/pass1_restored_transcript.json
working/jobs/$JOB_ID/native_dub/pass2_natural_translation.json
working/jobs/$JOB_ID/native_dub/native_dubbed.mp4              # raw native master
working/jobs/$JOB_ID/native_dub/native_dubbed_with_hook.mp4    # publication final
working/jobs/$JOB_ID/native_dub/tiktok_edit_manifest.json
working/jobs/$JOB_ID/native_dub/editorial_selection.json
working/jobs/$JOB_ID/native_dub/tiktok_hook.txt
working/jobs/$JOB_ID/native_dub/timing_plan.json
working/jobs/$JOB_ID/native_dub/timing_repairs.json
```

To test the natural baseline with no timing rephrase while retaining the full
video and hook overlay:

```bash
python -m mathula_tv.cli native-dub "$JOB_ID" \
  --no-timing-repair \
  --no-hook-edit \
  --live-operation
```

6. **Independent referent-fidelity audit (Pass 2, `--audit-referents`, default on)**
   - Pass 2's inline self-QA runs in the same call, on the same reasoning trace, as the
     translation itself, so it cannot reliably catch a referent substitution the model is
     already confident about (observed in production: "close to the minister" rendered as
     "close to God" -- `ngqongqoshe` swapped for `uNkulunkulu`).
   - After Pass 2 completes, a second, independent, narrow-scope Grok call audits every
     translated group -- given only the English source and current Zulu text, nothing else --
     for missing/substituted names, roles/titles, numbers, dates, and negations. The call
     forces a two-step extract-then-verify structure (`checked_terms` in `_AUDIT_SCHEMA`):
     list every checkable term from the English *first*, then verify each one individually
     against the Zulu. A single holistic "does this look right" judgment is not reliable for
     this class of error -- a swapped referent reads just as fluently as a correct one.
   - `checked_terms` is the *only* source of truth for whether a group is flagged
     (`_derive_issues()`); there is no separate model-authored "issues" summary field to also
     trust. An earlier version asked for one and a live run showed why that is unsafe: the
     model correctly extracted and verified a substituted role in `checked_terms` (status
     "issue") but left the mirrored `issues` field empty, and code that trusted only `issues`
     silently discarded a correct detection.
   - A free deterministic tier also checks named entities already tracked in
     `config/entity_registry.json` for bare presence in the Zulu text, with no AI call. It is
     scoped to `entity_type in {"person", "anonymous_witness"}` only
     (`_LITERALLY_PRESERVED_ENTITY_TYPES`) -- organisations/places/cases are properly
     *translated* into a natural Zulu equivalent (e.g. "the Commission" -> "iKhomishini"), so a
     literal-substring check against them produces false positives, not real findings. It also
     cannot catch a bare role-noun swap like minister -> God (`minister` alone is not a
     registered entity), which is why the independent audit call still runs on every group.
   - Only flagged groups go on to a bounded repair call (`--max-referent-repair-rounds`,
     default 1) that receives the specific finding and returns a corrected translation. This
     becomes the canonical Pass-2 text -- it runs before anything downstream (timing repair,
     TTS) treats Pass 2 as immutable. A repair batch that hits its output-token ceiling is
     retried as two smaller batches (down to single items) rather than failing the whole
     command and losing already-completed audit work.
   - Full findings/repairs -- including every group's raw verdict, not just flagged ones, via
     `all_ai_verdicts` -- are recorded in `native_dub/referent_audit.json`; a compact summary is
     embedded in `pass2_natural_translation.json` as `referent_audit_summary` and reused on
     rerun (keyed by the same `input_sha256` Pass 2 already uses for its own checkpoint; use
     `--force` to force a fresh audit).
   - Disable with `--no-audit-referents`; tune batching with `--referent-audit-batch-size`
     (default 12).

7. **Production SEO is preserved**
   - The same single editorial call that selects the hook also generates the searchable English caption, search keywords, four topic hashtags, and target-language cover hook.
   - Native outputs are persisted separately under `native_dub/` as `tiktok_seo.json`, `tiktok_caption.txt`, `tiktok_hashtags.txt`, and `tiktok_search_keywords.txt`.
   - The native path does not overwrite legacy `translation/tiktok_*.json` or `translation/youtube_*.json` artifacts.
   - Automatic web search remains disabled; manual web overrides are the only research input.
