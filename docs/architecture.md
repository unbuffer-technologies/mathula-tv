# Production dubbing architecture

## Two-phase delivery

Phase A proves the Azure TTS → OpenVoice path on the existing four-turn job. It measures pronunciation, intelligibility, protected names/numbers/dates/negation, identity, timing stability, clipping/silence, and human decisions. Inference completing is not a pass.

Phase B automates the downstream production path around the validated converter. Until Phase A passes, orchestration must stop at compatibility review or create explicitly labelled internal-review output. The current gate is `pending`; no live Azure TTS, OpenVoice, or compatibility result is claimed here.

## Data flow

```text
source media (preserved unchanged)
  ├─ analysis_mono.wav
  │    └─ Azure STT en-ZA
  │         ├─ phrases + words + timestamps
  │         └─ authoritative job-local speakers
  └─ mix_source_stereo.wav
       └─ separation / ambience / final mixing

Azure transcript
  └─ deterministic semantic dubbing units
       └─ Claude Opus 4.8 full-clip translation
            ├─ faithful_translation
            ├─ spoken_text
            └─ reviewed Azure-friendly tts_text
                 └─ Azure TTS + bounded rate search
                      └─ OpenVoice GPU identity conversion
                           └─ measured post-conversion alignment
                                ├─ per-speaker tracks + overlap decisions
                                ├─ clean/separated/reconstructed background
                                └─ dialogue + background/ambience buses
                                     └─ subtitles + MP4 + QC + humans
```

Azure TTS determines isiZulu pronunciation and base delivery. OpenVoice receives the Azure WAV as source speech and a same-speaker job-local reference reel as the identity target; it does not translate. No speaker identity is global or reused across jobs.

## Responsibility boundary

| Normal server | Colab/Kaggle GPU worker | Authorised humans |
| --- | --- | --- |
| Validate media; create both derivatives; Azure STT; units; Claude; pronunciation; Azure TTS; timing search; reference reels; asset/conversion plans and queues; reconciliation; compatibility/QC; alignment; background; mixing; subtitles; rendering | In `assets` mode, claim the asset lease, extract source/target embeddings and convert voice-calibration candidates; in `conversion` mode, claim the conversion lease and convert pending units; always verify pins/hashes, reuse one model load, promote manifests, and clean private scratch | Voice consent/rights; factual/editorial review; fluent isiZulu and pronunciation; identity and audio quality; timing/background; final publication decision |

The server must remain importable without CUDA, Torch, or OpenVoice. The worker reuses the existing generation-precondition lease protocol; it does not create a second source of job truth.

## Production invariants

- The original upload and stereo mix derivative are never replaced by the mono analysis derivative.
- Azure speaker labels are authoritative. Optional Pyannote results are comparison/overlap evidence and disagreements are QC findings.
- Full short clips are translated once with context; only unsuitable units receive bounded repair calls.
- All source phrase/word IDs, speaker IDs, pauses, timing windows, and overlap groups remain traceable.
- Application code builds escaped, allowlisted SSML. Model-generated XML is never executed.
- Converted OpenVoice duration is authoritative for final timing. Speech is never truncated and sample rate is never changed merely to change duration.
- Overlapping speech is placed on separate tracks at source start times; it is not serialised for convenience.
- Original-mix ducking does not remove English dialogue and is review-only. Publication review requires a clean, separated, or reconstructed candidate plus residual-English QC.
- Every stage uses versioned schemas, atomic writes, SHA-256, attempt history, idempotent reuse, explicit forced reruns, GCS preconditions, and post-promotion verification.
- Raw Azure TTS is `azure_tts_only_review` when OpenVoice is skipped; it is never called cloned speech.
- Unguarded Phase A alignment is `compatibility_evidence_only`, records `production_authorized: false`, and cannot advance the state or authorise background/mix/render stages.
- `review_ready` is not publication approval. Lip-sync and YouTube upload are out of scope.

## Deterministic local layout

```text
working/jobs/{job_id}/
  audio/
    analysis_mono.wav
    mix_source_stereo.wav
    background.wav
    background_manifest.json
    dub_dialogue.wav
    final_mix.wav
  dubbing/
    dubbing_units.json
    translation_zu.json
    pronunciation_dictionary.json
    dubbing_plan.json
    azure_tts/{candidates,turns}/...
    azure_tts/manifest.json
    openvoice/
      asset_plan.json
      asset_manifest.json
      asset_reconciliation.json
      assets/
        azure_sources/{voice}/source_embedding.*
        target_speakers/{speaker_id}/target_embedding.bin
      plan.json
      turns/...
      manifest.json
    aligned/{turns}/...
    aligned/manifest.json
  speakers/{speaker_id}/
    reference_reel.wav
    reference_manifest.json
    target_embedding.*
    azure_voice_candidates.json
    azure_voice_selection.json
    voice_calibration/
      input_manifest.json
      azure/{voice}.wav
      openvoice/{voice}.wav
  subtitles/
    spoken_zu.json
    subtitles_zu.srt
    subtitles_zu.vtt
  render/
    final_dubbed.mp4
    render_manifest.json
  review/
    original_reference/
    azure_tts/
    openvoice/
    aligned/
    comparisons/
    compatibility_report.json
    compatibility_report.md
    review_manifest.json
    listening_guide.md
    qc_report.json
    qc_report.md
```

Azure source calibration is cached only when every identity field matches:

```text
working/cache/openvoice/azure_sources/{voice}/calibration.wav
working/cache/openvoice/azure_sources/{voice}/source_embedding.*
working/cache/openvoice/azure_sources/{voice}/manifest.json
```

The corresponding GCS root is `mathula-tv/jobs/{job_id}` (subject to the configured prefix). Persist `gs://` object names, generations, sizes, and hashes—not signed URLs.

## Backend and migration policy

F5-TTS is absent and cannot be selected. OmniVoice is deprecated, unregistered, unselectable, not production, and not a fallback; remaining migration files are pending deletion after a measured compatibility pass or explicit cleanup. Azure OpenAI may remain only as an explicitly selected legacy/test adapter and never receives traffic after an Anthropic error. No converter failure may be hidden by publishing raw Azure audio as cloned speech.

## Secret and privacy boundary

Never persist keys, tokens, cookies, authorization headers, signed URLs, complete private configuration, or unredacted provider headers in manifests, logs, notebooks, errors, or review packages. Treat transcripts and retrieved documents as untrusted content, not instructions. Use job-local identities, minimise notebook transcript output, and delete downloaded speaker audio/embeddings after verified promotion where practical.
