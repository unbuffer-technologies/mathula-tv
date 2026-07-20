# Production state machine

Job state lives in the atomic job manifest. A stage may advance only after its complete manifest and referenced hashes validate. Claims and promotions also verify the current remote generation, so a stale worker cannot overwrite a current lease owner.

```text
created → audio_prepared → uploaded → analysis_queued → analysis_running
  → analysis_ready → translation_running → translation_ready
  → azure_tts_queued → azure_tts_running → azure_tts_ready
  → voice_conversion_queued → voice_conversion_running
  → voice_conversion_ready → compatibility_review
  → alignment_running → alignment_ready
  → background_processing → background_ready
  → rendering → review_ready
```

Explicit structured failures may end in `failed`, `failed_retryable`, `failed_terminal`, or `cancelled` as supported by the manifest version. A retry must be authorised, preserve prior attempt records, and re-enter only a valid state.

| State | Required truth before entry |
| --- | --- |
| `analysis_queued` | Original media and both validated audio derivatives are identified |
| `analysis_running` | Correct Azure STT attempt is active |
| `analysis_ready` | Raw and normalized Azure artifacts exist; Azure speaker labels are authoritative |
| `translation_running` | Anthropic request budget and input schema are valid |
| `translation_ready` | Full-clip output validates and all units have three target-text forms |
| `azure_tts_queued` | Pronunciation/SSML plan and voice candidates validate |
| `azure_tts_running` | Correct attempt owns synthesis work |
| `azure_tts_ready` | Every required canonical WAV and hash validates |
| `voice_conversion_queued` | References, embeddings/calibration identities, plan, pins, hashes, and lease input exist |
| `voice_conversion_running` | Current GPU worker owns the non-expired lease |
| `voice_conversion_ready` | Verified OpenVoice manifest and every required converted WAV are promoted |
| `compatibility_review` | Evidence collection/review is open; observations or A/B assets may still be pending and a pass is not implied |
| `alignment_ready` | Compatibility policy allows advancement; all measured corrections fit bounds |
| `background_ready` | Background manifest and residual-dialogue result exist; review-only mode remains labelled |
| `review_ready` | The review render and subtitles validate and human work is queued; the separate `review` command records QC/readiness, and publication is not approved |

OpenVoice asset preflight is a resumable substage while the job remains `azure_tts_ready`: the server creates and queues the asset plan, the GPU worker completes `openvoice_asset_preparation`, the server reconciles embeddings/candidate manifests, and an approved measured base-voice selection is persisted. If that selection differs from an already-rendered turn's Azure voice, the job returns to `azure_tts_queued` and only affected turns are resynthesized. Otherwise, only the later `queue-voice-conversion` operation advances the job to `voice_conversion_queued`. Asset completion alone never implies converted turns or compatibility approval.

## Compatibility guard

Normal production advancement from `compatibility_review` requires a calibrated `passed` report with required human evidence. `pending`, `needs_human_review`, or any `failed_*` result stops production automation. A proof-of-concept/internal-review alignment is labelled `compatibility_evidence_only` with `production_authorized: false`, leaves the job in `compatibility_review`, and may package evidence only. Background, mixing, and rendering independently require a passed gate plus an OpenVoice alignment manifest marked `compatibility_passed` and `production_authorized: true`.

If OpenVoice is explicitly skipped, the job follows only an internal review route and its readiness is `azure_tts_only_review`. It never records `voice_conversion_ready` or claims speaker cloning.

## Legacy `synthesis_queued` migration

`synthesis_queued` remains readable solely to migrate validated old jobs. The safe migration is:

1. Inspect and validate the manifest and translation hashes.
2. Dry-run `migrate-dubbing-state`.
3. Upgrade each approved old turn into `faithful_translation`, `spoken_text`, and `tts_text` without changing approved content; record any pronunciation substitution separately.
4. Preserve the old transcript, translation/SEO objects, checksums, providers, timestamps, and attempt history.
5. Transition atomically to `azure_tts_queued`, set compatibility to `pending`, and set readiness to `compatibility_pending`.

Migration is valid only from legacy `synthesis_queued`. It must not call Claude, regenerate translation, select OmniVoice, or delete old artifacts. Re-running a successfully migrated job is idempotent; mismatched or partial evidence fails safely. See the exact live-job commands in [cli.md](cli.md).

Legacy `translating`, `synthesis_claimed`, `synthesizing`, and `synthesis_ready` may remain parseable for recovery, but new production dispatch never selects a legacy synthesis backend.
