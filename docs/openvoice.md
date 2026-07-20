# OpenVoice identity conversion

OpenVoice is the intended production speaker-identity converter. Its source speech is the validated Azure isiZulu WAV; its target identity is a reference reel built only from that job-local source speaker. OpenVoice does not translate, and the server never calls it as a hidden fallback.

The Azure TTS → OpenVoice compatibility gate is currently `pending`. No live OpenVoice inference or identity-quality result is claimed here.

## Isolation and pins

Normal server imports must not import CUDA, Torch, or OpenVoice. A runtime-injected GPU backend loads these packages only in a Colab or Kaggle worker. The combined server plan/queue configuration, private worker runtime contract, and verified worker manifest bind and record:

- immutable OpenVoice revision and unit/input identities in the server plan;
- Python and Torch versions in the worker runtime contract/manifest;
- reviewed checkpoint names/SHA-256 identities in server `MATHULA_TV_OPENVOICE_CHECKPOINT_HASHES_JSON` and matching checkpoint/model-asset materialisation in the private runtime contract/manifest;
- device/precision in the runtime manifest and conversion settings in the plan/result;
- Azure voice/output format and source-calibration identity in the applicable plan/manifests.

The worker verifies pins and assets before model load, loads the model once per session, and reuses source and target embeddings. Missing dependencies/CUDA/checkpoints, revision/runtime mismatch, checkpoint/hash mismatch, CUDA OOM, lease loss, and invalid output are explicit secret-safe errors. Conversion is never bypassed silently and raw Azure TTS is never labelled cloned speech.

## Source embeddings

One 20–40-second reviewed Azure calibration WAV is generated per configured/discovered voice. Its source embedding is cached across jobs only while voice, region, format, calibration-text version/hash, OpenVoice revision, checkpoints, WAV hash, embedding-manifest canonical hash, and embedding hash all match. The server seeds both the embedding and manifest or neither; a partial/tampered pair fails rather than being reused. A changed identity field invalidates the cache. Do not extract a source embedding from every short unit.

## Target reference reels

Build one target reel per job-local speaker from several clean segments, aiming for 20–45 seconds where available. Prefer `isolated_dialogue`, then `clean_source_segment`; `denoised_source` and especially `mixed_source_fallback` need stronger review and normally do not qualify for publication review.

Reject or flag overlap, `UNKNOWN`, another speaker, insufficient voiced content, silence, clipping, heavy music/noise, reverb, and unstable speech. The manifest records included/rejected source segment IDs, timestamps, source text, audio-source type, quality metrics/reasons, duration, and SHA-256. Never borrow another speaker and never create a persistent global identity.

## Per-speaker Azure voice calibration

Once per job-local speaker, render the same reviewed sentence with all allowed/discovered Azure isiZulu voices, convert every candidate using that speaker's target embedding, and compare retranscription, identity similarity, artefacts, duration drift, and human listening. Record every candidate and any human override; cache only the selected result in that job's speaker manifest. A simplistic permanent gender/voice mapping is not sufficient.

An injected notebook evaluator may write completed candidate metrics. If it is absent, reconciliation truthfully preserves `pending_external_qc`; an authorised reviewer must then supply `calibrate-speaker-voices --evaluations FILE`. The JSON uses `openvoice-candidate-evaluations-v1`; `speakers` must contain every manifest speaker, and each speaker must contain every manifest Azure voice:

```json
{
  "schema_version": "openvoice-candidate-evaluations-v1",
  "speakers": {
    "SPEAKER_00": {
      "status": "approved",
      "reviewer": "named-reviewer",
      "reviewed_at": "2026-07-20T12:00:00+00:00",
      "human_override": "zu-ZA-REVIEWED-VOICE",
      "candidates": {
        "zu-ZA-REVIEWED-VOICE": {
          "speaker_similarity": 0.0,
          "word_error_approximation": 0.0,
          "audio_artifact_count": 0
        }
      }
    }
  }
}
```

The numeric values above illustrate types, not recommended scores or pass thresholds. `speaker_similarity` and `word_error_approximation` must be finite values from 0 through 1; `audio_artifact_count` must be a non-negative integer. Use the actual measured values and exact speaker/voice sets. `human_override` is optional, but when present it must name one of that speaker's allowed candidates.

## Conversion request and result

Each request binds job/unit/speaker IDs, Azure source WAV/hash, cached source embedding/hash, target embedding/hash, target language, selected Azure voice, conversion settings, output path, attempt, and current lease identity. Each result records backend/revision, all input/output hashes, source/converted durations and ratio, sample format, device/precision, attempt, and warnings.

OpenVoice output must be canonical validated WAV. Its measured duration—not Azure duration—drives alignment and is recorded as a per-unit duration ratio for QC. Predictive persisted per-speaker ratio planning is not yet applied by the production pipeline.

## Existing lease protocol

The worker reuses the repository's GCS generation-precondition lease:

1. Claim with `if_generation_match=0` or a reviewed reclaim generation.
2. Heartbeat during setup/model load/long conversion.
3. Confirm current ownership before and after each unit and immediately before promotion.
4. Stop on expiry or ownership loss; stale workers cannot promote.
5. Upload unique partials, verify size/SHA-256, promote with generation preconditions, and preserve attempt history.
6. Complete only after every required output and the final manifest are verified remotely.

Partial work is truthful and resumable. A new owner may convert only pending/invalid units after verifying prior completed hashes.

## Two-pass GPU handoff

OpenVoice setup is intentionally not hidden inside queueing:

1. The server runs `prepare-openvoice --prepare-only` to emit asset requirements after Azure calibration WAVs and speaker reels exist. If publication rights are not approved, the operator must select `--proof-of-concept`; the resulting voice-rights artifact permits only restricted internal evidence generation.
2. The server runs `queue-openvoice-assets` to promote the immutable plan and all verified inputs. Run the pinned Colab or Kaggle notebook in `assets` mode with its own explicit live acknowledgement, server asset-plan file SHA-256, and immutable runtime/checkpoint validation. It creates/reuses Azure source and job-local target embeddings, converts the per-speaker Azure voice candidates, optionally calls an injected measured-QC evaluator, promotes verified assets/manifests, and stops without converting dubbing turns.
3. The server runs `reconcile-openvoice-assets`, then `calibrate-speaker-voices` (with reviewed `--evaluations` when notebook metrics remain pending), and resynthesizes affected Azure units if the selected voice changed. Non-prepare `prepare-openvoice` then builds the hash-complete relative conversion plan.
4. The server runs `queue-voice-conversion`. Run the same notebook in `conversion` mode with a separate live acknowledgement and server conversion-plan file SHA-256; it claims the conversion lease, verifies preflight assets/selections, converts pending turns, promotes the final manifest, and completes the lease.
5. The server runs `reconcile-voice-conversion`; verified converted outputs may then enter the still-pending compatibility evaluation. Worker completion by itself is never a compatibility pass.

Do not queue a plan with placeholder embeddings and do not combine unverified preflight output with production conversion.

## Migration policy

F5-TTS is not a backend or fallback. OmniVoice is deprecated, unregistered, unselectable, not production, and unreachable; any remaining files are migration evidence pending deletion after the OpenVoice compatibility path passes or an explicit later cleanup. There is no silent OpenVoice → OmniVoice fallback.

Use the [Colab](colab.md) or [Kaggle](kaggle.md) runbook for a live worker and [compatibility-gate.md](compatibility-gate.md) before downstream production work.
