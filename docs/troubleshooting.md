# Troubleshooting

Failures should be actionable, stage-specific, bounded, and secret-safe. Inspect the redacted job manifest, current state, attempt history, stage manifest, referenced files, sizes/hashes, and lease generation before forcing a rerun.

| Symptom | Check and action |
| --- | --- |
| Protected live job command is refused | Add `--live-operation` only after confirming the literal job ID, credentials, expected state, and intended mutation. Never weaken the guard. |
| Legacy job remains `synthesis_queued` | Run inspect/validate and the migration dry-run. Migrate only from that exact state; preserve existing translations and hashes. Do not start GPT automatically. |
| GPT authentication/rate-limit/timeout/refusal/invalid JSON | Check `AZURE_AI_KEY`, endpoint, deployment agreement, output budget, retry budget, and the redacted validation summary. There is no model-provider fallback. |
| Azure STT rejects media | Verify `analysis_mono.wav`, endpoint/region/locale, size and 240-minute policy, ffprobe metadata, and credentials. Do not truncate or invent cross-chunk speaker continuity. |
| Azure speakers disagree with Pyannote | Keep Azure authoritative; record diagnostic disagreement/overlap in QC. Never overwrite labels silently. |
| Configured Azure TTS voice unavailable | Run voice discovery in the configured region and select only an allowed returned `zu-ZA` voice. A name in `.env` is not availability proof. |
| Azure TTS cancellation or malformed WAV | Inspect the secret-safe cancellation category, output format, SSML bounds, XML escaping, and retry class. Delete/promote only through atomic attempt handling. |
| Name/number/date is pronounced poorly | Review `spoken_text` vs `tts_text`, dictionary precedence, normalisation, and SSML substitution; listen again and record a reviewed override. Never put respelling in subtitles. |
| Azure speech cannot fit a hard window | Confirm the semantic-fit tolerance and measured Azure duration. Request a targeted GPT contraction; do not change speaker tempo, remove critical facts, or truncate words. |
| Same-speaker reference missing/poor | Review rejected reel candidates for overlap, `UNKNOWN`, cross-speaker audio, silence, clipping, music/reverb, and voiced duration. Never substitute another speaker. |
| OpenVoice import/CUDA/checkpoint/revision failure | Use the pinned Colab/Kaggle runtime, verify Python/Torch/OpenVoice revisions and checkpoint hashes before model load, and avoid installing GPU dependencies on the server. Server `MATHULA_TV_OPENVOICE_CHECKPOINT_HASHES_JSON` names/SHA-256 values must exactly match worker `MATHULA_TV_OPENVOICE_CHECKPOINTS_JSON`. |
| CUDA out of memory | Stop cleanly, preserve the partial manifest, restart/select adequate GPU, reclaim only after lease expiry, and resume pending units within attempt limits. |
| Lease lost or GCS precondition failed | Stop the stale worker immediately. Inspect current owner/expiry/generation; never promote or overwrite from the old session. |
| Hash mismatch or partial promotion | Quarantine/discard the invalid partial, redownload from the authoritative generation, verify size/SHA-256, and resume the affected unit with attempt history. |
| Asset worker completed but reconciliation is pending/refused | Check the `openvoice_asset_preparation` claim identity/status, server plan file/artifact hashes, runtime/checkpoint identity, expected speaker/voice sets, and every promoted output hash. Do not queue turn conversion from notebook-local files. |
| Speaker voice calibration reports `pending_external_qc` | Supply measured, approved `openvoice-candidate-evaluations-v1` data through `--evaluations FILE`, or rerun the asset worker with a reviewed evaluator. Never invent scores or silently select a candidate. |
| OpenVoice ran but compatibility is pending | This is expected when measurements, calibrated thresholds, or human reviews are absent. Generate missing evidence; never equate inference completion with a pass. |
| Identity/similarity score is weak | Check same-speaker reel, selected Azure base voice, backend/version and calibration. Do not lower an uncalibrated universal threshold to force a pass. |
| Converted unit exceeds timing bounds | Use measured converted duration and permitted pitch-preserving correction (`0.94`–`1.06` initially); otherwise repair the unit and reconvert it. Never resample or truncate to hide the overrun. |
| Background still contains English | Reject publication review, inspect separation/reconstruction and residual-speech evidence, and choose a better candidate. Ducking is internal-review-only. |
| Mix clips or sounds unbalanced | Inspect overlap placement, bus gains/fades, LUFS/LRA/true peak/clipping, and dialogue/background balance; peak normalisation alone is inadequate. |
| Render has missing streams/duration mismatch | Run ffprobe on source/final mix/output, ensure final speech fits, use AAC 48 kHz, and re-render. Do not truncate speech or add black frames. |
| Secret appears in logs/notebook output | Stop sharing/promotion, revoke/rotate it, remove unsafe outputs/artifacts, run the redaction/secret scan, and repeat with secret-store injection. |
| Someone proposes F5-TTS or OmniVoice fallback | Reject it. F5-TTS is absent; OmniVoice is deprecated, unregistered, unselectable, not production, and pending deletion. |

`--force` is not a general repair tool: it must preserve attempts, respect leases/generations, and never change approved text without explicit authorisation. Raw Azure output may be created only as `azure_tts_only_review`. No troubleshooting path uploads to YouTube.
