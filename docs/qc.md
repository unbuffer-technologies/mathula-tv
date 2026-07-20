# Automated and human-facing QC

QC produces machine-readable JSON and human-readable Markdown. It informs review; it cannot grant consent or publication approval. The Azure TTS → OpenVoice compatibility gate is currently `pending`.

## Signals

- Retranscribe raw Azure TTS and OpenVoice output with Azure STT `zu-ZA`; compare to `spoken_text`, protected names, numbers, dates, negation, and acronyms. Treat STT error as a signal because recognition itself may be weak.
- Compare original reference vs OpenVoice, original reference vs raw Azure, and—when available—OpenVoice vs unrelated job speakers through an injected versioned embedding backend. Record all input hashes and calibration status.
- Measure Azure/OpenVoice/aligned durations, ratios, final timing errors, correction factors, padding, clipping, silence, and artefacts.
- Validate same-speaker reel quality and selected Azure base-voice calibration evidence.
- Detect residual English in the background candidate and record mode/provider/retranscription evidence.
- Measure integrated loudness, loudness range, true peak, clipping, dialogue/background balance, final duration, and output streams.
- Validate translation/repair counts, protected terms, subtitles, render manifest, leases/promotions, and important artifact hashes.

Do not invent a universal speaker-similarity or intelligibility threshold. Provisional/uncalibrated thresholds are labelled and cannot pass compatibility.

## Report contents

The current consolidated report records job/source identity/duration; speaker/unit/overlap counts; Azure STT backend/version; Claude model/prompt version; Azure voices and TTS attempts; OpenVoice revision/attempts; source/target embedding hashes; reference and per-speaker voice-selection quality; per-unit durations/ratios/timing errors; translation repairs; retranscription/protected-item/similarity results; residual-English and background strategy; loudness/peak/clipping/silence; failures/fallbacks; human requirements; and hashes for important artifacts. The job manifest retains the selected translation provider, and the hash-linked verified OpenVoice worker manifest retains the exact checkpoint set/runtime identity; the consolidated QC JSON does not duplicate those fields today.

No report contains secrets, authorization headers, signed URLs, embeddings, or private full configuration.

## Readiness values

```text
compatibility_pending
compatibility_failed
azure_tts_only_review
needs_translation_review
needs_pronunciation_review
needs_voice_review
needs_timing_review
needs_background_review
needs_audio_review
review_ready
failed
```

`review_ready` means the artifacts are ready for authorised humans. It is deliberately not `production_ready`; only an explicit human publication decision outside automated readiness can approve use. A skipped OpenVoice stage, ducked original mix, pending gate, missing consent, or substantial residual English cannot be promoted by relabelling.

See [compatibility-gate.md](compatibility-gate.md) for Phase A evidence and [human-review.md](human-review.md) for the publication boundary.
