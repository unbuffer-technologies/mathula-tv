# Mathula TV engineering handoff

Last updated: 2026-07-20

## Authoritative status

The intended production path is Azure STT → Claude Opus 4.8 → Azure TTS → OpenVoice → measured alignment → background reconstruction → mix/subtitles/render/QC → mandatory human review.

The Azure TTS → OpenVoice compatibility gate is **pending**. Live Phase A execution is in progress for proof job `19ba6d69f1b84132ba4f20599101834a` (a 156.885-second clip). The legacy OmniVoice `review_ready` record was explicitly archived in its attempt history and reset into the Azure TTS/OpenVoice path with its approved translation hashes preserved. No OpenVoice inference, compatibility pass, background separation, final render, or human-approved production video has completed.

Completed live proof steps: GCS access validation; source audio derivative recovery and validation; Azure voice discovery (`zu-ZA-ThandoNeural` and `zu-ZA-ThembaNeural`); Azure source calibration WAV generation; 24 deterministic dubbing units and the three-text plan; and targeted Azure Foundry Claude timing repairs. The initial Azure timing pass found nine affected units; a second repair pass completed units `unit_0018`, `unit_0022`, and `unit_0023`. The next action is a normal Azure synthesis rerun, which reuses verified unaffected units and regenerates repaired units.

Use this exact next command from the remote server after starting a new session:

```bash
cd /home/mokgethwa/mathula-tv
unset AZURE_SPEECH_KEY AZURE_SPEECH_REGION
GOOGLE_APPLICATION_CREDENTIALS=/home/mokgethwa/.config/mathula-tv/gcp-worker.json .venv/bin/python -m mathula_tv.cli synthesize 19ba6d69f1b84132ba4f20599101834a --backend azure-tts --live-operation
```

Never commit or paste `.env`, the service-account JSON key, Azure/Foundry keys, clips, transcripts, generated audio, job manifests, logs, or Colab secrets. The public repository contains code, tests, and this non-secret handover only.

## Architectural decisions

- Azure Speech-to-Text is the production transcription provider. Its job-local speaker labels are authoritative.
- Pyannote is opt-in diagnostic/comparison/overlap analysis only and cannot overwrite Azure labels silently.
- Claude Opus 4.8 through the Anthropic provider is the runtime default for translation, repair, contextual analysis, SEO, and editorial metadata. Azure OpenAI is legacy/test-only when selected explicitly; failure never triggers a silent fallback.
- Azure Text-to-Speech creates the native isiZulu base delivery and owns pronunciation. Application code—not Claude—constructs allowlisted, escaped SSML.
- OpenVoice transfers job-local speaker identity after Azure TTS in a pinned Colab or Kaggle GPU runtime. It does not translate text, and the normal server does not import CUDA/OpenVoice packages.
- F5-TTS is absent from the Mathula TV architecture and must not be registered or restored.
- OmniVoice is deprecated, not registered, not selectable, not production, and unreachable. Any remaining source/notebook is migration evidence pending deletion after a compatibility pass or explicit cleanup; it is never a fallback.
- Genuine overlap remains overlap. Final timing is based on measured OpenVoice output, not raw Azure duration.
- Original English dialogue must be removed or sufficiently suppressed for publication review. Ducking the original mix is internal-review-only.
- Subtitles use `spoken_text`, never pronunciation-respelled `tts_text`. Automatic phoneme-level lip synchronisation is out of scope.
- Mathula TV does not upload to YouTube.

## Implemented, offline-testable surfaces

The repository contains versioned boundaries and deterministic artifacts for:

- atomic job manifests, explicit production states, legacy `synthesis_queued` migration, attempt history, and GCS generation-precondition leases;
- dual audio derivatives (`analysis_mono.wav` and `mix_source_stereo.wav`) and the 240-minute production source limit;
- Azure STT normalization with phrase/word timing and authoritative Azure speaker labels;
- deterministic semantic dubbing-unit construction with source-word/phrase provenance, pauses, hard windows, and overlap groups;
- the Anthropic/Claude provider, schema-validated full-clip requests, bounded per-turn repairs, prompt hashes/versions, usage metadata, retry/refusal classification, and redacted summaries;
- three target-text forms, a versioned pronunciation dictionary, and safe SSML construction;
- Azure TTS voice discovery, canonical WAV validation, bounded rate search, idempotent output, and repair signals;
- same-speaker reference reels, Azure source calibration caching, target embeddings, and per-speaker Azure voice comparison/override data;
- an isolated, pinned OpenVoice worker interface with a self-verifying asset plan, separate asset-preflight and turn-conversion leases, source/target embedding manifests, measured per-speaker Azure voice candidates, hash validation, resumable manifests, lease checks, and an injected/mockable GPU runtime;
- compatibility observations/reports and A/B package layout without invented universal thresholds;
- post-conversion timing, silence padding, bounded pitch-preserving correction, separate per-speaker tracks with recorded cross-speaker overlap decisions, and no spoken-word truncation;
- background-source selection, a separation-provider boundary, review-only ducking labels, residual-English QC, loudness/mixing metrics;
- `spoken_text` JSON/SRT/VTT subtitles, validated AAC/48-kHz MP4 rendering, and JSON/Markdown QC reports.

These are implementation capabilities, not evidence that external services or a production video have been exercised successfully.

## Phase A: next live milestone

Run the exact opt-in sequence in `docs/cli.md` for the four-turn job. It must preserve the existing approved translation, validate available Azure voices, generate raw Azure WAVs, build same-speaker references and calibration candidates, queue/reconcile the pinned GPU asset pass, persist a measured voice choice, queue/reconcile turn conversion, then produce the A/B package and compatibility report.

The gate cannot pass until every unit has measured Azure/OpenVoice duration, retranscription and protected-item results, calibrated speaker-similarity/timing/audio thresholds, and approved voice-rights, editorial, factual, and fluent-isiZulu reviews. Missing measurements, provisional thresholds, or pending reviews produce `pending`/`needs_human_review`, never `passed`.

## Phase B: guarded production rollout

After an authorised compatibility pass, exercise alignment, background preparation, residual-dialogue detection, overlap-safe mixing, subtitles, final rendering, and QC against representative material. `review_ready` means ready for humans; it is not publication approval. A clean/separated/reconstructed background and residual-English pass are mandatory for the publication-review path. Raw Azure output and original-mix ducking remain labelled internal-review fallbacks.

## External work still required

1. Provide Anthropic, Azure Speech, and private GCS credentials through secret stores; never put them in artifacts or notebooks.
2. Pin the OpenVoice revision in the server plan; configure the reviewed checkpoint name/SHA identities on the server before queueing; and bind matching Python/Torch/checkpoint materialisation through the private GPU runtime contract and verified worker manifest.
3. Run Azure voice discovery in the configured region; configured voice names are only candidates.
4. Calibrate compatibility/speaker-similarity thresholds from representative reviewed examples rather than adopting universal constants.
5. Run the complete four-turn proof, record failures truthfully, and have authorised humans complete all required reviews.
6. Select and validate a real clean-stem or separation provider plus a residual-speech detector/retranscriber integration; the server intentionally does not pull large unverified media models into its default dependencies, and the current CLI records residual QC as `not_run` without an injected detector.
7. Wire persisted per-speaker OpenVoice duration-ratio estimates into later planning if calibration proves useful; current alignment measures every converted unit directly and records per-unit ratios for QC.
8. Delete obsolete OmniVoice implementation files, tests, dependencies, and notebooks only after the OpenVoice path passes or a later explicit cleanup authorises deletion.

## Operational safety

The live job must never be used by unit tests. Live commands require `--live-operation`; paid/GPU integration tests require their individual `MATHULA_TV_RUN_LIVE_*_TESTS=1` opt-ins. Verify manifests and hashes before every promotion, stop a stale worker after lease loss, and never paste secrets, authorization headers, cookies, private signed URLs, full private configuration, or unnecessary transcripts into logs or notebook output.
