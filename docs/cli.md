# CLI and live compatibility runbook

Use `python -m mathula_tv.cli --help` and `python -m mathula_tv.cli COMMAND --help` from the activated environment. Commands are resumable and reuse verified artifacts unless `--force` is explicit. `--dry-run` must not mutate local or remote state.

## Production command surface

```bash
python -m mathula_tv.cli inspect JOB_ID
python -m mathula_tv.cli migrate-dubbing-state JOB_ID --dry-run
python -m mathula_tv.cli prepare-source-derivatives JOB_ID
python -m mathula_tv.cli build-dubbing-units JOB_ID
python -m mathula_tv.cli translate JOB_ID --provider anthropic --model claude-opus-4-8 --live-operation
python -m mathula_tv.cli repair-translation JOB_ID --provider anthropic --model claude-opus-4-8 --live-operation
python -m mathula_tv.cli prepare-dubbing JOB_ID
python -m mathula_tv.cli validate-azure-voices JOB_ID --live-operation
python -m mathula_tv.cli calibrate-azure-sources JOB_ID --live-operation
python -m mathula_tv.cli synthesize JOB_ID --backend azure-tts --live-operation
python -m mathula_tv.cli build-reference-reels JOB_ID --candidates /path/to/candidates.json
python -m mathula_tv.cli calibrate-speaker-voices JOB_ID --prepare-only --live-operation
python -m mathula_tv.cli prepare-openvoice JOB_ID --prepare-only --device cuda
python -m mathula_tv.cli queue-openvoice-assets JOB_ID --live-operation
python -m mathula_tv.cli reconcile-openvoice-assets JOB_ID --live-operation
python -m mathula_tv.cli calibrate-speaker-voices JOB_ID --evaluations /path/to/openvoice_candidate_evaluations.json --live-operation
python -m mathula_tv.cli prepare-openvoice JOB_ID --device cuda
python -m mathula_tv.cli queue-voice-conversion JOB_ID --live-operation
python -m mathula_tv.cli reconcile-voice-conversion JOB_ID --live-operation
python -m mathula_tv.cli evaluate-compatibility JOB_ID --observations /path/to/observations.json --thresholds /path/to/thresholds.json
python -m mathula_tv.cli align JOB_ID --require-compatibility-pass
python -m mathula_tv.cli build-review-package JOB_ID
python -m mathula_tv.cli prepare-background JOB_ID --clean-stem /path/to/clean_background.wav
python -m mathula_tv.cli mix JOB_ID
python -m mathula_tv.cli subtitles JOB_ID
python -m mathula_tv.cli render JOB_ID
python -m mathula_tv.cli review JOB_ID
python -m mathula_tv.cli process JOB_ID
```

If protected-job inspection proves a legacy OmniVoice job is incorrectly marked
`review_ready`, an authorised operator may preserve its old artifacts and reset
only its state with `migrate-dubbing-state JOB_ID --allow-legacy-review-reset
--live-operation`. The command rejects non-OmniVoice or non-legacy review jobs.

Important controls:

- `--live-operation` explicitly authorises touching the protected proof job or live GCS. Credentials are still required.
- `calibrate-speaker-voices --prepare-only` creates the identical reviewed Azure candidate sentence for every job-local speaker/allowed voice; `prepare-openvoice --prepare-only` builds the self-verifying GPU asset plan without pretending embeddings or conversion completed.
- `queue-openvoice-assets` and `reconcile-openvoice-assets` are the asset-preflight handoff. They are distinct from the later turn-conversion queue/reconciliation.
- `calibrate-speaker-voices --evaluations FILE` accepts approved external candidate metrics when the optional notebook evaluator was not injected. It validates the `openvoice-candidate-evaluations-v1` speaker/voice sets, reviewer, timezone-aware review time, finite 0–1 similarity/error values, and non-negative artefact counts before ranking.
- `build-reference-reels --proof-of-concept` permits truthful mixed-source internal references; `prepare-openvoice --proof-of-concept` permits identity-conversion evidence when rights are not publication-approved; `prepare-background --proof-of-concept` permits review-only source ducking. Each writes restrictive review metadata and none can approve publication.
- `align --skip-openvoice` creates only `azure_tts_only_review` output and never implies identity conversion.
- `align --require-compatibility-pass` is the normal production guard. `--no-require-compatibility-pass` is restricted to the clearly labelled Phase A alignment evidence described below.
- `prepare-openvoice --device cuda` rejects non-CUDA production planning.
- `--force` preserves previous attempt history and must never overwrite an owned lease or silently change approved text.

`process` is guidance-only for the production lifecycle: it inspects the current state and prints the next required command. It does not silently execute a paid provider call, GPU stage, or human-gated transition.

## Exact four-turn job sequence

The protected job is `19ba6d69f1b84132ba4f20599101834a`. The incoming handoff reports legacy state `synthesis_queued` with four approved isiZulu turns. **Do not run `translate` for this job** unless a human explicitly requests retranslation. The migration/prepare stages preserve the approved text while creating the three-text representation.

Before even the read-only inspection command, configure the intended private bucket and verified server credentials. The guard accepts an explicit `GOOGLE_APPLICATION_CREDENTIALS` path supplied outside the repository, or ADC/workload identity only when `MATHULA_TV_USE_APPLICATION_DEFAULT_CREDENTIALS=1` is deliberately set. That acknowledgement is not a credential and does not replace IAM validation. Keep its `.env.example` default at `0` outside an authorised live session.

Before either OpenVoice queue, the normal server also requires the immutable 40-character `MATHULA_TV_OPENVOICE_REVISION` and non-empty `MATHULA_TV_OPENVOICE_CHECKPOINT_HASHES_JSON`, preferably a JSON object mapping each unique reviewed checkpoint name to its 64-hex SHA-256. The private GPU runtime uses `MATHULA_TV_OPENVOICE_CHECKPOINTS_JSON` with the exact same identities plus worker-only `filename` and `source_path`; a server/worker checkpoint-set mismatch fails reconciliation. The server accepts the full worker variable only as a compatibility fallback, not the recommended secret boundary.

The commands below are an operator runbook, not commands executed during repository tests. Run one at a time, inspect its output and artifacts, and stop on any unexpected state/hash. Every protected-job command includes the required opt-in flag.

### 1. Inspect, validate, and migrate without retranslation

```bash
python -m mathula_tv.cli inspect 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli validate 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli migrate-dubbing-state 19ba6d69f1b84132ba4f20599101834a --dry-run --live-operation
python -m mathula_tv.cli migrate-dubbing-state 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli prepare-source-derivatives 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli build-dubbing-units 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli prepare-dubbing 19ba6d69f1b84132ba4f20599101834a --live-operation
```

Confirm that the source translation hashes/content are unchanged, each unit has `faithful_translation`, `spoken_text`, and `tts_text`, and every spoken/tts difference is recorded.

### 2. Validate Azure voices and generate raw Azure isiZulu WAVs

```bash
python -m mathula_tv.cli validate-azure-voices 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli calibrate-azure-sources 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli synthesize 19ba6d69f1b84132ba4f20599101834a --backend azure-tts --live-operation
```

Voice discovery proves availability in the configured region; names in `.env` are only candidates. Source calibration creates reviewed 20–40-second Azure WAVs whose OpenVoice source embeddings remain pending for GPU preflight. Synthesis performs bounded Azure rate search and stops for a targeted text repair if a unit cannot fit safely. None of these commands uses OmniVoice or F5-TTS.

If—and only if—synthesis writes a reviewed affected-unit timing-repair queue and a human explicitly authorises changing those approved turns, run the bounded repair and resynthesize:

```bash
python -m mathula_tv.cli repair-translation 19ba6d69f1b84132ba4f20599101834a --provider anthropic --model claude-opus-4-8 --live-operation
python -m mathula_tv.cli synthesize 19ba6d69f1b84132ba4f20599101834a --backend azure-tts --live-operation
```

This is a live Claude request and is not part of the default migration path. Review the old/new three-text forms, declared compression, protected entities, and attempt history before continuing; never resend the full clip for a unit-only timing repair. The repair artifact marks affected unit IDs, so the normal synthesis rerun regenerates those units and reuses verified unaffected WAVs.

### 3. Build references, calibrations, embeddings, and the GPU plan

```bash
python -m mathula_tv.cli build-reference-reels 19ba6d69f1b84132ba4f20599101834a --proof-of-concept --live-operation
python -m mathula_tv.cli calibrate-speaker-voices 19ba6d69f1b84132ba4f20599101834a --prepare-only --live-operation
python -m mathula_tv.cli prepare-openvoice 19ba6d69f1b84132ba4f20599101834a --prepare-only --proof-of-concept --device cuda --live-operation
python -m mathula_tv.cli queue-openvoice-assets 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli inspect 19ba6d69f1b84132ba4f20599101834a --live-operation
```

`--proof-of-concept` explicitly labels the default mixed-source reference candidates/voice-use evidence as internal-review-only. For publication-oriented evidence, replace the reference flag with `--candidates /absolute/path/to/reviewed_reference_candidates.json` and supply approved voice-rights metadata. Candidate preparation renders one reviewed sentence with every discovered allowed Azure voice. Prepare-only builds/promotes `dubbing/openvoice/asset_plan.json`; queueing claims no completed embeddings and leaves turn conversion unqueued.

Run either OpenVoice notebook in `assets` mode. In addition to all immutable runtime/checkpoint settings in [Colab](colab.md) or [Kaggle](kaggle.md), set:

```text
MATHULA_TV_JOB_ID=19ba6d69f1b84132ba4f20599101834a
MATHULA_TV_OPENVOICE_WORKER_MODE=assets
MATHULA_TV_OPENVOICE_ASSET_PLAN_SHA256=<raw plan file sha256 printed by the server>
MATHULA_TV_LIVE_OPERATION=1
MATHULA_TV_LIVE_OPERATION_ACK=I_ACKNOWLEDGE_LIVE_OPENVOICE_GCS_MUTATION
```

The worker claims `openvoice_asset_preparation`, verifies all pins/inputs, extracts source/target embeddings, converts voice candidates, optionally records injected similarity/retranscription/artefact metrics, promotes `dubbing/openvoice/asset_manifest.json`, cleans private scratch, and stops without dubbing-turn conversion. A completed GPU manifest with `pending_external_qc` candidates still cannot select a voice.

Back on the server:

```bash
python -m mathula_tv.cli reconcile-openvoice-assets 19ba6d69f1b84132ba4f20599101834a --live-operation
```

Inspect every reconciled `speakers/{speaker}/azure_voice_candidates.json`. If the injected evaluator produced completed measured metrics, persist the provisional, still-human-review-required selection with the first command below. If any candidate remains `pending_external_qc`, create the reviewed [candidate-evaluations document](openvoice.md#per-speaker-azure-voice-calibration) at the shown path and use the second command instead. Choose one, not both:

```bash
python -m mathula_tv.cli calibrate-speaker-voices 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli calibrate-speaker-voices 19ba6d69f1b84132ba4f20599101834a --evaluations /home/mokgethwa/mathula-tv/working/jobs/19ba6d69f1b84132ba4f20599101834a/review/openvoice_candidate_evaluations.json --live-operation
```

The evaluation file must name every actual speaker and allowed Azure voice exactly, contain measured values rather than invented thresholds, and record `status: approved`, a named reviewer, and a timezone-aware `reviewed_at`. It may include a reviewed `human_override` per speaker.

If and only if selection reports `azure_resynthesis_required`, inspect its affected-unit list and rerun synthesis without `--force`; request/voice identity makes the backend regenerate changed turns while reusing verified unaffected turns:

```bash
python -m mathula_tv.cli synthesize 19ba6d69f1b84132ba4f20599101834a --backend azure-tts --live-operation
```

Then build and queue the turn-conversion plan:

```bash
python -m mathula_tv.cli prepare-openvoice 19ba6d69f1b84132ba4f20599101834a --proof-of-concept --device cuda --live-operation
python -m mathula_tv.cli queue-voice-conversion 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli inspect 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli process 19ba6d69f1b84132ba4f20599101834a --live-operation
```

Asset reconciliation distrusts notebook-local paths, verifies the server plan/lease/runtime/checkpoints and every promoted hash/manifest, and keeps target embeddings job-local. Speaker calibration ranks only completed measured metrics plus any human override. If the chosen Azure voice changes, stop and rerun the affected Azure units before full preparation. The server then builds a hash-complete relative conversion plan and queues it through the existing GCS lease exchange. `process` must truthfully report that a GPU worker is required. The normal server does not run inference.

### 4. Run exactly one pinned Colab or Kaggle worker

Choose the same host guide, start a clean runtime, and set `MATHULA_TV_OPENVOICE_WORKER_MODE=conversion`, `MATHULA_TV_OPENVOICE_PLAN_SHA256=<raw conversion plan file sha256 printed by the server>`, the literal job ID, and the exact live acknowledgement. The worker claims `openvoice_conversion`, verifies every plan input/pin/hash and the reconciled embeddings/selections, converts only pending units with one model load, promotes verified outputs, completes the lease, and cleans private local assets. Stop immediately after lease loss or hash mismatch.

### 5. Reconcile and evaluate Phase A

```bash
python -m mathula_tv.cli reconcile-voice-conversion 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli evaluate-compatibility 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli align 19ba6d69f1b84132ba4f20599101834a --no-require-compatibility-pass --live-operation
python -m mathula_tv.cli build-review-package 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli inspect 19ba6d69f1b84132ba4f20599101834a --live-operation
```

The first evaluation deliberately writes a `pending` report with missing-measurement/uncalibrated-threshold warnings. The guarded bypass on the next line is allowed only to create Phase A aligned listening evidence; it writes `compatibility_evidence_only`/`production_authorized: false`, leaves the job in `compatibility_review`, and cannot authorise background, mixing, or rendering. The package then contains original reference, raw Azure, OpenVoice, and aligned assets.

Collect measured retranscription/protected-term/similarity/timing/audio observations and the required voice-rights, editorial, factual, and fluent-isiZulu decisions in the deterministic files below. Follow the exact [observation and threshold field schemas](compatibility-gate.md#observation-and-threshold-inputs): the observation array must cover every real unit and match its verified speaker/text/artifact hashes. Calibrate and version thresholds from reviewed evidence—do not invent universal values. Then rerun:

```bash
python -m mathula_tv.cli evaluate-compatibility 19ba6d69f1b84132ba4f20599101834a --observations /home/mokgethwa/mathula-tv/working/jobs/19ba6d69f1b84132ba4f20599101834a/review/compatibility_observations.json --thresholds /home/mokgethwa/mathula-tv/working/jobs/19ba6d69f1b84132ba4f20599101834a/review/compatibility_thresholds.json --live-operation
python -m mathula_tv.cli inspect 19ba6d69f1b84132ba4f20599101834a --live-operation
```

Do not continue merely because conversion or proof alignment ran. The truthful result remains `pending`, `needs_human_review`, or a specific `failed_*` until all pass conditions are present.

### 6. Only after an authorised compatibility pass, build Phase B review output

```bash
python -m mathula_tv.cli align 19ba6d69f1b84132ba4f20599101834a --force --require-compatibility-pass --live-operation
python -m mathula_tv.cli build-review-package 19ba6d69f1b84132ba4f20599101834a --force --live-operation
python -m mathula_tv.cli prepare-background 19ba6d69f1b84132ba4f20599101834a --clean-stem /home/mokgethwa/mathula-tv/working/jobs/19ba6d69f1b84132ba4f20599101834a/audio/clean_background_stem.wav --live-operation
python -m mathula_tv.cli mix 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli subtitles 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli render 19ba6d69f1b84132ba4f20599101834a --burn-subtitles --live-operation
python -m mathula_tv.cli review 19ba6d69f1b84132ba4f20599101834a --live-operation
python -m mathula_tv.cli inspect 19ba6d69f1b84132ba4f20599101834a --live-operation
```

The explicit clean-stem path is a prerequisite, not a claim that the file exists. If no publication-capable background is available, stop; for a clearly labelled temporary internal review only, substitute `--proof-of-concept` for `--clean-stem ...`. Official alignment uses measured OpenVoice durations. `render` writes `spoken_text` subtitles, a temporary review MP4, and its render manifest; the following `review` command writes JSON/Markdown QC and updates automated readiness. The current CLI does not inject a residual-speech detector/retranscriber, so it records residual QC as `not_run`/not passed; this sequence can create temporary review evidence but cannot satisfy the publication-background requirement until that reviewed integration is provided and passes. `review_ready` means ready for authorised humans, not human publication approval, and nothing uploads to YouTube.

## Opt-in live tests

Normal `pytest` must not use credentials, network, GPU, or real model assets. Enable only the exact integration under review:

```text
MATHULA_TV_RUN_LIVE_AZURE_STT_TESTS=1
MATHULA_TV_RUN_LIVE_AZURE_TTS_TESTS=1
MATHULA_TV_RUN_LIVE_ANTHROPIC_TESTS=1
MATHULA_TV_RUN_LIVE_GCS_TESTS=1
MATHULA_TV_RUN_LIVE_OPENVOICE_TESTS=1
MATHULA_TV_RUN_LIVE_RENDER_TESTS=1
MATHULA_TV_RUN_LIVE_FOUR_TURN_TESTS=1
```

The Azure STT test additionally needs a reviewed `MATHULA_TV_LIVE_AZURE_STT_AUDIO` fixture. OpenVoice conversion/assets/similarity, GCS recovery, synthetic rendering, and the complete protected-job orchestration use explicit injected `module:callable` runners:

```text
MATHULA_TV_OPENVOICE_LIVE_TEST_RUNNER=module:callable
MATHULA_TV_OPENVOICE_ASSET_LIVE_TEST_RUNNER=module:callable
MATHULA_TV_SPEAKER_SIMILARITY_LIVE_TEST_RUNNER=module:callable
MATHULA_TV_GCS_RECOVERY_LIVE_TEST_RUNNER=module:callable
MATHULA_TV_RENDER_LIVE_TEST_RUNNER=module:callable
MATHULA_TV_FOUR_TURN_LIVE_TEST_RUNNER=module:callable
```

Blank/missing runners skip their tests; the suite never invents a live runtime. These flags may incur cost or mutate external state. The complete four-turn test requires every provider/GCS/OpenVoice flag plus its own flag and runner, and it must preserve the existing translation. Never point contention/recovery tests at the protected job; use a disposable job/bucket fixture.
