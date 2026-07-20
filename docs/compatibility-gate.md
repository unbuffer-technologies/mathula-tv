# Azure TTS → OpenVoice compatibility gate

Phase A asks whether Azure isiZulu speech can pass through OpenVoice while retaining pronunciation, intelligibility, protected content, job-local speaker identity, audio quality, and timing stability. The current gate is `pending`; this repository does not claim a live pass.

## Per-unit evidence

Compare four assets: the intended `spoken_text`, raw Azure TTS, OpenVoice output, and the original same-speaker reference (plus final aligned audio when available). Record:

- Azure/OpenVoice duration, duration ratio, and final timing error;
- Azure and OpenVoice `zu-ZA` retranscriptions and word/character-error approximation;
- protected-name, number, date, negation, and acronym preservation;
- reference-vs-OpenVoice and reference-vs-raw-Azure similarity;
- backend/version/input hashes, calibration and threshold versions;
- artefact warnings, clipping, silence ratio, and other WAV quality;
- voice-rights, editorial, factual, and fluent-isiZulu human-review evidence.

Retranscription is a QC signal, not absolute linguistic truth. Speaker similarity has no universal threshold; a score without a calibrated, versioned threshold remains provisional.

## Observation and threshold inputs

`--observations` consumes a JSON array with exactly one object for every actual dubbing-plan unit—no omissions or extras. The server binds each unit/speaker, exact `spoken_text`, and reference/Azure/OpenVoice/aligned SHA-256 to local verified artifacts before it evaluates any score. This is a structural template only; replace every placeholder/null with evidence from the actual unit. Placeholder hashes or copied measurements are rejected and must never be used to force a pass.

```json
[
  {
    "unit_id": "unit_0001",
    "speaker_id": "SPEAKER_00",
    "intended_spoken_text": "EXACT APPROVED spoken_text",
    "reference_audio_sha256": null,
    "azure_audio_sha256": null,
    "openvoice_audio_sha256": null,
    "aligned_audio_sha256": null,
    "azure_duration_ms": null,
    "openvoice_duration_ms": null,
    "final_timing_error_ms": null,
    "azure_retranscription": null,
    "openvoice_retranscription": null,
    "azure_word_error_rate": null,
    "openvoice_word_error_rate": null,
    "protected_name_accuracy": null,
    "number_accuracy": null,
    "date_accuracy": null,
    "negation_accuracy": null,
    "speaker_similarity": null,
    "azure_speaker_similarity": null,
    "audio_quality_passed": null,
    "clipping_ratio": null,
    "silence_ratio": null,
    "protected_name_count": 0,
    "number_count": 0,
    "date_count": 0,
    "negation_count": 0,
    "audio_artifact_warnings": [],
    "human_reviews": {
      "voice_rights": {
        "status": "approved",
        "reviewer": "named-reviewer",
        "reviewed_at": "2026-07-20T12:00:00+00:00"
      },
      "editorial": {
        "status": "approved",
        "reviewer": "named-reviewer",
        "reviewed_at": "2026-07-20T12:00:00+00:00"
      },
      "factual": {
        "status": "approved",
        "reviewer": "named-reviewer",
        "reviewed_at": "2026-07-20T12:00:00+00:00"
      },
      "isizulu_language": {
        "status": "approved",
        "reviewer": "named-reviewer",
        "reviewed_at": "2026-07-20T12:00:00+00:00"
      }
    }
  }
]
```

Approved human decisions require real reviewer names and timezone-aware timestamps. Other valid statuses are `pending`, `changes_requested`, and `rejected`. Accuracy, clipping, and silence ratios are from 0 through 1; similarities are from -1 through 1; error rates are non-negative; durations are positive integer milliseconds. A protected accuracy is applicable when its corresponding count is non-zero.

`--thresholds` consumes one JSON object with all fields below. Start with `calibrated: false` and `null` values while collecting evidence. Set `calibrated: true` only after authorised calibration and replace every `null`; neither this template nor the numbers in other guides are universal thresholds.

```json
{
  "version": "project-calibration-version",
  "calibrated": false,
  "max_openvoice_word_error_rate": null,
  "max_word_error_rate_degradation": null,
  "min_protected_name_accuracy": null,
  "min_number_accuracy": null,
  "min_date_accuracy": null,
  "min_negation_accuracy": null,
  "min_speaker_similarity": null,
  "max_abs_timing_error_ms": null,
  "max_clipping_ratio": null,
  "min_silence_ratio": null,
  "max_silence_ratio": null
}
```

The evaluator rejects incomplete/un-calibrated thresholds as a pass condition, inverted silence bounds, invalid ranges, duplicate/missing units, and any artifact-provenance mismatch.

## States

```text
pending
passed
failed_pronunciation
failed_intelligibility
failed_identity
failed_timing
failed_audio_quality
needs_human_review
```

Missing measurements or thresholds keep the gate pending. Uncalibrated thresholds and incomplete human decisions cannot produce `passed`. Known failures use the most relevant `failed_*` state and identify affected units. A gate pass requires all configured machine checks plus documented approval by all required human roles; inference success alone is irrelevant.

Normal Phase B advancement is blocked for `pending`, `needs_human_review`, and every `failed_*`. `--skip-openvoice` produces `azure_tts_only_review` and cannot satisfy the gate.

## Truthful Phase A ordering

Aligned audio is itself required listening/timing evidence, so the proof workflow is deliberately two-step:

1. Reconcile verified OpenVoice outputs and run an initial evaluation with no invented observations/thresholds; it writes a truthful `pending` report.
2. Run `align --no-require-compatibility-pass` exactly once as an explicitly labelled Phase A evidence operation. Its manifest records `readiness: compatibility_evidence_only` and `production_authorized: false`, and the job stays in `compatibility_review`. This bypass does not mean the gate passed and cannot authorise background, mixing, or rendering.
3. Build the A/B package, collect machine measurements and all required human decisions, and write versioned observation/threshold inputs.
4. Rerun evaluation with those explicit inputs. Only `passed` permits normal production continuation.
5. Regenerate official alignment with `--force --require-compatibility-pass`. Only an OpenVoice alignment manifest with `readiness: compatibility_passed` and `production_authorized: true`, together with the passed job gate, can advance to `alignment_ready` and authorise downstream work.

This ordering avoids both false live claims and the circular requirement to approve timing before anyone can hear aligned evidence.

## A/B review package

```text
review/
  original_reference/{unit_id}.wav
  azure_tts/{unit_id}.wav
  openvoice/{unit_id}.wav
  aligned/{unit_id}.wav
  comparisons/{unit_id}.json
  compatibility_report.json
  compatibility_report.md
  review_manifest.json
  listening_guide.md
```

Each comparison links hashes, intended spoken text, timing, retranscriptions, protected-item results, similarity, quality warnings, and human decisions. The package contains no credentials, signed URLs, embeddings, or unnecessary private configuration.

## Human listening order

For each unit, first read `spoken_text`; listen to raw Azure for pronunciation/intelligibility; listen to the original reference for identity; listen to OpenVoice for identity transfer and new artefacts; then listen to aligned output for timing damage. Review names, numbers/dates, negation, attribution, and overlap in context. Record changes requested rather than forcing a pass.

After every unit passes, listen to the complete clip for speaker continuity, overlap, pacing, residual English, ambience, loudness, and editorial meaning. A compatibility pass permits production automation; it is not consent or publication approval.
