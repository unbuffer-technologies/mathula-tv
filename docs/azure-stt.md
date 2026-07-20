# Azure Speech-to-Text

Azure Speech-to-Text is the production transcription and diarization authority. Production requests use English South African locale `en-ZA`; all returned speaker labels are job-local and must never become persistent global identities.

## Input and limits

Media preparation preserves:

```text
working/jobs/{job_id}/audio/analysis_mono.wav
working/jobs/{job_id}/audio/mix_source_stereo.wav
```

Only `analysis_mono.wav` goes to Azure STT. The original upload and stereo derivative remain available for reference selection, separation, ambience, and final mixing. Each derivative records sample rate, channels, duration, ffprobe metadata, and SHA-256 and is written atomically.

The initial production limit is 240 minutes per source file. Files longer than that are rejected clearly until reviewed chunking and cross-chunk speaker stitching exist. Do not claim stable speaker identity across independently transcribed chunks.

Eligible files use Fast Transcription. An injectable router/batch boundary exists for longer or otherwise ineligible Fast inputs, but the current production CLI constructs the Fast backend only; a reviewed deployment must explicitly wire batch before relying on it. The server uploads the private local audio directly; it does not need to expose a public or signed URL. Provider limits still apply in addition to the Mathula policy.

## Output and authority

Persist the untouched raw provider response and a separate normalized artifact containing phrase timestamps, word timestamps, confidence/provenance where returned, text, and speaker labels. Hash and mirror both artifacts idempotently before entering `analysis_ready`.

Dubbing-unit construction consumes normalized Azure speakers. Optional Pyannote may detect overlap, compare diarization, or produce evaluation evidence, but it cannot silently replace Azure speaker labels. Record disagreements in QC and keep the original provider artifacts intact.

## Reliability and privacy

The boundary classifies authentication, rate-limit, timeout, malformed-response, and provider errors; supports bounded retries, forced retry, injected HTTP for offline tests, local hash checks, GCS generation preconditions, and remote hash verification. Reuse a verified result on normal reruns. A forced rerun creates attempt history rather than erasing prior evidence.

Never record subscription keys, authorization headers, cookies, signed URLs, or raw exception bodies that may contain secrets. Redact configuration snapshots and logs.

## Configuration

```dotenv
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_SPEECH_ENDPOINT=
AZURE_SPEECH_API_VERSION=2025-10-15
AZURE_SPEECH_LOCALE=en-ZA
AZURE_SPEECH_MAX_SPEAKERS=10
AZURE_SPEECH_TIMEOUT_SECONDS=600
MATHULA_TV_MAX_SOURCE_DURATION_MINUTES=240
```

`AZURE_SPEECH_ENDPOINT` is optional when a valid region can derive the endpoint. Do not put credentials in the endpoint. Live validation is opt-in with `MATHULA_TV_RUN_LIVE_AZURE_STT_TESTS=1`; normal tests inject responses and use no network.

No new live Azure STT run is claimed by this guide.
