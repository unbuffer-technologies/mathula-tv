# Azure Text-to-Speech

Azure TTS is the production isiZulu base-speech backend. It determines pronunciation and base delivery before OpenVoice transfers job-local speaker identity. Configured voices are candidates, not proof of regional availability, so a live discovery/validation stage is mandatory.

No live Azure TTS success is claimed by this guide.

## Configuration

Azure TTS shares `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` with STT.

```dotenv
MATHULA_TV_AZURE_TTS_VOICES=zu-ZA-ThandoNeural,zu-ZA-ThembaNeural
MATHULA_TV_AZURE_TTS_DEFAULT_VOICE=zu-ZA-ThandoNeural
MATHULA_TV_AZURE_TTS_SAMPLE_RATE=24000
MATHULA_TV_AZURE_TTS_CHANNELS=1
MATHULA_TV_AZURE_TTS_SAMPLE_WIDTH=2
MATHULA_TV_AZURE_TTS_TIMEOUT_SECONDS=120
MATHULA_TV_AZURE_TTS_MAX_RETRIES=3
MATHULA_TV_AZURE_RATE_MIN_PERCENT=-12
MATHULA_TV_AZURE_RATE_MAX_PERCENT=15
MATHULA_TV_AZURE_RATE_SEARCH_ATTEMPTS=3
MATHULA_TV_PREFERRED_END_TOLERANCE_MS=120
```

Output is canonical mono 24-kHz PCM16 WAV unless a reviewed pipeline version changes the format. The backend writes atomically, validates the WAV, records duration/SHA-256/input and SSML hashes, classifies cancellation/retry errors, and reuses verified output idempotently. Forced pipeline stages preserve job-level attempt history; regenerated per-output metadata replaces that output's prior sidecar.

## Safe SSML

Claude may propose delivery hints but never supplies executable SSML. Application code escapes all text and constructs only reviewed allowlisted elements such as voice, bounded prosody, bounded break, and verified emphasis/substitution. It rejects unknown elements/namespaces, arbitrary XML, external audio/URLs, and out-of-range rate, pitch, volume, or pauses.

Store the deterministic SSML artifact and SHA-256 plus a safe human-readable summary. See [pronunciation.md](pronunciation.md) for substitution rules.

## Timing search

For each dubbing unit:

1. Synthesize a neutral or near-neutral candidate.
2. Trim only accidental leading/trailing digital silence and measure the actual result.
3. Calculate a candidate rate and perform at most the configured bounded interpolation/binary-style attempts.
4. Accept naturally shorter speech and pad later; do not slow it merely to consume silence.
5. Reject speech beyond the hard timing window.
6. Request a bounded Claude unit repair when safe Azure rates cannot fit.

Rate bounds are Mathula quality policy, not Azure platform limits. The timing-search manifest records each synthesis candidate's voice, rate, duration, hashes, warnings, and whether repair is required. Terminal transport/cancellation failures remain classified and secret-safe; individual internal transport-retry errors are not represented as separate successful candidate records.

## Per-speaker base-voice selection

Do not permanently map speaker traits to one Azure voice with a simple rule. Once per job-local speaker, render the same reviewed calibration sentence through every discovered allowed isiZulu voice, convert each with the speaker's target embedding, then compare intelligibility, similarity, artefacts, and duration drift. Store all candidates, the selected voice, and any human override in the speaker manifest. Reuse that selection per turn.

## Azure source calibration for OpenVoice

Each allowed Azure voice needs one stable 20–40-second reviewed calibration WAV and cached source embedding. Cache identity includes voice, region, output format, calibration-text version/hash, WAV hash, OpenVoice revision, checkpoint hashes, embedding-manifest canonical hash, and embedding hash. A changed voice/format/text/converter revision invalidates the cache; a partial embedding/manifest pair fails closed. Do not extract a new source embedding from every short unit.

Raw Azure TTS may be packaged only as an explicitly labelled `azure_tts_only_review` internal fallback. It is not OpenVoice output and does not prove speaker identity.
