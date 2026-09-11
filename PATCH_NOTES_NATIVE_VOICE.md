# Native voice resolution cumulative fix — 2026-08-20

This overlay supersedes the earlier native auto-voice, legacy-classifier, and classifier-required voice overlays.

## Resolution order
1. Trusted current-job voice_resolution.json
2. SpeechBrain known-speaker identity cache
3. Historical mapping by stable identified person name
4. Legacy contextual speaker identity/family evidence
5. Legacy local voice-family classifier (JaesungHuh/voice-gender-classifier)
6. Existing speaker profile evidence
7. Grok only for genuinely unresolved speakers

## Fixes in this revision
- Detects per-speaker classifier failures recorded in `acoustic_analysis.json`/the legacy helper `failures` map. Missing local model/dependencies now stop before Grok and print the actual speaker failure.
- Canonical Pass-1 segment timings are the primary speaker-reel source. Voice classification no longer depends on Azure exposing one particular legacy diarization `turns` shape.
- Logs reel source counts and per-speaker classifier failures/probabilities.
- Keeps native Azure TTS rate hard-locked to 0%.
- Keeps existing/manual/current-job mappings authoritative and never reuses diarization IDs across jobs as person identity.
