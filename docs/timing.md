# Timing and overlap

Precise timing means accurate unit start/end alignment, preserved pauses/overlaps, controlled endings, and final audio/video duration agreement. It does not mean phoneme-level lip synchronisation.

## Dubbing-unit windows

The deterministic unit builder merges short same-speaker phrases and splits long phrases at punctuation/word boundaries while preserving source phrase/word IDs. It never merges across speaker changes, hard overlap boundaries, or configured long pauses. Each unit records source start, preferred end/duration, hard end/maximum duration, surrounding silence, overlap group, and manual overrides.

Preferred end is a quality target; hard end is a no-overrun boundary. A naturally shorter utterance is valid and is padded later rather than slowed just to fill silence.

## Before conversion

Azure TTS performs a bounded rate search from a neutral candidate. The initial quality bounds are `-12%` to `+15%`, at most three searches, and a 120-ms preferred-end tolerance. These are project quality policies, not platform limits. A unit that cannot fit its hard window at reasonable delivery triggers a bounded targeted translation repair; critical content may not disappear silently.

## After conversion

Measured OpenVoice duration is authoritative. For every converted unit:

1. measure validated converted audio;
2. compare it with preferred and hard ends;
3. record the actual source/converted duration ratio for QC; the current pipeline measures each output directly and does not yet consume a persisted per-speaker predictor;
4. apply only a verified pitch-preserving correction within `0.94`–`1.06` speed;
5. pad remaining time with silence;
6. reject excessive correction and request a targeted repair;
7. rerun only affected units and record every decision.

Use a verified FFmpeg `atempo` chain or equivalent pitch-preserving implementation. Never truncate spoken words, change sample rate merely to alter duration, or conceal an overrun. Initial final timing tolerance is 120 ms.

## Overlap

Place each unit at its original source start on a separate per-speaker track. Preserve genuine cross-speaker overlap, record the overlap decisions, apply controlled fades/gain, and peak-protect the sum. Do not assign words to a different speaker or serialise dialogue just to simplify rendering. Severe/ambiguous overlap is a human-review flag.

Each alignment result records unit/speaker identity, source/corrected/final durations, preferred/hard durations, correction speed/method, silence padding, final timing error, aligned output path/SHA-256, and warnings. The enclosing manifest records source backend, readiness, `production_authorized`, and the gate state; later dialogue-bus output records overlap decisions. The compatibility gate must permit normal production alignment. The only exception is the explicit `--no-require-compatibility-pass` Phase A evidence alignment described in [compatibility-gate.md](compatibility-gate.md); it is labelled `compatibility_evidence_only`, leaves the job in compatibility review, cannot authorise downstream production stages, and must be replaced by a guarded alignment after a pass.
