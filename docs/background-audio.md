# Background audio, overlap, and mixing

Publication-quality dubbing must remove or sufficiently suppress the original English dialogue while preserving music and ambience. Ducking lowers dialogue; it does not remove it.

## Background source priority

```text
clean_background_stem
separated_background
reconstructed_ambience
original_mix_ducked_review_only
```

1. Prefer a supplied, validated clean background stem.
2. Otherwise use a validated speech-reduced result from the configured separation-provider interface or a verified local cache.
3. Otherwise reconstruct ambience from suitable non-speech regions and record the method/limits.
4. Only for explicit internal review, envelope-duck the original mix and label it `original_mix_ducked_review_only`.

The separation boundary supports external stems, an explicitly configured provider, deterministic cached output, validation/hashing, provider revision metadata, and explicit failure. The normal server does not silently install or select a large unverified separation model.

## Residual-English QC

The publication workflow must run speech detection on every background candidate. The pipeline accepts an injected detector/retranscriber boundary, but the current CLI does not configure one and therefore records `not_run`/not passed. When a reviewed integration is supplied and speech is detected, retranscribe/compare recognised English words with the source transcript and record detected segment count, matched terms, residual-speech duration/confidence, threshold/version, and warnings.

Substantial residual source dialogue yields `needs_background_review` and blocks the publication-review path. A clean/separated/reconstructed label alone is not proof; the candidate also needs a passing residual-dialogue result. Retranscription is a QC signal and human listening remains required.

## Dialogue and mix buses

Build separate dialogue tracks by speaker at source start times. Cross-speaker overlaps remain simultaneous on those separate tracks, with recorded overlap decisions, controlled gain/fades, and peak limiting; do not serialize or reassign overlapping words. Mix through:

- dialogue bus;
- background/ambience bus, which retains approved music already present in the chosen background.

Support per-turn fades, overlap-safe summing, dialogue-keyed side-chain ducking, peak protection, integrated loudness, loudness range, true peak, clipping count, and dialogue-to-background balance. Initial targets are `-14 LUFS`, `-1.5 dBTP`, and `11 LU LRA`; they are review targets, not substitutes for listening. Peak normalization alone is insufficient.

The background and mix manifests record source/output hashes, mode/provider, residual QC, per-speaker tracks, overlap decisions, background gain/side-chain policy, loudness/peak/clipping metrics, duration, and review warnings.
