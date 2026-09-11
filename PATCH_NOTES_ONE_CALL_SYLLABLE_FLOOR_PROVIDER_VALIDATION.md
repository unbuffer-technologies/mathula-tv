# One-call temporal-mask syllable-floor provider validation

## Problem

A one-call temporal-mask response could be structurally valid yet still return a Zulu block below its deterministic syllable floor. Python correctly detected this, but the design then terminated because Python is forbidden from launching a residual repair call.

Observed example:

- native_phrase_0010
- estimated syllables: 182
- required minimum: 191

## Fix

The temporal-mask provider response schema now carries a per-group syllable floor contract.

For every requested group:

- `estimated_syllables` must be at least that group's deterministic minimum.
- `spoken_text` receives a conservative character-count floor derived from the required syllables.
- `group_id` remains an exact immutable enum.

The schema uses per-group `anyOf` variants so each returned group is constrained by its own floor while keeping the provider response as one logical request.

Python still performs the final deterministic syllable calculation and will not trust the model-reported count. The provider-side schema exists to force an internal structured-output correction before the logical call is returned.

## Invariant

Python temporal-mask Pass 2 still makes exactly one logical provider call per chronological window. There is no Python residual repair round.
