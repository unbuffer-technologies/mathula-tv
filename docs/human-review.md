# Mandatory human review

Automation cannot approve voice use, editorial accuracy, factual/legal risk, fluent isiZulu quality, or publication. `review_ready` means “ready to inspect,” not “approved.” The compatibility gate and publication decision require named, timestamped evidence; missing evidence remains pending.

## Voice consent and rights

Maintain a versioned job-local review artifact containing speaker ID, reference source, intended use, consent/rights status, reviewer, timezone-aware review timestamp, restrictions, expiry/re-review date, and notes. Never invent consent, reviewer identity, or a rights basis.

When rights metadata is absent, generation is allowed only with an explicit proof-of-concept/internal-review setting. Those artifacts must remain private, carry restrictions, and cannot be marked publication-approved. Speaker references/embeddings remain job-local and should be deleted from GPU scratch after verified promotion.

## Required compatibility roles

- **Voice rights:** reference source and intended identity conversion are authorised.
- **Editorial:** meaning, attribution, tone, uncertainty, claims, compression, and SEO are responsible.
- **Factual:** names, numbers, dates, currency, percentages, quotations, negation, and sensitive claims remain accurate.
- **Fluent isiZulu:** faithful/spoken wording, pronunciation, code-switching, Azure audio, and OpenVoice audio are acceptable.

Approved decisions require a reviewer and timezone-aware timestamp. `changes_requested` and `rejected` are first-class outcomes; do not coerce them into a pass.

## Full-clip review checklist

1. Read the source and all three target-text forms; verify every recorded compression/substitution.
2. Listen A/B per unit: raw Azure pronunciation, original identity, OpenVoice identity/artefacts, aligned timing.
3. Check speaker attribution and continuity, names/numbers/dates/negation, quotes/claims, and genuine overlap.
4. Listen to the complete mix for pacing, residual English, music/ambience damage, dialogue balance, loudness, clipping, and endings.
5. Verify `spoken_text` subtitles and final MP4 duration/frame/resolution.
6. Review QC warnings, provisional thresholds, failures/fallbacks, consent restrictions, and artifact hashes.
7. Record approval/restrictions or request targeted changes. Re-run only affected versioned units/stages.

A compatibility pass only validates the Azure-to-OpenVoice path; it does not approve a particular video. Publication review also requires a publication-capable background with residual-English validation, final render/QC, and all appropriate legal/editorial decisions. Mathula TV performs no YouTube upload.
