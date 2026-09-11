# Undersized syllables are warning-only

## Policy
- Semantic incompleteness remains fatal and must never be committed.
- A Zulu block below its English-derived syllable target is now NON-FATAL.
- The returned wording is kept unchanged.
- Python emits a red warning with the syllable shortfall.
- Native TTS later synthesizes that wording at normal rate (`rate_percent=0`); no slowing is introduced.
- The pipeline commits the block and immediately advances to the next four-block window.
- No temporal-mask residual/retry call is made for undersized syllables.
- The four-block self-QA remains a single Grok provider call with internal review.
