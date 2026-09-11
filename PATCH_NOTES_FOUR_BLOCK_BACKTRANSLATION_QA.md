# Four-block semantic window + internal back-translation QA

## What changed

- Keeps the four-block semantic/comprehension window contract.
- Keeps one joint candidate column across the whole window; candidate `a` is one coherent four-block Zulu combination.
- Grok must internally back-translate the complete four-block Zulu combination to English and compare it with the complete four-block English window before returning it.
- The internal QA explicitly checks names, numbers, dates, negation, uncertainty, speaker attribution, references, questions/answers, self-corrections, chronology and argument progression.
- Long English blocks must not be summarized.
- Natural isiZulu expansion is allowed through context-grounded tautology, pleonasm, periphrasis, natural repetition, discourse markers and fuller grammatical realization.
- Meaningless filler and invented facts remain forbidden.

## Syllable contract

- The English block's estimated syllable count is the linguistic floor.
- Zulu minimum = English syllable count.
- Preferred target = English syllable count + 1.
- There is no hard maximum syllable count.
- Azure milliseconds, source duration, TTS overhead and speaking-rate targets are not exposed to Grok for language composition.
- Azure remains the downstream physical measurement authority.

## Window boundaries

- Normal window: current block + up to the next three blocks.
- A two- or three-block suffix is used as-is.
- A final one-block suffix is paired with exactly one already-committed predecessor.
- No artificial padding to four blocks.

## Timing policy retained

- `+12%` remains a red console quality-warning threshold, not a hard pipeline failure.
- Semantically complete overlong Zulu is committed and can be rendered rushed.
- Undersized Zulu remains the important language-side failure because the pipeline does not slow speech to solve it.
