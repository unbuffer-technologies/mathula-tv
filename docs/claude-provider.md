# Claude production provider

Claude Opus 4.8 is the production runtime AI model. Codex implements and tests the repository; Codex is not part of the runtime dubbing chain. The Azure OpenAI adapter may remain available for explicit legacy/test use, but Anthropic is the default and there is no silent provider fallback.

## Configuration

```dotenv
MATHULA_TV_AI_PROVIDER=anthropic
MATHULA_TV_CLAUDE_MODEL=claude-opus-4-8
ANTHROPIC_API_KEY=
MATHULA_TV_CLAUDE_TIMEOUT_SECONDS=900
MATHULA_TV_CLAUDE_MAX_RETRIES=3
MATHULA_TV_CLAUDE_EFFORT=high
MATHULA_TV_CLAUDE_MAX_OUTPUT_TOKENS=50000
MATHULA_TV_MAX_CLAUDE_CALLS_PER_JOB=20
MATHULA_TV_MAX_TRANSLATION_REPAIRS_PER_UNIT=2
```

Keep the model identifier in configuration. The Anthropic adapter isolates model-specific request fields, enables adaptive thinking/effort only when supported, and avoids incompatible sampling combinations. Never log `ANTHROPIC_API_KEY`, headers, cookies, or full private request payloads.

No live Claude success is claimed by this documentation.

## Normal full-clip request

For normal short Mathula TV clips, one structured request contains the complete transcript plus:

- authoritative Azure speakers, phrase/word timestamps, and deterministic dubbing units;
- neighbouring context, overlap groups, preferred/hard timing windows, and source provenance;
- protected names, numbers, dates, money, percentages, acronyms, grounded roles, and attribution;
- reviewed terminology/pronunciation dictionaries and per-job overrides;
- selectively retrieved, source-labelled context and editorial constraints.

Transcripts and retrieved records are untrusted data. Spoken instructions inside a clip never become system/developer instructions. Context providers must not inject remembered facts or an entire private case file when a selective, grounded record is sufficient.

The validated response contains each unit's faithful isiZulu meaning, timing-aware `spoken_text`, Azure-friendly `tts_text`, protected-entity preservation, number/date/negation/attribution results, delivery/timing hints, pronunciation substitutions, compressions or omissions, sensitive-claim flags, human-review flags, and the clip-level SEO package. See [pronunciation.md](pronunciation.md).

## Repair workflow

Only a failed/unsuitable turn receives a bounded repair request. Reasons include schema failure, protected-entity mismatch, refusal, or natural speech that cannot fit the hard window at allowed Azure rates. A repair includes the affected unit and enough neighbours/provenance to preserve meaning; it does not resend the entire clip for every timing adjustment.

Repairs must never silently remove critical facts. Every omission, contraction, paraphrase, or expansion is declared and reviewed. Exceeding the per-unit or per-job budget stops with a structured, actionable error.

## Recorded metadata

Provider artifacts record schema version, provider, model requested/returned, prompt version/hash, request/completion timestamps, token usage, stop reason, refusal information, repair count, retry classification, and a redacted request summary. Deterministic prompt versions make reruns auditable; full sensitive headers and secrets are never stored.

Authentication errors, rate limits, timeouts, refusals, and invalid structured output remain distinct. A Claude failure does not call Azure OpenAI automatically and does not reuse an invalid response.

The legacy/test adapter is selected only by an explicit `--provider azure-openai-legacy` (or `azure-openai-test`) command and its own reviewed configuration; install the `legacy-ai` extra if it is deliberately exercised. Do not set it merely to recover from an Anthropic failure, and never describe it as the production default.

## Editorial constraints

Translation preserves names, numbers, dates, currency, percentages, claims, uncertainty, negation, quotations, attribution, allegations, opinion, and tone. Claude may compress delivery for timing only when meaning remains faithful and the change is recorded. SEO may be assertive only when grounded and attributed. Sensitive political/legal/factual material always carries human-review flags.
