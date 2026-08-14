# GPT-only provider release v13.18.52

This cumulative source overlay moves the v13.18.52 production AI boundary to
Azure OpenAI GPT 5.6 Sol through the Azure Responses API. It includes all
performance-plan, word-timed lip-sync, pronunciation, overlap, and review
normalization changes through v13.18.51.

## Release boundary

- v13.18.52 production calls are GPT-only.
- There is no runtime Claude fallback.
- The CLI rejects an explicit Claude provider selection.
- A stale `MATHULA_TV_AI_PROVIDER=azure-foundry-claude` value is ignored by the
  GPT-only factory so an existing `.env` cannot reroute or block this release.
- v13.18.51 and earlier source archives are unchanged and remain the Claude
  release line.

## Required environment

```dotenv
AZURE_AI_ENDPOINT=https://YOUR-RESOURCE.services.ai.azure.com
AZURE_AI_KEY=YOUR_SECRET
AZURE_AI_DEPLOYMENT=gpt-5.6-sol-1
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.6-sol-1
```

If both deployment variables are present they must be identical. The API key
is sent only in Azure's `api-key` header and is never included in metadata or
safe request summaries.

## Provider behavior

- Endpoint: `{AZURE_AI_ENDPOINT}/openai/v1/responses`
- Transport: non-streaming Responses API for deterministic structured parsing
- Storage: `store=false`
- Default reasoning effort: high; semantic-fit batches explicitly use medium
- Retry policy: bounded retry of timeouts, connection failures, HTTP 429, and
  transient 5xx failures
- Output policy: Azure strict JSON Schema where the schema stays within Azure's
  100-property/five-level limits; JSON-object mode plus unchanged local schema
  and semantic validation for deeper performance-plan candidate envelopes
- Optional fields are nullable only in the provider wire schema and are
  normalized before the unchanged application schema validates them
- Incomplete, refused, malformed, rate-limited, and unauthorized responses fail
  closed with GPT-specific diagnostics

Historical `claude-*` response schema identifiers remain readable because they
are persisted data-contract names, not provider routing. New prompt and
checkpoint strategy identifiers are GPT/version-specific.

## Verification

The cumulative overlay passes 84 offline tests in a reconstructed full source
tree. Coverage includes the Responses URL and headers, strict schema payload,
nullable optional fields, Azure schema-depth fallback, cached/reasoning token
telemetry, incomplete output, HTTP 429, conflicting deployments, stale Claude
environment values, explicit CLI Claude rejection, and all 77 v13.18.51
lip-sync/performance regressions.
