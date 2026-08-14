# Azure GPT provider

Mathula TV v13.18.54 sends every runtime AI operation through Azure OpenAI's
Responses API. Translation, semantic fitting, speaker-context inference,
autocorrect entity inventory, candidate ranking, SEO, hooks, and editorial
review all use `create_production_ai_provider()`; web research uses the same
Responses API family with the native `web_search` tool.

Required configuration:

```dotenv
AZURE_AI_ENDPOINT=https://RESOURCE.services.ai.azure.com
AZURE_AI_KEY=secret
AZURE_AI_DEPLOYMENT=gpt-5.6-sol-1
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.6-sol-1
```

If both deployment variables are present they must be identical. Historical
`MATHULA_TV_AI_PROVIDER=azure-foundry-claude` values are ignored by the sole
production factory, while explicit Claude CLI selections fail closed. No GPT
failure can trigger a provider fallback.

Structured output is validated twice: first against the provider schema and
then against Mathula's source-bound application schema. Local normalizers may
repair deterministic metadata only; they cannot invent translation text.
Provider responses, token usage, prompt versions, request hashes, and validation
failures remain persisted for resume and audit.

Old `claude-*` schema identifiers may appear inside previously completed job
artifacts. They are legacy wire-format names, not evidence of a new Claude
request, and remain readable for backward compatibility.
