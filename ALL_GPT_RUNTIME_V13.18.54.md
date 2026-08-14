# Mathula TV v13.18.54 — all runtime AI on Azure GPT

This cumulative source overlay completes the provider migration started in
v13.18.52. Every active runtime AI operation now uses Azure OpenAI GPT through
the Responses API:

- semantic-chunk translation and targeted translation repair;
- semantic-fit generation and independent review;
- transcript semantic analysis and contextual voice-family inference;
- autocorrect entity inventory and constrained candidate ranking;
- cited name research through native Azure GPT web search;
- SEO, hook selection, editorial packaging, and TikTok editing.

There is one production factory and one provider identity:
`azure-openai-gpt`. A stale `MATHULA_TV_AI_PROVIDER` value cannot reroute the
factory. Explicit `anthropic` or `azure-foundry-claude` CLI selections fail
closed. The retired five-call experiment remains at its old filename only as a
compatibility notice and starts no model request.

The standalone autocorrect command no longer posts to Anthropic. Translation no
longer rejects GPT. New job counters, progress messages, error codes, settings
snapshots, repair artifacts, and QC reports use provider-neutral or GPT names.
GPT incomplete responses now raise `AIInvalidStructuredOutput`, and rate limits
raise `AIRateLimit`, so diagnostics no longer incorrectly identify Claude.

Backward compatibility is deliberate:

- old `claude-*` response schema identifiers remain readable;
- old attempt counters contribute to current call and repair budgets;
- old pronunciation-source labels are removed when a GPT result supersedes
  them;
- historical exception names remain importable for cached jobs and external
  test doubles;
- the legacy streamed-response recovery tool performs no model call.

Required production variables:

```dotenv
AZURE_AI_ENDPOINT=https://RESOURCE.services.ai.azure.com
AZURE_AI_KEY=secret
AZURE_AI_DEPLOYMENT=gpt-5.6-sol-1
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.6-sol-1
```

If both deployment variables are set, they must agree. Claude credentials and
endpoints are not read by any production factory.

Install without changing metadata on an existing repository directory:

```bash
cd ~/mathula-tv
tar --no-overwrite-dir -xzf ~/mathula-all-gpt-v13.18.54-source.tar.gz -C .
source .venv/bin/activate
pip install -e .
```

The `--no-overwrite-dir` flag avoids the harmless root-directory `utime` and
mode errors seen when extracting earlier overlays into a protected checkout.

Offline verification for this release covers 143 tests: GPT configuration and
transport, every runtime operation family, explicit Claude rejection, legacy
artifact compatibility, output budgets, rate-limit resume, voice cache,
pronunciation, semantic fit, cadence, mastering, and transcript lip-sync.
