# Manual ChatGPT AI override

This mode keeps Mathula TV's deterministic pipeline, Azure Speech/TTS, timing, checkpoints, normalizers, JSON Schema validation, and application validators intact while replacing paid GPT network calls with a resumable human-mediated ChatGPT handoff.

## Start translation in manual mode

```bash
python -m mathula_tv.cli translate "$JOB_ID" \
  --provider manual-chatgpt \
  --live-operation
```

For direct Azure dubbing, including grammar/timing/publication AI calls:

```bash
python -m mathula_tv.cli dub-azure "$JOB_ID" \
  --ai-provider manual-chatgpt \
  --live-operation
```

You can also set `MATHULA_TV_AI_PROVIDER=manual-chatgpt` for commands that construct the production provider internally.

## Handoff loop

When an AI result is needed, the command writes a deterministic request under:

```text
working/jobs/<JOB_ID>/manual_ai/requests/
```

and exits with `Manual ChatGPT response required`. List outstanding requests with:

```bash
python -m mathula_tv.cli manual-ai "$JOB_ID" pending
```

Upload the request JSON to ChatGPT. ChatGPT should return a JSON file with this envelope:

```json
{
  "schema_version": "mathula-manual-ai-response-v1",
  "request_id": "<exact request_id from request>",
  "result": {}
}
```

`result` must match `provider_response_schema` from the request. Install the returned file:

```bash
python -m mathula_tv.cli manual-ai "$JOB_ID" install /path/to/manual-response.json
```

Then rerun the exact original `translate` or `dub-azure` command. Mathula validates the response with the provider wire schema, application normalizer, final output schema, and existing request validator before accepting it. A wrong request id or invalid result fails closed and does not overwrite the request.

The accepted audit artifact is stored under `manual_ai/accepted/`. Manual completions report zero API token usage and provider `manual-chatgpt`; they do not require Azure OpenAI credentials.
