# Mathula TV v10.2 — Azure Foundry Claude streaming translation

## Scope

This patch keeps the v10.1 architecture:

- one complete authoritative English transcript;
- one source-bound translation request;
- one Claude call through Azure AI Foundry;
- natural, concise, and compact variants returned together;
- zero automatic provider retries in the translation command;
- zero automatic structured-output repair calls.

It changes only the Foundry response transport from buffered JSON to server-sent
events (SSE).

## Streaming behavior

- Adds `stream: true` to the Foundry Messages request.
- Sends `Accept: text/event-stream`.
- Reconstructs the final Claude Messages envelope from:
  - `message_start`;
  - `content_block_start`;
  - `content_block_delta`;
  - `content_block_stop`;
  - `message_delta`;
  - `message_stop`.
- Accepts `ping` events and unknown future event types without failing.
- Handles streamed text, thinking, signatures, citations, and partial tool JSON.
- Does not persist or print streamed thinking text or signatures.
- Treats the configured timeout as a socket read-idle timeout rather than a
  total generation deadline. Each SSE event resets the underlying read wait.
- Never automatically retries after a stream has opened, because that could
  duplicate a paid full-transcript generation.

## Job artifacts

Each translation run writes:

- `provider_stream_events.jsonl` — sanitized stream events;
- `translation_response.partial.txt` — incrementally received response text;
- `provider_stream_progress.json` — latest safe progress counters;
- normal final response, validation, installation, status, and failure files.

Terminal progress reports response characters and output tokens but never prints
translated text.

## Validation

Focused suites passed:

- Foundry SSE transport and timeout handling;
- exact manual-export system/user prompt parity;
- one full-transcript provider request;
- no automatic repair or retry call;
- manual translation import;
- multivariant validation;
- CLI translation output;
- direct Azure multivariant selection.

33 focused regression tests passed, plus 25 provider/CLI transport tests.
