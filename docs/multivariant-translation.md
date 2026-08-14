# Multivariant translation and manual override

Mathula TV translates each deterministic English dubbing unit once into three
reviewable target-language deliveries:

1. `natural` — the richest natural translation.
2. `concise` — materially shorter, with every required fact preserved.
3. `compact` — the shortest fluent, truthful version.

Azure TTS measures prepared deliveries. In `semantic-fit` mode, GPT proposes
source-bound contractions only for measured misses; the server validates facts,
entities, cadence, and the exact Azure duration before accepting a candidate.

## Normal cloud translation

```bash
python -m mathula_tv.cli translate "$JOB_ID" --live-operation
```

The translation stage uses resumable semantic chunks and writes:

- `translation/transcript_<language>.json` — source-bound multivariant artifact;
- `dubbing/translation_<language>.json` — compatibility artifact retaining all variants;
- `translation/cloud_runs/<timestamp>/` — request, response, provider metadata,
  validation, backups, and installation manifest.

All three variants require human review. Application code, not the language
model, derives `tts_text` and SSML.

## Export a package for manual or external review

Build deterministic dubbing units first when they are not already present:

```bash
python -m mathula_tv.cli build-dubbing-units "$JOB_ID" --live-operation
```

Export the source-bound prompt package:

```bash
python -m mathula_tv.cli export-translation-package "$JOB_ID"
```

The package contains:

- `translation_request.json`
- `system_prompt.txt`
- `user_prompt.txt`
- `response_schema.json`
- `package_manifest.json`

The request is bound to the authoritative English transcript and exact timing
units. Any transcript or timing change makes the old response stale.

## Validate and install a manual response

Save the model's JSON response beside `translation_request.json`, or pass the
request explicitly.

Validate without changing the job:

```bash
python -m mathula_tv.cli import-translation "$JOB_ID" translated.json \
  --request translation_request.json \
  --reviewed-by "Reviewer Name" \
  --dry-run
```

Install after review:

```bash
python -m mathula_tv.cli import-translation "$JOB_ID" translated.json \
  --request translation_request.json \
  --reviewed-by "Reviewer Name" \
  --live-operation
```

Every installation is versioned under
`translation/manual_overrides/<timestamp>/`. Existing authoritative artifacts
are backed up, both replacement files are written atomically with rollback, and
provenance hashes are saved.

## Dub with GPT semantic fit

```bash
python -m mathula_tv.cli dub-azure "$JOB_ID" --live-operation
```

Use `--timing-mode semantic-fit` to build word-timed performance regions,
measure the approved delivery with the exact Azure voice, and ask GPT only for
unresolved regions. There is no Claude timing backend or provider fallback. The
system does not silently drop meaning, truncate speech, or hide overlaps.

## Language paths

- isiZulu (`zu-ZA`): `transcript_zu.json`, `translation_zu.json`
- Sepedi (`nso-ZA`): `transcript_nso.json`, `translation_nso.json`

The job target locale selects the paths. A manual response for another locale is
rejected.
