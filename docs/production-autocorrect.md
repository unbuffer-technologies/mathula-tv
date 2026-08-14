# PostgreSQL production autocorrect

Mathula TV uses an event-sourced autocorrect model similar to phone
autocorrect. Users can review every change, revert it, redo it, teach a private
dictionary, or type a completely new word or phrase.

## PostgreSQL

Autocorrect uses the existing PostgreSQL server through:

```text
MATHULA_TV_DATABASE_URL
```

`DATABASE_URL` is accepted as a fallback. Mathula TV creates and owns a
separate schema:

```text
MATHULA_TV_DATABASE_SCHEMA=mathula_autocorrect
```

It does not modify commission-ai tables.

Install the dependency:

```bash
.venv/bin/python -m pip install -e '.[database]'
```

Run the idempotent schema migration:

```bash
.venv/bin/python scripts/mathula_autocorrect.py migrate
```

## Immutable transcript model

The first `prepare` operation snapshots:

```text
analysis/transcript_en.json
→ analysis/transcript_en.autocorrect_source.json
```

The effective transcript is regenerated from the immutable snapshot and active
correction overlays. Raw Azure artifacts remain unchanged.

## Correction states

- `suggested`
- `needs_review`
- `auto_applied`
- `confirmed`
- `edited`
- `reverted`
- `dismissed`
- `suppressed`

Every state change is stored in PostgreSQL and exported to:

```text
analysis/autocorrection_events.json
analysis/autocorrection_state.json
analysis/autocorrection_report.json
```

## Prepare a video

```bash
JOB_ID="..."
USER_ID="..."
PROJECT_ID="..."

.venv/bin/python scripts/mathula_autocorrect.py prepare \
  "$JOB_ID" \
  --user-id "$USER_ID" \
  --project-id "$PROJECT_ID" \
  --ai-model "$AZURE_AI_DEPLOYMENT"
```

New AI or registry suggestions are reviewable. They do not auto-apply.
Only an exact private user/project rule that was explicitly marked “always” or
confirmed repeatedly can auto-apply.

## Review

```bash
.venv/bin/python scripts/mathula_autocorrect.py list "$JOB_ID"

.venv/bin/python scripts/mathula_autocorrect.py questions \
  "$JOB_ID" \
  --maximum 5
```

Each correction contains the Azure text, replacement, timestamp, context,
candidate list, confidence, current state, and source.

## Accept a known correction

```bash
.venv/bin/python scripts/mathula_autocorrect.py accept \
  "$JOB_ID" CORRECTION_ID \
  --actor-id "$USER_ID" \
  --user-id "$USER_ID" \
  --project-id "$PROJECT_ID" \
  --learn-scope user \
  --learn-scope project
```

Use `--always` to create a private exact autocorrect rule:

```bash
.venv/bin/python scripts/mathula_autocorrect.py accept \
  "$JOB_ID" CORRECTION_ID \
  --actor-id "$USER_ID" \
  --user-id "$USER_ID" \
  --project-id "$PROJECT_ID" \
  --learn-scope project \
  --always
```

## Type a completely new term

The `new-term` command is the free-text path used for words such as
`hoshkhari`, new kasi slang, emerging music terms, creator names, brands, and
other vocabulary that is not yet in the candidate list.

Private learning only:

```bash
.venv/bin/python scripts/mathula_autocorrect.py new-term \
  "$JOB_ID" CORRECTION_ID \
  --actor-id "$USER_ID" \
  --text "hoshkhari" \
  --user-id "$USER_ID" \
  --project-id "$PROJECT_ID" \
  --term-category slang \
  --language en-ZA \
  --domain "South African street culture" \
  --learn-scope user \
  --learn-scope project \
  --always
```

The new term becomes active in the current transcript and is available to the
user/project dictionary and future Azure phrase exports.

## Contribute a new term to the public dictionary

Public contribution is optional and must be explicit:

```bash
.venv/bin/python scripts/mathula_autocorrect.py new-term \
  "$JOB_ID" CORRECTION_ID \
  --actor-id "$USER_ID" \
  --text "new street term" \
  --user-id "$USER_ID" \
  --project-id "$PROJECT_ID" \
  --term-category slang \
  --language en-ZA \
  --domain "South African street culture" \
  --learn-scope user \
  --learn-scope project \
  --contribute-publicly
```

The public evidence stores pseudonymous user and job hashes, not raw user IDs.
It requires:

```text
MATHULA_TV_FEEDBACK_HASH_SALT
```

A contribution does not become a public rule immediately. PostgreSQL
aggregates independent users and jobs into a moderated candidate.

## Moderate public terms

List candidates:

```bash
.venv/bin/python scripts/mathula_autocorrect.py public-report \
  --status candidate
```

Approve after evidence and privacy review:

```bash
.venv/bin/python scripts/mathula_autocorrect.py approve-public-term \
  TERM_ID \
  --moderator-id MODERATOR_ID
```

Reject:

```bash
.venv/bin/python scripts/mathula_autocorrect.py reject-public-term \
  TERM_ID \
  --moderator-id MODERATOR_ID
```

Approved public terms become candidates for all users and are included in
future Azure phrase exports. They are not automatic replacements by
themselves.

## Revert, redo and suppress

```bash
.venv/bin/python scripts/mathula_autocorrect.py revert \
  "$JOB_ID" CORRECTION_ID \
  --actor-id "$USER_ID"
```

Revert and prevent the same project suggestion:

```bash
.venv/bin/python scripts/mathula_autocorrect.py revert \
  "$JOB_ID" CORRECTION_ID \
  --actor-id "$USER_ID" \
  --user-id "$USER_ID" \
  --project-id "$PROJECT_ID" \
  --learn-scope project \
  --never
```

Redo:

```bash
.venv/bin/python scripts/mathula_autocorrect.py redo \
  "$JOB_ID" CORRECTION_ID \
  --actor-id "$USER_ID"
```

Undo only automatically applied changes:

```bash
.venv/bin/python scripts/mathula_autocorrect.py undo-all \
  "$JOB_ID" \
  --actor-id "$USER_ID"
```

## Export phrases for future Azure Fast jobs

```bash
.venv/bin/python scripts/mathula_autocorrect.py export-phrases \
  "$JOB_ID" \
  --user-id "$USER_ID" \
  --project-id "$PROJECT_ID"
```

This exports private learned terms first, followed by moderator-approved public
terms.

## Fresh source transcripts

When source STT or diarization is regenerated:

```bash
.venv/bin/python scripts/mathula_autocorrect.py reset-source \
  "$JOB_ID" \
  --actor-id "$USER_ID"
```

The previous autocorrect artifacts are archived before the new source snapshot
is adopted.
