# Deploy Mathula TV v13.17.2

Patch:

`mathula-tv-claude-resilience-v13.17.2-20260730T143224Z.tar.gz`

This archive is cumulative and can be installed over the existing Mathula TV
checkout even if one of the earlier v13.16.x/v13.17.x patches was only partially
installed.

## 1. Put the archive on the server

Copy the downloaded archive to:

`/tmp/mathula-tv-claude-resilience-v13.17.2-20260730T143224Z.tar.gz`

For example, from the computer containing the download:

```bash
scp mathula-tv-claude-resilience-v13.17.2-20260730T143224Z.tar.gz \
  mokgethwa@YOUR_SERVER:/tmp/
```

## 2. Stop the active command

If `translate`, `dub-azure`, or publication rendering is running in the current
terminal, stop it with `Ctrl+C`. Completed translation checkpoints are retained.

## 3. Activate the project and verify the archive

```bash
cd /home/mokgethwa/mathula-tv
source .venv/bin/activate

PATCH="/tmp/mathula-tv-claude-resilience-v13.17.2-20260730T143224Z.tar.gz"

test -f "$PATCH" || {
  echo "Patch not found: $PATCH"
  exit 1
}

sha256sum "$PATCH"
tar -tzf "$PATCH"
```

Compare the printed SHA-256 value with the checksum supplied with the download.

## 4. Back up every file replaced by the patch

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="/tmp/mathula-tv-before-v13.17.2-${STAMP}"
mkdir -p "$BACKUP_DIR"

while IFS= read -r FILE
do
  if [[ -f "$FILE" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$FILE")"
    cp -a -- "$FILE" "$BACKUP_DIR/$FILE"
  fi
done < <(tar -tzf "$PATCH" | grep -E '^(src/|tests/|DEPLOY.md$|PATCH_NOTES.md$)')

echo "Rollback backup: $BACKUP_DIR"
```

Keep the printed `BACKUP_DIR` path until the resumed job has completed.

## 5. Install

The `--no-overwrite-dir` and `--touch` options avoid the directory permission
and `utime` errors seen with earlier archives.

```bash
tar -xzf "$PATCH" \
  --no-same-owner \
  --no-same-permissions \
  --no-overwrite-dir \
  --touch \
  -C /home/mokgethwa/mathula-tv
```

## 6. Compile and verify the installed markers

```bash
cd /home/mokgethwa/mathula-tv
source .venv/bin/activate

python -m compileall -q src/mathula_tv

rg -n \
  "foundry_structured_outputs|stream_json_salvaged|split_chunk_recovery_count" \
  src/mathula_tv/ai_provider.py \
  src/mathula_tv/one_call_translation.py

rg -n "TOTAL WALL TIME" src/mathula_tv/cli.py
```

All three commands must exit successfully.

## 7. Run the focused deployment tests

```bash
pytest -q \
  tests/test_claude_resilience_v13172.py \
  tests/test_ai_provider.py \
  tests/test_foundry_max_tokens_recovery.py \
  tests/test_multivariant_ai_provider.py \
  tests/test_targeted_validation_repair_v13170.py \
  tests/test_targeted_identity_repair_v13160.py \
  tests/test_editorial_ai_profiles.py \
  tests/test_multivariant_exact_manual_prompt.py
```

Expected: 45 tests pass.

Do not add the entire `test_translation_response_recovery.py` file to this
deployment gate. It contains two legacy identity tests already failing in the
v13.17.1 baseline. The focused recovery assertions used by this patch pass.

## 8. Structured-output mode

No environment change is required. The recommended default is:

```text
MATHULA_TV_FOUNDRY_CLAUDE_STRUCTURED_OUTPUTS=auto
```

Modes:

- `auto`: attempt native Structured Outputs, then remember a deployment-level
  fallback if Azure rejects the feature;
- `enabled`: require native Structured Outputs and fail if unavailable;
- `disabled`: always use schema-in-prompt plus local validation.

Use `auto` unless the specific Foundry deployment has already been verified and
you deliberately want fail-closed `enabled` behavior.

## 9. Resume the existing job

Use the same job ID. Do not start a new job:

```bash
export JOB_ID="8b670180f8e44ccf8e6a04fc9219b313"

python -m mathula_tv.cli translate "$JOB_ID" --live-operation
```

The command reuses completed chunk checkpoints. If the saved failed chunk is a
truncation/context-size failure, it proceeds to serial split recovery without
paying for the same parent request again.

If autocorrect research, rather than translation, is still the incomplete
phase, run:

```bash
python -m mathula_tv.cli resume-autocorrect-research \
  "$JOB_ID" \
  --live-operation

python -m mathula_tv.cli translate "$JOB_ID" --live-operation
```

## 10. Inspect consumption, progress, and timing

During execution, the terminal now reports request size/token estimates, actual
usage, retries/fallbacks, chunk progress, split recovery, and FFmpeg progress.
At command completion it prints every tracked phase plus `TOTAL WALL TIME`.

Inspect the cumulative machine-readable AI report:

```bash
python -m json.tool \
  "working/jobs/$JOB_ID/analysis/ai_consumption_report.json" \
  | less
```

Find the most important recovery and timing events in a captured log:

```bash
rg -n \
  "consumption|capability fallback|salvaged|split recovery|ffmpeg|TOTAL WALL TIME" \
  /path/to/your/run.log
```

## Rollback

Replace the example with the exact backup path printed during installation:

```bash
cd /home/mokgethwa/mathula-tv
source .venv/bin/activate

BACKUP_DIR="/tmp/mathula-tv-before-v13.17.2-YYYYMMDDTHHMMSSZ"
test -d "$BACKUP_DIR" || {
  echo "Backup not found: $BACKUP_DIR"
  exit 1
}

while IFS= read -r -d '' FILE
do
  RELATIVE="${FILE#"$BACKUP_DIR/"}"
  mkdir -p "$(dirname "$RELATIVE")"
  cp -a -- "$FILE" "$RELATIVE"
done < <(find "$BACKUP_DIR" -type f -print0)

python -m compileall -q src/mathula_tv
```

This restores every pre-existing file that the archive replaced. The new
regression test `tests/test_claude_resilience_v13172.py` can remain in place; it
does not affect runtime behavior.
