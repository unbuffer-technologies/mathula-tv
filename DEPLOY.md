# Deploy Mathula TV v13.18.26

Patch:

`mathula-tv-mastering-no-change-guard-v13.18.26-20260811T192638Z.tar.gz`

This is a cumulative patch containing every Mathula TV upgrade through
v13.18.26. It fixes the identical-audio mastering loop seen on job
`6f410d500d9e4e7ca07653550e3c883e`.

## 1. Copy the patch to the server

Place the downloaded archive at:

```text
/tmp/mathula-tv-mastering-no-change-guard-v13.18.26-20260811T192638Z.tar.gz
```

Then verify it without extracting:

```bash
cd /home/mokgethwa/mathula-tv
source .venv/bin/activate

PATCH="/tmp/mathula-tv-mastering-no-change-guard-v13.18.26-20260811T192638Z.tar.gz"

test -f "$PATCH" || {
  echo "ERROR: Patch not found: $PATCH"
  exit 1
}

sha256sum "$PATCH"
tar -tzf "$PATCH" | less
tar -xOf "$PATCH" DEPLOY.md | less
```

## 2. Stop an active old process

If `translate`, `dub-azure`, or `optimize-dub` is currently running, stop only
that command with `Ctrl+C`. Do not delete the job directory, translation,
speaker mappings, provider checkpoints, TTS cache, rendered media, or mastering
reports.

## 3. Back up every replaced file

```bash
cd /home/mokgethwa/mathula-tv

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="/tmp/mathula-tv-before-v13.18.26-${STAMP}"
mkdir -p "$BACKUP_DIR"

while IFS= read -r FILE
do
  if [[ -f "$FILE" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$FILE")"
    cp -a -- "$FILE" "$BACKUP_DIR/$FILE"
  fi
done < <(
  tar -tzf "$PATCH" |
    sed 's#^\./##' |
    grep -E '^(src/|scripts/|tests/|pyproject.toml$|requirements.txt$|DEPLOY.md$|PATCH_NOTES.md$)'
)

echo "Rollback backup: $BACKUP_DIR"
```

## 4. Install safely

```bash
tar -xzf "$PATCH" \
  --no-same-owner \
  --no-same-permissions \
  --no-overwrite-dir \
  --touch \
  -C /home/mokgethwa/mathula-tv
```

These flags prevent the directory `utime` and permission errors encountered
with earlier archives.

## 5. Compile and verify the installed markers

```bash
cd /home/mokgethwa/mathula-tv
source .venv/bin/activate

python -m compileall -q src/mathula_tv scripts

rg -n \
  "v13.18.26|mathula-production-dub-score-v3|mastering_variant_id|repair_render_produced_identical_audio" \
  src/mathula_tv/direct_azure_dub.py \
  src/mathula_tv/dub_mastering.py
```

All four markers must be present.

## 6. Run the focused regression suite

```bash
pytest -q \
  tests/test_dub_mastering_no_change_v131826.py \
  tests/test_dub_mastering_calibration_v131825.py \
  tests/test_dub_mastering_loop_v131824.py \
  tests/test_direct_azure_dub.py \
  tests/test_direct_azure_voice_cache.py \
  tests/test_dub_azure_cli_output.py \
  tests/test_zulu_native_honorifics_v131823.py \
  tests/test_zulu_native_dates_v131822.py \
  tests/test_target_text_pronunciation.py \
  tests/test_contextual_turn_repair_v131821.py \
  tests/test_zulu_coloured_code_switch_v131820.py \
  tests/test_reporter_attribution_word_alignment_v131819.py \
  tests/test_five_block_final_conductor_v131818.py
```

Expected result: `102 passed`.

## 7. Run mastering on the affected job

```bash
JOB_ID="6f410d500d9e4e7ca07653550e3c883e"

python -m mathula_tv.cli optimize-dub "$JOB_ID" \
  --max-rounds 2 \
  --live-operation
```

Do not add `--force`. On the first non-audit v13.18.26 run, the v2 automatic
override payload is archived under:

```text
working/jobs/JOB_ID/direct_dub/mastering/policy_migrations/
```

The code then rebuilds the unmodified approved-variant baseline under policy
v3. Existing Azure STT and TTS cache entries remain reusable.

The repair log should now show a changed audio checksum before a second STT
round. If any future repair still produces identical audio, the command stops
immediately with:

```text
repair_render_produced_identical_audio
```

It will not spend another Azure STT request on that unchanged mix.

## 8. Inspect the result

```bash
jq '{
  state,
  stopped_reason,
  best_round,
  best_production_score,
  best_round_restored,
  prior_policy_overrides_rebased,
  rounds
}' "working/jobs/$JOB_ID/direct_dub/mastering/report.json"

jq '.' \
  "working/jobs/$JOB_ID/direct_dub/mastering_overrides.json"
```

Check that a repair round includes:

```text
repair_audio_changed: true
```

If it is `false`, the loop must have stopped after that round and restored the
best override set.

## 9. Create or reuse the publication output once

After mastering finishes:

```bash
python -m mathula_tv.cli dub-azure "$JOB_ID" --live-operation
```

The mastering loop works on the clean dub and does not run the publication
editor in each feedback round. The final command reuses mastered assets and
performs the publication stage once.

## Rollback

```bash
cp -a "$BACKUP_DIR"/. /home/mokgethwa/mathula-tv/
python -m compileall -q /home/mokgethwa/mathula-tv/src/mathula_tv
```

Rollback restores code and tests only. It does not delete job media, cached
provider responses, mastering audits, or policy-migration evidence.
