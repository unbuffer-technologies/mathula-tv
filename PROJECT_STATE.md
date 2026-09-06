# Mathula TV engineering handoff

Last updated: 2026-08-22

## Read this first: this file was stale

Everything below the "Architectural decisions" section through "External work
still required" describes an **earlier OpenVoice-based production path** that
was accurate as of 2026-07-20. Since then, `src/mathula_tv/native_dub.py`
(Azure Fast Transcription → Foundry Grok Pass 1/2 → Azure TTS at rate 0 →
pitch-preserving `atempo` timing correction → hook/editorial/SEO overlay →
publication render) has become the active production path for English →
isiZulu dubbing, entirely uncommitted in git until reviewed/promoted. It does
not call OpenVoice at all. Do not treat the OpenVoice-centric sections below
as a description of what `native-dub` currently does — they are retained only
because some of the underlying architectural invariants (Azure STT authority,
pyannote as diagnostic-only, no YouTube upload, `spoken_text` subtitles, etc.)
still hold for the pipeline in general, not because the OpenVoice conversion
step is still in the loop for new work.

The runtime AI provider is **not** Claude/Anthropic despite legacy field
names (`claude_model`, `claude_effort`, ...) surviving in `config.py` for
backward-compatible job-snapshot reads. `Settings.ai_provider` is hardcoded to
`"azure-openai-gpt"`; a stale provider selector in an existing `.env` is
intentionally ignored. Treat any Claude/Anthropic references below and in
older docs as historical.

The old "next command from the remote server" instructions further down
referenced a Linux host (`/home/mokgethwa/mathula-tv`). Development and the
live job referenced in the most recent handover run directly on Windows at
`C:\mathula-tv` with `MATHULA_TV_WORK_DIR` defaulting to `C:\mathula-tv\working`.
Do not resurrect the Linux remote-server command below as a literal
instruction; it is historical context only.

## Native-dub path: current status (2026-08-22)

- CLI subcommands: `native-translate` (Pass 1 STT correction / Pass 2 natural,
  non-concise isiZulu translation, via `--pass 1|2|all`) and `native-dub`
  (Azure TTS synthesis, mouth-close timing repair loop, hook/editorial
  overlay, publication render). Flags and defaults are defined in
  `cli.py`'s `native-dub`/`native-translate` argparse blocks — inspect them
  directly rather than trusting any flag list in a handover doc, since they
  change.
- Fixed this session: `native-translate` and `native-dub` never updated
  `job.state`/`job.completed_stages`/`job.review_readiness` after a real run,
  so a job's `status.json` could claim `state: "analysis_running"` while its
  `native_dub/` folder already held a fully rendered, hook-edited MP4. Both
  command handlers in `cli.py` now append `native_pass1`/`native_pass2`/
  `native_dub` to `completed_stages` and set `state`/`review_readiness`
  (`"needs_timing_review"` when any phrase-boundary overlap in
  `timing_plan.json`'s `dialogue.overlaps` is flagged `human_review_required`,
  else `"review_ready"`).
- Fixed this session: a stale POSIX `MATHULA_TV_WORK_DIR` (or
  `COMMISSION_MASTER_CASE_PATH` / `POLITICS_CONTEXT_PATH`) value silently
  became a broken `C:\home\...`-style path on Windows instead of being
  rejected. `config.py`'s `_resolve_windows_safe_path`/`_resolve_work_dir` now
  reject root-relative POSIX values (with or without a drive letter, including
  backslash-escaped forms) on Windows and fall back to a project-local
  default with a `RuntimeWarning`. Covered by `tests/test_config_work_dir.py`.
- Added this session: an independent post-Pass-2 **referent-fidelity audit and
  bounded repair** (`src/mathula_tv/referent_audit.py`, wired into
  `run_native_pass2`/`run_native_translation`, CLI flag
  `--audit-referents`/`--no-audit-referents`, default **on**). This exists
  because a real production example was found in job `33cd7b46...`'s
  `pass2_natural_translation.json`: Pass 2's inline "self-QA" (same call as the
  translation) rendered "close to **the minister**" as "close to **God**"
  (`ngqongqoshe` → `uNkulunkulu`) and its own `semantic_guard_warnings` never
  flagged it. The new audit is a second, independent Grok call per group
  (English + current Zulu only, no translation reasoning) checking names,
  roles/titles, numbers/dates, and negation; a free deterministic tier also
  checks `config/entity_registry.json` named entities with no AI call. Only
  flagged groups get a bounded repair call. Findings/repairs are recorded in
  `native_dub/referent_audit.json`; a compact summary is embedded in
  `pass2_natural_translation.json` as `referent_audit_summary` and reused on
  rerun. See `docs/native-dub-grok.md` section 6. Covered by
  `tests/test_referent_audit.py` and
  `tests/test_native_translate_referent_audit_flags.py`.

- **First live run happened** (job `33cd7b46...`, `--pass 2 --temporal-mask
  --live-operation`) and surfaced two real bugs, both now fixed:
  1. The repair call's flat `300 tokens/item` output budget was far too small
     for this transcript's long testimony passages and hit `max_tokens`,
     which then killed the *entire* command and lost the already-paid-for
     audit findings. Fixed: budget now scales from actual text length
     (`_estimate_output_tokens`), and `_call_with_truncation_retry` retries a
     truncated batch as two smaller batches (recursing to single items)
     instead of propagating the failure — a token-budget miss on one group
     can no longer crash the whole audit.
  2. The deterministic entity check produced 5 false positives, all "the
     Commission" reported missing from Zulu text that correctly said
     `iKhomishini` — the entity registry's alias list includes that bare
     generic phrase, but organisations are properly *translated*
     (`iKhomishini`), unlike this corpus's people, who stay verbatim
     (`Fadiel Adams`). Fixed: the deterministic tier now only checks
     `entity_type in {"person", "anonymous_witness"}`
     (`_LITERALLY_PRESERVED_ENTITY_TYPES`); organisations/places/cases are
     left to the AI audit call, which can judge translated equivalence.
  - **Still open, and the reason this exists**: the AI audit call itself did
    **not** flag the original minister→God bug (`native_phrase_0006` in that
    job still reads `kunkulunkulu`, unfixed, as of this writing). The audit
    prompt/schema were then strengthened from a single holistic "ok/issue"
    judgment to a forced two-step "extract every checkable term from the
    English, then verify each one individually against the Zulu"
    (`checked_terms` in `_AUDIT_SCHEMA`) — a technique intended to prevent
    exactly the kind of skim-and-miss failure observed, but **this has not
     yet been tested live**. `referent_audit.json` now also persists
    `all_ai_verdicts` (every group's verdict, not just flagged ones) so a
    future miss can be diagnosed the way this one was.

- **Second live run** (same command, no `--force`) exposed exactly the checkpoint
  problem predicted above, plus a display bug: `REFERENT_AUDIT_SCHEMA_VERSION`
  hadn't been bumped when the `checked_terms`/`groups_flagged` schema changed,
  so the run correctly hit "Reusing matching completed checkpoint" against the
  *old* v1 summary still on disk (`issues_found: 10`, no `groups_flagged` key at
  all) — and the CLI printed "0 flagged" because it read a field name that v1
  never had, defaulting to 0. Not a real finding; a stale-cache display bug.
  Fixed: `REFERENT_AUDIT_SCHEMA_VERSION` is now
  `"mathula-native-referent-audit-v2-extract-then-verify"`, and
  `_apply_referent_audit`'s checkpoint check in `native_dub.py` now also
  requires `existing_summary.get("schema_version") == REFERENT_AUDIT_SCHEMA_VERSION`
  before reusing — an older on-disk summary now triggers a fresh audit
  automatically, without needing `--force`. Covered by
  `tests/test_native_dub_referent_audit_checkpoint.py`.

- **Third live run** (same command, schema-version fix now forced a fresh
  audit) printed "No referent issues found across 57 group(s)" — still
  missing `native_phrase_0006`. But `referent_audit.json`'s `all_ai_verdicts`
  (added specifically for this kind of debugging) showed the model's own
  `checked_terms` for that group correctly extracted `"minister"` as
  `role_or_title` **and** correctly verified it as `"substituted"`, with
  `status: "issue"` — the extract-then-verify step worked. The miss was a bug
  in this codebase's own orchestration, not the model: `flagged_group_ids`
  required a *separate*, model-authored `issues` array to also be non-empty
  before treating a group as flagged, and the model left `issues` empty/absent
  even though `checked_terms` correctly recorded the substitution — so a
  correct detection was silently discarded. Fixed: `issues` is no longer part
  of the audit schema or trusted from the model at all; `_derive_issues()`
  derives the authoritative issue list directly from `checked_terms`
  (`verification != "present"`) for both flagging and the repair payload.
  `REFERENT_AUDIT_SCHEMA_VERSION` bumped again to
  `"mathula-native-referent-audit-v3-checked-terms-authoritative"` so this
  job's stale checkpoint invalidates automatically on the next run — no
  `--force` needed. This is a stronger, more probable fix than the prompt
  rewrite alone (it was a real, demonstrated code bug, not a modeling
  problem), but it is still unverified against a live Grok call as of this
  writing — the next run is the real test of whether `native_phrase_0006`
  finally gets corrected.

- **Not fixed, and the highest-priority open issue**: the timing-repair loop
  (measure → decide → paraphrase → resynthesize → speed-up → verify) does not
  currently prevent one phrase's synthesized audio from overrunning into the
  next phrase's onset. When it happens, the renderer places the colliding
  audio on separate speaker tracks with limiter gain management and flags the
  pair `human_review_required: true` in `timing_plan.json`'s
  `dialogue.overlaps` rather than resolving the collision. This is the
  concrete mechanism behind the "Speaker 1 overflowed into Speaker 2"
  complaint from prior handovers; treat it as a real, unresolved architecture
  gap, not an anecdote.
- Repo hygiene: several unmerged patch-overlay directories and stale
  already-superseded `.patch`/`.tar.gz` archives that had accumulated at the
  repo root from prior incremental sessions were removed this session once
  confirmed their content was already present in (or superseded by) the live
  `src/` tree. Keep this file updated if that pattern recurs — loose overlay
  directories at the repo root are a sign a fix was written but never copied
  into `src/`.

## Architectural decisions

- Azure Speech-to-Text is the production transcription provider. Its job-local speaker labels are authoritative.
- Pyannote is opt-in diagnostic/comparison/overlap analysis only and cannot overwrite Azure labels silently.
- Azure Text-to-Speech creates the native isiZulu base delivery and owns pronunciation. Application code—not the AI provider—constructs allowlisted, escaped SSML.
- F5-TTS is absent from the Mathula TV architecture and must not be registered or restored.
- OmniVoice is deprecated, not registered, not selectable, not production, and unreachable. Any remaining source/notebook is migration evidence pending deletion after a compatibility pass or explicit cleanup; it is never a fallback.
- Original English dialogue must be removed or sufficiently suppressed for publication review. Ducking the original mix is internal-review-only.
- Subtitles use `spoken_text`, never pronunciation-respelled `tts_text`. Automatic phoneme-level lip synchronisation is out of scope.
- Mathula TV does not upload to YouTube.

## Older OpenVoice-path record (historical, 2026-07-20)

The following described the intended production path before `native_dub.py`
existed: Azure STT → Claude Opus 4.8 → Azure TTS → OpenVoice → measured
alignment → background reconstruction → mix/subtitles/render/QC → mandatory
human review. Live Phase A execution was in progress for proof job
`19ba6d69f1b84132ba4f20599101834a`. This is retained for historical context
only; do not treat it as the current plan.

Never commit or paste `.env`, the service-account JSON key, Azure/Foundry keys, clips, transcripts, generated audio, job manifests, logs, or Colab secrets. The public repository contains code, tests, and this non-secret handover only.

## Operational safety

The live job must never be used by unit tests. Live commands require `--live-operation` where the command gates on it (inspect `cli.py` per-command — not every native-dub/native-translate invocation currently requires it); paid/GPU integration tests require their individual `MATHULA_TV_RUN_LIVE_*_TESTS=1` opt-ins. Verify manifests and hashes before every promotion, stop a stale worker after lease loss, and never paste secrets, authorization headers, cookies, private signed URLs, full private configuration, or unnecessary transcripts into logs or notebook output.
