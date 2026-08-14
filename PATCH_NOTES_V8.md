# V8 — multivariant translation and manual override

- Normal cloud translation now returns natural, concise, and compact variants in
  one source-bound request.
- `dub-azure` uses prepared variants first and has no Claude timing-repair calls
  by default.
- Added prompt-package export and reviewed manual import commands.
- Added strict JSON/schema/timeline/protected-span/duration validation.
- Added immutable cloud-run and manual-override provenance histories.
- Added atomic paired artifact installation with rollback.
- Added dynamic isiZulu/Sepedi translation paths.
- Preserved application ownership of pronunciation-only `tts_text` and SSML.
- Preserved the explicit Claude timing-repair backend as an opt-in last resort.
