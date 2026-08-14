# Word-envelope lip-sync and contextual pronunciation v13.18.42

This cumulative source update fixes the two regressions reproduced from job
`d724a4c947404afeae1d748363b41827`: the job's reviewed McKenzie pronunciation
was lost after semantic rewriting, and word-timed speech was allowed to extend
into following silence even after the visible source speaker had stopped.

## Rendering contract

- Azure word timestamps are authoritative for the start, internal pauses, and
  final endpoint of a word-timed speech block.
- Semantic-fit accepts a word-timed delivery only within the configured
  tolerance of the final source word. It cannot borrow a same-speaker
  transition after that endpoint.
- Source pauses at 800ms or longer are retained as speech-island boundaries.
  Word-timed pauses are not shortened to make overlong wording pass. The
  semantic solver must instead find a shorter faithful contraction.
- The maximum transferable source pause is 3000ms, covering the 2360ms pause
  measured in this job while remaining within the existing Azure SSML bound.
- Each fitted block records its source endpoint, rendered endpoint, endpoint
  error, and deadline result in the direct-dub report.

For block 3 in the supplied failure archive, the source words run from
25,630ms to 30,830ms. The old result ran to 32,305ms. With a 250ms tolerance,
the new semantic-fit duration band is 4,950–5,450ms around the authoritative
5,200ms source speech envelope.

## Pronunciation contract

- The pronunciation dictionary is reapplied after every AI semantic-fit
  rewrite, including batched, serial-fallback, and final-conductor candidates.
- Reviewed job-local `noC <name>` entries now safely derive matching isiZulu
  subject-prefix forms such as `uC <name>` and `u<name>` for the same name.
- This is derived from the job's reviewed override rather than a hard-coded
  person. In the supplied job, `uC Mackenzie` therefore reaches Azure as
  `u Makenzi`, consistent with the existing `noC Mackenzie` → `no Makenzi`
  review decision.
- Substitutions applied after semantic rewrites are retained in the candidate
  audit.

## Backward compatibility

- The `mathula-tv dub-azure` command and all existing flags are unchanged.
- Jobs without usable Azure word timestamps retain the legacy block-window,
  transition-borrow, and pause-calibration behavior.
- New timing fields have defaults, so existing `SpeechBlock` callers remain
  valid.
- Approved/display translations remain immutable; only the hidden TTS layer is
  pronunciation-normalized.
- No VAD, visual mouth model, PyTorch, or GPU dependency is introduced.

## Install

```bash
tar -xzf mathula-word-envelope-lipsync-v13.18.42-source.tar.gz \
  -C ~/mathula-tv --no-same-owner --no-same-permissions
cd ~/mathula-tv
source .venv/bin/activate
pip install -e .
```

## Clean rerun of the reproduced job

Preserve the old semantic checkpoint for comparison, then rerun. Speaker and
voice-resolution caches are not removed.

```bash
job=~/mathula-tv/working/jobs/d724a4c947404afeae1d748363b41827
if [ -f "$job/direct_dub/semantic_fit.json" ]; then
  mv "$job/direct_dub/semantic_fit.json" \
    "$job/direct_dub/semantic_fit.pre-v13.18.42.json"
fi

mathula-tv dub-azure d724a4c947404afeae1d748363b41827 \
  --timing-mode semantic-fit \
  --semantic-fit-max-attempts 12 \
  --semantic-fit-tolerance-ms 250 \
  --live-operation \
  --force
```

## Validation

- 46 targeted regression tests passed across semantic fitting, transcript
  lip-sync, pronunciation, cadence, mastering, and voice-cache behavior.
- The exact supplied job dictionary transforms `uC Mackenzie` to `u Makenzi`.
- The exact block 3 source timing produces a 4,950–5,450ms target band.
- Python byte-compilation passed for every supplied source and test module.

