# Performance-plan v13.18.50

This release replaces parallel paraphrase guessing with a bounded,
measurement-driven region controller. It is cumulative and backwards
compatible: `balanced` and `semantic-fit` retain their existing behavior, while
the opt-in `performance-plan` mode receives the new controller.

## Production behavior

- Azure word-timed performance regions remain the linguistic and timing unit;
  STT blocks are retained only as audit members.
- Speaker rate stays locked at the resolved natural rate. Post-synthesis time
  stretching is not used.
- Continuous whole-region duration and immutable source endpoints are the hard
  lip-sync gate.
- Isolated island TTS is advisory. The controller removes isolated
  phrase-boundary scale error and reports normalized cumulative boundary drift;
  an already fitting region is never rejected because an island synthesized on
  its own has different prosody.
- Every feedback round asks Claude for exactly one targeted revision per
  scheduled region. The next request includes the latest Azure duration and an
  explicit `contract`, `expand`, or `maintain` direction.
- Provider requests contain no more than three region candidates, keeping
  structured output below the practical Claude response ceiling.
- Performance regions receive at most four measured AI revisions even when the
  general CLI attempt value is larger. Paid checkpoint candidates are replayed
  before new calls, and a strategy-version change grants a fresh bounded budget.
- Progress distinguishes selected regions, regions scheduled in the current
  shard, and total unresolved regions.
- Beat ownership accepts punctuation/spacing changes but still rejects lexical
  differences, missing islands, duplicates, and source-order changes.

## Install

From the Mathula TV repository root:

```bash
tar -xzf mathula-performance-plan-v13.18.50-source.tar.gz \
  -C "$HOME/mathula-tv" --no-same-owner --no-same-permissions
cd "$HOME/mathula-tv"
source .venv/bin/activate
pip install -e .
```

The archive has no `.` root entry, so extraction does not attempt to change the
repository root's timestamp or mode.

## Resume job d724

```bash
mathula-tv dub-azure d724a4c947404afeae1d748363b41827 \
  --timing-mode performance-plan \
  --semantic-fit-max-attempts 12 \
  --semantic-fit-tolerance-ms 250 \
  --live-operation \
  --force
```

The performance controller internally caps its fresh measured revisions at
four. The value `12` remains compatible with older semantic-fit behavior and
does not cause twelve performance-plan guesses.

## Verification

The cumulative source passed 76 regression tests. Added production cases cover:

- the observed d724 contract/expand/maintain decisions;
- acceptance of d724 region 5 when the whole region fits despite three isolated
  island misses;
- one measured revision per feedback round;
- the four-revision performance budget;
- punctuation-only beat ownership normalization;
- nine-region, three-shard checkpointing;
- locked rate and no time stretching.
