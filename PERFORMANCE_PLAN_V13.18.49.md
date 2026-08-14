# Performance plan mode — v13.18.49

`--timing-mode performance-plan` is an opt-in direct Azure dubbing mode.
`balanced` remains the default and the previous `semantic-fit` behavior remains
available.

## Timing model

- Azure STT word timestamps are immutable timing evidence.
- Consecutive word-timed speech islands on one speaker lane form a performance
  region. STT blocks remain audit/storage members, not linguistic bounds.
- Speaker changes, overlaps/interjections, gaps over 5000 ms, missing word
  timing, 30-second regions, and 12-island regions are hard boundaries.
- Claude plans semantic beats and assigns every source island exactly once, in
  source order. Meaning may move across internal STT member boundaries.
- Azure synthesizes at the persisted speaker rate. Rate changes and
  post-synthesis stretching remain forbidden.
- Acceptance requires both the full region and every island delivery to fit its
  word-timed window. Failed island measurements feed the next AI round.

## Bounded provider work

The first pass requests one holistic performance per region. Performance output
is scheduled in shards of at most three candidates. Each completed shard is
measured, reviewed, and written to `semantic_fit.json` before the next provider
request begins. A later `max_tokens`, rate-limit, or transport failure therefore
does not discard completed paid work.

The structured-output ceiling now scales with the candidate's source-island
count. A measured miss opens a three-candidate feedback round for only that
region; it does not regenerate already selected regions.

## Install

Run these commands from the repository root. `--no-same-owner` and
`--no-same-permissions` avoid the directory metadata error seen when extracting
an overlay over an existing checkout.

```bash
cd ~/mathula-tv
tar --no-same-owner --no-same-permissions \
  -xzf mathula-performance-plan-v13.18.49-source.tar.gz
source .venv/bin/activate
pip install -e .
```

## Redo job d724a4c947404afeae1d748363b41827

```bash
mathula-tv dub-azure d724a4c947404afeae1d748363b41827 \
  --timing-mode performance-plan \
  --semantic-fit-max-attempts 12 \
  --semantic-fit-tolerance-ms 250 \
  --live-operation \
  --force
```

Do not delete `direct_dub/semantic_fit.json`. Versioned strategy history and
completed candidates are intentionally reused.
