# Performance plan mode — v13.18.48

`--timing-mode performance-plan` is a new, opt-in direct Azure dubbing mode.
`balanced` and `semantic-fit` retain their existing behavior.

## Timing model

- Azure STT word timestamps are immutable timing evidence.
- Consecutive word-timed speech islands on one speaker lane form a performance
  region. STT blocks remain audit/storage members and are not linguistic bounds.
- Speaker changes, overlaps/interjections, gaps over 5000 ms, missing word
  timing, 30-second regions, and 12-island regions are hard boundaries.
- Claude plans semantic beats and assigns every source island exactly once, in
  source order. Meaning may move across internal STT member boundaries.
- Azure synthesizes at the persisted speaker rate. Rate changes and
  post-synthesis stretching remain forbidden.
- Acceptance requires both the full region and every island delivery to fit its
  measured word-timed window. Failed island measurements are returned as
  acoustic evidence in the next AI round.

## Provider budget

The first pass requests one holistic performance per region. A measured miss
opens a three-candidate feedback round. Existing checkpoints are resumable and
the strategy version creates a fresh attempt budget without deleting history.

## Compatibility

The CLI default remains `balanced`. The previous `semantic-fit` phrase-group
path is unchanged. Blocks without safe word timing and blocks at interaction
boundaries use the compatibility block path.

## Command

```bash
mathula-tv dub-azure d724a4c947404afeae1d748363b41827 \
  --timing-mode performance-plan \
  --semantic-fit-max-attempts 12 \
  --semantic-fit-tolerance-ms 250 \
  --live-operation \
  --force
```

The run writes `direct_dub/performance_plan.json` and preserves full selected
beat, island, region, measurement, and review evidence in the direct-dub report
and `semantic_fit.json` checkpoint.
