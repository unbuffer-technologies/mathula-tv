# Semantic-fit portfolio v13.18.45

This cumulative update resolves the two-block v13.18.44 exhaustion reproduced
from job `d724a4c947404afeae1d748363b41827`.

## Exact diagnosis

The remaining blocks had different causes and must not share one fallback.

### Block 21: Azure pause-realization error

The approved natural translation had already succeeded in an earlier render at
20,536ms. In v13.18.44 it was measured again with a 6,900ms SSML pause plan,
but Azure realized that plan as 21,088ms total audio. The word-timed branch then
refused all pause calibration and unnecessarily sent the block into semantic
rewriting.

The source contains 5,080ms of protected island-boundary pause requests plus
flexible phrase pauses. v13.18.45 calibrates only capacity above the protected
floors. Replaying the recorded Azure measurements reduces the flexible total
to 6,349ms and reproduces the 20,536ms accepted delivery without changing text,
tempo, or any protected island floor.

The final Azure word timestamp extends 38ms beyond the decoded media boundary.
Word-envelope centring now uses the playable endpoint, so the selected audio
cannot be positioned beyond the final video frame.

### Block 6: false fact and serial search

The coalesced translation metadata listed the bare complementizer `ukuthi` as
an independent required fact. The reviewer therefore rejected otherwise useful
clause fusion as `missing_fact_003_that`, while later contractions deleted real
content such as "tonight" or "many things" to recover time.

The acoustic requirement is now explicit: the 1,380ms island pause is mandatory,
leaving at most 4,880ms of plain locked-tempo speech inside the 6,260ms boundary
band. Bare `ukuthi` is removed from the proposition list, while the Patriots
address, tonight reference, plurality/extent, lack of knowledge, and continuing
"but" relationship remain subject to independent semantic review.

## Faster semantic search

- The attempt budget is no longer spent as one expensive Claude generation per
  block per round when the provider supports portfolios.
- A portfolio call returns up to six genuinely distinct grammatical
  contractions per unresolved block, ordered from the fullest likely fit to
  the shortest faithful fit.
- With the normal 12-attempt budget, candidate generation is therefore bounded
  to at most two Claude calls per unresolved block set.
- Azure measures every portfolio candidate locally at the locked voice rate.
- Timing-fit candidates are independently reviewed in fidelity-first order.
  If one fails review, the next already-generated fit is reviewed without
  paying for another candidate-generation call.
- Providers without the new portfolio method retain the v13.18.44 iterative
  batch implementation.
- The existing `semantic_fit.json` remains reusable. The v13.18.45 strategy
  rebases its fresh budget after the 24 recorded v13.18.43/v13.18.44 attempts.

## Preserved production rules

- No per-block Azure rate change.
- No post-synthesis time stretching.
- No VAD, visual mouth model, PyTorch, or GPU dependency.
- Protected speech-island pauses cannot calibrate below their requested floors.
- Phrase pauses remain capacity-aware and may absorb Azure's SSML realization
  difference.
- McKenzie pronunciation handling and 2,000ms-safe SSML pause chunking remain
  included.
- Jobs without authoritative Azure word timing retain the legacy calibration
  and timing behavior.

## Install

```bash
tar -xzf mathula-semantic-fit-portfolio-v13.18.45-source.tar.gz \
  -C ~/mathula-tv --no-same-owner --no-same-permissions
cd ~/mathula-tv
source .venv/bin/activate
pip install -e .
```

## Rerun

Keep the existing `direct_dub/semantic_fit.json` and run:

```bash
mathula-tv dub-azure d724a4c947404afeae1d748363b41827 \
  --timing-mode semantic-fit \
  --semantic-fit-max-attempts 12 \
  --semantic-fit-tolerance-ms 250 \
  --live-operation \
  --force
```

## Validation

- Exact block-21 pause realization and media-tail placement are covered.
- Block-6 fact normalization and its 4,880ms plain-speech budget are covered.
- Six-candidate single-call portfolio generation, review order, strategy
  fallback, cached-candidate recovery, pronunciation, cadence, transcript
  lip-sync, mastering, and voice-cache behavior are covered.
- 58 targeted regression tests pass.
- Every supplied Python source and test module byte-compiles.

