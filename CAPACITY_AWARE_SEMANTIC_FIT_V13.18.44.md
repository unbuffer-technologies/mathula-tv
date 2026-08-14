# Capacity-aware semantic fit v13.18.44

This cumulative update corrects the 19-minute v13.18.43 exhaustion reproduced
from job `d724a4c947404afeae1d748363b41827`.

## Exact diagnosis

The failure was not evidence that eleven translations were impossible. The
word-timed renderer added every detected source gap on top of Azure's natural
punctuation pauses, then required the resulting audio to end within a one-sided
250ms tail allowance.

For example, the shortest cached block 1 speech measured 8,812ms. The renderer
then added 2,960ms of mandatory breaks and rejected the 11,772ms result against
a 10,970ms ceiling. Only the 1,180ms island boundary is lip-sync-critical; the
three shorter phrase pauses must share whatever capacity remains after locked
tempo speech.

The supplied exhausted checkpoint contains 132 candidate observations for the
eleven failed blocks. Re-evaluation with this policy finds at least one
timing-feasible cached candidate for every block while retaining all long
island-boundary pauses.

## New timing policy

- The speaker's Azure-measured tempo remains locked at the established voice
  rate. No per-block speed change or post-synthesis stretch is introduced.
- Pauses of 800ms or longer remain protected speech-island boundaries and
  receive capacity first.
- Shorter phrase pauses use only the remaining capacity. They are cadence cues,
  not additional mandatory time layered over Azure's natural punctuation.
- When a faithful delivery ends early, available pause capacity is distributed
  across the detected source gaps to bring the word-timed endpoint back toward
  the source.
- A long candidate cannot erase an island to pass timing; it must yield to a
  shorter faithful contraction.
- Individual SSML breaks remain capped at 2,000ms, retaining the v13.18.43 safe
  chunking correction.
- The configured tolerance applies independently to the first and final mouth
  boundaries. A 250ms setting permits centring within 250ms at each boundary,
  not a large one-sided tail lag.
- Same-speaker transition borrowing remains disabled for word-timed blocks.

## Faster AI recovery

- Existing paid candidates in `semantic_fit.json` are remeasured before any new
  candidate-generation call.
- A strategy upgrade rebases the fresh attempt counter. If cached candidates
  fail semantic review, v13.18.44 can make a new bounded attempt instead of
  immediately reporting that the old v13.18.43 budget is exhausted.
- Repeated evidence is reduced to the useful timing frontier: shortest
  overflow, closest underflow, latest semantic rejection, and latest result.
- On the supplied checkpoint this reduces transmitted history from 132 to 25
  observations and from 185,998 to 9,277 serialized bytes (95.0%).
- Rejected wording history is bounded to the six most recent unique forms.

## Backward compatibility

- The command line and existing flags are unchanged.
- Jobs without authoritative Azure word timestamps retain the legacy pause,
  block-window, transition-borrow, and calibration implementation.
- Approved/display translations remain immutable.
- The McKenzie contextual pronunciation correction from v13.18.42 and Azure
  safe long-pause chunks from v13.18.43 remain included.
- No VAD, visual mouth model, PyTorch, GPU dependency, tempo change, or audio
  time stretching is introduced.

## Install

```bash
tar -xzf mathula-capacity-aware-semantic-fit-v13.18.44-source.tar.gz \
  -C ~/mathula-tv --no-same-owner --no-same-permissions
cd ~/mathula-tv
source .venv/bin/activate
pip install -e .
```

## Rerun the reproduced job

Keep the existing v13.18.43 `semantic_fit.json`; it contains the paid candidates
that v13.18.44 reuses.

```bash
mathula-tv dub-azure d724a4c947404afeae1d748363b41827 \
  --timing-mode semantic-fit \
  --semantic-fit-max-attempts 12 \
  --semantic-fit-tolerance-ms 250 \
  --live-operation \
  --force
```

## Validation

- All eleven failed blocks have a timing-feasible cached candidate under the
  capacity-aware island policy.
- Exact block 1, final-block underflow, island-preservation, two-boundary
  centring, strategy-rebase, evidence-compaction, and SSML chunk cases pass.
- 53 targeted regression tests pass across semantic fit, transcript lip-sync,
  pronunciation, cadence, mastering, and voice-cache behavior.
- Python byte-compilation passes for every supplied source and test module.

