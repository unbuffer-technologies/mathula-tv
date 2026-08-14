# Azure-safe long pause chunks v13.18.43

This cumulative hotfix corrects the v13.18.42 failure:

```text
error: SSML pause must be between 0 and 2000 ms
```

## Cause

v13.18.42 correctly preserved word-timed source pauses up to 3000ms, but emitted
the complete logical pause as one SSML `<break>`. Azure's production SSML
contract permits at most 2000ms per individual break even though the request's
total pause allowance is larger.

The supplied job contains a 2360ms source word gap. After the existing 100ms
speech baseline, the renderer requested a 2260ms SSML pause and the validator
rejected it before synthesis.

## Correction

- Logical source pauses remain capped at 3000ms.
- Each logical pause is divided into consecutive backend-safe chunks no larger
  than the active Azure backend's `pause_max_ms` value.
- The job's 2260ms logical pause is therefore rendered as 2000ms + 260ms.
- The audit retains both the logical pause duration and its physical SSML chunk
  durations.
- Leading word-timed silence uses the same safe chunking.
- Legacy pause calibration maps calibrated chunks back to their logical source
  insertion, preserving backward-compatible audit semantics.

The word-envelope timing and contextual McKenzie pronunciation corrections
from v13.18.42 are retained unchanged.

## Install

```bash
tar -xzf mathula-word-envelope-lipsync-v13.18.43-source.tar.gz \
  -C ~/mathula-tv --no-same-owner --no-same-permissions
cd ~/mathula-tv
source .venv/bin/activate
pip install -e .
```

The failed run stopped before completing baseline synthesis, so `--force` is
sufficient; its old semantic checkpoint can remain archived as described in
the v13.18.42 notes.

## Validation

- 47 targeted regression tests passed.
- The exact 2260ms logical pause renders as legal 2000ms and 260ms breaks.
- The production SSML validator accepts the resulting document.
- Python byte-compilation passed for every supplied source and test module.

