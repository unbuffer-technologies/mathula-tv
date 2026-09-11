# Install — Four-block temporal mask / non-fatal rush

Extract this cumulative overlay over the current Mathula TV checkout.

```powershell
cd C:\mathula-tv

tar -xzf "$env:USERPROFILE\Downloads\mathula-native-four-block-semantic-window-rush-nonfatal-20260821T1345SAST.tar.gz" `
  -C C:\mathula-tv `
  --strip-components=1

python -m py_compile src\mathula_tv\native_dub.py src\mathula_tv\cli.py
```

## Run Pass 2

Pass 1 can be reused. The defaults are now a fixed four-block semantic window and one joint candidate combination per window.

```powershell
python -m mathula_tv.cli native-translate "$JOB_ID" `
  --pass 2 `
  --temporal-mask `
  --temporal-mask-residual-rounds 2 `
  --temporal-mask-tts-workers 6 `
  --force `
  --live-operation
```

You no longer need to specify `--temporal-mask-batch-size 6` or `--temporal-mask-candidates 3`. The new defaults are 4 and 1 respectively; temporal composition itself is fixed to four-block semantic windows.

## Render

```powershell
python -m mathula_tv.cli native-dub "$JOB_ID" `
  --timing-repair `
  --preferred-raw-speed-percent 6 `
  --max-natural-speed-percent 12 `
  --mouth-close-lag-tolerance-ms 40 `
  --max-timing-repair-rounds 2 `
  --timing-repair-batch-size 12 `
  --protected-gap-overflow `
  --max-protected-gap-overflow-ms 400 `
  --min-protected-pause-ms 120 `
  --no-hook-edit `
  --force `
  --live-operation
```

`--max-natural-speed-percent 12` is now the red console warning threshold. It is not a hard speed ceiling. A phrase requiring +20%, +45%, or more is rendered at that pitch-preserving acceleration and the pipeline continues.
