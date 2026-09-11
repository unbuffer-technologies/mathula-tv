# Install — one-call temporal-mask self-QA

```powershell
cd C:\mathula-tv
tar -xzf "$env:USERPROFILE\Downloads\mathula-native-four-block-window-one-call-self-qa-20260821T2128SAST.tar.gz" `
  -C C:\mathula-tv `
  --strip-components=1
python -m py_compile src\mathula_tv\native_dub.py src\mathula_tv\cli.py
python -m pytest -q tests\test_native_temporal_mask_one_call_self_qa.py tests\test_native_four_block_temporal_policy.py
```

Then run Pass 2 with temporal-mask. `--temporal-mask-residual-rounds` and `--temporal-mask-tts-workers` are compatibility options only and do not trigger Python repair calls.
