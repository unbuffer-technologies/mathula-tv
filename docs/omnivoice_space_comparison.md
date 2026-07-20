# Deprecated OmniVoice migration note

> **Deprecated — not registered — not selectable — not production.**

This file remains only so old handoffs and links do not imply that OmniVoice is still active. OmniVoice has been removed from production configuration, backend selection, CLI dispatch, worker discovery, and the supported architecture. It is unreachable during the Azure TTS → OpenVoice compatibility phase and is never a fallback.

The previous Space comparison and calibration procedure are obsolete. Do not run `calibrate-omnivoice`, do not run the old OmniVoice notebook, and do not interpret remaining implementation files/tests/notebook cells as supported behavior.

After the OpenVoice compatibility gate passes—or an explicit later cleanup authorises it—delete the remaining OmniVoice implementation files, dependencies, notebooks, tests, and this migration note. Until then, preserve them only as clearly deprecated migration evidence.

F5-TTS is also not part of Mathula TV. The intended path is Azure TTS for isiZulu pronunciation followed by OpenVoice for speaker-identity conversion on a pinned Colab or Kaggle GPU worker. The compatibility gate is currently `pending`; no live OpenVoice success is claimed.
