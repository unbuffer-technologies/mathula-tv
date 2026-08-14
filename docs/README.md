# Mathula TV production dubbing guides

The Azure TTS → OpenVoice compatibility gate is currently `pending`; these guides describe the implemented, offline-testable architecture and the opt-in work needed to validate it. They do not document a live compatibility pass.

| Guide | Purpose |
| --- | --- |
| [Architecture](architecture.md) | End-to-end responsibilities, invariants, and artifacts |
| [State machine](state-machine.md) | Production lifecycle and legacy migration |
| [CLI](cli.md) | Normal commands and exact four-turn live runbook |
| [Azure GPT provider](gpt-provider.md) | Translation, repairs, metadata, and safety |
| [Azure STT](azure-stt.md) | Authoritative transcription/diarization |
| [Azure TTS](azure-tts.md) | Voice discovery, safe SSML, synthesis, and rate search |
| [Pronunciation](pronunciation.md) | Three text forms and reviewed substitutions |
| [OpenVoice](openvoice.md) | Conversion boundary, calibrations, references, and leases |
| [Colab](colab.md) / [Kaggle](kaggle.md) | Pinned GPU worker procedures |
| [Compatibility gate](compatibility-gate.md) | Measurements, gate states, A/B package, and pass rules |
| [Timing](timing.md) | Unit windows, converted-duration authority, overlap |
| [Background audio](background-audio.md) | Dialogue removal, separation, residual-English QC, mixing |
| [Rendering](rendering.md) | Subtitles, final mix, MP4 validation, and artifacts |
| [QC](qc.md) | Automated signals and readiness calculation |
| [Human review](human-review.md) | Voice rights, editorial/language review, publication boundary |
| [Troubleshooting](troubleshooting.md) | Actionable, secret-safe failure handling |

F5-TTS is not part of Mathula TV. OmniVoice is deprecated and unreachable pending deletion. Azure OpenAI GPT is the sole runtime AI provider; Azure STT and Azure TTS are authoritative for speaker labels and isiZulu base pronunciation respectively; OpenVoice is the intended identity converter; Pyannote is diagnostic only. Mathula TV never uploads to YouTube.
