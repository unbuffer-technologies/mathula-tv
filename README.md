# Mathula TV

Mathula TV is a resumable, human-reviewed pipeline for dubbing English South African news video into isiZulu. The production design uses Azure Speech-to-Text for transcription and authoritative job-local diarization, Claude Opus 4.8 for contextual translation and editorial metadata, Azure Text-to-Speech for isiZulu pronunciation, and OpenVoice for post-TTS speaker-identity conversion on a Colab or Kaggle GPU worker.

Codex is the engineering agent that implements this repository; it is not a runtime content provider. Claude Opus 4.8 is the configured production runtime AI provider.

> **Current status:** the Azure TTS → OpenVoice compatibility gate is **pending**. The repository contains offline-testable production boundaries and orchestration components, but this checkout makes no claim that a live Claude request, live Azure TTS request, live OpenVoice conversion, compatibility pass, publication-quality background, or approved production video has completed. Human voice-rights, editorial, factual, fluent-isiZulu, timing, and audio review are mandatory.

Mathula TV never uploads to YouTube.

## Architecture

```text
source video
  ├─ audio/analysis_mono.wav ──> Azure STT (en-ZA + authoritative diarization)
  └─ audio/mix_source_stereo.wav ───────────────────────────────────────┐
                                      ↓                                 │
dubbing units ─> Claude Opus 4.8 ─> pronunciation plan ─> Azure TTS     │
                                      ↓                                 │
                    OpenVoice GPU conversion (job-local speaker reels)  │
                                      ↓                                 │
                    measured alignment + overlap-safe speaker tracks    │
                                      ↓                                 │
            clean/separated/reconstructed background <─────────────────┘
                                      ↓
              mix + subtitles + MP4 + QC reports + human review
```

The rollout is deliberately two-phase:

1. **Phase A — compatibility proof:** run the existing four-turn job through Azure TTS and OpenVoice, compare raw Azure, converted, aligned, and original-reference audio, calibrate thresholds, and complete human reviews.
2. **Phase B — production automation:** permit the full alignment, background, mixing, subtitle, render, and QC flow only around the validated path.

F5-TTS is not part of this repository or its fallback chain. Legacy OmniVoice files may remain temporarily as migration evidence, but OmniVoice is deprecated, unregistered, unselectable, and not production; it is pending deletion after the compatibility path passes or an explicit cleanup. There is no silent fallback from Claude to Azure OpenAI, from OpenVoice to OmniVoice, or from converted speech to raw Azure TTS.

## Install

Python 3.11 or later and FFmpeg/ffprobe are required on the normal server.

```bash
cd /home/mokgethwa/mathula-tv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[cloud,synthesis,dev]'
cp .env.example .env
python -m mathula_tv.cli --help
```

Configure `.env` locally. Do not commit credentials, service-account JSON, tokens, signed URLs, private transcripts, speaker audio, embeddings, checkpoints, or generated media. OpenVoice/CUDA dependencies belong in the pinned GPU worker, not the normal server environment.

## Start a job

```bash
python -m mathula_tv.cli submit /path/to/news_clip.mp4 --target-language zu-ZA
python -m mathula_tv.cli inspect JOB_ID
python -m mathula_tv.cli process JOB_ID
```

`process` is a resumable guidance command: it inspects the current state and prints the next action, stopping truthfully when an external GPU worker or human review is required. It does not bypass OpenVoice or a failed/pending compatibility gate. `--skip-openvoice` may be used only for an explicitly labelled `azure_tts_only_review` internal artifact and never implies voice cloning.

The known four-turn proof job is `19ba6d69f1b84132ba4f20599101834a`. Do not run it casually: every command that reads or mutates that live job must use `--live-operation`, valid credentials, and the exact safe sequence in [docs/cli.md](docs/cli.md). The sequence preserves its approved isiZulu content and does not call Claude unless explicitly authorised.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and artifact layout](docs/architecture.md)
- [State machine and legacy migration](docs/state-machine.md)
- [CLI and exact four-turn runbook](docs/cli.md)
- [Claude provider](docs/claude-provider.md)
- [Azure STT](docs/azure-stt.md) and [Azure TTS](docs/azure-tts.md)
- [Pronunciation and three text forms](docs/pronunciation.md)
- [OpenVoice worker](docs/openvoice.md), [Colab](docs/colab.md), and [Kaggle](docs/kaggle.md)
- [Compatibility gate](docs/compatibility-gate.md) and [timing](docs/timing.md)
- [Background audio](docs/background-audio.md), [rendering](docs/rendering.md), and [QC](docs/qc.md)
- [Human review](docs/human-review.md) and [troubleshooting](docs/troubleshooting.md)
- [Engineering handoff](PROJECT_STATE.md)

Run offline checks with `pytest`. Credentialed, paid, GCS-mutating, and GPU integration tests are opt-in only; see [docs/cli.md](docs/cli.md).
