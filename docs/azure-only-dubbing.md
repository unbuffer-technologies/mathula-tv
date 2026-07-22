# Azure-Only Dubbing Path

This document describes the Azure-only dubbing path for Mathula TV, a CPU-only professional TTS dubbing workflow that preserves speaker context without requiring GPU resources or OpenVoice voice conversion.

## Overview

The Azure-only dubbing path provides a complete, production-ready dubbing pipeline using only Azure TTS for speech generation. This mode:

- **GPU-independent**: Runs entirely on CPU
- **Deterministic voice assignment**: Resolves Azure voices before synthesis based on speaker context
- **Speaker context preservation**: Maintains speaker identity throughout the pipeline
- **QC gates**: Includes mandatory quality control at each stage
- **OpenVoice-optional**: Preserves OpenVoice as an optional enhancement path

## Architecture

### Speaker Context Resolution

The pipeline builds an authoritative speaker context artifact that captures:

1. **Speaker identity**: From authoritative speaker registry (if available)
2. **Acoustic voice family**: CPU-only pitch analysis (masculine/feminine/unknown)
3. **TTS voice family**: Determined from identity or acoustic analysis
4. **Azure voice assignment**: Mapped from voice family to target locale voices

### Pipeline Stages

1. **Speaker Profiles** (`speaker_profiles.json`)
   - Extracts speakers from transcript and diarization
   - Resolves identity from authoritative sources
   - Records acoustic analysis results
   - Determines TTS voice family

2. **Acoustic Analysis** (`acoustic_analysis.json`)
   - CPU-only pitch (F0) analysis using autocorrelation
   - Classifies voice family as masculine/feminine/unknown
   - Provides confidence metrics

3. **Voice Resolution** (`azure_voice_assignments.json`)
   - Maps TTS voice family to Azure voices
   - Uses target locale voice catalogue
   - Supports fallback for unresolved speakers

4. **Dubbing Plan** (`dubbing/dubbing_plan.json`)
   - Built with `azure_only=True` flag
   - Marks OpenVoice as "skipped"
   - Records configuration mode

5. **Azure TTS Synthesis** (`dubbing/azure_tts/`)
   - Uses resolved voice assignments
   - Applies timing search for duration constraints
   - Includes intelligibility audit

6. **Azure TTS QC** (`dubbing/azure_tts/qc_report.json`)
   - Validates WAV decodability
   - Checks for effective silence
   - Detects clipping
   - Validates protected entities

7. **Alignment** (`dubbing/aligned/manifest.json`)
   - Aligns directly from Azure TTS output
   - Skips OpenVoice conversion
   - Readiness: `azure_tts_production_ready`

8. **Background Preparation** (`audio/background_manifest.json`)
   - Accepts `azure_tts_production_ready` alignment
   - Residual dialogue QC
   - CPU-only separation (if needed)

9. **Final Mix** (`audio/final_mix.wav`)
   - Mixes dialogue with background
   - Loudness normalization (LUFS, LRA, True Peak)
   - Final-mix intelligibility audit

10. **Review Video** (`render/final_dubbed.mp4`)
    - Combines final mix with source video
    - Generates subtitles separately
    - No subtitle burning

## CLI Usage

### Full Pipeline

```bash
mathula-tv dub-azure <job_id>
```

### Skip to Stage

```bash
mathula-tv dub-azure <job_id> --skip-to alignment
```

### Options

- `--fallback-allowed`: Allow fallback voice resolution for unresolved speakers
- `--min-confidence`: Minimum confidence threshold (default: 0.5)
- `--force`: Force re-run of stages

## Artifact Paths

All artifacts follow the deterministic path structure defined in `DubbingArtifacts`:

- `analysis/speaker_profiles.json` - Speaker context artifact
- `analysis/acoustic_analysis.json` - Acoustic analysis results
- `analysis/azure_voice_assignments.json` - Voice assignments
- `dubbing/dubbing_plan.json` - Dubbing plan (azure_only mode)
- `dubbing/azure_tts/manifest.json` - Azure TTS manifest
- `dubbing/azure_tts/qc_report.json` - Azure TTS QC report
- `dubbing/aligned/manifest.json` - Alignment manifest
- `audio/background_manifest.json` - Background manifest
- `audio/final_mix.wav` - Final mixed audio
- `render/final_dubbed.mp4` - Review video

## Voice Catalogue

### isiZulu (zu-ZA)

- `zu-ZA-ThembaNeural` - Masculine voice family
- `zu-ZA-ThandoNeural` - Feminine voice family

Additional locales can be added by extending `DEFAULT_CATALOGUES` in `azure_voice_catalogue.py`.

## QC Gates

### Azure TTS QC

- **WAV decodability**: All files must be valid mono PCM16
- **Effective silence**: Ratio must not exceed threshold (default: 0.8)
- **Clipping**: Clipping ratio must not exceed threshold (default: 0.01)
- **Protected entities**: All placeholders must be present in TTS text

### Intelligibility Audit

- **Azure TTS stage**: Blind WER must be below threshold
- **Aligned stage**: WER degradation must be within limits
- **Final-mix stage**: WER degradation must be within limits

## Configuration

### Settings

- `azure_tts_voices` - Tuple of available Azure voices
- `azure_tts_default_voice` - Default voice fallback
- `azure_rate_min_percent` - Minimum rate adjustment (default: -12)
- `azure_rate_max_percent` - Maximum rate adjustment (default: 15)
- `max_clean_unit_wer` - Maximum WER for clean units (default: 0.35)
- `max_final_mix_wer_degradation` - Maximum WER degradation (default: 0.15)

### Environment Variables

- `MATHULA_TV_AZURE_TTS_VOICES` - Comma-separated voice list
- `MATHULA_TV_AZURE_TTS_DEFAULT_VOICE` - Default voice

## Comparison with Standard Path

| Feature | Standard Path | Azure-Only Path |
|---------|--------------|-----------------|
| GPU Required | Yes (OpenVoice) | No |
| Voice Conversion | OpenVoice | None |
| Alignment Source | OpenVoice | Azure TTS |
| Compatibility Gate | Required | Skipped |
| Speaker Context | Preserved | Preserved |
| QC Gates | Full | Full (Azure-specific) |

## Migration from Standard Path

Jobs in standard mode can be migrated to azure-only mode:

1. Ensure Azure TTS synthesis is complete
2. Run `dub-azure --skip-to alignment` to start alignment
3. Background, mix, and render stages are compatible

## Implementation Notes

### Acoustic Analysis

The CPU-only acoustic analysis uses autocorrelation-based pitch estimation:

- Frame size: 25ms
- Frame shift: 10ms
- Min F0: 75Hz
- Max F0: 400Hz
- Masculine threshold: 165Hz
- Feminine threshold: 255Hz

This is a heuristic classification, not definitive gender determination.

### Voice Resolution Priority

1. Authoritative speaker registry (if available)
2. Entity/person metadata (diagnostic only)
3. Acoustic analysis (if confidence >= 0.7)
4. Fallback to default voice (if allowed)

### Alignment Mode

Azure-only alignment uses the existing `align_converted_unit` function with:

- Source: Azure TTS output directly
- No OpenVoice conversion
- Readiness: `azure_tts_production_ready`

## Testing

Run tests for Azure-only components:

```bash
pytest tests/test_speaker_profiles.py
pytest tests/test_azure_voice_catalogue.py
pytest tests/test_voice_resolver.py
```

## Future Enhancements

- **Speaker registry integration**: Support for authoritative speaker identity sources
- **Extended locales**: Add more target locale voice catalogues
- **Advanced acoustic models**: Optional GPU-based acoustic analysis
- **Entity-aware voice selection**: Consider entity context in voice assignment
