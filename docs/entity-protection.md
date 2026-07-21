# Entity Protection and Audio Intelligibility Gates

This document describes the entity protection and audio intelligibility quality gates implemented in the Mathula TV dubbing pipeline.

## Overview

The entity protection and intelligibility gates ensure that:
- Protected entities (names, organizations, places, etc.) are preserved deterministically throughout the dubbing pipeline
- Audio quality meets intelligibility thresholds at each production stage
- Quality gates block production if entity integrity or intelligibility fails

## Entity Registry

### Location
`config/entity_registry.json`

### Schema
- `schema_version`: `mathula-entity-registry-v1`
- `registry_id`: Unique identifier for the registry
- `entities`: Array of protected entity definitions

### Entity Structure
Each entity includes:
- `entity_id`: Unique identifier (e.g., `person_john_doe`)
- `entity_type`: One of `person`, `anonymous_witness`, `organisation`, `place`, `case`, `case_or_inquiry`, `alleged_network`, `phrase`
- `canonical_text`: Primary text form
- `display_text`: Text form for display/subtitles
- `aliases`: Alternative text forms for matching
- `stt_phrases`: Phrases for Azure STT phrase hints
- `spoken_forms`: Locale/voice-specific approved spoken forms
- `must_preserve`: Whether entity must be preserved in translation
- `translation_policy`: `protected_token` or `advisory`

### Entity Registry Library
`src/mathula_tv/entity_registry.py`

Key classes:
- `EntityRegistry`: Loads and validates the registry, matches entities in text
- `Entity`: Frozen dataclass representing a protected entity
- `EntityMatch`: Result of entity matching in source text
- `EntityBindingsArtifact`: Job-local entity binding artifact

## Entity Binding Pipeline

### 1. Entity Binding
**Command**: `mathula-tv bind-entities JOB_ID`

**Method**: `ProductionDubbingPipeline.bind_entities()`

**Process**:
1. Load entity registry
2. Load dubbing units from source transcript
3. Match entities in each unit's source text
4. Replace matched entities with stable placeholders: `[[MATHULA_ENTITY:entity_id]]`
5. Create job-local binding artifact: `working/jobs/<job_id>/analysis/entity_bindings.json`

**Output**:
- `analysis/entity_bindings.json`: Contains per-unit matches, protected text, placeholder mappings
- Job manifest updated with `entity_bindings_sha256` and `entity_registry_sha256`

### 2. Translation Protection
**Method**: `ProductionDubbingPipeline.translate()`

**Process**:
1. Load entity bindings
2. Protect source text with placeholders before sending to Claude
3. Add editorial constraints to Claude prompt:
   - "All [[MATHULA_ENTITY:...]] placeholders are immutable data tokens"
   - "No placeholder may move to another unit or be translated, inflected, split, merged, reordered across unrelated clauses, or removed"

### 3. Placeholder Validation
**Method**: After Claude translation response

**Process**:
1. Extract placeholders from `faithful_translation`, `spoken_text`, and `tts_text`
2. Validate that all placeholders appear exactly once in each text form
3. Check for missing, unexpected, or duplicate placeholders
4. If validation fails, raise error and block translation

### 4. Entity Restoration
**Method**: After placeholder validation

**Process**:
1. Restore `display_text` for `faithful_translation` and `spoken_text`
2. Restore approved spoken form for `tts_text` using locale/voice-specific spoken forms
3. Add entity metadata to units:
   - `protected_entities_expected`: Count of entities in unit
   - `protected_entities_restored`: Count successfully restored
   - `pronunciation_calibration_required`: Whether entity needs pronunciation calibration

## Dubbing Plan Enforcement

**Method**: `normalize_translation_units()` in `src/mathula_tv/dubbing_plan.py`

**Validations**:
- `protected_entities_expected` must equal `protected_entities_restored`
- No `protected_entities_missing`
- No `protected_entities_unexpected`

**Dubbing Plan Updates**:
- Include `entity_registry` and `entity_bindings` SHA256 references
- Per-unit entity metadata preserved in plan

## Audio Intelligibility Gates

### Intelligibility Auditor
**Module**: `src/mathula_tv/intelligibility.py`

**Key Classes**:
- `IntelligibilityAuditor`: Runs Azure STT intelligibility audits
- `UnitIntelligibilityObservation`: Per-unit measurements
- `IntelligibilityAuditReport`: Complete audit report

### Audit Stages
Intelligibility audits run after:
1. Azure TTS synthesis
2. OpenVoice voice conversion
3. Alignment
4. Final mix (optional)

### Audit Process
1. Transcribe audio using Azure STT (blind pass)
2. Calculate Word Error Rate (WER) against expected text
3. Check protected entity recognition in transcription
4. Aggregate metrics across all units
5. Compare against thresholds
6. Generate JSON and Markdown reports

### Thresholds
Configured via environment variables:
- `MATHULA_TV_MAX_CLEAN_UNIT_WER`: Default 0.35
- `MATHULA_TV_MAX_OPENVOICE_WER_DEGRADATION`: Default 0.10
- `MATHULA_TV_MAX_FINAL_MIX_WER_DEGRADATION`: Default 0.15
- `MATHULA_TV_MIN_PROTECTED_ENTITY_SIMILARITY`: Default 0.85

### Quality Gate Behavior
- If audit fails, stage is blocked and error is raised
- Reports written to `working/jobs/<job_id>/qc/intelligibility/<stage>/`
- Job manifest updated with audit state and SHA256

### CLI Command
**Command**: `mathula-tv audit-intelligibility JOB_ID --stage <stage>`

**Stages**: `azure-tts`, `openvoice`, `aligned`, `final-mix`

## Background and Mixing Safety

### Residual Dialogue QC
**Method**: `prepare_background()` in `src/mathula_tv/production_pipeline.py`

**Safety Check**:
- If residual dialogue QC fails and not `proof_of_concept`, raise error
- Error message: "Background contains residual dialogue that would interfere with dubbing"
- Use `proof_of_concept=True` for preview-only ducking
- Provide clean stem for production

### Background Manifest
Updated with:
- `proof_of_concept_ducking`: Boolean flag
- `residual_dialogue_status`: QC status

## Safe Rebuild Command

### Command
**Command**: `mathula-tv rebuild-with-entities JOB_ID`

**Method**: `ProductionDubbingPipeline.rebuild_with_entities()`

### Purpose
Safely rebuild downstream artifacts after entity registry changes while preserving source artifacts.

### Process
1. Validate source artifacts exist (dubbing units, transcript)
2. Check job state (must be in valid states)
3. Record what will be invalidated
4. Remove downstream artifacts:
   - Entity bindings
   - Translation
   - Dubbing plan
   - Azure TTS outputs
   - OpenVoice outputs
   - Alignment outputs
   - Background
   - Final mix
   - Intelligibility reports
5. Update job manifest (remove SHA256 references)
6. Reset state to `analysis_ready`
7. Clear completed stages (preserve `source_derivatives`, `dubbing_units`)
8. Record rebuild in attempt history

### Preserved Artifacts
- Dubbing units
- Source transcript
- Source derivatives (analysis audio, mix source audio)

### Next Step
Run `translate()` to re-bind entities and re-translate with updated registry.

## CLI Commands

### Entity Management
```bash
# Bind entities to a job
mathula-tv bind-entities JOB_ID [--force]

# Rebuild after entity registry changes
mathula-tv rebuild-with-entities JOB_ID [--force]
```

### Intelligibility Audit
```bash
# Audit intelligibility for a specific stage
mathula-tv audit-intelligibility JOB_ID --stage azure-tts
mathula-tv audit-intelligibility JOB_ID --stage openvoice
mathula-tv audit-intelligibility JOB_ID --stage aligned
mathula-tv audit-intelligibility JOB_ID --stage final-mix
```

## Configuration

### Environment Variables
```bash
# Intelligibility thresholds
MATHULA_TV_MAX_CLEAN_UNIT_WER=0.35
MATHULA_TV_MAX_OPENVOICE_WER_DEGRADATION=0.10
MATHULA_TV_MAX_FINAL_MIX_WER_DEGRADATION=0.15
MATHULA_TV_MIN_PROTECTED_ENTITY_SIMILARITY=0.85
```

### Settings
Updated in `src/mathula_tv/config.py`:
- `max_clean_unit_wer`: Maximum WER for clean Azure TTS units
- `max_openvoice_wer_degradation`: Maximum WER degradation from OpenVoice
- `max_final_mix_wer_degradation`: Maximum WER degradation in final mix
- `min_protected_entity_similarity`: Minimum protected entity recognition rate

## Artifacts

### New Artifacts
- `working/jobs/<job_id>/analysis/entity_bindings.json`: Job-local entity bindings
- `working/jobs/<job_id>/qc/intelligibility/<stage>/report.json`: Intelligibility audit JSON
- `working/jobs/<job_id>/qc/intelligibility/<stage>/report.md`: Intelligibility audit Markdown

### Updated Artifacts
- `working/jobs/<job_id>/dubbing/dubbing_plan.json`: Includes entity metadata
- Job manifest: Includes entity SHA256 references and intelligibility states

## Testing

### Unit Tests
- `tests/test_entity_registry.py`: Entity registry library tests
- `tests/test_intelligibility.py`: Intelligibility auditor tests

### Test Coverage
- Entity registry loading and validation
- Entity matching in text
- Placeholder protection and restoration
- Placeholder integrity validation
- WER calculation
- Protected entity recognition
- Intelligibility audit workflow
- Report generation

## Migration for Existing Jobs

For jobs created before entity protection:
1. Run `mathula-tv rebuild-with-entities JOB_ID` to rebuild with entity protection
2. This preserves source artifacts and re-runs translation with entity protection
3. Downstream artifacts are regenerated with entity integrity enforced

## Error Handling

### Entity Protection Errors
- **Placeholder validation failure**: Translation blocked, requires manual review
- **Entity integrity failure**: Dubbing plan blocked, requires entity registry update
- **Pronunciation calibration required**: Entity needs approved spoken form before TTS

### Intelligibility Errors
- **WER threshold exceeded**: Stage blocked, requires audio quality improvement
- **Entity recognition below threshold**: Stage blocked, requires entity calibration or audio improvement
- **STT backend not configured**: Audit skipped, warning logged

## Live Operation Safety

### Protected Job
The live compatibility job (`19ba6d69f1b84132ba4f20599101834a`) requires:
- `--live-operation` flag for all mutating commands
- GCS bucket configuration
- Application credentials

### Entity Registry Updates
When updating `config/entity_registry.json`:
1. Validate schema and content
2. Test with sample jobs
3. Run `rebuild-with-entities` for affected jobs
4. Verify entity matches and calibrations
5. Commit registry changes with job rebuilds

## Future Enhancements

### Planned Features
- Phrase-hinted Azure STT passes for diagnostic purposes
- Automatic pronunciation calibration suggestions
- Entity collision resolution UI
- Intelligibility trend analysis across jobs
- Automated entity suggestion from transcripts
