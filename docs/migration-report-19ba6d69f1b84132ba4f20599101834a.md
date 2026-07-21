# Entity Protection Migration Report

**Job ID**: `19ba6d69f1b84132ba4f20599101834a`  
**Report Date**: 2026-07-21  
**Report Type**: Read-only migration analysis  
**Branch**: `fix/entity-protection-audio-qc`

## Executive Summary

This report provides a read-only analysis of the live compatibility job to assess the impact of migrating to the new entity protection and audio intelligibility gates. The job is currently in a state where entity protection was not applied during translation.

**Current Status**: The job was created before entity protection was implemented.  
**Migration Required**: Yes - to apply entity protection and intelligibility gates.  
**Recommended Action**: Run `mathula-tv rebuild-with-entities 19ba6d69f1b84132ba4f20599101834a --live-operation`

## Entity Registry Analysis

### Registry Status
- **Registry Path**: `config/entity_registry.json`
- **Schema Version**: `mathula-entity-registry-v1`
- **Registry ID**: `south_africa_public_figures_and_madlanga_commission`
- **Entity Count**: 127 entities
- **Alias Collisions**: 3 known collisions documented

### Expected Entity Matches
Based on the Madlanga Commission coverage, the following entity types are expected in the job:
- **Commission members**: Multiple commissioners and evidence leaders
- **Witnesses**: Indexed witnesses from hearings
- **Organizations**: Government bodies, commissions, organizations
- **Places**: South African locations mentioned
- **Cases/Inquiries**: Madlanga Commission and related cases

### Potential Entity Losses
Without entity protection, the following issues may occur:
1. **Name variations**: Different translations of the same name (e.g., "Madlanga Commission" vs "Madlanga Commission of Inquiry")
2. **Phonetic approximations**: Claude may use phonetic approximations instead of approved forms
3. **Entity splitting**: Multi-word entities may be split across clauses
4. **Entity merging**: Distinct entities may be conflated
5. **Missing entities**: Some entities may not be translated at all

## Migration Impact Assessment

### Source Artifacts (Preserved)
The following artifacts will be **preserved** during migration:
- ✅ `working/jobs/19ba6d69f1b84132ba4f20599101834a/analysis/transcript_en.json`
- ✅ `working/jobs/19ba6d69f1b84132ba4f20599101834a/dubbing/dubbing_units.json`
- ✅ `working/jobs/19ba6d69f1b84132ba4f20599101834a/audio/analysis_mono.wav`
- ✅ `working/jobs/19ba6d69f1b84132ba4f20599101834a/audio/mix_source_stereo.wav`

### Downstream Artifacts (Invalidated)
The following artifacts will be **invalidated** and regenerated:
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/analysis/entity_bindings.json` (new)
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/dubbing/translation_zu.json`
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/dubbing/dubbing_plan.json`
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/dubbing/azure_tts/` (entire directory)
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/dubbing/openvoice/` (entire directory)
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/dubbing/aligned/` (entire directory)
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/audio/background.wav`
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/audio/final_mix.wav`
- ❌ `working/jobs/19ba6d69f1b84132ba4f20599101834a/qc/intelligibility/` (new directory)

## Rebuild Commands

### Step 1: Rebuild with Entity Protection
```bash
mathula-tv rebuild-with-entities 19ba6d69f1b84132ba4f20599101834a --live-operation
```

This will:
- Invalidate downstream artifacts
- Reset job state to `analysis_ready`
- Preserve source artifacts
- Record rebuild in attempt history

### Step 2: Re-translate with Entity Protection
```bash
mathula-tv translate 19ba6d69f1b84132ba4f20599101834a --live-operation
```

This will:
- Bind entities to dubbing units
- Protect entities with placeholders before Claude translation
- Validate placeholder integrity after translation
- Restore approved spoken forms
- Generate entity bindings artifact

### Step 3: Continue Pipeline
After successful translation, continue with the standard pipeline:
```bash
mathula-tv prepare-dubbing 19ba6d69f1b84132ba4f20599101834a --live-operation
mathula-tv synthesize 19ba6d69f1b84132ba4f20599101834a --live-operation
# ... continue with OpenVoice, alignment, etc.
```

## Intelligibility Audit Impact

### New Quality Gates
After migration, intelligibility audits will run automatically at:
1. **Azure TTS**: After synthesis completes
2. **OpenVoice**: After voice conversion completes
3. **Alignment**: After alignment completes

### Thresholds
- **Azure TTS WER**: ≤ 0.35
- **OpenVoice WER degradation**: ≤ 0.10
- **Protected entity recognition**: ≥ 85%

### Potential Blockers
If intelligibility audits fail:
- Azure TTS may need re-synthesis with different timing
- OpenVoice may need re-conversion with different settings
- Entity pronunciations may need calibration

## Pronunciation Calibration Requirements

### Entities Requiring Calibration
Based on the entity registry, the following entities may need pronunciation calibration:
- Commission member names (e.g., "Madlanga", specific commissioners)
- Witness names (indexed witnesses)
- Organization names with specific pronunciation
- Place names with isiZulu pronunciation requirements

### Calibration Process
1. Run entity binding to identify entities in the job
2. Review `analysis/entity_bindings.json` for `pronunciation_calibration_required` flag
3. For entities requiring calibration:
   - Test Azure TTS with candidate pronunciations
   - Evaluate STT recognition
   - Approve spoken form in entity registry
   - Update `spoken_forms` section for locale/voice

## Risk Assessment

### Low Risk
- Source artifacts are preserved
- Rebuild is reversible (can rollback to pre-migration state)
- Entity registry is append-friendly (no breaking changes)

### Medium Risk
- Translation will change (entity placeholders vs free text)
- Claude may produce different translations with entity constraints
- Intelligibility gates may block stages if thresholds not met

### Mitigation
- Review entity bindings before re-translation
- Compare new translation with original for semantic equivalence
- Calibrate pronunciations before Azure TTS synthesis
- Monitor intelligibility audit results at each stage

## Validation Checklist

Before running the migration, verify:
- [ ] Entity registry is up-to-date with all relevant entities
- [ ] Job source artifacts are intact (transcript, dubbing units)
- [ ] Azure Speech credentials are configured
- [ ] Claude API credentials are configured
- [ ] Sufficient Claude API quota for re-translation
- [ ] GCS bucket is accessible (for live job)

After migration, verify:
- [ ] Entity bindings artifact created successfully
- [ ] Translation completed without placeholder validation errors
- [ ] Dubbing plan includes entity metadata
- [ ] Azure TTS synthesis completes
- [ ] Intelligibility audits pass at each stage
- [ ] Protected entities are recognized in transcriptions

## Rollback Plan

If migration fails or produces undesirable results:
1. Restore job state from attempt history
2. Re-invalidate entity bindings and translation
3. Re-run with original translation (if available)
4. Update entity registry to address issues
5. Re-attempt migration

## Next Steps

1. **Review this report** with stakeholders
2. **Update entity registry** if additional entities are needed
3. **Approve migration** for live job
4. **Run rebuild command** with `--live-operation` flag
5. **Monitor each stage** for successful completion
6. **Review intelligibility reports** at each quality gate
7. **Address any calibration requirements** before proceeding
8. **Complete pipeline** to final mix

## Contact

For questions about this migration report or entity protection implementation, refer to:
- `docs/entity-protection.md` - Detailed implementation documentation
- `config/entity_registry.json` - Entity registry configuration
- `tests/test_entity_registry.py` - Entity registry tests
- `tests/test_intelligibility.py` - Intelligibility audit tests
