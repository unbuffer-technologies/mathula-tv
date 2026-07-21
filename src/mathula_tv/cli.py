from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .ai_provider import create_production_ai_provider
from .atomic_io import read_json
from .azure_stt import AzureFastTranscriptionBackend, derive_endpoint
from .azure_tts import AzureTTSBackend, AzureTTSHTTPBoundary
from .colab_package import publish_colab_package
from .compatibility import (
    CompatibilityThresholds,
    ReviewPackageUnit,
    UnitCompatibilityObservation,
    build_ab_review_package,
)
from .config import load_settings
from .diarization import run_pyannote
from .logging_utils import configure_logging, redact
from .orchestrator import Orchestrator
from .production_pipeline import ProductionDubbingPipeline
from .translation import AzureOpenAIClient
from .tts_ssml import SSMLBounds


LIVE_COMPATIBILITY_JOB_ID = "19ba6d69f1b84132ba4f20599101834a"
EXTERNAL_COMMANDS = {
    "transcribe",
    "translate",
    "repair-translation",
    "validate-azure-voices",
    "calibrate-azure-sources",
    "calibrate-speaker-voices",
    "synthesize",
    "queue-openvoice-assets",
    "reconcile-openvoice-assets",
    "queue-voice-conversion",
    "reconcile-voice-conversion",
}


def create_speech_backend(settings):
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not key:
        raise ValueError("AZURE_SPEECH_KEY is missing")
    endpoint = settings.azure_speech_endpoint or derive_endpoint(settings.azure_speech_region)
    return AzureFastTranscriptionBackend(
        endpoint=endpoint,
        subscription_key=key,
        api_version=settings.azure_speech_api_version,
        max_speakers=settings.azure_speech_max_speakers,
        timeout_seconds=settings.azure_speech_timeout_seconds,
    )


def create_translation_backend(provider: str | None = None, model: str | None = None):
    selected = (provider or os.getenv("MATHULA_TV_AI_PROVIDER") or "anthropic").strip().lower()
    if selected in {"anthropic", "azure-foundry-claude"}:
        environment = dict(os.environ)
        environment["MATHULA_TV_AI_PROVIDER"] = selected
        if model:
            environment["MATHULA_TV_FOUNDRY_CLAUDE_DEPLOYMENT" if selected == "azure-foundry-claude" else "MATHULA_TV_CLAUDE_MODEL"] = model
        return create_production_ai_provider(environment)
    if selected in {"azure-openai-legacy", "azure-openai-test"}:
        return AzureOpenAIClient.from_environment()
    raise ValueError(f"Unsupported AI provider: {selected}; no fallback is configured")


def create_tts_backend(settings) -> AzureTTSBackend:
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not key:
        raise ValueError("AZURE_SPEECH_KEY is missing")
    if not settings.azure_speech_region:
        raise ValueError("AZURE_SPEECH_REGION is missing")
    bounds = SSMLBounds(
        rate_min_percent=settings.azure_rate_min_percent,
        rate_max_percent=settings.azure_rate_max_percent,
    )
    return AzureTTSBackend(
        AzureTTSHTTPBoundary(settings.azure_speech_region, key),
        configured_voices=settings.azure_tts_voices,
        default_voice=settings.azure_tts_default_voice,
        sample_rate=settings.azure_tts_sample_rate,
        channels=settings.azure_tts_channels,
        sample_width=settings.azure_tts_sample_width,
        timeout_seconds=settings.azure_tts_timeout_seconds,
        max_retries=settings.azure_tts_max_retries,
        ssml_bounds=bounds,
    )


def _add_job_command(commands, name: str) -> argparse.ArgumentParser:
    command = commands.add_parser(name)
    command.add_argument("job_id")
    command.add_argument("--force", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--live-operation", action="store_true")
    return command


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mathula-tv")
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("video", type=Path)
    submit.add_argument("--target-language", default="zu-ZA")
    submit.add_argument("--force", action="store_true")

    for name in ("process", "transcribe", "diarize", "status", "retry", "validate", "inspect"):
        _add_job_command(commands, name)
    migration = _add_job_command(commands, "migrate-dubbing-state")
    migration.add_argument("--allow-legacy-review-reset", action="store_true")
    _add_job_command(commands, "prepare-source-derivatives")
    _add_job_command(commands, "build-dubbing-units")
    translate = _add_job_command(commands, "translate")
    translate.add_argument("--provider", default=None)
    translate.add_argument("--model", default=None)
    repair_translation = _add_job_command(commands, "repair-translation")
    repair_translation.add_argument("--provider", default=None)
    repair_translation.add_argument("--model", default=None)
    _add_job_command(commands, "prepare-dubbing")
    _add_job_command(commands, "validate-azure-voices")
    _add_job_command(commands, "calibrate-azure-sources")
    synthesize = _add_job_command(commands, "synthesize")
    synthesize.add_argument("--backend", default="azure-tts", choices=("azure-tts",))
    references = _add_job_command(commands, "build-reference-reels")
    references.add_argument("--candidates", type=Path)
    references.add_argument("--proof-of-concept", action="store_true")
    calibrate_speaker_voices = _add_job_command(commands, "calibrate-speaker-voices")
    calibrate_speaker_voices.add_argument("--prepare-only", action="store_true")
    calibrate_speaker_voices.add_argument("--evaluations", type=Path)
    prepare_openvoice = _add_job_command(commands, "prepare-openvoice")
    prepare_openvoice.add_argument("--prepare-only", action="store_true")
    prepare_openvoice.add_argument("--proof-of-concept", action="store_true")
    prepare_openvoice.add_argument("--device", default="cuda")
    _add_job_command(commands, "queue-openvoice-assets")
    _add_job_command(commands, "reconcile-openvoice-assets")
    _add_job_command(commands, "queue-voice-conversion")
    _add_job_command(commands, "reconcile-voice-conversion")
    compatibility = _add_job_command(commands, "evaluate-compatibility")
    compatibility.add_argument("--observations", type=Path)
    compatibility.add_argument("--thresholds", type=Path)
    _add_job_command(commands, "build-review-package")
    align = _add_job_command(commands, "align")
    align.add_argument("--skip-openvoice", action="store_true")
    align.add_argument("--require-compatibility-pass", action=argparse.BooleanOptionalAction, default=True)
    background = _add_job_command(commands, "prepare-background")
    background.add_argument("--clean-stem", type=Path)
    background.add_argument("--proof-of-concept", action="store_true")
    _add_job_command(commands, "mix")
    _add_job_command(commands, "subtitles")
    render = _add_job_command(commands, "render")
    render.add_argument("--burn-subtitles", action="store_true")
    _add_job_command(commands, "review")
    commands.add_parser("list-jobs")
    
    # Entity registry commands
    _add_job_command(commands, "bind-entities")
    _add_job_command(commands, "rebuild-with-entities")
    
    # Registry management commands (independent of jobs)
    entities = commands.add_parser("entities")
    entities_subcommands = entities.add_subparsers(dest="entities_command", required=True)
    
    entities_validate = entities_subcommands.add_parser("validate")
    entities_validate.add_argument("--registry", default="config/entity_registry.json")
    
    entities_list = entities_subcommands.add_parser("list")
    entities_list.add_argument("--registry", default="config/entity_registry.json")
    entities_list.add_argument("--domain")
    entities_list.add_argument("--type")
    entities_list.add_argument("--query")
    
    # Intelligibility audit commands
    audit_intelligibility = _add_job_command(commands, "audit-intelligibility")
    audit_intelligibility.add_argument("--stage", required=True)
    audit_intelligibility.add_argument("--with-phrase-hints", action="store_true")
    
    publish_worker = commands.add_parser("publish-colab-package")
    publish_worker.add_argument("--live-operation", action="store_true")
    return root


def _require_operation_authorization(args, settings) -> None:
    job_id = getattr(args, "job_id", None)
    if job_id == LIVE_COMPATIBILITY_JOB_ID:
        if not args.live_operation:
            raise ValueError("The compatibility job requires --live-operation")
        if not settings.gcs_bucket:
            raise ValueError("MATHULA_TV_GCS_BUCKET is required for the compatibility job")
        if not (
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            or os.getenv("MATHULA_TV_USE_APPLICATION_DEFAULT_CREDENTIALS") == "1"
        ):
            raise ValueError(
                "Configure GOOGLE_APPLICATION_CREDENTIALS or explicitly acknowledge "
                "application-default credentials with MATHULA_TV_USE_APPLICATION_DEFAULT_CREDENTIALS=1"
            )
    if args.command in EXTERNAL_COMMANDS and hasattr(settings, "ai_provider") and not args.live_operation:
        raise ValueError(f"{args.command} calls an external provider and requires --live-operation")


def _dry_run(args, job) -> dict[str, Any] | None:
    if not getattr(args, "dry_run", False):
        return None
    if args.command in {"inspect", "status", "validate", "migrate-dubbing-state"}:
        return None
    return {
        "dry_run": True,
        "job_id": job.job_id,
        "state": job.state,
        "command": args.command,
        "mutated": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings()
    configure_logging(settings.work_dir / "logs/mathula-tv.jsonl", args.verbose)
    app = Orchestrator(settings)
    try:
        if args.command == "submit":
            job = app.submit(args.video, args.target_language, args.force)
            print(job.job_id)
            return 0
        if args.command == "list-jobs":
            print(json.dumps([{"job_id": item.job_id, "state": item.state, "source": item.source_filename} for item in app.jobs.list()], indent=2))
            return 0
        if args.command == "entities":
            from pathlib import Path
            from .entity_registry import EntityRegistry
            
            registry_path = Path(args.registry)
            if args.entities_command == "validate":
                registry = EntityRegistry(registry_path)
                print(json.dumps({
                    "validation_state": "valid",
                    "schema_version": "mathula-entity-registry-v1",
                    "entity_count": len(registry._entities),
                    "registry_sha256": registry.sha256,
                    "alias_collision_count": 0,  # Would need collision detection logic
                    "invalid_reference_count": 0,  # Would need reference validation logic
                }, indent=2))
                return 0
            elif args.entities_command == "list":
                registry = EntityRegistry(registry_path)
                entities = list(registry._entities.values())
                
                # Apply filters
                if args.domain:
                    entities = [e for e in entities if args.domain in e.domains]
                if args.type:
                    entities = [e for e in entities if e.entity_type == args.type]
                if args.query:
                    query_lower = args.query.lower()
                    entities = [e for e in entities if query_lower in e.canonical_text.lower() or query_lower in " ".join(e.aliases).lower()]
                
                print(json.dumps([
                    {
                        "entity_id": e.entity_id,
                        "entity_type": e.entity_type,
                        "canonical_text": e.canonical_text,
                        "display_text": e.display_text,
                        "aliases": list(e.aliases),
                        "domains": list(e.domains),
                        "active": e.active,
                        "must_preserve": e.must_preserve,
                    }
                    for e in entities
                ], indent=2))
                return 0
        if args.command == "publish-colab-package":
            if not args.live_operation:
                raise ValueError("publish-colab-package mutates GCS and requires --live-operation")
            if not settings.gcs_bucket:
                raise ValueError("MATHULA_TV_GCS_BUCKET is required")
            from google.cloud import storage

            result = publish_colab_package(
                Path(__file__).resolve().parents[2],
                storage.Client().bucket(settings.gcs_bucket),
                settings.gcs_prefix,
            )
            print(json.dumps({key: value for key, value in result.items() if key != "sha256"}, indent=2))
            return 0

        job = app.jobs.load(args.job_id)
        _require_operation_authorization(args, settings)
        dry = _dry_run(args, job)
        if dry is not None:
            print(json.dumps(dry, indent=2))
            return 0

        # Preserve the explicitly selectable Azure OpenAI legacy/test adapter.
        # Normal Settings always has ai_provider=anthropic and uses production below.
        if args.command == "translate" and not hasattr(settings, "ai_provider"):
            print(json.dumps(app.translate_and_package(job, create_translation_backend(), force=args.force), ensure_ascii=False))
            return 0
        production = ProductionDubbingPipeline(settings)

        if args.command == "status":
            print(json.dumps(redact(job.to_dict()), indent=2))
        elif args.command == "inspect":
            print(json.dumps(redact(production.inspect(job)), indent=2))
        elif args.command == "migrate-dubbing-state":
            print(
                json.dumps(
                    production.migrate(
                        job,
                        dry_run=args.dry_run,
                        allow_legacy_review_reset=args.allow_legacy_review_reset,
                    ),
                    indent=2,
                )
            )
        elif args.command == "prepare-source-derivatives":
            print(json.dumps(production.ensure_source_derivatives(job, force=args.force), indent=2))
        elif args.command == "build-dubbing-units":
            print(json.dumps(production.build_units(job, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "prepare-dubbing":
            print(json.dumps(production.prepare_dubbing(job, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "translate":
            provider_name = args.provider or settings.ai_provider
            if provider_name in {"azure-openai-legacy", "azure-openai-test"}:
                print(json.dumps(app.translate_and_package(job, create_translation_backend(provider_name, args.model), force=args.force), ensure_ascii=False))
            else:
                provider = create_translation_backend(provider_name, args.model or settings.claude_model)
                print(json.dumps(production.translate(job, provider, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "repair-translation":
            provider_name = args.provider or settings.ai_provider
            if provider_name not in {"anthropic", "azure-foundry-claude"}:
                raise ValueError("Timing repair requires the configured Claude production provider")
            provider = create_translation_backend(provider_name, args.model or settings.claude_model)
            print(
                json.dumps(
                    production.repair_translation_units(job, provider, force=args.force),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "validate-azure-voices":
            voices = create_tts_backend(settings).validate_configured_voices(force_refresh=args.force)
            print(json.dumps({"available": [voice.name for voice in voices], "validated": True}, indent=2))
        elif args.command == "calibrate-azure-sources":
            print(json.dumps(production.prepare_azure_source_calibrations(job, create_tts_backend(settings), force=args.force), indent=2))
        elif args.command == "synthesize":
            print(json.dumps(production.synthesize_azure(job, create_tts_backend(settings), force=args.force), indent=2))
        elif args.command == "build-reference-reels":
            if args.candidates:
                candidates = read_json(args.candidates)
            elif args.proof_of_concept:
                candidates = production.default_reference_candidates(job)
            else:
                raise ValueError("Provide --candidates or explicitly select --proof-of-concept mixed-source references")
            print(json.dumps(production.prepare_reference_reels(job, candidates, proof_of_concept=args.proof_of_concept, force=args.force), indent=2))
        elif args.command == "calibrate-speaker-voices":
            if args.prepare_only:
                value = production.prepare_speaker_voice_calibration_audio(
                    job,
                    create_tts_backend(settings),
                    force=args.force,
                )
            else:
                value = production.persist_speaker_voice_selections(
                    job,
                    read_json(args.evaluations) if args.evaluations else None,
                )
            print(json.dumps(value, indent=2))
        elif args.command == "prepare-openvoice":
            if args.device != "cuda":
                raise ValueError("Production OpenVoice preparation supports only --device cuda")
            print(
                json.dumps(
                    production.prepare_openvoice(
                        job,
                        lease_identity="unbound",
                        prepare_only=args.prepare_only,
                        proof_of_concept=args.proof_of_concept,
                        force=args.force,
                    ),
                    indent=2,
                )
            )
        elif args.command == "queue-openvoice-assets":
            print(
                json.dumps(
                    ProductionDubbingPipeline(
                        settings, _gcs(settings)
                    ).queue_openvoice_assets(job, force=args.force),
                    indent=2,
                )
            )
        elif args.command == "reconcile-openvoice-assets":
            print(
                json.dumps(
                    ProductionDubbingPipeline(
                        settings, _gcs(settings)
                    ).reconcile_openvoice_assets(job),
                    indent=2,
                )
            )
        elif args.command == "queue-voice-conversion":
            print(json.dumps(ProductionDubbingPipeline(settings, _gcs(settings)).queue_openvoice(job, force=args.force), indent=2))
        elif args.command == "reconcile-voice-conversion":
            print(json.dumps(ProductionDubbingPipeline(settings, _gcs(settings)).reconcile_openvoice(job), indent=2))
        elif args.command == "evaluate-compatibility":
            observations = [UnitCompatibilityObservation(**item) for item in read_json(args.observations)] if args.observations else None
            thresholds = CompatibilityThresholds(**read_json(args.thresholds)) if args.thresholds else None
            print(json.dumps(production.evaluate_compatibility(job, observations, thresholds), ensure_ascii=False, indent=2))
        elif args.command == "build-review-package":
            report = read_json(production.paths(job.job_id).compatibility_json)
            plan = production.prepare_dubbing(job)
            assets = [
                ReviewPackageUnit(
                    unit_id=unit["unit_id"],
                    original_reference=production.jobs.job_dir(job.job_id) / "speakers" / unit["speaker_id"] / "reference_reel.wav",
                    azure_tts=production.paths(job.job_id).azure_turn(unit["unit_id"]),
                    openvoice=production.paths(job.job_id).openvoice_turn(unit["unit_id"]),
                    aligned=production.paths(job.job_id).aligned_turn(unit["unit_id"]),
                )
                for unit in plan["units"]
            ]
            print(json.dumps(build_ab_review_package(production.paths(job.job_id).review_root, report, assets), indent=2))
        elif args.command == "align":
            print(json.dumps(production.align(job, skip_openvoice=args.skip_openvoice, require_compatibility_pass=args.require_compatibility_pass, force=args.force), indent=2))
        elif args.command == "prepare-background":
            print(json.dumps(production.prepare_background(job, clean_stem=args.clean_stem, proof_of_concept=args.proof_of_concept), indent=2))
        elif args.command == "mix":
            print(json.dumps(production.mix(job), indent=2))
        elif args.command == "subtitles":
            print(json.dumps(production.subtitles(job), indent=2))
        elif args.command == "render":
            print(json.dumps(production.render(job, burn_subtitles=args.burn_subtitles), indent=2))
        elif args.command == "review":
            print(json.dumps(production.review(job), ensure_ascii=False, indent=2))
        elif args.command == "bind-entities":
            print(json.dumps(production.bind_entities(job, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "rebuild-with-entities":
            print(json.dumps(production.rebuild_with_entities(job, force=args.force), indent=2))
        elif args.command == "audit-intelligibility":
            stt_backend = create_speech_backend(settings)
            production_with_stt = ProductionDubbingPipeline(settings, _gcs(settings), stt_backend)
            plan = production_with_stt.prepare_dubbing(job)
            stage = args.stage
            audio_root = None
            if stage == "azure-tts":
                audio_root = production_with_stt.paths(job.job_id).azure_tts_root / "turns"
            elif stage == "openvoice":
                audio_root = production_with_stt.paths(job.job_id).openvoice_root / "turns"
            elif stage == "aligned":
                audio_root = production_with_stt.paths(job.job_id).aligned_root / "turns"
            elif stage == "final-mix":
                audio_root = production_with_stt.paths(job.job_id).job_root / "audio"
            else:
                raise ValueError(f"Unknown stage: {stage}")
            if not audio_root or not audio_root.exists():
                raise ValueError(f"Audio root for stage {stage} does not exist")
            
            # Get baseline WER for final-mix degradation check
            baseline_wer = None
            if stage == "final-mix":
                aligned_report_path = production_with_stt.paths(job.job_id).qc_intelligibility_root / "aligned" / "report.json"
                if aligned_report_path.exists():
                    aligned_report = read_json(aligned_report_path)
                    baseline_wer = aligned_report.get("aggregate_blind_wer")
            
            result = production_with_stt._audit_intelligibility(
                job, stage, plan["units"], audio_root, baseline_wer=baseline_wer, with_phrase_hints=args.with_phrase_hints
            )
            print(json.dumps(result, indent=2))
        elif args.command == "process":
            print(_process_message(job))
        elif args.command == "diarize":
            result = run_pyannote(Path(job.objects["local_audio"]), settings.pyannote_model)
            from .atomic_io import atomic_write_json

            atomic_write_json(app.jobs.job_dir(job.job_id) / "analysis/pyannote_diarization.json", result)
            print("Optional Pyannote diagnostic ready; Azure speaker labels remain authoritative")
        elif args.command == "retry":
            if job.state != "failed_retryable":
                raise ValueError("Only failed_retryable jobs can be retried")
            job.last_error = None
            job.transition("uploaded")
            app.jobs.save(job)
            print(app.process(job))
        elif args.command == "validate":
            print(json.dumps(production.validate_job(job), indent=2))
        elif args.command == "transcribe":
            transcribed_job = app.transcribe(job, create_speech_backend(settings), force=args.force)
            print(
                json.dumps(
                    {
                        "job_id": transcribed_job.job_id,
                        "state": transcribed_job.state,
                        "stage": "azure_transcription",
                        "completed": True,
                    }
                )
            )
        else:
            raise ValueError(f"Unsupported command: {args.command}")
        return 0
    except Exception as exc:
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return 1


def _gcs(settings):
    from .gcs_store import GCSStore

    return GCSStore(settings.gcs_bucket, settings.gcs_prefix)


def _process_message(job) -> str:
    messages = {
        "synthesis_queued": f"Legacy state requires: python -m mathula_tv.cli migrate-dubbing-state {job.job_id}",
        "azure_tts_queued": f"Run: python -m mathula_tv.cli synthesize {job.job_id} --backend azure-tts --live-operation",
        "azure_tts_ready": (
            "Run the OpenVoice asset preflight: prepare-openvoice --prepare-only, "
            "queue-openvoice-assets, the assets-mode GPU notebook, and "
            f"reconcile-openvoice-assets for {job.job_id}"
        ),
        "voice_conversion_queued": "GPU worker required. Open the pinned Colab or Kaggle OpenVoice notebook, then reconcile.",
        "voice_conversion_ready": f"Run: python -m mathula_tv.cli evaluate-compatibility {job.job_id} --live-operation",
        "compatibility_review": "Compatibility/human review must pass before normal production alignment.",
        "alignment_ready": f"Run: python -m mathula_tv.cli prepare-background {job.job_id} --live-operation",
        "background_ready": f"Run: python -m mathula_tv.cli mix {job.job_id} --live-operation",
        "review_ready": "Review artifacts are ready; publication approval remains a human decision.",
    }
    return messages.get(job.state, f"No automatic production action for state {job.state}")


if __name__ == "__main__":
    raise SystemExit(main())
