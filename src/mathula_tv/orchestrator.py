from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .atomic_io import atomic_write_json, read_json
from .azure_stt import AzureSpeechError, AzureSpeechTranscriber, SpeechBackend, normalize
from .config import Settings
from .domain_classifier import classify
from .diarization import select_authoritative
from .gcs_store import GCSStore
from .job_store import JobStore
from .media import checksum, prepare_source_derivatives, probe, render_video
from .models import JobManifest
from .context_providers import providers_for
from .transcript_merge import reconcile
from .translation import AzureAIError, JsonClient, TRANSLATION_PROMPT_VERSION, translate as translate_segments, validate_translation
from .seo import generate_seo, validate_seo
from .stt_phrases import configure_source_stt_backend


class Orchestrator:
    def __init__(self, settings: Settings, gcs: GCSStore | None = None):
        self.settings, self.jobs, self.gcs = settings, JobStore(settings.work_dir), gcs

    def submit(self, source: Path, target_language: str = "zu-ZA", force: bool = False) -> JobManifest:
        if target_language != "zu-ZA": raise ValueError("Only zu-ZA is enabled in this proof of concept")
        info = probe(source); digest = checksum(source)
        if float(info["duration"]) > self.settings.max_source_duration_minutes * 60:
            raise ValueError(f"Source exceeds the reviewed {self.settings.max_source_duration_minutes}-minute limit")
        job = self.jobs.create(source_filename=source.name, local_source_path=str(source.resolve()), source_checksum=digest, source_duration=info["duration"], target_language=target_language, configuration=self.settings.safe_snapshot(), human_review_flags=["Human review required before publication"])
        job_dir = self.jobs.job_dir(job.job_id); preserved = job_dir / "input" / source.name; preserved.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, preserved)
        job.local_source_path = str(preserved)
        analysis_audio = job_dir / "audio/analysis_mono.wav"
        mix_audio = job_dir / "audio/mix_source_stereo.wav"
        job.media = prepare_source_derivatives(
            preserved,
            analysis_audio,
            mix_audio,
            max_duration_seconds=self.settings.max_source_duration_minutes * 60,
        )
        job.objects.update(
            local_audio=str(analysis_audio),
            analysis_mono=str(analysis_audio),
            mix_source_stereo=str(mix_audio),
            original_media=str(preserved),
        )
        self.jobs.complete_stage(job, "audio_preparation", "audio_prepared", force=force)
        return job

    def upload(self, job: JobManifest, force: bool = False) -> JobManifest:
        if "upload" in job.completed_stages and not force: return job
        if not self.gcs: self.gcs = GCSStore(self.settings.gcs_bucket, self.settings.gcs_prefix)
        analysis = Path(job.objects["analysis_mono"])
        mix_source = Path(job.objects["mix_source_stereo"])
        job.objects["analysis_audio"] = self.gcs.upload(job.job_id, "audio/analysis_mono.wav", analysis)
        job.objects["mix_source_audio"] = self.gcs.upload(job.job_id, "audio/mix_source_stereo.wav", mix_source)
        # Legacy diagnostic workers may still read this object during migration.
        job.objects["source_audio"] = self.gcs.upload(job.job_id, "input/source_audio.wav", analysis)
        job.objects["manifest"] = self.gcs.upload_json(job.job_id, "input/manifest.json", job.to_dict())
        self.jobs.complete_stage(job, "upload", "uploaded", force=force); self.gcs.upload_json(job.job_id, "status.json", job.to_dict())
        return job

    def classify(self, job: JobManifest) -> dict:
        transcript = read_json(self.jobs.job_dir(job.job_id) / "analysis/transcript_en.json")
        result = classify(" ".join(s["source_text"] for s in transcript["segments"]))
        atomic_write_json(self.jobs.job_dir(job.job_id) / "analysis/domain_classification.json", result)
        return result

    def transcribe(self, job: JobManifest, backend: SpeechBackend, *, force: bool = False) -> JobManifest:
        stage = "azure_transcription"
        job_dir = self.jobs.job_dir(job.job_id)
        phrase_bundle = configure_source_stt_backend(backend)
        phrase_path = job_dir / "analysis/azure_stt_phrase_list.json"
        if phrase_bundle is not None:
            atomic_write_json(phrase_path, phrase_bundle.to_dict())
        raw_path = job_dir / "analysis/azure_stt.json"
        normalized_path = job_dir / "analysis/azure_diarization.json"
        if stage in job.completed_stages and not force:
            # Completed means both local artifacts and their durable GCS objects were verified.
            normalize(read_json(raw_path))
            if not normalized_path.is_file():
                raise ValueError("Completed Azure transcription is missing normalized diarization")
            return job
        if job.state not in {"uploaded", "analysis_running"}:
            raise ValueError(f"Azure transcription requires uploaded or analysis_running state, not {job.state}")
        if job.state == "uploaded":
            job.transition("analysis_running")
            self.jobs.save(job)
        job.attempt_counters[stage] = job.attempt_counters.get(stage, 0) + 1
        self.jobs.save(job)
        try:
            if raw_path.is_file() and not force:
                raw = read_json(raw_path)
                normalized = normalize(raw, provider=backend.provider)
            else:
                raw, normalized = AzureSpeechTranscriber(backend).transcribe(Path(job.objects["local_audio"]), self.settings.azure_speech_locale)
                atomic_write_json(raw_path, raw)
            atomic_write_json(normalized_path, normalized)
            if not self.gcs:
                self.gcs = GCSStore(self.settings.gcs_bucket, self.settings.gcs_prefix)
            # File upload preserves the successful Azure response byte-for-byte.
            job.objects["azure_stt"] = self.gcs.upload(job.job_id, "analysis/azure_stt.json", raw_path)
            job.objects["azure_diarization"] = self.gcs.upload(job.job_id, "analysis/azure_diarization.json", normalized_path)
            if phrase_bundle is not None:
                job.objects["azure_stt_phrase_list"] = self.gcs.upload(
                    job.job_id,
                    "analysis/azure_stt_phrase_list.json",
                    phrase_path,
                )
                job.providers["azure_stt_phrase_registry_sha256"] = phrase_bundle.registry_sha256
                job.providers["azure_stt_phrase_count"] = len(phrase_bundle.phrases)
                job.providers["azure_stt_phrase_biasing_weight"] = phrase_bundle.biasing_weight
            job.providers["azure_stt"] = backend.provider
            job.providers["azure_stt_api_version"] = backend.api_version
            job.media["azure_transcription_duration"] = normalized["duration"]
            if backend.request_duration_seconds is not None:
                job.media["azure_request_duration"] = round(backend.request_duration_seconds, 3)
            if stage not in job.completed_stages:
                job.completed_stages.append(stage)
            job.last_error = None
            self.jobs.save(job)
            self.gcs.upload_json(job.job_id, "status.json", job.to_dict())
            return job
        except Exception as exc:
            retryable = not isinstance(exc, AzureSpeechError) or exc.retryable
            job.last_error = {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)[:500], "retryable": retryable, "at": datetime.now(timezone.utc).isoformat()}
            job.transition("failed_retryable" if retryable else "failed_terminal")
            self.jobs.save(job)
            raise

    def render(self, job: JobManifest) -> Path:
        if job.state != "synthesis_ready": raise ValueError("Job must be synthesis_ready before rendering")
        job.transition("rendering"); self.jobs.save(job)
        audio = self.jobs.job_dir(job.job_id) / "output/dubbed_zu.wav"; output = self.jobs.job_dir(job.job_id) / "output/review_zu.mp4"
        render_video(Path(job.local_source_path), audio, output)
        job.objects["review_video"] = str(output); self.jobs.complete_stage(job, "render", "review_ready")
        return output

    def translate_and_package(self, job: JobManifest, client: JsonClient, *, force: bool = False) -> dict:
        if job.state == "synthesis_queued" and not force:
            translated = read_json(self.jobs.job_dir(job.job_id) / "translation/transcript_zu.json")
            seo = read_json(self.jobs.job_dir(job.job_id) / "translation/youtube_zu.json")
            validate_translation(translated, read_json(self.jobs.job_dir(job.job_id) / "analysis/transcript_en.json")["segments"]); validate_seo(seo)
            return self._translation_summary(job, translated, seo)
        if job.state not in {"analysis_ready", "translating", "translation_ready", "failed_retryable"}:
            raise ValueError(f"Translation requires analysis_ready or a retryable translation state, not {job.state}")
        if job.state == "failed_retryable" and (job.last_error or {}).get("stage") not in {"translation", "seo_packaging"}:
            raise ValueError("Retryable failure does not belong to the translation stage")
        job_dir = self.jobs.job_dir(job.job_id)
        transcript_path = job_dir / "analysis/transcript_en.json"
        classification_path = job_dir / "analysis/domain_classification.json"
        context_path = job_dir / "analysis/context.json"
        inputs = {}
        for name, path in (("transcript_en",transcript_path),("domain_classification",classification_path),("context",context_path)):
            if not path.is_file() or not path.stat().st_size: raise ValueError(f"Required analysis artifact is missing or empty: {path.name}")
            inputs[name] = checksum(path)
        previous = job.media.get("analysis_input_hashes")
        if previous and previous != inputs: raise ValueError("Analysis artifact checksum changed after analysis_ready")
        job.media["analysis_input_hashes"] = inputs
        transcript, classification, context_document = read_json(transcript_path), read_json(classification_path), read_json(context_path)
        selected_context = [p for p in context_document.get("providers",[]) if p.get("enrichment_applied")]
        translated_path = job_dir / "translation/transcript_zu.json"
        seo_path = job_dir / "translation/youtube_zu.json"
        translation_duration = seo_duration = 0.0
        try:
            if job.state in {"analysis_ready","failed_retryable"}: job.transition("translating"); self.jobs.save(job)
            translated = None
            if translated_path.is_file() and not force:
                translated = validate_translation(read_json(translated_path), transcript["segments"])
                if "request_duration_seconds" not in translated: translated=None
            if translated is None:
                job.attempt_counters["translation"] = job.attempt_counters.get("translation",0)+1; self.jobs.save(job)
                before=len(client.request_durations); translated=translate_segments(client,transcript,selected_context); translation_duration=sum(client.request_durations[before:])
                atomic_write_json(translated_path,translated)
                job.media["translation_request_duration"]=round(translation_duration,3)
                job.media["translation_prompt_version"]=TRANSLATION_PROMPT_VERSION
                job.media["translation_repair_attempts"]=translated.get("repair_attempts",0)
                self.jobs.save(job)
            seo = None
            if seo_path.is_file() and not force:
                seo=validate_seo(read_json(seo_path))
            if seo is None:
                job.attempt_counters["seo_packaging"] = job.attempt_counters.get("seo_packaging",0)+1; self.jobs.save(job)
                before=len(client.request_durations); seo=generate_seo(client,transcript,translated,classification,selected_context); seo_duration=sum(client.request_durations[before:])
                atomic_write_json(seo_path,seo)
            if not self.gcs: self.gcs=GCSStore(self.settings.gcs_bucket,self.settings.gcs_prefix)
            for relative,path,key in (("translation/transcript_zu.json",translated_path,"transcript_zu"),("translation/youtube_zu.json",seo_path,"youtube_zu")):
                digest=checksum(path); temporary=f"{relative}.partial.server-{uuid.uuid4().hex[:12]}"
                self.gcs.upload(job.job_id,temporary,path,if_generation_match=0)
                job.objects[key]=self.gcs.promote(job.job_id,temporary,relative,digest,checksum)
                job.media[f"{key}_sha256"]=digest
            job.providers["translation"]="azure-openai"
            job.providers["translation_deployment"]=client.deployment
            job.providers["translation_api_version"]=client.api_version
            job.media["translation_request_duration"]=round(translation_duration,3) if translation_duration else job.media.get("translation_request_duration",translated.get("request_duration_seconds",0))
            job.media["seo_request_duration"]=round(seo_duration,3)
            job.media["translation_prompt_version"]=TRANSLATION_PROMPT_VERSION
            job.media["translation_repair_attempts"]=translated.get("repair_attempts",0)
            if "translation" not in job.completed_stages: job.completed_stages.append("translation")
            if job.state == "translating": job.transition("translation_ready")
            if "seo_packaging" not in job.completed_stages: job.completed_stages.append("seo_packaging")
            job.transition("synthesis_queued")
            job.last_error=None; self.jobs.save(job); self.gcs.upload_json(job.job_id,"status.json",job.to_dict())
            return self._translation_summary(job,translated,seo)
        except Exception as exc:
            retryable=isinstance(exc,AzureAIError) and exc.retryable
            stage="seo_packaging" if translated_path.is_file() else "translation"
            job.last_error={"stage":stage,"error_type":type(exc).__name__,"message":str(exc)[:500],"retryable":retryable,"at":datetime.now(timezone.utc).isoformat()}
            if job.state not in {"failed_retryable","failed_terminal"}: job.transition("failed_retryable" if retryable else "failed_terminal")
            self.jobs.save(job); raise

    @staticmethod
    def _translation_summary(job: JobManifest, translated: dict, seo: dict) -> dict:
        return {"job_id":job.job_id,"state":job.state,"translated_segments":len(translated["turns"]),"target_language":translated["language"],"title":seo["title"],"translation_request_duration":job.media.get("translation_request_duration",0),"seo_request_duration":job.media.get("seo_request_duration",0)}

    def process(self, job: JobManifest, speech_backend: SpeechBackend | None = None, translation_client: JsonClient | None = None):
        if job.state == "audio_prepared": self.upload(job)
        if job.state == "uploaded" and speech_backend:
            self.transcribe(job, speech_backend)
        if job.state == "analysis_running":
            pyannote = None
            pyannote_path = self.jobs.job_dir(job.job_id) / "analysis/pyannote_diarization.json"
            diagnostic_warning = None
            if getattr(self.settings, "pyannote_enabled", False):
                if not self.gcs: self.gcs = GCSStore(self.settings.gcs_bucket, self.settings.gcs_prefix)
                try:
                    self.gcs.download(job.job_id, "analysis/pyannote_diarization.json", pyannote_path)
                    pyannote = read_json(pyannote_path)
                except Exception as exc:
                    if isinstance(exc, FileNotFoundError) or "404" in str(exc) or "not found" in str(exc).lower():
                        diagnostic_warning = "not supplied"
                    else:
                        raise
            azure = read_json(self.jobs.job_dir(job.job_id) / "analysis/azure_diarization.json")
            turns, source, warnings = select_authoritative(pyannote, azure, diagnostic_warning)
            job_dir = self.jobs.job_dir(job.job_id)
            transcript_path = job_dir / "analysis/transcript_en.json"
            
            # Check if a corrected transcript already exists
            if transcript_path.is_file() and transcript_path.stat().st_size > 0:
                # Use existing corrected transcript
                transcript = read_json(transcript_path)
                transcript["warnings"] = transcript.get("warnings", [])
                transcript["warnings"].extend(warnings)
                transcript["warnings"].append("Using existing corrected transcript")
                atomic_write_json(transcript_path, transcript)
            else:
                # Reconcile from Azure STT words
                transcript = reconcile(azure["words"], turns, source=source, tolerance=self.settings.speaker_tolerance_seconds)
                transcript["warnings"].extend(warnings)
                atomic_write_json(transcript_path, transcript)
            
            classification = classify(" ".join(segment["source_text"] for segment in transcript["segments"]))
            classification_path = job_dir / "analysis/domain_classification.json"
            atomic_write_json(classification_path, classification)
            text = " ".join(segment["source_text"] for segment in transcript["segments"])
            context = {"providers": [provider.enrich(classification, text) for provider in providers_for(classification, self.settings.commission_master_case_path, self.settings.politics_context_path)]}
            context_path = job_dir / "analysis/context.json"
            atomic_write_json(context_path, context)
            
            # Update analysis input hashes based on actual file bytes
            inputs = {}
            for name, path in (("transcript_en", transcript_path), ("domain_classification", classification_path), ("context", context_path)):
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"Required analysis artifact is missing or empty: {path.name}")
                inputs[name] = checksum(path)
            job.media["analysis_input_hashes"] = inputs
            
            if not self.gcs:
                self.gcs = GCSStore(self.settings.gcs_bucket, self.settings.gcs_prefix)
            if pyannote_path.is_file():
                job.objects["pyannote_diarization"] = self.gcs.upload(job.job_id, "analysis/pyannote_diarization.json", pyannote_path)
            job.objects["transcript_en"] = self.gcs.upload(job.job_id, "analysis/transcript_en.json", transcript_path)
            job.objects["domain_classification"] = self.gcs.upload(job.job_id, "analysis/domain_classification.json", classification_path)
            job.objects["context"] = self.gcs.upload(job.job_id, "analysis/context.json", context_path)
            job.providers["authoritative_diarization"] = source
            if pyannote is not None and "pyannote_analysis" not in job.completed_stages: job.completed_stages.append("pyannote_analysis")
            if "analysis_reconciliation" not in job.completed_stages: job.completed_stages.append("analysis_reconciliation")
            job.transition("analysis_ready")
            self.jobs.save(job)
            self.gcs.upload_json(job.job_id, "status.json", job.to_dict())
            return "Analysis ready: Azure diarization retained, transcript classified, and context selected"
        if job.state == "uploaded": return "Waiting for Azure transcription"
        if job.state == "analysis_ready":
            if translation_client is None: raise ValueError("Azure OpenAI translation backend is required")
            return self.translate_and_package(job,translation_client)
        if job.state == "synthesis_queued":
            return "Legacy synthesis_queued must be migrated to azure_tts_queued; no legacy voice backend is registered"
        if job.state in {"synthesis_claimed", "synthesizing", "synthesis_ready"}:
            raise ValueError("Deprecated legacy synthesis state is not dispatched; reconcile or migrate it explicitly")
        return f"No automatic action for state {job.state}"
