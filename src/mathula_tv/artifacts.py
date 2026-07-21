"""Repository-consistent deterministic dubbing artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DubbingArtifacts:
    job_root: Path

    @property
    def analysis_audio(self) -> Path:
        return self.job_root / "audio/analysis_mono.wav"

    @property
    def mix_source_audio(self) -> Path:
        return self.job_root / "audio/mix_source_stereo.wav"

    @property
    def dubbing_units(self) -> Path:
        return self.job_root / "dubbing/dubbing_units.json"

    @property
    def three_text_translation(self) -> Path:
        return self.job_root / "dubbing/translation_zu.json"

    @property
    def pronunciation_dictionary(self) -> Path:
        return self.job_root / "dubbing/pronunciation_dictionary.json"

    @property
    def plan(self) -> Path:
        return self.job_root / "dubbing/dubbing_plan.json"

    @property
    def azure_tts_root(self) -> Path:
        return self.job_root / "dubbing/azure_tts"

    def azure_candidate(self, unit_id: str, rate: int) -> Path:
        return self.azure_tts_root / "candidates" / unit_id / f"rate_{rate:+d}.wav"

    def azure_turn(self, unit_id: str) -> Path:
        return self.azure_tts_root / "turns" / f"{unit_id}.wav"

    @property
    def azure_manifest(self) -> Path:
        return self.azure_tts_root / "manifest.json"

    @property
    def openvoice_root(self) -> Path:
        return self.job_root / "dubbing/openvoice"

    def openvoice_turn(self, unit_id: str) -> Path:
        return self.openvoice_root / "turns" / f"{unit_id}.wav"

    @property
    def openvoice_plan(self) -> Path:
        return self.openvoice_root / "plan.json"

    @property
    def openvoice_asset_plan(self) -> Path:
        return self.openvoice_root / "asset_plan.json"

    @property
    def openvoice_asset_manifest(self) -> Path:
        return self.openvoice_root / "asset_manifest.json"

    @property
    def openvoice_manifest(self) -> Path:
        return self.openvoice_root / "manifest.json"

    @property
    def aligned_root(self) -> Path:
        return self.job_root / "dubbing/aligned"

    def aligned_turn(self, unit_id: str) -> Path:
        return self.aligned_root / "turns" / f"{unit_id}.wav"

    @property
    def alignment_manifest(self) -> Path:
        return self.aligned_root / "manifest.json"

    @property
    def background(self) -> Path:
        return self.job_root / "audio/background.wav"

    @property
    def background_manifest(self) -> Path:
        return self.job_root / "audio/background_manifest.json"

    @property
    def dialogue(self) -> Path:
        return self.job_root / "audio/dub_dialogue.wav"

    @property
    def final_mix(self) -> Path:
        return self.job_root / "audio/final_mix.wav"

    @property
    def subtitle_root(self) -> Path:
        return self.job_root / "subtitles"

    @property
    def render_root(self) -> Path:
        return self.job_root / "render"

    @property
    def final_video(self) -> Path:
        return self.render_root / "final_dubbed.mp4"

    @property
    def render_manifest(self) -> Path:
        return self.render_root / "render_manifest.json"

    @property
    def review_root(self) -> Path:
        return self.job_root / "review"

    @property
    def compatibility_json(self) -> Path:
        return self.review_root / "compatibility_report.json"

    @property
    def compatibility_markdown(self) -> Path:
        return self.review_root / "compatibility_report.md"

    @property
    def qc_json(self) -> Path:
        return self.review_root / "qc_report.json"

    @property
    def qc_markdown(self) -> Path:
        return self.review_root / "qc_report.md"

    @property
    def entity_bindings(self) -> Path:
        return self.job_root / "analysis/entity_bindings.json"

    @property
    def qc_intelligibility_root(self) -> Path:
        return self.job_root / "qc/intelligibility"

    def intelligibility_report(self, stage: str) -> Path:
        return self.qc_intelligibility_root / stage / "report.json"

    def intelligibility_markdown(self, stage: str) -> Path:
        return self.qc_intelligibility_root / stage / "report.md"

    @property
    def preview_root(self) -> Path:
        return self.job_root / "audio/previews"

    @property
    def preview_background(self) -> Path:
        return self.preview_root / "background_review.wav"

    @property
    def preview_final_mix(self) -> Path:
        return self.preview_root / "final_mix_review.wav"
