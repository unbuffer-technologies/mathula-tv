"""Azure STT intelligibility audit for dubbed audio quality gates."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .azure_stt import SpeechBackend
from .config import Settings


INTELLIGIBILITY_AUDIT_SCHEMA = "mathula-stt-intelligibility-audit-v1"


@dataclass(frozen=True)
class UnitIntelligibilityObservation:
    """Per-unit intelligibility measurements."""
    unit_id: str
    expected_text: str
    transcribed_text: str
    word_error_rate: float
    protected_entities_expected: list[str]
    protected_entities_recognized: list[str]
    protected_entities_missing: list[str]
    best_recognized_phrases: dict[str, str]
    duration_seconds: float | None = None
    audio_sha256: str | None = None


@dataclass(frozen=True)
class IntelligibilityAuditReport:
    """Complete intelligibility audit report."""
    schema_version: str
    job_id: str
    stage: str
    locale: str
    generated_at: str
    aggregate_blind_wer: float
    aggregate_hinted_wer: float | None
    per_unit_observations: dict[str, dict[str, Any]]
    protected_entities_summary: dict[str, Any]
    pass_fail_reasons: list[str]
    state: str
    audio_paths: dict[str, str]
    raw_azure_response: dict[str, Any] | None = None


class IntelligibilityAuditor:
    """Reusable Azure STT intelligibility auditor."""

    def __init__(self, settings: Settings, backend: SpeechBackend):
        self.settings = settings
        self.backend = backend

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (Unicode, punctuation, case)."""
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # Lowercase
        text = text.lower()
        # Remove punctuation except apostrophes in contractions
        text = re.sub(r"[^\w\s']", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate word error rate between reference and hypothesis."""
        ref_words = self.normalize_text(reference).split()
        hyp_words = self.normalize_text(hypothesis).split()
        
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        
        # Levenshtein distance for word-level WER
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1],  # substitution
                    )
        
        return dp[m][n] / len(ref_words)

    def check_protected_entity(
        self,
        entity: str,
        transcribed: str,
        aliases: list[str] | None = None,
        stt_phrases: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Check if a protected entity is recognized in transcription."""
        norm_entity = self.normalize_text(entity)
        norm_transcribed = self.normalize_text(transcribed)
        
        # Check exact match
        if norm_entity in norm_transcribed:
            return True, entity
        
        # Check aliases
        if aliases:
            for alias in aliases:
                if self.normalize_text(alias) in norm_transcribed:
                    return True, alias
        
        # Check STT phrases
        if stt_phrases:
            for phrase in stt_phrases:
                if self.normalize_text(phrase) in norm_transcribed:
                    return True, phrase
        
        # Fuzzy match for punctuation/apostrophe variations
        fuzzy_entity = re.sub(r"[^\w\s]", "", norm_entity)
        fuzzy_transcribed = re.sub(r"[^\w\s]", "", norm_transcribed)
        if fuzzy_entity in fuzzy_transcribed:
            return True, entity
        
        return False, ""

    def audit_unit(
        self,
        unit_id: str,
        expected_text: str,
        audio_path: Path,
        protected_entities: dict[str, dict[str, Any]] | None = None,
        locale: str = "zu-ZA",
    ) -> UnitIntelligibilityObservation:
        """Audit a single unit's intelligibility."""
        # Transcribe audio
        result = self.backend.transcribe(audio_path, locale)
        transcribed = self._extract_transcription(result)
        
        # Calculate WER
        wer = self.calculate_wer(expected_text, transcribed)
        
        # Check protected entities
        protected_entities = protected_entities or {}
        expected = list(protected_entities.keys())
        recognized = []
        missing = []
        best_phrases = {}
        
        for entity_id, entity_data in protected_entities.items():
            canonical = entity_data.get("canonical_text", "")
            aliases = entity_data.get("aliases", [])
            stt_phrases = entity_data.get("stt_phrases", [])
            
            is_recognized, best_phrase = self.check_protected_entity(
                canonical, transcribed, aliases, stt_phrases
            )
            
            if is_recognized:
                recognized.append(entity_id)
                best_phrases[entity_id] = best_phrase
            else:
                missing.append(entity_id)
                best_phrases[entity_id] = ""
        
        return UnitIntelligibilityObservation(
            unit_id=unit_id,
            expected_text=expected_text,
            transcribed_text=transcribed,
            word_error_rate=wer,
            protected_entities_expected=expected,
            protected_entities_recognized=recognized,
            protected_entities_missing=missing,
            best_recognized_phrases=best_phrases,
            audio_sha256=self._audio_checksum(audio_path) if audio_path.exists() else None,
        )

    def audit_stage(
        self,
        job_id: str,
        stage: str,
        units: list[dict[str, Any]],
        audio_root: Path,
        locale: str = "zu-ZA",
        with_phrase_hints: bool = False,
        protected_entities: dict[str, dict[str, Any]] | None = None,
    ) -> IntelligibilityAuditReport:
        """Audit an entire stage (azure-tts, openvoice, aligned, final-mix)."""
        observations = {}
        aggregate_wer = 0.0
        aggregate_hinted_wer = None
        
        for unit in units:
            unit_id = unit.get("unit_id")
            expected_text = unit.get("tts_text", "")
            audio_path = audio_root / f"{unit_id}.wav"
            
            if not audio_path.exists():
                continue
            
            # Blind audit
            obs = self.audit_unit(
                unit_id, expected_text, audio_path, protected_entities, locale
            )
            observations[unit_id] = asdict(obs)
            aggregate_wer += obs.word_error_rate
            
            # Optional hinted audit (diagnostic only)
            if with_phrase_hints and protected_entities:
                # This would require a separate STT call with phrase hints
                # For now, we skip this as it's diagnostic-only
                pass
        
        if observations:
            aggregate_wer /= len(observations)
        
        # Determine pass/fail
        max_wer = self._get_max_wer_for_stage(stage)
        state = "passed" if aggregate_wer <= max_wer else "failed"
        reasons = []
        
        if state == "failed":
            reasons.append(f"Aggregate WER {aggregate_wer:.4f} exceeds threshold {max_wer}")
        
        # Check protected entities
        total_expected = sum(len(o.get("protected_entities_expected", [])) for o in observations.values())
        total_recognized = sum(len(o.get("protected_entities_recognized", [])) for o in observations.values())
        
        if total_expected > 0:
            recognition_rate = total_recognized / total_expected
            min_similarity = self.settings.__dict__.get("min_protected_entity_similarity", 0.85)
            if recognition_rate < min_similarity:
                state = "failed"
                reasons.append(
                    f"Protected entity recognition {recognition_rate:.2%} below threshold {min_similarity:.2%}"
                )
        
        return IntelligibilityAuditReport(
            schema_version=INTELLIGIBILITY_AUDIT_SCHEMA,
            job_id=job_id,
            stage=stage,
            locale=locale,
            generated_at=datetime.now(timezone.utc).isoformat(),
            aggregate_blind_wer=aggregate_wer,
            aggregate_hinted_wer=aggregate_hinted_wer,
            per_unit_observations=observations,
            protected_entities_summary={
                "total_expected": total_expected,
                "total_recognized": total_recognized,
                "recognition_rate": total_recognized / total_expected if total_expected > 0 else 1.0,
            },
            pass_fail_reasons=reasons,
            state=state,
            audio_paths={"stage_root": str(audio_root)},
        )

    def _extract_transcription(self, result: dict[str, Any]) -> str:
        """Extract transcription text from Azure STT result."""
        if "combinedResults" in result:
            combined = result["combinedResults"]
            if isinstance(combined, list) and combined:
                return combined[0].get("lexical", "")
        if "DisplayText" in result:
            return result["DisplayText"]
        return ""

    def _audio_checksum(self, path: Path) -> str:
        """Calculate SHA-256 of audio file."""
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _get_max_wer_for_stage(self, stage: str) -> float:
        """Get maximum allowed WER for a stage."""
        thresholds = {
            "azure-tts": self.settings.__dict__.get("max_clean_unit_wer", 0.35),
            "openvoice": self.settings.__dict__.get("max_openvoice_wer_degradation", 0.10),
            "aligned": self.settings.__dict__.get("max_aligned_wer_degradation", 0.10),
            "final-mix": self.settings.__dict__.get("max_final_mix_wer_degradation", 0.15),
        }
        return thresholds.get(stage, 0.35)

    def write_report(
        self,
        report: IntelligibilityAuditReport,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Write JSON and Markdown reports."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        json_path = output_dir / "report.json"
        atomic_write_json(json_path, asdict(report))
        
        # Markdown report
        md_path = output_dir / "report.md"
        md_content = self._markdown_report(report)
        md_path.write_text(md_content, encoding="utf-8")
        
        return json_path, md_path

    def _markdown_report(self, report: IntelligibilityAuditReport) -> str:
        """Generate Markdown report."""
        hinted_wer_str = f"{report.aggregate_hinted_wer:.4f}" if report.aggregate_hinted_wer is not None else "N/A"
        lines = [
            "# Intelligibility Audit Report",
            "",
            f"**Job ID:** {report.job_id}",
            f"**Stage:** {report.stage}",
            f"**Locale:** {report.locale}",
            f"**Generated:** {report.generated_at}",
            "",
            "## Summary",
            "",
            f"**State:** {report.state}",
            f"**Aggregate Blind WER:** {report.aggregate_blind_wer:.4f}",
            f"**Aggregate Hinted WER:** {hinted_wer_str}",
            "",
        ]
        
        if report.pass_fail_reasons:
            lines.append("### Pass/Fail Reasons")
            lines.append("")
            for reason in report.pass_fail_reasons:
                lines.append(f"- {reason}")
            lines.append("")
        
        lines.extend([
            "### Protected Entities",
            "",
            f"- Expected: {report.protected_entities_summary['total_expected']}",
            f"- Recognized: {report.protected_entities_summary['total_recognized']}",
            f"- Recognition Rate: {report.protected_entities_summary['recognition_rate']:.2%}",
            "",
            "## Per-Unit Observations",
            "",
        ])
        
        for unit_id, obs in report.per_unit_observations.items():
            lines.extend([
                f"### {unit_id}",
                "",
                f"- WER: {obs['word_error_rate']:.4f}",
                f"- Expected: {obs['protected_entities_expected']}",
                f"- Recognized: {obs['protected_entities_recognized']}",
                f"- Missing: {obs['protected_entities_missing']}",
                "",
            ])
        
        return "\n".join(lines)


__all__ = [
    "INTELLIGIBILITY_AUDIT_SCHEMA",
    "UnitIntelligibilityObservation",
    "IntelligibilityAuditReport",
    "IntelligibilityAuditor",
]
