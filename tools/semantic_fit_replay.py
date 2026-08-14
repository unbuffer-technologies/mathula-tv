from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import wave
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathula_tv.ai_provider import AzureOpenAIGPTProvider  # noqa: E402
from mathula_tv.atomic_io import read_json  # noqa: E402
from mathula_tv.azure_tts import AzureTTSResult  # noqa: E402
from mathula_tv.direct_azure_dub import (  # noqa: E402
    DirectAzureDubRenderer,
    DirectDubArtifacts,
    DirectDubError,
    DirectDubOptions,
    SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
    TIMING_MODE_SEMANTIC_FIT,
    _DirectDubProgress,
    _source_block_id,
    attach_source_pause_anchors,
    attach_transcript_lipsync_plan,
    attach_word_anchored_island_contracts,
    build_semantic_phrase_groups,
    coalesce_same_speaker_segments,
    extend_final_speech_block_into_source_tail,
    load_approved_segments,
)
from mathula_tv.pronunciation import (  # noqa: E402
    PronunciationDictionary,
    with_default_organisation_initialisms,
)

JOB_ID = "d724a4c947404afeae1d748363b41827"
SOURCE_DURATION_MS = 227_072
SPEAKER_RATE = 0
VOICE = "zu-ZA-ThembaNeural"


def norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


class TimingOracleBackend:
    sample_rate = 16_000
    ssml_bounds = SimpleNamespace(
        rate_min_percent=-50,
        rate_max_percent=50,
        total_pause_max_ms=10_000,
    )

    def __init__(self, checkpoint: Mapping[str, Any]) -> None:
        self.recorded: dict[tuple[str, int], int] = {}
        self.overrides: dict[tuple[str, int], int] = {}
        self.token_overrides: dict[tuple[str, int], int] = {}
        self.calls: list[dict[str, Any]] = []
        self._learn(checkpoint)

    def _learn(self, checkpoint: Mapping[str, Any]) -> None:
        samples: list[tuple[int, int]] = []
        for case in (checkpoint.get("cases") or {}).values():
            if not isinstance(case, Mapping):
                continue
            for attempt in case.get("attempts") or []:
                if not isinstance(attempt, Mapping):
                    continue
                text = str(attempt.get("tts_text") or attempt.get("spoken_text") or "").strip()
                measured = int(attempt.get("measured_duration_ms") or 0)
                rate = int(attempt.get("speaker_rate_percent") or 0)
                if text and measured > 0:
                    self.recorded[(norm(text), rate)] = measured
                    if rate == SPEAKER_RATE:
                        samples.append((max(1, len(text)), measured))
                for measurement in attempt.get("performance_island_measurements") or []:
                    if not isinstance(measurement, Mapping):
                        continue
                    island_text = str(measurement.get("spoken_text") or measurement.get("tts_text") or "").strip()
                    island_ms = int(measurement.get("measured_duration_ms") or 0)
                    if island_text and island_ms > 0:
                        self.recorded[(norm(island_text), rate)] = island_ms
        # Robust fallback calibrated from this exact speaker/job.
        ratios = sorted(measured / chars for chars, measured in samples if chars >= 8)
        self.ms_per_char = ratios[len(ratios) // 2] if ratios else 95.0

    def register(self, text: str, rate: int, duration_ms: int, *, token: str | None = None) -> None:
        self.overrides[(norm(text), int(rate))] = max(80, int(duration_ms))
        if token:
            self.token_overrides[(token.casefold(), int(rate))] = max(80, int(duration_ms))

    def duration_for(self, text: str, rate: int) -> int:
        key = (norm(text), int(rate))
        if key in self.overrides:
            return self.overrides[key]
        if key in self.recorded:
            return self.recorded[key]
        lowered = norm(text)
        for (token, token_rate), duration in reversed(list(self.token_overrides.items())):
            if token_rate == int(rate) and token in lowered:
                return duration
        # Preserve punctuation pauses modestly; this is only an oracle fallback
        # for unseen text, never an acoustic-quality claim.
        punctuation = sum(str(text).count(ch) for ch in ",.;:?!")
        estimate = int(round(max(120, len(str(text)) * self.ms_per_char + punctuation * 55)))
        # Approximate rate scaling around the recorded -4% calibration.
        if rate != SPEAKER_RATE:
            estimate = int(round(estimate * (100 - SPEAKER_RATE) / max(50, 100 - rate)))
        return max(120, estimate)

    def synthesize(self, request, output_path: Path, *, force: bool):
        duration_ms = self.duration_for(request.text, request.rate_percent)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = max(1, round(duration_ms * self.sample_rate / 1000))
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(b"\0\0" * frames)
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        self.calls.append(
            {
                "turn_id": request.turn_id,
                "text": request.text,
                "rate": request.rate_percent,
                "duration_ms": duration_ms,
                "preferred_duration_ms": request.preferred_duration_ms,
                "maximum_duration_ms": request.maximum_duration_ms,
            }
        )
        return AzureTTSResult(
            backend="replay-oracle",
            turn_id=request.turn_id,
            speaker_id=request.speaker_id,
            voice=request.voice or VOICE,
            language=request.language,
            input_text_hash=digest,
            ssml_hash=digest,
            output_path=str(output_path),
            sample_rate=self.sample_rate,
            channels=1,
            sample_width=2,
            duration_ms=duration_ms,
            preferred_duration_ms=request.preferred_duration_ms,
            maximum_duration_ms=request.maximum_duration_ms,
            rate_percent=request.rate_percent,
            attempt=1,
            sha256=digest,
            warnings=(),
            request_hash=digest,
            ssml_path=str(output_path.with_suffix(".ssml")),
            manifest_path=str(output_path.with_suffix(".json")),
            ssml_summary="semantic-fit replay timing oracle",
            idempotent_reuse=False,
        )


class _ReplayResponse:
    def __init__(self, data: Mapping[str, Any], *, operation: str) -> None:
        self.data = dict(data)
        self.operation = operation

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": "semantic-fit-production-provider-replay-v1",
            "operation": self.operation,
            "data": self.data,
        }


class ProductionContractReplayProvider(AzureOpenAIGPTProvider):
    """Deterministic replay through the REAL production provider methods.

    The inherited fit/review/phrase-repair methods construct the exact
    StructuredAIRequest used in production.  Only complete_structured is
    replaced, so provider capability omissions, schemas, validators,
    normalizers, prompt wiring and operation names are exercised by replay.
    """

    def __init__(self, backend: TimingOracleBackend, *, reject_unit: str | None = None) -> None:
        # Deliberately do not initialize a network client.  The inherited
        # high-level provider methods do not need one before complete_structured.
        self.backend = backend
        self.global_portfolio_calls = 0
        self.global_fit_calls = 0
        self.phrase_calls = 0
        self.review_calls = 0
        self.reject_unit = reject_unit
        self.config = SimpleNamespace(model="production-contract-replay")
        self.provider = "azure-openai-gpt-replay"
        self.sleep = lambda _seconds: None
        self.event_callback = None

    def with_request_options(self, *, timeout_seconds=None, max_retries=None, event_callback=None):
        self.event_callback = event_callback
        return self

    def _phrase_response(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.phrase_calls += 1
        repairs = []
        for repair in payload.get("repairs") or []:
            groups = []
            round_no = int(repair.get("round") or 1)
            for group in repair.get("breath_groups") or []:
                island_id = str(group.get("island_id") or "")
                target = max(1, int(group.get("target_duration_ms") or 1))
                minimum = max(1, int(group.get("minimum_duration_ms") or target))
                maximum = max(minimum, int(group.get("maximum_duration_ms") or target))
                current = " ".join(str(group.get("current_spoken_text") or "").split())
                source_text = " ".join(str(group.get("source_text") or "").split())
                base = current or source_text or "isiZulu phrase"
                raw_token = hashlib.sha1(
                    f"{repair.get('unit_id')}::{island_id}::{round_no}".encode()
                ).hexdigest()[:6]
                token = "sim" + "".join(chr(ord("a") + int(ch, 16) % 16) for ch in raw_token)
                candidates = []
                roles = ["natural_fit", "alternate_grammar", "aggressive_fit"]
                # Exercise measured feedback on a deterministic subset.
                needs_second_round = int(raw_token[:2], 16) % 5 == 0 and round_no == 1
                for pos, role in enumerate(roles):
                    role_token = f"{token}{chr(ord('a') + pos)}"
                    text = f"{base} {role_token}".strip()
                    if needs_second_round:
                        duration = maximum + 220 + pos * 80
                    elif role == "natural_fit":
                        duration = min(maximum, max(minimum, target + 60))
                    elif role == "alternate_grammar":
                        duration = min(maximum, max(minimum, target - 30))
                    else:
                        duration = min(maximum, max(minimum, target))
                    self.backend.register(text, SPEAKER_RATE, duration, token=role_token)
                    candidates.append(
                        {
                            "candidate_id": role,
                            "spoken_text": text,
                            "tts_text": text,
                        }
                    )
                groups.append({"island_id": island_id, "candidates": candidates})
            repairs.append({"unit_id": str(repair.get("unit_id") or ""), "breath_groups": groups})
        return {
            "schema_version": "gpt-localized-breath-group-repair-batch-v1",
            "repairs": repairs,
        }

    def _review_batch_response(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.review_calls += 1
        reviews = []
        for req in payload.get("reviews") or []:
            unit_id = str(req.get("unit_id") or "")
            rejected = bool(self.reject_unit and unit_id.startswith(self.reject_unit))
            reviews.append(
                {
                    "unit_id": unit_id,
                    "review": {
                        "schema_version": "claude-semantic-fit-review-v1",
                        "accepted": not rejected,
                        "required_fact_ids_preserved": [
                            item.get("fact_id")
                            for item in req.get("required_facts") or []
                            if isinstance(item, Mapping) and item.get("fact_id")
                        ],
                        "protected_entities_preserved": list(req.get("protected_entities") or []),
                        "attribution_preserved": True,
                        "uncertainty_preserved": True,
                        "negation_preserved": True,
                        "no_new_facts": True,
                        "natural_spoken_target_language": True,
                        "semantic_fidelity_score": 40 if rejected else 99,
                        "naturalness_score": 95,
                        "reason_codes": ["simulated_semantic_rejection"] if rejected else [],
                    },
                }
            )
        return {
            "schema_version": "claude-semantic-fit-review-batch-v1",
            "reviews": reviews,
        }

    def complete_structured(self, request):
        operation = str(request.operation)
        if operation == "localized_breath_group_repair_batch":
            data = self._phrase_response(request.payload)
        elif operation == "semantic_fit_review_batch":
            data = self._review_batch_response(request.payload)
        elif operation == "semantic_fit_portfolio_batch":
            self.global_portfolio_calls += 1
            repairs = [x for x in request.payload.get("repairs") or [] if isinstance(x, Mapping)]
            ids = [str(x.get("unit_id") or "") for x in repairs]
            sample = repairs[0] if repairs else {}
            raise AssertionError(f"exact exhausted checkpoint unexpectedly called global portfolio for {ids}; sample_keys={list(sample.keys())}; sample={json.dumps(sample, ensure_ascii=False)[:5000]}")
        elif operation == "semantic_fit_batch":
            self.global_fit_calls += 1
            raise AssertionError("exact exhausted checkpoint must not call global fit")
        else:
            raise AssertionError(f"unexpected production provider operation in replay: {operation}")

        if callable(getattr(request, "normalizer", None)):
            data = request.normalizer(data)
        if callable(getattr(request, "validator", None)):
            request.validator(data)
        return _ReplayResponse(data, operation=operation)

class ProgressRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))
        message = str(event.get("message") or "")
        if (
            "phrase-only" in message
            or "exhausted" in message
            or "selected" in message
            or "source-anchored" in message
        ):
            print(f"[{event.get('stage')}] {message}")


def prepare_blocks(job_root: Path):
    translation = read_json(job_root / "translation" / "transcript_zu.json")
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary.from_dict(
            read_json(job_root / "dubbing" / "pronunciation_dictionary.json")
        )
    )
    segments = load_approved_segments(
        translation,
        source_duration_ms=SOURCE_DURATION_MS,
        pronunciation_dictionary=dictionary,
    )
    blocks = coalesce_same_speaker_segments(segments, max_gap_ms=1200)
    blocks, _ = extend_final_speech_block_into_source_tail(
        blocks, source_duration_ms=SOURCE_DURATION_MS
    )
    source_transcript = read_json(job_root / "analysis" / "transcript_en.json")
    blocks, _ = attach_source_pause_anchors(blocks, source_transcript)
    blocks, _ = attach_transcript_lipsync_plan(blocks, source_transcript)
    blocks, _ = build_semantic_phrase_groups(blocks)
    blocks, _ = attach_word_anchored_island_contracts(blocks)
    return tuple(blocks), dictionary


def force_exhausted_resume(checkpoint: dict[str, Any], block_ids: list[str]) -> None:
    checkpoint["strategy_version"] = SEMANTIC_FIT_BATCH_STRATEGY_VERSION
    cases = checkpoint.setdefault("cases", {})
    for unit_id in block_ids:
        case = cases.setdefault(unit_id, {"attempts": []})
        case["strategy_version"] = SEMANTIC_FIT_BATCH_STRATEGY_VERSION
        case["speaker_id"] = "SPEAKER_00"
        case["voice"] = VOICE
        case["speaker_rate_percent"] = SPEAKER_RATE
        # Keep acoustic audit evidence but remove cached candidate objects so
        # the replay cannot accidentally select an old GPT object instead of
        # proving the rescue path.
        for attempt in case.get("attempts") or []:
            if isinstance(attempt, dict):
                attempt.pop("candidate", None)
                attempt.pop("review", None)
        base = max(
            [int(a.get("attempt") or 0) for a in case.get("attempts") or [] if isinstance(a, Mapping)]
            or [0]
        )
        for strategy_attempt in range(1, 7):
            case.setdefault("attempts", []).append(
                {
                    "attempt": base + strategy_attempt,
                    "strategy_attempt": strategy_attempt,
                    "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                    "status": "simulated_exhausted_budget",
                    "speaker_rate_percent": SPEAKER_RATE,
                }
            )
        case["status"] = "simulated_exhausted_resume"
        case["attempt_limit"] = 6


def run_replay(job_root: Path, checkpoint_path: Path, work_dir: Path, *, reject_unit: str | None = None) -> dict[str, Any]:
    blocks, dictionary = prepare_blocks(job_root)
    if len(blocks) != 20:
        raise AssertionError(f"expected 20 semantic-fit blocks, got {len(blocks)}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    force_exhausted_resume(checkpoint, [_source_block_id(b.block_id) for b in blocks])

    sim_job_root = work_dir / "jobs" / JOB_ID
    artifacts = DirectDubArtifacts(sim_job_root)
    artifacts.root.mkdir(parents=True, exist_ok=True)
    artifacts.semantic_fit.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    backend = TimingOracleBackend(checkpoint)
    provider = ProductionContractReplayProvider(backend, reject_unit=reject_unit)
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=work_dir),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    recorder = ProgressRecorder()
    options = DirectDubOptions(
        timing_mode=TIMING_MODE_SEMANTIC_FIT,
        semantic_fit_max_attempts=12,
        semantic_fit_tolerance_ms=250,
        source_word_alignment_tolerance_ms=200,
        same_speaker_gap_ms=1200,
        max_time_stretch_ratio=1.03,
        final_product_max_time_stretch_ratio=1.03,
    )

    error = None
    results = None
    try:
        results = renderer._synthesize_semantic_fit_batch(
            blocks,
            artifacts,
            synthesis_profiles={i: (VOICE, SPEAKER_RATE, 0, 0) for i in range(len(blocks))},
            options=options,
            force=False,
            progress=_DirectDubProgress(recorder),
            pronunciation_dictionary=dictionary,
        )
    except Exception as exc:  # simulator reports rather than hides failure
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"

    final_checkpoint = read_json(artifacts.semantic_fit)
    summary = {
        "job_id": JOB_ID,
        "replay_mode": "latest-checkpoint-forced-exhausted-production-provider-contract",
        "block_count": len(blocks),
        "result_count": len(results or {}),
        "completed": results is not None and len(results) == len(blocks),
        "error": error,
        "global_portfolio_calls": provider.global_portfolio_calls,
        "global_fit_calls": provider.global_fit_calls,
        "phrase_repair_calls": provider.phrase_calls,
        "semantic_review_calls": provider.review_calls,
        "tts_oracle_calls": len(backend.calls),
        "phrase_repair_events": [
            e.get("message") for e in recorder.events if "phrase-only" in str(e.get("message") or "")
        ],
        "case_statuses": {
            _source_block_id(block.block_id): (final_checkpoint.get("cases") or {}).get(
                _source_block_id(block.block_id), {}
            ).get("status")
            for block in blocks
        },
        "single_island_word_timed_blocks": [
            _source_block_id(block.block_id)
            for block in blocks
            if block.source_timing_evidence == "azure_word_timestamps"
            and len(block.source_speech_islands) == 1
            and not block.performance_region_id
        ],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reject-unit", default=None)
    args = parser.parse_args()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    summary = run_replay(
        args.job_root, args.checkpoint, args.work_dir, reject_unit=args.reject_unit
    )
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
