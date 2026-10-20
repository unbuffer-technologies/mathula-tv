"""The dubbing agent's environment: real Azure TTS measurement and deterministic isiZulu checks.

A ``Workbench`` is plain Python with no transport. The hand-written agent loop
(``dub_agent``) calls its methods as tools directly, and the MCP server
(``dub_workbench_mcp``) exposes the very same methods to any other agent -- so
what an agent measures here is what ``submit_dub`` grades it on (both go through
``dub_fit.measure_text_fit``).

It prescribes no dubbing method. One-shot, candidate search, anything: the agent
chooses, then submits units.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .atomic_io import atomic_write_json, read_json
from .dub_fit import measure_text_fit
from .dub_submission import score_submission_dir, validate_submission, write_submission
from .episode_scoring import (
    GATE_NAMES, _known_entity_tokens, english_leakage, find_fabricated_entities,
    find_restated_references, find_unsupported_doubling,
)
from .dub_pronunciation import register_clip_pronunciation
from .dub_voices import load_clip_voices, pick_voice_info
from .experience_store import ExperienceStore

DEFAULT_TTS_BUDGET = 800
MAX_MEASURE_BATCH = 40     # a whole clip's worth of units; the pool measures them six at a time


_BAD_TIMING_FLAGS = frozenset({"overrun", "rushed", "dead_air"})


def segment_timing_status(card: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per source segment: does the unit(s) covering it fit its time frame? (ok, plus why not.)

    Keyed by segment, not by unit, because an agent may regroup units between submissions while the
    segments they cover stay put. A segment split across several units fits only if all of them do.
    """
    by_unit = {str(g["group_id"]): g for g in card["groups"]}
    status: dict[str, dict[str, Any]] = {}
    for item in items:
        group = by_unit.get(str(item["item_id"]))
        if group is None:
            continue
        why = sorted(_BAD_TIMING_FLAGS & set(group["flags"]))
        if group.get("collision_ms"):
            why.append(f"collides {group['collision_ms']}ms with {group['collides_with']}")
        for segment_id in item["source_segment_ids"]:
            entry = status.setdefault(str(segment_id), {"ok": True, "unit": str(item["item_id"]), "why": []})
            if why:
                entry.update(ok=False, unit=str(item["item_id"]), why=why)
    return status


def merge_partial(
    previous: list[dict[str, Any]], update: list[dict[str, Any]], remove: list[str] | tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fold a partial update into the last accepted submission.

    A current unit is replaced only if an updated unit has its item_id, or its id is in ``remove``; every other
    current unit carries over untouched. (Matching on shared source segments was too aggressive: units may
    legitimately share a boundary segment, and sending one of them silently dropped its neighbours.)
    Returns (merged units in time order, ids replaced or removed).
    """
    new_ids = {str(u["item_id"]) for u in update}
    removed = {str(r) for r in remove}
    kept, replaced = [], []
    for unit in previous:
        if str(unit["item_id"]) in new_ids or str(unit["item_id"]) in removed:
            replaced.append(str(unit["item_id"]))
        else:
            kept.append(unit)
    merged = sorted([*kept, *update], key=lambda u: int(u["start_ms"]))
    return merged, replaced


def timing_key(headline: dict[str, Any]) -> tuple[float, float, float]:
    """Lower is better: cross-speaker collisions, then collisions, then overruns."""
    return (
        float(headline.get("cross_speaker_collisions", 0)), float(headline.get("collisions", 0)),
        float(headline.get("groups_rushed", 0)),
    )


def _timing_summary(headline: dict[str, Any]) -> dict[str, Any]:
    return {k: headline.get(k) for k in ("cross_speaker_collisions", "collisions", "groups_rushed", "residual_dead_air_pct")}


def ratchet_regressions(locked: set[str], status: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Segments that fitted before and no longer do."""
    return [
        {"segment": sid, "unit": status[sid]["unit"], "now": status[sid]["why"]}
        for sid in sorted(locked) if sid in status and not status[sid]["ok"]
    ]


class BudgetExhausted(RuntimeError):
    """Raised when the agent asks for more measurements than its episode budget allows."""


class Workbench:
    def __init__(
        self, store: ExperienceStore, tts_factory: Callable[[], Any], *,
        scope: dict[str, Any] | None = None, subdir: str | None = None, measure_root: Path | None = None,
        timing_ratchet: bool = True, voices: dict[str, dict[str, Any]] | None = None, judge: Any = None,
    ) -> None:
        """``scope`` ({segment_ids, next_start_ms}) narrows this workbench to one speaker turn of the
        clip; ``subdir`` puts its episode files under ``<episode>/turns/<subdir>``; ``measure_root``
        shares one synthesis cache across workbenches (same text -> same audio file)."""
        self.store = store
        self._tts_factory = tts_factory
        self._tts: Any = None
        self.scope = scope
        self.subdir = subdir
        self._measure_root = Path(measure_root) if measure_root else None
        self.last_items: list[dict[str, Any]] | None = None
        self.voices: dict[str, dict[str, Any]] = dict(voices or {})
        self.pronunciation_root: Path | None = None
        self.segments: list[dict[str, Any]] = []
        # Timing ratchet: once a source segment has fitted its time frame in an accepted submission it must
        # keep fitting. A submission that breaks one is rejected, so fixing one unit can never silently
        # break another, and the clip's flow only ever improves.
        self.timing_ratchet = timing_ratchet
        self.judge = judge        # a dub_judge.UnitJudge, or None for no meaning check
        self.locked_segments: set[str] = set()
        self.episode: dict[str, Any] | None = None
        self.last_submit: dict[str, Any] | None = None   # the latest accepted submit_dub result, for the harness

    # -- plumbing -----------------------------------------------------------------
    @property
    def tts(self) -> Any:
        if self._tts is None:
            self._tts = self._tts_factory()
        return self._tts

    def _episode(self) -> dict[str, Any]:
        if not self.episode:
            raise RuntimeError("No clip is open. Call open_clip first.")
        return self.episode

    def _measure_dir(self) -> Path:
        return self._measure_root or Path(self._episode()["dir"]) / "measure"

    def _spend(self, count: int) -> int:
        episode = self._episode()
        budget = int(episode["tts_budget"])
        if episode["tts_used"] + count > budget:
            raise BudgetExhausted(
                f"TTS measurement budget exhausted ({episode['tts_used']}/{budget}). "
                "Submit with what you have."
            )
        episode["tts_used"] += count
        atomic_write_json(Path(episode["dir"]) / "episode.json", dict(episode))
        return budget - episode["tts_used"]

    def _present(self, result: dict[str, Any]) -> dict[str, Any]:
        """Show timing the way this episode's speed policy sees it.

        Under "natural" there is no speed-up to lean on, so the agent is shown an overrun,
        not a "required speed", and none of the speed-rescue bookkeeping."""
        result.pop("audio_path", None)
        if self._episode().get("speed_policy") == "natural":
            result["overrun_pct"] = result.pop("required_speed_percent", 0.0)
            # The pipeline's fit flag forgives a speed-up of up to ~10%. Here nothing is forgiven:
            # it fits only if it needs no speed-up and is not undersized.
            result["fits_naturally"] = result["overrun_pct"] == 0 and result.get("direction") != "expand"
            result.pop("speed_fit_mode", None)
            result.pop("rank", None)
            # Fields the agent never needs: the voice is fixed, the ms duplicates the pct, the tolerance
            # is already folded into overrun_pct.
            for key in ("voice", "late_tolerance_ms", "shortfall_ms", "residual_shortfall_ms"):
                result.pop(key, None)
        return result

    # -- tools --------------------------------------------------------------------
    def list_clips(self) -> dict[str, Any]:
        corpus = self.store.load_corpus()["clips"]
        return {"clips": [
            {"clip_id": cid, **{k: v[k] for k in ("duration_s", "segment_count", "notes")}}
            for cid, v in corpus.items()
        ]}

    def open_clip(
        self, clip_id: str, run_id: str, repeat: int = 1, tts_budget: int | None = None,
        speed_policy: str = "natural",
    ) -> dict[str, Any]:
        if clip_id not in self.store.load_corpus()["clips"]:
            raise ValueError(f"unknown clip {clip_id!r}; call list_clips")
        native = self.store.clip_dir(clip_id) / "native_dub"
        directory = self.store.repeat_dir(run_id, clip_id, repeat)
        if self.subdir:
            directory = directory / "turns" / self.subdir
        directory.mkdir(parents=True, exist_ok=True)
        previous = read_json(directory / "episode.json") if (directory / "episode.json").is_file() else {}
        budget = int(tts_budget or os.getenv("MATHULA_DUB_TTS_BUDGET", DEFAULT_TTS_BUDGET))
        self.episode = {
            "clip_id": clip_id, "run_id": run_id, "repeat": int(repeat), "dir": str(directory),
            "native": str(native), "tts_budget": budget, "tts_used": int(previous.get("tts_used", 0)),
            "submissions": int(previous.get("submissions", 0)), "speed_policy": speed_policy,
        }
        if not self.store.read_run(run_id):
            self.store.write_run(run_id, parameters={"approach": "agent", "tts_budget": budget, "speed_policy": speed_policy})
        pass1 = read_json(native / "pass1_restored_transcript.json")
        self.segments = list(pass1["segments"])
        if not self.voices:
            self.voices = load_clip_voices(self.store, clip_id)
        # The renderer applies the job's pronunciation dictionary; measure with the same one so what is measured
        # is what is rendered.
        self.pronunciation_root = register_clip_pronunciation(self.store, clip_id)
        glossary_doc = read_json(native / "zulu_terminology_glossary.json")
        ledger_doc = read_json(native / "context_ledger.json")
        modes_path = native / "speaker_seriousness_modes.json"
        modes = read_json(modes_path).get("modes", {}) if modes_path.is_file() else {}
        mine = set(self.scope["segment_ids"]) if self.scope else None
        opened = {
            "clip_id": clip_id,
            "segments": [
                {"segment_id": s["segment_id"], "speaker_id": s.get("speaker_id"), "start_ms": s["start_ms"],
                 "end_ms": s["end_ms"], "english": s.get("restored_text") or s.get("source_text"),
                 **({"yours": s["segment_id"] in mine} if mine is not None else {})}
                for s in pass1["segments"]
            ],
            "context_ledger": ledger_doc.get("context_ledger", ledger_doc),
            "glossary": glossary_doc.get("zulu_terminology_glossary", glossary_doc),
            "speaker_modes": modes,
            "tts_budget_remaining": budget - self.episode["tts_used"],
            # Measurements use each speaker's assigned voice, the one the renderer will use.
            "voices": {sid: v.get("selected_voice") for sid, v in self.voices.items()},
        }
        if mine is not None:
            opened["your_scope"] = (
                "You dub ONLY the segments marked yours:true (one speaker turn). The rest of the clip is "
                "context so your retelling fits the whole story. submit_dub covers just your segments; "
                f"the next speech after your turn starts at {self.scope.get('next_start_ms')} ms."
            )
        return opened

    def measure_tts(
        self, text: str, start_ms: int, end_ms: int, next_start_ms: int | None = None,
        speaker_id: str = "SPEAKER_00", source_text: str = "",
    ) -> dict[str, Any]:
        remaining = self._spend(1)
        result = measure_text_fit(
            self.tts, text=text, start_ms=start_ms, end_ms=end_ms, next_start_ms=next_start_ms,
            speaker_id=speaker_id, source_text=source_text,
            output_root=self._measure_dir(),
            voice_info=pick_voice_info(self.voices, speaker_id, start_ms, end_ms, self.segments),
            pronunciation_root=self.pronunciation_root,
        )
        result = self._present(result)
        result["tts_budget_remaining"] = remaining
        return result

    def measure_tts_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        if not 1 <= len(requests) <= MAX_MEASURE_BATCH:
            raise ValueError(f"send between 1 and {MAX_MEASURE_BATCH} requests (you sent {len(requests)})")
        remaining = self._spend(len(requests))
        root = self._measure_dir()
        tts = self.tts

        def _one(request: dict[str, Any]) -> dict[str, Any]:
            out = measure_text_fit(
                tts, text=request["text"], start_ms=request["start_ms"], end_ms=request["end_ms"],
                next_start_ms=request.get("next_start_ms"), speaker_id=request.get("speaker_id", "SPEAKER_00"),
                source_text=request.get("source_text", ""), output_root=root,
                voice_info=pick_voice_info(self.voices, request.get("speaker_id"), request["start_ms"],
                                           request["end_ms"], self.segments),
                pronunciation_root=self.pronunciation_root,
            )
            return out

        with ThreadPoolExecutor(max_workers=min(6, len(requests))) as pool:
            results = [self._present(r) for r in pool.map(_one, requests)]
        return {"results": results, "tts_budget_remaining": remaining}

    def preview_turn_padding(
        self, members: list[dict[str, Any]], last_free_gap_ms: int = 0,
    ) -> dict[str, Any]:
        """Free arithmetic: if a speaker turn of these units finishes early, how much silence does the
        renderer add, and how much is left as dead air? members: {start_ms, end_ms, measured_ms}."""
        from .dub_padding import plan_turn_padding

        if not members:
            raise ValueError("members is empty")
        ordered = sorted(members, key=lambda m: int(m["start_ms"]))
        window = max(int(m["end_ms"]) for m in ordered) - int(ordered[0]["start_ms"])
        measured = sum(int(m["measured_ms"]) for m in ordered)
        plan = plan_turn_padding(ordered, shortfall_ms=max(0, window - measured), last_free_gap_ms=int(last_free_gap_ms))
        plan.update(window_ms=window, measured_ms=measured)
        return plan

    def check_text(self, zulu_text: str, source_text: str) -> dict[str, Any]:
        from .native_dub import _missing_required_literals

        native = Path(self._episode()["native"])
        glossary_doc = read_json(native / "zulu_terminology_glossary.json")
        ledger_doc = read_json(native / "context_ledger.json")
        known = _known_entity_tokens(
            source_text, glossary_doc.get("zulu_terminology_glossary", glossary_doc),
            ledger_doc.get("context_ledger", ledger_doc),
        )
        missing = _missing_required_literals(source_text=source_text, candidate_text=zulu_text)
        restated = find_restated_references(zulu_text)
        fabricated = find_fabricated_entities(zulu_text, known)
        problems: list[dict[str, Any]] = []
        if missing:
            problems.append({"gate": "required_literals", "evidence": [f"{m['literal']} x{m['required_count']}" for m in missing]})
        if restated:
            problems.append({"gate": "restated_reference", "evidence": [f"{h['first']} / {h['second']}" for h in restated]})
        if fabricated:
            problems.append({"gate": "fabricated_entity", "evidence": fabricated})
        doubled = find_unsupported_doubling(zulu_text, source_text)
        if doubled:
            problems.append({"flag": "unsupported_doubling", "evidence": doubled})
        if english_leakage(zulu_text) >= 0.2:
            problems.append({"flag": "english_leakage"})
        return {"ok": not any("gate" in p for p in problems), "problems": problems, "gates_checked": list(GATE_NAMES)}

    def recall_lessons(self, limit: int = 25) -> dict[str, Any]:
        lessons: list[dict[str, Any]] = []
        for clip_id in self.store.clip_ids():
            for label in self.store.load_labels(clip_id):
                if label.get("verdict") == "bad":
                    lessons.append({k: label.get(k) for k in (
                        "clip_id", "tags", "english", "zulu", "bad_substring", "corrected_zulu", "note")})
        lessons.sort(key=lambda item: item.get("clip_id") or "")
        return {"count": len(lessons), "lessons": lessons[-int(limit):]}

    def get_submission(self, unit_ids: list[str] | None = None) -> dict[str, Any]:
        """The current accepted units (all, or just ``unit_ids``), with any timing/gate flags from the last score."""
        if not self.last_items:
            return {"units": [], "count": 0, "note": "nothing has been accepted yet"}
        flags = {f["item_id"]: {"gates": f.get("gates"), "flags": f.get("flags")}
                 for f in (self.last_submit or {}).get("flagged_units", [])}
        wanted = set(unit_ids) if unit_ids else None
        units = [
            {k: u.get(k) for k in ("item_id", "start_ms", "end_ms", "speaker_id", "zulu_text", "source_segment_ids")}
            | ({"flagged": flags[str(u["item_id"])]} if str(u["item_id"]) in flags else {})
            for u in self.last_items if wanted is None or str(u["item_id"]) in wanted
        ]
        return {"units": units, "count": len(units)}

    def _judge_meaning(self, card: dict[str, Any], directory: Path) -> dict[str, Any] | None:
        """Ask the independent judge whether each unit keeps the English meaning (only changed units cost tokens).

        Units it scores 0-1 are material meaning loss: they are counted in the scorecard headline so the harness can
        hold the agent to them. A judge failure never blocks a submission; the headline simply carries no count.
        """
        if self.judge is None:
            return None
        units = [{"id": str(g["group_id"]), "source": g["source_text"], "dub": g["spoken_text"]}
                 for g in card["groups"] if g.get("source_text") and g.get("spoken_text")]
        try:
            verdicts = self.judge.judge(units)
        except Exception as exc:                      # a model outage must not lose the agent's work
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        from .dub_judge import summarise

        loss = [v for v in verdicts if v["meaning_preserved"] is not None and v["meaning_preserved"] <= 1]
        card["headline"]["meaning_loss_units"] = len(loss)
        atomic_write_json(directory / "judge.json", {"verdicts": verdicts, "summary": summarise(verdicts),
                                                      "usage": dict(self.judge.usage)})
        return {"loss": [{"item_id": v["id"], "issues": [i["detail"] for i in v["issues"]][:3],
                          "reads_back_as": (v.get("back_translation") or "")[:300]} for v in loss],
                "unjudged": sum(1 for v in verdicts if v["meaning_preserved"] is None)}

    def submit_dub(
        self, items: list[dict[str, Any]], approach_notes: str = "", partial: bool = False,
        remove: list[str] | None = None,
    ) -> dict[str, Any]:
        replaced: list[str] = []
        sent = len(items)
        if partial:
            if not self.last_items:
                return {"accepted": False, "errors": [
                    "partial=true updates your last accepted submission, but none has been accepted yet: send the "
                    "full set of units first (partial is false)."]}
            unknown = sorted({str(r) for r in remove or []} - {str(u["item_id"]) for u in self.last_items})
            if unknown:
                return {"accepted": False, "errors": [f"remove: no current unit(s) {unknown}; see get_submission"]}
            items, replaced = merge_partial(self.last_items, items, remove or ())
        episode = self._episode()
        native = Path(episode["native"])
        segments = read_json(native / "pass1_restored_transcript.json")["segments"]
        if self.scope:
            mine = set(self.scope["segment_ids"])
            segments = [s for s in segments if s["segment_id"] in mine]
        submission = {"clip_id": episode["clip_id"], "run_id": episode["run_id"], "repeat": episode["repeat"],
                      "approach_notes": approach_notes, "items": items}
        errors, warnings = validate_submission(submission, segments)
        if errors:
            if partial:
                errors = [*errors, f"(partial update: it replaced your current unit(s) {replaced or 'none'} by item_id; "
                                   "units are replaced only by the same item_id or listed in remove. To change how "
                                   "units are split, send the new units and put the old ids in remove.)"]
            return {"accepted": False, "errors": errors, "warnings": warnings}
        directory = Path(episode["dir"])
        card = score_submission_dir(
            native, submission, tts=self.tts, measure_root=self._measure_dir(),
            labels=self.store.load_labels(episode["clip_id"]),
            usage={"tts_measurements": episode["tts_used"]},
            speed_policy=episode.get("speed_policy", "pipeline"),
            segments=segments if self.scope else None,
            next_start_after_ms=self.scope.get("next_start_ms") if self.scope else None,
            voices=self.voices, pronunciation_root=self.pronunciation_root,
        )
        status = segment_timing_status(card, items)
        regressions = ratchet_regressions(self.locked_segments, status) if self.timing_ratchet else []
        key = timing_key(card["headline"])
        if regressions:
            # A submission that breaks segments which had fitted is rejected -- unless it is clearly better
            # overall (fewer collisions, then overruns). A strict ratchet can freeze a bad first draft: a rewrite
            # that repairs many units but breaks a few is a step forward, not something to block.
            previous = timing_key((self.last_submit or {}).get("headline") or {})
            if self.last_submit is None or key >= previous:
                return {
                    "accepted": False,
                    "errors": [
                        f"timing regression: {len(regressions)} segment(s) fitted their time frame in your previous "
                        "submission but no longer do, and this submission is not better overall, so it was not "
                        "accepted. Fix these and resubmit (the earlier submission still stands)."
                    ],
                    "regressions": regressions[:12],
                    "this_submission": _timing_summary(card["headline"]),
                    "previous_submission": _timing_summary((self.last_submit or {}).get("headline") or {}),
                }
        meaning = self._judge_meaning(card, directory)
        write_submission(directory / "submission.json", submission)
        episode["submissions"] += 1
        self.locked_segments = {sid for sid, entry in status.items() if entry["ok"]}
        atomic_write_json(directory / "scorecard.json", card)
        self.last_items = [dict(i) for i in items]
        self.last_submit = {
            "accepted": True, "warnings": warnings, "headline": card["headline"], "gates": card["gates"],
            **({"merged": {"sent": sent, "replaced": replaced, "units_now": len(items)}} if partial else {}),
            **({"regressions_accepted": regressions[:12],
                "note": "This submission broke segments that had fitted before but is better overall, so it was "
                        "accepted; fix those segments next."} if regressions else {}),
            **({"meaning": meaning} if meaning else {}),
            "timing_lock": {"fitting_segments": len(self.locked_segments), "total_segments": len(segments)},
            "turn_padding": [
                {"turn_id": t["turn_id"], "units": t["group_ids"], "shortfall_ms": t["shortfall_ms"],
                 "mechanism": t["padding"]["mechanism"], "padded_ms": t["padding"]["padded_ms"],
                 "residual_ms": t["padding"]["residual_ms"]}
                for t in card["turns"] if t["shortfall_ms"]
            ],
            "flagged_units": [
                {"item_id": g["group_id"], "gates": [n for n, v in g["gates"].items() if v["failed"]], "flags": g["flags"]}
                for g in card["groups"] if g["flags"] or any(v["failed"] for v in g["gates"].values())
            ],
        }
        return self.last_submit
