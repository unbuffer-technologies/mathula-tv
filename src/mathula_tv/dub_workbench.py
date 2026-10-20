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
    previous: list[dict[str, Any]], update: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fold a partial update into the last accepted submission.

    A previous unit is replaced if an updated unit has its item_id or covers any of its source segments;
    every other previous unit carries over untouched. Returns (merged units in time order, replaced ids).
    """
    new_ids = {str(u["item_id"]) for u in update}
    new_segments = {str(s) for u in update for s in u.get("source_segment_ids") or []}
    kept, replaced = [], []
    for unit in previous:
        covers = {str(s) for s in unit.get("source_segment_ids") or []}
        if str(unit["item_id"]) in new_ids or covers & new_segments:
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
        timing_ratchet: bool = True,
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
        # Timing ratchet: once a source segment has fitted its time frame in an accepted submission it must
        # keep fitting. A submission that breaks one is rejected, so fixing one unit can never silently
        # break another, and the clip's flow only ever improves.
        self.timing_ratchet = timing_ratchet
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

    def submit_dub(
        self, items: list[dict[str, Any]], approach_notes: str = "", partial: bool = False,
    ) -> dict[str, Any]:
        replaced: list[str] = []
        sent = len(items)
        if partial:
            if not self.last_items:
                return {"accepted": False, "errors": [
                    "partial=true updates your last accepted submission, but none has been accepted yet: send the "
                    "full set of units first (partial is false)."]}
            items, replaced = merge_partial(self.last_items, items)
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
                errors = [*errors, f"(partial update: it replaced your current unit(s) {replaced or 'none'}; every "
                                   "segment those covered must be covered again by your update)"]
            return {"accepted": False, "errors": errors, "warnings": warnings}
        directory = Path(episode["dir"])
        card = score_submission_dir(
            native, submission, tts=self.tts, measure_root=self._measure_dir(),
            labels=self.store.load_labels(episode["clip_id"]),
            usage={"tts_measurements": episode["tts_used"]},
            speed_policy=episode.get("speed_policy", "pipeline"),
            segments=segments if self.scope else None,
            next_start_after_ms=self.scope.get("next_start_ms") if self.scope else None,
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
