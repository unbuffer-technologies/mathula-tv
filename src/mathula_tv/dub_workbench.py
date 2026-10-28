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

import json
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
from .experience_store import ExperienceStore, glossary_changes

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
        self.source_corrections: list[dict[str, Any]] = []
        self.seed_unit_texts: dict[str, str] | None = None       # repair runs: the English each unit of the earlier submission was written against
        self.glossary_changes: list[dict[str, str]] = []
        self._pronunciation_lab: Any = None
        self._lost_sound_warned: set[tuple[str, str]] = set()
        self._readings_changed = 0                                # how many times a reading was changed in this run
        self.foreign_words_listed = False                       # whether the agent has looked at the English words its Zulu keeps
        self.pronunciation_choices: list[dict[str, Any]] = []   # spellings the agent chose with the lab, for the report
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
        """Where measured speech is kept.  Outputs are named by the unit's text, but what the voice says for a text depends on the readings in
        force, so every change of reading starts a new folder: the same text measured again then makes new speech instead of colliding with
        the old (the run that changed a reading re-measured a unit, and every measure and submit after it failed with a TTS idempotency error)."""
        base = self._measure_root or Path(self._episode()["dir"]) / "measure"
        return base / f"readings-{self._readings_changed}" if self._readings_changed else base

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
        corrections_path = native / "source_corrections.json"
        self.source_corrections = (read_json(corrections_path).get("corrections") or []) if corrections_path.is_file() else []
        previous_path = native / "previous_glossary.json"
        self.glossary_changes = (glossary_changes(read_json(previous_path), glossary_doc)
                                 if previous_path.is_file() else [])
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
        sheet = self._delivery_sheet(clip_id)
        if sheet:
            from .delivery import for_agent

            opened["delivery"] = for_agent(sheet, sorted(mine) if mine is not None else None)
        switched = self._code_switch(clip_id, sorted(mine) if mine is not None else None)
        if switched:
            opened["code_switch"] = switched
        if self.source_corrections:
            opened["source_corrections"] = {
                "note": ("The English below was corrected against the speaker's audio: the transcript had misheard these "
                         "words. The segments' english already carries the correction; translate that meaning, never the "
                         "misheard word."),
                "corrections": [{"segment_id": c["segment_id"], "transcript_had": c["token"],
                                 "speaker_said": c["replacement"] or "(nothing: he broke the sentence off and started again; it is not part of the speech)",
                                 "why": c.get("reason", "")} for c in self.source_corrections],
            }
        return opened

    def _code_switch(self, clip_id: str, segment_ids: list[str] | None) -> dict[str, Any] | None:
        """What the speaker really said where they switch language (see code_switch): read from the job, where prepare_agent_inputs writes it."""
        from .code_switch import FILE, for_agent, load

        entry = self.store.load_corpus()["clips"][clip_id]
        doc = load(Path(self.store.root).parent / "jobs" / str(entry["job_id"]) / "native_dub" / FILE)
        return for_agent(doc, segment_ids) if doc else None

    def _delivery_sheet(self, clip_id: str) -> dict[str, Any] | None:
        """How the clip's speakers deliver (see delivery): read from the job, where prepare_agent_inputs writes it."""
        from .delivery import SHEET_NAME, load_sheet

        entry = self.store.load_corpus()["clips"][clip_id]
        return load_sheet(Path(self.store.root).parent / "jobs" / str(entry["job_id"]) / "native_dub" / SHEET_NAME)

    # -- pronunciation lab (see pronunciation_lab) ----------------------------------
    def _lab(self) -> Any:
        """The clip's pronunciation lab: the speaker's sounds for a word, and the Zulu voice's sounds for a spelling."""
        if self._pronunciation_lab is None:
            from .pronunciation_lab import PronunciationLab

            episode = self._episode()
            entry = self.store.load_corpus()["clips"][episode["clip_id"]]
            voice = next((v.get("selected_voice") for v in self.voices.values() if v.get("selected_voice")), "zu-ZA-ThembaNeural")
            self._pronunciation_lab = PronunciationLab(
                Path(self.store.root).parent / "jobs" / str(entry["job_id"]), voice=voice, key=os.environ["AZURE_SPEECH_KEY"],
                region=os.environ["AZURE_SPEECH_REGION"].strip().lower(), work_dir=Path(episode["dir"]) / "pronunciation_lab")
        return self._pronunciation_lab

    def _sync_lab_context(self) -> None:
        """Give the lab the Zulu as written now and the readings already chosen, so a word is also heard in the sentence it is spoken in."""
        from .pronunciation_memory import entries_for, memory_path

        lab = self._lab()
        submission = Path(self._episode()["dir"]) / "submission.json"
        items = read_json(submission).get("items", []) if submission.is_file() else []
        lab.sentences = [str(i.get("zulu_text") or "") for i in items]
        self._lab_item_ids = [str(i.get("item_id") or "") for i in items]
        readings: dict[str, str] = {}
        lexicon = Path(self.store.clip_dir(self._episode()["clip_id"])) / "native_dub" / "pronunciation_names.json"
        if lexicon.is_file():
            readings.update({str(e["display_text"]): str(e["tts_text"]) for e in read_json(lexicon).get("entries", []) if e.get("display_text") and e.get("tts_text")})
        for entry in entries_for(memory_path(Path(self.store.root).parent), lab.voice):
            if entry.get("display_text") and entry.get("tts_text") and not entry.get("disabled"):
                readings.setdefault(str(entry["display_text"]), str(entry["tts_text"]))
        lab.readings = readings

    def _known_readings(self) -> set[str]:
        """Texts that already have a reading (the clip's lexicon and the shared memory): the lab does not need to look at them."""
        from .pronunciation_lab import failing_readings
        from .pronunciation_memory import entries_for, memory_path

        self._sync_lab_context()
        known: set[str] = set()
        lexicon = Path(self.store.clip_dir(self._episode()["clip_id"])) / "native_dub" / "pronunciation_names.json"
        if lexicon.is_file():
            known |= {str(e["display_text"]).casefold() for e in read_json(lexicon).get("entries", [])}
        saved = entries_for(memory_path(Path(self.store.root).parent), self._lab().voice)
        known |= {str(e["display_text"]).casefold() for e in saved}
        known -= failing_readings(self._lab(), saved)          # a saved reading an English listener does not understand is back on the list
        return known

    def pronunciation_check_pending(self) -> list[str]:
        """The English words that still need a look, for the harness's one reminder; empty when there is no need (or the agent already listed them).

        It runs the same sweep as ``list_foreign_words`` (the model's reading of the Zulu is cached, so the agent's own call costs nothing more)
        and leaves out words that already have a reading an English listener understands.  A failure means no reminder, never a blocked run."""
        if self.foreign_words_listed or not hasattr(self, "list_foreign_words"):
            return []
        try:
            return [str(entry["word"]) for entry in self._foreign_candidates()["words"]]
        except Exception:
            return []

    def _restore_written_words(self, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """Units whose text carries a chosen respelling instead of the word get the word back (see ``restore_written_words``)."""
        from .pronunciation_memory import load, memory_path, restore_written_words

        try:
            entries = list(load(memory_path(Path(self.store.root).parent))["entries"].values())
        except Exception:
            return items, []
        return restore_written_words(items, entries)

    def pronunciation_evidence(self, words: list[str]) -> list[dict[str, Any]]:
        """What the lab measures for each of these words as the Zulu writes them now, for the harness's reminder: no tries are spent, and a
        word that cannot be measured is left out."""
        found: list[dict[str, Any]] = []
        try:
            self._sync_lab_context()
        except Exception:
            return found
        for word in words[:6]:
            try:
                measured = self._lab().as_written(word)
            except Exception:
                measured = None
            if measured:
                measured["units"] = self._lab().units_with(word, getattr(self, "_lab_item_ids", ()))
                found.append(measured)
        return found

    def saved_readings_evidence(self) -> list[dict[str, Any]]:
        """How the readings the shared memory already holds measure now, in the sentences of this text, for the harness's reminder.  Only
        readings of English words that this text has; no tries are spent, nothing is scored or cut off, and one that cannot be measured is left out."""
        from .pronunciation_memory import entries_for, memory_path

        rows: list[dict[str, Any]] = []
        try:
            self._sync_lab_context()
            lab = self._lab()
            saved = [e for e in entries_for(memory_path(Path(self.store.root).parent), lab.voice) if e.get("kind") == "english_code_switch"]
        except Exception:
            return rows
        for entry in saved:
            word, spelling = str(entry["display_text"]), str(entry["tts_text"])
            if lab._context(word) is None:
                continue
            try:
                measured = lab.as_written(word, spelling)
            except Exception:
                measured = None
            if measured:
                measured["units"] = lab.units_with(word, getattr(self, "_lab_item_ids", ()))
                rows.append(measured)
            if len(rows) >= 10:
                break
        return sorted(rows, key=lambda row: row["closeness"])

    def list_foreign_words(self) -> dict[str, Any]:
        """English words that the current Zulu keeps and that the speaker says in the source audio: the ones worth checking."""
        self.foreign_words_listed = True
        return self._foreign_candidates()

    def _foreign_candidates(self) -> dict[str, Any]:
        from .pronunciation_lab import MAX_TRIES_PER_CLIP, candidate_words
        from .pronunciation_search import looks_isizulu

        submission = Path(self._episode()["dir"]) / "submission.json"
        if not submission.is_file():
            raise ValueError("submit a dub first: there is no Zulu text to look through yet")
        texts = [str(i.get("zulu_text") or "") for i in read_json(submission).get("items", [])]
        named: list[dict[str, str]] = []
        if self.judge is not None and hasattr(self.judge, "foreign_words"):
            try:
                named = self.judge.foreign_words(texts)          # the English words a model reads out of the Zulu, whatever their spelling
            except Exception:                                    # an outage leaves the rule-based sweep, never an empty list
                named = []
        found = candidate_words(texts, self._lab().transcript, looks_isizulu=looks_isizulu, exclude=self._known_readings(), named=named)
        return {"words": found, "note": "Check only the ones that matter. Never glue a Zulu prefix onto an English word the speaker says as it is.",
                "tries_left_in_clip": MAX_TRIES_PER_CLIP - self._lab().total_tries}

    def try_pronunciation(self, word: str, spellings: list[str]) -> dict[str, Any]:
        """Hear what the Zulu voice does with up to three spellings of ``word`` and how they differ from the speaker's sounds."""
        from .pronunciation_lab import PronunciationBudgetExhausted

        try:
            self._sync_lab_context()
            return self._lab().try_spellings(word, spellings)
        except PronunciationBudgetExhausted as exc:
            raise BudgetExhausted(str(exc)) from exc

    def use_pronunciation(self, word: str, spelling: str) -> dict[str, Any]:
        """Make ``spelling`` the way ``word`` is spoken: in your own measurements now, in the render, and in the shared memory for later clips."""
        from . import dub_pronunciation
        from .pronunciation_memory import memory_path, remember

        lab, episode = self._lab(), self._episode()
        tried = lab.tried.get(word, {}).get(spelling)
        if tried is None:
            raise ValueError(f"try {spelling!r} for {word!r} with try_pronunciation before using it")
        if tried.get("lost_in_its_sentence") and (word, spelling) not in self._lost_sound_warned:
            self._lost_sound_warned.add((word, spelling))       # said once: choosing it again after this is the agent's decision
            raise ValueError(
                f"in the sentence {word!r} is spoken in, the voice loses {' '.join(tried['lost_in_its_sentence'])} with {spelling!r} (it says "
                f"{tried.get('said_in_its_sentence')!r}; the speaker says {' '.join(lab.source(word)['phones'])!r}), so it will sound different in the video "
                "than in the lab. Try letters that keep the sound, then choose; if you still want this spelling, call use_pronunciation again.")
        clip_dir = Path(self.store.clip_dir(episode["clip_id"]))
        lexicon_path = clip_dir / "native_dub" / "pronunciation_names.json"
        lexicon = read_json(lexicon_path) if lexicon_path.is_file() else {"schema": "mathula-names-lexicon-v1", "voice": lab.voice, "entries": []}
        lexicon["entries"] = [e for e in lexicon.get("entries", []) if str(e.get("display_text", "")).casefold() != word.casefold()]
        lexicon["entries"].append({"display_text": word, "tts_text": spelling, "kind": "english_code_switch", "confidence": 0.7,
                                   "source": "agent-pronunciation", "closeness": tried["closeness"]})
        lexicon_path.parent.mkdir(parents=True, exist_ok=True)
        lexicon_path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=1), encoding="utf-8")
        dub_pronunciation._REGISTERED.discard(str(clip_dir))                 # measure from now on with the new reading
        self.pronunciation_root = dub_pronunciation.register_clip_pronunciation(self.store, episode["clip_id"])
        self._readings_changed += 1
        entry = self.store.load_corpus()["clips"][episode["clip_id"]]
        remember(memory_path(Path(self.store.root).parent), word, spelling, source="agent", voice=lab.voice, job_id=str(entry["job_id"]),
                 kind="english_code_switch", note=f"chosen by the agent with the lab; closeness {tried['closeness']}; voice sounds {tried['voice_sounds']}")
        choice = {"word": word, "spelling": spelling, "closeness": tried["closeness"], "speaker_sounds": " ".join(lab.source(word)["phones"]),
                  "voice_sounds": tried["voice_sounds"]}
        self.pronunciation_choices = [c for c in self.pronunciation_choices if c["word"] != word] + [choice]
        return {"ok": True, **choice, "note": "your measurements and the render now use this reading wherever the word is written. Keep the WORD in the text: never type the spelling into it (the voice respells the word when it speaks it, and the text is what the captions show). Re-measure any unit that contains the word."}

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
        items, restored_words = self._restore_written_words(items)
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
        warnings = [*warnings, *restored_words]
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
