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
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .atomic_io import atomic_write_json, read_json
from .dub_fit import measure_text_fit
from .voice_limits import MURMUR_NOTE, murmurs
from .dub_submission import score_submission_dir, validate_submission, write_submission
from .episode_scoring import (
    GATE_NAMES, _known_entity_tokens, describe_missing_literal, english_leakage, find_fabricated_entities,
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


# How much Zulu fills a source window at a voice's natural speed: characters (spaces included, pauses and ~markers excluded) per second of
# window.  Measured on the finished, accepted dubs of one 25-unit clip from three runs (12.4, 12.5, 12.4; per unit 8 to 15: figures and
# English words run longer).  Known for the voice measured only; a voice without an entry gets no guide.
CHARS_PER_SECOND = {"zu-ZA-ThembaNeural": 12.4}


def length_guide(segments: list[dict[str, Any]], voices: dict[str, Any]) -> dict[str, int]:
    """segment_id -> the Zulu length (characters, to the nearest 5) that fills the segment's window at its speaker's voice's natural speed."""
    guide: dict[str, int] = {}
    for s in segments:
        rate = CHARS_PER_SECOND.get(str((voices.get(s.get("speaker_id")) or {}).get("selected_voice") or ""))
        seconds = (int(s["end_ms"]) - int(s["start_ms"])) / 1000
        if rate and seconds > 0:
            guide[str(s["segment_id"])] = max(5, int(round(seconds * rate / 5)) * 5)
    return guide


LENGTH_GAP_MIN_PCT = 6           # a unit within this share of its window gets no length advice


def spoken_chars(text: str) -> int:
    """Characters the length guide counts: spaces included, pause and ~ markers not."""
    from .delivery import PAUSE_MARKER

    return len(re.sub(r"~|\[[^\]]*\]", "", PAUSE_MARKER.sub(" ", text or "")))     # a ~pause800 marker is silence: its letters are not spoken


SHORT_SHORTFALL_MS = 450         # a unit this close to its window (or closer) is told to stay as it is, not given characters to add
ADD_NOTE = ("only with what the English says; if it is all said, a shorter unit is correct and a ~pause the speaker took is the honest filler")
LEAVE_SHORT_NOTE = ("silence closes this gap and it is a share of the whole clip's time, so leave the unit as it is: adding a word the speaker did not say "
                    "to fill it costs the listener an odd or wrong phrase")


def length_gap(text: str, measured: dict[str, Any]) -> dict[str, Any]:
    """How much to write to fill the window, from this very measurement: the unit's characters, and, when it speaks more than 6% short of its window
    (or past it), the characters to add (or cut), to the nearest 5, at the rate this text measured.  The agent saw only 'shortfall_pct' before and had
    to guess the size of the edit, so a first draft 27% short took most of a run to close.  A gap of SHORT_SHORTFALL_MS or less gets no characters to
    add: prod-74's 'Sanibonani.' (825 of 1120 ms) was told to add 5 characters and became 'Sanibonani ntambama.', which Zulu does not say; a short
    faithful line that leaves a fraction of a second is right, and the renderer closes it with silence."""
    chars, spoken, window = spoken_chars(text), int(measured.get("measured_ms") or 0), int(measured.get("window_ms") or 0)
    if not (chars and spoken and window):
        return {}
    gap = chars * (window / spoken - 1)
    if abs(gap) < chars * LENGTH_GAP_MIN_PCT / 100:
        return {"chars": chars}
    if gap > 0 and window - spoken <= SHORT_SHORTFALL_MS:
        return {"chars": chars, "shortfall_ms": window - spoken, "leave_short": LEAVE_SHORT_NOTE}
    amount = max(5, int(round(abs(gap) / 5)) * 5)
    if gap > 0:
        return {"chars": chars, "add_about_chars": amount, "add_note": ADD_NOTE}
    return {"chars": chars, "cut_about_chars": amount}


WORDS_MEASURED = 30             # words, names and initialisms the word list measures as written against the speaker (one clip of the voice each): all of the list, which candidate_words caps at 30. At 20 only the first 20 in the order of the text were measured, so Western (0.42) and the others after them sat unmeasured at the end of prod-52's list
PAIR_EXPECTED_MIN = 5.0          # naturalness: a pair of common words that never meets although chance would put it 5 times in the corpora (false alarm 1.0-1.5% of real sentences)
GATE_EVIDENCE_CHARS = 700          # characters of one failed gate's evidence a submit result carries (check_text gives it whole)
MAX_DECISIONS = 60              # settled terms the workbench keeps and shows (submit_dub decisions)
FIGURES_SHOWN = 30             # written numbers a submit result shows with the words the voice is given for them (figures_as_spoken)
UNIT_TEXTS_SHOWN = 8           # flagged units and claims whose current Zulu text a submit result carries, so the agent need not call get_submission to read them
UNIT_TEXT_CHARS = 360
CLAIMS_PER_SUBMIT = 6           # units a submit result asks the agent to check against the English, at most

SAVED_READINGS_MEASURED = 24   # saved readings measured for the reminder (the weakest six are shown): the cost is one clip each, once per run


class BudgetExhausted(RuntimeError):
    """Raised when the agent asks for more measurements than its episode budget allows."""




def units_overview(card: dict[str, Any]) -> list[str]:
    """Every unit in one short string, so the agent can see the whole submission's fit without reading it: ``u14 858ch -1.0s`` is 858 characters ending 1.0 s before its window, ``+0.4s`` running
    0.4 s over, ``0`` fitting.  The latest submit result is the one thing compaction keeps, so this is what stops a get_submission turn just to see how the rest stands."""
    lines = []
    for g in card.get("groups", []):
        window, measured = int(g.get("window_ms") or 0), int(g.get("measured_ms") or 0)
        gap = measured - window
        fit = "0" if abs(gap) < 250 else f"{'+' if gap > 0 else '-'}{abs(gap) / 1000:.1f}s"
        lines.append(f"{g['group_id']} {len(str(g.get('spoken_text') or ''))}ch {fit}")
    return lines

def settled_units(card: dict[str, Any]) -> list[str]:
    """The units with no flag and no failed gate at this submit: the ones to leave alone unless a point names them (kept in the latest submit result, so compaction does not lose it)."""
    return [str(g["group_id"]) for g in card.get("groups", []) if not g.get("flags") and not any(v["failed"] for v in g["gates"].values())]


def speaker_rates(card: dict[str, Any]) -> dict[str, float]:
    """Characters per second of speech each speaker's voice measured in this submit (the median over the speaker's units): the rate a length edit should be reckoned at."""
    rates: dict[str, list[float]] = {}
    for g in card.get("groups", []):
        measured = int(g.get("measured_ms") or 0)
        text = str(g.get("spoken_text") or "")
        if measured > 0 and text:
            rates.setdefault(str(g.get("speaker_id") or ""), []).append(len(text) / (measured / 1000.0))
    return {speaker: round(sorted(values)[len(values) // 2], 1) for speaker, values in rates.items() if values}


class Workbench:
    _spoken_cache: dict[str, Any] = {}               # the spoken-Zulu transcripts, read once per process

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
        self.decisions: dict[str, dict[str, Any]] = {}              # how the agent settled recurring terms (submit_dub decisions), kept across compaction and into a repair run
        self.seed_decisions: dict[str, dict[str, Any]] = {}
        self.seed_unit_texts: dict[str, str] | None = None       # repair runs: the English each unit of the earlier submission was written against
        self.glossary_changes: list[dict[str, str]] = []
        self._pronunciation_lab: Any = None
        self._lost_sound_warned: set[tuple[str, str]] = set()
        self._readings_changed = 0                                # how many times a reading was changed in this run
        self.foreign_words_listed = False                       # whether the agent has looked at the English words its Zulu keeps
        self.pronunciation_choices: list[dict[str, Any]] = []   # spellings the agent chose with the lab, for the report
        self.glossary_corrections: list[str] = []                # glossary terms put right when the clip was opened (see english_prefix)
        from .pronunciation_ledger import Ledger

        self.ledger = Ledger()                                   # what the agent was shown about each word and what it did (see pronunciation_ledger)
        # Timing ratchet: once a source segment has fitted its time frame in an accepted submission it must
        # keep fitting. A submission that breaks one is rejected, so fixing one unit can never silently
        # break another, and the clip's flow only ever improves.
        self.timing_ratchet = timing_ratchet
        self.judge = judge        # a dub_judge.UnitJudge, or None for no meaning check
        self.locked_segments: set[str] = set()
        self.episode: dict[str, Any] | None = None
        self.last_submit: dict[str, Any] | None = None   # the latest accepted submit_dub result, for the harness
        self.claims_shown: set[tuple[str, str]] = set()  # (unit id, spoken text) the agent has already been asked to check

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

    SENTENCE_TIMING_NOTE = ("per sentence of this unit: its characters, the seconds the voice took (dub_s), where it starts in the unit (at_s), and the source speaker's matching sentence "
                            "(source_s, source_at_s: matched by share of the speech, so only a guide). Where a unit is over or short, the sentences with the most seconds for their characters "
                            "or that run past where the source speaker's sentence was are where the time is.")

    @staticmethod
    def _measured_sentences(audio_path: Any) -> dict[str, Any] | None:
        """The sentences of a measured unit with their seconds, from the manifest the voice wrote beside the audio (OmniVoice: where each sentence falls, from the service's word times)."""
        try:
            manifest = Path(str(audio_path)).with_suffix(".omnivoice.json")
            natural = json.loads(manifest.read_text(encoding="utf-8")).get("natural") or {}
            sentences = natural.get("sentences")
        except Exception:  # noqa: BLE001 - another voice, a cached file without it: no sentence timing, never a failed measurement
            return None
        if not sentences:
            return None
        rows = [{"n": x["n"], "chars": x["chars"], "dub_s": x["dub_s"], "at_s": x["start_s"], "source_s": x.get("source_s"), "source_at_s": x.get("source_start_s"), "starts": x["text"][:28]} for x in sentences]
        return {"sentences": rows, "note": Workbench.SENTENCE_TIMING_NOTE}

    FIT_NOISE_PCT = 1.0          # a unit this close to its window cannot be told from a fit by ear: the same unit's length moves by about 2% between wordings of nearly the same size
    FIT_NOTE = ("to_window_ms is the window minus the measured length (negative: it runs past the window); chars_to_change is that gap at this unit's own measured pace (ms_per_char): positive, add about that many "
                "characters; negative, take about that many out, anywhere in the unit's speech. The voice's length is not linear in characters at this size: this clip's unit measured 0.3% short at 864 characters and 2.5% short at "
                "868 after a different wording, so when within_noise is true the unit fits as far as anyone can hear, and a further edit is as likely to move it away from the window as nearer.")

    @staticmethod
    def _fit_hint(window_ms: Any, measured_ms: Any, chars: Any) -> dict[str, Any] | None:
        """The gap between a unit's measured length and its window, and what that is in characters at the unit's own measured pace.  Arithmetic from the measurement; the agent chooses where the words change."""
        try:
            window, measured, count = int(window_ms), int(measured_ms), int(chars)
        except (TypeError, ValueError):
            return None
        if window <= 0 or measured <= 0 or count <= 0:
            return None
        per_char = measured / count
        gap = window - measured
        return {"to_window_ms": gap, "ms_per_char": round(per_char, 1), "chars_to_change": int(round(gap / per_char)), "within_noise": abs(gap) <= Workbench.FIT_NOISE_PCT / 100.0 * window}

    @staticmethod
    def _verdict(result: dict[str, Any]) -> str | None:
        """What a measurement means, in one sentence, from the numbers it carries.  The agent spent turns 9-13 of prod-114 working out why a unit that ended 340 ms past its window reported
        fits_naturally true and overrun_pct 0 (the free silence before the next speech is what is allowed); the answer was in the fields, which were not shown together."""
        measured, window = result.get("measured_ms"), result.get("window_ms")
        if not measured or not window:
            return None
        over = int(measured) - int(window)
        gap, tolerance = result.get("free_gap_after_ms"), result.get("late_tolerance_ms")
        if result.get("direction") == "expand":
            left = result.get("residual_shortfall_pct", result.get("shortfall_pct"))
            return f"ends {-over} ms before its window ends" + (f"; {left}% of the window is left as dead air after the renderer's own padding" if left else "")
        if over <= 0:
            return f"fits: ends {-over} ms inside its window"
        allowed = int(tolerance if tolerance is not None else (gap or 0))
        if float(result.get("required_speed_percent") or result.get("overrun_pct") or 0.0) == 0.0:
            return (f"runs {over} ms past its window end into the {gap if gap is not None else allowed} ms of free silence before the next speech: allowed, it collides with nothing")
        return f"runs {over} ms past its window end; only {allowed} ms of that is allowed (the free silence before the next speech), so it overruns by {over - allowed} ms and would collide"

    def _present(self, result: dict[str, Any]) -> dict[str, Any]:
        """Show timing the way this episode's speed policy sees it.

        Under "natural" there is no speed-up to lean on, so the agent is shown an overrun,
        not a "required speed", and none of the speed-rescue bookkeeping."""
        timing = self._measured_sentences(result.pop("audio_path", None))
        if timing:
            if self.__dict__.get("_sentence_note_given"):
                timing.pop("note", None)                                  # said once: the agent has it, and every measurement result is re-read on every later turn
            else:
                self.__dict__["_sentence_note_given"] = True
            result["sentence_timing"] = timing
        verdict = self._verdict(result)
        if verdict:
            result["verdict"] = verdict
        hint = self._fit_hint(result.get("window_ms"), result.get("measured_ms"), result.get("chars"))
        if hint:
            if not self.__dict__.get("_fit_note_given"):
                self.__dict__["_fit_note_given"] = True                    # said once: every measurement result is re-read on every later turn
                hint["note"] = self.FIT_NOTE
            result["to_fit"] = hint
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
        saved = directory / "decisions.json"
        if not self.decisions:          # a resumed run keeps its own; a repair run starts with the earlier run's
            try:
                loaded = read_json(saved).get("decisions") if saved.is_file() else None
            except (OSError, ValueError):
                loaded = None
            for entry in (loaded or list(self.seed_decisions.values())):
                if isinstance(entry, dict) and entry.get("english") and entry.get("zulu"):
                    self.decisions[str(entry["english"]).casefold()] = dict(entry)
        pass1 = read_json(native / "pass1_restored_transcript.json")
        self.segments = list(pass1["segments"])
        if not self.voices:
            self.voices = load_clip_voices(self.store, clip_id)
        # The renderer applies the job's pronunciation dictionary; measure with the same one so what is measured
        # is what is rendered.
        try:        # the clip's lexicon carries what the memory holds now, not what an earlier run left in it (a reading switched off must stop counting as known)
            from .pronunciation_memory import apply_to_lexicon, memory_path as _memory_path

            first_voice = next((v.get("selected_voice") for v in self.voices.values() if v.get("selected_voice")), None)
            clip_lexicon = Path(self.store.clip_dir(clip_id)) / "native_dub" / "pronunciation_names.json"
            if clip_lexicon.is_file():
                apply_to_lexicon(clip_lexicon, _memory_path(Path(self.store.root).parent), first_voice, skip=self._ordinary_zulu, per_voice=True)
        except Exception:
            pass
        self.pronunciation_root = register_clip_pronunciation(self.store, clip_id)
        glossary_doc = read_json(native / "zulu_terminology_glossary.json")
        from .english_prefix import fix_glossary

        if isinstance(glossary_doc.get("zulu_terminology_glossary"), dict):          # a form like maFighter never reaches the agent, whatever the file says
            glossary_doc["zulu_terminology_glossary"], self.glossary_corrections = fix_glossary(glossary_doc["zulu_terminology_glossary"])
        ledger_doc = read_json(native / "context_ledger.json")
        modes_path = native / "speaker_seriousness_modes.json"
        modes = read_json(modes_path).get("modes", {}) if modes_path.is_file() else {}
        mine = set(self.scope["segment_ids"]) if self.scope else None
        corrections_path = native / "source_corrections.json"
        self.source_corrections = (read_json(corrections_path).get("corrections") or []) if corrections_path.is_file() else []
        previous_path = native / "previous_glossary.json"
        self.glossary_changes = (glossary_changes(read_json(previous_path), glossary_doc)
                                 if previous_path.is_file() else [])
        guide = length_guide(list(pass1["segments"]), self.voices)
        omni_guide = self._omnivoice_guide(list(pass1["segments"]))
        if omni_guide:                                                # OmniVoice: the length that fills a window at the pace of THAT segment's reference clip, not an Azure voice's 12.4 characters a second
            guide = omni_guide
        clip_ms = max([int(s["end_ms"]) for s in pass1["segments"]] or [1])
        opened = {
            "clip_id": clip_id,
            "segments": [
                {"segment_id": s["segment_id"], "speaker_id": s.get("speaker_id"), "start_ms": s["start_ms"],
                 "end_ms": s["end_ms"], "share_of_clip_pct": round((int(s["end_ms"]) - int(s["start_ms"])) / clip_ms * 100.0, 2), "english": s.get("restored_text") or s.get("source_text"),
                 **({"fits_about_chars": guide[str(s["segment_id"])]} if str(s["segment_id"]) in guide else {}),
                 **({"murmur_in_the_english": murmurs(s.get("restored_text") or s.get("source_text"))} if murmurs(s.get("restored_text") or s.get("source_text")) else {}),
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
        opened["share_note"] = ("share_of_clip_pct is a segment's window as a share of the whole clip. The bar counts dead air over the whole clip, so a short segment can cost the bar little "
                                "however far it is off: weigh your effort by it.")
        if any(s.get("murmur_in_the_english") for s in opened["segments"]):
            opened["murmur_note"] = MURMUR_NOTE
        if guide:
            opened["length_guide"] = (
                "fits_about_chars is the Zulu length (characters, spaces included, pauses and ~markers not counted) that fills a segment's window at "
                "this voice's natural speed, about 12 characters per second. Dubs that fitted landed within roughly 20% of it; first drafts that "
                "retold compactly landed about 30% short, which then took most of the run to close unit by unit. Draft to this length from the start, "
                "with what the speaker said (padding with additions fails the restatement and fabrication gates). Each measurement then gives the unit's "
                "chars and add_about_chars or cut_about_chars, the edit that closes the gap at the rate that text measured. A unit "
                "that leaves under half a second stays as it is: leave_short means add nothing.")
            if omni_guide:
                opened["length_guide"] = opened["length_guide"].replace(
                    "this voice's natural speed, about 12 characters per second", "the pace of the reference clip cut from the source speaker's own audio in that window (the voice is cloned from "
                    "it, and its pace sets how fast the voice speaks, so the length differs from one segment to the next)")
        if os.environ.get("MATHULA_AGENT_HINTS", "0").strip().lower() in {"1", "on", "true", "yes"}:
            hints = self._pairs_db().hints([(str(s["segment_id"]), str(s["english"] or "")) for s in opened["segments"] if s.get("yours", True)])
            if hints:
                opened["human_translation_hints"] = hints
                opened["human_translation_note"] = (
                    "How professional translators wrote phrases that occur in this clip (Umsuka news pairs, Autshumato government text): evidence to weigh, not answers. The pairs are "
                    "sometimes shifted by a line, the government text is older and formal, and a phrase can have several good renderings. Ask human_translations(phrase) for more pairs.")
        if mine is not None:
            opened["your_scope"] = (
                "You dub ONLY the segments marked yours:true (one speaker turn). The rest of the clip is "
                "context so your retelling fits the whole story. submit_dub covers just your segments; "
                f"the next speech after your turn starts at {self.scope.get('next_start_ms')} ms."
            )
        self._add_phrasing(opened, clip_id)
        self._add_dialogue(opened, clip_id, ledger_doc.get("context_ledger", ledger_doc))
        self._add_listener_answers(opened, clip_id)
        self._add_video(opened, clip_id)
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

    def _source_word_timings(self, clip_id: str) -> list[dict[str, Any]]:
        """The source audio's word timings for this clip's job (read once)."""
        if getattr(self, "_source_words", None) is None:
            from .comma_check import load_words

            entry = self.store.load_corpus()["clips"][clip_id]
            self._source_words = load_words(Path(self.store.root).parent / "jobs" / str(entry["job_id"]))
        return self._source_words

    VIDEO_NOTE = ("What the uploader wrote about this video: its title, channel and description (and chapter titles when it has them). A human wrote it, so it spells the names of the story "
                  "(people, parties, places, events) and says what the report is about; the transcript and the captions are machine readings. Use it to settle a garbled name or a doubtful "
                  "word, and to know the setting, never as a source of facts the speaker did not say, and never to put a name where the speaker said something else.")

    def _add_video(self, opened: dict[str, Any], clip_id: str) -> None:
        """The video's own title, channel and description, from ``input/youtube_source.json``; nothing for a job that did not come from YouTube."""
        from .youtube_captions import video_context

        try:
            entry = self.store.load_corpus()["clips"][clip_id]
            video = video_context(Path(self.store.root).parent / "jobs" / str(entry["job_id"]))
        except Exception:                                             # noqa: BLE001 - context for the agent must never stop a run
            return
        if video:
            opened["video"] = {**video, "note": self.VIDEO_NOTE}

    def _add_listener_answers(self, opened: dict[str, Any], clip_id: str) -> None:
        """What the listener answered to this video's earlier missing-evidence questions (console, "What the agent could not find out"): the words that were really said where the
        transcript and the captions were both garbled.  Tied to what the agent asked for, not to a unit id, because the units are cut afresh in every run.  Opened with the clip, so
        compaction keeps it."""
        from .language_questions import notes_for_agent

        try:
            entry = self.store.load_corpus()["clips"][clip_id]
            answers = notes_for_agent(Path(self.store.root).parent / "jobs" / str(entry["job_id"]))
        except Exception:                                             # noqa: BLE001 - an advisory view must never fail opening a clip
            return
        if answers:
            opened["listener_answers"] = answers
            opened["listener_answers_note"] = ("The listener heard the source audio and answered what you said you needed (missing_evidence) in an earlier run on this video. Where an answer "
                                               "covers a stretch of your English, it replaces both machine readings of that stretch: translate it, and say so in a choice point.")

    def _add_dialogue(self, opened: dict[str, Any], clip_id: str, ledger: dict[str, Any]) -> None:
        """The clip as a conversation (see dialogue_view): turns in order, the question each answers, and the murmurs of other people inside a turn, from the segments and YouTube's speaker
        marks.  Opened with the clip, so compaction keeps it.  Evidence, never a gate; a clip with one speaker has none."""
        from . import youtube_captions
        from .dialogue_view import NOTE, build

        try:
            if len({str(s.get("speaker_id") or "") for s in opened["segments"]}) < 2:
                return
            entry = self.store.load_corpus()["clips"][clip_id]
            lines: list[tuple[float, float, str]] = []
            found = youtube_captions.find(Path(self.store.root).parent / "jobs" / str(entry["job_id"]))
            if found:
                lines = youtube_captions.events(found[0])
            roles = {}
            for role in ledger.get("speaker_roles") or []:
                if isinstance(role, dict) and role.get("speaker_id"):
                    roles[str(role["speaker_id"])] = str(role.get("name") or role.get("identified_name") or role.get("role") or "")[:60]
            from .dialogue_view import people_from_ledger

            turns = build(opened["segments"], lines, roles, people_from_ledger(ledger))
            if turns:
                opened["dialogue"] = turns
                opened["dialogue_note"] = NOTE
        except Exception:                                             # noqa: BLE001 - an advisory view must never fail opening a clip
            return

    def _add_phrasing(self, opened: dict[str, Any], clip_id: str) -> None:
        """Who talks to whom, in what order (see comma_check.conversation).  The speakers' own pauses are not put in front of the agent any more: the voice clones the speaker, so the Zulu is
        written for fluency and for being understood, not to imitate the speaker's breathing.  Evidence, never a gate."""
        from .comma_check import conversation

        try:
            talk = conversation(opened["segments"])
            if talk:
                opened["conversation"] = talk
        except Exception:                                             # noqa: BLE001 - an advisory view must never stop a clip from opening
            return

    def _speaker_pauses(self, clip_id: str) -> dict[str, int]:
        """segment id -> milliseconds of held pauses (0.3 s or more) the source speaker took inside it, from the delivery sheet: the silence a dub of that segment may leave without it being dead air."""
        try:
            sheet = self._delivery_sheet(clip_id) or {}
        except Exception:  # noqa: BLE001 - no sheet just means no allowance
            return {}
        return {str(sid): sum(int(p.get("ms") or 0) for p in (row.get("pauses") or [])) for sid, row in (sheet.get("segments") or {}).items()}

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

    def _memory_entries(self, voice: str | None) -> list[dict[str, Any]]:
        """The shared memory's enabled readings for ``voice`` that may be applied: not an ordinary Zulu word (eNkandla), which the Zulu voice reads
        natively and which a reading saved by an earlier run must not respell."""
        from .pronunciation_memory import entries_for, memory_path

        return [e for e in entries_for(memory_path(Path(self.store.root).parent), voice) if not self._ordinary_zulu(str(e.get("display_text") or ""))]

    def _sync_lab_context(self) -> None:
        """Give the lab the Zulu as written now and the readings already chosen, so a word is also heard in the sentence it is spoken in."""
        lab = self._lab()
        submission = Path(self._episode()["dir"]) / "submission.json"
        items = read_json(submission).get("items", []) if submission.is_file() else []
        lab.sentences = [str(i.get("zulu_text") or "") for i in items]
        self._lab_item_ids = [str(i.get("item_id") or "") for i in items]
        readings: dict[str, str] = {}
        lexicon = Path(self.store.clip_dir(self._episode()["clip_id"])) / "native_dub" / "pronunciation_names.json"
        if lexicon.is_file():
            readings.update({str(e["display_text"]): str((e.get("tts_text_by_voice") or {}).get(lab.voice) or e["tts_text"])         # this voice's reading, else the shared one
                             for e in read_json(lexicon).get("entries", []) if e.get("display_text") and e.get("tts_text")})
        for entry in self._memory_entries(lab.voice):
            if entry.get("display_text") and entry.get("tts_text") and not entry.get("disabled"):
                readings.setdefault(str(entry["display_text"]), str(entry["tts_text"]))
        lab.readings = readings

    def _known_readings(self) -> set[str]:
        """Texts that already have a reading (the clip's lexicon and the shared memory): the lab does not need to look at them."""
        from .pronunciation_lab import failing_readings

        self._sync_lab_context()
        known: set[str] = set()
        lexicon = Path(self.store.clip_dir(self._episode()["clip_id"])) / "native_dub" / "pronunciation_names.json"
        if lexicon.is_file():
            known |= {str(e["display_text"]).casefold() for e in read_json(lexicon).get("entries", [])}
        saved = self._memory_entries(self._lab().voice)
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
                from .pronunciation_ledger import problems_of

                self._ledger("shown", measured["word"], "measured as written, in the reminder", closeness=measured.get("closeness"), problems=problems_of(measured))
        return found

    def saved_readings_evidence(self) -> list[dict[str, Any]]:
        """How the readings the shared memory already holds measure now, in the sentences of this text, for the harness's reminder.  Only
        readings of English words that this text has; no tries are spent, nothing is scored or cut off, and one that cannot be measured is left out."""
        rows: list[dict[str, Any]] = []
        try:
            self._sync_lab_context()
            lab = self._lab()
            saved = [e for e in self._memory_entries(lab.voice) if e.get("kind") == "english_code_switch"]
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
                from .pronunciation_ledger import problems_of

                self._ledger("shown", word, "saved reading, in the reminder", closeness=measured.get("closeness"), problems=problems_of(measured))
            if len(rows) >= SAVED_READINGS_MEASURED:
                break
        # readings with something named wrong come first, then by closeness: a missing final s costs the score almost nothing but is heard at once
        return sorted(rows, key=lambda row: (0 if (row.get("differences") or row.get("ending") or row.get("lost_in_its_sentence")) else 1, row["closeness"]))

    def loanwords(self) -> list[dict[str, str]]:
        """English common nouns the Zulu text keeps ({"written", "english", "zulu"}), from a model's reading of the Zulu text.  ``zulu`` is the everyday
        Zulu word for it, or the form created for it in an earlier clip, or "" when Zulu has none (then one can be created).  Every such word is
        listed, whether or not a pronunciation was already chosen for it.  Empty when it cannot be asked."""
        submission = Path(self._episode()["dir"]) / "submission.json"
        if self.judge is None or not hasattr(self.judge, "foreign_words") or not submission.is_file():
            return []
        try:
            from .pronunciation_lab import _stem

            spoken = {_stem(str(w.get("text") or "")) for w in self._lab().transcript}          # only English the speaker really says in the source audio
            texts = [str(i.get("zulu_text") or "") for i in read_json(submission).get("items", [])]
            from .created_loanwords import path_for, zulu_for

            created = path_for(Path(self.store.root).parent)
            found = [{"written": w["written"], "english": w["english"],
                      "zulu": next((z for z in (zulu_for(created, w["english"]), w.get("zulu")) if z and self._attested(z)), "")}      # a form real text never writes is not offered
                     for w in self.judge.foreign_words(texts)
                     if w.get("kind") != "name" and len(_stem(w["english"])) >= 4 and _stem(w["english"]) in spoken]
            return list({(item["english"].casefold()): item for item in found}.values())          # one entry per English word, whatever prefix it carries
        except Exception:
            return []

    def _ledger(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Record in the ledger; a recording problem never touches the run."""
        try:
            getattr(self.ledger, method)(*args, **kwargs)
        except Exception:
            pass

    def note_pronunciation(self, word: str, decision: str, why: str) -> dict[str, Any]:
        """The agent's own reason for a word it checked and left or changed (decision: kept_as_is, changed or not_a_problem)."""
        return self.ledger.note(word, decision, why)

    def ledger_unexamined(self) -> list[dict[str, Any]]:
        return self.ledger.unexamined()

    def ledger_untried_far(self) -> list[dict[str, Any]]:
        return self.ledger.untried_far()

    def ledger_summary(self) -> dict[str, Any]:
        return self.ledger.summary()

    def write_pronunciation_ledger(self, directory: Path) -> Path:
        path = Path(directory) / "pronunciation_ledger.json"
        path.write_text(json.dumps(self.ledger.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        return path

    def _resources(self) -> Any:
        from . import attestation

        return attestation.load(Path(self.store.root).parent)

    def _ordinary_zulu(self, token: str) -> bool:
        if getattr(self, "_zulu_guard", None) is None:
            from . import attestation

            self._zulu_guard = attestation.zulu_guard(Path(self.store.root).parent)
        return self._zulu_guard(token)

    def _attested(self, form: str) -> bool:
        """Whether real text writes this Zulu form; True when the resources are not installed (nothing can be judged, so nothing is dropped)."""
        resources = self._resources()
        return True if resources is None else resources.attested(form)

    def _pairs_db(self):
        if getattr(self, "_human_pairs", None) is None:
            from .human_translations import PairsDB

            self._human_pairs = PairsDB(Path(self.store.root).parent / "resources" / "grammar" / "human_pairs.sqlite")
        return self._human_pairs

    def human_translations(self, phrase: str, limit: int = 5) -> dict[str, Any]:
        """How professional translators wrote an English word or phrase: the English-isiZulu pairs (Umsuka news, Autshumato government text) whose English contains it, the shortest
        first, with where each came from and how many pairs there are in all.  Free: no try, no model.  Evidence to weigh, never an answer picked for the agent."""
        return self._pairs_db().lookup(str(phrase), limit=max(1, min(int(limit or 5), 8)))

    def source_readings(self, segment_id: str = "", phrase: str = "") -> dict[str, Any]:
        """What the pipeline heard at one moment of the source (see source_readings.py): the recogniser's text and what pass 1 changed, its per-word confidence, YouTube's caption for the same
        moment and where the two differ, and the sentences either side.  Free: no model, no speech.  Evidence to weigh, never an answer picked for the agent."""
        import json as _json

        from . import source_readings, youtube_captions

        episode = self._episode()
        entry = self.store.load_corpus()["clips"][episode["clip_id"]]
        job_root = Path(self.store.root).parent / "jobs" / str(entry["job_id"])
        if getattr(self, "_raw_words", None) is None:
            try:
                self._raw_words = _json.loads((job_root / "analysis" / "transcript_en_raw.json").read_text(encoding="utf-8")).get("words") or []
            except (OSError, ValueError):
                self._raw_words = []
        files = youtube_captions.find(job_root)
        captions = youtube_captions.events(files[0]) if files else []
        track = files[0].name.split(".")[-2] if files else ""
        return source_readings.build(self.segments, words=self._raw_words, captions=captions, track=track, segment_id=str(segment_id or ""), phrase=str(phrase or ""))

    def say_it(self, text: str, speaker_id: str = "SPEAKER_00") -> dict[str, Any]:
        """The Zulu line as the voice will receive it (see say_it.py): the spoken form, each change the dictionary made, and what stayed as written.  Free: no try, no speech."""
        from .native_dub import _native_dub_pronunciation_dictionary, _NATIVE_DUB_PRONUNCIATION_DICTIONARY_BY_JOB
        from .say_it import say_it

        dictionary = _NATIVE_DUB_PRONUNCIATION_DICTIONARY_BY_JOB.get(str(self.pronunciation_root), None) if self.pronunciation_root else None
        dictionary = dictionary or _native_dub_pronunciation_dictionary()
        voice = None
        try:
            voice = str(pick_voice_info(self.voices, speaker_id, 0, 1, self.segments).get("selected_voice") or "") or None
        except Exception:                                             # noqa: BLE001 - the voice only selects a per-voice spelling; the default is fine
            voice = None
        return say_it(dictionary, str(text), voice)

    def figure_style(self, phrase: str) -> dict[str, Any]:
        """How professional translators wrote a figure in isiZulu (the prefix before the digits, the unit word, or the number in words), from the pairs whose English has the same
        shape: "31 percent", "20 years", "in 2006", "11 municipalities".  Free: no try, no model.  Evidence to weigh, never an answer picked for the agent."""
        from .zulu_figures import figure_style

        return figure_style(self._pairs_db().db, str(phrase))

    def attested_zulu(self, word: str, zulu_forms: list[str] | None = None) -> dict[str, Any]:
        """What real written isiZulu does with an English word, and how often forms the agent is thinking of have been written (see attestation).  No try
        is spent and no model is asked."""
        from . import attestation

        resources = attestation.load(Path(self.store.root).parent)
        if resources is None:
            return {"error": "the attestation resources are not installed on this machine; decide as you would without them and keep an English word as the speaker says it"}
        answer = attestation.lookup(resources, word, [str(f) for f in zulu_forms or []])
        spoken = Workbench._spoken_zulu(self)
        if spoken is not None:                                  # what people SAY, from human-transcribed speech, beside the formal written evidence above
            answer["spoken"] = spoken.lookup(word, [str(f) for f in zulu_forms or []])
        return answer

    def punctuation_examples(self, phrase: str, limit: int = 4) -> dict[str, Any]:
        """Real Zulu news sentences in which a word follows a comma and others in which it does not (see attestation.punctuation_examples).  Free, no model."""
        from . import attestation

        return attestation.punctuation_examples(Path(self.store.root).parent, phrase, limit)

    def _spoken_zulu(self):
        from . import spoken_zulu

        path = Path(self.store.root).parent / "resources" / "spoken"
        cache = Workbench._spoken_cache
        if str(path) not in cache:
            try:
                cache[str(path)] = spoken_zulu.SpokenZulu.from_directory(path)
            except Exception:
                cache[str(path)] = None                            # unreadable transcripts must not stop a lookup
        return cache[str(path)]

    def glued_class_prefix_words(self) -> list[dict[str, str]]:
        """English words that the Zulu text carries with a noun-class prefix glued straight on ("MaFighter") although the speaker says the English
        word as it is ("Fighters").  Found in the text itself: the candidate list for the lab leaves out names, the model reads "Fighters" as a form of
        address, and words that already have a saved reading, and MaFighter was all three.  A hyphenated particle (ne-, ye-), a locative e-, the u- / o- of
        a personal name (uMalema) and a word that has an everyday Zulu word (the loanword note covers it) are ordinary Zulu and are not listed."""
        import re

        from .pronunciation_lab import _stem

        try:
            submission = Path(self._episode()["dir"]) / "submission.json"
            texts = [str(i.get("zulu_text") or "") for i in read_json(submission).get("items", [])] if submission.is_file() else []
            spoken = {_stem(str(w.get("text") or "")): str(w.get("text")) for w in self._lab().transcript}
        except Exception:
            return []
        prefix = re.compile(r"(?<![A-Za-z])((?i:ma|ama|isi|izi|aba|abe|imi|izin|i))(?=[A-Z][a-z]{3,})([A-Za-z]+)")
        found: dict[str, dict[str, str]] = {}
        for text in texts:
            for match in prefix.finditer(text):
                stem = _stem(match.group(2))
                if len(stem) >= 4 and stem in spoken and match.group(0) not in found:
                    found[match.group(0)] = {"written": match.group(0), "speaker_says": re.sub(r"[^A-Za-z']", "", spoken[stem]), "_stem": stem}
        try:
            has_zulu = {_stem(str(item.get("english") or "")) for item in self.loanwords() if item.get("zulu")}
        except Exception:
            has_zulu = set()
        return [{k: v for k, v in item.items() if k != "_stem"} for item in found.values() if item["_stem"] not in has_zulu]

    def grammar_review(self, item_ids: list[str] | None = None) -> dict[str, Any]:
        """A second reader of the current Zulu: per unit, the phrases a model finds ungrammatical or too fused to say in one breath, each with a suggested rewrite.  It is an opinion, not a gate: an English-speaking
        model cannot judge all of isiZulu, so the writer decides; it costs no TTS measurement."""
        if self.judge is None or not hasattr(self.judge, "grammar"):
            return {"error": "no reviewer is configured for this run; read your own text for phrases that run together or do not parse, and use check_text"}
        submission = Path(self._episode()["dir"]) / "submission.json"
        if not submission.is_file():
            raise ValueError("submit a dub first: there is no Zulu text to review yet")
        items = read_json(submission).get("items", [])
        current = {str(s["segment_id"]): str(s.get("restored_text") or s.get("source_text") or "") for s in getattr(self, "segments", None) or []}
        wanted = {str(i) for i in item_ids} if item_ids else None
        units = [{"id": str(i.get("item_id")), "english": " ".join(current.get(str(x), "") for x in i.get("source_segment_ids") or []), "zulu": str(i.get("zulu_text") or "")}
                 for i in items if wanted is None or str(i.get("item_id")) in wanted]
        try:
            issues = self.judge.grammar(units)
        except Exception as exc:                                              # noqa: BLE001 - an outage of the reviewer never stops the run
            return {"error": f"the reviewer could not be reached ({type(exc).__name__}); read your own text instead"}
        return {"units": [{"item_id": u["id"], "issues": issues.get(u["id"], [])} for u in units if issues.get(u["id"])],
                "reviewed": [u["id"] for u in units],
                "note": "Phrases a model found ungrammatical (grammar) or too fused to say in one breath (fused), with a suggested fix. Apply a fix only where you agree it is better isiZulu and says the same; keep every "
                        "name, number and date as written; then check_text and re-measure the unit. A unit not listed had nothing to polish."}

    def list_foreign_words(self) -> dict[str, Any]:
        """English words that the current Zulu keeps and that the speaker says in the source audio: the ones worth checking."""
        self.foreign_words_listed = True
        result = self._foreign_candidates()
        for item in result.get("words", []):
            self._ledger("shown", item["word"], "listed by list_foreign_words")
        return result

    def _foreign_candidates(self, *, with_known: bool = False) -> dict[str, Any]:
        from .pronunciation_lab import GUARANTEED_TRIES_PER_WORD, candidate_words
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
        found = candidate_words(texts, self._lab().transcript, looks_isizulu=looks_isizulu, exclude=set() if with_known else self._known_readings(), named=named,
                                ordinary_zulu=self._ordinary_zulu)
        found = self._measured_weakest_first(found)
        return {"words": found, "note": "Loanwords first: an entry with zulu_word has an everyday Zulu word; write that word in the text instead of keeping the "
                                         "English one, unless it would change what the speaker said. Check with the lab only the English words you keep. "
                                         "Never glue a Zulu prefix onto an English word the speaker says as it is. Names and initialisms (is_name) are "
                                         "here too: the speaker's sounds are the reading to match. as_written is how close the voice is now, weakest first; "
                                         "a name already close (an isiZulu name, say) needs nothing.",
                "tries_per_word": f"every word can be tried {GUARANTEED_TRIES_PER_WORD} times; a 4th and 5th come from a shared pool, {self._lab().shared_extra_tries_left} left",
                "shared_extra_tries_left": self._lab().shared_extra_tries_left}

    def _measured_weakest_first(self, found: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Every word the lab may be asked about, measured as the Zulu writes it against the speaker's sounds in the source audio, weakest first.

        The speaker's audio is the baseline: ``as_written`` is how far the voice's reading is from it.  Until now only the six words the reminder showed
        were measured, in list order, so a name that sat late in the list ("Heidi", "Western Cape High Court") was never looked at, and names and
        initialisms were not listed at all.  One clip of the voice per word (no try is spent), at most WORDS_MEASURED of them; a word that cannot
        be measured goes last, never missing."""
        from .pronunciation_ledger import problems_of

        for entry in found[:WORDS_MEASURED]:
            try:
                measured = self._lab().as_written(entry["word"])
            except Exception:
                measured = None
            if measured and measured.get("reference_warning"):
                entry["as_written"] = {"closeness": None, "reference_unreliable": measured["reference_warning"]}          # a score against the wrong sounds says nothing
                self._ledger("shown", entry["word"], "measured as written, in the word list: the speaker's sounds do not fit the word")
            elif measured:
                entry["as_written"] = {"closeness": measured.get("closeness"), "problems": problems_of(measured),
                                       **({"reference": "a standard South African English voice, not the speaker (their audio was too fast to decode)"} if measured.get("reference_note") else {})}
                self._ledger("shown", entry["word"], "measured as written, in the word list", closeness=measured.get("closeness"), problems=problems_of(measured))
        return sorted(found, key=lambda e: (e.get("as_written", {}).get("closeness") is None, (e.get("as_written") or {}).get("closeness") or 0))

    def try_pronunciation(self, word: str, spellings: list[str]) -> dict[str, Any]:
        """Hear what the Zulu voice does with up to three spellings of ``word`` and how they differ from the speaker's sounds."""
        from .pronunciation_lab import PronunciationBudgetExhausted

        try:
            self._sync_lab_context()
            result = self._lab().try_spellings(word, spellings)
            self._ledger("tried", word, spellings)
            return result
        except PronunciationBudgetExhausted as exc:
            raise BudgetExhausted(str(exc)) from exc

    def suggest_spellings(self, word: str) -> dict[str, Any]:
        """Spellings of ``word`` worked out from the speaker's sounds and what this voice turns letters into (counts as one try)."""
        from .pronunciation_lab import PronunciationBudgetExhausted

        try:
            self._sync_lab_context()
            result = self._lab().suggest_spellings(word)
            self._ledger("tried", word, ["suggest_spellings"])
            return result
        except PronunciationBudgetExhausted as exc:
            raise BudgetExhausted(str(exc)) from exc

    def _loan_phones(self, word: str) -> list[str]:
        """The dictionary's sounds for the English part of ``word`` (CMUdict, then the guesser for a word it lacks): the same sounds the pronunciation rules start from.
        Empty when there are none.  No recogniser and no voice is asked."""
        from .english_sounds import EnglishSounds, g2p_guesser
        from .interpreter import Interpreter
        from .pronunciation_lab import without_prefix, written_prefix

        if getattr(self, "_loan_interpreter", None) is None:
            sounds = EnglishSounds.load(Path(self.store.root).parent / "resources" / "cmudict" / "cmudict.dict")
            if sounds is None:                                        # no dictionary on this machine: no sounds
                return []
            sounds.guesser = g2p_guesser()
            self._loan_interpreter = Interpreter(sounds, {}, None, {})
        stem = without_prefix(word.split("-")[-1], written_prefix(word))
        reading = self._loan_interpreter.main_reading(stem)
        return list(reading["phones"]) if reading else []

    def create_loanword(self, word: str) -> dict[str, Any]:
        """Zulu words made for an English word that has none, by the loan rules (``loan_adapt``: each English sound to the letters Zulu writes it with, open syllables,
        a vowel inserted where Zulu has no cluster and at the end) from the dictionary's sounds, the way the pronunciation rules work.  Nothing is listened to, spoken
        or scored: the forms come back with what each changes from the most typical, and the agent chooses."""
        from . import loan_adapt

        phones = self._loan_phones(word)
        if not phones:
            return {"word": word, "error": "the dictionary has no sounds for this word, so there is nothing to build a Zulu word from: keep the English word as the speaker says it"}
        rows = [{"zulu_word": c["stem"], "with_i": c["with_i"], "syllables": c["syllables"], "changes_from_the_most_typical": c["choices"]}
                for c in loan_adapt.adapt(phones, limit=5)]
        made = self.__dict__.setdefault("_loans", {})
        for row in rows:
            made.setdefault(word, set()).add(row["zulu_word"])
        return {"word": word, "dictionary_sounds": " ".join(phones), "zulu_words": rows,
                "how_these_were_made": "the dictionary's sounds rebuilt in Zulu syllables: English p/t/k/b written ph/th/kh/bh, vowels reduced to a e i o u, a vowel "
                                       "inserted where Zulu has no cluster (u after a labial or velar, i after the others) and at the end of the word; the first "
                                       "is the most typical, the others change one choice each (see changes_from_the_most_typical)",
                "next": "choose one, write it in the Zulu text where the English word stands, with the noun prefix the sentence needs (i-, isi-, ama-, u-), "
                        "then call use_loanword so later clips use the same word"}

    def use_loanword(self, word: str, zulu_word: str) -> dict[str, Any]:
        """Remember the Zulu word chosen for an English word, for every later clip.  The agent writes it in the Zulu text itself."""
        from .created_loanwords import path_for, remember

        episode = self._episode()
        made = self.__dict__.setdefault("_loans", {}).get(word, set())
        if zulu_word not in made:
            # the form with its noun prefix (iphurovideni, isayensi) is the word made (phurovideni, sayensi): the prefix is the sentence's
            stem = next((m for m in made if zulu_word.casefold().endswith(m.casefold()) and 0 < len(zulu_word) - len(m) <= 4), None)
            if stem is None:
                raise ValueError(f"create {zulu_word!r} for {word!r} with create_loanword before using it")
            zulu_word = stem
        entry = self.store.load_corpus()["clips"][episode["clip_id"]]
        if not self._attested(zulu_word):                      # the default for a word nobody writes is to keep the English as the speaker says it
            self._ledger("used", word, zulu_word)
            return {"ok": True, "remembered": False, "word": word, "zulu_word": zulu_word,
                    "note": f"{zulu_word!r} is not written anywhere in real isiZulu text (corpus_count 0), so it is not remembered for later clips. Keep {word!r} as the "
                            "speaker says it unless you find an attested form with attested_zulu."}
        remember(path_for(Path(self.store.root).parent), word, zulu_word, job_id=str(entry["job_id"]),
                 note=f"made by the loan rules from the dictionary's sounds {' '.join(self._loan_phones(word))}")
        self._ledger("used", word, zulu_word)
        return {"ok": True, "word": word, "zulu_word": zulu_word,
                "note": f"remembered for later clips. Write {zulu_word!r} in the Zulu text where {word!r} stands, with the noun prefix the sentence needs "
                        "(i-, isi-, ama-, u-), then re-measure and resubmit the units you change. A created word is Zulu text: the captions show it and the "
                        "voice reads it as Zulu, so it needs no respelling."}

    def _refuse_unsound_choice(self, lab: Any, word: str, spelling: str, tried: dict[str, Any]) -> None:
        """A closeness score can reward a reading that is plainly worse: prod-56 chose "Nyuz" for News (the voice then says only "n"), "E" for AI (the voice
        says nothing), "Rhamaphosa" for Ramaphosa (the voice says "hamaposa") and "Ey En Si" for ANC at 0.683 against 0.70 as written, and saved them for later clips.
        A reading is refused when the voice says less than 60% of the reference's sounds, or does not beat the word as it is written."""
        reference, said = list(lab.source(word).get("phones") or []), str(tried.get("voice_sounds") or "").split()
        if reference and len(said) < 0.6 * len(reference):
            raise ValueError(f"with {spelling!r} the voice says only [{' '.join(said)}] of the reference [{' '.join(reference)}]: it loses sounds, so it is not a reading of "
                             f"{word!r}. Choose a spelling that keeps them, or leave the word as it is written (note_pronunciation kept_as_is).")
        try:
            written = lab.as_written(word)
        except Exception:
            written = None
        if (spelling.casefold() != word.casefold() and written and written.get("closeness") is not None
                and float(tried["closeness"]) <= float(written["closeness"])):
            raise ValueError(f"{spelling!r} measures {tried['closeness']}, no closer than {word!r} as it is written ({written['closeness']}): leave the word as it is "
                             "written (note_pronunciation kept_as_is, citing the tries) or find a spelling that is closer.")

    def use_pronunciation(self, word: str, spelling: str) -> dict[str, Any]:
        """Make ``spelling`` the way ``word`` is spoken: in your own measurements now, in the render, and in the shared memory for later clips."""
        from . import dub_pronunciation
        from .pronunciation_memory import memory_path, remember

        lab, episode = self._lab(), self._episode()
        if self._ordinary_zulu(word):
            raise ValueError(f"{word!r} is ordinary isiZulu (real Zulu text writes it this way): the Zulu voice reads it natively, so it is not respelled to sound like "
                             "English. Leave it as it is written (note_pronunciation kept_as_is).")
        tried = lab.tried.get(word, {}).get(spelling)
        if tried is None:
            raise ValueError(f"try {spelling!r} for {word!r} with try_pronunciation before using it")
        self._refuse_unsound_choice(lab, word, spelling, tried)
        if tried.get("lost_in_its_sentence") and (word, spelling) not in self._lost_sound_warned:
            self._lost_sound_warned.add((word, spelling))       # said once: choosing it again after this is the agent's decision
            raise ValueError(
                f"in the sentence {word!r} is spoken in, the voice loses {' '.join(tried['lost_in_its_sentence'])} with {spelling!r} (it says "
                f"{tried.get('said_in_its_sentence')!r}; the speaker says {' '.join(lab.source(word)['phones'])!r}), so it will sound different in the video "
                "than in the lab. Try letters that keep the sound, then choose; if you still want this spelling, call use_pronunciation again.")
        # A reading respells the English word, not the Zulu grammar around it: the prefix glued on in the text ("we" of "weScience", the possessive
        # that makes "minister of science") stays on the spoken form even when the spelling tried was only the English part.
        from .pronunciation_lab import written_prefix

        prefix, tried_spelling = written_prefix(word), spelling
        if prefix and not spelling.casefold().startswith(prefix.casefold()):
            spelling = prefix + spelling
        if lab.source(word).get("reference_warning"):
            raise ValueError(
                f"{lab.source(word)['reference_warning']}. Keep {word!r} as the Zulu writes it (note_pronunciation kept_as_is, citing this); a reading "
                "matched to those sounds would not make it sound right and would be saved for later clips.")
        from .pronunciation_memory import beaten_by_saved, saved as saved_reading

        held = saved_reading(memory_path(Path(self.store.root).parent), word, lab.voice)
        if beaten_by_saved(held, spelling, float(tried["closeness"])):
            raise ValueError(
                f"the shared memory already holds {held['tts_text']!r} for {word!r}, which measured closer to the speaker (closeness {held['closeness']}; "
                f"{tried_spelling!r} measures {tried['closeness']}), and the render uses saved readings. Measure {held['tts_text']!r} here with "
                f"try_pronunciation and choose between them, or find a spelling that beats it.")
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
                 kind="english_code_switch", note=f"chosen by the agent with the lab; closeness {tried['closeness']}; voice sounds {tried['voice_sounds']}",
                 closeness=float(tried["closeness"]))
        choice = {"word": word, "spelling": spelling, "closeness": tried["closeness"], "speaker_sounds": " ".join(lab.source(word)["phones"]),
                  "voice_sounds": tried["voice_sounds"]}
        self.pronunciation_choices = [c for c in self.pronunciation_choices if c["word"] != word] + [choice]
        self._ledger("used", word, spelling, tried["closeness"])
        kept = f" The Zulu prefix {prefix!r} stays on it: the spoken form is {spelling!r}." if spelling != tried_spelling else ""
        return {"ok": True, **choice, "note": "your measurements and the render now use this reading wherever the word is written." + kept + " Keep the WORD in the text: never type the spelling into it (the voice respells the word when it speaks it, and the text is what the captions show). Re-measure any unit that contains the word."}

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
        result.update(length_gap(text, result))
        result["tts_budget_remaining"] = remaining
        self._learn_duration(text, pick_voice_info(self.voices, speaker_id, start_ms, end_ms, self.segments), result)
        self._show_omnivoice_estimate(text, speaker_id, start_ms, end_ms, result)
        return result

    # -- the duration model: a free estimate of how long the voice takes, learned from every measurement -----------------------------------------------------------
    ESTIMATE_NOTE = ("an estimate from the text alone (for the OmniVoice voice: the text's weight at the pace of the unit's own reference clip; otherwise per voice, learned from every duration "
                     "measured so far): it is not a measurement and it cannot confirm a fit (the bar is tighter than its error). Use it to leave out the candidates that are far from their window "
                     "and to send the rest to measure_tts_batch.")
    ESTIMATE_MAX = 40
    RUN_CORRECTION_MIN_SAMPLES = 3         # measurements of one voice in this run before its estimates are corrected by how this run's measurements compare

    def _duration_state(self) -> dict[str, Any]:
        if "_duration" not in self.__dict__:
            from . import duration_model

            work = Path(self.store.root).parent
            try:
                model = duration_model.load(work)
            except Exception:  # noqa: BLE001 - a missing or unreadable model means no estimates, never a failed run
                model = {}
            self.__dict__["_duration"] = {"work": work, "model": model, "ratios": {}, "seen": 0}
        return self.__dict__["_duration"]

    def _voice_of(self, voice_info: Any) -> tuple[str, float]:
        from . import duration_model

        voice = str((voice_info or {}).get("selected_voice") or getattr(self.tts, "default_voice", "") or "")
        return voice, duration_model.rate_of(voice_info)

    def _estimate_ms(self, voice_info: Any, text: str) -> dict[str, Any] | None:
        """The model's estimate for a text, corrected by this run's own measurements of the voice once there are RUN_CORRECTION_MIN_SAMPLES of them."""
        from . import duration_model

        state = self._duration_state()
        voice, rate = self._voice_of(voice_info)
        if not state["model"] or not voice:
            return None
        found = duration_model.predict(state["model"], voice=voice, text=text, rate_pct=rate)
        if not found:
            return None
        ratios = sorted(state["ratios"].get(voice, []))
        if len(ratios) >= self.RUN_CORRECTION_MIN_SAMPLES:
            factor = min(1.25, max(0.8, ratios[len(ratios) // 2]))
            found["run_correction"] = round(factor, 3)
            found["predicted_ms"] = int(round(found["predicted_ms"] * factor))
        return found

    OMNIVOICE_BAND_PCT = 10.0          # the PRIOR for the estimate's error band, used until OMNIVOICE_BAND_MIN_PAIRS real pairs exist; after that the band is learned (see _omnivoice_band_pct)
    OMNIVOICE_BAND_MIN_PAIRS = 20
    OMNIVOICE_BAND_WINDOW = 300        # the most recent pairs the band is read from, so it follows the estimator as the reference clips and the model change
    OMNIVOICE_BAND_FLOOR_PCT = 3.0
    OMNIVOICE_PAIRS = "omnivoice_estimates.jsonl"

    def _omnivoice_guide(self, segments: list[dict[str, Any]]) -> dict[str, int]:
        """segment_id -> the Zulu length (characters, to the nearest 5) that fills the segment's window at the pace of its own reference clip; empty when the voice is not OmniVoice."""
        try:
            backend = self.tts
        except Exception:  # noqa: BLE001 - a voice that cannot be built yet (no credentials) just means no OmniVoice guide: open_clip must not fail on it
            return {}
        if not hasattr(backend, "reference_for"):
            return {}
        from . import omnivoice_duration as od

        guide: dict[str, int] = {}
        for s in segments:
            try:
                reference = backend.reference_for({"speaker_id": s.get("speaker_id"), "start_ms": s["start_ms"], "source_end_ms": s["end_ms"]}, self.pronunciation_root)
                seconds = (int(s["end_ms"]) - int(s["start_ms"])) / 1000
                guide[str(s["segment_id"])] = max(5, int(round(od.chars_to_fill(seconds, reference.text, reference.seconds, backend.speed) / 5)) * 5)
            except Exception:  # noqa: BLE001 - a segment with no usable reference just gets no guide
                continue
        return guide

    def _omnivoice_pairs_path(self) -> Path:
        return Path(self.store.root).parent / self.OMNIVOICE_PAIRS

    def _record_estimate_pair(self, estimated_ms: int, measured_ms: int) -> None:
        """One real measurement beside the free estimate that was made for the same text: the data the error band is learned from.  Best effort, never a failed measurement."""
        try:
            path = self._omnivoice_pairs_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"estimated_ms": int(estimated_ms), "measured_ms": int(measured_ms)}) + "\n")
        except Exception:  # noqa: BLE001
            return

    def _omnivoice_band_pct(self) -> float:
        """The band nine in ten free estimates fall inside, learned from the recorded pairs (the 90th percentile of the absolute error over the latest OMNIVOICE_BAND_WINDOW);
        the prior until there are OMNIVOICE_BAND_MIN_PAIRS pairs."""
        try:
            lines = self._omnivoice_pairs_path().read_text(encoding="utf-8").splitlines()[-self.OMNIVOICE_BAND_WINDOW:]
            errors = sorted(abs(p["estimated_ms"] - p["measured_ms"]) / p["measured_ms"] * 100.0 for p in (json.loads(x) for x in lines) if p.get("measured_ms"))
        except Exception:  # noqa: BLE001
            return self.OMNIVOICE_BAND_PCT
        if len(errors) < self.OMNIVOICE_BAND_MIN_PAIRS:
            return self.OMNIVOICE_BAND_PCT
        return round(max(self.OMNIVOICE_BAND_FLOOR_PCT, errors[min(len(errors) - 1, int(0.9 * len(errors)))]), 1)

    def _omnivoice_ms(self, text: str, speaker_id: str, start_ms: int, end_ms: int) -> int | None:
        """OmniVoice's free estimate of this text's natural length, from the pace of its reference clip (see omnivoice_duration); None when the voice is not OmniVoice."""
        estimator = getattr(self.tts, "estimate_seconds", None)
        if estimator is None:
            return None
        from .dub_fit import _group

        group = _group(item_id="estimate", speaker_id=speaker_id, start_ms=start_ms, end_ms=end_ms, source_text="")
        group["parts"] = [{"type": "text", "segment_id": "estimate_seg", "text": text}]
        info = pick_voice_info(self.voices, speaker_id, start_ms, end_ms, self.segments) or {"selected_voice": getattr(self.tts, "default_voice", "")}
        try:
            return int(round(estimator(group, info, self.pronunciation_root) * 1000))
        except Exception:  # noqa: BLE001 - no reference clip or an unreadable corpus means no estimate, never a failed measurement
            return None

    def _show_omnivoice_estimate(self, text: str, speaker_id: str, start_ms: int, end_ms: int, result: dict[str, Any]) -> None:
        """Beside a real OmniVoice measurement, what the free estimate said, so the agent sees how far to trust it."""
        estimated = self._omnivoice_ms(text, speaker_id, start_ms, end_ms)
        measured = int(result.get("measured_ms") or 0)
        if estimated and measured > 0:
            result["estimated_ms"] = estimated
            result["estimate_off_pct"] = round((estimated - measured) / measured * 100.0, 1)
            self._record_estimate_pair(estimated, measured)

    def _learn_duration(self, text: str, voice_info: Any, result: dict[str, Any]) -> None:
        """A real measurement: show the agent how its estimate did, keep it in the ledger for the next fit, and let this run's estimates lean on it."""
        from . import duration_model

        if hasattr(self.tts, "estimate_seconds"):          # OmniVoice: its estimate is a formula of the reference's pace, not a model fitted to Azure measurements
            return
        try:
            measured = int(result.get("measured_ms") or 0)
            voice, rate = self._voice_of(voice_info)
            if measured <= 0 or not voice:
                return
            state = self._duration_state()
            raw = duration_model.predict(state["model"], voice=voice, text=text, rate_pct=rate) if state["model"] else None
            if raw and raw["predicted_ms"] > 0:
                state["ratios"].setdefault(voice, []).append(measured / raw["predicted_ms"])
                result["estimated_ms"] = raw["predicted_ms"]
                result["estimate_off_pct"] = round((raw["predicted_ms"] - measured) / measured * 100.0, 1)
            duration_model.record(state["work"], voice=voice, text=text, measured_ms=measured, rate_pct=rate, source="measure")
        except Exception:  # noqa: BLE001 - learning must never fail a measurement
            return

    def estimate_fit(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        """Free (no TTS, no budget): for up to 40 candidate texts, the voice's estimated duration against the source window, with the error band of the estimate."""
        if not 1 <= len(requests) <= self.ESTIMATE_MAX:
            raise ValueError(f"send between 1 and {self.ESTIMATE_MAX} requests (you sent {len(requests)})")
        results = []
        for request in requests:
            start, end = int(request["start_ms"]), int(request["end_ms"])
            if end <= start:
                raise ValueError("end_ms must be greater than start_ms")
            omni = self._omnivoice_ms(str(request.get("text") or ""), str(request.get("speaker_id") or "SPEAKER_00"), start, end)
            if omni is not None:                          # OmniVoice: the formula from the reference clip's pace, with the band it showed on a whole job
                window = end - start
                gap = (omni - window) / window * 100.0
                band = self._omnivoice_band_pct()
                results.append({"estimated_ms": omni, "window_ms": window, "estimated_gap_pct": round(gap, 1), "band_pct": band,
                                "reading": ("inside the estimate's error band: worth a measurement" if abs(gap) <= band
                                            else f"probably {'over' if gap > 0 else 'short of'} the window by more than the band")})
                continue
            info = pick_voice_info(self.voices, request.get("speaker_id"), start, end, self.segments)
            found = self._estimate_ms(info, str(request.get("text") or ""))
            if not found:
                results.append({"available": False, "note": "no duration model is fitted yet for this voice: measure it"})
                continue
            window = end - start
            gap = (found["predicted_ms"] - window) / window * 100.0
            band = float(found["band_pct"])
            reading = ("inside the estimate's error band: worth a measurement" if abs(gap) <= band
                       else f"probably {'over' if gap > 0 else 'short of'} the window by more than the band")
            results.append({"estimated_ms": found["predicted_ms"], "window_ms": window, "estimated_gap_pct": round(gap, 1), "band_pct": band,
                            "reading": reading, **({"run_correction": found["run_correction"]} if "run_correction" in found else {})})
        return {"results": results, "typical_error_pct": next((r.get("band_pct") for r in results if r.get("band_pct")), None), "note": self.ESTIMATE_NOTE}

    TIMELINE_MAX = 60
    TIMELINE_MIN_GAP_MS = 300          # the least silence between two units that run into each other (as the renderer keeps)
    TIMELINE_LATE_MS = 400             # a sentence starting this much after its source onset is late enough to matter

    def _text_ms(self, text: str, speaker_id: str, start_ms: int, end_ms: int) -> tuple[int | None, float | None]:
        """The free estimate of a text's length and its error band: OmniVoice's formula when that is the voice, else the Azure duration model; (None, None) when there is none."""
        omni = self._omnivoice_ms(text, speaker_id, start_ms, end_ms)
        if omni is not None:
            return omni, self._omnivoice_band_pct()
        found = self._estimate_ms(pick_voice_info(self.voices, speaker_id, start_ms, end_ms, self.segments), text)
        return (found["predicted_ms"], float(found["band_pct"])) if found else (None, None)

    def timeline_fit(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Free: a draft laid out on the timeline the way a person would watch it, sentence by sentence.  Each item is one source segment and the Zulu you would say for it.
        A sentence starts where its source starts, or after the previous one ends (plus the least gap) when that one ran long; the lateness it carries is what every later sentence
        inherits.  Returns, per sentence, its estimated start and end against its source's, how late it is, what it would overrun into, and, where lateness begins, which earlier
        sentences of the run are over their windows and could give the time back."""
        if not 1 <= len(items) <= self.TIMELINE_MAX:
            raise ValueError(f"send between 1 and {self.TIMELINE_MAX} items (you sent {len(items)})")
        ordered = sorted(self.segments, key=lambda s: (int(s["start_ms"]), str(s["segment_id"])))
        index = {str(s["segment_id"]): i for i, s in enumerate(ordered)}
        windows: list[dict[str, Any]] = []
        for item in items:                                        # an item is one source segment (segment_id) or one of the agent's own units (start_ms and end_ms, which may span several segments)
            if item.get("segment_id") is not None and (item.get("start_ms") is None or item.get("end_ms") is None):
                sid = str(item["segment_id"])
                if sid not in index:
                    raise ValueError(f"unknown segment_id {sid!r}: use the ids from open_clip")
                segment = ordered[index[sid]]
                windows.append({"label": sid, "start": int(segment["start_ms"]), "end": int(segment["end_ms"]), "speaker": str(segment.get("speaker_id") or "SPEAKER_00"), "text": item.get("text")})
            elif item.get("start_ms") is not None and item.get("end_ms") is not None:
                start, end = int(item["start_ms"]), int(item["end_ms"])
                if end <= start:
                    raise ValueError("end_ms must be greater than start_ms")
                covering = next((s for s in ordered if int(s["start_ms"]) <= start < int(s["end_ms"])), None)
                windows.append({"label": str(item.get("item_id") or f"{start}-{end}"), "start": start, "end": end,
                                "speaker": str(item.get("speaker_id") or (covering or {}).get("speaker_id") or "SPEAKER_00"), "text": item.get("text")})
            else:
                raise ValueError("each item needs a segment_id, or a start_ms and an end_ms")
        rows: list[dict[str, Any]] = []
        previous_end: int | None = None
        for window in sorted(windows, key=lambda w: (w["start"], w["end"])):
            sid, start, end, speaker = window["label"], window["start"], window["end"], window["speaker"]
            estimated, band = self._text_ms(str(window.get("text") or ""), speaker, start, end)
            if estimated is None:
                rows.append({"segment_id": sid, "available": False, "note": "no estimate for this voice yet: measure it"})
                previous_end = None
                continue
            placed = start if previous_end is None else max(start, previous_end + self.TIMELINE_MIN_GAP_MS)
            placed_end = placed + estimated
            following = next((s for s in ordered if int(s["start_ms"]) >= end - 50 and int(s["start_ms"]) > start), None)          # the first speech that starts after this window
            next_start = int(following["start_ms"]) if following else None
            other = bool(following and str(following.get("speaker_id") or "") != str(speaker))
            rows.append({
                "segment_id": sid, "speaker_id": speaker, "source_start_ms": start, "source_end_ms": end, "window_ms": end - start, "estimated_ms": estimated,
                "starts_ms": placed, "ends_ms": placed_end, "late_by_ms": placed - start, "over_window_by_ms": placed_end - end,
                "free_gap_after_ms": (next_start - end) if next_start is not None else None,
                "runs_into_next_speaker_ms": max(0, placed_end - next_start) if (other and next_start is not None) else 0,
                "band_pct": band})
            previous_end = placed_end
        usable = [r for r in rows if r.get("available", True)]
        late = [r for r in usable if r["late_by_ms"] > self.TIMELINE_LATE_MS]
        hints: list[dict[str, Any]] = []
        for position, row in enumerate(usable):
            if row["late_by_ms"] <= self.TIMELINE_LATE_MS or (position and usable[position - 1]["late_by_ms"] > self.TIMELINE_LATE_MS):
                continue                                              # only where lateness BEGINS: the rows after it only inherit it
            chain_start = position
            while chain_start > 0 and usable[chain_start - 1]["ends_ms"] + self.TIMELINE_MIN_GAP_MS >= usable[chain_start]["source_start_ms"] and usable[chain_start - 1]["over_window_by_ms"] > 0:
                chain_start -= 1
            givers = [r for r in usable[chain_start:position + 1] if r["over_window_by_ms"] > 0]
            hints.append({"late_from": row["segment_id"], "late_by_ms": row["late_by_ms"],
                          "shorten_these": [{"segment_id": r["segment_id"], "over_window_by_ms": r["over_window_by_ms"]} for r in sorted(givers, key=lambda r: -r["over_window_by_ms"])[:4]],
                          "reading": "the time is lost in the sentences listed (they run past their own windows): shortening one of them brings this sentence, and every one after it, back"})
        return {"sentences": rows, "late_sentences": [r["segment_id"] for r in late], "lateness_at_end_ms": usable[-1]["late_by_ms"] if usable else 0,
                "runs_into_another_speaker": [r["segment_id"] for r in usable if r["runs_into_next_speaker_ms"] > 0], "where_to_look": hints,
                "note": "an estimate, not a measurement: lateness here is how far each sentence would start after the speaker's, so the earlier a sentence is over its window the more it costs the ones after it."}

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
        for request, result in zip(requests, results):
            result.update(length_gap(request["text"], result))
            self._learn_duration(request["text"], pick_voice_info(self.voices, request.get("speaker_id"), request["start_ms"], request["end_ms"], self.segments), result)
            self._show_omnivoice_estimate(request["text"], str(request.get("speaker_id") or "SPEAKER_00"), request["start_ms"], request["end_ms"], result)
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

    def _source_explains(self):
        """``explains(token, source_text)`` for the fabricated-entity gate: does real parallel text write this Zulu word where the English has a word of the source (attestation.translates)?
        None when the resources are not installed (the gate then behaves as it always did)."""
        if not hasattr(self, "_explains_hook"):
            from . import attestation

            try:
                resources = attestation.load(Path(self.store.root).parent)
            except Exception:                                             # noqa: BLE001 - an optional resource must never stop a check
                resources = None
            self._explains_hook = (lambda token, source: bool(resources.translates(token, source))) if resources is not None else None
        return self._explains_hook

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
        from .delivery import strip_pauses

        spoken = strip_pauses(zulu_text)          # the scorer reads the text without its ~pauseN markers; a capital after one is a sentence start, not a name
        hook = self._source_explains()
        restated = find_restated_references(spoken)
        fabricated = find_fabricated_entities(spoken, known, (lambda token: hook(token, source_text)) if hook else None)
        problems: list[dict[str, Any]] = []
        if missing:
            problems.append({"gate": "required_literals", "evidence": [describe_missing_literal(m, zulu_text) for m in missing]})
        if restated:
            problems.append({"gate": "restated_reference", "evidence": [f"{h['first']} / {h['second']}" for h in restated]})
        if fabricated:
            problems.append({"gate": "fabricated_entity", "evidence": fabricated})
        from .english_prefix import find_prefixed_english

        prefixed = find_prefixed_english(zulu_text, source_text)
        if prefixed:
            problems.append({"gate": "english_with_class_prefix", "evidence": prefixed})
        from .english_prefix import find_soft_prefixed_english

        soft_prefixed = find_soft_prefixed_english(zulu_text, source_text)
        if soft_prefixed:
            problems.append({"flag": "english_word_with_noun_prefix", "evidence": soft_prefixed})
        from .english_prefix import find_unattested_prefixed

        unattested = find_unattested_prefixed(zulu_text, source_text, self._resources())
        if unattested:
            problems.append({"flag": "prefix_forms_to_compare_with_the_corpus", "evidence": unattested})
        doubled = find_unsupported_doubling(zulu_text, source_text)
        if doubled:
            problems.append({"flag": "unsupported_doubling", "evidence": doubled})
        from .zulu_word_notes import weekday_notes

        weekdays = weekday_notes(zulu_text)
        if weekdays:
            problems.append({"flag": "english_weekday", "evidence": weekdays})
        if english_leakage(zulu_text) >= 0.2:
            problems.append({"flag": "english_leakage"})
        from .listening_load import figures_run_together, long_runs

        together = figures_run_together(zulu_text)
        if together:
            problems.append({"flag": "figures_run_together", "evidence": together})          # a listener needs a breath to take a figure in
        runs = long_runs(zulu_text)
        if runs:
            problems.append({"flag": "long_run_without_breath", "evidence": runs})
        from .voice_limits import find as voice_limit_lines

        limits = voice_limit_lines(zulu_text)
        if limits:
            problems.append({"flag": "word_the_voice_stumbles_on", "evidence": limits, "note": "a listener heard this word stutter; a different word is better than a respelling"})
        return {"ok": not any("gate" in p for p in problems), "problems": problems, "gates_checked": list(GATE_NAMES)}

    def recall_lessons(self, limit: int = 25) -> dict[str, Any]:
        lessons: list[dict[str, Any]] = []
        for clip_id in self.store.clip_ids():
            for label in self.store.load_labels(clip_id):
                if label.get("verdict") == "bad":
                    lessons.append({k: label.get(k) for k in (
                        "clip_id", "tags", "english", "zulu", "bad_substring", "corrected_zulu", "note")})
        lessons.sort(key=lambda item: item.get("clip_id") or "")
        from . import ear_lessons

        return {"count": len(lessons), "lessons": lessons[-int(limit):], "ear_lessons": ear_lessons.for_agent(Path(self.store.root).parent)}

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

    def _judge_meaning(self, card: dict[str, Any], directory: Path, items: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
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
                "claims": self._claims_to_check(verdicts, {v["id"] for v in loss}, units, items or []),
                "unjudged": sum(1 for v in verdicts if v["meaning_preserved"] is None)}

    def _claims_to_check(self, verdicts: list[dict[str, Any]], already_shown_as_loss: set[str], units: list[dict[str, str]],
                         items: list[dict[str, Any]] = ()) -> list[dict[str, Any]]:
        """Units whose Zulu an independent reader thinks says something the English does not (an addition or a distortion), apart from
        the ones that lose meaning outright.  Seven runs of one clip carried 7 to 12 additions each: the agent fills a window with
        claims the speaker never made, and the submit result showed it only the units scoring 0-1.  Information, not a bar: the judge
        is noisy, so the agent weighs each against the English.  Each unit is shown once per text (an edit shows it again), and at
        most CLAIMS_PER_SUBMIT at a time, the lowest scores first, so this can never become an open-ended repair loop.

        Each claim carries the unit's characters and what its window takes (fits_about_chars): taking an addition out shortens the unit, and a
        unit left short is dead air, which the timing ratchet rejects when the unit had fitted (4 to 8 of the 10 to 19 submissions of two of
        three runs were rejected that way, against one per run before the flag)."""
        spoken = {u["id"]: u["dub"] for u in units}
        guide = length_guide(self.segments, self.voices)
        by_item = {str(i["item_id"]): i for i in items}
        found = []
        for v in verdicts:
            issues = [i for i in v.get("issues", []) if i.get("kind") in ("addition", "distortion")]
            key = (str(v["id"]), spoken.get(v["id"], ""))
            if not issues or v["id"] in already_shown_as_loss or key in self.claims_shown:
                continue
            score = v["meaning_preserved"] if v["meaning_preserved"] is not None else 3
            window = sum(guide.get(str(sid), 0) for sid in (by_item.get(str(v["id"])) or {}).get("source_segment_ids", ()))
            found.append((score, key, {"item_id": v["id"], "kinds": sorted({i["kind"] for i in issues}),
                                       "issues": [i["detail"] for i in issues][:2],
                                       "reads_back_as": (v.get("back_translation") or "")[:300],
                                       "chars_now": spoken_chars(spoken.get(v["id"], "")),
                                       "zulu_text": (lambda t: t if len(t) <= UNIT_TEXT_CHARS else t[:UNIT_TEXT_CHARS] + "…")(spoken.get(v["id"], "")),
                                       **({"fits_about_chars": window} if window else {})}))
        found.sort(key=lambda f: f[0])
        shown = found[:CLAIMS_PER_SUBMIT]
        self.claims_shown.update(key for _, key, _ in shown)
        return [claim for _, _, claim in shown]

    def submit_dub(
        self, items: list[dict[str, Any]], approach_notes: str = "", partial: bool = False,
        remove: list[str] | None = None, decisions: list[dict[str, Any]] | None = None,
        choice_points: list[dict[str, Any]] | None = None, unit_strategies: list[dict[str, Any]] | None = None, missing_evidence: list[dict[str, Any]] | None = None,
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
            voices=self.voices, pronunciation_root=self.pronunciation_root, source_explains=self._source_explains(),
            speaker_pauses=self._speaker_pauses(episode["clip_id"]),
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
        meaning = self._judge_meaning(card, directory, items)
        write_submission(directory / "submission.json", submission)
        episode["submissions"] += 1
        self.locked_segments = {sid for sid, entry in status.items() if entry["ok"]}
        atomic_write_json(directory / "scorecard.json", card)
        self.last_items = [dict(i) for i in items]
        self._keep_decisions(decisions, directory)
        self._log_thinking(choice_points, unit_strategies, items, directory, int(episode["submissions"]), missing_evidence)
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
            "flagged_units": self._with_texts([self._flagged_entry(g) for g in card["groups"] if g["flags"] or any(v["failed"] for v in g["gates"].values())], items),
            "listening_load": self._listening_load(items),
            "spoken_form_gaps": self._spoken_form_gaps(items),
            "sentence_timing": self._sentence_timing(items, card),
            "voice_limit_words": self._voice_limit_words(items),
            "figures_as_spoken": self._figures_as_spoken(items),
            "decisions_so_far": self._decisions_view(items),
            "settled_units": settled_units(card),
            "units_overview": units_overview(card),
            "naturalness": self._naturalness(items),
            "speaker_rates_chars_per_s": speaker_rates(card),
        }
        return self.last_submit

    def _naturalness(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """How natural the units are against real Zulu (see naturalness and cadence): word pairs that never meet in the corpora although two common words would, and units chopped into sentences
        shorter than almost any real speaker's.  Evidence on every submit, free, never a gate; the pair check needs the index built by scripts/naturalness_eval.py --final (the cadence check does not)."""
        from .cadence import cadence_line
        from .naturalness import NgramIndex, check, directory, register_check

        try:
            if not hasattr(self, "_ngram_index"):
                self._ngram_index = NgramIndex.load(directory(Path(self.store.root).parent))
            found = []
            for item in items:
                entry: dict[str, Any] = {}
                if self._ngram_index is not None:
                    pairs = check(self._ngram_index, str(item.get("zulu_text") or ""), expected_min=PAIR_EXPECTED_MIN)["unseen_pairs"][:2]
                    if pairs:
                        entry["pairs_that_never_meet_in_real_zulu"] = pairs
                    register = register_check(self._ngram_index, str(item.get("zulu_text") or ""))
                    if register:
                        entry["reads_like_translated_government_zulu"] = (
                            f"{register['above']} of {register['sentences']} sentences score like translated government text and not like native speech or news (only about 5% of native sentences score this high; "
                            + (f"words pulling it: {', '.join(register['pulling'])}, which may be right for this subject). " if register["pulling"] else "). ")
                            + "Say it the way a speaker would, in native word order and wording, unless the wording is exactly what the English needs.")
                line = cadence_line(item)
                if line:
                    entry["cadence"] = line
                if entry:
                    found.append({"item_id": str(item.get("item_id")), **entry})
            return found[:UNIT_TEXTS_SHOWN]
        except Exception:                                             # noqa: BLE001 - an advisory view must never fail a submission
            return []

    def _keep_decisions(self, decisions: list[dict[str, Any]] | None, directory: Path) -> None:
        """The terms the agent says it settled (english, zulu, note), merged by the English; saved beside the submission so a resumed or repair run starts with them."""
        for entry in decisions or []:
            if not isinstance(entry, dict):
                continue
            english, zulu = " ".join(str(entry.get("english") or "").split())[:60], " ".join(str(entry.get("zulu") or "").split())[:80]
            if english and zulu and (english.casefold() in self.decisions or len(self.decisions) < MAX_DECISIONS):
                self.decisions[english.casefold()] = {"english": english, "zulu": zulu, "note": " ".join(str(entry.get("note") or "").split())[:120]}
        try:
            atomic_write_json(directory / "decisions.json", {"decisions": list(self.decisions.values())})
        except OSError:
            pass

    @staticmethod
    def _stem_analyses(terms: list[str], items: list[dict[str, Any]]) -> dict[str, list[str]]:
        """ZulMorph analyses (zulu_stems) for the words of the terms and for the words of the units that could hold one of their stems.  The letters only choose which words are worth
        analysing (a stem is part of every form that holds it); the morphology decides whether they are the word."""
        from . import zulu_stems

        wanted = {w for term in terms for w in zulu_stems.words_of(term)}
        analyses = dict(zulu_stems.analyse(sorted(wanted)))
        known = set(wanted)
        for word in wanted:
            known |= zulu_stems.extract_stems(analyses.get(word) or ())
        candidates = {w for item in items for w in zulu_stems.words_of(str(item.get("zulu_text") or "")) if any(stem in w for stem in known)}
        analyses.update(zulu_stems.analyse(sorted(candidates)))
        return analyses

    def _decisions_view(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Each settled term with the units that still use it.  "Still uses it" is decided by stem (zulu_stems: ZulMorph reads isu, amasu and ngamasu as one word), not by the letters; when
        the analyser cannot be reached the comparison is of the plain forms and the message says so, so the agent is not told a term drifted when only its prefix changed."""
        from . import zulu_stems

        if not self.decisions:
            return []
        analyses = self._stem_analyses([entry["zulu"] for entry in self.decisions.values()], items)
        view = []
        for entry in self.decisions.values():
            units = [str(i["item_id"]) for i in items if zulu_stems.holds(entry["zulu"], str(i.get("zulu_text") or ""), analyses)]
            gone = ("no unit uses a form of this word" if analyses else "not found as written in any unit (the stem analyser is not available here, so a prefixed form would be missed)")
            view.append({"english": entry["english"], "zulu": entry["zulu"], **({"note": entry["note"]} if entry.get("note") else {}), "in_units": units[:12] if units else gone})
        return view[:MAX_DECISIONS]

    def _log_thinking(self, choice_points: list[dict[str, Any]] | None, unit_strategies: list[dict[str, Any]] | None, items: list[dict[str, Any]], directory: Path, submission: int,
                      missing_evidence: list[dict[str, Any]] | None = None) -> None:
        """Training mode: the agent's choice points and unit strategies, one JSON line each in thinking.jsonl beside the submission.  A choice point is checked against the text it was sent
        with (``chosen_in_text``, by stem), so a record that does not match what was written shows.  Live mode sends none and this does nothing."""
        if not choice_points and not unit_strategies and not missing_evidence:
            return
        from . import zulu_stems

        def clip(value: Any, limit: int = 400) -> str:
            return " ".join(str(value or "").split())[:limit]

        texts = {str(i["item_id"]): str(i.get("zulu_text") or "") for i in items}
        points = [c for c in (choice_points or []) if isinstance(c, dict)][:40]
        try:
            analyses = self._stem_analyses([clip(c.get("zulu"), 120) for c in points], items)
        except Exception:                                             # noqa: BLE001 - a record must never fail a submission
            analyses = {}
        lines = []
        for c in points:
            unit, zulu = clip(c.get("unit"), 20), clip(c.get("zulu"), 120)
            lines.append({"submission": submission, "kind": "choice", "unit": unit, "english": clip(c.get("english")), "zulu": zulu,
                          "alternatives": [clip(a, 160) for a in (c.get("alternatives") or [])[:6]] if isinstance(c.get("alternatives"), list) else [],
                          "why": clip(c.get("why")), "evidence": clip(c.get("evidence")),
                          "chosen_in_text": bool(zulu and unit in texts and zulu_stems.holds(zulu, texts[unit], analyses)), "unit_known": unit in texts})
        for m in [m for m in (missing_evidence or []) if isinstance(m, dict)][:20]:
            lines.append({"submission": submission, "kind": "missing_evidence", "unit": clip(m.get("unit"), 20), "wanted": clip(m.get("wanted"), 300), "why": clip(m.get("why"), 300)})
        for u in [u for u in (unit_strategies or []) if isinstance(u, dict)][:40]:
            lines.append({"submission": submission, "kind": "strategy", "unit": clip(u.get("unit"), 20), "strategy": clip(u.get("strategy"), 300)})
        try:
            with open(directory / "thinking.jsonl", "a", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError:
            pass

    @staticmethod
    def _flagged_entry(g: dict[str, Any]) -> dict[str, Any]:
        """One flagged unit as the agent sees it in a submit result: the gates that failed WITH their evidence (which required literal, which name), not only that a gate failed.  prod-98: after a
        hard-gate failure on u08 the agent got the unit id and had to infer which literal ("I inferred it was the source's three officers")."""
        failed = {n: v for n, v in g["gates"].items() if v["failed"]}
        entry: dict[str, Any] = {"item_id": g["group_id"], "gates": list(failed), "flags": g["flags"], "window_ms": g.get("window_ms"), "measured_ms": g.get("measured_ms")}
        evidence = {n: [str(e)[:GATE_EVIDENCE_CHARS] for e in (v.get("evidence") or [])[:3]] for n, v in failed.items() if v.get("evidence")}
        if evidence:
            entry["gate_evidence"] = evidence
        if "dead_air" in g["flags"]:
            entry["ends_early_ms"] = int(g.get("shortfall_ms") or 0) - int(g.get("tail_pad_ms") or 0)
        return entry

    @staticmethod
    def _voice_limit_words(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Words in the units that the voice cannot say well (see voice_limits): shown on every submit, free, never a gate."""
        from .voice_limits import for_items

        try:
            return for_items(items)[:UNIT_TEXTS_SHOWN]
        except Exception:                                             # noqa: BLE001 - an advisory view must never fail a submission
            return []

    def _sentence_timing(self, items: list[dict[str, Any]], card: dict[str, Any]) -> list[dict[str, Any]]:
        """Where each unit's sentences will land against the source speaker's once the renderer has moved the silence between them (see sentence_timing): the sentences that would still
        start well after the speaker's, and the units whose sentences mostly could not be matched.  On every submit, free, never a gate."""
        from .sentence_timing import check

        try:
            return check(items, card, self._source_word_timings(self._episode()["clip_id"]), speaker_rates(card), self._word_aligner())
        except Exception:                                             # noqa: BLE001 - an advisory view must never fail a submission
            return []

    def _word_aligner(self):
        """The aligner that says which Zulu word says what an English word said (word_alignment, from the parallel corpus), so a held pause is looked for where its words are and not at the
        same fraction of the sentence; None when the corpus is not installed (the check then uses the position alone)."""
        if not hasattr(self, "_aligner_hook"):
            from . import attestation, word_alignment

            try:
                resources = attestation.load(Path(self.store.root).parent)
            except Exception:                                             # noqa: BLE001 - an optional resource must never stop a check
                resources = None
            self._aligner_hook = word_alignment.make_aligner(resources.translation_forms) if resources is not None else None
        return self._aligner_hook

    def _spoken_form_gaps(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Units whose text still holds digits, an abbreviation or capitals that no rule converted (see say_it): shown on every submit, so the agent sees what the voice will guess at
        without having to ask.  Free, and never blocks a submission."""
        found = []
        for item in items:
            try:
                gaps = self.say_it(str(item.get("zulu_text") or ""), str(item.get("speaker_id") or "SPEAKER_00")).get("still_as_written")
            except Exception:                                             # noqa: BLE001 - an advisory view must never fail a submission
                continue
            if gaps:
                found.append({"item_id": str(item["item_id"]), "still_as_written": gaps})
        return found[:UNIT_TEXTS_SHOWN]

    def _figures_as_spoken(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Each figure, year, time or number the units write, and the words the voice is given for it (see say_it): shown on every submit so the agent reads its own numbers as a
        listener will hear them ("ngo-9.30" -> "ngo-nayn thethi") without having to ask.  Free, bounded, and never blocks a submission."""
        found, total = [], 0
        for item in items:
            try:
                changes = self.say_it(str(item.get("zulu_text") or ""), str(item.get("speaker_id") or "SPEAKER_00")).get("changes") or []
            except Exception:                                             # noqa: BLE001 - an advisory view must never fail a submission
                continue
            said = [f"{c['was']} -> {c['now']}" for c in changes if any(ch.isdigit() for ch in str(c.get("was") or ""))]
            if said:
                found.append({"item_id": str(item["item_id"]), "spoken": said[:6]})
                total += len(said[:6])
            if total >= FIGURES_SHOWN:
                break
        return found

    @staticmethod
    def _listening_load(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Units whose text runs figures together or goes on too long without a breath (see listening_load): soft evidence the agent sees on every submit, whether or not it calls check_text."""
        from .listening_load import figures_run_together, long_runs

        found = []
        for item in items:
            text = str(item.get("zulu_text") or "")
            together, runs = figures_run_together(text), long_runs(text, limit=1)
            if together or runs:
                found.append({"item_id": str(item["item_id"]), **({"figures_run_together": together} if together else {}), **({"long_run_without_breath": runs} if runs else {})})
        return found[:UNIT_TEXTS_SHOWN]

    @staticmethod
    def _with_texts(entries: list[dict[str, Any]], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The first UNIT_TEXTS_SHOWN entries carry the unit's current Zulu text.  In lenfix-2, 14 of 75 turns were get_submission calls made right after a
        submit only to read the text of the units it flagged; each such turn re-sends the whole 22,000-token context."""
        texts = {str(i["item_id"]): str(i.get("zulu_text") or "") for i in items}
        for entry in entries:                                          # the gap to the window, in characters at the unit's own measured pace (to_fit, as in a measurement)
            hint = Workbench._fit_hint(entry.pop("window_ms", None), entry.pop("measured_ms", None), len(texts.get(str(entry["item_id"]), "")))
            if hint:
                entry["to_fit"] = hint
        for entry in entries[:UNIT_TEXTS_SHOWN]:
            text = texts.get(str(entry["item_id"]), "")
            entry["zulu_text"] = text if len(text) <= UNIT_TEXT_CHARS else text[:UNIT_TEXT_CHARS] + "…"
        return entries
