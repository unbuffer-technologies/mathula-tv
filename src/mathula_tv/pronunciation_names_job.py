"""Run the names stage over one job: plan every researched name and write the per-job lexicon.

See ``pronunciation_names`` for the rules. This module is the part that needs audio, the phoneme model and Azure.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .pronunciation_names import NAMES_FILE, approved_names, plan_name, write_lexicon

_DEFAULT_CONTEXTS = (
    {"before": "Manje sibuza ", "after": "ngalolu daba.", "pitch": "-6%"},
    {"before": "Kuthiwe ", "after": "ubekhona ngaleso sikhathi.", "pitch": "-6%"},
)
_KINDS = {"person": "personal_name", "organisation": "organisation_name", "place": "place_name"}


def plan_job_names(job_root: Path, *, key: str, region: str, voice: str = "zu-ZA-ThembaNeural", gop: Any = None,
                   margin: float = 0.15, progress: Callable[[str], None] = print) -> Path | None:
    """Plan every researched name of a job and write ``native_dub/pronunciation_names.json``. Returns the path, or None."""
    from . import pronunciation_search as ps
    from .native_dub import _extract_entity_english_source_clip
    from .pronunciation import PronunciationDictionary, with_default_organisation_initialisms
    from .pronunciation_assess import (
        LocalGop, english_reading, gop_goodness, phone_tokens, regroup, sanitize_ipa, strip_length,
    )

    job_root = Path(job_root)
    research_path = job_root / "native_dub" / "pronunciation_research.json"
    if not research_path.is_file():
        progress("[names] no pronunciation research for this job; names stage skipped")
        return None
    research = json.loads(research_path.read_text(encoding="utf-8"))
    gop = gop or LocalGop()
    gop._load()
    work = job_root / "native_dub" / "names_work"
    synthesize = ps.bookmark_synthesizer(voice, key, region)
    approved = approved_names(with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id=str(job_root))))
    contexts = [ps.Context(**c) for c in _DEFAULT_CONTEXTS]
    plans, kinds = [], {}
    for correction in research.get("accepted_corrections") or []:
        name = str(correction.get("canonical_text") or "").strip()
        ipa = str(correction.get("pronunciation_ipa") or "")
        if not name or not ipa or name.casefold() in approved:
            continue
        try:
            base = sanitize_ipa(ipa, gop._vocab)
            phone_tokens(base, gop._vocab)
        except ValueError as exc:
            progress(f"[names] {name}: its pronunciation note cannot be read ({str(exc)[:80]}); left alone")
            continue
        if len(base.split()) != len(name.split()):
            progress(f"[names] {name}: the pronunciation note has {len(base.split())} sound groups for "
                     f"{len(name.split())} words; left alone")
            continue
        slug = re.sub(r"\W+", "_", name.lower())
        clip = _extract_entity_english_source_clip(job_root=job_root, entity_id=slug, canonical_text=name,
                                                   audio_dir=work / "reference")
        if clip is None:
            clip = english_reading(name, "en-ZA-LukeNeural", key, region, work / "reference" / f"{slug}_en_tts.wav")
        try:
            picked = gop.choose_expected(clip, base)
        except Exception as exc:            # one name that cannot be analysed must not stop the others
            progress(f"[names] {name}: could not be analysed ({type(exc).__name__}: {str(exc)[:80]}); left alone")
            continue
        words = [strip_length(w) if ps.looks_isizulu(word) else w
                 for w, word in zip(regroup(picked["chosen_tokens"], base, gop._vocab), name.split())]
        windows = ps.NameWindows(synthesize, lambda wav: (), work / slug)
        memo: dict[tuple[str, str], float] = {}

        def goodness(wav: Path, expected: str, memo: dict = memo) -> float:
            if (wav.name, expected) not in memo:
                memo[(wav.name, expected)] = gop_goodness(gop.assess(wav, expected)["score"])
            return memo[(wav.name, expected)]

        scorer = ps.WordScorer(windows, goodness, words, contexts)
        plan = plan_name(name, words, scorer, lambda w: phone_tokens(w, gop._vocab),
                         looks_isizulu=ps.looks_isizulu, margin=margin)
        plans.append(plan)
        kinds[name] = _KINDS.get(str(correction.get("entity_type") or ""), "personal_name")
        progress(f"[names] {name} -> {plan['result']!r}" + ("" if plan["changed"] else "  (unchanged)"))
    return write_lexicon(job_root / "native_dub" / NAMES_FILE, plans, voice=voice, kinds=kinds)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.pronunciation_names_job")
    parser.add_argument("--job", required=True)
    parser.add_argument("--voice", default="zu-ZA-ThembaNeural")
    args = parser.parse_args(argv)
    path = plan_job_names(Path(load_settings().work_dir) / "jobs" / args.job, key=os.environ["AZURE_SPEECH_KEY"],
                          region=os.environ["AZURE_SPEECH_REGION"].strip().lower(), voice=args.voice)
    print(f"lexicon: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
