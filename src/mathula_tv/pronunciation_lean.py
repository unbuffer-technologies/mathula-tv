"""Lean names: how much of the web-research respelling is still needed?

For every researched name in a job this compares two readings of the name:

  current  what the pipeline speaks today (built-in entries + this job's researched respellings)
  lean     what the names stage would speak if the research respelling did not exist: isiZulu-shaped words as written,
           acronyms and already-structured readings as they are, every other word spelt by rule from the sounds the
           speaker in the source audio (or an English voice reading the name) actually used

and scores each on the name's own audio (GOP against the speaker-derived sounds, and Azure pronunciation assessment). It
also writes takes of every name whose two readings differ so that a person can judge them by ear. The scores guide;
the ear decides, and nothing here changes what the pipeline speaks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .pronunciation_names import approved_names, phones_to_spelling

_CONTEXTS = (
    {"before": "Manje sibuza ", "after": "ngalolu daba.", "pitch": "-6%"},
    {"before": "Kuthiwe ", "after": "ubekhona ngaleso sikhathi.", "pitch": "-6%"},
)


def lean_reading(name: str, word_ipas: Sequence[str], current: Sequence[str], tokenise: Callable[[str], Sequence[str]], *,
                 looks_isizulu: Callable[[str], bool]) -> list[dict[str, Any]]:
    """The lean spelling of each word, with why. Words the rules cannot or should not touch keep their current reading."""
    words = name.split()
    out: list[dict[str, Any]] = []
    for i, word in enumerate(words):
        if looks_isizulu(word):
            out.append({"word": word, "lean": word, "why": "isiZulu shape: as written"})
        elif len(word) > 1 and word.isupper():
            out.append({"word": word, "lean": current[i], "why": "acronym: left to the initialism rules"})
        else:
            spelling, unconfirmed = phones_to_spelling(tokenise(word_ipas[i]), like=word)
            out.append({"word": word, "lean": spelling or word, "why": "rule from the speaker's sounds",
                        "unconfirmed_rules": unconfirmed})
    return out


def verdict(current_score: float, lean_score: float, margin: float = 0.1) -> str:
    if lean_score >= current_score + margin:
        return "lean better"
    if current_score >= lean_score + margin:
        return "current better"
    return "about equal"


def compare_job(job_root: Path, *, key: str, region: str, voice: str = "zu-ZA-ThembaNeural", gop: Any = None,
                progress: Callable[[str], None] = print) -> dict[str, Any]:
    from . import pronunciation_search as ps
    from .native_dub import _extract_entity_english_source_clip
    from .pronunciation import (
        PronunciationDictionary, with_default_organisation_initialisms, with_web_researched_organisation_pronunciations,
    )
    from .pronunciation_assess import (
        LocalGop, azure_assessor, azure_goodness, concatenate_takes, english_reading, gop_goodness, phone_tokens, regroup,
        sanitize_ipa, strip_length,
    )

    job_root = Path(job_root)
    research = json.loads((job_root / "native_dub" / "pronunciation_research.json").read_text(encoding="utf-8"))
    gop = gop or LocalGop()
    gop._load()
    work = job_root / "native_dub" / "lean_work"
    synthesize = ps.bookmark_synthesizer(voice, key, region)
    builtin = with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id=str(job_root)))
    approved = approved_names(builtin)
    current_dictionary = with_web_researched_organisation_pronunciations(builtin, research)
    contexts = [ps.Context(**c) for c in _CONTEXTS]
    rows: list[dict[str, Any]] = []
    take_files: list[Path] = []
    take_labels: list[str] = []
    for correction in research.get("accepted_corrections") or []:
        name = str(correction.get("canonical_text") or "").strip()
        ipa = str(correction.get("pronunciation_ipa") or "")
        if not name or not ipa:
            continue
        current_text = current_dictionary.apply(name).tts_text
        row: dict[str, Any] = {"name": name, "current": current_text, "approved_by_person": name.casefold() in approved,
                               "researched_respelling": correction.get("pronunciation_tts_text")}
        current = current_text.split()
        if row["approved_by_person"]:
            row["note"] = "a person approved this reading; not compared"
            rows.append(row)
            continue
        if len(current) != len(name.split()):
            row["note"] = f"current reading has a different structure ({len(current)} words for {len(name.split())}); left to existing rules"
            rows.append(row)
            continue
        try:
            base = sanitize_ipa(ipa, gop._vocab)
            phone_tokens(base, gop._vocab)
            if len(base.split()) != len(name.split()):
                raise ValueError("sound groups do not match the words")
            slug = re.sub(r"\W+", "_", name.lower())
            clip = _extract_entity_english_source_clip(job_root=job_root, entity_id=slug, canonical_text=name,
                                                       audio_dir=work / "reference")
            source = "speaker's audio"
            if clip is None:
                clip = english_reading(name, "en-ZA-LukeNeural", key, region, work / "reference" / f"{slug}_en_tts.wav")
                source = "English voice reading"
            picked = gop.choose_expected(clip, base)
            words = [strip_length(w) if ps.looks_isizulu(word) else w
                     for w, word in zip(regroup(picked["chosen_tokens"], base, gop._vocab), name.split())]
        except Exception as exc:
            row["note"] = f"could not analyse ({type(exc).__name__}: {str(exc)[:80]})"
            rows.append(row)
            continue
        lean_words = lean_reading(name, words, current, lambda w: phone_tokens(w, gop._vocab), looks_isizulu=ps.looks_isizulu)
        lean_text = " ".join(w["lean"] for w in lean_words)
        row.update(lean=lean_text, expected_from=source, expected_sounds=" ".join(words), lean_detail=lean_words)
        if lean_text.casefold() == current_text.casefold():
            row["verdict"] = "same reading"
            rows.append(row)
            continue
        whole_ipa = "".join(words)
        windows = ps.NameWindows(synthesize, lambda wav: (), work / slug)
        azure = azure_assessor(key, region, "en-US", name)
        scores: dict[str, dict[str, float]] = {}
        for label, text in (("current", current_text), ("lean", lean_text)):
            gops, azures = [], []
            for context in contexts:
                audio = windows.audio(context, ps.Candidate(text))
                gops.append(gop_goodness(gop.assess(audio, whole_ipa)["score"]))
                azures.append(azure_goodness(azure(audio)["score"]))
            scores[label] = {"gop": round(sum(gops) / len(gops), 3), "azure": round(sum(azures) / len(azures), 3)}
        row.update(scores=scores, verdict={"gop": verdict(scores["current"]["gop"], scores["lean"]["gop"]),
                                           "azure": verdict(scores["current"]["azure"], scores["lean"]["azure"])})
        for text in (current_text, lean_text):
            audio = windows.audio(contexts[0], ps.Candidate(text))
            take_files.append(audio.with_name(audio.name.replace(".name.wav", ".wav")))
        take_labels.append(f"{name}: current {current_text!r} then lean {lean_text!r}")
        progress(f"[lean] {name}: current {current_text!r} | lean {lean_text!r} | GOP {scores['current']['gop']} vs "
                 f"{scores['lean']['gop']} | Azure {scores['current']['azure']} vs {scores['lean']['azure']}")
        rows.append(row)
    report = {"job": job_root.name, "voice": voice, "rows": rows, "takes_order": take_labels}
    if take_files:
        report["takes"] = str(concatenate_takes(take_files, job_root / "native_dub" / "lean_takes.wav", gap_s=0.6))
    (job_root / "native_dub" / "lean_comparison.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.pronunciation_lean")
    parser.add_argument("--job", required=True)
    parser.add_argument("--voice", default="zu-ZA-ThembaNeural")
    args = parser.parse_args(argv)
    report = compare_job(Path(load_settings().work_dir) / "jobs" / args.job, key=os.environ["AZURE_SPEECH_KEY"],
                         region=os.environ["AZURE_SPEECH_REGION"].strip().lower(), voice=args.voice)
    print(f"\n{'name':<40} {'current':<34} {'lean':<34} verdict")
    for row in report["rows"]:
        shown = row.get("verdict")
        verdict_text = json.dumps(shown) if isinstance(shown, dict) else (shown or row.get("note", ""))
        print(f"{row['name'][:39]:<40} {row['current'][:33]:<34} {str(row.get('lean', ''))[:33]:<34} {verdict_text}")
    print("\ntakes:", report.get("takes"), "\norder:", *report["takes_order"], sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
