"""Names from the speaker's audio.

Web research says which names a clip contains, how each is written and what kind of name it is. Nothing in it says how a name
sounds. The speaker's sounds come from the source audio: each English-shaped word is decoded straight from its occurrences,
the speaker's vowels replace the written vowels, and the consonants come from the written name spelt for an isiZulu voice.
isiZulu-shaped words and acronyms are left to the voice and to the initialism rules.

A vowel the recogniser hears only as a weak vowel (schwa and its kin), or a word whose count of heard vowels does not match
its written vowels, keeps what is written: the recogniser is less reliable than the page there.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from .pronunciation_names import NAMES_FILE, approved_names, write_lexicon

_NUCLEUS = re.compile(r"([aeiouy]+(?:r(?![aeiouy]))?)")
_HEARD_VOWELS = {
    "a": "a", "ɑ": "a", "æ": "a", "ʌ": "a", "aː": "aa", "ɑː": "aa",
    "e": "e", "ɛ": "e", "eː": "e", "i": "i", "iː": "ii", "o": "o", "ɔ": "o", "oː": "o", "ɔː": "o", "u": "u", "ʊ": "u", "uː": "uu",
    "eɪ": "ey", "aɪ": "ay", "ɔɪ": "oy", "əʊ": "o", "oʊ": "o", "aʊ": "aw",
}
_WEAK_VOWELS = {"ə", "ɐ", "ɪ", "ɜ", "ɜː", "ɨ"}
_KINDS = {"person": "personal_name", "organisation": "organisation_name", "place": "place_name"}


def is_vowel(token: str) -> bool:
    return token in _HEARD_VOWELS or token in _WEAK_VOWELS


def respell(word: str, heard: Sequence[str]) -> str:
    """The word spelt for an isiZulu voice: the speaker's vowels where the recogniser is sure, the page's elsewhere."""
    parts = _NUCLEUS.split(word.lower())                 # consonants, nucleus, consonants, nucleus, ... consonants
    nuclei = parts[1::2]
    vowels = [token for token in heard if is_vowel(token)]
    use_audio = len(vowels) == len(nuclei)
    out: list[str] = []
    for index, part in enumerate(parts):
        if index % 2:
            written = part.rstrip("r") if len(part) > 1 else part     # a vowel before a consonant is not followed by a sounded r
            if part.endswith("r") and len(part) > 1:
                written = part[:-1]
            spoken = _HEARD_VOWELS.get(vowels[index // 2]) if use_audio else None
            single = len(written) == 1
            out.append(spoken if spoken and single else ("i" if written == "y" else written))
            continue
        following = parts[index + 1] if index + 1 < len(parts) else ""
        out.append(_consonants(part, following))
    spelled = "".join(out)
    return spelled[:1].upper() + spelled[1:] if word[:1].isupper() else spelled


def _consonants(cluster: str, following: str) -> str:
    """A written consonant cluster in isiZulu spelling: soft and hard c, one letter for a doubled one, aspirated k/p/t."""
    text = cluster
    text = re.sub(r"ck", "k", text)
    text = re.sub(r"c(?!h)", lambda m: "s" if following[:1] in {"e", "i", "y"} and m.end() == len(text) else "k", text)
    text = re.sub(r"(.)\1", r"\1", text)
    if following and text and text[-1] in "kpt" and not text.endswith(("sk", "sp", "st")) and (len(text) == 1 or text[-2] not in "sh"):
        text += "h"
    return text


def audio_names(job_root: Path, *, voice: str = "zu-ZA-ThembaNeural", gop: Any = None,
                progress: Callable[[str], None] = print) -> Path | None:
    """Write the job's names lexicon from the speaker's audio. Returns the path, or None when there is no research."""
    from . import pronunciation_search as ps
    from .pronunciation import (
        PronunciationDictionary, with_default_organisation_initialisms, with_web_researched_organisation_pronunciations,
    )
    from .pronunciation_assess import LocalGop
    from .pronunciation_evidence import heard_word

    job_root = Path(job_root)
    research_path = job_root / "native_dub" / "pronunciation_research.json"
    if not research_path.is_file():
        progress("[names] no pronunciation research for this job; names stage skipped")
        return None
    research = json.loads(research_path.read_text(encoding="utf-8"))
    transcript_path = job_root / "analysis" / "azure_diarization.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8")).get("words") or [] if transcript_path.is_file() else []
    gop = gop or LocalGop()
    gop._load()
    builtin = with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id=str(job_root)))
    approved = approved_names(builtin)
    existing = with_web_researched_organisation_pronunciations(builtin, research)
    heard_cache: dict[str, Any] = {}
    plans: list[dict[str, Any]] = []
    kinds: dict[str, str] = {}
    for correction in research.get("accepted_corrections") or []:
        name = str(correction.get("canonical_text") or "").strip()
        if not name or name.casefold() in approved:
            continue
        spoken = existing.apply(name).tts_text.split()
        words = name.split()
        if len(spoken) != len(words):
            progress(f"[names] {name}: spoken as {len(spoken)} words (an existing structure); left to existing rules")
            continue
        decisions = []
        for index, word in enumerate(words):
            if not word.isalpha() or ps.looks_isizulu(word) or (len(word) > 1 and word.isupper()):
                decisions.append({"word": word, "decision": spoken[index],
                                  "why": "isiZulu shape, acronym, title or compound: left to the existing rules"})
                continue
            if word not in heard_cache:
                heard_cache[word] = heard_word(job_root / "audio" / "analysis_mono.wav", transcript, word, gop,
                                               job_root / "native_dub" / "audio_work")
            heard = heard_cache[word]
            if heard is None:
                progress(f"[names] {name}: {word!r} not found in the transcript under that spelling (a lookup failure to fix)")
                decisions.append({"word": word, "decision": spoken[index], "why": "not found in the transcript"})
                continue
            decisions.append({"word": word, "decision": respell(word, heard["phones"]), "heard": " ".join(heard["phones"]),
                              "why": "vowels from the speaker's audio, consonants from the written name"})
        reading = " ".join(d["decision"] for d in decisions)
        changed = reading.casefold() != " ".join(spoken).casefold()
        plans.append({"name": name, "result": reading, "changed": changed, "decisions": decisions})
        kinds[name] = _KINDS.get(str(correction.get("entity_type") or ""), "personal_name")
        progress(f"[names] {name} -> {reading!r}" + ("" if changed else "  (unchanged)"))
    path = write_lexicon(job_root / "native_dub" / NAMES_FILE, plans, voice=voice, kinds=kinds)
    try:        # what earlier jobs learned about this voice (verified repairs, readings a person approved)
        from .pronunciation_memory import apply_for_job

        remembered = apply_for_job(job_root)
        if remembered:
            progress(f"[names] from the shared memory: {', '.join(remembered)}")
    except Exception as exc:
        progress(f"[names] shared memory not applied: {type(exc).__name__}: {str(exc)[:120]}")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.pronunciation_audio")
    parser.add_argument("--job", required=True)
    parser.add_argument("--voice", default="zu-ZA-ThembaNeural")
    args = parser.parse_args(argv)
    print("lexicon:", audio_names(Path(load_settings().work_dir) / "jobs" / args.job, voice=args.voice))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
