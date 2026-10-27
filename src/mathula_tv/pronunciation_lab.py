"""A pronunciation lab for the dubbing agent: hear what the voice does with a spelling, and compare it with the speaker.

The agent writes text and never hears audio, and the Zulu TTS voice takes only text, so the only way to make it say an English word
the way the speaker did is to try spellings. This is that loop, made available to the agent as tools:

* ``source(word)``      the sounds the speaker produced for the word, decoded from their own audio (all occurrences, consensus);
* ``try_spelling``      synthesise a spelling in the Zulu voice, decode the sounds it produced with the same recogniser, and say how
                         they differ from the speaker's, in plain terms ("voice drops the d", "vowel o instead of ɔɪ");
* ``candidate_words``   the English words the Zulu text keeps, found in the speaker's audio.

It measures; it never decides. The agent chooses a spelling from the differences, the number of tries is capped, and a person's ear stays
the check on a first use. The comparison below is guidance for the agent, not a pass/fail score.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

MAX_TRIES_PER_WORD = 5
MAX_TRIES_PER_CLIP = 15
MAX_SPELLINGS_PER_CALL = 3

_VOWEL_LETTERS = set("aeiouɑɒɔəɛɜɪʊʌæɐøɨ")
_VOICING = {("p", "b"), ("t", "d"), ("k", "ɡ"), ("k", "g"), ("s", "z"), ("f", "v"), ("ʃ", "ʒ"), ("tʃ", "dʒ"), ("θ", "ð")}
_EDGE_COST = 0.3          # sounds at the edges of a word window are noisy in the recogniser: skipping them costs little
_MARKS = "ːˑʰʲʷ̩̯̃"


class PronunciationBudgetExhausted(Exception):
    pass


def _is_vowel(phone: str) -> bool:
    return any(c in _VOWEL_LETTERS for c in phone)


def _base(phone: str) -> str:
    return "".join(c for c in phone if c not in _MARKS)


def substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    base_a, base_b = _base(a), _base(b)
    if base_a == base_b:
        return 0.1                                    # length or aspiration only
    if _is_vowel(a) and _is_vowel(b):
        return 0.3 if base_a[:1] == base_b[:1] else 0.6
    if _is_vowel(a) != _is_vowel(b):
        return 1.0
    if (base_a, base_b) in _VOICING or (base_b, base_a) in _VOICING:
        return 0.3
    return 0.7


def align(source: Sequence[str], heard: Sequence[str]) -> dict[str, Any]:
    """Weighted alignment of the speaker's sounds with the voice's. Returns ``distance`` (0 = identical), ``closeness`` (1 - distance)
    and ``differences`` in plain words.

    The speaker's window is cut from running speech by word timings, so its first and last sounds are noisy: skipping them costs
    little. The voice's window is cut cleanly with SSML bookmarks, so a sound the voice adds anywhere (the "ma" of "mafector") is real."""
    n, m = len(source), len(heard)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i * _EDGE_COST, (i - 1, 0, "skip_source")
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j * 1.0, (0, j - 1, "skip_heard")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(dp[i - 1][j - 1] + substitution_cost(source[i - 1], heard[j - 1]), (i - 1, j - 1, "pair")),
                       (dp[i - 1][j] + 1.0, (i - 1, j, "skip_source")),
                       (dp[i][j - 1] + 1.0, (i, j - 1, "skip_heard"))]
            dp[i][j], back[i][j] = min(options, key=lambda option: option[0])
    # the speaker's last sounds may be skipped cheaply (noisy edge); the voice's extra last sounds may not
    finish = min([(dp[n][j] + (m - j) * 1.0, n, j) for j in range(m + 1)] + [(dp[i][m] + (n - i) * _EDGE_COST, i, m) for i in range(n + 1)])
    total, i, j = finish
    differences: list[str] = []
    if m - j:
        differences.append(f"the voice adds {' '.join(heard[j:])} at the end")
    steps = []
    while (i, j) != (0, 0):
        previous_i, previous_j, step = back[i][j]  # type: ignore[misc]
        steps.append((step, i - 1, j - 1))
        i, j = previous_i, previous_j
    steps.reverse()
    for step, si, hj in steps:
        speaker_edge = si <= 0 or si >= n - 1
        if step == "pair" and source[si] != heard[hj] and substitution_cost(source[si], heard[hj]) > 0.1:
            kind = "vowel" if _is_vowel(source[si]) else "consonant"
            differences.append(f"speaker {source[si]}, voice {heard[hj]} ({kind})")
        elif step == "skip_source" and not speaker_edge:
            differences.append(f"the voice leaves out {source[si]}")
        elif step == "skip_heard":
            differences.append(f"the voice adds {heard[hj]}" + (" at the start" if hj == 0 else ""))
    distance = total / max(n, m, 1)
    return {"distance": round(distance, 3), "closeness": round(max(0.0, 1.0 - distance), 3), "differences": differences[:6]}


# A Zulu prefix written straight onto an English word, no hyphen ("weScience", "iScience", "NeTechnology", "MaFighter"): one to four letters,
# then a capital that starts the English word.  Missed before, so "Science" was never offered for checking and was spoken with Zulu letter rules.
_GLUED_PREFIX = re.compile(r"^[A-Za-z][a-z]{0,3}(?=[A-Z][a-z]{3,})")


def _stem(word: str) -> str:
    word = re.sub(r"[^a-z]", "", word.casefold())
    return word[:-1] if word.endswith("s") and len(word) > 4 else word


def candidate_words(texts: Iterable[str], source_words: Sequence[Mapping[str, Any]], *, looks_isizulu: Callable[[str], bool],
                    exclude: set[str] = frozenset(), limit: int = 16, named: Sequence[Mapping[str, str]] = ()) -> list[dict[str, Any]]:
    """English words that the Zulu text keeps and that the speaker says in the source audio.

    A token counts when its stem (after any isiZulu prefix glued on, with a hyphen or straight onto a capital: "weScience") is a word of the
    English transcript.  A glued prefix proves the writer meant an English word, so it is kept even when the word happens to end in a vowel and
    so looks isiZulu ("Science").  Names already given a reading (``exclude``) are skipped.

    ``named`` is what a model read out of the Zulu text itself ({"written": "iScience", "english": "science"}): it finds the English words
    by meaning, whatever their spelling or prefix.  Only the speaker's side (the transcript) is still matched here, because the lab needs
    the speaker's own sounds for the word."""
    texts = list(texts)
    spoken: dict[str, dict[str, Any]] = {}
    for word in source_words:
        text = str(word.get("text") or "")
        stem = _stem(text)
        if len(stem) >= 4:
            spoken.setdefault(stem, {"as_spoken": re.sub(r"[^A-Za-z']", "", text), "count": 0})["count"] += 1
    found: dict[str, dict[str, Any]] = {}
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z'\-]*", text):
            written = token.split("-")[-1]
            unglued = _GLUED_PREFIX.sub("", written)
            for variant, glued in ((written, None), (unglued, "prefix") if unglued != written else (written, None)):
                stem = _stem(variant)
                if len(stem) < 4 or stem not in spoken or (looks_isizulu(variant) and not glued) or written.casefold() in exclude or variant.casefold() in exclude:
                    continue
                entry = found.setdefault(written, {
                    "word": written, "occurrences_in_dub": 0, "speaker_says": spoken[stem]["as_spoken"],
                    "times_the_speaker_says_it": spoken[stem]["count"],
                    "note": "a Zulu prefix is glued on: the speaker says the English word as it is" if glued else ""})
                entry["occurrences_in_dub"] += 1
                break
    for item in named:
        written, english = str(item.get("written") or "").strip(), str(item.get("english") or "").strip()
        stem = _stem(english)
        if len(stem) < 4 or stem not in spoken or written in found or written.casefold() in exclude or english.casefold() in exclude:
            continue
        count = sum(text.count(written) for text in texts)
        if count:
            found[written] = {"word": written, "occurrences_in_dub": count, "speaker_says": spoken[stem]["as_spoken"],
                              "times_the_speaker_says_it": spoken[stem]["count"],
                              "note": "a Zulu prefix is glued on: the speaker says the English word as it is" if written.casefold() != english.casefold() else ""}
    return sorted(found.values(), key=lambda e: (-e["occurrences_in_dub"], e["word"]))[:limit]


class PronunciationLab:
    """The sounds of a word as the speaker said it, and as the voice says a spelling. One instance per clip; models load on first use."""

    def __init__(self, job_root: Path, *, voice: str, key: str, region: str, work_dir: Path, gop: Any = None) -> None:
        self.job_root, self.voice, self.key, self.region, self.work_dir = Path(job_root), voice, key, region, Path(work_dir)
        self._gop = gop
        self._windows: Any = None
        self._transcript: list[Mapping[str, Any]] | None = None
        self.tries_per_word: dict[str, int] = {}
        self.tried: dict[str, dict[str, dict[str, Any]]] = {}
        self._source_cache: dict[str, dict[str, Any]] = {}

    @property
    def gop(self) -> Any:
        if self._gop is None:
            from .pronunciation_assess import LocalGop

            self._gop = LocalGop()
        self._gop._load()
        return self._gop

    @property
    def transcript(self) -> list[Mapping[str, Any]]:
        if self._transcript is None:
            path = self.job_root / "analysis" / "azure_diarization.json"
            self._transcript = json.loads(path.read_text(encoding="utf-8")).get("words") or [] if path.is_file() else []
        return self._transcript

    @property
    def total_tries(self) -> int:
        return sum(self.tries_per_word.values())

    def source(self, word: str) -> dict[str, Any]:
        """The sounds the speaker used for ``word`` (consensus over its occurrences in the source audio)."""
        spoken_stems = {_stem(str(w.get("text"))): str(w["text"]) for w in self.transcript}
        stem = next((c for c in (_stem(word), _stem(re.sub(r"^[Mm]a(?=[A-Z])", "", word))) if c in spoken_stems), _stem(word))
        if stem not in self._source_cache:
            from .pronunciation_evidence import heard_word

            spoken_form = spoken_stems.get(stem, word)
            heard = heard_word(self.job_root / "audio" / "analysis_mono.wav", self.transcript, re.sub(r"[^A-Za-z']", "", spoken_form), self.gop,
                               self.work_dir / "source", limit=4)
            self._source_cache[stem] = ({"found": True, "phones": heard["phones"], "occurrences": len(heard["spans"]),
                                         "each_occurrence": [" ".join(d) for d in heard["decoded"]]} if heard else {"found": False, "phones": []})
        return self._source_cache[stem]

    def voice_reading(self, spelling: str) -> list[str]:
        """The sounds the Zulu voice produces for ``spelling``, decoded from its own audio with the same recogniser."""
        from . import pronunciation_search as ps

        if self._windows is None:
            self._windows = ps.NameWindows(ps.bookmark_synthesizer(self.voice, self.key, self.region), lambda wav: (), self.work_dir / "voice")
        # the word alone, with nothing before or after it, so the neighbouring words' sounds do not leak into its window
        audio = self._windows.audio(ps.Context(before="", after=".", pitch="-6%"), ps.Candidate(spelling))
        return list(self.gop.decode(audio))

    def try_spellings(self, word: str, spellings: Sequence[str]) -> dict[str, Any]:
        """Try up to three spellings of ``word``; each is compared with the speaker's sounds. Counts against the budget."""
        spellings = [s.strip() for s in spellings if str(s).strip()][:MAX_SPELLINGS_PER_CALL]
        if not spellings:
            raise ValueError("give at least one spelling to try")
        source = self.source(word)
        if not source["found"]:
            return {"word": word, "error": "the speaker never says this word in the source audio, so there is nothing to match"}
        results = []
        for spelling in spellings:
            if self.total_tries >= MAX_TRIES_PER_CLIP or self.tries_per_word.get(word, 0) >= MAX_TRIES_PER_WORD:
                raise PronunciationBudgetExhausted(
                    f"pronunciation tries used: {self.tries_per_word.get(word, 0)}/{MAX_TRIES_PER_WORD} for {word!r}, {self.total_tries}/{MAX_TRIES_PER_CLIP} in all")
            self.tries_per_word[word] = self.tries_per_word.get(word, 0) + 1
            heard = self.voice_reading(spelling)
            comparison = align(source["phones"], heard)
            record = {"spelling": spelling, "voice_sounds": " ".join(heard), **comparison}
            self.tried.setdefault(word, {})[spelling] = record
            results.append(record)
        best = max(self.tried[word].values(), key=lambda r: r["closeness"])
        return {"word": word, "speaker_sounds": " ".join(source["phones"]), "speaker_occurrences": source["occurrences"], "results": results,
                "closest_so_far": {"spelling": best["spelling"], "closeness": best["closeness"]},
                "tries_left_for_word": MAX_TRIES_PER_WORD - self.tries_per_word[word], "tries_left_in_clip": MAX_TRIES_PER_CLIP - self.total_tries}
