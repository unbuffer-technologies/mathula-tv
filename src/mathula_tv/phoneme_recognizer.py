"""Language-independent phone recognition + phonetic-distance scoring, used to
compare the ORIGINAL English source pronunciation of a name against synthesized
Zulu TTS candidates in a shared phone space -- see native_dub.py's real TTS+STT
round-trip verification (Phase 18, `_measure_pronunciation_round_trip`) for the
downstream consumer this is meant to cheaply pre-filter candidates for.

Backed by Allosaurus (github.com/xinjli/allosaurus), a pretrained universal
phone recognizer covering 2000+ languages via a shared, language-independent
encoder. Deliberately called in its UNIVERSAL mode (``lang_id="ipa"``, ~230
phones across all languages) rather than pinning a language-specific
inventory: pinning a DIFFERENT lang code per side (e.g. "eng" for the English clip, "zul"
for the Zulu candidate) would map each side through a different language's
allophone layer and break the apples-to-apples comparison this module exists
for. Distance between two phone sequences is scored with panphon's
articulatory-feature-weighted edit distance (github.com/dmort27/panphon), not
plain string/Levenshtein edit distance, so a near-miss substitution (e.g. /b/
for /p/) costs less than an unrelated one (e.g. /b/ for /k/).

HONEST LIMITATION: Allosaurus's own phone recognition has a real error rate,
especially on short, isolated-word clips outside their original sentence
context. This module is a SUPPLEMENTARY, cheap, local-only ranking signal for
narrowing many candidate respellings down to one before spending a real (paid,
network) Azure TTS+STT round trip on it -- it never replaces that round trip,
which is still the only thing that reflects what the actual production zu-ZA
STT will hear.

Requires the optional ``phonetics`` extra (allosaurus + panphon + torch).
Every public function raises PhoneRecognitionUnavailable when the extra isn't
installed -- this must never be a hard requirement for the rest of the
pipeline to run.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class PhoneRecognitionUnavailable(RuntimeError):
    """Raised internally when the optional phonetics extra isn't installed."""


@dataclass(frozen=True)
class PhoneSequence:
    phones: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.phones)

    def as_ipa(self) -> str:
        """Phones joined with no separator, panphon's expected input shape --
        it re-segments an IPA string itself using its own longest-match
        segmenter, so a space (not a valid IPA segment) would corrupt that."""
        return "".join(self.phones)


def is_available() -> bool:
    """Whether the optional phonetics extra (allosaurus + panphon) is
    installed. Does not trigger Allosaurus's model download -- that only
    happens lazily inside recognize(), on first real use."""
    try:
        import allosaurus.app  # noqa: F401
        import panphon.distance  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _load_recognizer():
    try:
        from allosaurus.app import read_recognizer
    except ImportError as exc:
        raise PhoneRecognitionUnavailable(
            "allosaurus is not installed; install the 'phonetics' extra"
        ) from exc
    return read_recognizer()


@lru_cache(maxsize=1)
def _load_distance():
    try:
        from panphon.distance import Distance
    except ImportError as exc:
        raise PhoneRecognitionUnavailable(
            "panphon is not installed; install the 'phonetics' extra"
        ) from exc
    return Distance()


def recognize(audio_path: Path) -> PhoneSequence:
    """Universal (language-independent) phone sequence for one audio clip.

    Raises PhoneRecognitionUnavailable if the phonetics extra isn't
    installed; propagates any real Allosaurus inference error unchanged (a
    corrupt/unreadable audio file must surface, not silently score as empty).
    """
    recognizer = _load_recognizer()
    # Real bug, confirmed against the installed allosaurus version: its
    # Recognizer.recognize() parameter is named lang_id, not lang (checked
    # directly via inspect.signature -- (filename, lang_id='ipa', topk=1,
    # emit=1.0, timestamp=False)). The universal "ipa" value itself is
    # unchanged; only the keyword name was wrong.
    raw = recognizer.recognize(str(audio_path), lang_id="ipa")
    return PhoneSequence(tuple(raw.split()))


def phonetic_similarity(a: PhoneSequence, b: PhoneSequence) -> float:
    """Articulatory-feature-weighted similarity in [0, 1] between two phone
    sequences. Two empty sequences are trivially identical (1.0); one empty
    and one non-empty are maximally dissimilar (0.0), matching how an
    entirely missing/failed recognition should be scored -- never a false
    mid-range tie.

    Real bug, confirmed against panphon directly (2026-09-11): normalizing by
    raw phone COUNT (as if this were plain Levenshtein, cost ~1 per edit)
    silently floored every real-world comparison to 0.0. panphon's actual
    per-substitution cost is far larger -- a real measured case scored 43.5
    for two ~12-phone sequences (~3.6/phone), so dividing by phone count
    (~12) always overshot past 1.0 and hit the max(0.0, ...) clamp
    regardless of how similar the sequences actually were. Fixed to
    normalize against each sequence's own real "delete everything" distance
    (the cost of aligning it against nothing) -- the true worst case any
    real alignment is bounded by, in the SAME units as the raw distance
    (unlike phone count). Verified on real data: an identical pair now
    scores 1.0 (unchanged), a real English-vs-Zulu-TTS name comparison
    scores 0.500 (previously always exactly 0.0, indistinguishable from
    total garbage), and a genuinely unrelated short pair scores 0.138.

    Raises PhoneRecognitionUnavailable if the phonetics extra isn't installed.
    """
    if not a.phones and not b.phones:
        return 1.0
    if not a.phones or not b.phones:
        return 0.0
    distance_engine = _load_distance()
    a_ipa, b_ipa = a.as_ipa(), b.as_ipa()
    raw_distance = distance_engine.weighted_feature_edit_distance(a_ipa, b_ipa)
    worst_case = max(
        distance_engine.weighted_feature_edit_distance(a_ipa, ""),
        distance_engine.weighted_feature_edit_distance(b_ipa, ""),
    )
    if worst_case <= 0.0:
        return 1.0
    return max(0.0, 1.0 - raw_distance / worst_case)


def best_window_similarity(haystack: PhoneSequence, needle: PhoneSequence) -> float:
    """Best phonetic_similarity between `needle` and any equal-length
    contiguous window of `haystack`.

    Mirrors the text-space "best matching window" technique in native_dub.py's
    _pronunciation_round_trip_score (the partial-ratio trick) -- needed here
    for the same reason: a synthesized candidate's phone sequence includes the
    fixed carrier sentence's phones around the target name, which would
    otherwise dilute the name's own signal identically for every candidate,
    making the comparison useless for ranking them against each other.
    """
    if not needle.phones or not haystack.phones:
        return 0.0
    window_size = len(needle.phones)
    if window_size >= len(haystack.phones):
        return phonetic_similarity(haystack, needle)
    best = 0.0
    for start in range(0, len(haystack.phones) - window_size + 1):
        window = PhoneSequence(haystack.phones[start:start + window_size])
        best = max(best, phonetic_similarity(window, needle))
    return best
