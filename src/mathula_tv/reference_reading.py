"""The English reference for a word: a dictionary reading, chosen by the speaker's audio, instead of whatever the recogniser decodes.

The lab used to take the speaker's sounds from a free decode of the word cut out of running speech.  That is the weakest listening there is: the cut
carries a neighbour's sounds (the "ko" after ANC, the "mɪ" after News), the recogniser blurs voicing and the exact vowel, and a fast word comes out
short.  Scoring is a different and easier job than decoding.  The dictionary (or a model's guess for a word it lacks) offers readings, with the ways an
English speaker reduces them (a reduced vowel, the t of Western, no r after a vowel), and the audio is asked only which of them it fits best.  A sound
that the cut brings in from a neighbour costs every reading the same, so it cannot change the choice.

On 200 clean Multipron names (every word in the dictionary) the reading chosen this way was 0.861 close to what people labelled, against 0.835 for
the free decode, 0.852 for the dictionary without the audio, and 0.865 for the best possible choice.

Whether the audio contains the word at all is asked against decoys: other words of about the same length, scored on the same clips.  The clips' extra sounds cost
the decoys what they cost the reading, so the reading must beat them, and a window that points at other speech (the transcript's time for "High" held "p aː k")
gives it no advantage.  A free decode could not be the yardstick: it explains the neighbour's sounds too, so a good reading looked bad against it.
"""
from __future__ import annotations

import random
import statistics
from typing import Any, Mapping, Sequence

REDUCED = frozenset({"ə", "ɚ", "ɪ"})
REDUCTION_CHOICES = ("ə", "ɪ", "ɛ", "a", "i", "o")
MAX_REDUCED_POSITIONS = 3          # the reductions tried at once; the rest keep the dictionary's vowel
MAX_VARIANTS = 600
MIN_GAIN = 0.5                     # per phone: another reading (a rare variant, the other accent) must beat the first by this much, or the first stands (Western's
                                   # "hw" variant won by 0.33 on a window with a stray onset; Party's non-rhotic reading wins by 0.9, AI's letters over the word by 6)
DECOYS = 40
SEPARATION = 1.5                   # per phone: how much better than the median decoy the reading must fit for the audio to contain the word


def variants(tokens: Sequence[str]) -> list[list[str]]:
    """The dictionary's reading first, then how a speaker may say it: the t of "Western" and "Christmas" dropped, an unstressed vowel reduced to another."""
    base = list(tokens)
    out: list[list[str]] = [base]
    # t between s and a syllabic n / schwa / l is not said: Wesen, Chrismas
    for i in range(1, len(base) - 1):
        if base[i] == "t" and base[i - 1] == "s" and base[i + 1] in {"ə", "ɚ", "n", "l", "ɪ"}:
            out.append(base[:i] + base[i + 1:])
    # an unstressed vowel is said as another reduced vowel
    for form in list(out):
        positions = [i for i, token in enumerate(form) if token in REDUCED][:MAX_REDUCED_POSITIONS]
        grown = [form]
        for position in positions:
            grown = grown + [v[:position] + [choice] + v[position + 1:] for v in grown for choice in REDUCTION_CHOICES if choice != v[position]]
            grown = grown[:MAX_VARIANTS * 2]
        out.extend(grown)
    seen: dict[tuple[str, ...], None] = {}
    for form in out:
        seen.setdefault(tuple(form), None)
    return [list(form) for form in list(seen)[:MAX_VARIANTS]]


def decoy_tokens(words: Mapping[str, Sequence[Sequence[str]]], stem: str, length: int, vocab: Mapping[str, int], count: int = DECOYS) -> list[list[str]]:
    """Dictionary words of about ``length`` sounds other than ``stem`` (the same ones every time for the same word), as recogniser tokens."""
    from .english_sounds import to_ipa
    from .pronunciation_assess import phone_tokens

    pool = [word for word, readings in words.items() if word != stem.lower() and readings and abs(len(readings[0]) - length) <= 1]
    random.Random(sum(ord(c) for c in stem)).shuffle(pool)
    out: list[list[str]] = []
    for word in pool:
        tokens = [t for t in phone_tokens("".join(to_ipa(list(words[word][0]))), vocab) if t in vocab]
        if tokens:
            out.append(tokens)
        if len(out) >= count:
            break
    return out


def choose(clips: Sequence[Any], readings: Sequence[Mapping[str, Any]], decoded: Sequence[str], vocab: Mapping[str, int], blank: int,
           decoys: Sequence[Sequence[str]] = ()) -> dict[str, Any] | None:
    """The reading (with its variants) that the clips' log-probabilities support best, and whether the audio contains the word at all.

    ``readings`` are the interpreter's ({"how", "phones", "accent", "stress"}), the dictionary's first reading first; ``decoded`` is the free decode, kept for the
    caller to show.  The first reading stands unless another beats it by ``MIN_GAIN``.  None when there is nothing to score."""
    from .pronunciation_assess import gop_excess, phone_tokens

    if not clips or not readings:
        return None

    def cost(tokens: Sequence[str]) -> float:
        ids = [vocab[t] for t in tokens]
        return sum(gop_excess(clip, ids, blank) for clip in clips) / len(clips)

    families: list[tuple[float, list[str], Mapping[str, Any]]] = []
    for reading in readings:
        tokens = [t for t in phone_tokens("".join(reading["phones"]), vocab) if t in vocab]
        if not tokens:
            continue
        best = min(((cost(form), form) for form in variants(tokens)), key=lambda scored: scored[0])
        families.append((best[0], best[1], reading))
    if not families:
        return None
    first, fastest = families[0], min(families, key=lambda family: family[0])
    score, form, reading = fastest if fastest[0] <= first[0] - MIN_GAIN else first
    decoy_costs = sorted(cost(tokens) for tokens in decoys)
    median = statistics.median(decoy_costs) if decoy_costs else None
    separation = None if median is None else median - score
    return {"phones": form, "reading": reading, "excess": round(score, 3), "first_reading_excess": round(first[0], 3),
            "readings_scored": [(" ".join(family[2]["phones"]), family[2]["accent"], round(family[0], 3)) for family in families],
            "decoy_median": None if median is None else round(median, 3), "decoy_best": round(decoy_costs[0], 3) if decoy_costs else None,
            "separation": None if separation is None else round(separation, 3), "beaten_by_decoys": sum(1 for c in decoy_costs if c < score),
            "accepted": separation is not None and separation >= SEPARATION}
