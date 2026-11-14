"""Does this Zulu sound like Zulu people say or write?  Evidence from word-pair statistics over real text, for the dub agent's submit feedback.

The scorer rewards fit, fidelity and the gates; nothing in the loop pulls the agent toward speech that sounds natural and lifelike (it is third in the prompt and in no measurement).  This
module gives that goal a first, independent signal that no model produces: how real isiZulu behaves.  It counts words and adjacent word pairs in

* spoken: African Next Voices isiZulu, 56,479 human-written transcripts of speakers talking (about 2.0 million words);
* news: Zulu news-site text from FineWeb-2 (about 7.8 million words);
* government: the Autshumato isiZulu side (about 3.3 million words, translated and formal),

kept apart so that register can be told: a word that is common in writing and almost never said is bookish in a casual interview.

Two things are reported for a text, each only when the evidence is strong, and never as a verdict:

* ``unseen_pairs``: two common words side by side that never meet in the corpus although, by their frequencies, they would meet at least ``EXPECTED_MIN`` times by chance.  Sparse
  statistics make most unseen pairs harmless in a language that builds words with prefixes, which is why the expectation is required: absence is only evidence where presence was likely.
* ``bookish_words``: a word the written corpora use often (enough that the spoken corpus should have it several times) and speakers almost never do.

How the thresholds were chosen (and what they cost) is in ``scripts/naturalness_eval.py``: held-out real text must pass at a low false-alarm rate, and deliberately corrupted text must be caught.
This is evidence for the agent to weigh, the same as the others; it is not a judge of Zulu and does not read meaning.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SOURCES = ("spoken", "news", "gov")
_SENTENCE_BREAK = re.compile(r"[.!?;:\n]+|~pause\d+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
MIN_WORD = 20               # a word is "common" when the corpora hold it this often: only then does the absence of a pair mean something
EXPECTED_MIN = 3.0          # a pair of common words that never meets although chance alone would put it this many times in the corpus
BOOKISH_WRITTEN_MIN = 40    # written count before a word's absence from speech is read
BOOKISH_EXPECTED_SPOKEN = 10.0
SHOWN = 3


def sentences(text: str) -> list[list[str]]:
    """The sentences of a text as lists of lower-case word forms.  Words with digits are dropped; a hyphenated form (i-microphone, ama-details) is one word; names are kept as words (they are rare, so
    they never count as common)."""
    found = []
    for chunk in _SENTENCE_BREAK.split(str(text or "")):
        words = [w.lower().strip("-'") for w in _TOKEN.findall(chunk.replace("’", "'"))]
        words = [w for w in words if w]
        if words:
            found.append(words)
    return found


class NgramIndex:
    """Word and adjacent-pair counts for each source.  Pairs are stored as sorted uint64 codes (first word id times the vocabulary size plus second word id) with their counts."""

    def __init__(self, vocab: Sequence[str], unigrams: Mapping[str, np.ndarray], pair_codes: Mapping[str, np.ndarray], pair_counts: Mapping[str, np.ndarray], tokens: Mapping[str, int]) -> None:
        self.vocab = list(vocab)
        self.ids = {w: i for i, w in enumerate(self.vocab)}
        self.unigrams = dict(unigrams)
        self.pair_codes = dict(pair_codes)
        self.pair_counts = dict(pair_counts)
        self.tokens = dict(tokens)
        self.size = len(self.vocab)
        self.total_tokens = max(1, sum(self.tokens.values()))

    # -- building -----------------------------------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, corpora: Mapping[str, Iterable[str]]) -> "NgramIndex":
        """``corpora`` maps a source name to its texts (a line or a document each)."""
        per_source: dict[str, list[list[list[str]]]] = {}
        vocab: dict[str, int] = {}
        for name, texts in corpora.items():
            kept = []
            for text in texts:
                for sentence in sentences(text):
                    for word in sentence:
                        if word not in vocab:
                            vocab[word] = len(vocab)
                    kept.append(sentence)
            per_source[name] = kept
        size = len(vocab)
        unigrams, codes, counts, tokens = {}, {}, {}, {}
        for name, kept in per_source.items():
            uni = np.zeros(size, dtype=np.int64)
            pairs: list[np.ndarray] = []
            total = 0
            for sentence in kept:
                ids = np.fromiter((vocab[w] for w in sentence), dtype=np.int64, count=len(sentence))
                total += len(ids)
                np.add.at(uni, ids, 1)
                if len(ids) > 1:
                    pairs.append(ids[:-1].astype(np.uint64) * np.uint64(size) + ids[1:].astype(np.uint64))
            unigrams[name], tokens[name] = uni, total
            if pairs:
                joined = np.concatenate(pairs)
                codes[name], counts[name] = np.unique(joined, return_counts=True)
            else:
                codes[name], counts[name] = np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
        return cls(list(vocab), unigrams, codes, counts, tokens)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"vocab": np.array(self.vocab, dtype=object).astype(str)}
        for name in self.unigrams:
            arrays[f"uni_{name}"], arrays[f"codes_{name}"], arrays[f"counts_{name}"] = self.unigrams[name], self.pair_codes[name], self.pair_counts[name]
            arrays[f"tokens_{name}"] = np.array([self.tokens[name]])
        np.savez_compressed(directory / "ngrams.npz", **arrays)

    @classmethod
    def load(cls, directory: Path) -> "NgramIndex | None":
        path = Path(directory) / "ngrams.npz"
        if not path.is_file():
            return None
        data = np.load(path, allow_pickle=False)
        names = [k[len("uni_"):] for k in data.files if k.startswith("uni_")]
        return cls(data["vocab"].tolist(), {n: data[f"uni_{n}"] for n in names}, {n: data[f"codes_{n}"] for n in names}, {n: data[f"counts_{n}"] for n in names},
                   {n: int(data[f"tokens_{n}"][0]) for n in names})

    # -- counts -------------------------------------------------------------------------------------------------------------------------
    def word(self, word: str, source: str | None = None) -> int:
        i = self.ids.get(word)
        if i is None:
            return 0
        return int(self.unigrams[source][i]) if source else int(sum(self.unigrams[s][i] for s in self.unigrams))

    def pair(self, first: str, second: str, source: str | None = None) -> int:
        a, b = self.ids.get(first), self.ids.get(second)
        if a is None or b is None:
            return 0
        code = np.uint64(a) * np.uint64(self.size) + np.uint64(b)
        total = 0
        for name in ([source] if source else list(self.pair_codes)):
            codes = self.pair_codes[name]
            at = int(np.searchsorted(codes, code))
            if at < len(codes) and codes[at] == code:
                total += int(self.pair_counts[name][at])
        return total


def check(index: NgramIndex, text: str, *, min_word: int = MIN_WORD, expected_min: float = EXPECTED_MIN) -> dict[str, Any]:
    """The evidence for a text: ``unseen_pairs`` (strongest expectation first) and ``bookish_words``.  Empty lists when there is nothing worth saying."""
    unseen: dict[tuple[str, str], float] = {}
    bookish: dict[str, float] = {}
    spoken_tokens = max(1, index.tokens.get("spoken", 0))
    written_tokens = max(1, index.tokens.get("news", 0) + index.tokens.get("gov", 0))
    for sentence in sentences(text):
        for first, second in zip(sentence, sentence[1:]):
            count_a, count_b = index.word(first), index.word(second)
            if count_a < min_word or count_b < min_word or first == second:
                continue
            expected = count_a * count_b / index.total_tokens
            if expected >= expected_min and index.pair(first, second) == 0:
                unseen[(first, second)] = max(unseen.get((first, second), 0.0), expected)
        for word in sentence:
            spoken, written = index.word(word, "spoken"), index.word(word, "news") + index.word(word, "gov")
            if written >= BOOKISH_WRITTEN_MIN and spoken <= 1:
                expected_spoken = written / written_tokens * spoken_tokens
                if expected_spoken >= BOOKISH_EXPECTED_SPOKEN:
                    bookish[word] = max(bookish.get(word, 0.0), expected_spoken)
    return {"unseen_pairs": [f"{a} {b}" for (a, b), _ in sorted(unseen.items(), key=lambda kv: -kv[1])],
            "bookish_words": [w for w, _ in sorted(bookish.items(), key=lambda kv: -kv[1])]}


# -- translationese: does a unit read like translated government text or like native Zulu? ------------------------------------------------------------------------------------
# The government text is translated from English, so it carries English-shaped wording; the news and the speech are written and said by Zulu speakers.  A word far more common in the translated
# text than in native text pulls a sentence toward translationese.  Measured on held-out text (scripts/naturalness_eval.py --register): the sentence score separates translated from native
# sentences (AUC 0.875; 0.82 against speech alone, 0.93 against news alone).  It works on a sentence, not on a word: the words that pull a sentence are often ordinary words that government text
# happens to use a lot (iqhaza, incazelo), so a word is never reported as wrong, only named as one of what pulls.  Political and legal subjects are well covered on the native side (the news
# corpus is Isolezwe, IOL, News24, Scrolla, Bayede, Ilanga, Impempe), which is why this contrast is used and not speech against writing.  Caveat: some news sites translate English wire stories.
REGISTER_MIN_WORD_TOTAL = 20         # a word counts only when the corpora hold it this often
REGISTER_MIN_SCORED = 3              # scored words a sentence needs
REGISTER_MIN_SENTENCE = 6            # words a sentence needs to be scored at all
REGISTER_CUT = 0.63                  # native 95th percentile of the sentence score (held out, 10,413 native sentences)
REGISTER_CUT_SINGLE = 0.96           # native 99th percentile: a unit with one scored sentence needs this much
REGISTER_PULL_MIN_GOV = 15


def word_register(index: NgramIndex, word: str) -> float | None:
    """log of how much more often the word is written in the translated government text than in native speech and news (per word, smoothed); None for a word too rare to say anything."""
    spoken, news, gov = index.word(word, "spoken"), index.word(word, "news"), index.word(word, "gov")
    if spoken + news + gov < REGISTER_MIN_WORD_TOTAL:
        return None
    native_tokens = max(1, index.tokens.get("spoken", 0) + index.tokens.get("news", 0))
    return float(np.log(((gov + 1) / max(1, index.tokens.get("gov", 0))) / ((spoken + news + 1) / native_tokens)))


def sentence_register(index: NgramIndex, sentence: Sequence[str]) -> float | None:
    if len(sentence) < REGISTER_MIN_SENTENCE:
        return None
    scores = [x for x in (word_register(index, w) for w in sentence) if x is not None]
    return sum(scores) / len(scores) if len(scores) >= REGISTER_MIN_SCORED else None


def register_check(index: NgramIndex, text: str, *, cut: float = REGISTER_CUT, cut_single: float = REGISTER_CUT_SINGLE) -> dict[str, Any] | None:
    """A unit whose sentences read like translated government Zulu: at least half of its scored sentences (two or more) above the native 95th percentile, or one scored sentence above the
    stricter single-sentence cut.  ``{"sentences": n, "above": k, "pulling": [words]}``, or None."""
    scored = [(sentence, sentence_register(index, sentence)) for sentence in sentences(text)]
    scored = [(sen, x) for sen, x in scored if x is not None]
    if not scored:
        return None
    above = [(sen, x) for sen, x in scored if x > cut]
    flagged = (len(scored) >= 2 and len(above) * 2 >= len(scored) and len(above) >= 2) or (len(scored) == 1 and scored[0][1] > cut_single)
    if not flagged:
        return None
    pull: dict[str, float] = {}
    for sen, _ in above:
        for word in sen:
            x = word_register(index, word)
            if x is not None and index.word(word, "gov") >= REGISTER_PULL_MIN_GOV and x > 0:
                pull[word] = max(pull.get(word, 0.0), x)
    return {"sentences": len(scored), "above": len(above), "pulling": [w for w, _ in sorted(pull.items(), key=lambda kv: -kv[1])[:SHOWN]]}


def directory(work_dir: Path) -> Path:
    return Path(work_dir) / "resources" / "naturalness"
