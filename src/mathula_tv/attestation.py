"""Attestation: what real written isiZulu does with a word.

A model produces fluent Zulu, and it also produces forms nobody writes (the glossary made "Fighters" into "maFighter"); nothing inside the model says "I
have never seen this".  This module is the check the model cannot give itself: a word is attested when real text has it.  The ladder it serves is the
user's rule for a word the system does not know what to do with: use an attested Zulu word or loan if there is one, otherwise code-switch (the English
word as the speaker says it).  "Knowing" is attestation, not a model's confidence.

Resources (all downloaded from the SADiLaR repository, repo.sadilar.org, licences CC BY 2.5 ZA and CC BY 4.0; credit CTexT, North-West University, and the
Department of Arts and Culture):

* the Autshumato English-isiZulu parallel corpus: 233,691 aligned sentence pairs, 2.87 million Zulu tokens;
* the NCHLT isiZulu lexicon files (nouns, verbs, lemma list): 43,178 entries.

A third source, in ``web/``, is Zulu written on the web (FineWeb-2 zul_Latn, ODC-By 1.0: 127,335 documents, 56.8 million words, Common Crawl
2013-2024), counted apart for Zulu news sites; ``build-web`` makes it from the downloaded file.  It shows how real writers spell an English word in
Zulu text (irally 5, i-rally 4 on news sites) that the translated corpus is too small to show.

They are plain text in ``<work_dir>/resources/attestation/``: ``autshumato.zu.txt`` and ``autshumato.en.txt`` (one sentence per line, the two files
aligned line by line) and ``lexicon/*.txt`` (one entry per line, or token<TAB>lemma).  Nothing here calls a model.
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")
_EN_TOKEN = re.compile(r"[a-z][a-z']*")
TRANSLATES_MIN_SHARED = 5          # aligned sentences a Zulu stem must share with an English word
TRANSLATES_MIN_DICE = 0.25         # and the Dice score of the pair (commission/ikhomishana 0.35, english/isingisi 0.29, sergeant/usayitsheni 0.67; a frequent word such as president beside a name stays far below)
TRANSLATES_MIN_CORE = 5            # letters a capitalised core needs before it is looked up
ORDINARY_MIN = 20                # a prefixed place name written this often in real text is ordinary Zulu
ORDINARY_MIN_NO_ENGLISH_STEM = 5     # the same when the place's stem (Nkandla) is not an English word
ORDINARY_COMMON = 100            # a lower-case form this common is an everyday Zulu word
MIN_PAIRS = 3                    # a Zulu form must share at least this many sentences with the English word to count as its equivalent
TRANSLATION_TOP = 5              # forms kept per English word when the dub is aligned to the speaker (word_alignment)
TRANSLATION_MIN_SCORE = 0.10     # and their Dice score (speaker/somlomo .47, lawyer/ummeli .27, signature/ukusayina .23, initials/osonhlamvukazi .13; noise stays under .02)
TRANSLATION_MAX_LINES = 20000    # a word in more sentences than this is a function word


WEB_MIN = 2                      # a form written this often on Zulu news sites is attested (one occurrence may be a typo)
WEB_NEWS_DOMAINS = frozenset({
    "isolezwe.co.za", "iol.co.za", "izindaba.scrolla.africa", "bayedenews.com", "news24.com", "isizulu.news24.com", "impempe.com", "ilanganews.co.za",
    "pressreader.com", "ndabaonline.ukzn.ac.za", "umbele.co.za"})
_WEB_TOKEN = re.compile(r"[A-Za-z](?:[A-Za-z'\-]*[A-Za-z])?")
_ZULU_PARTICLES = "i|ama|izi|izin|isi|imi|aba|ne|nge|nga|we|ye|kwi|kwe|yama|ngama|le|se|e|ku|ka|o|u|a|ba"


def resources_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "resources" / "attestation"


def tokens(line: str) -> list[str]:
    """Lowercase word tokens; a hyphen splits ("i-Tshela" -> i, tshela), so a particle before an English word is counted apart from it."""
    return [t.lower() for t in _TOKEN.findall(line)]


class WebText:
    """Zulu written on the web (FineWeb-2 zul_Latn: 127,335 documents, 56.8 million words, Common Crawl 2013-2024, ODC-By 1.0), counted apart for Zulu
    news sites and for everything else, because the rest includes machine-translated product and tech pages that prove less."""

    def __init__(self, counts: dict[str, tuple[int, int]], news_text: str = "") -> None:
        self.counts = counts
        self._news = news_text.lower()

    @classmethod
    def from_directory(cls, directory: Path) -> "WebText":
        counts: dict[str, tuple[int, int]] = {}
        for line in (Path(directory) / "web_counts.tsv").read_text(encoding="utf-8").splitlines():
            token, news, other = line.split("\t")
            counts[token] = (int(news), int(other))
        news_file = Path(directory) / "web_news.txt"
        return cls(counts, news_file.read_text(encoding="utf-8") if news_file.is_file() else "")

    def written(self, form: str) -> tuple[int, int]:
        """(news, other): how often Zulu web text writes this form; a phrase is counted in the news text only."""
        key = " ".join(form.lower().split())
        if " " not in key:
            return self.counts.get(key, (0, 0))
        found = len(re.findall(r"(?<![a-z'])" + re.escape(key) + r"(?![a-z'])", self._news)) if self._news else 0
        return found, 0

    def forms_with(self, english_word: str, top: int = 6) -> list[dict[str, Any]]:
        """How Zulu news sites write an English word: bare, with a Zulu particle (irally), or with a hyphenated one (i-rally, kwi-rally)."""
        word = re.escape(english_word.lower().strip())
        if not self._news or not word:
            return []
        pattern = re.compile(r"(?<![a-z'])([a-z]{1,6}-" + word + r"|(?:" + _ZULU_PARTICLES + r")?" + word + r")(?:s|es)?(?![a-z'])")
        found = Counter(match.group(0) for match in pattern.finditer(self._news))
        return [{"form": form, "count": count} for form, count in found.most_common(top)]


def build_web_index(documents: Path | Sequence[Path], out_dir: Path) -> dict[str, int]:
    """Make the web resources from ``host<TAB>date<TAB>text`` files: ``web_counts.tsv`` (token, news count, other count; tokens seen at least twice, a
    hyphenated token counted whole and by its parts) and ``web_news.txt`` (one news document per line).  A document whose first 100 letters have been
    seen already (the same story in two sources) is counted once."""
    news: Counter[str] = Counter()
    other: Counter[str] = Counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    documents_in_news = 0
    duplicates = 0
    seen: set[str] = set()
    files = [Path(documents)] if isinstance(documents, (str, Path)) else [Path(d) for d in documents]
    with (out_dir / "web_news.txt").open("w", encoding="utf-8", newline="\n") as news_file:
        for path in files:
            for line in path.open(encoding="utf-8"):
                host, _date, text = line.rstrip("\n").split("\t", 2)
                is_news = host in WEB_NEWS_DOMAINS
                if is_news:
                    key = re.sub(r"[^a-z]", "", text[:160].lower())[:100]
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                target = news if is_news else other
                for token in _WEB_TOKEN.findall(text):
                    low = token.lower()
                    target[low] += 1
                    if "-" in low:
                        target.update(part for part in low.split("-") if part)
                if is_news:
                    news_file.write(text + "\n")
                    documents_in_news += 1
    kept = [token for token in set(news) | set(other) if news[token] + other[token] >= 2]
    with (out_dir / "web_counts.tsv").open("w", encoding="utf-8", newline="\n") as counts_file:
        for token in sorted(kept):
            counts_file.write(f"{token}\t{news[token]}\t{other[token]}\n")
    return {"tokens": len(kept), "news_documents": documents_in_news, "duplicates_skipped": duplicates}


class Attestation:
    def __init__(self, zulu_lines: Sequence[str], english_lines: Sequence[str], lexicon: Iterable[str] = (), web: WebText | None = None) -> None:
        self.web = web
        if len(zulu_lines) != len(english_lines):
            raise ValueError(f"the two sides must be aligned line by line ({len(zulu_lines)} against {len(english_lines)})")
        self.zulu_lines = list(zulu_lines)
        self.frequency: Counter[str] = Counter()
        self.document_frequency: Counter[str] = Counter()
        for line in self.zulu_lines:
            found = tokens(line)
            self.frequency.update(found)
            self.document_frequency.update(set(found))
        lowered = [line.lower() for line in english_lines]
        self._en_lines = lowered
        self._english = "\n".join(lowered)
        self._starts: list[int] = []
        position = 0
        for line in lowered:
            self._starts.append(position)
            position += len(line) + 1
        self.lexicon = {entry.lower() for entry in lexicon if entry}
        self._zulu_text = "\n".join(self.zulu_lines)
        self._zulu_lower = self._zulu_text.lower()
        self._zulu_line_starts: set[int] = set()
        position = 0
        for line in self.zulu_lines:
            self._zulu_line_starts.add(position)
            position += len(line) + 1

    @classmethod
    def from_directory(cls, directory: Path) -> "Attestation":
        directory = Path(directory)
        zulu = (directory / "autshumato.zu.txt").read_text(encoding="utf-8").split("\n")
        english = (directory / "autshumato.en.txt").read_text(encoding="utf-8").split("\n")
        size = min(len(zulu), len(english))
        entries: list[str] = []
        for path in sorted((directory / "lexicon").glob("*.txt")) if (directory / "lexicon").is_dir() else []:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                entries.extend(part.strip() for part in line.split("\t") if part.strip())
        web = WebText.from_directory(directory / "web") if (directory / "web" / "web_counts.tsv").is_file() else None
        return cls(zulu[:size], english[:size], entries, web)

    # -- a Zulu form -----------------------------------------------------------------
    def status(self, form: str) -> dict[str, Any]:
        """How often real text writes this exact form, and whether the lexicon knows it.  A form of several words ("isikhwama sempesheni") or with a
        hyphenated particle ("se-provident") is counted as written, exactly, and each of its words is counted on its own."""
        key = " ".join(form.lower().split())
        parts = [p for p in re.split(r"[\s\-]+", key) if p]
        if len(parts) <= 1 and "-" not in key:
            answer = {"token": form, "corpus_count": self.frequency.get(key, 0), "in_lexicon": key in self.lexicon}
            return self._with_web(answer, key)
        written = len(re.findall(r"(?<![a-z'])" + re.escape(key) + r"(?![a-z'])", self._zulu_lower)) if key else 0
        words = [self._with_web({"token": p, "corpus_count": self.frequency.get(p, 0), "in_lexicon": p in self.lexicon}, p) for p in parts]
        answer = {"token": form, "corpus_count": written, "in_lexicon": False, "words": words,
                  "all_words_attested": bool(words) and all(w["corpus_count"] > 0 or w["in_lexicon"] or w.get("web_news", 0) >= WEB_MIN for w in words)}
        return self._with_web(answer, key)

    def _with_web(self, answer: dict[str, Any], key: str) -> dict[str, Any]:
        if self.web is not None and key:
            news, other = self.web.written(key)
            answer["web_news"], answer["web_other"] = news, other
        return answer

    def attested(self, form: str) -> bool:
        """A single form is attested when real text writes it or the lexicon has it; a phrase when real text writes the phrase or every one of its
        words is attested (the parts of a new combination are real words)."""
        found = self.status(form)
        web = found.get("web_news", 0) >= WEB_MIN
        if "words" in found:
            return found["corpus_count"] > 0 or web or found["all_words_attested"]
        return found["corpus_count"] > 0 or found["in_lexicon"] or web

    def written_count(self, token: str) -> int:
        """How many times real text writes exactly this form, with this capitalisation (a capital at the start of a sentence proves nothing and is not
        counted): "eMpumalanga" 115 times, "Tshela" 12 (the verb), "Imali" 146 in mid-sentence."""
        pattern = re.compile(r"(?<![A-Za-z'])" + re.escape(token) + r"(?![A-Za-z'])")
        count = 0
        for match in pattern.finditer(self._zulu_text):
            if token[:1].isupper() and match.start() in self._zulu_line_starts:
                continue
            count += 1
        return count

    def ordinary_zulu(self, token: str, english_has: Callable[[str], bool] | None = None) -> bool:
        """Whether this written form is ordinary Zulu that the Zulu voice reads as Zulu, and not a foreign word to be matched to the speaker's sounds:
        a place written with a Zulu locative prefix that real text writes often (eMpumalanga, eGauteng), or a word whose lower-case form is very common
        and whose capitalised form is only a sentence start (Imali).  A capitalised name that real text rarely writes that way (Tshela: a Zulu verb
        in lower case, a Sesotho name here) is not.

        ``english_has`` says whether a word is in an English dictionary.  A prefixed place whose stem is no English word (Nkandla, the home of a former
        president, written "eNkandla" 11 times in a small corpus) needs far fewer sightings: nothing English is competing for it."""
        if re.match(r"^(?:e|ku|kwa|ase|eku)[A-Z][a-z]", token):
            minimum = ORDINARY_MIN
            if english_has is not None and not english_has(re.sub(r"^(?:eku|ase|kwa|ku|e)(?=[A-Z])", "", token)):
                minimum = ORDINARY_MIN_NO_ENGLISH_STEM
            return self.written_count(token) >= minimum
        lower = token[:1].lower() + token[1:]
        common = self.written_count(lower)
        return token[:1].isupper() and common >= ORDINARY_COMMON and self.written_count(token) <= 0.2 * common

    # -- a Zulu form that translates a word of the English --------------------------------
    def _zulu_lines_holding(self, core: str) -> list[int]:
        """The aligned lines whose Zulu side holds a word that contains ``core`` with a few letters of grammar round it (a noun-class prefix, a locative, a plural)."""
        if not hasattr(self, "_zulu_starts_sorted"):
            self._zulu_starts_sorted = sorted(self._zulu_line_starts)
        pattern = re.compile(r"(?<![a-z'])[a-z]{0,5}" + re.escape(core) + r"[a-z]{0,4}(?![a-z'])")
        found: list[int] = []
        last = -1
        for match in pattern.finditer(self._zulu_lower):
            index = bisect.bisect_right(self._zulu_starts_sorted, match.start()) - 1
            if index != last:
                found.append(index)
                last = index
        return found

    def english_associates(self, core: str, *, min_shared: int = TRANSLATES_MIN_SHARED, min_dice: float = TRANSLATES_MIN_DICE) -> dict[str, float]:
        """The English words that share the aligned sentences of a Zulu stem (a Dice score, so common function words do not win): ``{"commission": 0.35}`` for ikhomishana.  Empty
        when the stem is too rare to say anything."""
        cache = self.__dict__.setdefault("_associates", {})
        if core in cache:
            return cache[core]
        lines = self._zulu_lines_holding(core)
        answer: dict[str, float] = {}
        if len(lines) >= min_shared:
            if not hasattr(self, "_en_df"):
                self._en_df = Counter()
                for line in self._en_lines:
                    self._en_df.update(set(_EN_TOKEN.findall(line)))
            together: Counter[str] = Counter()
            for index in lines:
                together.update(set(_EN_TOKEN.findall(self._en_lines[index])))
            for word, shared in together.items():
                if shared >= min_shared:
                    dice = 2 * shared / (len(lines) + self._en_df[word])
                    if dice >= min_dice:
                        answer[word] = round(dice, 3)
        cache[core] = answer
        return answer

    def translates(self, zulu_token: str, english_source: str) -> bool:
        """Does real parallel text write this Zulu word where the English has a word of ``english_source``?  iKhomishana translates "Commission", isiNgisi "English", uSayitsheni "Sergeant":
        capitalised Zulu words that are not names and have nothing to do with an invented one.  An invented name has no aligned sentences (or only the weak, frequent-word co-occurrences that
        the Dice score rejects)."""
        token = str(zulu_token or "")
        token = token.split("-")[-1]
        upper_at = next((i for i, ch in enumerate(token) if ch.isupper()), 0)
        core = token[upper_at:].lower()
        if len(core) < TRANSLATES_MIN_CORE:
            return False
        words = set(_EN_TOKEN.findall(str(english_source or "").lower()))
        words |= {w[:-1] for w in words if w.endswith("s")} | {w + "s" for w in words}
        # the stem as written, then shortened by the grammar that can follow it: eKhomishaneni (locative -ni) has no sentence in the corpus, its stem khomishan has 388
        for variant in [core] + [core[:-k] for k in (2, 3, 4) if len(core) - k >= TRANSLATES_MIN_CORE]:
            associates = self.english_associates(variant)
            if not associates:
                continue
            if any(word[:5] == variant[:5] for word in associates):          # the English side writes the same word (Ramaphosa, Novemba/November): a name or a localised spelling, which the gate's own rules judge, never a translation
                return False
            if any(word in words for word in associates):
                return True
        return False

    # -- an English word ---------------------------------------------------------------
    def _lines_with(self, word: str) -> list[int]:
        pattern = re.compile(r"(?<![a-z'])" + re.escape(word.lower()) + r"(?:s|es)?(?![a-z'])")
        found: list[int] = []
        last = -1
        for match in pattern.finditer(self._english):
            index = bisect.bisect_right(self._starts, match.start()) - 1
            if index != last:
                found.append(index)
                last = index
        return found

    def _english_index(self) -> dict[str, list[int]]:
        """English word -> the aligned lines that have it (built once, about two seconds; the regex scan of ``_lines_with`` takes a quarter of a second per word)."""
        if not hasattr(self, "_en_index"):
            index: dict[str, list[int]] = {}
            for number, line in enumerate(self._en_lines):
                for word in set(_EN_TOKEN.findall(line)):
                    index.setdefault(word, []).append(number)
            self._en_index = index
        return self._en_index

    def translation_forms(self, english_word: str, *, top: int = TRANSLATION_TOP, min_score: float = TRANSLATION_MIN_SCORE) -> list[tuple[str, float]]:
        """The Zulu forms real parallel text writes where the English has this word, as (form, Dice score) best first: ``speaker`` -> somlomo .47, usomlomo .35.  The same evidence as
        ``equivalents`` without the report around it, from an index, so the dub's words can be aligned to the speaker's by it.  Empty for a word the corpus has too few or too many times."""
        word = str(english_word or "").lower().strip()
        cache = self.__dict__.setdefault("_translation_forms", {})
        if word in cache:
            return cache[word]
        index = self._english_index()
        lines = sorted(set(index.get(word, ())) | set(index.get(word + "s", ())) | set(index.get(word + "es", ())))
        answer: list[tuple[str, float]] = []
        if MIN_PAIRS <= len(lines) <= TRANSLATION_MAX_LINES:
            together: Counter[str] = Counter()
            for number in lines:
                together.update(set(tokens(self.zulu_lines[number])))
            scored = sorted(((2 * pairs / (len(lines) + self.document_frequency[form]), form) for form, pairs in together.items()
                             if pairs >= MIN_PAIRS and form not in (word, word + "s", word + "es")), reverse=True)
            answer = [(form, round(score, 3)) for score, form in scored[:top] if score >= min_score]
        cache[word] = answer
        return answer

    def equivalents(self, english_word: str, *, top: int = 8) -> dict[str, Any]:
        """What the Zulu side of the corpus writes where the English side has this word: the Zulu forms that share the most sentences with it (a Dice
        score, so common function words do not win), and how often the English word itself stays in the Zulu sentence (code-switching)."""
        word = english_word.lower().strip()
        lines = self._lines_with(word)
        result: dict[str, Any] = {"english": english_word, "sentences": len(lines), "forms": [], "kept_in_english": 0, "with_particle": []}
        if not lines:
            return result
        together: Counter[str] = Counter()
        kept = 0
        word_forms = {word, word + "s", word + "es"}
        particle = re.compile(r"(?<![A-Za-z'])([A-Za-z]{1,6}-" + re.escape(word) + r"(?:s|es)?)(?![A-Za-z'])", re.IGNORECASE)
        particles: Counter[str] = Counter()
        for index in lines:
            found = set(tokens(self.zulu_lines[index]))
            if found & word_forms:
                kept += 1
            together.update(found)
            particles.update(m.lower() for m in particle.findall(self.zulu_lines[index]))
        result["with_particle"] = [{"form": form, "count": count} for form, count in particles.most_common(5)]
        scored = []
        for form, pairs in together.items():
            if pairs < MIN_PAIRS or form in word_forms:
                continue
            scored.append((2 * pairs / (len(lines) + self.document_frequency[form]), form, pairs))
        scored.sort(reverse=True)
        result["kept_in_english"] = kept
        result["forms"] = [{"form": form, "shared_sentences": pairs, "score": round(score, 3), "corpus_count": self.frequency[form]}
                           for score, form, pairs in scored[:top]]
        return result


def lookup(attestation: Attestation, english_word: str, zulu_forms: Sequence[str] = ()) -> dict[str, Any]:
    """The answer for one English word, shaped for the agent: the Zulu forms real text pairs with it, whether it stays English there, and the status
    of any form the agent is thinking of writing."""
    answer = attestation.equivalents(english_word)
    answer["forms"] = [f for f in answer["forms"] if f["score"] >= MIN_FORM_SCORE]            # a function word (i, ukuthi) shares sentences with everything; its score is ~0 and it only crowds the answer
    answer["checked_forms"] = [attestation.status(form) for form in zulu_forms]
    web = attestation.web
    answer["web_forms"] = web.forms_with(english_word) if web is not None else []
    if web is None:
        answer["caution"] = ("the corpus is translated government and news text, 2.9 million words: a form missing from it is weak evidence, not proof that nobody writes it"
                             + ("; only %d sentences here have the word" % answer["sentences"] if answer["sentences"] < 50 else ""))
    else:
        answer["caution"] = ("evidence is translated government text (2.9 million words) and Zulu news sites 2013-2024 (isolezwe, news24, scrolla, bayede, ilanga, impempe); "
                             "web_news counts only those sites, web_other is everything else on the Zulu web, some of it machine translated; a form missing from both is weak evidence")
    sentences, kept = answer["sentences"], answer["kept_in_english"]
    strong = [f for f in answer["forms"] if f["score"] >= 0.1][:3]
    if not sentences:
        answer["reading"] = ("the English word does not occur in the translated corpus" if answer["web_forms"]
                             else "the English word does not occur in the corpus: nothing is attested; keep it as the speaker says it")
    elif strong:
        answer["reading"] = ("real text pairs it with " + ", ".join(f"{f['form']!r} ({f['shared_sentences']} of {sentences} sentences)" for f in strong)
                             + (f"; the English word itself stays in {kept} of them" if kept else ""))
    elif kept and kept * 5 >= sentences:
        answer["reading"] = f"no Zulu form clearly pairs with it, and it stays English in {kept} of {sentences} sentences: real text code-switches it"
    elif answer["web_forms"]:
        answer["reading"] = (f"the translated corpus has no Zulu form that clearly pairs with it ({kept} of {sentences} sentences keep the English); "
                             "a form written at least twice by Zulu news sites is attested")
    else:
        answer["reading"] = (f"no Zulu form clearly pairs with it and it is rarely kept in English ({kept} of {sentences} sentences): nothing is attested; "
                             "keep it as the speaker says it")
    if sentences and answer["with_particle"]:
        answer["reading"] += "; written with a hyphenated particle as " + ", ".join(f"{p['form']} ({p['count']})" for p in answer["with_particle"][:3])
    if answer["web_forms"]:
        answer["reading"] += "; Zulu news sites write it as " + ", ".join(f"{p['form']} ({p['count']})" for p in answer["web_forms"][:4])
    return answer


MIN_FORM_SCORE = 0.02
_cache: dict[str, Attestation] = {}


def load(work_dir: Path) -> Attestation | None:
    """The resources for this work directory, loaded once per process; None when they have not been put there."""
    directory = resources_dir(work_dir)
    key = str(directory)
    if key not in _cache:
        if not (directory / "autshumato.zu.txt").is_file():
            return None
        _cache[key] = Attestation.from_directory(directory)
    return _cache[key]


def zulu_guard(work_dir: Path) -> Callable[[str], bool]:
    """Whether a written token is ordinary Zulu that must not be respelled to sound like English: the attestation's rule, with the English dictionary
    (``resources/cmudict``) telling whether the stem of a prefixed place is an English word.  False for everything when the resources are not installed."""
    resources = load(work_dir)
    if resources is None:
        return lambda token: False
    from .english_sounds import EnglishSounds

    sounds = EnglishSounds.load(Path(work_dir) / "resources" / "cmudict" / "cmudict.dict")
    english_has = (lambda word: bool(sounds.word(word))) if sounds is not None else None
    answers: dict[str, bool] = {}             # counting a form in the whole corpus is slow and the workbench asks about every memory entry on every tool call

    def guard(token: str) -> bool:
        if token not in answers:
            answers[token] = bool(resources.ordinary_zulu(token, english_has=english_has))
        return answers[token]

    return guard


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.attestation")
    sub = parser.add_subparsers(dest="command", required=True)
    form = sub.add_parser("form", help="how often real text writes an isiZulu form")
    form.add_argument("token", nargs="+")
    english = sub.add_parser("english", help="what real text writes where the English side has this word")
    english.add_argument("word")
    english.add_argument("--zulu", nargs="*", default=[], help="forms to check as well")
    web = sub.add_parser("build-web", help="make the web resources from host<TAB>date<TAB>text lines")
    web.add_argument("documents", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "build-web":
        print(build_web_index(args.documents, resources_dir(load_settings().work_dir) / "web"))
        return 0
    resources = load(load_settings().work_dir)
    if resources is None:
        print("no resources in", resources_dir(load_settings().work_dir))
        return 1
    if args.command == "form":
        for token in args.token:
            print(json.dumps(resources.status(token), ensure_ascii=False))
    else:
        print(json.dumps(lookup(resources, args.word, args.zulu), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
