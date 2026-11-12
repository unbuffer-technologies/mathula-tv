"""How professional translators wrote it: look up an English phrase in the human-translated English-isiZulu pairs we hold.

Sources (all downloaded under working/resources/grammar): Umsuka (about 10.7 k news pairs, professional translators), the Autshumato bilingual corpus (about 233 k lines of government
text, 2022 revision), the 2013 Autshumato release (27 k) and its translation memory.  They are loaded once into a SQLite full-text index (``build``), so a lookup is milliseconds and a
run does not hold 270 k pairs in memory.  A lookup returns the pairs whose English contains the phrase, shortest English first (a short segment is the most direct translation of the
phrase), each with where it came from.  Alignment is imperfect (a pair is sometimes shifted by a line), so a length-ratio filter drops the obvious misses.  This is evidence for the agent
to weigh, never an answer picked for it.  Measured on the English of 8 dubbed jobs: 97% of single words, 76% of two-word phrases, 37% of three-word and 11% of four-word phrases occur.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

GRAMMAR = Path.home() / "mathula-tv" / "working" / "resources" / "grammar"
DEFAULT_DB = GRAMMAR / "human_pairs.sqlite"
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOP = frozenset("the a an of to in and is are was were be been it that this for on with as at by from or but not have has had they we you i he she his her their our your will would can could "
                 "do does did so if than then there which who what when how".split())
RATIO_LOW, RATIO_HIGH = 0.4, 2.0         # Zulu words per English word: outside this a pair is probably shifted by a line


def words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("’", "'"))


@dataclass(frozen=True)
class Pair:
    en: str
    zu: str
    source: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("�", "").strip())


def load_pairs(root: Path = GRAMMAR) -> list[Pair]:
    """Every English-isiZulu pair we hold, de-duplicated."""
    pairs: list[Pair] = []
    for name, flip in (("en-zu.training.csv", False), ("en-zu.eval.csv", False), ("zu-en.training.csv", True)):
        path = root / "umsuka_en_zu" / name
        if path.is_file():
            for row in csv.reader(open(path, encoding="utf-8", errors="replace")):
                if len(row) >= 2 and row[0] not in ("en", "zu"):
                    en, zu = (row[1], row[0]) if flip else (row[0], row[1])
                    pairs.append(Pair(_clean(en), _clean(zu), "Umsuka (news)"))
    z = root / "autshumato_en_zu_parallel_2022" / "Autshumato.BilingualCorpus(English-isiZulu).v2.0.zip"
    if z.is_file():
        zf = zipfile.ZipFile(z)
        zu_name = next(n for n in zf.namelist() if n.endswith(".zu.txt"))
        en_name = next(n for n in zf.namelist() if n.endswith(".en.txt"))          # the release names this file after Xitsonga; it is the English side of the isiZulu corpus
        en_lines = zf.read(en_name).decode("utf-8", "replace").split("\n")
        zu_lines = zf.read(zu_name).decode("utf-8", "replace").split("\n")
        if len(en_lines) == len(zu_lines):
            pairs += [Pair(_clean(e), _clean(u), "Autshumato 2022 (government)") for e, u in zip(en_lines, zu_lines) if e.strip() and u.strip()]
    z = root / "autshumato_en_zu_parallel_2013" / "en-zu.release.zip"
    if z.is_file():
        zf = zipfile.ZipFile(z)
        en_name = next(n for n in zf.namelist() if ".eng." in n and n.endswith(".txt"))
        zu_name = next(n for n in zf.namelist() if ".zul." in n and n.endswith(".txt"))
        en_lines = zf.read(en_name).decode("utf-8", "replace").split("\n")
        zu_lines = zf.read(zu_name).decode("utf-8", "replace").split("\n")
        if len(en_lines) == len(zu_lines):
            pairs += [Pair(_clean(e), _clean(u), "Autshumato 2013 (government)") for e, u in zip(en_lines, zu_lines) if e.strip() and u.strip()]
    z = root / "autshumato_en_zu_tm" / "translationmemory.eng-zul.1.0.0.zip"
    if z.is_file():
        zf = zipfile.ZipFile(z)
        for name in zf.namelist():
            if name.lower().endswith(".tmx"):
                text = zf.read(name).decode("utf-8", "replace")
                for tu in re.findall(r"<tu>(.*?)</tu>", text, flags=re.S):
                    segs = re.findall(r"<tuv lang=\"([a-z-]+)\">\s*<seg>(.*?)</seg>", tu, flags=re.S | re.I)
                    d = {lang[:3].lower(): seg for lang, seg in segs}
                    if "eng" in d and "zul" in d:
                        pairs.append(Pair(_clean(d["eng"]), _clean(d["zul"]), "Autshumato TM (government)"))
    pairs += load_masakhane_pairs(root / ".." / "masakhane")
    pairs += load_dsfsi_pairs(root / ".." / "dsfsi")
    seen, out = set(), []
    for p in pairs:
        key = (p.en.lower(), p.zu.lower())
        if p.en and p.zu and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_masakhane_pairs(root: Path) -> list[Pair]:
    """Human-made English-isiZulu pairs from the Masakhane datasets (``scripts/fetch_masakhane.py`` puts them in ``resources/masakhane``): AfriDocMT sentences (health and IT
    documents), AfriScience-MT (science sentences, plus its term glossary), MAFAND-MT and NTREX (news), InjongoIntent (assistant-style utterances matched to their English by id).  The
    machine-translated sets (african-ultrachat, african-translated-alpaca) are never loaded: they are not evidence of how people write.  A missing folder gives no pairs."""
    root = Path(root)
    pairs: list[Pair] = []
    if not root.is_dir():
        return pairs
    csv.field_size_limit(10**9)
    for path in sorted((root / "AfriDocMT" / "data" / "sentence").rglob("*.csv")) if (root / "AfriDocMT" / "data" / "sentence").is_dir() else []:
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("en") and row.get("zu"):
                    pairs.append(Pair(_clean(row["en"]), _clean(row["zu"]), "AfriDocMT (health and IT documents)"))
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    if pq is not None:
        for path in sorted((root / "afriscience_mt" / "corpus").glob("*.parquet")) if (root / "afriscience_mt" / "corpus").is_dir() else []:
            for row in pq.read_table(path, columns=["lang_pair", "source", "target"]).to_pylist():
                if row["lang_pair"] == "eng-zul" and row["source"] and row["target"]:
                    pairs.append(Pair(_clean(row["source"]), _clean(row["target"]), "AfriScience-MT (science)"))
        glossary = root / "afriscience_mt" / "glossary" / "terms-00000.parquet"
        if glossary.is_file():
            for row in pq.read_table(glossary).to_pylist():
                if row.get("target_lang") == "zul" and row.get("eng") and row.get("target"):
                    pairs.append(Pair(_clean(row["eng"]), _clean(row["target"]), "AfriScience-MT glossary (terms)"))
    for path in sorted((root / "mafand" / "en-zul").glob("*.json")) if (root / "mafand" / "en-zul").is_dir() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                tr = json.loads(line)["translation"]
            except (ValueError, KeyError):
                continue
            if tr.get("en") and tr.get("zul"):
                pairs.append(Pair(_clean(tr["en"]), _clean(tr["zul"]), "MAFAND-MT (news)"))
    ntrex = root / "ntrex_african" / "ntrex_african" / "zul_Latn.tsv"
    if ntrex.is_file():
        for row in csv.DictReader(open(ntrex, encoding="utf-8", newline=""), delimiter="	"):
            if row.get("sentence_eng_Latn") and row.get("sentence_zul_Latn"):
                pairs.append(Pair(_clean(row["sentence_eng_Latn"]), _clean(row["sentence_zul_Latn"]), "NTREX (news)"))
    injongo = root / "InjongoIntent"
    for split in ("train", "dev", "test"):
        zu_path, en_path = injongo / "zul" / f"{split}.jsonl", injongo / "eng" / f"{split}.jsonl"
        if zu_path.is_file() and en_path.is_file():
            english = {}
            for line in en_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                english[row.get("example_id")] = row.get("text")
            for line in zu_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("text") and english.get(row.get("example_id")):
                    pairs.append(Pair(_clean(english[row["example_id"]]), _clean(row["text"]), "InjongoIntent (assistant utterances)"))
    return pairs


ALIGN_MIN = 0.7           # the dsfsi aligned sets carry a similarity score; below this a pair is probably misaligned


def load_dsfsi_pairs(root: Path) -> list[Pair]:
    """Human-made English-isiZulu pairs from the dsfsi (University of Pretoria) sets that ``scripts/fetch_more_zulu.py`` puts in ``resources/dsfsi``: the sentence-aligned Vukuzenzele
    government magazine and cabinet statements (kept when their alignment score is at least ``ALIGN_MIN``), and the za-mafoko terminology lists (each entry's English term against each of
    its isiZulu renderings; "a / b / c" is three pairs, so an alternative is not lost).  A missing folder gives no pairs."""
    root = Path(root)
    pairs: list[Pair] = []
    if not root.is_dir():
        return pairs
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
    if pq is not None:
        for name, label in (("vukuzenzele-sentence-aligned", "Vukuzenzele (government magazine)"), ("govza-sa-cabinet-statements-sentence-aligned", "Cabinet statements (government)")):
            for path in sorted((root / name / "eng-zul").rglob("*.parquet")) if (root / name / "eng-zul").is_dir() else []:
                for row in pq.read_table(path, columns=["eng", "zul", "score"]).to_pylist():
                    try:
                        good = float(row.get("score") or 0) >= ALIGN_MIN
                    except ValueError:
                        good = False
                    if good and row.get("eng") and row.get("zul"):
                        pairs.append(Pair(_clean(row["eng"]), _clean(row["zul"]), label))
    for path in sorted(root.glob("za-mafoko-*/*.jsonl")):
        label = "Term list: " + path.stem.replace("_", " ")[:60]
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            english = _clean(str(row.get("eng") or ""))
            for rendering in str(row.get("zul") or "").split(" / "):
                rendering = _clean(rendering)
                if english and rendering and len(rendering) < 120:
                    pairs.append(Pair(english, rendering, label))
    return pairs


def build(path: Path, pairs: list[Pair]) -> int:
    """Write the pairs into a SQLite database with a full-text index on the English."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE pairs (id INTEGER PRIMARY KEY, en TEXT, zu TEXT, source TEXT, en_words INTEGER, zu_words INTEGER)")
    db.executemany("INSERT INTO pairs (en, zu, source, en_words, zu_words) VALUES (?, ?, ?, ?, ?)",
                   [(p.en, p.zu, p.source, len(words(p.en)), len(p.zu.split())) for p in pairs])
    db.execute("CREATE VIRTUAL TABLE fts USING fts5(en, content='pairs', content_rowid='id', tokenize='unicode61 remove_diacritics 2')")
    db.execute("INSERT INTO fts(fts) VALUES('rebuild')")
    db.execute("CREATE VIRTUAL TABLE fts_zu USING fts5(zu, content='pairs', content_rowid='id', tokenize='unicode61 remove_diacritics 2')")
    db.execute("INSERT INTO fts_zu(fts_zu) VALUES('rebuild')")
    db.commit()
    db.close()
    return len(pairs)


def _query(phrase: str) -> str | None:
    toks = words(phrase)
    return '"' + " ".join(toks).replace('"', "") + '"' if toks else None


class PairsDB:
    def __init__(self, path: Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) if self.path.is_file() else None

    @property
    def available(self) -> bool:
        return self.db is not None

    def count(self, phrase: str) -> int:
        q = _query(phrase)
        if not q or not self.db:
            return 0
        return self.db.execute("SELECT count(*) FROM fts WHERE fts MATCH ?", (q,)).fetchone()[0]

    def lookup(self, phrase: str, limit: int = 5) -> dict:
        """The usable pairs whose English contains the phrase, shortest English first, and how many pairs there are in all."""
        q = _query(phrase)
        if not q:
            return {"phrase": phrase, "pairs_in_all": 0, "pairs": []}
        if not self.db:
            return {"phrase": phrase, "error": "the human-translation pairs are not installed on this machine"}
        rows = self.db.execute("SELECT p.en, p.zu, p.source, p.en_words, p.zu_words FROM fts JOIN pairs p ON p.id = fts.rowid WHERE fts MATCH ? "
                               "ORDER BY p.en_words, p.id LIMIT ?", (q, max(limit * 8, 30))).fetchall()
        total = self.count(phrase)
        out = []
        consensus = self.consensus(phrase, min_pairs=3) if total >= 3 else []
        for en, zu, source, ew, zw in rows:
            if not (RATIO_LOW <= zw / max(ew, 1) <= RATIO_HIGH) or en.isupper():
                continue                                    # a shifted pair, or a heading in capitals
            out.append({"english": en, "isizulu": zu, "source": source})
            if len(out) >= limit:
                break
        return {"phrase": phrase, "pairs_in_all": total, **({"usually_written": consensus} if consensus else {}), "pairs": out}

    def _usable(self, rows: list[tuple]) -> list[tuple]:
        return [r for r in rows if RATIO_LOW <= r[3] / max(r[2], 1) <= RATIO_HIGH and not r[0].isupper()]

    def consensus(self, phrase: str, max_pairs: int = 300, min_pairs: int = 3, min_share: float = 0.35, min_lift: float = 8.0) -> list[dict]:
        """What the Zulu of the pairs that contain the English phrase has in common: the Zulu words or short phrases (1 to 3 words) found in at least ``min_share`` of the usable pairs
        and far more often there than in Zulu text at large (``min_lift``).  A pair that is shifted by a line is simply outvoted.  The longest candidate wins when a shorter one has
        the same support.  Empty when there are too few pairs or nothing is shared: no consensus is not evidence of anything."""
        q = _query(phrase)
        if not q or not self.db:
            return []
        rows = self.db.execute("SELECT p.en, p.zu, p.en_words, p.zu_words FROM fts JOIN pairs p ON p.id = fts.rowid WHERE fts MATCH ? ORDER BY p.en_words, p.id LIMIT ?",
                               (q, max_pairs * 3)).fetchall()
        rows = self._usable(rows)[:max_pairs]
        n = len(rows)
        if n < min_pairs:
            return []
        df: Counter[tuple[str, ...]] = Counter()
        for _, zu, _, _ in rows:
            toks = words(zu)
            df.update({tuple(toks[i:i + k]) for k in (1, 2, 3) for i in range(len(toks) - k + 1)})
        total = self.db.execute("SELECT count(*) FROM pairs").fetchone()[0]
        scored = []
        for gram, c in df.items():
            if c < min_pairs or c / n < min_share:
                continue
            outside = self.db.execute("SELECT count(*) FROM fts_zu WHERE fts_zu MATCH ?", ('"' + " ".join(gram) + '"',)).fetchone()[0]
            lift = (c / n) / max(outside / total, 1 / total)
            if lift >= min_lift:
                scored.append((gram, c, lift))
        keep = []
        for gram, c, lift in sorted(scored, key=lambda t: (-len(t[0]), -t[1])):
            if not any(set(gram) < set(k[0]) and c <= k[1] * 1.1 for k in keep):
                keep.append((gram, c, lift))
        keep.sort(key=lambda t: (-t[1], -len(t[0])))
        return [{"isizulu": " ".join(g), "in_pairs": c, "of": n} for g, c, _ in keep[:3]]

    def hints(self, segments: list[tuple[str, str]], max_total: int = 10, per_segment: int = 2) -> list[dict]:
        """Evidence for the agent, kept short: for a short stock line, how the human pairs translate that exact line; for a longer segment, the 2 to 4 word content phrases whose Zulu has a
        clear consensus.  Nothing is shown for a phrase with no consensus (a shifted pair or a generic phrase proves nothing) or for single words (attested_zulu covers those)."""
        out: list[dict] = []
        stock: list[dict] = []
        for seg_id, text in segments:
            toks = words(text)
            if len(toks) < 2:
                continue
            if len(toks) <= 5:
                found = self.lookup(" ".join(toks), limit=40)
                exact = [p for p in found.get("pairs", []) if words(p["english"]) == toks][:2]
                cons = self.consensus(" ".join(toks), min_pairs=2)
                if exact or cons:
                    stock.append({"segment_id": seg_id, "phrase": " ".join(toks), "usually_written": cons, "exact_pairs": exact})
                continue
            shown = 0
            for n in (4, 3, 2):
                for i in range(len(toks) - n + 1):
                    g = tuple(toks[i:i + n])
                    if g[0] in STOP or g[-1] in STOP or any(w.isdigit() for w in g):
                        continue
                    cons = self.consensus(" ".join(g), min_pairs=5, min_share=0.4)            # strict: a hint must be worth its tokens
                    if cons:
                        found = self.lookup(" ".join(g), limit=1)
                        out.append({"segment_id": seg_id, "phrase": " ".join(g), "usually_written": cons, "example": (found.get("pairs") or [None])[0]})
                        shown += 1
                        break
                if shown >= per_segment:
                    break
        return (stock + out)[:max_total]
