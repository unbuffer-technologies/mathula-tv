"""The stems of isiZulu words, from the ZulMorph analyser, so that "the same word" is decided by morphology and not by how the letters happen to line up.

isiZulu words carry their grammar on both sides of the stem (isu, amasu, ngamasu, lesisu all hold the stem "su"; ilungelo and lelilungelo hold "lungelo"), so asking whether a unit
still uses a word the agent settled on cannot be answered by looking for the letters.  ZulMorph (Pretorius/Bosch) analyses a word into prefixes and a stem, e.g.
``ngamasu -> nga[AdvPre]a[NPrePre][6]ma[BPre][6]su.5-6[NStem]``; the stem is what two forms of one word share.

ZulMorph is an optional dependency (see zulu_morphology).  It is used in this process when it loads; otherwise through the interpreter of its own environment
(``MATHULA_ZULMORPH_PYTHON``, the one the grammar scripts use on the server), once for a whole batch of words.  When neither works every function here returns the evidence it has
and says so, and the caller falls back to the plain forms: nothing is ever invented.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_STEM = re.compile(r"([A-Za-z']+)(?:\.[0-9\-]+)?\[\w*Stem\]")
_WORD = re.compile(r"[A-Za-z']+")
_CACHE: dict[str, list[str]] = {}
_HELPER = r"""
import json, sys, resource
try:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))          # the analyser's memory grows word by word
except Exception:
    pass
import importlib.resources as resources
path = str(resources.files("zulmorph.res").joinpath("zul.fom"))
try:
    import zulmorph.zulmorph as zm
    fst = zm.load_fst(path)
except Exception:
    from foma_with_lib.foma import FST
    fst = FST.load(path)
out = {}
for w in json.load(sys.stdin):
    found = []
    try:
        for analysis in fst.apply_up(w):
            found.append(analysis)
            if len(found) >= 6:
                break
    except Exception:
        pass
    out[w] = found
json.dump(out, sys.stdout)
"""
HELPER_BATCH = 150          # words to one helper process: a fresh process takes the next batch


def words_of(text: str) -> list[str]:
    """The word forms of a text, lower case, a hyphenated prefix split off (ama-details -> ama, details)."""
    return [w.lower() for w in _WORD.findall(str(text or ""))]


def extract_stems(analyses: Iterable[str]) -> set[str]:
    """The stems named in ZulMorph analyses (``...su.5-6[NStem]`` -> su; verbs and relatives too)."""
    return {m.group(1).lower() for analysis in analyses for m in _STEM.finditer(analysis)}


def _in_process(words: Sequence[str]) -> dict[str, list[str]] | None:
    try:
        from . import zulu_morphology

        if not zulu_morphology.is_available():
            return None
        fst = zulu_morphology._load_fst()
        return {w: list(fst.apply_up(w))[:6] for w in words}
    except Exception:                                             # noqa: BLE001 - an optional analyser must never stop a run
        return None


def _through_helper(words: Sequence[str]) -> dict[str, list[str]] | None:
    found: dict[str, list[str]] = {}
    for start in range(0, len(words), HELPER_BATCH):
        part = _helper_once(words[start:start + HELPER_BATCH])
        if part is None:
            return found or None
        found.update(part)
    return found


def _helper_once(words: Sequence[str]) -> dict[str, list[str]] | None:
    python = os.environ.get("MATHULA_ZULMORPH_PYTHON") or str(Path.home() / "sadilar" / "venv_zulmorph" / "bin" / "python")
    if not Path(python).is_file():
        return None
    library = Path.home() / "sadilar" / "libfoma" / "usr" / "lib" / "x86_64-linux-gnu"
    env = dict(os.environ)
    if (library / "libfoma.so.0").is_file():
        env["LD_LIBRARY_PATH"] = f"{library}:{env.get('LD_LIBRARY_PATH', '')}"
        env["LD_PRELOAD"] = str(library / "libfoma.so.0")
    try:
        done = subprocess.run([python, "-c", _HELPER], input=json.dumps(list(words)), capture_output=True, text=True, timeout=60, env=env)
        return json.loads(done.stdout) if done.returncode == 0 and done.stdout else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def analyse(words: Iterable[str]) -> dict[str, list[str]]:
    """ZulMorph's analyses for the words (those it could not be asked about are absent).  Cached for the life of the process; one call for the whole batch."""
    wanted = sorted({w for w in words if w and w not in _CACHE})
    if wanted:
        got = _in_process(wanted) or _through_helper(wanted) or {}
        _CACHE.update(got)
    return {w: _CACHE[w] for w in words if w in _CACHE}


def available() -> bool:
    return bool(analyse(["isu"]).get("isu"))


def status() -> dict[str, object]:
    """How the analyser is reached from this process, tested with a word it must parse: ``{"available": True, "via": "in_process" | "helper"}`` or ``{"available": False, "via": None}``.
    Printed at the start of every agent run, so a run that would compare plain forms (no analyser) says so up front and not only in its results."""
    _CACHE.pop("isu", None)
    if (_in_process(["isu"]) or {}).get("isu"):
        return {"available": True, "via": "in_process"}
    if (_through_helper(["isu"]) or {}).get("isu"):
        return {"available": True, "via": "helper"}
    return {"available": False, "via": None}


def stems_of(word: str, analyses: Mapping[str, Sequence[str]]) -> set[str]:
    """The stems a form can have: ZulMorph's, and the form itself (a word it does not parse, or one standing without its prefix, "su" after "lelo", is its own stem)."""
    word = word.lower()
    return {word} | extract_stems(analyses.get(word) or ())


def holds(term: str, text: str, analyses: Mapping[str, Sequence[str]]) -> bool:
    """Does the text use the word (or every word of the phrase) the term names, in whatever prefixed form?  Compared by stem: each word of the term must share a stem with some
    word of the text.  With no analyses at all this is the plain-form comparison, which is the honest limit of the evidence."""
    wanted = words_of(term)
    if not wanted:
        return False
    present: set[str] = set()
    for word in words_of(text):
        present |= stems_of(word, analyses)
    return all(stems_of(w, analyses) & present for w in wanted)
