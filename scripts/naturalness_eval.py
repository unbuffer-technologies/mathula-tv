"""Build the word-pair index and measure what its evidence is worth, before the agent is shown any of it.

    cd ~/mathula-tv && set -a && . ./.env && set +a && .venv/bin/python scripts/naturalness_eval.py [--final]

Without --final: builds an index from everything EXCEPT a held-out sample, then reports, for several thresholds, how often the check fires on
  * held-out real spoken sentences, held-out real news sentences, held-out real government sentences (these should rarely fire: the false-alarm rate), and
  * deliberately corrupted spoken sentences (two adjacent words swapped, one word replaced by a random common word, all words shuffled: these should fire),
  * the agent's own submissions (prod-89, prod-91), with the pairs and words it would have shown.
There were no human quality labels to test against (0 verdicts across the 16 clips), so this is the honest test available: the check can tell broken word order and wrong collocations from
real Zulu, at a stated false-alarm rate.  It does not show that the check finds stiffness, and it does not read meaning.
With --final: builds the index from everything and writes it to <work_dir>/resources/naturalness/ngrams.npz.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mathula_tv import naturalness  # noqa: E402
from mathula_tv.config import load_settings  # noqa: E402

SEED = 7
HOLD_SPOKEN, HOLD_NEWS, HOLD_GOV = 2000, 120, 2000


def read_sources(work_dir: Path) -> dict[str, list[str]]:
    resources = work_dir / "resources"
    spoken = [json.loads(line).get("transcript", "") for line in (resources / "spoken" / "anv_zul.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    news = (resources / "attestation" / "web" / "web_news.txt").read_text(encoding="utf-8").splitlines()
    gov = (resources / "attestation" / "autshumato.zu.txt").read_text(encoding="utf-8").splitlines()
    return {"spoken": [t for t in spoken if t], "news": [t for t in news if t.strip()], "gov": [t for t in gov if t.strip()]}


def long_sentences(texts: list[str], minimum: int = 5) -> list[list[str]]:
    return [s for t in texts for s in naturalness.sentences(t) if len(s) >= minimum]


def corrupt(sentences: list[list[str]], kind: str, common: list[str], rng: random.Random) -> list[list[str]]:
    out = []
    for s in sentences:
        s = list(s)
        if kind == "swap_adjacent":
            i = rng.randrange(len(s) - 1)
            s[i], s[i + 1] = s[i + 1], s[i]
        elif kind == "replace_word":
            s[rng.randrange(len(s))] = rng.choice(common)
        else:
            rng.shuffle(s)
        out.append(s)
    return out


def rates(index: naturalness.NgramIndex, sentences: list[list[str]], expected_min: float, min_word: int) -> dict[str, float]:
    pairs = bookish = 0
    for s in sentences:
        found = naturalness.check(index, " ".join(s) + ".", min_word=min_word, expected_min=expected_min)
        pairs += bool(found["unseen_pairs"])
        bookish += bool(found["bookish_words"])
    n = max(1, len(sentences))
    return {"pairs": 100 * pairs / n, "bookish": 100 * bookish / n, "n": len(sentences)}


def units_of(texts: list[str], size: int = 3, limit: int = 1500) -> list[str]:
    """Held-out units: ``size`` consecutive sentences of one text (a transcript, a news document, a run of government lines), joined into a unit the way the agent's units are."""
    out = []
    for t in texts:
        s = [x for x in naturalness.sentences(t) if len(x) >= 3]
        for i in range(0, len(s) - size + 1, size):
            out.append(". ".join(" ".join(x) for x in s[i:i + size]))
    return out[:limit]


def register_eval(sources: dict[str, list[str]], work_dir: Path) -> int:
    import numpy as np

    rng = random.Random(11)
    held, train = {}, {}
    for name, size in (("spoken", 3000), ("news", 200), ("gov", 3000)):
        order = list(range(len(sources[name])))
        rng.shuffle(order)
        cut = set(order[:size])
        held[name] = [sources[name][i] for i in sorted(cut)]
        train[name] = [t for i, t in enumerate(sources[name]) if i not in cut]
    index = naturalness.NgramIndex.build(train)

    def scores(texts):
        return [x for x in (naturalness.sentence_register(index, s) for t in texts for s in naturalness.sentences(t)) if x is not None]

    native = scores(held["spoken"]) + scores(held["news"])
    gov = scores(held["gov"])
    print(f"sentence score: native n {len(native)}, p90 {np.percentile(native, 90):+.3f} p95 {np.percentile(native, 95):+.3f} p99 {np.percentile(native, 99):+.3f}; translated gov n {len(gov)}, median {np.median(gov):+.3f}")
    print(f"  constants in the module: cut {naturalness.REGISTER_CUT}, single-sentence cut {naturalness.REGISTER_CUT_SINGLE}")
    print("\nunit-level rule (3 consecutive sentences per unit): share of held-out units flagged")
    for label, texts in (("native spoken", held["spoken"]), ("native news", held["news"]), ("translated gov", held["gov"])):
        units = units_of(texts)
        flagged = sum(1 for u in units if naturalness.register_check(index, u))
        print(f"  {label:16s} units {len(units):5d}  flagged {100 * flagged / max(1, len(units)):5.1f}%")
    print("\nsingle-sentence units: share flagged at the single cut")
    for label, texts in (("native spoken", held["spoken"]), ("native news", held["news"]), ("translated gov", held["gov"])):
        units = [" ".join(s) for t in texts for s in naturalness.sentences(t) if len(s) >= 6][:3000]
        flagged = sum(1 for u in units if naturalness.register_check(index, u))
        print(f"  {label:16s} units {len(units):5d}  flagged {100 * flagged / max(1, len(units)):5.1f}%")
    print("\nthe agent's own units:")
    for run in ("prod-89", "prod-91", "prod-93"):
        path = work_dir / "experience" / "runs" / run / "37bae6ec" / "r1" / "submission.json"
        if not path.is_file():
            continue
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        hits = [(i["item_id"], naturalness.register_check(index, i["zulu_text"])) for i in items]
        hits = [(u, h) for u, h in hits if h]
        print(f"  {run}: {len(hits)} of {len(items)} units flagged:", [(u, h['above'], h['sentences'], h['pulling']) for u, h in hits][:6])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="build the index from everything and save it")
    parser.add_argument("--register", action="store_true", help="evaluate the translated-versus-native register check on held-out units")
    args = parser.parse_args()
    work_dir = Path(load_settings().work_dir)
    started = time.time()
    sources = read_sources(work_dir)
    if args.register:
        return register_eval(sources, work_dir)
    print({k: len(v) for k, v in sources.items()}, flush=True)
    if args.final:
        index = naturalness.NgramIndex.build(sources)
        index.save(naturalness.directory(work_dir))
        print(f"final index: {index.size} words, tokens {index.tokens}, saved to {naturalness.directory(work_dir)} in {time.time() - started:.0f}s")
        return 0

    rng = random.Random(SEED)
    held = {}
    train = {}
    for name, size in (("spoken", HOLD_SPOKEN), ("news", HOLD_NEWS), ("gov", HOLD_GOV)):
        order = list(range(len(sources[name])))
        rng.shuffle(order)
        cut = set(order[:size])
        held[name] = [sources[name][i] for i in sorted(cut)]
        train[name] = [t for i, t in enumerate(sources[name]) if i not in cut]
    index = naturalness.NgramIndex.build(train)
    print(f"held-out index: {index.size} words, tokens {index.tokens}, built in {time.time() - started:.0f}s", flush=True)

    sets = {"REAL spoken": long_sentences(held["spoken"])[:3000], "REAL news": long_sentences(held["news"])[:3000], "REAL gov": long_sentences(held["gov"])[:3000]}
    base = sets["REAL spoken"]
    common = [w for w in index.vocab if index.word(w) >= 200]
    for kind in ("swap_adjacent", "replace_word", "shuffle"):
        sets[f"BROKEN spoken: {kind}"] = corrupt(base, kind, common, rng)

    print("\nshare of sentences that fire (%), by threshold (expected_min / min_word)")
    grid = [(1.0, 20), (3.0, 20), (5.0, 20), (10.0, 20), (3.0, 50), (5.0, 50)]
    print(f"{'set':36s} {'n':>5s} " + " ".join(f"{e:>4.0f}/{m:<3d}" for e, m in grid) + "   | bookish words (%)")
    for label, sentences in sets.items():
        cells = [rates(index, sentences, e, m) for e, m in grid]
        print(f"{label:36s} {cells[0]['n']:5d} " + " ".join(f"{c['pairs']:8.1f}" for c in cells) + f"   | {cells[1]['bookish']:5.1f}", flush=True)

    print("\nthe agent's own text (same thresholds), and what it would be shown at 3/20:")
    for run in ("prod-89", "prod-91"):
        path = work_dir / "experience" / "runs" / run / "37bae6ec" / "r1" / "submission.json"
        if not path.is_file():
            continue
        texts = [i.get("zulu_text", "") for i in json.loads(path.read_text(encoding="utf-8"))["items"]]
        sentences = [s for t in texts for s in naturalness.sentences(t) if len(s) >= 5]
        cells = [rates(index, sentences, e, m) for e, m in grid]
        print(f"{run:36s} {cells[0]['n']:5d} " + " ".join(f"{c['pairs']:8.1f}" for c in cells) + f"   | {cells[1]['bookish']:5.1f}")
        shown = {"pairs": [], "bookish": []}
        for t in texts:
            found = naturalness.check(index, t)
            shown["pairs"] += found["unseen_pairs"][:2]
            shown["bookish"] += found["bookish_words"][:2]
        print("   unseen pairs :", shown["pairs"][:14])
        print("   bookish words:", sorted(set(shown["bookish"]))[:14])
    print(f"\ndone in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
