"""An independent fidelity judge for the agent's isiZulu.

The scorer measures timing and hard gates; it cannot tell whether the Zulu *means* what the English meant.
This judge is a screen for that, not ground truth. It works in two blind-then-compare passes so the verdict
is not anchored on the English:

1. back-translate: a model sees only the Zulu and writes plain English for it (marking what it cannot read);
2. compare: a model sees the source English, the back-translation and the Zulu, and scores meaning
   preserved 0-3 with typed issues (omission / addition / distortion / unclear).

It is the same model family as the agent, so its errors correlate with the agent's. Treat the output as a
way to rank units for the human reviewers (and, later, calibrate it against their verdicts), never as a
reason to skip them. Nothing here feeds the agent or the improvement gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .delivery import strip_pauses

CHUNK = 10
ISSUE_KINDS = ("omission", "addition", "distortion", "unclear")

BACK_TRANSLATE_PROMPT = """\
You translate isiZulu news speech into plain, literal English so a reviewer can see exactly what the Zulu says.
You are NOT given the original English: do not guess it, do not smooth over problems, do not add what is not
there. Keep every name, number and date as written. If a word or phrase is not understandable, write [unclear]
in its place.
Reply with JSON only: {"items":[{"id":"...","english":"..."}]} with one entry per input id.
"""

COMPARE_PROMPT = """\
You check whether an isiZulu dub preserves the meaning of the English it was made from.
For each item you get: the SOURCE English, the DUB (isiZulu) and BACK (a blind English back-translation of the
dub). Judge meaning, not word order: a good dub may reorder, merge or retell, but must not lose, add or change
facts, who did what, negation, quantities, names, dates or attitude.
Score meaning_preserved:
  3 = same meaning, nothing material lost or added
  2 = same facts, minor nuance or emphasis lost
  1 = a material omission, addition or distortion
  0 = wrong or unrelated meaning
List each problem in issues as {"kind":"omission|addition|distortion|unclear","detail":"<=20 words"}.
Spoken delivery may legitimately drop filler ("um", repeated words); do not count that. Be strict about facts.
Reply with JSON only: {"items":[{"id":"...","meaning_preserved":0-3,"issues":[...]}]} with one entry per input id.
"""

FOREIGN_PROMPT = """\
You read isiZulu text spoken by a Zulu voice that reads every word with Zulu letter rules, so English words kept inside it come out
wrong unless someone checks them. For each item list every English (or other non-Zulu) word or name the Zulu keeps as it is, in
whatever form it has: bare ("medical aid"), with a hyphen prefix ("ne-medical"), or with a Zulu prefix written straight onto it
("weScience", "iScience", "NeTechnology", "MaFighter"). Give "written" exactly as it appears in the text (the whole token) and
"english" as the bare English word, and "kind": "word" for an ordinary English word ("medical", "science", "benefit") or "name" for
the name of a person, place, party or organisation, or an acronym ("Malema", "Nkomazi", "EFF", "Leopard Creek"): names are handled
elsewhere, from the speaker's audio, and an English listener cannot judge a South African name. Do not list ordinary isiZulu words, nor
English words that are fully adapted into Zulu spelling ("isikole", "ibhasi"), nor numbers.
Reply with JSON only: {"items":[{"id":"...","words":[{"written":"...","english":"...","kind":"word|name"}]}]} with one entry per input id.
"""

CONFIRM_PROMPT = """A reviewer flagged each item below as losing material meaning. Check that claim strictly; flags are often wrong,
especially on long units, where a dub that follows the source clause by clause gets described as "adding"
things that the source already says.
For each item re-read SOURCE, DUB and BACK. List only problems you can prove with exact words:
  {"kind":"omission|distortion|addition","source_quote":"<exact words copied from SOURCE>",
   "dub_quote":"<exact words copied from BACK showing what the dub says instead, or empty for an omission>","detail":"<=20 words"}
An omission needs a source_quote that the dub genuinely does not carry; a distortion needs both quotes; an addition
needs a dub_quote that the source does not support. Retelling, reordering and dropping filler are not problems.
Then score meaning_preserved 0-3 as before (3 = nothing material lost). If you cannot quote a real problem, score 2 or 3.
Reply with JSON only: {"items":[{"id":"...","meaning_preserved":0-3,"issues":[...]}]}
"""


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", str(text or "").lower()).strip()


def _grounded(issue: Mapping[str, Any], source: str, back: str) -> bool:
    """A confirmed issue must quote real words: the source quote from the source, the dub quote from the dub's reading."""
    kind, sq, dq = issue.get("kind"), _norm(issue.get("source_quote")), _norm(issue.get("dub_quote"))
    if kind == "omission":
        return bool(sq) and sq in _norm(source)
    if kind == "distortion":
        return bool(sq) and sq in _norm(source) and bool(dq) and dq in _norm(back)
    if kind == "addition":
        return bool(dq) and dq in _norm(back)
    return False


def _parse_items(text: str) -> dict[str, dict[str, Any]]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    items = data.get("items") if isinstance(data, dict) else None
    return {str(i["id"]): i for i in items or [] if isinstance(i, dict) and "id" in i}


def _call(client: Any, system: str, payload: Sequence[Mapping[str, Any]], usage: dict[str, int]) -> dict[str, dict[str, Any]]:
    response = client.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(list(payload), ensure_ascii=False)}],
        json_mode=True, max_output_tokens=8000)
    for key in ("prompt_tokens", "completion_tokens"):
        usage[key] = usage.get(key, 0) + int((response.get("usage") or {}).get(key) or 0)
    message = (response.get("choices") or [{}])[0].get("message") or {}
    return _parse_items(message.get("content") or "")


def _clean_issues(raw: Any) -> list[dict[str, str]]:
    issues = []
    for issue in raw if isinstance(raw, list) else []:
        if isinstance(issue, dict):
            kind = str(issue.get("kind", "")).lower()
            issues.append({"kind": kind if kind in ISSUE_KINDS else "distortion", "detail": str(issue.get("detail", ""))[:200]})
    return issues


def judge_units(client: Any, units: Sequence[Mapping[str, str]], *, chunk: int = CHUNK) -> dict[str, Any]:
    """Judge ``units`` ({"id","source","dub"}). Returns per-unit verdicts plus the token cost."""
    usage: dict[str, int] = {}
    verdicts: list[dict[str, Any]] = []
    units = [{**u, "dub": strip_pauses(u["dub"])} for u in units]       # ~pause800 is a direction to the voice, not part of the meaning
    for start in range(0, len(units), chunk):
        part = units[start:start + chunk]
        back = _call(client, BACK_TRANSLATE_PROMPT, [{"id": u["id"], "zulu": u["dub"]} for u in part], usage)
        compared = _call(client, COMPARE_PROMPT, [
            {"id": u["id"], "source": u["source"], "dub": u["dub"], "back": (back.get(u["id"]) or {}).get("english", "")}
            for u in part], usage)
        doubtful = [u for u in part if (compared.get(u["id"]) or {}).get("meaning_preserved") in (0, 1)]
        confirmed = _call(client, CONFIRM_PROMPT, [
            {"id": u["id"], "source": u["source"], "dub": u["dub"], "back": (back.get(u["id"]) or {}).get("english", "")}
            for u in doubtful], usage) if doubtful else {}
        for u in doubtful:
            second = confirmed.get(u["id"]) or {}
            proven = [i for i in second.get("issues") or [] if isinstance(i, dict)
                      and _grounded(i, u["source"], (back.get(u["id"]) or {}).get("english", ""))]
            if second.get("meaning_preserved") in (0, 1) and proven:
                compared[u["id"]] = {**second, "issues": proven}
            else:                                         # not provable with quotes: a doubt, not a defect
                compared[u["id"]] = {"meaning_preserved": 2, "issues": compared[u["id"]].get("issues"), "unconfirmed": True}
        for u in part:
            result = compared.get(u["id"])
            score = result.get("meaning_preserved") if result else None
            verdicts.append({
                "id": u["id"], "source": u["source"], "dub": u["dub"],
                "back_translation": (back.get(u["id"]) or {}).get("english"),
                "meaning_preserved": int(score) if isinstance(score, (int, float)) and 0 <= score <= 3 else None,
                "issues": _clean_issues(result.get("issues") if result else None),
                **({"unconfirmed": True} if result and result.get("unconfirmed") else {}),
            })
    return {"verdicts": verdicts, "summary": summarise(verdicts), "usage": usage}


def foreign_words(client: Any, texts: Sequence[str], usage: dict[str, int], *, chunk: int = CHUNK) -> list[dict[str, str]]:
    """The English words a model finds kept inside ``texts`` (isiZulu), each as {"written", "english"}. Text only: no audio, no source."""
    found: dict[tuple[str, str], dict[str, str]] = {}
    for start in range(0, len(texts), chunk):
        part = [{"id": str(start + i), "zulu": strip_pauses(text)} for i, text in enumerate(texts[start:start + chunk])]
        for item in _call(client, FOREIGN_PROMPT, part, usage).values():
            for word in item.get("words") or []:
                if isinstance(word, dict) and str(word.get("written") or "").strip() and str(word.get("english") or "").strip():
                    written, english = str(word["written"]).strip(), str(word["english"]).strip()
                    kind = "name" if str(word.get("kind") or "").strip().lower() == "name" else "word"
                    found.setdefault((written.casefold(), english.casefold()), {"written": written, "english": english, "kind": kind})
    return list(found.values())


class UnitJudge:
    """A judge that remembers its verdicts, keyed by the exact (source, dub) text pair.

    The judge is a sampled model, so asking twice can give different scores. Inside one run the agent
    resubmits mostly-unchanged units; re-judging them would turn that noise into a moving target the agent
    chases. With the cache a unit keeps its verdict until its text changes, and only changed units cost tokens.
    """

    def __init__(self, client: Any, *, chunk: int = CHUNK) -> None:
        self.client, self.chunk = client, chunk
        self._cache: dict[str, dict[str, Any]] = {}
        self.usage: dict[str, int] = {}

    @staticmethod
    def _key(unit: Mapping[str, str]) -> str:
        return hashlib.sha1(f"{unit['source']}\x00{unit['dub']}".encode("utf-8")).hexdigest()

    def foreign_words(self, texts: Sequence[str]) -> list[dict[str, str]]:
        """English words kept in the Zulu texts. One model call per ten new texts; a text keeps its answer until it changes."""
        cache = self.__dict__.setdefault("_foreign", {})
        fresh = [t for t in dict.fromkeys(texts) if t not in cache]
        for start in range(0, len(fresh), self.chunk):
            part = fresh[start:start + self.chunk]
            for text in part:
                cache[text] = []
            for word in foreign_words(self.client, part, self.usage, chunk=self.chunk):
                for text in part:
                    if word["written"] in strip_pauses(text):
                        cache[text].append(word)
        merged: dict[tuple[str, str], dict[str, str]] = {}
        for text in texts:
            for word in cache.get(text, []):
                merged.setdefault((word["written"].casefold(), word["english"].casefold()), word)
        return list(merged.values())

    def judge(self, units: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
        fresh = {self._key(u): u for u in units if self._key(u) not in self._cache}
        if fresh:
            # Unit ids are only labels for the model; the cache is keyed by text, so two units with equal text share one call.
            labelled = [{"id": key, "source": u["source"], "dub": u["dub"]} for key, u in fresh.items()]
            result = judge_units(self.client, labelled, chunk=self.chunk)
            for verdict in result["verdicts"]:
                if verdict["meaning_preserved"] is not None:        # an unjudged unit is retried next time
                    self._cache[verdict["id"]] = verdict
            for key, value in result["usage"].items():
                self.usage[key] = self.usage.get(key, 0) + value
        out = []
        for unit in units:
            cached = self._cache.get(self._key(unit))
            out.append({**(cached or {"meaning_preserved": None, "issues": [], "back_translation": None}), "id": unit["id"],
                        "source": unit["source"], "dub": unit["dub"]})
        return out


def summarise(verdicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [v for v in verdicts if v["meaning_preserved"] is not None]
    words = lambda v: max(1, len(str(v["source"]).split()))  # noqa: E731 - weight by how much speech a unit carries
    total = sum(words(v) for v in scored)
    return {
        "units": len(verdicts), "unjudged": len(verdicts) - len(scored),
        "mean_score": round(sum(v["meaning_preserved"] for v in scored) / len(scored), 2) if scored else None,
        "word_weighted_mean": round(sum(v["meaning_preserved"] * words(v) for v in scored) / total, 2) if total else None,
        "pct_units_fully_faithful": round(100 * sum(v["meaning_preserved"] == 3 for v in scored) / len(scored), 1) if scored else None,
        "units_with_material_problems": [v["id"] for v in scored if v["meaning_preserved"] <= 1],
    }


def units_from_scorecard(run_dir: Path) -> list[dict[str, str]]:
    card = json.loads((Path(run_dir) / "scorecard.json").read_text(encoding="utf-8"))
    return [{"id": str(g["group_id"]), "source": g["source_text"], "dub": g["spoken_text"]}
            for g in card["groups"] if g.get("source_text") and g.get("spoken_text")]


def judge_run(client: Any, run_dir: Path) -> dict[str, Any]:
    result = judge_units(client, units_from_scorecard(run_dir))
    (Path(run_dir) / "judge.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_judge")
    parser.add_argument("--run", required=True, help="run id, e.g. base-v2")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args(argv)

    from .config import load_settings
    from .dub_agent import ChatClient
    from .experience_store import ExperienceStore
    from .foundry_grok import native_gpt_config_from_environment

    store = ExperienceStore(load_settings().work_dir)
    client = ChatClient(native_gpt_config_from_environment(dict(os.environ)))
    for repeat in args.repeats:
        run_dir = store.repeat_dir(args.run, args.clip, repeat)
        if not (run_dir / "scorecard.json").is_file():
            print(f"r{repeat}: no scorecard, skipped")
            continue
        result = judge_run(client, run_dir)
        print(f"r{repeat}: {json.dumps(result['summary'], ensure_ascii=False)} tokens={result['usage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
