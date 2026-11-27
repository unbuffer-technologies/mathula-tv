"""Correct one unit of a finished dub, and make only that unit again.  No agent turn, no time-fitting of the other units.

    python -m mathula_tv.dub_fix --job <job id> --run <run id> --unit 2 --replace "ngo-1994=>ngo-1994 kwaba"      # a word or phrase changed in one unit's text
    python -m mathula_tv.dub_fix --job <job id> --run <run id> --unit u-03 --text "the whole new text of the unit"
    python -m mathula_tv.dub_fix --job <job id> --run <run id> --unit 2 --check                                     # only measure it: nothing is edited
    ... --finish                                                                                                     # then the video and the files to post it with (dub_finish)

The dub is one text per unit (``submission.json``), and every unit is made and kept on its own: the Studio names a unit's audio by its text, its draws, its reference clip and the run's voice settings, so
changing one unit's text makes that unit again and finds every other unit as it was.  The agent is only needed to WRITE a better text; a correction a person has already made needs none of that:

1. the edit is made on a copy (``submission.before_fix_<n>.json`` is the text as it was) and written back;
2. the unit is spoken as the render speaks it and measured against its own window, the time from its start to where the next unit starts: spoken seconds, room, and what is left over or lacking;
   nothing is stretched, squeezed or cut to make it fit: a unit that does not fit is told so, and the person (or one scoped agent turn) decides;
3. with ``--finish`` the whole dub is laid out again from the units (only the changed one is made) and the video is finished.

The draws a person committed belong to the exact text of a sentence, so an edit only loses the commits of the sentences it changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping


def find_unit(items: list[dict[str, Any]], unit: str | int) -> int:
    """The index of a unit given as its number or its item id ("u-03")."""
    if isinstance(unit, int) or str(unit).isdigit():
        index = int(unit)
        if not 0 <= index < len(items):
            raise ValueError(f"no unit {unit}: the dub has {len(items)}")
        return index
    for index, item in enumerate(items):
        if str(item.get("item_id")) == str(unit):
            return index
    raise ValueError(f"no unit {unit}")


def edit_unit(items: list[dict[str, Any]], unit: str | int, *, text: str | None = None, replace: str | None = None) -> tuple[list[dict[str, Any]], int, str, str]:
    """(the items with one unit's text changed, its index, the text before, the text after).  ``replace`` is "old=>new", every occurrence in that unit."""
    index = find_unit(items, unit)
    before = str(items[index].get("zulu_text") or "")
    if text is not None:
        after = text
    elif replace is not None:
        if "=>" not in replace:
            raise ValueError('--replace is "old=>new"')
        old, new = replace.split("=>", 1)
        if not old or old not in before:
            raise ValueError(f"{old!r} is not in unit {index}")
        after = before.replace(old, new)
    else:
        after = before
    changed = [dict(item) for item in items]
    changed[index]["zulu_text"] = after
    return changed, index, before, after


def window(items: list[dict[str, Any]], index: int) -> dict[str, float]:
    """The unit's own time: from its start to where the next unit starts (the last: to its own source end), and its source length."""
    item = items[index]
    start, end = int(item.get("start_ms") or 0), int(item.get("end_ms") or 0)
    following = int(items[index + 1].get("start_ms") or end) if index + 1 < len(items) else end
    return {"source_s": round((end - start) / 1000.0, 2), "room_s": round((max(following, end) - start) / 1000.0, 2)}


def fit(spoken_s: float, room: Mapping[str, float]) -> dict[str, Any]:
    """What the voice's own reading of the unit leaves in its room: ``over_s`` is how far it runs into the next unit, ``spare_s`` how much room is left after it."""
    over = round(spoken_s - room["room_s"], 2)
    return {"spoken_s": round(spoken_s, 2), **room, "over_s": max(over, 0.0), "spare_s": max(-over, 0.0), "fits": over <= 0.0}


def measure(console: Any, job: str, run: str, items: list[dict[str, Any]], index: int) -> dict[str, Any]:
    """The unit spoken as the render speaks it (its draws as committed, the run's voice), and its fit."""
    corpus, group, spans, gaps, _ref = console._draw_unit(job, run, index)
    backend = console._omni_with(console._run_voice(job, run))
    audio, rate, _answer, report, _reference = backend.speak(group, spans, gaps, corpus)
    return {**fit(len(audio) / rate, window(items, index)), "spans": len(spans), "picked": report.get("picked")}


def fix(settings: Any, *, job_id: str, run_id: str, unit: str | int, text: str | None = None, replace: str | None = None, check: bool = False, finish: bool = False,
        clip_id: str | None = None, progress: Callable[[str], None] = print, console: Any = None) -> dict[str, Any]:
    from .experience_store import ExperienceStore

    clip_id = clip_id or job_id[:8]
    path = ExperienceStore(settings.work_dir).repeat_dir(run_id, clip_id, 1) / "submission.json"
    submission = json.loads(path.read_text(encoding="utf-8"))
    items = list(submission["items"])
    if check:
        changed, index, before, after = items, find_unit(items, unit), "", ""
    else:
        changed, index, before, after = edit_unit(items, unit, text=text, replace=replace)
    result: dict[str, Any] = {"unit": index, "item_id": items[index].get("item_id")}
    if not check and after != before:          # nothing changed (no edit asked for): the unit is only measured and made again
        n = 1
        while path.with_name(f"submission.before_fix_{n}.json").exists():
            n += 1
        path.with_name(f"submission.before_fix_{n}.json").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(json.dumps({**submission, "items": changed}, ensure_ascii=False, indent=2), encoding="utf-8")
        result["edited"] = {"before": before, "after": after, "backup": f"submission.before_fix_{n}.json"}
        progress(f"[fix] unit {index} ({result['item_id']}) text changed; the text as it was is in submission.before_fix_{n}.json")
        items = changed
    if console is None:
        from .console import Console

        console = Console(Path(settings.work_dir))
    result["fit"] = measure(console, job_id, run_id, items, index)
    progress(f"[fix] unit {index}: {json.dumps(result['fit'])}")
    if finish:
        from .dub_finish import finish as finish_dub

        result["finish"] = finish_dub(settings, job_id=job_id, run_id=run_id, clip_id=clip_id, progress=progress, console=console)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_fix")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--unit", required=True, help="the unit's number (0 is the first) or its item id (u-03)")
    parser.add_argument("--replace", default=None, help='"old=>new": every occurrence in that unit\'s text')
    parser.add_argument("--text", default=None, help="the whole new text of the unit")
    parser.add_argument("--check", action="store_true", help="only measure the unit: nothing is edited")
    parser.add_argument("--finish", action="store_true", help="then finish the video and the files to post it with (dub_finish)")
    parser.add_argument("--clip", default=None)
    args = parser.parse_args(argv)
    if sum(x is not None for x in (args.replace, args.text)) > 1:
        parser.error("give --replace or --text, not both")

    from .config import load_settings

    print(json.dumps(fix(load_settings(), job_id=args.job, run_id=args.run, unit=args.unit, text=args.text, replace=args.replace, check=args.check, finish=args.finish, clip_id=args.clip),
                     indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
