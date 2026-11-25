"""An inspection of the Zulu a run ended with: what a second reader finds wrong with its grammar, once the agent is done.

This reads the FINAL submission of a run, with the English of each unit, and reports per unit the phrases that are ungrammatical or too dense to say (dub_judge.grammar_issues).  The reader is an English-speaking
model: an opinion, which flagged different phrases in identical text from one reading to the next (prod-132 against prod-133), so its count is not a measure of quality.  It is for a person to weigh; it changes
nothing.  It is written beside the run as ``grammar_inspect.json``.

    python -m mathula_tv.grammar_inspect --job <job id> --run <run id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def source_english(job_root: Path) -> dict[str, str]:
    """{segment id: the English of that segment} from the job's transcript (the corrected one first)."""
    for name in ("corrected_transcript.json", "transcript_en.json", "transcript_en_raw.json"):
        path = Path(job_root) / "analysis" / name
        if path.is_file():
            try:
                segments = json.loads(path.read_text(encoding="utf-8")).get("segments") or []
            except (OSError, ValueError):
                continue
            return {str(s.get("segment_id")): str(s.get("restored_text") or s.get("source_text") or "") for s in segments if s.get("segment_id") is not None}
    return {}


def inspect(items: Sequence[Mapping[str, Any]], english: Mapping[str, str], client: Any) -> dict[str, Any]:
    """The reviewer's findings for each unit: {units: [{item_id, issues}], counts}."""
    from .dub_judge import grammar_issues

    units = [{"id": str(i.get("item_id")), "english": " ".join(english.get(str(x), "") for x in i.get("source_segment_ids") or []), "zulu": str(i.get("zulu_text") or "")} for i in items]
    usage: dict[str, int] = {}
    found = grammar_issues(client, units, usage, chunk=1)
    rows = [{"item_id": u["id"], "issues": found.get(u["id"], [])} for u in units]
    kinds: dict[str, int] = {}
    for row in rows:
        for issue in row["issues"]:
            kinds[issue["kind"]] = kinds.get(issue["kind"], 0) + 1
    return {"units": rows, "counts": {"units": len(rows), "units_with_issues": sum(1 for r in rows if r["issues"]), "issues": sum(kinds.values()), "by_kind": kinds}, "usage": usage}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.grammar_inspect")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    from .config import load_settings
    from .dub_agent import native_client_config, new_chat_client
    from .experience_store import ExperienceStore

    settings = load_settings()
    work = Path(settings.work_dir)
    run_dir = ExperienceStore(work).repeat_dir(args.run, args.job[:8], 1)
    items = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))["items"]
    report = inspect(items, source_english(work / "jobs" / args.job), new_chat_client(native_client_config()))
    (run_dir / "grammar_inspect.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in report["units"]:
        print(f"{row['item_id']}: {len(row['issues'])} issues")
        for issue in row["issues"]:
            print(f"   [{issue['kind']}] {issue['phrase']}: {issue['problem']}  ->  {issue['fix']}")
    print(json.dumps(report["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
