"""Native-speaker review: a CSV sheet to fill in, label ingestion, scorer calibration.

Native-speaker judgement is the one signal no off-the-shelf metric supplies for
isiZulu news (AfriCOMET's evaluation data has no isiZulu), so the sheet is built
to spend a reviewer's time where it teaches the most. A reviewer's CORRECTED Zulu
(``corrected_zulu``) is the highest-value field: it becomes a post-edit pair for
retrieval-based adaptation and, later, a preference pair. Time goes where it teaches
the scorer the most: every flagged
unit first, then a seeded random sample of UNflagged units -- the latter is how
the scorer's blind spots are found.
"""
from __future__ import annotations

import csv
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

VERDICTS = ("ok", "bad", "unsure")
TAGS = (
    "restated_reference", "mistranslation", "dropped_content", "fabricated_content", "unnatural",
    "grammar", "pronunciation", "timing_dead_air", "timing_rush", "register", "other",
)
SHEET_COLUMNS = (
    "clip_id", "group_id", "zulu_hash", "priority", "why_flagged", "start_s", "window_s", "measured_s",
    "shortfall_pct", "rush_pct", "english", "zulu",
    "verdict", "tags", "bad_substring", "corrected_zulu", "note",
)


def _why(group: Mapping[str, Any]) -> list[str]:
    reasons = [f"gate:{name}" for name, gate in group["gates"].items() if gate["failed"]]
    reasons.extend(f"flag:{flag}" for flag in group["flags"])
    return reasons


def build_review_rows(
    clip_id: str, scorecard: Mapping[str, Any], *, spot_check: int = 4, seed: int = 7,
) -> list[dict[str, Any]]:
    groups = list(scorecard["groups"])
    already = {g["zulu_hash"] for g in groups if g.get("labels")}
    flagged = [g for g in groups if _why(g) and g["zulu_hash"] not in already]
    quiet = [g for g in groups if not _why(g) and g["zulu_hash"] not in already]
    rng = random.Random(f"{seed}:{clip_id}")
    sampled = rng.sample(quiet, min(spot_check, len(quiet)))

    def _priority(group: Mapping[str, Any]) -> tuple[int, float]:
        gate_hits = sum(1 for gate in group["gates"].values() if gate["failed"])
        return (-gate_hits, -max(group["shortfall_pct"], group["rush_pct"]))

    rows = []
    for rank, (group, kind) in enumerate(
        [*((g, "flagged") for g in sorted(flagged, key=_priority)), *((g, "spot_check") for g in sampled)], start=1,
    ):
        rows.append({
            "clip_id": clip_id,
            "group_id": group["group_id"],
            "zulu_hash": group["zulu_hash"],
            "priority": rank,
            "why_flagged": "; ".join(_why(group)) if kind == "flagged" else "spot_check (nothing flagged)",
            "start_s": round(group["start_ms"] / 1000.0, 1),
            "window_s": round(group["window_ms"] / 1000.0, 1),
            "measured_s": round(group["measured_ms"] / 1000.0, 1),
            "shortfall_pct": group["shortfall_pct"],
            "rush_pct": group["rush_pct"],
            "english": group["source_text"],
            "zulu": group["spoken_text"],
            "verdict": "", "tags": "", "bad_substring": "", "corrected_zulu": "", "note": "",
        })
    return rows


def write_review_sheet(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_review_sheet(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (labels for rows the reviewer filled in, problems found)."""
    labels: list[dict[str, Any]] = []
    problems: list[str] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            verdict = (row.get("verdict") or "").strip().lower()
            if not verdict:
                continue
            if verdict not in VERDICTS:
                problems.append(f"line {line_number}: verdict {verdict!r} must be one of {VERDICTS}")
                continue
            tags = [t.strip().lower() for t in (row.get("tags") or "").replace(";", ",").split(",") if t.strip()]
            unknown = [t for t in tags if t not in TAGS]
            if unknown:
                problems.append(f"line {line_number}: unknown tags {unknown} (allowed: {', '.join(TAGS)})")
                continue
            labels.append({
                "clip_id": row["clip_id"], "group_id": row["group_id"], "zulu_hash": row["zulu_hash"],
                "verdict": verdict, "tags": tags,
                "bad_substring": (row.get("bad_substring") or "").strip(),
                "corrected_zulu": (row.get("corrected_zulu") or "").strip(),
                "note": (row.get("note") or "").strip(),
                "english": row.get("english", ""), "zulu": row.get("zulu", ""),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            })
    return labels, problems


def scorer_calibration(scorecard: Mapping[str, Any], labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How well does 'the scorer flagged it' predict 'the native speaker said bad'?"""
    latest = {(label["group_id"], label["zulu_hash"]): label for label in labels}
    tp = fp = fn = tn = 0
    missed: list[str] = []
    false_alarms: list[str] = []
    per_gate: dict[str, dict[str, int]] = {}
    for group in scorecard["groups"]:
        label = latest.get((group["group_id"], group["zulu_hash"]))
        if label is None or label["verdict"] == "unsure":
            continue
        human_bad = label["verdict"] == "bad"
        scorer_flagged = bool(_why(group))
        if human_bad and scorer_flagged:
            tp += 1
        elif human_bad:
            fn += 1
            missed.append(group["group_id"])
        elif scorer_flagged:
            fp += 1
            false_alarms.append(group["group_id"])
        else:
            tn += 1
        for name, gate in group["gates"].items():
            if gate["failed"]:
                bucket = per_gate.setdefault(name, {"flagged_and_bad": 0, "flagged_and_ok": 0})
                bucket["flagged_and_bad" if human_bad else "flagged_and_ok"] += 1
    labeled = tp + fp + fn + tn
    return {
        "labeled_units": labeled,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "missed_by_scorer": missed,
        "false_alarms": false_alarms,
        "per_gate": per_gate,
    }
