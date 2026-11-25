"""Which of a run's accepted submissions is the best, by the order the scorer itself uses: hard gates first, then cross-speaker collisions, collisions, overruns, and last the dead air left.

``dub_workflow`` already chose the best round by this order; the agent loop now records every accepted submission with it, so a run's best one is never lost and the final one can be
checked against it (prod-114 made three submissions whose last was also its best: more gates and collisions fixed, at the price of dead air).
"""
from __future__ import annotations

from typing import Any, Mapping


def score_key(result: Mapping[str, Any] | None) -> tuple[float, float, float, float, float]:
    """Lower is better: hard gates first, then cross-speaker collisions, collisions, overruns, then dead air that is left."""
    headline = (result or {}).get("headline") or {}
    return (float(headline.get("gate_failures", 1e9)), float(headline.get("cross_speaker_collisions", 0)), float(headline.get("collisions", 0)),
            float(headline.get("groups_rushed", 1e9)), float(headline.get("residual_dead_air_pct", 1e9)))


class BestSubmission:
    """Every accepted submission of a run, and which is best."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None

    def observe(self, turn: int, result: Mapping[str, Any] | None, items: list[Mapping[str, Any]], unit_texts: Mapping[str, str] | None = None) -> bool:
        """Record one accepted submission; True when it is now the best (ties go to the later one: it is the agent's latest word).
        ``unit_texts``: the English each unit was written against, kept with it so a later run can tell which units the source has since changed."""
        if not result or not result.get("accepted"):
            return False
        entry = {"turn": int(turn), "key": list(score_key(result)), "headline": dict(result.get("headline") or {}), "items": [dict(i) for i in items],
                 "unit_texts": dict(unit_texts or {})}
        self.history.append({k: entry[k] for k in ("turn", "key", "headline")})
        if self.best is None or tuple(entry["key"]) <= tuple(self.best["key"]):
            self.best = entry
            return True
        return False

    def report(self) -> dict[str, Any]:
        if not self.history:
            return {"submissions": 0}
        final = self.history[-1]
        return {"submissions": len(self.history), "best_turn": self.best["turn"] if self.best else None, "final_turn": final["turn"],
                "final_is_best": bool(self.best and tuple(final["key"]) <= tuple(self.best["key"])), "scores": [{"turn": h["turn"], "key": h["key"]} for h in self.history]}

    def to_dict(self) -> dict[str, Any]:
        return {"history": self.history, "best": self.best}

    def load(self, state: Mapping[str, Any] | None) -> None:
        if state:
            self.history = list(state.get("history") or [])
            self.best = state.get("best")
