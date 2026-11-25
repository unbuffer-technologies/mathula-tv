"""The run's token budget as a structure, so that a run cannot end having produced nothing.

A token ceiling that is only a hard stop has one failure: the agent spends it all on research and checks, and the run ends with no submission (prod-113 on Gemini passed 6.3M
tokens with none).  The governor makes the ceiling a sequence of phases the harness enforces, whatever the agent does:

* draft deadline (before any submission exists, from DRAFT_FRACTION of the ceiling, or DRAFT_MIN_TURNS turns of the current prompt if that is more, never past the soft line): the agent is told once to submit
  a complete first draft, every tool but submitting is refused until it does, and after DRAFT_GRACE_TURNS more turns the harness submits the agent's own working drafts when they cover the clip.  The first
  submission is what gives the agent real timing, gates and collisions for every unit at once, so the budget before it is bounded, not left to free research and checks (prod-117: 39 turns of free tools
  before the first submission, then the ceiling).
* normal: the work.
* soft (from SOFT_FRACTION of the ceiling): the agent is told once where it stands and what remains, and to submit a complete draft while it can still be improved.
* final (the last ``reserve`` tokens, sized from the real prompt so that FINAL_TURNS turns always fit): the tool list is cut to submitting, so the reserve cannot be spent
  on anything else, and the agent is told it is the last chance.
* over: the run stops, and if there is still no submission the harness submits the agent's own working drafts (the draft ledger) when they cover the whole clip, and
  saves them to the run folder when they do not.

The fractions are policy for every run, not tuned to one: they decide how early the agent is warned and how large the protected reserve is.
"""
from __future__ import annotations

from typing import Any, Mapping

SOFT_FRACTION = 0.70
RESERVE_FRACTION = 0.10
DRAFT_FRACTION = 0.30                                     # the share of the ceiling by which a complete first submission is due
DRAFT_MIN_TURNS = 8                                       # ... but never less than this many turns at the current prompt size (a long clip has a large prompt)
DRAFT_GRACE_TURNS = 3                                     # turns after the note before the harness submits the agent's drafts itself
FINAL_TURNS = 3                                           # at most this many turns in the final phase, and the reserve is sized to hold them
FINAL_TOOLS = frozenset({"submit_dub", "get_submission"})
DEFAULT_MAX_OUTPUT = 16_000


class BudgetGovernor:
    def __init__(self, ceiling: int, *, max_output: int = DEFAULT_MAX_OUTPUT) -> None:
        self.ceiling = int(ceiling)
        self.max_output = int(max_output)
        self.told: set[str] = set()
        self.final_turns = 0
        self.reached: str = "normal"
        self.auto_submit_result: dict[str, Any] | None = None
        self.draft_told = False                           # the draft deadline note was sent (kept apart from ``told``: phase() clears that one while the work phase lasts)
        self.draft_turns = 0                              # turns taken since the draft deadline note, with no submission
        self.draft_acted = False                          # the harness has tried to submit the drafts itself

    # -- the phases -----------------------------------------------------------------------------------------------------------------------------------------
    def reserve(self, last_prompt_tokens: int) -> int:
        """What the final phase protects: room for FINAL_TURNS turns at the prompt size of the moment (never more than the share between the soft line and the ceiling)."""
        wanted = max(int(RESERVE_FRACTION * self.ceiling), FINAL_TURNS * (int(last_prompt_tokens) + self.max_output))
        return min(wanted, int((1.0 - SOFT_FRACTION) * self.ceiling))

    def phase(self, used: int, last_prompt_tokens: int = 0) -> str:
        if used >= self.ceiling:
            current = "over"
        elif used >= self.ceiling - self.reserve(last_prompt_tokens):
            current = "final"
        elif used >= SOFT_FRACTION * self.ceiling:
            current = "soft"
        else:
            current = "normal"
        if current == "normal":                           # a raised ceiling (a resumed run) starts the warnings again
            self.told.clear()
            self.final_turns = 0
        order = ("normal", "soft", "final", "over")
        if order.index(current) > order.index(self.reached):
            self.reached = current
        return current

    # -- the draft deadline ---------------------------------------------------------------------------------------------------------------------------------
    def draft_deadline(self, last_prompt_tokens: int = 0) -> int:
        """The token count by which a first submission is due."""
        wanted = max(int(DRAFT_FRACTION * self.ceiling), DRAFT_MIN_TURNS * int(last_prompt_tokens))
        return min(wanted, int(SOFT_FRACTION * self.ceiling))

    def draft_state(self, used: int, last_prompt_tokens: int = 0, *, submitted: bool) -> str:
        """open (before the deadline, or a submission exists) | due (the note is to be sent now) | grace (told, turns remain) | overdue (the harness acts now) | acted (it has)."""
        if submitted or used < self.draft_deadline(last_prompt_tokens):
            return "open"
        if not self.draft_told:
            return "due"
        if self.draft_acted:
            return "acted"
        return "overdue" if self.draft_turns >= DRAFT_GRACE_TURNS else "grace"

    def draft_restricted(self, *, submitted: bool) -> bool:
        """After the deadline note and until a submission exists, only submitting is available."""
        return self.draft_told and not submitted

    def draft_note(self, used: int, drafts: str = "") -> str:
        share = int(100 * used / max(self.ceiling, 1))
        return (f"DRAFT DEADLINE: {share}% of the token ceiling is used ({used:,} of {self.ceiling:,}) and there is no submission yet. Only submit_dub and get_submission are available until there is one (any other call is "
                f"refused and costs nothing). Call submit_dub now with complete text for EVERY unit: size a unit you have not settled by the length guide you already have, because the score shows which units need work "
                f"and the other tools come back as soon as it is in. If there is no submission within {DRAFT_GRACE_TURNS} turns the harness submits your working drafts itself."
                + (f"\n\nYour working drafts, as the harness recorded them:\n{drafts}" if drafts else ""))

    @staticmethod
    def draft_uncovered_note(result: Mapping[str, Any]) -> str:
        uncovered = ", ".join(str(u) for u in (result.get("uncovered") or [])[:30])
        return ("The harness could not submit your working drafts: " + str(result.get("reason") or "they do not cover the clip") + (f" (no text yet for: {uncovered})" if uncovered else "")
                + ". Call submit_dub now with complete text for EVERY unit, including these. Only submitting is available until you do.")

    def due(self, phase: str) -> bool:
        """Is this phase's note still to be sent?  Once per phase."""
        return phase in ("soft", "final") and phase not in self.told

    def sent(self, phase: str) -> None:
        self.told.add(phase)

    def final_specs(self, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The tools of the final phase: submitting only."""
        return [s for s in specs if s["function"]["name"] in FINAL_TOOLS]

    def final_turn_allowed(self) -> bool:
        self.final_turns += 1
        return self.final_turns <= FINAL_TURNS

    # -- what the agent is told -----------------------------------------------------------------------------------------------------------------------------
    def note(self, phase: str, used: int, *, submitted: bool, drafts: str = "") -> str:
        share = int(100 * used / max(self.ceiling, 1))
        if phase == "soft":
            return (f"Budget: this run has used {share}% of its token ceiling ({used:,} of {self.ceiling:,}). Plan to finish with what remains. "
                    + ("Your submission stands: spend what is left on the units it flags. " if submitted else
                       "You have not submitted yet: call submit_dub now with complete text for EVERY unit (a draft that is not perfect can still be improved afterwards with partial submissions). ")
                    + "The last part of the budget is protected for one final submission and cannot be used for anything else.")
        body = (f"FINAL PHASE: {share}% of the token ceiling is used ({used:,} of {self.ceiling:,}) and only submit_dub and get_submission are available now. "
                + ("Send what you still want changed with submit_dub (partial=true: only the units you change), then reply with one short sentence and stop."
                   if submitted else "You have not submitted: call submit_dub now with your best complete text for EVERY unit, then reply with one short sentence and stop."))
        return body + (f"\n\nYour working drafts, as the harness recorded them:\n{drafts}" if drafts and not submitted else "")

    # -- the last resort ------------------------------------------------------------------------------------------------------------------------------------
    def auto_submit(self, workbench: Any, ledger: Any) -> dict[str, Any]:
        """The run is ending with no accepted submission: submit the agent's own working drafts if they cover every segment of the clip.  Never raises."""
        segments = list(getattr(workbench, "segments", None) or [])
        if not segments or not ledger.entries:
            self.auto_submit_result = {"submitted": False, "reason": "there are no drafts to submit" if not ledger.entries else "the clip is not open"}
            return self.auto_submit_result
        items, uncovered = ledger.as_submission(segments)
        if uncovered or not items:
            self.auto_submit_result = {"submitted": False, "reason": f"the drafts cover {len(segments) - len(uncovered)} of {len(segments)} segments", "uncovered": uncovered[:30]}
            return self.auto_submit_result
        try:
            result = workbench.submit_dub(items, approach_notes="auto-submitted by the budget governor from the agent's own working drafts: the run ended with no submission", partial=False)
            self.auto_submit_result = {"submitted": bool(result.get("accepted")), "units": len(items), "errors": result.get("errors")}
        except Exception as error:  # noqa: BLE001 - a failed last resort must not turn the run's ending into a crash
            self.auto_submit_result = {"submitted": False, "reason": f"{type(error).__name__}: {str(error)[:200]}"}
        return self.auto_submit_result

    # -- the checkpoint and the report ----------------------------------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"told": sorted(self.told), "final_turns": self.final_turns, "reached": self.reached, "draft_told": self.draft_told, "draft_turns": self.draft_turns, "draft_acted": self.draft_acted}

    def load(self, state: Mapping[str, Any] | None) -> None:
        if state:
            self.told = set(state.get("told") or [])
            self.final_turns = int(state.get("final_turns") or 0)
            self.reached = str(state.get("reached") or "normal")
            self.draft_told = bool(state.get("draft_told"))
            self.draft_turns = int(state.get("draft_turns") or 0)
            self.draft_acted = bool(state.get("draft_acted"))

    def report(self) -> dict[str, Any]:
        return {"ceiling": self.ceiling, "phase_reached": self.reached, "final_turns": self.final_turns,
                **({"draft_deadline": {"turns_after_note": self.draft_turns, "harness_submitted": self.draft_acted}} if self.draft_told else {}), **({"auto_submit": self.auto_submit_result} if self.auto_submit_result else {})}
