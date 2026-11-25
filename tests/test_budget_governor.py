import pytest

from mathula_tv.budget_governor import DRAFT_FRACTION, DRAFT_GRACE_TURNS, DRAFT_MIN_TURNS, FINAL_TOOLS, FINAL_TURNS, RESERVE_FRACTION, SOFT_FRACTION, BudgetGovernor
from mathula_tv.draft_ledger import DraftLedger

CEILING = 1_000_000


def test_the_phases_follow_the_ceiling_and_the_reserve_is_sized_from_the_real_prompt():
    governor = BudgetGovernor(CEILING)
    reserve = governor.reserve(50_000)
    assert reserve == FINAL_TURNS * (50_000 + 16_000)          # room for the final turns at today's prompt size, above the fractional floor of 100k
    assert governor.phase(600_000, 50_000) == "normal"
    assert governor.phase(int(SOFT_FRACTION * CEILING), 50_000) == "soft"
    assert governor.phase(CEILING - reserve, 50_000) == "final"
    assert governor.phase(CEILING, 50_000) == "over"


def test_the_reserve_never_swallows_the_working_phase_whatever_the_prompt_size():
    for ceiling in (200_000, 1_000_000, 8_000_000, 16_000_000):
        governor = BudgetGovernor(ceiling)
        for prompt in (0, 10_000, 70_000, 155_000, 900_000):
            reserve = governor.reserve(prompt)
            assert reserve >= min(int(RESERVE_FRACTION * ceiling), int((1 - SOFT_FRACTION) * ceiling))
            assert ceiling - reserve >= SOFT_FRACTION * ceiling                     # the final phase never starts before the soft line
            assert reserve < ceiling


def test_each_phase_is_announced_once_and_a_raised_ceiling_starts_the_warnings_again():
    governor = BudgetGovernor(CEILING)
    assert governor.due("soft") and not governor.due("normal")
    governor.sent("soft")
    assert not governor.due("soft") and governor.due("final")
    governor.phase(100_000)                                                                # back to normal (a resumed run with a higher ceiling)
    assert governor.due("soft") and governor.final_turns == 0


def test_the_final_phase_offers_only_submitting_and_is_bounded_in_turns():
    governor = BudgetGovernor(CEILING)
    specs = [{"function": {"name": n}} for n in ("open_clip", "measure_tts", "submit_dub", "check_text", "get_submission")]
    assert {s["function"]["name"] for s in governor.final_specs(specs)} == FINAL_TOOLS
    assert [governor.final_turn_allowed() for _ in range(FINAL_TURNS + 1)] == [True] * FINAL_TURNS + [False]


def test_the_notes_say_where_the_run_stands_and_what_to_do():
    governor = BudgetGovernor(CEILING)
    soft = governor.note("soft", 700_000, submitted=False)
    assert "70%" in soft and "700,000 of 1,000,000" in soft and "submit_dub now" in soft and "protected" in soft
    assert "Your submission stands" in governor.note("soft", 700_000, submitted=True)
    final = governor.note("final", 900_000, submitted=False, drafts="0.0-1.0s: my draft")
    assert final.startswith("FINAL PHASE") and "only submit_dub and get_submission" in final and "my draft" in final
    assert "my draft" not in governor.note("final", 900_000, submitted=True, drafts="0.0-1.0s: my draft")


SEGMENTS = [{"segment_id": "s1", "speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 4000}, {"segment_id": "s2", "speaker_id": "SPEAKER_01", "start_ms": 4500, "end_ms": 8000}]


class Bench:
    def __init__(self, fail=False):
        self.segments = SEGMENTS
        self.fail = fail
        self.items = None

    def submit_dub(self, items, approach_notes="", partial=False, remove=None):
        if self.fail:
            raise RuntimeError("scoring is down")
        self.items = items
        return {"accepted": True}


def _ledger(*drafts):
    ledger = DraftLedger(SEGMENTS)
    for start, end, text in drafts:
        ledger._put(start, end, text, speaker="SPEAKER_00" if start < 4500 else "SPEAKER_01")
    return ledger


def test_the_last_resort_submits_complete_drafts_names_the_gap_otherwise_and_never_raises():
    governor, bench = BudgetGovernor(CEILING), Bench()
    done = governor.auto_submit(bench, _ledger((0, 4000, "one"), (4500, 8000, "two")))
    assert done["submitted"] and done["units"] == 2 and [i["zulu_text"] for i in bench.items] == ["one", "two"] and bench.items[1]["speaker_id"] == "SPEAKER_01"
    gap = BudgetGovernor(CEILING).auto_submit(Bench(), _ledger((0, 4000, "one")))
    assert gap["submitted"] is False and gap["uncovered"] == ["s2"] and "1 of 2" in gap["reason"]
    assert BudgetGovernor(CEILING).auto_submit(Bench(), DraftLedger(SEGMENTS))["submitted"] is False
    broken = BudgetGovernor(CEILING).auto_submit(Bench(fail=True), _ledger((0, 4000, "one"), (4500, 8000, "two")))
    assert broken["submitted"] is False and "scoring is down" in broken["reason"]


def test_where_drafts_overlap_the_newer_one_wins_when_they_become_a_submission():
    ledger = _ledger((0, 8000, "a long unit over both segments"))
    ledger._put(4500, 8000, "a newer sentence for the second segment", speaker="SPEAKER_01")
    items, uncovered = ledger.as_submission(SEGMENTS)
    assert [i["zulu_text"] for i in items] == ["a newer sentence for the second segment"] and uncovered == ["s1"]        # the newer draft took its window; the older unit overlapped it, so it is out
    older_first = _ledger((4500, 8000, "older"))
    older_first._put(0, 8000, "newer long unit", speaker="SPEAKER_00")
    items, uncovered = older_first.as_submission(SEGMENTS)
    assert [i["zulu_text"] for i in items] == ["newer long unit"] and uncovered == [] and items[0]["source_segment_ids"] == ["s1", "s2"]


def test_the_state_round_trips_through_a_checkpoint():
    governor = BudgetGovernor(CEILING)
    governor.phase(900_000, 50_000)
    governor.sent("soft")
    governor.final_turn_allowed()
    again = BudgetGovernor(CEILING)
    again.load(governor.to_dict())
    assert again.told == {"soft"} and again.final_turns == 1 and again.reached == "final" and again.report()["phase_reached"] == "final"
    with pytest.raises(AssertionError):
        assert again.report().get("auto_submit")


def test_the_draft_deadline_is_a_share_of_the_ceiling_but_at_least_some_turns_of_the_prompt_and_never_past_the_soft_line():
    governor = BudgetGovernor(2_000_000)
    assert governor.draft_deadline(40_000) == int(DRAFT_FRACTION * 2_000_000)                    # a small prompt: the share decides
    assert governor.draft_deadline(120_000) == DRAFT_MIN_TURNS * 120_000                           # a long clip has a large prompt: it gets its turns
    assert governor.draft_deadline(900_000) == int(SOFT_FRACTION * 2_000_000)                      # and never past the soft line


def test_the_draft_state_walks_open_due_grace_overdue_acted_and_a_submission_closes_it():
    governor = BudgetGovernor(1_000_000)
    deadline = governor.draft_deadline(10_000)
    assert governor.draft_state(deadline - 1, 10_000, submitted=False) == "open" and not governor.draft_restricted(submitted=False)
    assert governor.draft_state(deadline, 10_000, submitted=False) == "due"
    governor.draft_told = True
    assert governor.draft_restricted(submitted=False) and not governor.draft_restricted(submitted=True)
    for turn in range(DRAFT_GRACE_TURNS):
        assert governor.draft_state(deadline + 1, 10_000, submitted=False) == "grace"
        governor.draft_turns += 1
    assert governor.draft_state(deadline + 1, 10_000, submitted=False) == "overdue"
    governor.draft_acted = True
    assert governor.draft_state(deadline + 1, 10_000, submitted=False) == "acted"
    assert governor.draft_state(deadline + 1, 10_000, submitted=True) == "open"


def test_the_draft_notes_say_what_is_due_and_what_happens_and_the_state_round_trips():
    governor = BudgetGovernor(1_000_000)
    note = governor.draft_note(310_000, "0.0-1.0s: my draft")
    assert note.startswith("DRAFT DEADLINE: 31%") and "submit_dub" in note and f"{DRAFT_GRACE_TURNS} turns" in note and "my draft" in note
    assert "no text yet for: s2" in governor.draft_uncovered_note({"reason": "the drafts cover 1 of 2 segments", "uncovered": ["s2"]})
    governor.draft_told = True
    governor.draft_turns, governor.draft_acted = 2, True
    again = BudgetGovernor(1_000_000)
    again.load(governor.to_dict())
    assert again.draft_turns == 2 and again.draft_acted and again.draft_told and again.report()["draft_deadline"] == {"turns_after_note": 2, "harness_submitted": True}
