"""Experience loop: SA-specific scoring checks, submission contract, corpus store, review labels.

The heuristic cases are real failures from job fb3d08b63fed4d90922b08f7e325b906 and
earlier -- each was first caught by a human listening, which is exactly what the
scorer exists to stop depending on.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.dub_submission import score_submission_dir, validate_submission
from mathula_tv.episode_scoring import (
    english_leakage, find_fabricated_entities, find_restated_references, normalise_number_format, split_restatements,
    find_unsupported_doubling, score_items, _known_entity_tokens,
)
from mathula_tv.experience_cli import compare_scorecards
from mathula_tv.experience_store import ExperienceStore, text_hash
from mathula_tv.review_sheet import (
    build_review_rows, read_review_sheet, scorer_calibration, write_review_sheet,
)


# --- heuristics, from real failures -------------------------------------------------------


def test_restated_municipality_from_the_real_regression_is_flagged():
    hits = find_restated_references(
        "Sifuna umsebenzi ongashintshi kamasipala kuMasipala waseLekwa."
    )
    assert any({h["first"].lower(), h["second"].lower()} == {"kamasipala", "kumasipala"} for h in hits)


def test_emphatic_doubling_is_not_a_restated_reference():
    assert find_restated_references("Sifuna umsebenzi oqine kakhulu, kakhulu, eLekwa.") == []


def test_doubling_is_only_supported_when_the_english_doubled_a_word():
    assert find_unsupported_doubling("kakhulu, kakhulu", "to work very, very permanent") == []
    assert find_unsupported_doubling("kakhulu, kakhulu", "to work permanently") == ["kakhulu"]


def test_fabricated_first_name_is_caught_but_localised_names_are_not():
    known = _known_entity_tokens(
        "I was at Mr. Lincoln's house in Mbombela on November 1.",
        {"terms": [], "do_not_translate": ["Lincoln"]}, {"entities": []},
    )
    assert find_fabricated_entities("Ngangise endlini ka uAndre Lincoln.", known) == ["uAndre"]
    assert find_fabricated_entities("Ngangise-ndlini kaLincoln eMbombela ngomhla ka-1 kuNovemba.", known) == []


def test_sentence_initial_capitals_are_not_names():
    assert find_fabricated_entities("Izakhamuzi zafika. Bahlala phansi.", set()) == []


def test_capitalised_zulu_common_nouns_and_quote_starts_are_not_fabricated_names():
    # Real false positives from the first baseline scoring of four remote jobs.
    for text in (
        "Uthi uMbuso akazange ahambe eNkantolo ngoLwesihlanu.",
        "Bazovela ku-Agasti nakuDisemba, ngoMeyi, kuSepthemba.",
        "WayeyiNobhala-Jikelele; uyise wakhe wavela eGoli neThekwini.",
        "Uthe: “Ngingasolwa kanjani?” — Kunzima kakhulu.",
        "Uthe “Kunzima lokhu.” Kusukela ekuqaleni kwakunjalo; ngokoHlelo 6.",
    ):
        assert find_fabricated_entities(text, set()) == [], text
    assert find_fabricated_entities("Wafika uNkosana eMtubatuba.", set()) == ["uNkosana", "eMtubatuba"]


def test_repeated_names_are_only_a_gate_when_the_english_said_it_fewer_times():
    zulu = "U-Armindo Pakula wathi, kaPakula ngokwakhe, uyavuma."
    gate, soft = split_restatements(zulu, "Armindo Pakula said Pakula himself agrees.")
    assert gate == [] and soft
    gate, _ = split_restatements(zulu, "Armindo said he agrees.")
    assert [h["stem"] for h in gate] == ["pakula"]


def test_repeated_common_nouns_are_a_soft_flag_not_a_gate():
    gate, soft = split_restatements("babendawonye ehhotela, ndawonye", "they were together at the hotel")
    assert gate == [] and [h["stem"] for h in soft] == ["ndawonye"]


def test_number_grouping_differences_do_not_count_as_a_missing_literal():
    assert normalise_number_format("about 7,000 rand", source=True) == "about 7000 rand"
    assert normalise_number_format("ama-randi ayizi-7 000 kuphela", source=False) == "ama-randi ayizi-7000 kuphela"
    assert normalise_number_format("7 000 000", source=False) == "7000000"


def test_english_leakage_needs_several_function_words():
    assert english_leakage("Sifuna the people and the jobs of the town") > 0.2
    assert english_leakage("Sifuna umsebenzi") == 0.0


# --- scoring core --------------------------------------------------------------------------


def _item(group_id="g1", *, source="The report cites 24 documents.", spoken="Umbiko ukhomba amadokhumenti angu-24.",
          start=0, end=4000, measured=4000, direction=None, rush=0.0, variant="agent", turn_id="t1"):
    return {
        "group_id": group_id, "turn_id": turn_id, "speaker_id": "S0", "start_ms": start, "end_ms": end,
        "source_text": source, "spoken_text": spoken, "measured_ms": measured, "direction": direction,
        "required_speed_percent": rush, "variant_id": variant, "candidate_id": group_id, "requires_review": False,
    }


def _score(items, **kwargs):
    return score_items(items, glossary={"terms": [], "do_not_translate": []}, ledger={"entities": []}, **kwargs)


def test_a_clean_unit_has_no_gate_failures_or_flags():
    card = _score([_item()])
    assert card["headline"]["gate_failures"] == 0
    assert card["groups"][0]["flags"] == []


def test_dead_air_is_measured_against_the_real_window():
    card = _score([_item(measured=2000, direction="expand")])
    group = card["groups"][0]
    assert group["shortfall_ms"] == 2000 and group["shortfall_pct"] == 50.0
    assert "dead_air" in group["flags"]
    assert card["headline"]["dead_air_pct"] == 50.0


def test_dropping_a_number_fails_the_literal_gate():
    card = _score([_item(spoken="Umbiko ukhomba amadokhumenti amaningi.")])
    assert card["gates"]["required_literals"] == ["g1"]


def test_last_resort_compression_is_waived_from_the_literal_gate():
    card = _score([_item(spoken="Umbiko ukhomba amadokhumenti amaningi.", variant="protected_content_last_resort")])
    assert card["gates"]["required_literals"] == []


def test_a_human_bad_substring_becomes_a_regression_gate_on_new_output():
    label = {"group_id": "other", "verdict": "bad", "bad_substring": "kamasipala kuMasipala"}
    card = _score([_item(spoken="Umbiko 24 kamasipala kuMasipala")], labels=[label])
    assert card["gates"]["known_bad_text"] == ["g1"]


def test_untranslated_copy_through_fails():
    card = _score([_item(source="Hello there", spoken="Hello there.")])
    assert card["gates"]["untranslated"] == ["g1"]


def test_turn_rollup_uses_the_whole_turn_window():
    items = [
        _item("a", start=0, end=4000, measured=2000, direction="expand", turn_id="t"),
        _item("b", start=4000, end=8000, measured=4000, turn_id="t"),
    ]
    turn = _score(items)["turns"][0]
    assert turn["window_ms"] == 8000 and turn["measured_ms"] == 6000 and turn["shortfall_pct"] == 25.0


# --- submission contract -------------------------------------------------------------------

SEGMENTS = [
    {"segment_id": "seg-1", "speaker_id": "S0", "start_ms": 0, "end_ms": 4000, "restored_text": "The report cites 24 documents."},
    {"segment_id": "seg-2", "speaker_id": "S0", "start_ms": 4000, "end_ms": 8000, "restored_text": "Mbombela voted."},
]


def _sub(*items):
    return {"items": list(items)}


def _u(item_id, start, end, ids, text="Isibonelo."):
    return {"item_id": item_id, "start_ms": start, "end_ms": end, "speaker_id": "S0",
            "zulu_text": text, "source_segment_ids": ids}


def test_a_complete_ordered_submission_validates():
    errors, _ = validate_submission(_sub(_u("a", 0, 4000, ["seg-1"]), _u("b", 4000, 8000, ["seg-2"])), SEGMENTS)
    assert errors == []


def test_an_uncovered_segment_is_an_error():
    errors, _ = validate_submission(_sub(_u("a", 0, 4000, ["seg-1"])), SEGMENTS)
    assert any("seg-2" in e and "not covered" in e for e in errors)


def test_overlapping_and_out_of_clip_windows_are_errors():
    errors, _ = validate_submission(
        _sub(_u("a", 0, 5000, ["seg-1"]), _u("b", 4500, 20000, ["seg-2"])), SEGMENTS,
    )
    assert any("overlaps" in e for e in errors) and any("outside the clip" in e for e in errors)


def test_a_shared_segment_requires_explicit_source_text():
    errors, warnings = validate_submission(
        _sub(_u("a", 0, 2000, ["seg-1"]), _u("b", 2000, 4000, ["seg-1"]), _u("c", 4000, 8000, ["seg-2"])), SEGMENTS,
    )
    assert any("explicit source_text" in e for e in errors) and warnings


class _FakeTts:
    default_voice = "zu-ZA-ThandoNeural"

    def __init__(self, durations):
        self.durations = durations

    def synthesize(self, request, output_path, *, force=False):
        return SimpleNamespace(duration_ms=self.durations.get(request.text, 3000))


def _write_native(tmp_path: Path) -> Path:
    native = tmp_path / "job" / "native_dub"
    atomic_write_json(native / "pass1_restored_transcript.json", {"segments": SEGMENTS})
    atomic_write_json(native / "context_ledger.json", {"context_ledger": {"entities": []}})
    atomic_write_json(native / "zulu_terminology_glossary.json",
                      {"zulu_terminology_glossary": {"terms": [], "do_not_translate": ["Mbombela"]}})
    return native


def test_submission_is_remeasured_and_scored_by_the_scorer_not_the_agent(tmp_path):
    native = _write_native(tmp_path)
    submission = _sub(
        _u("a", 0, 4000, ["seg-1"], "Umbiko ukhomba amadokhumenti angu-24."),
        _u("b", 4000, 8000, ["seg-2"], "IMbombela ikhethile."),
    )
    tts = _FakeTts({"Umbiko ukhomba amadokhumenti angu-24.": 1000, "IMbombela ikhethile.": 4000})
    card = score_submission_dir(native, submission, tts=tts, measure_root=tmp_path / "m", usage={"tts_measurements": 2})
    assert card["approach"] == "agent_submission"
    first = card["groups"][0]
    assert first["measured_ms"] == 1000 and "dead_air" in first["flags"]
    assert card["gates"]["required_literals"] == []
    assert card["headline"]["tts_measurements"] == 2
    assert card["turns"][0]["group_ids"] == ["a", "b"]


def test_an_invalid_submission_refuses_to_score(tmp_path):
    native = _write_native(tmp_path)
    with pytest.raises(ValueError, match="not covered"):
        score_submission_dir(native, _sub(_u("a", 0, 4000, ["seg-1"])), tts=_FakeTts({}), measure_root=tmp_path / "m")


# --- corpus store ---------------------------------------------------------------------------


def _fake_job(tmp_path: Path) -> Path:
    job = tmp_path / "work" / "jobs" / "abcdef1234567890"
    native = job / "native_dub"
    atomic_write_json(native / "pass1_restored_transcript.json", {"segments": SEGMENTS})
    atomic_write_json(native / "context_ledger.json", {"context_ledger": {}})
    atomic_write_json(native / "zulu_terminology_glossary.json", {"zulu_terminology_glossary": {}})
    atomic_write_json(native / "pass2_candidate_pool.json", {"candidate_pool": [], "usage": {}})
    return job


def test_freeze_copies_inputs_and_baseline_and_seeds_a_clean_replay_dir(tmp_path):
    job = _fake_job(tmp_path)
    store = ExperienceStore(tmp_path / "work")
    entry = store.freeze_clip(job, notes="first")
    assert entry["clip_id"] == "abcdef12" and entry["duration_s"] == 8.0
    assert store.baseline_native_dub("abcdef12").joinpath("pass2_candidate_pool.json").is_file()
    seeded = store.seed_replay_dir("abcdef12", tmp_path / "replay")
    assert (seeded / "native_dub" / "pass1_restored_transcript.json").is_file()
    assert not (seeded / "native_dub" / "pass2_candidate_pool.json").exists()
    with pytest.raises(FileExistsError):
        store.freeze_clip(job)


def test_freeze_refuses_a_job_without_a_candidate_pool(tmp_path):
    job = _fake_job(tmp_path)
    (job / "native_dub" / "pass2_candidate_pool.json").unlink()
    with pytest.raises(FileNotFoundError):
        ExperienceStore(tmp_path / "work").freeze_clip(job)


# --- review loop ----------------------------------------------------------------------------


def _card_for_review():
    flagged = _item("bad1", spoken="Umbiko amaningi.")  # drops "24"
    quiet = [_item(f"ok{i}", spoken=f"Umbiko ukhomba amadokhumenti angu-24 {i}.") for i in range(6)]
    return _score([flagged, *quiet])


def test_review_sheet_puts_flagged_first_then_a_seeded_spot_check(tmp_path):
    rows = build_review_rows("clip", _card_for_review(), spot_check=3)
    assert rows[0]["group_id"] == "bad1" and "gate:required_literals" in rows[0]["why_flagged"]
    assert [r["why_flagged"] for r in rows[1:]] == ["spot_check (nothing flagged)"] * 3
    assert rows == build_review_rows("clip", _card_for_review(), spot_check=3)  # deterministic
    path = write_review_sheet(tmp_path / "sheet.csv", rows)
    assert path.read_text(encoding="utf-8-sig").startswith("clip_id,group_id")


def test_ingest_validates_verdicts_and_tags_and_skips_blank_rows(tmp_path):
    rows = build_review_rows("clip", _card_for_review(), spot_check=2)
    rows[0].update(verdict="bad", tags="dropped_content", bad_substring="amaningi", corrected_zulu="Umbiko ukhomba amadokhumenti angu-24.", note="lost the number")
    rows[1].update(verdict="maybe")
    rows[2].update(verdict="ok", tags="not_a_tag")
    path = write_review_sheet(tmp_path / "sheet.csv", rows)
    labels, problems = read_review_sheet(path)
    assert len(labels) == 1 and labels[0]["tags"] == ["dropped_content"]
    assert labels[0]["corrected_zulu"] == "Umbiko ukhomba amadokhumenti angu-24."
    assert len(problems) == 2


def test_calibration_reports_what_the_scorer_missed_and_falsely_flagged():
    card = _card_for_review()
    labels = [
        {"group_id": "bad1", "zulu_hash": card["groups"][0]["zulu_hash"], "verdict": "bad"},
        {"group_id": "ok0", "zulu_hash": card["groups"][1]["zulu_hash"], "verdict": "bad"},   # scorer missed
        {"group_id": "ok1", "zulu_hash": card["groups"][2]["zulu_hash"], "verdict": "ok"},
    ]
    report = scorer_calibration(card, labels)
    assert report["precision"] == 1.0 and report["recall"] == 0.5
    assert report["missed_by_scorer"] == ["ok0"]


# --- comparison -----------------------------------------------------------------------------


def test_compare_flags_a_new_gate_failure_as_a_regression_even_if_dead_air_improves():
    base = _score([_item(measured=2000, direction="expand")])
    cand = _score([_item(measured=4000, spoken="Umbiko amaningi.")])
    result = compare_scorecards(base, cand)
    assert result["verdict"] == "regression" and "new gate failures" in result["regressions"]
    assert result["deltas"]["dead_air_pct"] < 0


def test_compare_reports_improvement_when_dead_air_drops_and_nothing_regresses():
    base = _score([_item(measured=2000, direction="expand")])
    cand = _score([_item(measured=3900)])
    assert compare_scorecards(base, cand)["verdict"] == "improvement"
