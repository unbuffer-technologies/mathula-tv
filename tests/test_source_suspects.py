import json
from types import SimpleNamespace

from mathula_tv.source_suspects import (
    apply_corrections, apply_to_frozen, corrections_from, find_suspects, judge,
)

SEGMENTS = [
    {"segment_id": "seg-1", "restored_text": "Julius Malema is in Mpumalanga. They must hire security tenders before we come in. "
                                              "Because when we come in, Masia must know, no more tender. Leopard Creek must pay."},
    {"segment_id": "seg-2", "restored_text": "Masia is not Malema. Rupert owes the Nkomazi municipality."},
]
KNOWN = ["Julius Malema", "Mpumalanga", "Leopard Creek", "Rupert", "Nkomazi municipality"]


def test_only_unfamiliar_capitalised_mid_sentence_words_are_suspects():
    tokens = [s["token"] for s in find_suspects(SEGMENTS, KNOWN)]
    assert tokens == ["Masia"]                               # known names, sentence starters and "Because" are not suspects


def test_a_word_used_many_times_is_not_a_suspect():
    many = [{"segment_id": "s", "restored_text": "We met. Then Zodwa spoke. Zodwa agreed. Later Zodwa left. Zodwa returned."}]
    assert find_suspects(many, []) == []


def test_only_a_clear_misrecognised_verdict_with_a_replacement_becomes_a_correction():
    suspects = [{"id": 0, "segment_id": "seg-1", "token": "Masia", "sentence": "x"}, {"id": 1, "segment_id": "seg-1", "token": "Zodwa", "sentence": "y"}]
    verdicts = [{"id": 0, "verdict": "misrecognised", "replacement": "the municipality", "reason": "masipala"},
                {"id": 1, "verdict": "unsure", "replacement": "", "reason": ""}]
    assert [c["token"] for c in corrections_from(suspects, verdicts)] == ["Masia"]


def test_the_judge_sends_the_audio_sounds_and_context_in_one_call():
    seen = {}

    class Provider:
        def complete_json(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(data={"verdicts": [{"id": 0, "verdict": "misrecognised", "replacement": "the municipality", "reason": "r"}]})

    suspects = [{"id": 0, "segment_id": "seg-1", "token": "Masia", "sentence": "when we come in, Masia must know",
                 "previous_sentence": "", "audio_phones": "m a s i a m a"}]
    verdicts = judge(Provider(), suspects, KNOWN)
    assert verdicts[0]["verdict"] == "misrecognised" and seen["payload"]["tokens"][0]["audio_phones"] == "m a s i a m a"


def test_the_correction_is_applied_to_the_frozen_copy_only_and_recorded(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"segments": [dict(s, source_text=s["restored_text"]) for s in SEGMENTS]}))
    corrections = tmp_path / "source_corrections.json"
    corrections.write_text(json.dumps({"corrections": [{"segment_id": "seg-1", "token": "Masia", "replacement": "the municipality",
                                                        "reason": "masipala"}]}))
    changed = apply_to_frozen(frozen, corrections)
    data = json.loads(frozen.read_text())
    assert "when we come in, the municipality must know" in data["segments"][0]["restored_text"]
    assert "Masia is not Malema" in data["segments"][1]["restored_text"]            # other segments are untouched
    assert changed[0]["was"] == "Masia" and data["corrections"][-1]["source"] == "audio source check"
    assert apply_to_frozen(frozen, tmp_path / "missing.json") == []


def test_a_repair_run_is_told_which_units_carry_a_corrected_segment():
    from mathula_tv.dub_agent import _source_correction_note

    workbench = SimpleNamespace(source_corrections=[{"segment_id": "seg-4", "token": "Masia", "replacement": "the municipality"}])
    seed = [{"item_id": "u3", "source_segment_ids": ["seg-3"]}, {"item_id": "u4", "source_segment_ids": ["seg-4"]}]
    note = _source_correction_note(workbench, seed)
    assert "Masia" in note and "the municipality" in note and "units: u4" in note and "u3" not in note
    assert _source_correction_note(SimpleNamespace(source_corrections=[]), seed) == ""


def test_a_clip_is_refrozen_when_its_source_corrections_change(tmp_path):
    from mathula_tv.dub_production import ensure_frozen
    from mathula_tv.experience_store import ExperienceStore, source_corrections_sha

    job = tmp_path / "jobs" / "abc12345"
    native = job / "native_dub"
    native.mkdir(parents=True)
    for name in ("pass1_restored_transcript.json", "context_ledger.json", "zulu_terminology_glossary.json"):
        (native / name).write_text(json.dumps({"segments": [{"segment_id": "seg-1", "restored_text": "when we come in, Masia must know",
                                                             "source_text": "x", "start_ms": 0, "end_ms": 1000}]}))
    store = ExperienceStore(tmp_path / "experience")
    first = ensure_frozen(store, job, "abc12345")
    assert not first.get("already_frozen") and ensure_frozen(store, job, "abc12345").get("already_frozen")
    (native / "source_corrections.json").write_text(json.dumps({"corrections": [
        {"segment_id": "seg-1", "token": "Masia", "replacement": "the municipality", "reason": "masipala"}]}))
    again = ensure_frozen(store, job, "abc12345")
    assert not again.get("already_frozen") and again["source_corrections_sha"] == source_corrections_sha(native)
    frozen = json.loads((store.clip_dir("abc12345") / "native_dub" / "pass1_restored_transcript.json").read_text())
    assert "the municipality must know" in frozen["segments"][0]["restored_text"]
    assert "Masia" in json.loads((native / "pass1_restored_transcript.json").read_text())["segments"][0]["restored_text"]   # the job's own file is untouched
