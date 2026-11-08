import json

import pytest

from mathula_tv import job_pronunciations as jp
from mathula_tv import language_questions as lq

ENGLISH = {"heidi", "wehle", "lobo", "buffalo"}
LEARNED = "zulu-shaped|no-capital-inside"


def find(texts, *, said=None, lessons=None, unsure=None, resolved=None):
    return jp.candidates(texts, researched={}, dictionary=lambda w: w in ENGLISH, zulu_count=lambda w: 19 if w in ENGLISH else 900, said=said, lessons=lessons, unsure=unsure,
                         resolved=resolved)


def test_a_dictionary_word_nobody_said_is_not_respelled_as_english_but_asked_about():
    unsure = []
    said = lambda w: w.casefold() in {"heidi", "buffalo"}
    assert find(["Wehle ngesenti. Sawubona Heidi eBuffalo."], said=said, unsure=unsure) == ["Heidi", "Buffalo"]          # Wehle was written by the translator, not said
    assert unsure == [{"word": "Wehle", "signature": LEARNED}]
    assert find(["Wehle ngesenti."], said=None) == ["Wehle"]                                                              # no restored transcript yet: nothing to compare with, the old rule stands


def test_the_listeners_answer_for_a_word_decides_in_every_later_job():
    said, unsure = (lambda w: False), []
    assert find(["Wehle ngesenti."], said=said, lessons={"words": {"wehle": {"language": "zulu"}}}, unsure=unsure) == [] and unsure == []          # left as written, not asked again
    assert find(["Wehle ngesenti."], said=said, lessons={"words": {"wehle": {"language": "sepedi"}}}, unsure=unsure) == [] and unsure == []
    assert find(["Wehle ngesenti."], said=said, lessons={"words": {"wehle": {"language": "english"}}}, unsure=unsure) == ["Wehle"]               # English is respelled by the rules
    assert find(["Wehle ngesenti."], said=lambda w: True, lessons={"words": {"wehle": {"language": "zulu"}}}) == []                              # an answer outranks what was said


def test_words_that_look_like_ones_already_answered_the_same_way_are_settled_without_asking():
    lessons = {"words": {k: {"language": "zulu", "signature": LEARNED} for k in ("a", "b", "c")}}
    unsure, resolved = [], []
    assert find(["Lobo bufakazi."], said=lambda w: False, lessons=lessons, unsure=unsure, resolved=resolved) == []
    assert unsure == [] and resolved == [{"word": "Lobo", "language": "zulu", "signature": LEARNED}]
    few = {"words": {k: {"language": "zulu", "signature": LEARNED} for k in ("a", "b")}}                                                   # two answers are not a pattern
    unsure, resolved = [], []
    find(["Lobo bufakazi."], said=lambda w: False, lessons=few, unsure=unsure, resolved=resolved)
    assert [u["word"] for u in unsure] == ["Lobo"] and resolved == []
    split = {"words": {**{k: {"language": "zulu", "signature": LEARNED} for k in ("a", "b", "c")}, "d": {"language": "english", "signature": LEARNED}}}          # 3 of 4 is not enough
    unsure = []
    find(["Lobo bufakazi."], said=lambda w: False, lessons=split, unsure=unsure)
    assert [u["word"] for u in unsure] == ["Lobo"]
    english = {"words": {k: {"language": "english", "signature": LEARNED} for k in ("a", "b", "c")}}
    assert find(["Lobo bufakazi."], said=lambda w: False, lessons=english) == ["Lobo"]                                                    # settled as English: respelled


def test_a_signature_tells_zulu_shaped_words_from_others_and_a_capital_inside_a_sentence_from_one_that_opens_it():
    assert lq.signature("Wehle", inside=False) == LEARNED
    assert lq.signature("Pakistan", inside=True) == "zulu-shaped|capital-inside"
    assert lq.signature("Court", inside=False) == "not-zulu-shaped|no-capital-inside"


def _job(tmp_path, segments, captions=None):
    job = tmp_path / "work" / "jobs" / ("ab" * 16)
    (job / "native_dub").mkdir(parents=True)
    (job / "native_dub" / "pass1_restored_transcript.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    if captions is not None:
        folder = job / "input" / "youtube_captions"
        folder.mkdir(parents=True)
        (folder / "vid.zu-orig.json3").write_text(json.dumps({"events": [{"tStartMs": s, "dDurationMs": e - s, "segs": [{"utf8": t}]} for s, e, t in captions]}), encoding="utf-8")
    return job


SEGMENTS = [{"segment_id": "seg-1", "start_ms": 1000, "end_ms": 5000, "speaker_id": "S0", "source_text": "Siyipatala through tax rise now yes please",
             "restored_text": "Siyipatala through tax rise now yes please", "changed": False},
            {"segment_id": "seg-2", "start_ms": 6000, "end_ms": 9000, "speaker_id": "S0", "source_text": "We went down by one cent this week",
             "restored_text": "We went down by one cent this week", "changed": False}]
CAPTIONS = [(900, 5200, "ngiyabonga kakhulu bantu bakithi sengiyaphuma"), (6000, 9000, "We went down by one cent this week")]


def test_what_was_said_is_the_transcript_the_restored_text_and_the_captions_together(tmp_path):
    job = _job(tmp_path, SEGMENTS, CAPTIONS)
    said = lq.said_words(job)
    assert {"siyipatala", "cent", "ngiyabonga", "kakhulu"} <= said and "wehle" not in said
    assert lq.heard(said, "Cents") and lq.heard(said, "cent") and not lq.heard(said, "Wehle")
    assert lq.said_words(tmp_path / "work" / "jobs" / "nothing") is None


def test_only_the_stretches_the_captions_still_do_not_support_are_asked_and_the_answer_is_kept_as_a_lesson(tmp_path):
    job, work = _job(tmp_path, SEGMENTS, CAPTIONS), tmp_path / "work"
    asked = lq.stretch_questions(job)
    assert [q["id"] for q in asked] == ["seg-1"] and asked[0]["caption"].startswith("ngiyabonga") and asked[0]["answer"] is None and asked[0]["track"] == "zu-orig"
    kept = lq.answer_stretch(job, work, "seg-1", "typed", "  Ngiyabonga kakhulu bantu   bakithi ")
    assert kept["choice"] == "typed" and kept["text"] == "Ngiyabonga kakhulu bantu bakithi"
    assert lq.stretch_questions(job)[0]["answer"]["choice"] == "typed"                                           # an answered stretch stays listed so it can be changed
    lesson = lq.load_lessons(work)["stretches"]
    assert len(lesson) == 1 and lesson[0]["final"] == "Ngiyabonga kakhulu bantu bakithi" and lesson[0]["transcript"].startswith("Siyipatala") and lesson[0]["track"] == "zu-orig"
    lq.answer_stretch(job, work, "seg-1", "transcript")                                                          # a changed mind replaces the lesson, it does not add a second
    assert [e["choice"] for e in lq.load_lessons(work)["stretches"]] == ["transcript"]


def test_an_answer_is_applied_to_the_restored_text_once_and_the_models_own_text_is_kept(tmp_path):
    job, work = _job(tmp_path, SEGMENTS, CAPTIONS), tmp_path / "work"
    lq.answer_stretch(job, work, "seg-1", "typed", "Ngiyabonga kakhulu")
    segments = json.loads((job / "native_dub" / "pass1_restored_transcript.json").read_text())["segments"]
    assert lq.apply_to_segments(job, segments) == 1 and lq.apply_to_segments(job, segments) == 0
    first, second = segments
    assert first["restored_text"] == "Ngiyabonga kakhulu" and first["model_text"].startswith("Siyipatala") and first["reason_code"] == "listener_confirmed" and first["changed"] is True
    assert second["restored_text"] == SEGMENTS[1]["restored_text"] and "listener_confirmed" not in second
    lq.answer_stretch(job, work, "seg-1", "model")                                                               # the model was right after all: its own text comes back
    assert lq.apply_to_segments(job, segments) == 1 and segments[0]["restored_text"].startswith("Siyipatala")
    lq.answer_stretch(job, work, "seg-1", "transcript")                                                          # the transcript's text is the model's here: nothing differs, nothing changes
    assert lq.apply_to_segments(job, segments) == 0 and segments[0]["restored_text"] == SEGMENTS[0]["source_text"] and segments[0]["changed"] is False and segments[0]["listener_confirmed"] == "transcript"


def test_the_listeners_corrections_become_examples_for_the_next_pass_1_prompt(tmp_path):
    job, work = _job(tmp_path, SEGMENTS, CAPTIONS), tmp_path / "work"
    assert lq.examples(work) == []
    lq.answer_stretch(job, work, "seg-1", "typed", "Ngiyabonga kakhulu")
    example = lq.examples(work)[0]
    assert example["transcript"].startswith("Siyipatala") and example["caption_track"] == "zu-orig" and example["listener_decided"] == "Ngiyabonga kakhulu" and example["winner"] == "the listener's own words"
    many = {"words": {}, "stretches": [{"job": f"j{n}", "segment_id": "x", "choice": "model", "model": "m", "final": f"final {n}", "track": "en-orig"} for n in range(10)]}
    lq._write(lq.lessons_path(work), many)
    shown = lq.examples(work)
    assert len(shown) == lq.EXAMPLES_MAX and shown[-1]["listener_decided"] == "final 9"                          # the latest ones, a bounded number


def test_bad_answers_are_refused_and_a_word_answer_can_be_taken_back(tmp_path):
    job, work = _job(tmp_path, SEGMENTS, CAPTIONS), tmp_path / "work"
    with pytest.raises(ValueError):
        lq.answer_stretch(job, work, "seg-1", "guess")
    with pytest.raises(ValueError):
        lq.answer_stretch(job, work, "seg-1", "typed", "   ")
    with pytest.raises(ValueError):
        lq.answer_stretch(job, work, "seg-404", "model")
    with pytest.raises(ValueError):
        lq.answer_word(work, "Wehle", "klingon")
    lq.answer_word(work, "Wehle", "zulu", LEARNED)
    assert lq.load_lessons(work)["words"]["wehle"]["language"] == "zulu" and lq.forget_word(work, "WEHLE") and lq.load_lessons(work)["words"] == {}


def test_a_word_said_in_another_form_was_said_but_a_word_nobody_said_is_not_made_up_by_a_stem(tmp_path):
    said = {"we", "re", "street", "fighting", "pakistanis", "somalis", "went", "down"}
    assert lq.heard(said, "Pakistan") and lq.heard(said, "Somalia") and lq.heard(said, "somali")          # "Pakistanis" and "Somalis" were said; the dub keeps "namaPakistan"
    assert not lq.heard(said, "Wehle") and not lq.heard(said, "lobo") and not lq.heard(said, "")


def test_a_glued_prefix_needs_a_capital_so_nhlobo_is_not_lobo_and_kwehle_is_not_wehle():
    from mathula_tv import ear_clips

    assert ear_clips.word_pattern("lobo").search("abukadambi nhlobo") is None and ear_clips.word_pattern("Wehle").search("ukwesekwa kwenu kwehle kusuka") is None
    assert ear_clips.word_pattern("Wehle").search("Wehle ngesenti").group(2) == "Wehle"
    glued = ear_clips.word_pattern("Pakistan").search("silwa namaPakistan namaSomalia")
    assert glued.group(1) == "nama" and glued.group(2) == "Pakistan"
    assert ear_clips.word_pattern("Jacob").search("UJacob Zuma").group(1) == "U" and ear_clips.word_pattern("Jacob").search("i-Jacob").group(1) == "i-"


def test_a_short_segment_is_never_a_question_because_the_neighbours_caption_lines_overlap_it(tmp_path):
    short = [{"segment_id": "seg-1", "start_ms": 1000, "end_ms": 2000, "speaker_id": "S0", "source_text": "ultimately the lesson here", "restored_text": "ultimately the lesson here?", "changed": True}]
    job = _job(tmp_path, short, [(900, 2200, "something else entirely different from it")])
    assert lq.stretch_questions(job) == []
