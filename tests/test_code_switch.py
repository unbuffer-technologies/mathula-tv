import json
from types import SimpleNamespace

from mathula_tv import code_switch as cs


def test_disagreement_is_zero_for_the_same_words_and_high_for_few_shared():
    assert cs.disagreement("The ANC has no money", "the anc has no money!") == 0
    assert cs.disagreement("Siyipatala through tax", "Lethu siyayibadala thru taxt") > 0.5
    assert cs.disagreement("", "anything") == 0


def test_the_most_different_readings_are_chosen_capped_and_returned_in_time_order():
    rows = [{"id": i, "start_s": float(i), "end_s": i + 1.0, "english": e, "isizulu": z} for i, (e, z) in enumerate([
        ("what is our job", "what is our job"),
        ("Siyipatala through tax", "lethu siyayibadala thru taxt"),
        ("ANC Aina Mali", "enzi annamali"),
        ("Let us go", "let us go now"),
    ])]
    assert [r["id"] for r in cs.select_stretches(rows)] == [1, 2]
    assert len(cs.select_stretches(rows, cap=1)) == 1


def test_neighbours_are_the_phrases_just_before_and_after():
    phrases = [{"start": 0.0, "end": 2.0, "text": "one"}, {"start": 2.5, "end": 4.0, "text": "two"}, {"start": 4.5, "end": 6.0, "text": "three"},
               {"start": 10.0, "end": 12.0, "text": "four"}, {"start": 12.5, "end": 14.0, "text": "five"}, {"start": 14.5, "end": 16.0, "text": "six"}]
    assert cs.neighbours(phrases, 6.5, 9.5) == ("two three", "four five")
    assert cs.neighbours(phrases, 0.0, 2.0) == ("", "two three")


def test_phones_are_grouped_into_words_where_the_speaker_pauses_and_segments_found_by_overlap():
    timed = [("n", 0.0, 0.05), ("a", 0.05, 0.1), ("m", 0.4, 0.45), ("a", 0.45, 0.5)]
    assert cs.group_phones(timed) == "na | ma"
    segments = [{"segment_id": "a", "start_ms": 0, "end_ms": 5000}, {"segment_id": "b", "start_ms": 5000, "end_ms": 9000}]
    assert cs.segment_for(segments, 4.0, 7.0) == "b"
    assert cs.segment_for(segments, 20.0, 22.0) is None


def test_topic_memory_is_what_he_already_said_about_the_same_things_before_the_stretch():
    segments = [{"segment_id": "a", "start_ms": 10_000, "restored_text": "The grant for the grandmothers will rise. We love football."},
                {"segment_id": "b", "start_ms": 60_000, "restored_text": "The grant for children is small."},
                {"segment_id": "c", "start_ms": 200_000, "restored_text": "Grants after the stretch must not appear."}]
    memory = cs.topic_memory(segments, 100.0, "that grant imali yethu, the grant for gogo", limit=5)
    assert memory == ["[10s] The grant for the grandmothers will rise.", "[60s] The grant for children is small."]


def test_a_reading_is_resolved_only_when_it_is_sure_sounds_best_and_clearly_beats_rivals_that_mean_something_else():
    def cands(*pairs):
        return [{"english_meaning": m, "closeness": c} for m, c in pairs]

    clear = cands(("we are increasing the children's grant", 0.80), ("the people's money", 0.66))
    assert cs.is_resolved(clear, 0, "high")
    assert not cs.is_resolved(clear, 0, "medium")                                       # the model is not sure
    assert not cs.is_resolved(clear, 1, "high")                                         # it chose the one that sounds worse
    close = cands(("the children's grant", 0.71), ("the people's grant", 0.71))
    assert not cs.is_resolved(close, 0, "high")                                         # a tie between different meanings: uncertain
    same = cands(("the children's grant", 0.80), ("Grant the children's", 0.75))
    assert cs.is_resolved(same, 0, "high")                                              # a rival that means the same does not count against it
    assert not cs.is_resolved(clear, -1, "high")
    assert cs.is_resolved(cands(("x", None), ("y", None)), 0, "high")                   # no phone check possible at all: the model's word is all there is
    assert not cs.is_resolved(cands(("x", None), ("y", None)), 0, "high", checked=True) # a check was possible but did not run: not resolved


def _fixture(tmp_path, with_gap=False):
    native, analysis = tmp_path / "native_dub", tmp_path / "analysis"
    native.mkdir()
    analysis.mkdir()
    (native / "pass1_restored_transcript.json").write_text(json.dumps({"segments": [
        {"segment_id": "seg-1", "start_ms": 0, "end_ms": 11000, "restored_text": "The grant for the grandmothers will rise. Mali"},
        {"segment_id": "seg-2", "start_ms": 12000, "end_ms": 15000, "restored_text": "Then we go."}]}), encoding="utf-8")
    (native / "context_ledger.json").write_text(json.dumps({"context_ledger": {"summary": "A rally in Mpumalanga.", "story_beats": ["Grants"]}}), encoding="utf-8")
    (analysis / "azure_diarization.json").write_text(json.dumps({"phrases": [
        {"start": 1.0, "end": 4.0, "text": "What is our job"}, {"start": 7.0, "end": 10.0, "text": "Siyipatala through tax"}]}), encoding="utf-8")
    if with_gap:
        (native / "source_corrections.json").write_text(json.dumps({"corrections": [], "unresolved": [
            {"segment_id": "seg-1", "heard": "Nale yaban", "after_word": "Mali", "gap_start_s": 10.5, "gap_end_s": 11.8, "extends_to_ms": 11600}]}), encoding="utf-8")
    return native


class Provider:
    """Proposes a children's-grant reading and a people's-money reading; picks whichever the test says."""

    def __init__(self, pick=0, confidence="high"):
        self.calls, self.pick, self.confidence = [], pick, confidence

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["operation"] == "native_code_switch_candidates":
            return SimpleNamespace(data={"candidates": [
                {"words": [{"text": "yabantwana siyayinyusa", "language": "isizulu"}], "spelling_for_sounds": "yabantwana siyayinyusa",
                 "english_meaning": "we are increasing the children's grant", "fits_because": "continues the grants"},
                {"words": [{"text": "yabantu", "language": "isizulu"}], "spelling_for_sounds": "yabantu",
                 "english_meaning": "the people's money", "fits_because": "sounds a bit like it"}]})
        return SimpleNamespace(data={"chosen": self.pick, "confidence": self.confidence, "reason": "r"})


def test_build_solves_stretches_in_order_tests_candidates_against_the_audio_and_marks_clear_winners_resolved(tmp_path, monkeypatch):
    native = _fixture(tmp_path)
    provider = Provider()
    spoken = []

    def speak(text):
        spoken.append(text)
        return ["j", "a", "b", "a", "n", "t", "w", "a", "n", "a"] if "bantwana" in text else ["i", "m", "a", "l", "i"]

    path = cs.build(tmp_path, provider, zulu_reading=lambda a, b: "lethu siyayibadala thru taxt" if a >= 7 else "what is our job", afrikaans_reading=lambda a, b: "sy baddala",
                    phones=lambda a, b: [("j", 0.0, 0.1), ("a", 0.1, 0.2), ("b", 0.2, 0.3), ("a", 0.3, 0.4), ("n", 0.4, 0.5), ("t", 0.5, 0.6),
                                         ("a", 0.6, 0.7), ("n", 0.7, 0.8), ("a", 0.8, 0.9)],
                    speak=speak, progress=lambda m: None)
    doc = cs.load(path)
    stretch = doc["stretches"][0]
    assert stretch["segment_id"] == "seg-1"
    assert stretch["status"] == "resolved" and stretch["english_meaning"] == "we are increasing the children's grant"
    assert [c["closeness"] > 0 for c in stretch["candidates"]] == [True, True]
    assert stretch["candidates"][0]["closeness"] - stretch["candidates"][1]["closeness"] >= cs.MIN_CLOSENESS_MARGIN
    assert spoken == ["yabantwana siyayinyusa", "yabantu"]                             # each candidate was spoken and tested
    first = provider.calls[0]["payload"]
    assert first["rally_story"].startswith("A rally in Mpumalanga.")                   # the story of the speech is in the context
    assert first["stretch"]["said_just_before"] == "What is our job"
    assert first["earlier_sentences_about_the_same_things"] == []                       # its own paragraph is given whole, not repeated as memory
    # cached: the same inputs do not call the model again
    provider.calls.clear()
    assert cs.build(tmp_path, provider, zulu_reading=lambda a, b: "", afrikaans_reading=lambda a, b: "", phones=lambda a, b: [], speak=speak,
                    progress=lambda m: None) == path
    assert not provider.calls
    # a new version of the method rebuilds it
    monkeypatch.setattr(cs, "EVIDENCE_VERSION", cs.EVIDENCE_VERSION + 1)
    assert cs.build(tmp_path, provider, zulu_reading=lambda a, b: "lethu siyayibadala thru taxt" if a >= 7 else "what is our job", afrikaans_reading=lambda a, b: "",
                    phones=lambda a, b: [("j", 0.0, 0.1)], speak=speak, progress=lambda m: None) == path
    assert provider.calls
    view = cs.for_agent(doc, [stretch["segment_id"]])
    assert view["stretches"][0]["status"] == "resolved" and "or_perhaps" not in view["stretches"][0]


def test_a_stretch_whose_best_sounding_candidate_is_not_clearly_best_is_uncertain_and_listed_for_review(tmp_path):
    _fixture(tmp_path)
    provider = Provider(pick=0, confidence="high")
    path = cs.build(tmp_path, provider, zulu_reading=lambda a, b: "lethu siyayibadala thru taxt" if a >= 7 else "what is our job", afrikaans_reading=lambda a, b: "",
                    phones=lambda a, b: [("j", 0.0, 0.1), ("a", 0.1, 0.2)], speak=lambda text: ["j", "a"], progress=lambda m: None)
    doc = cs.load(path)
    assert doc["stretches"][0]["status"] == "uncertain"                                 # both candidates sound identical: a tie
    review = cs.review_items(doc)
    assert len(review) == 1 and len(review[0]["readings"]) == 2
    view = cs.for_agent(doc)
    assert view["stretches"][0]["or_perhaps"] == "the people's money"


def test_missed_speech_in_another_language_joins_the_dub_only_when_resolved(tmp_path):
    native = _fixture(tmp_path, with_gap=True)
    provider = Provider()
    sounds = [("j", 0.0, 0.1), ("a", 0.1, 0.2), ("b", 0.2, 0.3), ("a", 0.3, 0.4), ("n", 0.4, 0.5), ("t", 0.5, 0.6), ("w", 0.6, 0.7), ("a", 0.7, 0.8)]
    path = cs.build(tmp_path, provider, zulu_reading=lambda a, b: "nale yabantwana" if a >= 7 else "what is our job", afrikaans_reading=lambda a, b: "", phones=lambda a, b: sounds,
                    speak=lambda t: ["j", "a", "b", "a", "n", "t", "w", "a"] if "bantwana" in t else ["i", "m", "a"], progress=lambda m: None)
    doc = cs.load(path)
    gap = [s for s in doc["stretches"] if s.get("missed_speech")]
    assert len(gap) == 1 and gap[0]["status"] == "resolved"
    added = cs.add_missed_speech_to_corrections(tmp_path, lambda m: None)
    assert len(added) == 1 and added[0]["extends_to_ms"] == 11600 and "children's grant" in added[0]["replacement"]
    assert cs.add_missed_speech_to_corrections(tmp_path, lambda m: None) == []          # added once
    # an uncertain reading is never added
    gap[0]["status"] = "uncertain"
    (native / cs.FILE).write_text(json.dumps(doc), encoding="utf-8")
    saved = json.loads((native / "source_corrections.json").read_text(encoding="utf-8"))
    saved["corrections"] = []
    (native / "source_corrections.json").write_text(json.dumps(saved), encoding="utf-8")
    assert cs.add_missed_speech_to_corrections(tmp_path, lambda m: None) == []


def test_candidates_the_voice_could_not_test_are_recorded_and_the_stretch_stays_uncertain(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)

    def broken(text):
        raise RuntimeError("throttled")

    path = cs.build(tmp_path, Provider(), zulu_reading=lambda a, b: "lethu siyayibadala thru taxt" if a >= 7 else "what is our job",
                    afrikaans_reading=lambda a, b: "", phones=lambda a, b: [("j", 0.0, 0.1), ("a", 0.1, 0.2)], speak=broken, progress=lambda m: None)
    stretch = cs.load(path)["stretches"][0]
    assert stretch["status"] == "uncertain"                                             # sure of itself, but never checked against the audio
    assert all("throttled" in c["untested"] for c in stretch["candidates"])

def test_youtube_captions_are_evidence_for_a_stretch_but_never_what_selects_one(tmp_path):
    _fixture(tmp_path)
    provider = Provider()
    # the isiZulu recogniser disagrees with the English one only for the second phrase; YouTube's (Zulu-model) captions garble BOTH
    path = cs.build(tmp_path, provider, zulu_reading=lambda a, b: "lethu siyayibadala thru taxt" if a >= 7 else "What is our job",
                    afrikaans_reading=lambda a, b: "", phones=lambda a, b: [("j", 0.0, 0.1)], speak=lambda t: ["j"],
                    youtube_reading=lambda a, b: "imali yethu siyayibhadala through tax" if a >= 7 else "wat is awa jobe", progress=lambda m: None)
    doc = cs.load(path)
    assert [s["start_s"] for s in doc["stretches"]] == [7.0]                              # the first phrase was not dragged in by YouTube's garbling
    assert doc["stretches"][0]["youtube_reading"] == "imali yethu siyayibhadala through tax"
    assert provider.calls[0]["payload"]["stretch"]["youtube_caption_reading"] == "imali yethu siyayibhadala through tax"
    assert "YouTube" in provider.calls[0]["system_prompt"]

def test_candidates_are_spoken_as_plain_words_without_sentence_punctuation():
    assert cs.speakable("What is our job?") == "What is our job"
    assert cs.speakable("Nale yabantwana, siyayinyusa!") == "Nale yabantwana siyayinyusa"
    assert cs.speakable("ANC ayinamali… ANC ayibhadali") == "ANC ayinamali ANC ayibhadali"
    assert cs.speakable("i-grant") == "i-grant"                                          # hyphens inside a word stay


def test_independent_agreement_lets_a_medium_confidence_reading_count_as_resolved():
    youtube = "ku ANC imali ye grant iphuma la from you South Africa nale le yabantwana siyayinyusa,"
    assert cs.corroborated(["Naleyo", "yabantwana", "siyayinyusa"], [youtube, "Nale yabantu anase yusa", ""])      # YouTube has the same words
    assert not cs.corroborated(["Imali", "yabantu", "ANC", "ayinyusi"], [youtube])                                  # it does not
    assert cs.corroborated(["yabantwana"], [youtube])                                                              # the only distinctive word, found
    assert not cs.corroborated(["a", "b"], [youtube])                                                              # nothing distinctive to find
    clear = [{"english_meaning": "increase the children's one", "closeness": 0.705}, {"english_meaning": "the children's money", "closeness": 0.659}]
    assert not cs.is_resolved(clear, 0, "medium")                                                                  # medium alone is not enough
    assert cs.is_resolved(clear, 0, "medium", agreed=True)                                                         # medium plus an independent source is
    assert not cs.is_resolved(clear, 0, "low", agreed=True)                                                        # low never is
    close = [{"english_meaning": "x", "closeness": 0.70}, {"english_meaning": "y", "closeness": 0.69}]
    assert not cs.is_resolved(close, 0, "high")                                                                    # without an independent reading the sounds still need a clear margin
    close = [{"english_meaning": "x", "closeness": 0.70, "words": "alpha beta"}, {"english_meaning": "y", "closeness": 0.69, "words": "gamma delta"}]
    assert cs.is_resolved(close, 0, "medium", agreed=True, readings=["xx alpha yy"])                              # a reading has the choice's own words: a near tie does not block it
    assert not cs.is_resolved(close, 0, "medium", agreed=True, readings=["xx gamma yy"])                          # it has only the rival's words



def test_candidates_that_say_the_same_thing_do_not_count_as_rivals_to_each_other():
    # the real case: "siyayinyusa" (are increasing) was chosen; "sizoyinyusa" (will increase) sounds a little closer but says the same thing
    cands = [{"english_meaning": "and the one for the children, we are increasing it", "closeness": 0.736},
             {"english_meaning": "we are increasing the children's money", "closeness": 0.686},
             {"english_meaning": "and the one for the children, we will increase it", "closeness": 0.791}]
    cands = [dict(c, words=w) for c, w in zip(cands, ["Nale yabantwana siyayinyusa", "Nale yabantwana imali", "Nale yabantwana sizoyinyusa"])]
    assert not cs.is_resolved(cands, 0, "medium", checked=True, agreed=True)                          # agreed alone does not outvote a closer rival
    assert cs.is_resolved(cands, 0, "medium", checked=True, agreed=True, readings=["nale yabantwana siyayinyusa"])      # but the words that tell it from the rival do
    assert cs.is_resolved(cands, 0, "medium", checked=True, agreed=True, same_message=[2])             # the model says 2 is the same message: resolved
    assert not cs.is_resolved(cands, 0, "high", checked=True, same_message=[1])                        # sure but with no independent reading: a different closer rival (2) still blocks
    assert not cs.is_resolved(cands, 0, "medium", checked=True, agreed=False, same_message=[2])        # still needs the model's word or an independent source
    near = [dict(cands[0]), dict(cands[1], closeness=0.77), dict(cands[2])]
    assert not cs.is_resolved(near, 0, "high", checked=True, same_message=[2])                         # a different message must still be clearly worse


def test_a_reader_of_both_languages_outvotes_a_closer_sounding_rival_when_other_sources_agree():
    # the real case at 6:36: "Unga- confuse the two" (do not confuse) was the model's reading, backed by the English transcript, the isiZulu
    # recogniser and YouTube's "ungacon"; "Confuse the two" sounds closer to an English-trained phone recogniser, but loses the negation
    cands = [{"english_meaning": "Do not confuse the two.", "closeness": 0.536, "words": "Unga- confuse the two"},
             {"english_meaning": "Confuse the two.", "closeness": 0.708, "words": "Confuse the two"}]
    readings = ["Unga confuse the two.", "I confused itu?", "iphuma i izohamba our grand ayihambi i ANC yahamba ungacon"]
    assert cs.is_resolved(cands, 0, "medium", checked=True, agreed=True, readings=readings)             # "unga" is what tells the choice from the rival, and two sources have it
    assert not cs.is_resolved(cands, 0, "medium", checked=True, agreed=True, readings=["I confused itu?"])      # no reading has it: the sounds decide
    assert not cs.is_resolved(cands, 0, "medium", checked=True, agreed=False, readings=readings)       # no independent agreement: the sounds decide
    poor = [{"english_meaning": "a", "closeness": 0.30, "words": "unga"}, {"english_meaning": "b", "closeness": 0.60, "words": "other"}]
    assert not cs.is_resolved(poor, 0, "medium", checked=True, agreed=True, readings=readings)         # readings cannot outvote poor sounds
    # a word both candidates share proves nothing: the SASSA case, where the sounds and the English transcript both prefer "work for"
    sassa = [{"english_meaning": "All those who are on SASSA", "closeness": 0.60, "words": "All those who are on SASSA"},
             {"english_meaning": "All those who work for SASSA", "closeness": 0.78, "words": "All those who work for SASSA"}]
    assert not cs.is_resolved(sassa, 0, "medium", checked=True, agreed=True, readings=["U-oldosuek fossasa.", "All those who work for SASA"])


def test_the_english_transcriber_nearly_agreeing_with_the_choice_counts_as_an_independent_reading():
    assert cs.english_agrees_with("All those who work for SASA", ["All", "those", "who", "work", "for", "SASSA"])
    assert not cs.english_agrees_with("Rupert is", ["one thousand five hundred"])
    cands = [{"english_meaning": "All those who work for SASSA", "closeness": 0.78}, {"english_meaning": "All those who are on SASSA", "closeness": 0.60}]
    assert not cs.is_resolved(cands, 0, "medium", checked=True)                                         # medium alone is not enough
    assert cs.is_resolved(cands, 0, "medium", checked=True, english_agrees=True)                       # the transcriber and the sounds agree


def test_reassess_applies_the_current_rules_to_stored_evidence_without_a_model_call():
    doc = {"stretches": [{"start_s": 506.7, "english_reading": "Unga confuse the two.", "isizulu_reading": "I confused itu?",
                          "youtube_reading": "iphuma i izohamba our grand ayihambi i ANC yahamba ungacon",
                          "reconstruction": [{"text": "Unga-", "language": "isizulu"}, {"text": "confuse", "language": "english"}, {"text": "the two", "language": "english"}],
                          "confidence": "medium", "status": "uncertain",
                          "candidates": [{"english_meaning": "Do not confuse the two.", "words": "Unga- confuse the two", "closeness": 0.536},
                                         {"english_meaning": "Confuse the two.", "words": "Confuse the two", "closeness": 0.708}]},
                         {"start_s": 1.0, "reconstruction": [], "candidates": [], "status": "none"}]}
    assert cs.reassess(doc) == 1 and doc["stretches"][0]["status"] == "resolved"
    assert cs.reassess(doc) == 0                                                                        # nothing more to change
    doc["stretches"][0]["status"] = "resolved"
    doc["stretches"][0]["candidates"][0]["closeness"] = 0.10                                            # a stretch already resolved is never downgraded
    assert cs.reassess(doc) == 0 and doc["stretches"][0]["status"] == "resolved"


def test_the_agent_is_told_to_decide_an_uncertain_stretch_and_is_given_the_reasoning():
    doc = {"schema_version": cs.SCHEMA_VERSION, "stretches": [
        {"segment_id": "s1", "start_s": 1.0, "english_reading": "Unga confuse", "reconstruction": [{"text": "Unga-", "language": "isizulu"}],
         "english_meaning": "Do not confuse", "status": "uncertain", "alternative": "Confuse", "reason": "The recognisers and the caption have unga-."}]}
    out = cs.for_agent(doc)
    assert "never drop an uncertain stretch" in out["note"].lower() and "do not rely" not in out["note"].lower()
    row = out["stretches"][0]
    assert row["or_perhaps"] == "Confuse" and row["why"].startswith("The recognisers")
