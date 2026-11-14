from mathula_tv.dialogue_view import NOTE, build, murmur_lines


def seg(i, speaker, start, end, text):
    return {"segment_id": f"seg-{i:05d}", "speaker_id": speaker, "start_ms": int(start * 1000), "end_ms": int(end * 1000), "restored_text": text}


def test_turns_are_runs_of_one_speaker_and_a_question_is_answered_by_the_next_turn():
    segments = [seg(1, "A", 0, 10, "Welcome. What do you make of the statement?"), seg(2, "B", 11, 30, "I think it is weak."), seg(3, "B", 30, 40, "And that is the point."),
                seg(4, "A", 41, 50, "Thank you, sir.")]
    turns = build(segments, None, {"A": "the anchor", "B": "the analyst"})
    assert [t["turn"] for t in turns] == [1, 2, 3] and turns[1]["segments"] == ["seg-00002", "seg-00003"]
    assert turns[0]["asks"] == "What do you make of the statement?" and turns[1]["answers"] == "What do you make of the statement?"
    assert "answers" not in turns[2] and turns[1]["speaker"] == "the analyst"


def test_a_short_murmur_line_inside_a_turn_is_another_speaker_listening():
    lines = [(100.0, 103.0, "the right of the public to safety."), (103.0, 103.6, ">> Mm-hmm."), (103.6, 106.0, ">> And then see there are two rights")]
    segments = [seg(1, "A", 90, 104, "...to safety."), seg(2, "B", 105, 120, "And then see there are two rights.")]
    turns = build(segments, lines)
    assert turns[0]["murmur_of_another_speaker_inside"] == ["Mm-hmm. at 103.0 s"]          # the case the agent copied into the Zulu
    assert "murmur_of_another_speaker_inside" not in turns[1]


def test_a_turn_that_opens_with_the_speakers_own_murmur_is_marked_and_ordinary_lines_are_not():
    lines = [(151.69, 154.0, ">> Mhm. There's quite a lot that you've"), (160.0, 162.0, ">> Yes, indeed.")]
    found = murmur_lines(lines)
    assert found == [{"at_s": 151.69, "text": "Mhm.", "alone": False}]
    turns = build([seg(1, "A", 150, 200, "There's quite a lot that you've mentioned.")], lines)
    assert turns[0]["opens_with_the_speakers_own_murmur"] == "Mhm."


def test_the_note_tells_the_agent_what_to_do_with_a_murmur():
    assert "leave it out" in NOTE and "Yebo" in NOTE


def test_the_workbench_puts_the_dialogue_in_the_opened_clip_with_roles_and_the_murmur(tmp_path):
    import json

    from mathula_tv.dub_workbench import Workbench

    job_dir = tmp_path / "work" / "jobs" / "job1" / "input" / "youtube_captions"
    job_dir.mkdir(parents=True)
    events = [{"tStartMs": 100000, "dDurationMs": 3000, "segs": [{"utf8": "the right of the public to safety."}]},
              {"tStartMs": 103000, "dDurationMs": 600, "segs": [{"utf8": ">> Mm-hmm."}]}]
    (job_dir / "captions-x.en-orig.json3").write_text(json.dumps({"events": events}), encoding="utf-8")

    class Store:
        root = tmp_path / "work" / "experience"
        def load_corpus(self):
            return {"clips": {"c1": {"job_id": "job1"}}}

    class Bench:
        store = Store()
        _add_dialogue = Workbench._add_dialogue

    opened = {"segments": [{"segment_id": "seg-1", "speaker_id": "A", "start_ms": 90000, "end_ms": 104000, "english": "...to safety."},
                           {"segment_id": "seg-2", "speaker_id": "B", "start_ms": 105000, "end_ms": 120000, "english": "Is that right?"}]}
    ledger = {"speaker_roles": [{"role": "Independent analyst", "speaker_id": "A"}, {"role": "News anchor", "speaker_id": "B"}]}
    Bench()._add_dialogue(opened, "c1", ledger)
    turns = opened["dialogue"]
    assert turns[0]["speaker"] == "Independent analyst" and turns[0]["murmur_of_another_speaker_inside"] == ["Mm-hmm. at 103.0 s"]
    assert turns[1]["speaker"] == "News anchor" and "dialogue_note" in opened
    solo = {"segments": [{"segment_id": "s", "speaker_id": "A", "start_ms": 0, "end_ms": 1000, "english": "Hello."}]}
    Bench()._add_dialogue(solo, "c1", ledger)
    assert "dialogue" not in solo                                                              # a monologue has no conversation to show
