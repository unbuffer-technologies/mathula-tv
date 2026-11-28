

def test_a_burst_of_the_next_word_at_the_end_of_a_cut_span_is_taken_off_and_a_real_last_word_is_not():
    import numpy as np
    from mathula_tv import voice_takes as vt

    rate = 24000
    rng = np.random.default_rng(1)
    speech = lambda s: (rng.normal(0, 0.08, int(s * rate))).astype(np.float32)
    quiet = lambda s: (rng.normal(0, 0.0003, int(s * rate))).astype(np.float32)
    leaked = np.concatenate([speech(1.5), quiet(0.5), speech(0.04)])                       # the year, a real silence, and the first 40 ms of "Nge-"
    cut, removed = vt.strip_leaked_onset(leaked, rate, not_before_s=1.0)
    assert removed > int(0.2 * rate) and abs(len(cut) / rate - 1.75) < 0.03 and len(cut) + removed == len(leaked)          # cut in the middle of the quiet
    clean = np.concatenate([speech(1.5), quiet(0.5)])
    assert vt.strip_leaked_onset(clean, rate)[1] == 0                                       # it ends in quiet: nothing to do
    pause_then_word = np.concatenate([speech(1.5), quiet(0.4), speech(0.25)])               # a short last word after a pause is the span's own word
    assert vt.strip_leaked_onset(pause_then_word, rate, not_before_s=1.9)[1] == 0           # the aligner puts its start at 1.9 s: the cut may not fall before 1.9 + 0.15
    assert vt.strip_leaked_onset(np.concatenate([speech(1.5), quiet(0.05), speech(0.04)]), rate)[1] == 0          # too little quiet to be sure
    assert vt.strip_leaked_onset(np.concatenate([speech(1.0), quiet(0.5), speech(0.8)]), rate)[1] == 0            # longer than a fragment


def _tail_fixture():
    import numpy as np

    rate = 24000
    rng = np.random.default_rng(7)
    speech = lambda s, a=0.08: rng.normal(0, a, int(s * rate)).astype(np.float32)
    quiet = lambda s: rng.normal(0, 0.0003, int(s * rate)).astype(np.float32)
    return rate, speech, quiet


def test_the_real_end_of_the_last_word_and_the_next_onset_are_found_in_the_signal_when_the_aligner_puts_them_together():
    import numpy as np
    from mathula_tv import voice_takes as vt

    rate, speech, quiet = _tail_fixture()
    core = np.concatenate([speech(2.0), quiet(0.45), speech(0.4)])                           # the last word ends at 2.0 s; the next word begins at 2.45 s
    found = vt.locate_tail(core, rate, last_word_start_s=1.4, next_word_start_s=2.9)           # the aligner: the word ends where the next begins, 0.45 s late (here its next start is 0.45 s late too)
    assert found and abs(found["speech_end_s"] - 2.0) < 0.03 and abs(found["next_onset_s"] - 2.45) < 0.03 and abs(found["pause_s"] - 0.45) < 0.04
    assert vt.locate_tail(np.concatenate([speech(2.0), quiet(0.04), speech(0.4)]), rate, 1.4, 2.04) is None          # no pause to cut in: continuous speech into the next word
    assert vt.locate_tail(np.concatenate([speech(2.0), quiet(0.45)]), rate, 1.4, 2.45) is None                      # nothing after it: the next word was not said


def test_the_longest_quiet_is_the_pause_not_a_closure_inside_the_next_word():
    import numpy as np
    from mathula_tv import voice_takes as vt

    rate, speech, quiet = _tail_fixture()
    core = np.concatenate([speech(2.0), quiet(0.5), speech(0.12), quiet(0.09), speech(0.3)])          # the pause, then a word with a closure in it
    found = vt.locate_tail(core, rate, 1.4, 2.62)
    assert abs(found["speech_end_s"] - 2.0) < 0.03 and abs(found["pause_s"] - 0.5) < 0.04


def test_a_cut_checks_its_own_end():
    import numpy as np
    from mathula_tv import voice_takes as vt

    rate, speech, quiet = _tail_fixture()
    assert vt.verify_cut_end(np.concatenate([speech(2.0), quiet(0.2)]), rate) is None                                   # ends in quiet
    assert vt.verify_cut_end(np.concatenate([speech(2.0), quiet(0.5), speech(0.04)]), rate) == "leaked_onset"            # a sliver of the next word
    assert vt.verify_cut_end(speech(2.0), rate) == "cut_in_sound"                                                         # a cut inside a word


def test_a_sound_no_word_accounts_for_is_found_and_replaced_by_the_rooms_quiet_and_a_word_is_not():
    import numpy as np
    from mathula_tv import voice_takes as vt

    rate = 24000
    rng = np.random.default_rng(11)
    speech = lambda s: rng.normal(0, 0.08, int(s * rate)).astype(np.float32)
    quiet = lambda s: rng.normal(0, 0.0004, int(s * rate)).astype(np.float32)
    audio = np.concatenate([speech(2.0), quiet(0.7), speech(0.3), quiet(0.4), speech(2.0)])          # a sentence, a pause, a 0.3 s sound, a pause, the next sentence
    words = [(0.0, 2.0), (3.4, 5.4)]                                                                    # what the aligner says the text is: the sound is no word's
    found = vt.find_stray_sounds(audio, rate, words)
    assert len(found) == 1 and abs(found[0][0] / rate - 2.7) < 0.03 and abs((found[0][1] - found[0][0]) / rate - 0.3) < 0.03
    assert vt.find_stray_sounds(audio, rate, words + [(2.6, 3.1)]) == []                                # a word the aligner places there: it is speech
    assert vt.find_stray_sounds(audio, rate, [(0.0, 2.0), (2.9, 5.4)]) == []                            # within 0.2 s of a word: its onset, not an extra sound
    cleaned = vt.quiet_over(audio, rate, found)
    assert np.sqrt((cleaned[int(2.72 * rate): int(2.98 * rate)] ** 2).mean()) < 0.003 and np.array_equal(cleaned[: int(2.0 * rate)], audio[: int(2.0 * rate)])          # quiet there, the speech untouched
    assert len(cleaned) == len(audio)
