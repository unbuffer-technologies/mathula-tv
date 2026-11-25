from mathula_tv import omnivoice_tts as ot
from mathula_tv import span_cuts as sc


def test_a_long_unit_is_cut_where_the_speaker_paused_and_not_where_the_length_overflows():
    lengths = [118, 140, 150, 31, 170, 110]                      # the six sentences of prod-119 unit-01
    # the speaker paused after sentences 1 and 2 and 5, and ran straight on after 3 and 4 (the 0:27 join)
    cuts = sc.choose_cuts(lengths, [0.3, 0.4, 0.0, 0.0, 0.3])
    assert cuts == [1, 4]                                         # [S1 S2] [S3 S4 S5] [S6]: nothing is cut between "Ayanda." and "Ngesonto"
    greedy = sc.choose_cuts(lengths, [0.0] * 5)                    # nobody paused anywhere: only length decides, as before
    assert 3 not in cuts and len(greedy) >= 1


def test_a_span_may_run_past_the_soft_limit_to_reach_a_pause_but_never_past_the_hard_one():
    lengths = [140, 140, 100, 100]                               # 140+140 fits 280; adding the third (381) is over the soft limit, under the hard one
    assert sc.choose_cuts(lengths, [0.0, 0.0, 0.5]) == [2]         # held on to the pause after sentence 3, 381 chars
    assert sc.choose_cuts([200, 200, 200], [0.0, 0.0]) != []       # 600 chars cannot be one span
    assert sc.choose_cuts([100, 120], [0.0]) == []                 # short enough for one generation: no cut at all


def test_the_speakers_pauses_are_placed_at_the_sentence_boundary_they_belong_to_by_position():
    zulu = "Sentensi yokuqala ende kakhulu inamagama amaningi. Okwesibili kufushane. Okwesithathu kuphinde kube side kakhulu futhi."
    words, at = [], 0.0
    for i, w in enumerate("The first sentence is quite long and has many words. The second is short. The third is long again and ends here.".split()):
        words.append({"text": w, "start": at, "end": at + 0.3})
        at += 0.3 + (0.5 if w in ("words.",) else 0.0)               # a 0.5 s pause after the first sentence and none after the second
    gaps = sc.boundary_gaps(zulu, words)
    assert gaps is not None and len(gaps) == 2 and gaps[0] >= 0.5 and gaps[1] == 0.0


def test_too_little_to_go_on_gives_none_and_the_spans_are_cut_for_length_as_before():
    assert sc.boundary_gaps("One sentence only.", [{"text": "a", "start": 0, "end": 1}] * 10) is None
    long_text = [" ".join(["Isigaba eside kakhulu sokuzama."] * 9)]                 # one piece, nine sentences of ~31 chars
    spans, gaps = ot.to_spans(long_text)
    assert len(spans) >= 2 and all(len(s) <= ot.MAX_SPAN_CHARS for s in spans) and gaps == [ot.SPAN_GAP_MS] * (len(spans) - 1)


def test_to_spans_uses_the_sentence_gaps_and_ignores_them_when_their_count_does_not_match():
    sentences = ["Okokuqala kuyahamba kuze kuphele nje ngoba kunjalo." for _ in range(4)]
    spans, _ = ot.to_spans([" ".join(sentences)], [0.0, 0.5, 0.0])
    assert len(spans) == len(sentences)                                         # one sentence, one span: the voice is not asked to say two at a time
    big = ["Sentence number %d is a fairly long one with enough words to take up room in a span." % i for i in range(8)]
    spans, gaps = ot.to_spans([" ".join(big)], [0, 0, 0, 0.6, 0, 0, 0])
    assert len(spans) == 8 and all(s.count("Sentence number") == 1 for s in spans) and gaps[3] == sc.join_gap_ms(0.6)          # every sentence is a span, the join after the fourth as long as the speaker's pause
    again, again_gaps = ot.to_spans([" ".join(big)], [0.6])                     # the wrong number of gaps: the same cuts, with the default gap
    assert len(again) == 8 and again_gaps == [ot.SPAN_GAP_MS] * 7
    held, held_gaps = ot.to_spans([" ".join(big[:2]), 700, " ".join(big[2:4])], [0.0, 0.0, 0.0])
    assert held_gaps[1] == 700 and len(held) == 4                                 # a held pause still cuts, with its own silence


def test_a_join_is_as_long_as_the_speakers_pause_and_as_short_as_possible_where_the_speaker_ran_on():
    assert sc.join_gap_ms(0.4) == 200 and sc.join_gap_ms(2.5) == 700 and sc.join_gap_ms(0.0) == -sc.PAD_MS and sc.join_gap_ms(0.1) == -sc.PAD_MS
    big = ["Sentence number %d is a fairly long one with enough words to take up room in a span." % i for i in range(8)]
    spans, gaps = ot.to_spans([" ".join(big)], [0, 0, 0, 0.6, 0, 0, 0])
    assert gaps[3] == sc.join_gap_ms(0.6) and gaps[0] == -sc.PAD_MS               # the cut at the 0.6 s pause is 0.6 s long, padding included
    spans, gaps = ot.to_spans([" ".join(big)], [0.0] * 7)
    assert len(spans) == 8 and all(g == -sc.PAD_MS for g in gaps)                 # nobody paused anywhere: every join is tight


def test_audio_is_joined_without_any_filler_where_the_gap_cancels_the_models_padding():
    import numpy as np
    from mathula_tv import voice_takes as vt

    a, b = (np.ones(2400, dtype=np.float32) * 0.1, 0, 0), (np.ones(2400, dtype=np.float32) * 0.1, 0, 0)
    audio, starts = vt.assemble([a, b], [-200], 24000, np.zeros(10, dtype=np.float32), 40.0)
    assert len(audio) == 4800 and starts == [0.0, 0.1]
    audio, starts = vt.assemble([(a[0], 2400, 2400), (b[0], 2400, 2400)], [-200], 24000, np.zeros(10, dtype=np.float32), 40.0)
    assert len(audio) == 2400 + 2400 + 2400 + 2400 and starts == [0.0, 0.1]          # the two 100 ms pads and the -200 ms gap cancel: the second span follows the first at once
