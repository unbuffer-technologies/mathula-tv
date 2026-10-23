import json
from pathlib import Path

import pytest

from mathula_tv.pronunciation_assess import cached, gop_excess, phones_to_ids
from mathula_tv.pronunciation_search import AssessmentScorer, Candidate, Context, NameWindows, agreement

VOCAB = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "k": 4, "kʰ": 5, "dʒ": 6, "d": 7, "ʒ": 8, "eɪ": 9, "e": 10,
         "ɪ": 11, "ə": 12, "b": 13, "uː": 14, "u": 15, "z": 16, "m": 17}


def test_phones_are_matched_longest_first_and_spaces_are_ignored():
    assert phones_to_ids("dʒ eɪ k ə b", VOCAB) == [6, 9, 4, 12, 13]       # "dʒ" is one phone, not "d" + "ʒ"
    assert phones_to_ids("kʰuːm", VOCAB) == [5, 14, 17]                      # "kʰ" and "uː" are single phones


def test_an_unknown_phone_is_reported_not_silently_dropped():
    with pytest.raises(ValueError, match="not in the recogniser's vocabulary"):
        phones_to_ids("dʒeɪxb", VOCAB)


def test_special_tokens_are_not_phones():
    with pytest.raises(ValueError):
        phones_to_ids("<pad>", VOCAB)


def _peaked(sequence, vocab_size=18, frames_per_phone=3, blank=1):
    """Log-probabilities that confidently spell out ``sequence`` (a blank frame between phones)."""
    import torch

    rows = []
    for phone in sequence:
        for _ in range(frames_per_phone):
            rows.append(phone)
        rows.append(blank)
    probs = torch.full((len(rows), vocab_size), 0.002)
    for t, phone in enumerate(rows):
        probs[t, phone] = 1.0
    return torch.log(probs / probs.sum(dim=-1, keepdim=True))


def test_audio_that_says_the_expected_phones_has_a_smaller_excess_than_audio_that_says_something_else():
    pytest.importorskip("torch")
    expected = [6, 9, 4, 12, 13]
    right = gop_excess(_peaked(expected), expected, blank=1)
    wrong = gop_excess(_peaked([6, 10, 5, 11, 13]), expected, blank=1)      # different vowels, aspirated k
    assert right < wrong and right < 0.5 and wrong > 2 * right


def test_one_wrong_phone_costs_less_than_several():
    pytest.importorskip("torch")
    expected = [6, 9, 4, 12, 13]
    one = gop_excess(_peaked([6, 9, 4, 11, 13]), expected, blank=1)
    many = gop_excess(_peaked([7, 10, 5, 11, 15]), expected, blank=1)
    assert one < many


def test_the_cache_assesses_each_wav_once_and_keeps_the_details(tmp_path):
    calls = []

    def assess(wav):
        calls.append(wav)
        return {"score": 0.75, "text": "hello"}

    run = cached("fake", assess, tmp_path / "store")
    wav = tmp_path / "a.name.wav"
    assert run(wav) == 0.75 and run(wav) == 0.75 and len(calls) == 1
    assert json.loads((tmp_path / "store" / "a.name.fake.json").read_text(encoding="utf-8"))["text"] == "hello"


def test_an_assessment_scorer_plugs_into_the_agreement_check(tmp_path):
    import wave

    texts = {}

    def synthesize(marked, pitch, wav):
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x00\x00" * 48000)
        texts[wav.stem] = marked
        return {"n0": 500.0, "n1": 1500.0}

    def assess(piece: Path) -> float:                      # a fake assessor that likes the spelling "good"
        return 1.0 if "good" in texts[piece.stem.removesuffix(".name")] else 0.2

    scorer = AssessmentScorer(NameWindows(synthesize, lambda wav: (), tmp_path), assess, [Context("a ", "b"), Context("c ", "d")])
    result = agreement(scorer, [(Candidate("good"), Candidate("bad")), (Candidate("bad"), Candidate("good"))])
    assert result["agreement"] == 0.5 and scorer.evaluations == 2


def test_a_candidate_must_look_good_to_every_assessor():
    from mathula_tv.pronunciation_assess import CombinedScorer, azure_goodness, gop_goodness

    class Fixed:
        def __init__(self, values):
            self.values = values

        def score(self, candidate):
            return self.values[candidate]

    azure = Fixed({"a": 0.99, "b": 0.99, "c": 0.80})
    gop = Fixed({"a": -0.2, "b": -2.5, "c": -0.1})
    combined = CombinedScorer([(azure, azure_goodness), (gop, gop_goodness)])
    scores = {c: combined.score(c) for c in "abc"}
    assert scores["a"] > scores["b"] and scores["a"] > scores["c"]      # b fools Azure only, c fools GOP only
    assert combined.evaluations == 3 and abs(combined.detail["b"][0] - 0.95) < 1e-6


def test_goodness_transforms_are_bounded():
    from mathula_tv.pronunciation_assess import azure_goodness, gop_goodness

    assert azure_goodness(1.0) == 1.0 and azure_goodness(0.5) == 0.0 and 0 < azure_goodness(0.9) < 1
    assert gop_goodness(0.3) == 1.0 and 0 < gop_goodness(-2.0) < gop_goodness(-0.5) < 1


def test_takes_are_joined_with_a_pause_after_each(tmp_path):
    import wave

    from mathula_tv.pronunciation_assess import concatenate_takes

    paths = []
    for i in range(2):
        path = tmp_path / f"t{i}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x01\x00" * 24000)                 # 1 s
        paths.append(path)
    out = concatenate_takes(paths, tmp_path / "all.wav", gap_s=0.5)
    with wave.open(str(out), "rb") as w:
        assert abs(w.getnframes() / w.getframerate() - 3.0) < 0.01          # two 1 s takes + two 0.5 s pauses


def test_reduced_vowels_are_tried_as_every_full_vowel():
    from mathula_tv.pronunciation_assess import ipa_variants, phone_tokens

    tokens = phone_tokens("dʒeɪkəb", VOCAB)
    assert tokens == ["dʒ", "eɪ", "k", "ə", "b"]
    variants = ["".join(v) for v in ipa_variants(tokens)]
    assert variants[0] == "dʒeɪkəb" and "dʒeɪkob" in variants and "dʒeɪkab" in variants
    assert all(v.startswith("dʒeɪk") and v.endswith("b") for v in variants)           # only the reduced vowel changes
    assert len(variants) == len(set(variants))


def test_the_speakers_audio_picks_the_vowel_they_said():
    pytest.importorskip("torch")
    from mathula_tv.pronunciation_assess import choose_expected, phone_tokens

    vocab = {**VOCAB, "o": 18, "a": 19}
    said = [vocab[t] for t in ["dʒ", "eɪ", "k", "o", "b"]]            # the speaker says a full "o"
    picked = choose_expected(_peaked(said, vocab_size=20), phone_tokens("dʒeɪkəb", vocab), vocab, blank=1)
    assert picked["chosen"] == "dʒeɪkob" and picked["chosen_excess"] < picked["base_excess"]


def test_a_chosen_token_sequence_is_regrouped_into_the_words_of_the_name():
    from mathula_tv.pronunciation_assess import regroup

    assert regroup(["dʒ", "eɪ", "k", "o", "b", "z", "uː", "m", "a"], "dʒeɪkəb zuːmə", VOCAB) == ["dʒeɪkob", "zuːma"]


def test_a_research_pronunciation_note_is_cleaned_to_the_models_symbols():
    from mathula_tv.pronunciation_assess import sanitize_ipa

    assert sanitize_ipa("/ˈkʰeɪ.ɪʱ zuː/", {"k": 1, "kʰ": 2, "eɪ": 3, "ɪ": 4, "z": 5, "uː": 6}) == "kʰeɪɪ zuː"
