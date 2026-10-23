import re
import wave
from pathlib import Path

from mathula_tv.pronunciation_search import (
    Candidate, Context, NameWindows, Scorer, agreement, neighbours, normalise_phones, search, slice_wav,
)


def _fake_voice(tmp_path):
    """Writes a real (silent, 2 s) wav per sentence and remembers its text; 'recognises' the name's own letters only."""
    texts: dict[str, str] = {}

    def synthesize(marked, pitch, wav):
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x00\x00" * 48000)
        texts[wav.stem] = marked
        return {"n0": 500.0, "n1": 1500.0}

    def recognise(piece):
        marked = texts[piece.stem.removesuffix(".name")]
        name = re.search(r'<bookmark mark="n0"/>(.*?)<bookmark mark="n1"/>', marked).group(1)
        return list(name.replace(" ", "").lower())

    return NameWindows(synthesize, recognise, tmp_path), texts


def _similarity(name, reference):
    matches = sum(1 for a, b in zip(name, reference) if a == b)
    return matches / max(len(name), len(reference), 1)


def _scorer(tmp_path, ideal, contexts=None, references=None):
    windows, _ = _fake_voice(tmp_path)
    return Scorer(windows, _similarity, references or [list(ideal)], contexts or [Context("a ", "b")])


def test_neighbours_are_single_edits_of_the_spelling():
    near = neighbours("Jacob")
    assert "Jacob" not in near and len(near) > 3
    assert "Jakob" in near and "Jaacob" in near                       # c -> k, doubled vowel
    assert "Kheyi" not in neighbours("Key") and "Khey" in neighbours("Key")      # k -> kh aspiration
    assert all(0 < len(n) <= 45 for n in neighbours("Em Khey Phaati"))


def test_bookmarks_bracket_exactly_the_name_including_a_glued_prefix():
    c = Candidate("Jeykhob Zuma")
    assert Context("Lokhu owayengumengameli u", ", futhi").render_marked(c) == \
        'Lokhu owayengumengameli <bookmark mark="n0"/>uJeykhob Zuma<bookmark mark="n1"/>, futhi'
    assert Context("Izindaba: i-", "ishintshe").render_marked(Candidate("Em Khey Phaathi", ", ")) == \
        'Izindaba: <bookmark mark="n0"/>i-Em Khey Phaathi<bookmark mark="n1"/>, ishintshe'
    assert Context("seqembu, ", "inqume").render_marked(Candidate("Em Khey Phaathi", " ")) == \
        'seqembu, <bookmark mark="n0"/>Em Khey Phaathi<bookmark mark="n1"/> inqume'
    assert Context("ye ", ".").render_marked(Candidate("Em Khey Phaathi", ", ")) == \
        'ye <bookmark mark="n0"/>Em Khey Phaathi<bookmark mark="n1"/>.'


def test_a_comma_is_not_doubled_when_the_sentence_already_has_one():
    assert Context("wowayengumengameli u", ", futhi").render(Candidate("Jeykhob Zuma", ", ")) == "wowayengumengameli uJeykhob Zuma, futhi"
    assert Context("seqembu, ", "inqume").render(Candidate("Em Khey Phaathi", ", ")) == "seqembu, Em Khey Phaathi, inqume"
    assert Context("ye ", ".").render(Candidate("Em Khey Phaathi", ", ")) == "ye Em Khey Phaathi."


def test_modifiers_such_as_aspiration_and_length_do_not_count_as_differences():
    assert normalise_phones(["kʰ", "o", "uː", "t͡ɕ", "l̪", "ʲ"]) == ["k", "o", "u", "tɕ", "l"]


def test_only_the_names_own_audio_is_scored_not_the_words_around_it(tmp_path):
    # Sentences around the name are identical; with whole-sentence scoring a lucky neighbour could win. Here the ideal
    # spelling wins whatever surrounds it, because only the bookmarked slice is recognised.
    contexts = [Context("nje kokuba ", ", ongumengameli"), Context("zzz ", "yyy")]
    scorer = _scorer(tmp_path, "jeykhobzuma", contexts=contexts)
    assert scorer.score(Candidate("Jeykhob Zuma")) == 1.0
    assert scorer.score(Candidate("Jeyikab Zuma")) < 1.0


def test_the_slice_is_cut_around_the_markers(tmp_path):
    windows, _ = _fake_voice(tmp_path)
    windows.phones(Context("a ", "b"), Candidate("xy"))
    piece = next(tmp_path.glob("*.name.wav"))
    with wave.open(str(piece), "rb") as w:
        assert abs(w.getnframes() / w.getframerate() - (1.0 + 0.040 + 0.120)) < 0.01      # name + lead + trail


def test_the_search_climbs_towards_the_reference_and_stays_inside_its_budget(tmp_path):
    scorer = _scorer(tmp_path, "jeykhobzuma")
    ranked = search(scorer, ["Jacob Zuma"], width=3, depth=4, budget=60)
    assert scorer.evaluations <= 60
    assert ranked[0][0] >= scorer.score(Candidate("Jacob Zuma")) and ranked[0][0] > 0.5


def test_agreement_measures_how_often_the_score_prefers_the_listeners_choice(tmp_path):
    scorer = _scorer(tmp_path, "jeykhobzuma")
    result = agreement(scorer, [(Candidate("Jeykhob Zuma"), Candidate("Jeyikab Zuma")),
                                (Candidate("Jeykhob Zuma"), Candidate("Zzzz"))])
    assert result["agreement"] == 1.0 and result["pairs"][0]["chosen_score"] > result["pairs"][0]["rejected_score"]
    assert agreement(scorer, [(Candidate("Jeyikab Zuma"), Candidate("Jeykhob Zuma"))])["agreement"] == 0.0


def test_scores_are_cached_so_a_candidate_is_synthesised_once_per_context(tmp_path):
    scorer = _scorer(tmp_path, "abc", contexts=[Context("", ""), Context("x ", " y")])
    scorer.score(Candidate("abc"))
    scorer.score(Candidate("abc"))
    assert scorer.evaluations == 1 and scorer.windows.syntheses == 2


def test_two_references_are_averaged(tmp_path):
    scorer = _scorer(tmp_path, "abc", references=[list("abc"), list("xyz")])
    assert abs(scorer.score(Candidate("abc")) - 0.5) < 1e-9


def test_slice_wav_clamps_to_the_file(tmp_path):
    windows, _ = _fake_voice(tmp_path)
    windows.phones(Context("a ", "b"), Candidate("xy"))
    wav = next(p for p in tmp_path.glob("*.wav") if not p.name.endswith(".name.wav"))
    out = slice_wav(wav, -500, 99_999, tmp_path / "all.wav")
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() == 48000
