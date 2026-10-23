from pathlib import Path

from mathula_tv.pronunciation_search import Candidate, Context, Scorer, agreement, neighbours, search


def _scorer(tmp_path, ideal, contexts=None):
    """A fake voice whose 'phones' are the spelling's letters, so closeness to ``ideal`` is easy to compute."""
    def synthesize(text, pitch, wav):
        wav.write_text(text, encoding="utf-8")

    def recognise(wav):
        return list(Path(wav).read_text(encoding="utf-8").replace(" ", "").replace(",", "").lower())

    def similarity(haystack, needle):
        matches = sum(1 for a, b in zip(haystack, needle) if a == b)
        return matches / max(len(needle), len(haystack), 1)

    return Scorer(synthesize, recognise, similarity, list(ideal), contexts or [Context("", "")], tmp_path)


def test_neighbours_are_single_edits_of_the_spelling():
    near = neighbours("Jacob")
    assert "Jacob" not in near and len(near) > 3
    assert "Jakob" in near and "Jaacob" in near                       # c -> k, doubled vowel
    assert "Kheyi" not in neighbours("Key") and "Khey" in neighbours("Key")      # k -> kh aspiration
    assert all(0 < len(n) <= 45 for n in neighbours("Em Khey Phaati"))


def test_a_comma_is_not_doubled_when_the_sentence_already_has_one():
    context = Context("wowayengumengameli u", ", futhi manje")
    assert context.render(Candidate("Jeykhob Zuma", ", ")) == "wowayengumengameli uJeykhob Zuma, futhi manje"
    assert Context("seqembu, ", "inqume").render(Candidate("Em Khey Phaathi", ", ")) == "seqembu, Em Khey Phaathi, inqume"
    assert Context("seqembu, ", "inqume").render(Candidate("Em Khey Phaathi", " ")) == "seqembu, Em Khey Phaathi inqume"
    assert Context("ye ", ".").render(Candidate("Em Khey Phaathi", ", ")) == "ye Em Khey Phaathi."


def test_the_search_climbs_towards_the_reference_and_stays_inside_its_budget(tmp_path):
    scorer = _scorer(tmp_path, "jeykhobzuma")
    ranked = search(scorer, ["Jacob Zuma"], width=3, depth=4, budget=60)
    assert scorer.evaluations <= 60
    best_score, _best = ranked[0]
    assert best_score >= scorer.score(Candidate("Jacob Zuma")) and best_score > 0.5


def test_agreement_measures_how_often_the_score_prefers_the_listeners_choice(tmp_path):
    scorer = _scorer(tmp_path, "jeykhobzuma")
    result = agreement(scorer, [(Candidate("Jeykhob Zuma"), Candidate("Jeyikab Zuma")),
                                (Candidate("Jeykhob Zuma"), Candidate("Zzzz"))])
    assert result["agreement"] == 1.0 and result["pairs"][0]["chosen_score"] > result["pairs"][0]["rejected_score"]
    wrong = agreement(scorer, [(Candidate("Jeyikab Zuma"), Candidate("Jeykhob Zuma"))])
    assert wrong["agreement"] == 0.0


def test_scores_are_cached_so_a_candidate_is_synthesised_once(tmp_path):
    scorer = _scorer(tmp_path, "abc", contexts=[Context("", ""), Context("x ", " y")])
    scorer.score(Candidate("abc"))
    scorer.score(Candidate("abc"))
    assert scorer.evaluations == 1
