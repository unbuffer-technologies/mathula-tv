from mathula_tv import human_translations as ht
from mathula_tv.dub_agent import SYSTEM_PROMPT, TOOL_SPECS
from mathula_tv.zulu_figures import figure_style

PAIRS = [
    ht.Pair("Support fell by 31 percent last year , the party said .", "Ukwesekwa kwehle ngamaphesenti angu-31 onyakeni odlule , kusho iqembu .", "Umsuka (news)"),
    ht.Pair("Turnout was 45 percent in the province and the count continued .", "Inani labavoti belingamaphesenti angama-45 esifundazweni futhi kwaqhubeka ukubalwa .", "Umsuka (news)"),
    ht.Pair("About 60 percent of the water comes from the dam in the north .", "Cishe u-60 wamaphesenti wamanzi avela edamini enyakatho yezwe .", "Umsuka (news)"),
    ht.Pair("The party lost 12 percent of its seats in the last vote .", "Iqembu lilahlekelwe yishumi nambili lamaphesenti ezihlalo zalo ekhethweni lokugcina .", "Umsuka (news)"),
    ht.Pair("Prices rose 5 percent and the bank reacted quickly this time .", "Amanani akhuphuke ngo-5% futhi ibhange liphendule ngokushesha kule nkathi .", "Umsuka (news)"),
    ht.Pair("A shifted line about 31 percent .", "Umgubho wawubanjwe ngoMgqibelo ekhaya lemizi yezindaba zasekhaya kakhulu lapha ngempela nje kodwa futhi kuthe kakhulu lapho .", "Umsuka (news)"),
    ht.Pair("The man is 20 years old and works in town every day .", "Indoda ineminyaka engu-20 yobudala futhi isebenza edolobheni nsuku zonke .", "Umsuka (news)"),
]


def _db(tmp_path):
    path = tmp_path / "pairs.sqlite"
    ht.build(path, PAIRS)
    return ht.PairsDB(path).db


def test_the_ways_the_isizulu_writes_a_percentage_are_counted_by_prefix_with_an_example(tmp_path):
    out = figure_style(_db(tmp_path), "31 percent")
    assert out["pairs_checked"] == 3                                    # figures of 20-99 before "percent": the shifted pair is dropped, 12 is a teen, "5 percent" has one digit
    written = {row["written"]: row for row in out["ways_the_figure_is_written"]}
    assert "ngamaphesenti angu-31" in written["angu-<figure>"]["example"] or "angu-31" in written["angu-<figure>"]["example"]
    assert set(written) == {"angu-<figure>", "angama-<figure>", "u-<figure>"}          # every prefix the pairs use
    assert out["figure_spelled_out_or_left_out"] == 0
    assert "few_pairs" in out                                           # fewer than 8 pairs: the lookup says to ask with a commoner unit
    assert abs(sum(row["share_of_digit_pairs"] for row in out["ways_the_figure_is_written"]) - 1.0) < 0.05


def test_any_figure_gives_the_same_shape_not_the_same_number(tmp_path):
    assert figure_style(_db(tmp_path), "99 percent")["pairs_checked"] == 3
    only_years = figure_style(_db(tmp_path), "5 years old")
    assert only_years["pairs_checked"] == 0 or only_years["pairs_checked"] == 1
    assert figure_style(_db(tmp_path), "20 years")["ways_the_figure_is_written"][0]["written"] == "engu-<figure>"


def test_a_phrase_without_a_figure_or_without_pairs_says_what_to_give(tmp_path):
    assert "digits" in figure_style(_db(tmp_path), "thirty percent")["error"]
    assert "not installed" in figure_style(None, "31 percent")["error"]
    assert figure_style(_db(tmp_path), "7 helicopters")["pairs_checked"] == 0


def test_the_agent_has_the_figure_lookup_and_is_told_the_renderer_speaks_small_figures_in_english():
    spec = next(s for s in TOOL_SPECS if s["function"]["name"] == "figure_style")
    assert "prefix" in spec["function"]["description"] and "unit word" in spec["function"]["description"]
    assert "figure_style" in SYSTEM_PROMPT and "code-switched English after its prefix" in SYSTEM_PROMPT


def test_a_figure_is_compared_only_with_figures_of_its_size(tmp_path):
    pairs = [ht.Pair(f"The council has {n} members here in the city today .", f"Umkhandlu unamalungu {p}{n} lapha edolobheni namuhla .", "Umsuka (news)")
             for n, p in ((11, "abayi-"), (12, "abayi-"), (63, "abangama-"), (64, "abangu-"), (65, "abangu-"))]
    path = tmp_path / "sizes.sqlite"
    ht.build(path, pairs)
    db = ht.PairsDB(path).db
    teens = figure_style(db, "13 members")["ways_the_figure_is_written"]
    assert [row["written"] for row in teens] == ["abayi-<figure>"]
    twenties_up = {row["written"] for row in figure_style(db, "40 members")["ways_the_figure_is_written"]}
    assert twenties_up == {"abangama-<figure>", "abangu-<figure>"}


def test_the_agent_prompt_says_what_each_punctuation_mark_does_and_no_longer_carries_the_azure_era_costs():
    # the 0.12-0.18 s comma and 0.9 s full stop were measured on the Azure voices (2026-11-11); the cloned voice joins sentences in about 0.35-0.5 s and the prompt must not frighten the agent off full stops
    assert "Pacing and punctuation" in SYSTEM_PROMPT and "ellipsis" in SYSTEM_PROMPT and "a comma marks a clause boundary" in " ".join(SYSTEM_PROMPT.split()).replace("A comma", "a comma")
    assert "0.12-0.18 s" not in SYSTEM_PROMPT and "about 0.9 s" not in SYSTEM_PROMPT
    assert "half a second after a sentence" not in SYSTEM_PROMPT
