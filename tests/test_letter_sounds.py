import json

from mathula_tv import letter_sounds as ls

TOY = {"s": ["s"], "z": ["z"], "n": ["n"], "ts": ["ts"], "nts": ["n", "ts"], "t": ["t"], "a": ["a"], "e": ["e"], "ai": ["aɪ"], "ay": ["aɪ"],
       "i": ["i"], "ya": ["j", "a"], "ng": ["ŋ"]}


def fake_reading(spelling):
    """A voice that says each letter group as in TOY, between whatever single-letter neighbours the probe has."""
    return [spelling[0], *TOY.get(spelling[1:-1], []), spelling[-1]]


def test_the_table_is_what_the_voice_said_between_the_fixed_neighbours(tmp_path):
    table = ls.build_table(fake_reading, voice="v")
    assert table["graphemes"]["ts"] == [{"sounds": ["ts"], "seen": 3, "of": 3}] and table["graphemes"]["ai"][0]["sounds"] == ["aɪ"]
    assert table["graphemes"]["ya"][0]["sounds"] == ["j", "a"] and table["graphemes"]["ai"][0]["of"] == 4                 # four neighbourhoods for a vowel
    assert "c" not in table["graphemes"] and "q" not in table["graphemes"]                      # clicks are never probed
    path = tmp_path / "letters.json"
    calls = []

    def counting(spelling):
        calls.append(spelling)
        return fake_reading(spelling)

    ls.load_or_build(path, counting, voice="v")
    first = len(calls)
    assert first > 100 and json.loads(path.read_text(encoding="utf-8"))["voice"] == "v"
    ls.load_or_build(path, counting, voice="v")
    assert len(calls) == first                                                                   # kept: the voice is not asked again


def test_spellings_are_worked_out_backwards_from_the_speakers_sounds():
    table = {"graphemes": TOY}
    found = dict(ls.candidates(["s", "aɪ", "n", "ts", "ə"], table, limit=40))
    found = dict(ls.candidates(["s", "aɪ", "n", "ts", "ə"], table, limit=400))
    assert {"saintsa", "sayntsa", "saintsah"} <= set(found)                                        # different letters reach the same sounds, a schwa as a or ah
    assert "saizsa" not in found                                                                   # but a different consonant (z for ts) is never reached
    best = ls.candidates(["s", "aɪ", "n", "ts", "ə"], table, limit=3)
    assert best[0][0] == "saintsa" and best == sorted(best, key=lambda item: item[1])            # every sound written by its most ordinary letters comes first
    assert "saints" in dict(ls.candidates(["s", "aɪ", "n", "ts", "ə"], table, limit=400))        # a noisy last sound may be left out too
    assert ls.candidates([], table) == []
    assert ls.candidates(["s"], {"graphemes": {}})[0][0] == "s"                                    # the prior works before any table is built


def test_a_sound_is_reached_through_its_nearest_letters_when_none_is_exact():
    table = {"graphemes": {"f": ["f"], "ai": ["aɪ"], "t": ["t"], "a": ["a"], "s": ["s"]}}
    found = dict(ls.candidates(["f", "aɪ", "t", "ə", "s"], table, limit=10))
    assert "faitas" in found                                                                       # the schwa is written with a: the nearest vowel
    assert all(ls._step_cost("t", made) is None for made in ("k", "s", "n"))                      # but a consonant is never swapped for a different consonant


def test_a_letter_group_that_is_read_differently_next_to_different_letters_keeps_every_outcome_with_how_often():
    def moody(spelling):                      # "ai" is a diphthong before n and t, something else elsewhere
        if spelling[1:-1] == "ai":
            return [spelling[0], *(["aɪ"] if spelling[-1] in "nt" else ["eɪ"]), spelling[-1]]
        return fake_reading(spelling)

    options = ls.build_table(moody)["graphemes"]["ai"]
    assert {tuple(o["sounds"]) for o in options} == {("aɪ",), ("eɪ",)} and sum(o["seen"] for o in options) == 4
    found = dict(ls.candidates(["s", "aɪ", "n"], {"graphemes": {"s": [{"sounds": ["s"], "seen": 1, "of": 1}], "ai": options,
                                                         "n": [{"sounds": ["n"], "seen": 1, "of": 1}]}}, limit=5))
    assert "sain" in found                                                                       # the outcome that reaches the speaker's sound is used


def test_a_spelling_is_varied_by_swapping_in_the_groups_the_voice_reads_the_same_way():
    table = {"graphemes": {
        "s": [{"sounds": ["s"], "seen": 4, "of": 4}], "n": [{"sounds": ["n"], "seen": 4, "of": 4}], "ts": [{"sounds": ["ts"], "seen": 3, "of": 4}],
        "nts": [{"sounds": ["n", "ts"], "seen": 3, "of": 4}], "ns": [{"sounds": ["n", "ts"], "seen": 4, "of": 4}],
        "ai": [{"sounds": ["eɪ"], "seen": 3, "of": 4}, {"sounds": ["aɪ"], "seen": 1, "of": 4}], "ay": [{"sounds": ["eɪ"], "seen": 4, "of": 4}],
        "ei": [{"sounds": ["aɪ"], "seen": 3, "of": 4}], "a": [{"sounds": ["a"], "seen": 4, "of": 4}]}}
    assert ls.split("saintsa", table) == ["s", "ai", "nts", "a"]                                  # longest known group first
    swaps = ls.substitutes(table)
    assert swaps["ai"][:3] == ["ay", "ei", "ey"] and "ai" in swaps["ay"] and "ns" in swaps["nts"]          # groups that write the same sound, most ordinary first
    near = ls.neighbours("saintsa", table, limit=100)
    assert "sayntsa" in near                                                                       # ai -> ay: read the same way
    assert "sainsa" in near and "saints" in near and "saintsha" in near                            # nts -> ns; the last group left out; a breath added
    assert any(spelling.startswith("sa") and spelling.endswith("a") and len(spelling) == len("saintsa") + 1 for spelling in near)    # a letter added
    assert "saintsa" not in near


def test_the_vowels_of_a_spelling_are_varied_before_its_consonants():
    table = {"graphemes": {}}
    near = ls.neighbours("sainsa", table, limit=100)
    first_vowel_swap = next(i for i, spelling in enumerate(near) if spelling.startswith("s") and spelling != "sainsa" and spelling[1:3] in ("ay", "ei", "ey"))
    first_consonant_swap = next(i for i, spelling in enumerate(near) if spelling.startswith(("z", "ss", "zai", "ssai")))
    assert first_vowel_swap < first_consonant_swap                                                # "sayn..." before "zain..."
    assert "saainsa" in ls.neighbours("sainsa", table, limit=200)                                  # the open-start way of writing the diphthong is reachable
