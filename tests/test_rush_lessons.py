import json

from mathula_tv import ear_lessons, rush_lessons


class _Client:
    def __init__(self, answer):
        self.answer, self.sent = answer, []

    def complete(self, messages, **_):
        self.sent.append(json.loads(messages[1]["content"]))
        items = [{"id": x["id"], **self.answer} for x in self.sent[-1]]
        return {"choices": [{"message": {"content": json.dumps({"items": items})}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_one_pressed_word_is_investigated_with_its_sentence_and_the_next_one_and_no_pace():
    client = _Client({"cause": "a long chain of modifiers taken for one compound", "rule": "When a verb-noun, an adverb and an adjective are stacked, split them.", "rewrite": "kahle, okubucayi", "next_note": ""})
    found = rush_lessons.investigate(client, sentence="a kahle b.", next_sentence="Umane ulandele.", word="kahle", phrase="a kahle b")
    assert found["cause"].startswith("a long chain") and found["rewrite"] == "kahle, okubucayi" and found["next_note"] == ""
    sent = client.sent[0][0]
    assert sent["next_sentence"] == "Umane ulandele." and set(sent) == {"id", "sentence", "next_sentence", "word", "phrase"}          # nothing counted is sent: no pace, no syllables
    assert "do not reason from speed" in rush_lessons.INVESTIGATE_PROMPT and "swallows a syllable" in rush_lessons.INVESTIGATE_PROMPT


def test_the_listeners_three_observations_are_lessons_every_agent_starts_with(tmp_path):
    concepts = ear_lessons.for_agent(tmp_path)["concepts"]
    classes = {(c["item_class"], c["written"][:12]) for c in concepts}
    assert ("speakability", "u-Ayanda Nya") in classes and ("speakability", "ukungaziphat") in classes and ("grammar", "bobabili eko") in classes
    assert not any("rushed_words" in c["fix"] for c in concepts)                                              # no lesson points at a measure that no longer exists
