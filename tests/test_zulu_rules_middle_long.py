"""A long vowel in a letter in the middle of an initialism: an open choice, single by default, written long as the listener wrote NEC and SABC."""
from __future__ import annotations

import pytest

from mathula_tv import zulu_rules as z


@pytest.fixture(autouse=True)
def rules_as_written(tmp_path, monkeypatch):
    monkeypatch.setenv(z.RULES_CONFIG_ENV, str(tmp_path / "rules_config.json"))
    z._config_seen.update(path=None, mtime=None)


def _main(word):
    return z.respell_initialism(word, limit=1)[0]["spelling"]


def _spellings(word):
    return [r["spelling"] for r in z.respell_initialism(word, limit=8)]


def test_by_default_a_middle_letter_keeps_its_vowel_single():
    assert [_main(w) for w in ("NEC", "SABC", "IEC", "ANC", "DA")] == ["en-i-siii", "es-eyi-bi-siii", "ayi-i-siii", "eyi-en-siii", "di-eeyi"]


def test_the_long_middle_letter_is_an_alternative_the_listener_can_hear_and_commit():
    assert "en-iii-siii" in _spellings("NEC") and "es-eyi-bii-siii" in _spellings("SABC")          # a bare vowel is held three times, a consonant's vowel twice
    found = z.respell_initialism("NEC", limit=8)
    long = next(r for r in found if r["spelling"] == "en-iii-siii")
    assert [d["rule"] for d in z.choice_differences(found[0]["choice"], long["choice"])] == ["middle_long"]
    assert "a long vowel in a letter in the middle of an initialism is written long: iii" in long["derivation"]


def test_only_the_middle_is_lengthened_never_the_first_letter_or_the_last_or_a_word():
    z.write_rule_default("middle_long", 1, path=z.rules_config_path())
    assert [_main(w) for w in ("NEC", "SABC", "DA", "ANC", "MK", "PSL")] == ["en-iii-siii", "es-eyi-bii-siii", "di-eeyi", "eyi-en-siii", "em-kheeyi", "pi-es-eel"]
    assert z.respell_word(["k", "ɔː", "t"], [None, 1, None], limit=1)[0]["spelling"] == "khoot"
