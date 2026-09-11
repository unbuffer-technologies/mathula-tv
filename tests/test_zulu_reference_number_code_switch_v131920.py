from __future__ import annotations

from mathula_tv.pronunciation import (
    PronunciationDictionary,
    ZU_ENGLISH_REFERENCE_NUMBER_CODE_SWITCH_VERSION,
    with_default_organisation_initialisms,
)


def _zulu_dictionary() -> PronunciationDictionary:
    return with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )


def test_bare_reference_number_keeps_protected_digits_but_uses_english_code_switch() -> None:
    # Confirmed real defect: "3978." transcribed as bare digits measured 4930ms of
    # synthesized audio against a 1120ms source window because Azure's zu-ZA voice
    # reads bare numerals as a full isiZulu cardinal-number expansion.
    result = _zulu_dictionary().apply("3978. 3978.")

    assert result.spoken_text == "3978. 3978."
    assert result.tts_text == "triiy nayn seven eyt. triiy nayn seven eyt."
    assert len(result.substitutions) == 2
    assert all(item.kind == "english_code_switch" for item in result.substitutions)
    assert all(item.before == "3978" for item in result.substitutions)


def test_reference_number_code_switch_generalizes_to_other_digit_combinations() -> None:
    dictionary = _zulu_dictionary()
    cases = {
        "2015": "tuu zeeroh wan fayiv",
        "60614": "seeks zeeroh seeks wan four",
    }
    for digits, expected_alias in cases.items():
        result = dictionary.apply(f"Idokethi {digits} yagcwaliswa.")
        assert expected_alias in result.tts_text
        assert digits in result.spoken_text


def test_two_digit_number_is_not_code_switched() -> None:
    # A bare 1-2 digit number is far more likely to be an ordinary quantity than a
    # reference/identifier number; the feature is scoped to 3+ digits (see the
    # confirmed real docket-number case) to avoid over-triggering on small counts.
    result = _zulu_dictionary().apply("Wafika ngo 17.")
    assert result.tts_text == "Wafika ngo 17."
    assert not result.substitutions


def test_isizulu_prefixed_cardinal_is_not_overridden_by_english_code_switch() -> None:
    # A number with a real isiZulu grammatical prefix already gets a natural
    # isiZulu cardinal reading (_zulu_dynamic_number_candidates) -- the English
    # code-switch must not compete with or override that existing, correct path.
    result = _zulu_dictionary().apply("emahoreni angaphansi ezingu-24.")
    assert "triiy" not in result.tts_text
    assert "tuu" not in result.tts_text
    assert "ezinga" in result.tts_text


def test_rand_million_amount_is_not_overridden_by_english_code_switch() -> None:
    result = _zulu_dictionary().apply("u-R31 million.")
    assert "izigidi" in result.tts_text
    assert "wan" not in result.tts_text
    assert "triiy" not in result.tts_text


def test_space_grouped_thousand_reads_as_english_magnitude_not_digit_spelled() -> None:
    # Real, confirmed defect, job b15075e7268049b491ee9e2222e5811f, user-
    # reported live: "bad pronunciation of 10000". The real SSML sent to
    # Azure read "amaRandi ayi-10 zeeroh zeeroh zeeroh ..." -- the space
    # between "10" and "000" meant the trailing group independently matched
    # the bare-reference-number pattern and got digit-spelled literally,
    # instead of the whole amount reading as a magnitude. Fixed per direct
    # user direction ("we should code switch '10 thousand'", "the model
    # should be able to pronounce money like a normal person").
    result = _zulu_dictionary().apply("Uphumile ngebheyili yamaRandi ayi-10 000 ngemibandela.")

    assert result.spoken_text == "Uphumile ngebheyili yamaRandi ayi-10 000 ngemibandela."
    assert "10 thousand" in result.tts_text
    assert "zeeroh" not in result.tts_text


def test_space_grouped_thousand_generalizes_to_other_round_amounts() -> None:
    # Matches the real confirmed shape (a hyphen/space separator between any
    # preceding concord/word and the number, e.g. "ayi-10 000") -- this
    # mechanism is currency-word-agnostic by design (see the regex's own
    # docstring), not tied to "R" glued directly onto the digits the way
    # _ZU_CODE_SWITCH_RAND_MILLION's compact-form convention is.
    dictionary = _zulu_dictionary()
    for amount in ("5 000", "100 000", "999 000"):
        result = dictionary.apply(f"Kukhokhwe ayi-{amount} ngenyanga.")
        assert "zeeroh" not in result.tts_text
        assert amount.split()[0] + " thousand" in result.tts_text


def test_non_thousand_three_digit_group_is_still_digit_spelled() -> None:
    # A trailing 3-digit group that is NOT exactly "000" is not a thousands
    # separator -- e.g. "10 123" is not a real-world magnitude shape this
    # mechanism should claim; the existing reference-number fallback should
    # still own it (unaffected by this change).
    result = _zulu_dictionary().apply("Idokethi 10 123 yagcwaliswa.")
    assert "wan zeeroh" not in result.tts_text  # "10" itself untouched
    assert "wan tuu triiy" in result.tts_text  # "123" still digit-spelled


def test_reference_number_version_note_is_attached() -> None:
    # Notes live on the underlying PronunciationEntry, not the lightweight
    # PronunciationSubstitution record returned by apply().
    from mathula_tv.pronunciation import _zulu_dynamic_reference_number_candidates

    candidates = _zulu_dynamic_reference_number_candidates("3978.", language="zu-ZA")
    assert len(candidates) == 1
    _, _, entry = candidates[0]
    assert ZU_ENGLISH_REFERENCE_NUMBER_CODE_SWITCH_VERSION in entry.notes


def test_english_code_switch_default_does_not_leak_into_other_locales() -> None:
    xhosa = PronunciationDictionary("v1", language="xh-ZA", job_id="job-123")
    augmented = with_default_organisation_initialisms(xhosa)
    assert augmented.apply("3978.").tts_text == "3978."
