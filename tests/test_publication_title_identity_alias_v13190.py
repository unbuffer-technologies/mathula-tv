from __future__ import annotations

from mathula_tv.tiktok_editor import apply_publication_title_authority


def _authority() -> dict[str, object]:
    return {
        "sha256": "authority-test",
        "corrections": [
            {
                "raw_text": "Sibiya",
                "canonical_text": "Lieutenant-General Sibiya",
                "source": "authoritative_transcript.name_research",
                "derived": False,
            },
            {
                "raw_text": "Sabir",
                "canonical_text": "Lieutenant-General Sibiya",
                "source": "authoritative_transcript.name_research",
                "derived": False,
            },
        ],
    }


def test_valid_surname_inside_canonical_expansion_is_not_rejected() -> None:
    title = (
        "USibiya uthi ujeziswa ngokusebenza kahle—kodwa udluliselwe "
        "emacaleni obugebengu"
    )

    selected, policy = apply_publication_title_authority(title, _authority())

    assert selected == title
    assert policy["adjusted"] is False
    assert policy["applied_corrections"] == []
    assert policy["accepted_identity_aliases"] == [
        {
            "raw_text": "Sibiya",
            "canonical_text": "Lieutenant-General Sibiya",
            "source": "authoritative_transcript.name_research",
            "reason": "canonical_expansion_preserves_valid_identity_alias",
        }
    ]


def test_actual_stt_corruption_is_still_replaced_once() -> None:
    selected, policy = apply_publication_title_authority(
        "USabir udluliselwe ukuze kuthathwe esinye isinyathelo",
        _authority(),
    )

    assert selected == (
        "ULieutenant-General Sibiya udluliselwe ukuze kuthathwe esinye "
        "isinyathelo"
    )
    assert "Lieutenant-General Lieutenant-General" not in selected
    assert policy["adjusted"] is True
    assert policy["applied_corrections"][0]["raw_text"] == "Sabir"


def test_full_correct_name_is_not_expanded_inside_itself() -> None:
    title = "ULt-Gen Shadrack Sibiya udluliselwe ukuze kuthathwe isinyathelo"

    selected, policy = apply_publication_title_authority(title, _authority())

    assert selected == title
    assert policy["adjusted"] is False
    assert len(policy["accepted_identity_aliases"]) == 1
