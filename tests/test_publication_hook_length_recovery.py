from __future__ import annotations

from mathula_tv import tiktok_editor


def test_editorial_schema_defers_title_length_to_local_renderer() -> None:
    properties = tiktok_editor.EDITORIAL_PACKAGE_SCHEMA["properties"]["candidates"][
        "items"
    ]["properties"]

    assert "maxLength" not in properties["hook_text"]
    assert "maxLength" not in properties["accessible_hook_text"]
    assert "maxLength" not in properties["withheld_answer"]
    assert "maxLength" not in properties["title_lead"]["properties"]["text"]
    assert tiktok_editor.TIKTOK_HOOK_PROMPT_VERSION == (
        "mathula-editorial-package-v12-featured-people-discovery"
    )


def test_very_long_accessible_hook_uses_concise_grounded_person_rewrite() -> None:
    accessible = (
        "U-Andrea Johnson, owayehola isikhungo esiphenya inkohlakalo, "
        "uthi izinsolo zemali eya kowayenguNgqongqoshe uBheki Cele "
        "bezifanele ziphenywe—kodwa akaqiniseki ukuthi kwenzeka ngempela"
    )
    assert len(accessible) > 160

    selected, policy = tiktok_editor._choose_accessible_hook_text(
        {
            "hook_text": (
                "IDAC: UJohnson uthi izinsolo zemali eya kuBheki Cele "
                "bekufanele ziphenywe"
            ),
            "accessible_hook_text": accessible,
            "title_lead": {"type": "person", "text": "Andrea Johnson"},
        },
        target_language="zu-ZA",
    )

    assert selected.startswith("UJohnson:")
    assert "IDAC" not in selected
    assert len(selected) < len(accessible)
    assert policy["deterministic_rewrite"]["strategy"] == "grounded_person_lead"

    fitted, length_policy = tiktok_editor._fit_title_card_hook_text(
        selected,
        title_lead={"type": "person", "text": "Andrea Johnson"},
        target_language="zu-ZA",
    )
    assert len(fitted) <= 90
    assert length_policy["maximum_characters"] == 90
