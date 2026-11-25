from mathula_tv.dub_workbench import Workbench


def test_the_gap_to_the_window_is_shown_in_characters_at_the_units_own_pace():
    hint = Workbench._fit_hint(63660, 61743, 875)                        # 1917 ms short, 70.6 ms per character
    assert hint["to_window_ms"] == 1917 and abs(hint["ms_per_char"] - 70.6) < 0.1 and hint["chars_to_change"] == 27 and hint["within_noise"] is False
    over = Workbench._fit_hint(10000, 10800, 100)                        # 800 ms past the window, 108 ms per character: take out about 7
    assert over["to_window_ms"] == -800 and over["chars_to_change"] == -7
    assert Workbench._fit_hint(63660, 63444, 864)["within_noise"] is True      # 216 ms short: inside the noise
    assert Workbench._fit_hint(63660, 0, 864) is None and Workbench._fit_hint(None, 1000, 10) is None and Workbench._fit_hint(1000, 900, 0) is None


def test_a_flagged_unit_carries_its_gap_and_the_note_is_said_once_in_a_measurement():
    group = {"group_id": "item-1", "gates": {}, "flags": ["overrun"], "window_ms": 63660, "measured_ms": 64800}
    entries = Workbench._with_texts([Workbench._flagged_entry(group)], [{"item_id": "item-1", "zulu_text": "a" * 870}])
    assert entries[0]["to_fit"]["to_window_ms"] == -1140 and entries[0]["to_fit"]["chars_to_change"] < 0
    assert "window_ms" not in entries[0] and "measured_ms" not in entries[0]                   # only the hint is shown
    assert "pace" in Workbench.FIT_NOTE and "within_noise" in Workbench.FIT_NOTE
