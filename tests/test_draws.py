from pathlib import Path

import pytest

from mathula_tv import draws


def test_a_committed_retake_is_found_for_the_same_text_and_clip_and_not_for_another_clip(tmp_path):
    draws.commit(tmp_path, "Mayelana nalokhu   sidlulela.", 23.9, 32.4, 2, unit="unit-01", note="Phineas")
    store = draws.load(tmp_path)
    assert draws.retake_for(store, "Mayelana nalokhu sidlulela.", 23.9, 32.4) == 2          # spacing does not matter
    assert draws.retake_for(store, "Mayelana nalokhu sidlulela.", 30.0, 38.0) == 0          # another reference clip: another draw
    assert draws.is_stale(store, "Mayelana nalokhu sidlulela.", 30.0, 38.0) and not draws.is_stale(store, "Mayelana nalokhu sidlulela.", 23.9, 32.4)
    assert draws.retake_for(store, "A different sentence.", 23.9, 32.4) == 0
    assert draws.load(tmp_path / "nowhere") == {"marks": {}} and draws.retake_for({}, "x", 0, 1) == 0


def test_a_retake_can_be_taken_back_and_must_be_a_small_whole_number(tmp_path):
    draws.commit(tmp_path, "Sawubona.", 1.0, 2.0, 3)
    assert draws.revoke(tmp_path, "Sawubona.") is True and draws.revoke(tmp_path, "Sawubona.") is False
    assert draws.retake_for(draws.load(tmp_path), "Sawubona.", 1.0, 2.0) == 0
    for bad in (-1, 10, 1.5, "2"):
        with pytest.raises(ValueError):
            draws.commit(tmp_path, "Sawubona.", 1.0, 2.0, bad)
