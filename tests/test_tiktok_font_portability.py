from pathlib import Path

from mathula_tv import tiktok_editor


def test_windows_system_font_is_discovered(tmp_path, monkeypatch):
    windir = tmp_path / "Windows"
    fonts = windir / "Fonts"
    fonts.mkdir(parents=True)
    expected = fonts / "segoeuib.ttf"
    expected.write_bytes(b"placeholder")
    monkeypatch.setenv("WINDIR", str(windir))
    monkeypatch.setenv("SystemRoot", str(windir))
    monkeypatch.setattr(tiktok_editor, "_FONT_PATH", None)

    assert tiktok_editor._resolve_bold_font_path() == expected


def test_explicit_font_override_has_priority(tmp_path, monkeypatch):
    explicit = tmp_path / "custom-bold.ttf"
    explicit.write_bytes(b"placeholder")
    monkeypatch.setattr(tiktok_editor, "_FONT_PATH", explicit)
    monkeypatch.delenv("WINDIR", raising=False)
    monkeypatch.delenv("SystemRoot", raising=False)

    assert tiktok_editor._resolve_bold_font_path() == explicit


def test_missing_preferred_font_falls_back_without_crashing(monkeypatch):
    monkeypatch.setattr(tiktok_editor, "_bold_font_candidates", lambda: [])
    font = tiktok_editor._load_title_font(42)
    assert font is not None
