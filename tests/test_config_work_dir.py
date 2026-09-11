from __future__ import annotations

import warnings
from pathlib import Path

from mathula_tv.config import _resolve_windows_safe_path, _resolve_work_dir, load_settings


def test_work_dir_rejects_posix_path_on_windows():
    root = Path("C:/mathula-tv")
    default = root / "working"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_work_dir(root, "/home/mokgethwa/mathula-tv/working", platform_name="nt")
    assert resolved == default
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_work_dir_rejects_backslash_escaped_posix_path_on_windows():
    root = Path("C:/mathula-tv")
    default = root / "working"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_work_dir(root, "\\home\\mokgethwa\\mathula-tv\\working", platform_name="nt")
    assert resolved == default
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_work_dir_accepts_drive_letter_path_on_windows():
    root = Path("C:/mathula-tv")
    resolved = _resolve_work_dir(root, "D:/mathula-tv/working", platform_name="nt")
    assert resolved == Path("D:/mathula-tv/working")


def test_work_dir_accepts_unc_path_on_windows():
    root = Path("C:/mathula-tv")
    resolved = _resolve_work_dir(root, "\\\\fileserver\\share\\mathula-tv\\working", platform_name="nt")
    assert resolved == Path("\\\\fileserver\\share\\mathula-tv\\working")


def test_work_dir_accepts_posix_path_on_linux():
    root = Path("/opt/mathula-tv")
    resolved = _resolve_work_dir(root, "/home/mokgethwa/mathula-tv/working", platform_name="posix")
    assert resolved == Path("/home/mokgethwa/mathula-tv/working")


def test_work_dir_rejects_windows_backslash_path_on_linux():
    # Real production failure: a shared .env carried MATHULA_TV_WORK_DIR=
    # C:\mathula-tv\working onto a remote Linux server. PosixPath never
    # treats backslashes as separators, so every job/file path built from it
    # silently corrupted (confirmed: ffprobe was invoked against a mangled
    # "C:mathula-tvworking/jobs/..." path that never existed).
    root = Path("/opt/mathula-tv")
    default = root / "working"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_work_dir(root, r"C:\mathula-tv\working", platform_name="posix")
    assert resolved == default
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_work_dir_rejects_windows_forward_slash_drive_path_on_linux():
    root = Path("/opt/mathula-tv")
    default = root / "working"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_work_dir(root, "C:/mathula-tv/working", platform_name="posix")
    assert resolved == default
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_work_dir_rejects_windows_unc_path_on_linux():
    root = Path("/opt/mathula-tv")
    default = root / "working"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_work_dir(root, r"\\fileserver\share\mathula-tv\working", platform_name="posix")
    assert resolved == default
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_work_dir_accepts_a_real_linux_path_that_happens_to_start_with_a_letter():
    # Must not false-positive on ordinary relative-looking POSIX values.
    root = Path("/opt/mathula-tv")
    resolved = _resolve_work_dir(root, "data/mathula-tv/working", platform_name="posix")
    assert resolved == Path("data/mathula-tv/working")


def test_work_dir_defaults_when_unset():
    root = Path("C:/mathula-tv")
    assert _resolve_work_dir(root, None, platform_name="nt") == root / "working"
    assert _resolve_work_dir(root, "", platform_name="nt") == root / "working"


def test_resolve_windows_safe_path_rejects_stale_posix_override():
    default = Path("C:/mathula-tv/working/context/politics_context.json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_windows_safe_path(
            default, "/home/mokgethwa/mathula-tv/working/context/politics_context.json", platform_name="nt"
        )
    assert resolved == default
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_load_settings_commission_default_is_windows_safe(monkeypatch, tmp_path):
    monkeypatch.delenv("COMMISSION_MASTER_CASE_PATH", raising=False)
    monkeypatch.delenv("POLITICS_CONTEXT_PATH", raising=False)
    monkeypatch.delenv("MATHULA_TV_WORK_DIR", raising=False)
    monkeypatch.setattr("mathula_tv.config.os.name", "nt")
    settings = load_settings(tmp_path)
    assert str(settings.commission_master_case_path).startswith(str(settings.work_dir))
    assert str(settings.politics_context_path).startswith(str(settings.work_dir))
    assert "home" not in settings.commission_master_case_path.parts
    assert "home" not in settings.politics_context_path.parts
