import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "install_manual_translation.py"
spec = importlib.util.spec_from_file_location("install_manual_translation", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_validate_alignment_accepts_equal_segments(tmp_path):
    source = {"segments": [{"start": 0, "end": 1, "source_text": "Hello"}]}
    translated = {"segments": [{"start": 0, "end": 1, "translated_text": "Sawubona"}]}
    result = module.validate_alignment(
        source,
        translated,
        tmp_path / "source.json",
        tmp_path / "translated.json",
    )
    assert result["segment_count"] == 1


def test_help_describes_deferred_single_editorial_call():
    source = MODULE_PATH.read_text(encoding="utf-8")
    main_source = source[source.index("def main()") :]
    assert "single editorial AI call is deferred" in source
    assert "build_deferred_localized_seo(" in main_source
    assert "generate_localized_seo(" not in main_source
