from __future__ import annotations

import ast
from pathlib import Path


def test_publication_renderer_uses_source_rate_and_frame_accurate_trim() -> None:
    import mathula_tv.tiktok_editor as module

    path = Path(module.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_tiktok_hook_edit"
    )
    function_source = ast.get_source_segment(source, function) or ""
    assert "select_playback_frame_rate" in function_source
    assert "trim=start=" in function_source
    assert "atrim=start=" in function_source
    assert '"-ss"' not in function_source
    assert '"90000"' in function_source
    assert 'MATHULA_TV_FINAL_VIDEO_CRF' in function_source
    assert 'MATHULA_TV_FINAL_AUDIO_BITRATE' in function_source
