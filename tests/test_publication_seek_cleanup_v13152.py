from __future__ import annotations

import ast
from pathlib import Path

import mathula_tv.tiktok_editor as tiktok_editor


def test_publication_renderer_has_no_input_seek() -> None:
    path = Path(tiktok_editor.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_tiktok_hook_edit"
    )
    function_source = ast.get_source_segment(source, function) or ""
    assert '"-ss"' not in function_source
    assert "'-ss'" not in function_source
    assert "trim=start=" in function_source
    assert "atrim=start=" in function_source
    assert "selected_rate = select_playback_frame_rate" in function_source


def test_publication_render_version_invalidates_old_output() -> None:
    assert "frame-accurate-seek-v13.15.2" in tiktok_editor.TIKTOK_EDIT_RENDER_VERSION
