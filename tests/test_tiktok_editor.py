from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv import tiktok_editor


class FakeProvider:
    def __init__(self, selected_block_id: str = "block_0002"):
        self.selected_block_id = selected_block_id
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        score_sets = [
            {
                "visual_impact": 9,
                "curiosity": 9,
                "specificity": 9,
                "stakes": 7,
                "immediacy": 8,
                "audience_relevance": 9,
            },
            {
                "visual_impact": 5,
                "curiosity": 6,
                "specificity": 7,
                "stakes": 6,
                "immediacy": 6,
                "audience_relevance": 7,
            },
            {
                "visual_impact": 4,
                "curiosity": 5,
                "specificity": 6,
                "stakes": 5,
                "immediacy": 5,
                "audience_relevance": 6,
            },
        ]
        return SimpleNamespace(
            data={
                "candidates": [
                    {
                        "candidate_id": f"candidate_{index + 1}",
                        "selected_block_id": (
                            self.selected_block_id if index == 0 else "block_0001"
                        ),
                        "hook_text": (
                            "I-EFF ayihlehli!"
                            if index == 0
                            else f"Umcimbi we-EFF uqala manje {index}"
                        ),
                        "rationale": "Grounded, specific and self-contained.",
                        "confidence": 0.91 - (index * 0.1),
                        "grounded": True,
                        "preserves_attribution": True,
                        "no_sensationalism": True,
                        "scores": score_sets[index],
                        "human_review_flags": [],
                    }
                    for index in range(3)
                ],
            },
            metadata=SimpleNamespace(to_dict=lambda: {"provider": "fake"}),
        )


def _blocks():
    return [
        {
            "block_id": "block_0001",
            "speaker_id": "SPEAKER_00",
            "start_ms": 2000,
            "source_text": "Welcome to the programme.",
            "translated_text": "Siyanamukela ohlelweni.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "SPEAKER_01",
            "start_ms": 19840,
            "source_text": "The EFF is not deterred.",
            "translated_text": "I-EFF ayihlehli.",
        },
    ]


def test_ai_selects_only_server_supplied_block_boundaries() -> None:
    result = tiktok_editor.select_tiktok_hook(
        provider=FakeProvider(),
        job_id="job1",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={},
        source_duration_seconds=120.0,
    )

    assert result["selected_block_id"] == "block_0002"
    assert result["cut_start_seconds"] == 19.84
    assert result["approved_speech_immutable"] is True
    assert result["engagement_ranking"]["winner_score"] > 8
    assert len(result["engagement_ranking"]["ranked_candidates"]) == 3


def test_ai_cannot_select_a_non_candidate_boundary() -> None:
    with pytest.raises(ValueError, match="non-candidate"):
        tiktok_editor.select_tiktok_hook(
            provider=FakeProvider("block_9999"),
            job_id="job1",
            target_language="zu-ZA",
            blocks=_blocks(),
            seo={},
            source_duration_seconds=120.0,
        )


def test_hook_edit_preserves_translation_and_keeps_remainder(
    tmp_path, monkeypatch
) -> None:
    master = tmp_path / "master.mp4"
    translation = tmp_path / "translation.json"
    output = tmp_path / "edited.mp4"
    manifest = tmp_path / "edit.json"
    hook = tmp_path / "hook.txt"
    master.write_bytes(b"master")
    translation.write_text('{"segments": [{"translated_text": "akuguquki"}]}')
    source_info = {
        "duration": 120.0,
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080},
            {"codec_type": "audio"},
        ],
    }
    rendered_info = {
        "duration": 100.16,
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "Main",
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "25/1",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
            },
        ],
    }
    monkeypatch.setattr(
        tiktok_editor,
        "probe",
        lambda path: source_info if path == master else rendered_info,
    )
    monkeypatch.setattr(tiktok_editor, "_FONT_PATH", tmp_path / "font.ttf")
    tiktok_editor._FONT_PATH.write_bytes(b"font")
    def fake_title_panel(*, output_path, title, width, height):
        output_path.write_bytes(b"panel")
        return {
            "rendered_text": title,
            "line_count": 1,
            "font_size": 48,
            "minimum_font_size": 35,
            "maximum_font_size": 84,
            "fit_action": "unchanged",
            "truncated": False,
            "panel_bounds": [88, 852, 1832, 1068],
        }

    monkeypatch.setattr(tiktok_editor, "_render_title_panel", fake_title_panel)
    commands = []

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"edited")
        return SimpleNamespace(returncode=0)

    before = translation.read_bytes()
    result = tiktok_editor.render_tiktok_hook_edit(
        master_video=master,
        output_path=output,
        manifest_path=manifest,
        hook_text_path=hook,
        selection={
            "selected_block_id": "block_0002",
            "cut_start_seconds": 19.84,
            "hook_text": "I-EFF ayihlehli!",
        },
        translation_path=translation,
        title_text="Isihloko esigqamile #ZuluTikTok #IsiZulu",
        runner=fake_runner,
    )

    assert translation.read_bytes() == before
    assert result["all_content_after_boundary_preserved"] is True
    assert result["translation"]["immutable"] is True
    assert result["title_panel"]["text"] == "Isihloko esigqamile"
    assert result["title_panel"]["position"] == "footer"
    assert result["title_panel"]["font_weight"] == "bold"
    assert result["top_hook_overlay"] is False
    command = commands[0]
    assert command[command.index("-ss") + 1] == "19.840000"
    assert command.index("-ss") < command.index("-i")
    video_filter = command[command.index("-filter_complex") + 1]
    assert "setpts=PTS-STARTPTS" in video_filter
    assert "overlay=0:0" in video_filter
    assert command[command.index("-af") + 1] == "asetpts=PTS-STARTPTS"
    assert result["render_version"] == tiktok_editor.TIKTOK_EDIT_RENDER_VERSION
    assert output.read_bytes() == b"edited"


def test_caption_card_removes_all_hashtags() -> None:
    assert tiktok_editor.caption_without_hashtags(
        "Izindaba #ZuluTikTok zanamuhla #Mzansi"
    ) == "Izindaba zanamuhla"


def test_selected_hook_becomes_authoritative_seo_cover_hook(tmp_path) -> None:
    seo_path = tmp_path / "translation" / "tiktok_zu.json"
    output_root = tmp_path / "output"
    output_json = output_root / "tiktok_seo_zu_job1.json"
    output_cover = output_root / "tiktok_cover_zu_job1.txt"
    seo_path.parent.mkdir(parents=True)
    output_root.mkdir()
    output_json.write_text('{"cover_hook": "Generic"}', encoding="utf-8")
    output_cover.write_text("Generic\n", encoding="utf-8")

    updated = tiktok_editor._synchronize_seo_cover_hook(
        seo={"cover_hook": "Generic", "title": "Generic"},
        seo_path=seo_path,
        hook_text="I-EFF yamukelwa yi-helicopter ebiyihlola",
        selection={
            "selected_block_id": "block_0001",
            "confidence": 0.9,
            "rationale": "Strongest grounded visual detail.",
            "human_review_flags": ["Preserve attribution."],
        },
        output_root=output_root,
        job_id="job1",
    )

    assert updated["cover_hook"] == "I-EFF yamukelwa yi-helicopter ebiyihlola"
    assert tiktok_editor.read_json(seo_path)["cover_hook"] == updated["cover_hook"]
    assert tiktok_editor.read_json(output_json)["cover_hook"] == updated["cover_hook"]
    assert output_cover.read_text(encoding="utf-8").strip() == updated["cover_hook"]


def test_title_panel_expands_and_shrinks_font_to_fill_two_lines(tmp_path) -> None:
    short = tiktok_editor._render_title_panel(
        output_path=tmp_path / "short.png",
        title="I-EFF eLimpopo",
        width=1920,
        height=1080,
    )
    long = tiktok_editor._render_title_panel(
        output_path=tmp_path / "long.png",
        title=(
            "I-EFF ingena enqabeni ye-ANC eLimpopo, yamukelwa "
            "yi-helicopter ebiyihlola"
        ),
        width=1920,
        height=1080,
    )

    assert short["fit_action"] == "expanded"
    assert long["fit_action"] == "expanded"
    assert short["font_size"] > long["font_size"]
    assert long["line_count"] == 2
    assert short["truncated"] is False
    assert long["truncated"] is False


def test_edit_reuses_same_output_across_relative_and_absolute_paths(
    tmp_path,
) -> None:
    job_id = "job1"
    root = tmp_path / "jobs" / job_id
    master = root / "direct_dub" / "final_dubbed.mp4"
    output = root / "output" / f"final_dubbed_{job_id}.mp4"
    translation = root / "translation" / "transcript_zu.json"
    seo = root / "translation" / "tiktok_zu.json"
    report = root / "direct_dub" / "report.json"
    manifest = root / "output" / f"tiktok_edit_manifest_{job_id}.json"
    for path in (master, output, translation, seo, report, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"master")
    output.write_bytes(b"publication")
    translation.write_text('{"target_language": "zu-ZA"}', encoding="utf-8")
    seo.write_text('{"cover_hook": "Isihloko se-SEO"}', encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "blocks": [{"block_id": "block_0001"}],
                "outputs": {"canonical_final_video": str(master)},
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "render_version": tiktok_editor.TIKTOK_EDIT_RENDER_VERSION,
                "selection_prompt_version": (
                    tiktok_editor.TIKTOK_HOOK_PROMPT_VERSION
                ),
                "hook_text": "Isihloko se-SEO",
                "master_video": {"sha256": tiktok_editor.checksum(master)},
                "translation": {
                    "sha256_after": tiktok_editor.checksum(translation)
                },
                "title_panel": {"text": "Isihloko se-SEO"},
                "output": {
                    "path": os.path.relpath(output, Path.cwd()),
                    "sha256": tiktok_editor.checksum(output),
                },
            }
        ),
        encoding="utf-8",
    )
    job = SimpleNamespace(job_id=job_id, target_language="zu-ZA", media={})

    result = tiktok_editor.edit_tiktok_job(
        work_dir=tmp_path,
        job=job,
        provider=SimpleNamespace(),
    )

    assert result["idempotent_reuse"] is True
    assert job.media["tiktok_publication_video"] == str(output)
