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
                "opening_boundary_block_id": self.selected_block_id,
                "opening_payoff_block_id": self.selected_block_id,
                "opening_relationship": "standalone",
                "opening_boundary_rationale": (
                    "Earliest safe self-contained opening."
                ),
                "candidates": [
                    {
                        "candidate_id": f"candidate_{index + 1}",
                        "selected_block_id": self.selected_block_id,
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
            "end_ms": 8000,
            "source_text": "Welcome to the programme.",
            "translated_text": "Siyanamukela ohlelweni.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "SPEAKER_01",
            "start_ms": 19840,
            "end_ms": 26000,
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


def test_late_title_evidence_is_demoted_without_moving_opening_cut() -> None:
    class LaterTitleProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["candidates"][0]["selected_block_id"] = "block_0002"
            response.data["candidates"][0]["hook_text"] = (
                "I-EFF iveza imininingwane emangazayo"
            )
            return response

    result = tiktok_editor.select_tiktok_hook(
        provider=LaterTitleProvider("block_0001"),
        job_id="job1",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={},
        source_duration_seconds=120.0,
    )

    assert result["selected_block_id"] == "block_0001"
    assert result["cut_start_seconds"] == 2.0
    assert result["engagement_ranking"]["winner_source_block_id"] == "block_0001"
    assert result["selected_title_quality"]["payoff_start_latency_seconds"] == 0.0


def test_title_candidate_before_cut_is_rejected_without_aborting_publication() -> None:
    class MixedGroundingProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["candidates"][0]["selected_block_id"] = "block_0001"
            response.data["candidates"][0]["hook_text"] = "Isingeniso esidala"
            return response

    result = tiktok_editor.select_tiktok_hook(
        provider=MixedGroundingProvider("block_0002"),
        job_id="mixed-title-grounding",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={},
        source_duration_seconds=120.0,
    )

    assert result["selected_block_id"] == "block_0002"
    assert result["engagement_ranking"]["winner_candidate_id"] == "candidate_2"
    assert [
        item["candidate_id"]
        for item in result["engagement_ranking"]["rejected_candidates"]
    ] == ["candidate_1"]
    assert result["title_grounding_guard"]["opening_recovery_applied"] is False
    assert len(result["engagement_ranking"]["ranked_candidates"]) == 2


def test_all_titles_before_cut_expand_opening_to_earliest_grounded_block() -> None:
    class AllEarlierTitlesProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0002"
            response.data["opening_payoff_block_id"] = "block_0002"
            response.data["opening_relationship"] = "standalone"
            for candidate in response.data["candidates"]:
                candidate["selected_block_id"] = "block_0001"
            return response

    result = tiktok_editor.select_tiktok_hook(
        provider=AllEarlierTitlesProvider(),
        job_id="9bcc4df93dae4bb3a4fc15377d77b73e",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={},
        source_duration_seconds=120.0,
    )

    assert result["ai_requested_opening_block_id"] == "block_0002"
    assert result["selected_block_id"] == "block_0001"
    assert result["cut_start_seconds"] == 2.0
    assert result["opening_payoff_block_id"] == "block_0002"
    assert result["title_grounding_guard"]["opening_recovery_applied"] is True
    assert result["title_grounding_guard"]["effective_opening_block_id"] == "block_0001"
    assert result["engagement_ranking"]["rejected_candidates"] == []
    assert "opening_expanded_for_title_grounding" in result["editorial_seo"][
        "human_review_flags"
    ]


def test_title_grounding_recovery_still_preserves_preceding_question() -> None:
    class EarlierAnswerTitlesProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0003"
            response.data["opening_payoff_block_id"] = "block_0003"
            response.data["opening_relationship"] = "standalone"
            for candidate in response.data["candidates"]:
                candidate["selected_block_id"] = "block_0002"
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "INTERVIEWER",
            "start_ms": 0,
            "end_ms": 4_000,
            "source_text": "Why were the charges withdrawn?",
            "translated_text": "Kungani amacala asusiwe?",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "KGANYAGO",
            "start_ms": 4_500,
            "end_ms": 12_000,
            "source_text": "They were withdrawn because the investigation was not finalised.",
            "translated_text": "Asusiwe ngoba uphenyo belungakapheli.",
        },
        {
            "block_id": "block_0003",
            "speaker_id": "HOST",
            "start_ms": 30_000,
            "end_ms": 36_000,
            "source_text": "We will continue following the story.",
            "translated_text": "Sizoqhubeka nokulandela le ndaba.",
        },
    ]

    result = tiktok_editor.select_tiktok_hook(
        provider=EarlierAnswerTitlesProvider("block_0003"),
        job_id="9bcc4df93dae4bb3a4fc15377d77b73e",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=180.0,
    )

    assert result["ai_requested_opening_block_id"] == "block_0003"
    assert result["selected_block_id"] == "block_0001"
    assert result["cut_start_seconds"] == 0.0
    assert result["title_grounding_guard"]["opening_recovery_applied"] is True
    assert result["title_grounding_guard"]["question_anchor_recovery_applied"] is True
    assert result["title_grounding_guard"]["effective_opening_block_id"] == "block_0001"


def test_answer_opening_is_moved_back_to_preceding_question() -> None:
    class AnswerOnlyProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0003"
            response.data["opening_payoff_block_id"] = "block_0003"
            response.data["opening_relationship"] = "standalone"
            for candidate in response.data["candidates"]:
                candidate["selected_block_id"] = "block_0003"
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "HOST",
            "start_ms": 0,
            "end_ms": 4000,
            "source_text": "Welcome to the programme.",
            "translated_text": "Siyanamukela ohlelweni.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "INTERVIEWER",
            "start_ms": 5000,
            "end_ms": 11000,
            "source_text": "Why did you request immediate release from office?",
            "translated_text": "Kungani wacela ukukhululwa ngokushesha esikhundleni?",
        },
        {
            "block_id": "block_0003",
            "speaker_id": "JOHNSON",
            "start_ms": 11500,
            "end_ms": 22000,
            "source_text": "I requested it because the attacks had become unbearable.",
            "translated_text": "Ngakucela ngoba ukuhlaselwa kwase kungasabekezeleleki.",
        },
    ]

    result = tiktok_editor.select_tiktok_hook(
        provider=AnswerOnlyProvider(),
        job_id="question-anchor-job",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=60.0,
    )

    assert result["ai_requested_opening_block_id"] == "block_0003"
    assert result["selected_block_id"] == "block_0002"
    assert result["opening_anchor_block_id"] == "block_0002"
    assert result["opening_payoff_block_id"] == "block_0003"
    assert result["opening_relationship"] == "question_answer"
    assert result["cut_start_seconds"] == 5.0
    assert result["question_anchor_guard"]["guard_applied"] is True
    assert result["question_anchor_guard"]["protected_question_block_ids"] == [
        "block_0002"
    ]


def test_explicit_question_answer_opening_stays_on_question() -> None:
    class QuestionAnswerProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0002"
            response.data["opening_payoff_block_id"] = "block_0003"
            response.data["opening_relationship"] = "question_answer"
            for candidate in response.data["candidates"]:
                candidate["selected_block_id"] = "block_0003"
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "HOST",
            "start_ms": 0,
            "end_ms": 4000,
            "source_text": "Welcome to the programme.",
            "translated_text": "Siyanamukela ohlelweni.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "INTERVIEWER",
            "start_ms": 5000,
            "end_ms": 11000,
            "source_text": "Why did you request immediate release from office?",
            "translated_text": "Kungani wacela ukukhululwa ngokushesha esikhundleni?",
        },
        {
            "block_id": "block_0003",
            "speaker_id": "JOHNSON",
            "start_ms": 11500,
            "end_ms": 22000,
            "source_text": "I requested it because the attacks had become unbearable.",
            "translated_text": "Ngakucela ngoba ukuhlaselwa kwase kungasabekezeleleki.",
        },
    ]

    result = tiktok_editor.select_tiktok_hook(
        provider=QuestionAnswerProvider(),
        job_id="question-anchor-job",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=60.0,
    )

    assert result["selected_block_id"] == "block_0002"
    assert result["opening_payoff_block_id"] == "block_0003"
    assert result["opening_relationship"] == "question_answer"
    assert result["question_anchor_guard"]["guard_applied"] is False


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
    assert "-ss" not in command
    video_filter = command[command.index("-filter_complex") + 1]
    assert "trim=start=19.840000" in video_filter
    assert "atrim=start=19.840000" in video_filter
    assert "fps=fps=25/1:round=near" in video_filter
    assert "overlay=0:0" in video_filter
    # TikTok re-encodes every upload regardless, so the default favors a much
    # faster local encode over one optimized for standalone fidelity.
    assert command[command.index("-preset") + 1] == "veryfast"
    assert command[command.index("-crf") + 1] == "20"
    assert command[command.index("-b:a") + 1] == "256k"
    assert result["render_version"] == tiktok_editor.TIKTOK_EDIT_RENDER_VERSION
    assert output.read_bytes() == b"edited"


def test_hook_edit_cpu_threads_caps_the_libx264_encode(tmp_path, monkeypatch) -> None:
    """native-dub's --lite mode threads a cpu_threads cap through to this, the
    most CPU-heavy step in that pipeline -- confirms it lands as ffmpeg's own
    encoder-scoped ``-threads`` option, immediately after the codec name.
    """
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
            {"codec_type": "video", "codec_name": "h264", "profile": "Main", "pix_fmt": "yuv420p", "avg_frame_rate": "25/1"},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }
    monkeypatch.setattr(tiktok_editor, "probe", lambda path: source_info if path == master else rendered_info)
    monkeypatch.setattr(tiktok_editor, "_FONT_PATH", tmp_path / "font.ttf")
    tiktok_editor._FONT_PATH.write_bytes(b"font")

    def fake_title_panel(*, output_path, title, width, height):
        output_path.write_bytes(b"panel")
        return {
            "rendered_text": title, "line_count": 1, "font_size": 48, "minimum_font_size": 35,
            "maximum_font_size": 84, "fit_action": "unchanged", "truncated": False,
            "panel_bounds": [88, 852, 1832, 1068],
        }

    monkeypatch.setattr(tiktok_editor, "_render_title_panel", fake_title_panel)
    commands = []

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"edited")
        return SimpleNamespace(returncode=0)

    tiktok_editor.render_tiktok_hook_edit(
        master_video=master, output_path=output, manifest_path=manifest, hook_text_path=hook,
        selection={"selected_block_id": "block_0002", "cut_start_seconds": 19.84, "hook_text": "I-EFF ayihlehli!"},
        translation_path=translation, title_text="Isihloko esigqamile #ZuluTikTok #IsiZulu",
        runner=fake_runner, cpu_threads=4,
    )
    command = commands[0]
    codec_index = command.index("libx264")
    assert command[codec_index + 1] == "-threads"
    assert command[codec_index + 2] == "4"


def test_hook_edit_without_cpu_threads_never_adds_the_flag(tmp_path, monkeypatch) -> None:
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
            {"codec_type": "video", "codec_name": "h264", "profile": "Main", "pix_fmt": "yuv420p", "avg_frame_rate": "25/1"},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }
    monkeypatch.setattr(tiktok_editor, "probe", lambda path: source_info if path == master else rendered_info)
    monkeypatch.setattr(tiktok_editor, "_FONT_PATH", tmp_path / "font.ttf")
    tiktok_editor._FONT_PATH.write_bytes(b"font")

    def fake_title_panel(*, output_path, title, width, height):
        output_path.write_bytes(b"panel")
        return {
            "rendered_text": title, "line_count": 1, "font_size": 48, "minimum_font_size": 35,
            "maximum_font_size": 84, "fit_action": "unchanged", "truncated": False,
            "panel_bounds": [88, 852, 1832, 1068],
        }

    monkeypatch.setattr(tiktok_editor, "_render_title_panel", fake_title_panel)
    commands = []

    def fake_runner(command, check=True, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"edited")
        return SimpleNamespace(returncode=0)

    tiktok_editor.render_tiktok_hook_edit(
        master_video=master, output_path=output, manifest_path=manifest, hook_text_path=hook,
        selection={"selected_block_id": "block_0002", "cut_start_seconds": 19.84, "hook_text": "I-EFF ayihlehli!"},
        translation_path=translation, title_text="Isihloko esigqamile #ZuluTikTok #IsiZulu",
        runner=fake_runner,
    )
    assert "-threads" not in commands[0]


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
                "opening_anchor_block_id": "block_0001",
                "opening_payoff_block_id": "block_0001",
                "opening_relationship": "standalone",
                "question_anchor_guard": {"guard_applied": False},
                "selected_candidate": {"block_id": "block_0001"},
                "opening_payoff_candidate": {"block_id": "block_0001"},
                "editorial_seo": {
                    "source": "single-high-thinking-editorial-call",
                    "cover_hook": "Isihloko se-SEO",
                },
                "editorial_call_count": 1,
                "translation_operation_performed": False,
                "title_acronym_policy": {"fallback_applied": False},
                "title_person_prefix_policy": {"prefix_applied": False},
                "title_quality_guard": {"policy_version": "test"},
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


def test_publication_final_uses_master_and_records_hook_panel(tmp_path, monkeypatch) -> None:
    job_id = "1d3c57ad6f0b4718a637c14626ee6857"
    root = tmp_path / "jobs" / job_id
    master = root / "direct_dub" / "dubbed_master.mp4"
    master.parent.mkdir(parents=True)
    master.write_bytes(b"master")
    translation = root / "translation" / "transcript_zu.json"
    translation.parent.mkdir(parents=True)
    translation.write_text('{"target_language":"zu-ZA"}', encoding="utf-8")
    seo = root / "translation" / "tiktok_zu.json"
    seo.write_text('{"cover_hook":"Iphaneli"}', encoding="utf-8")
    report_path = root / "direct_dub" / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "blocks": _blocks(),
                "outputs": {"canonical_dubbed_master": str(master)},
            }
        ),
        encoding="utf-8",
    )
    job = SimpleNamespace(job_id=job_id, target_language="zu-ZA", media={})
    monkeypatch.setattr(tiktok_editor, "probe", lambda path: {"duration": 120.0})

    monkeypatch.setattr(
        tiktok_editor,
        "select_tiktok_hook",
        lambda **kwargs: {
            "selected_block_id": "block_0001",
            "cut_start_seconds": 2.0,
            "hook_text": "Iphaneli",
            "selection_prompt_version": tiktok_editor.TIKTOK_HOOK_PROMPT_VERSION,
        },
    )

    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"panel-final")
        kwargs["manifest_path"].write_text("{}", encoding="utf-8")
        return {"output": {"path": str(kwargs["output_path"])}}

    monkeypatch.setattr(tiktok_editor, "render_tiktok_hook_edit", fake_render)

    result = tiktok_editor.edit_tiktok_job(
        work_dir=tmp_path,
        job=job,
        provider=object(),
    )

    final_path = root / "output" / f"final_dubbed_{job_id}.mp4"
    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert captured["master_video"] == master
    assert captured["output_path"] == final_path
    assert final_path.read_bytes() == b"panel-final"
    assert updated["outputs"]["final_video"] == str(final_path)
    assert updated["outputs"]["publication_final_video"] == str(final_path)
    assert updated["outputs"]["hook_panel_included"] is True
    assert result["idempotent_reuse"] is False


def test_hook_request_uses_full_transcript_and_hook_profile() -> None:
    class ConfiguredProvider(FakeProvider):
        config = SimpleNamespace(
            hook_effort="low",
            hook_thinking_type="disabled",
            hook_max_output_tokens=9000,
        )

    blocks = [
        *_blocks(),
        {
            "block_id": "block_0003",
            "speaker_id": "SPEAKER_01",
            "start_ms": 90_000,
            "source_text": "The later evidence is important.",
            "translated_text": "Ubufakazi bakamuva bubalulekile.",
        },
    ]
    provider = ConfiguredProvider("block_0001")

    tiktok_editor.select_tiktok_hook(
        provider=provider,
        job_id="job-profile",
        target_language="zu-ZA",
        blocks=blocks,
        seo={"cover_hook": "Isihloko"},
        source_duration_seconds=120.0,
        maximum_intro_cut_seconds=30.0,
    )

    request = provider.requests[0]
    assert request.effort == "low"
    assert request.thinking_type == "disabled"
    assert request.max_output_tokens == 9000
    assert [
        item["block_id"] for item in request.payload["candidate_boundaries"]
    ] == ["block_0001", "block_0002"]
    assert [
        item["block_id"] for item in request.payload["full_rendered_transcript"]
    ] == ["block_0001", "block_0002", "block_0003"]
    assert request.payload["requirements"][
        "never_cut_between_question_and_direct_response"
    ] is True
    assert request.payload["requirements"][
        "opening_boundary_is_anchor_not_payoff"
    ] is True


def test_hook_title_can_reference_later_full_context_block() -> None:
    class LaterContextProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            for candidate in response.data["candidates"]:
                candidate["selected_block_id"] = "block_0003"
            return response

    blocks = [
        *_blocks(),
        {
            "block_id": "block_0003",
            "speaker_id": "SPEAKER_01",
            "start_ms": 90_000,
            "source_text": "The later evidence is important.",
            "translated_text": "Ubufakazi bakamuva bubalulekile.",
        },
    ]

    result = tiktok_editor.select_tiktok_hook(
        provider=LaterContextProvider("block_0001"),
        job_id="job-later-context",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=120.0,
        maximum_intro_cut_seconds=30.0,
    )

    assert result["selected_block_id"] == "block_0001"
    assert result["engagement_ranking"]["winner_source_block_id"] == "block_0003"


def test_explicit_response_cue_restores_real_interview_question() -> None:
    class RealClipProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0002"
            response.data["opening_payoff_block_id"] = "block_0002"
            response.data["opening_relationship"] = "standalone"
            for candidate in response.data["candidates"]:
                candidate["selected_block_id"] = "block_0002"
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "INTERVIEWER",
            "start_ms": 960,
            "end_ms": 27900,
            "source_text": (
                "Johnson says the attacks against her have taken a personal and "
                "professional toll. Are these the inevitable consequences of "
                "serious allegations against a public office bearer?"
            ),
            "translated_text": "Ingabe lena imiphumela engenakugwenywa?",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "GUEST",
            "start_ms": 29520,
            "end_ms": 140000,
            "source_text": (
                "Thank you for that question, Lopam. Before I address it directly, "
                "let me say we welcome the decision by Advocate Johnson."
            ),
            "translated_text": (
                "Ngiyabonga ngalowo mbuzo, Lopam. Ngaphambi kokuwuphendula ngqo, "
                "ake ngithi siyasamukela isinqumo sika-Advocate Johnson."
            ),
        },
    ]

    result = tiktok_editor.select_tiktok_hook(
        provider=RealClipProvider(),
        job_id="17d8715b86ce4eb7977f91af455c18d5",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=396.0,
    )

    assert result["ai_requested_opening_block_id"] == "block_0002"
    assert result["opening_anchor_block_id"] == "block_0001"
    assert result["opening_payoff_block_id"] == "block_0002"
    assert result["opening_relationship"] == "question_answer"
    assert result["cut_start_seconds"] == pytest.approx(0.96)
    assert result["question_anchor_guard"]["guard_applied"] is True
    assert result["question_anchor_guard"]["response_cue_detected"] is True


def test_legacy_manifest_without_question_anchor_contract_is_not_current() -> None:
    assert not tiktok_editor._is_current_question_anchor_selection(
        {
            "selection_prompt_version": tiktok_editor.TIKTOK_HOOK_PROMPT_VERSION,
            "selected_candidate": {"block_id": "block_0002"},
            "hook_text": "Legacy hook",
        }
    )


def _write_publication_title_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _publication_title_authority_job(tmp_path: Path, job_id: str = "job-title") -> Path:
    root = tmp_path / "jobs" / job_id
    _write_publication_title_json(
        root / "analysis" / "autocorrection_name_research.json",
        {
            "applied_corrections": [
                {
                    "raw_text": "Kaiser Kanyaku",
                    "canonical_text": "Kaizer Kganyago",
                    "entity_type": "person",
                },
                {
                    "raw_text": "Mr. Kanyaku",
                    "canonical_text": "Mr. Kganyago",
                    "entity_type": "person",
                },
            ]
        },
    )
    return root


def test_publication_title_authority_derives_surname_and_corrects_zulu_prefix(
    tmp_path: Path,
) -> None:
    root = _publication_title_authority_job(tmp_path)
    authority = tiktok_editor.load_publication_title_authority(root)

    title, policy = tiktok_editor.apply_publication_title_authority(
        "UKanyaku: Amacala kaKhumalo asusiwe njengoba uphenyo lungakapheli",
        authority,
    )

    assert title == (
        "UKganyago: Amacala kaKhumalo asusiwe njengoba uphenyo lungakapheli"
    )
    assert policy["adjusted"] is True
    assert any(
        item["raw_text"] == "Kanyaku"
        and item["canonical_text"] == "Kganyago"
        and item["derived"] is True
        for item in policy["applied_corrections"]
    )


def test_cached_publication_title_is_rerendered_without_ai(
    tmp_path: Path, monkeypatch
) -> None:
    job_id = "job-cached-title"
    root = _publication_title_authority_job(tmp_path, job_id)
    master = root / "direct_dub" / "dubbed_master.mp4"
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"master")
    translation = root / "translation" / "transcript_zu.json"
    _write_publication_title_json(translation, {"target_language": "zu-ZA"})
    seo_path = root / "translation" / "tiktok_zu.json"
    wrong = "UKanyaku: Amacala kaKhumalo asusiwe njengoba uphenyo lungakapheli"
    correct = "UKganyago: Amacala kaKhumalo asusiwe njengoba uphenyo lungakapheli"
    _write_publication_title_json(seo_path, {"title": wrong, "cover_hook": wrong})
    report_path = root / "direct_dub" / "report.json"
    _write_publication_title_json(
        report_path,
        {
            "blocks": [
                {
                    "block_id": "block_0001",
                    "speaker_id": "speaker-1",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "source_text": "Question?",
                    "translated_text": "Umbuzo?",
                }
            ],
            "outputs": {"canonical_dubbed_master": str(master)},
        },
    )
    output = root / "output" / f"final_dubbed_{job_id}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"old-panel")
    manifest_path = root / "output" / f"tiktok_edit_manifest_{job_id}.json"
    _write_publication_title_json(
        manifest_path,
        {
            "schema_version": "mathula-tiktok-edit-manifest-v1",
            "selection_prompt_version": tiktok_editor.TIKTOK_HOOK_PROMPT_VERSION,
            "opening_anchor_block_id": "block_0001",
            "opening_payoff_block_id": "block_0001",
            "opening_relationship": "standalone",
            "question_anchor_guard": {"guard_applied": False},
            "selected_candidate": {"block_id": "block_0001"},
            "opening_payoff_candidate": {"block_id": "block_0001"},
            "editorial_seo": {
                "source": "single-high-thinking-editorial-call",
                "title": wrong,
                "cover_hook": wrong,
            },
            "editorial_call_count": 1,
            "translation_operation_performed": False,
            "title_acronym_policy": {},
            "title_person_prefix_policy": {
                "lead_type": "person",
                "lead_text": "Kanyaku",
            },
            "title_length_policy": {},
            "title_quality_guard": {"policy_version": "test"},
            "hook_text": wrong,
            "cut_start_seconds": 0.0,
            "render_version": tiktok_editor.TIKTOK_EDIT_RENDER_VERSION,
            "master_video": {"sha256": tiktok_editor.checksum(master)},
            "translation": {"sha256_after": tiktok_editor.checksum(translation)},
            "output": {"path": str(output), "sha256": tiktok_editor.checksum(output)},
            "title_panel": {"text": wrong},
            "hook_edit_enabled": True,
        },
    )

    class NoAiProvider:
        def complete_structured(self, request):  # pragma: no cover - must not run
            raise AssertionError("publication title correction must not call AI")

    captured: dict = {}

    def fake_render(**kwargs):
        captured.update(kwargs)
        kwargs["output_path"].write_bytes(b"corrected-panel")
        return {
            **dict(kwargs["selection"]),
            "title_panel": {"text": kwargs["title_text"]},
            "output": {"path": str(kwargs["output_path"])},
        }

    monkeypatch.setattr(tiktok_editor, "render_tiktok_hook_edit", fake_render)
    monkeypatch.setattr(tiktok_editor, "_record_publication_artifacts", lambda **kwargs: None)

    result = tiktok_editor.edit_tiktok_job(
        work_dir=tmp_path,
        job=SimpleNamespace(job_id=job_id, target_language="zu-ZA", media={}),
        provider=NoAiProvider(),
    )

    assert captured["title_text"] == correct
    assert captured["selection"]["hook_text"] == correct
    assert result["title_panel"]["text"] == correct
    seo = json.loads(seo_path.read_text(encoding="utf-8"))
    assert seo["cover_hook"] == correct
    assert output.read_bytes() == b"corrected-panel"


def test_new_ai_title_is_bound_to_authoritative_correction(tmp_path: Path) -> None:
    root = _publication_title_authority_job(tmp_path, "job-fresh-title")
    authority = tiktok_editor.load_publication_title_authority(root)

    class Metadata:
        def to_dict(self):
            return {"provider": "test"}

    class TitleProvider:
        def complete_structured(self, request):
            scores = {
                "visual_impact": 8,
                "curiosity": 8,
                "specificity": 8,
                "stakes": 8,
                "immediacy": 8,
                "audience_relevance": 8,
            }
            candidates = []
            for index, title in enumerate(
                [
                    "Kanyaku: Amacala kaKhumalo asusiwe njengoba uphenyo lungakapheli",
                    "Kanyaku: Uphenyo lukaKhumalo alukapheli",
                    "Kanyaku: Kungani amacala kaKhumalo asusiwe",
                ]
            ):
                candidates.append(
                    {
                        "candidate_id": f"candidate_{index}",
                        "selected_block_id": "block_0001",
                        "hook_text": title,
                        "accessible_hook_text": title,
                        "title_lead": {"type": "person", "text": "Kanyaku"},
                        "rationale": "Grounded",
                        "confidence": 0.9 - index * 0.1,
                        "grounded": True,
                        "preserves_attribution": True,
                        "no_sensationalism": True,
                        "scores": scores,
                        "human_review_flags": [],
                    }
                )
            return SimpleNamespace(
                data={
                    "opening_boundary_block_id": "block_0001",
                    "opening_payoff_block_id": "block_0001",
                    "opening_relationship": "standalone",
                    "opening_boundary_rationale": "Self contained",
                    "primary_story_angle": "Charges withdrawn while investigation continues",
                    "caption": "Amacala asusiwe kodwa uphenyo luyaqhubeka.",
                    "search_keywords": ["Khumalo"],
                    "topic_hashtags": ["#Khumalo"],
                    "human_review_flags": [],
                    "candidates": candidates,
                },
                metadata=Metadata(),
            )

    result = tiktok_editor.select_tiktok_hook(
        provider=TitleProvider(),
        job_id="job-fresh-title",
        target_language="zu-ZA",
        blocks=[
            {
                "block_id": "block_0001",
                "speaker_id": "speaker-1",
                "start_ms": 0,
                "end_ms": 4000,
                "source_text": "Kganyago said the charges were withdrawn.",
                "translated_text": "UKganyago uthe amacala asusiwe.",
            }
        ],
        seo={},
        source_duration_seconds=30.0,
        publication_title_authority=authority,
    )

    assert result["hook_text"].startswith("UKganyago:")
    assert "Kanyaku" not in result["hook_text"]
    assert result["publication_title_authority"]["adjusted"] is True


def test_generic_reporter_title_is_rejected_when_specific_contradiction_exists() -> None:
    class TitleQualityProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0001"
            response.data["opening_payoff_block_id"] = "block_0001"
            response.data["opening_relationship"] = "standalone"
            candidates = response.data["candidates"]

            candidates[0].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002"],
                    "editorial_angle": "other",
                    "hook_text": (
                        "Umbiki we-SABC: kungani abaphenyi befuna amalungu wonke?"
                    ),
                    "accessible_hook_text": (
                        "Umbiki we-SABC: kungani abaphenyi befuna amalungu wonke?"
                    ),
                    "title_lead": {"type": "topic", "text": ""},
                    "scores": {
                        "visual_impact": 10,
                        "curiosity": 10,
                        "specificity": 10,
                        "stakes": 10,
                        "immediacy": 10,
                        "audience_relevance": 10,
                    },
                    "confidence": 0.99,
                }
            )
            candidates[1].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002", "block_0003"],
                    "editorial_angle": "contradiction",
                    "hook_text": (
                        "UJohnson uthi wayengazi ngeTask Team—kodwa i-subpoena yakhe yafuna amalungu"
                    ),
                    "accessible_hook_text": (
                        "UJohnson uthi wayengazi ngeTask Team—kodwa i-subpoena yakhe yafuna amalungu"
                    ),
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {
                        "visual_impact": 6,
                        "curiosity": 6,
                        "specificity": 6,
                        "stakes": 6,
                        "immediacy": 6,
                        "audience_relevance": 6,
                    },
                    "confidence": 0.88,
                }
            )
            candidates[2].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002"],
                    "editorial_angle": "document_evidence",
                    "hook_text": "I-subpoena kaJohnson yafuna imininingwane yeTask Team",
                    "accessible_hook_text": (
                        "I-subpoena kaJohnson yafuna imininingwane yeTask Team"
                    ),
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {
                        "visual_impact": 5,
                        "curiosity": 5,
                        "specificity": 5,
                        "stakes": 5,
                        "immediacy": 5,
                        "audience_relevance": 5,
                    },
                    "confidence": 0.80,
                }
            )
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "ANCHOR",
            "start_ms": 0,
            "end_ms": 5_000,
            "source_text": "Let us go live to our reporter.",
            "translated_text": "Asiye ngqo kumbiki wethu.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "REPORTER",
            "start_ms": 5_000,
            "end_ms": 12_000,
            "source_text": (
                "The subpoena authorised by Advocate Andrea Johnson requested "
                "information about the commander and members of the Task Team."
            ),
            "translated_text": (
                "I-subpoena ka-Advocate Andrea Johnson yafuna ulwazi ngomkhuzi "
                "namalungu eTask Team."
            ),
        },
        {
            "block_id": "block_0003",
            "speaker_id": "REPORTER",
            "start_ms": 12_000,
            "end_ms": 18_000,
            "source_text": (
                "She said she had no idea IDAC was investigating the Task Team."
            ),
            "translated_text": (
                "Uthe wayengazi ukuthi i-IDAC yayiphenya iTask Team."
            ),
        },
    ]

    result = tiktok_editor.select_tiktok_hook(
        provider=TitleQualityProvider("block_0001"),
        job_id="title-quality",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=60.0,
    )

    assert result["engagement_ranking"]["winner_candidate_id"] == "candidate_2"
    assert result["engagement_ranking"]["winner_evidence_block_ids"] == [
        "block_0002",
        "block_0003",
    ]
    assert result["hook_text"].startswith("UJohnson")
    assert "Umbiki we-SABC" not in result["hook_text"]
    rejected = result["title_quality_guard"]["rejected_candidates"]
    assert [item["candidate_id"] for item in rejected] == [
        "candidate_1",
        "candidate_3",
    ]
    assert rejected[0]["reason"] == "generic_newsroom_lead"
    assert rejected[0]["payoff_start_latency_seconds"] == 5.0
    assert rejected[1]["reason"] == "not_tabloid_ready"


def test_multiblock_title_is_rejected_when_any_required_evidence_is_cut() -> None:
    class MultiBlockGroundingProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0002"
            response.data["opening_payoff_block_id"] = "block_0002"
            response.data["opening_relationship"] = "standalone"
            response.data["candidates"][0].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0001", "block_0002"],
                    "editorial_angle": "contradiction",
                    "hook_text": "UJohnson uyaphika—kodwa i-subpoena yakhe iyamphikisa",
                    "accessible_hook_text": (
                        "UJohnson uyaphika—kodwa i-subpoena yakhe iyamphikisa"
                    ),
                    "title_lead": {"type": "person", "text": "Johnson"},
                }
            )
            for candidate in response.data["candidates"][1:]:
                candidate["selected_block_id"] = "block_0002"
                candidate["evidence_block_ids"] = ["block_0002"]
                candidate["editorial_angle"] = "decision_consequence"
            return response

    result = tiktok_editor.select_tiktok_hook(
        provider=MultiBlockGroundingProvider("block_0002"),
        job_id="multi-block-grounding",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={},
        source_duration_seconds=120.0,
    )

    rejected = result["title_grounding_guard"]["rejected_candidates"]
    assert rejected[0]["candidate_id"] == "candidate_1"
    assert rejected[0]["removed_evidence_block_ids"] == ["block_0001"]
    assert result["engagement_ranking"]["winner_candidate_id"] == "candidate_2"


def test_truth_constrained_tabloid_hook_beats_neutral_document_summary() -> None:
    class TabloidProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0001"
            response.data["opening_payoff_block_id"] = "block_0001"
            response.data["opening_relationship"] = "standalone"
            candidates = response.data["candidates"]
            candidates[0].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002"],
                    "editorial_angle": "document_evidence",
                    "hook_text": "I-subpoena kaJohnson yafuna imininingwane yeTask Team",
                    "accessible_hook_text": "I-subpoena kaJohnson yafuna imininingwane yeTask Team",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {
                        "visual_impact": 10,
                        "curiosity": 10,
                        "specificity": 10,
                        "stakes": 10,
                        "immediacy": 10,
                        "audience_relevance": 10,
                    },
                    "confidence": 0.99,
                }
            )
            candidates[1].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002", "block_0003"],
                    "editorial_angle": "contradiction",
                    "hook_text": "UJohnson uthi wayengazi—kodwa i-subpoena yakhe iveza okunye",
                    "accessible_hook_text": "UJohnson uthi wayengazi—kodwa i-subpoena yakhe iveza okunye",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {
                        "visual_impact": 6,
                        "curiosity": 6,
                        "specificity": 6,
                        "stakes": 6,
                        "immediacy": 6,
                        "audience_relevance": 6,
                    },
                    "confidence": 0.85,
                }
            )
            candidates[2].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002", "block_0003"],
                    "editorial_angle": "institutional_overreach",
                    "hook_text": "I-Item 6 ishiya i-IDAC nemibuzo emikhulu",
                    "accessible_hook_text": "I-Item 6 ishiya i-IDAC nemibuzo emikhulu",
                    "title_lead": {"type": "institution", "text": "IDAC"},
                    "scores": {
                        "visual_impact": 5,
                        "curiosity": 5,
                        "specificity": 5,
                        "stakes": 5,
                        "immediacy": 5,
                        "audience_relevance": 5,
                    },
                    "confidence": 0.8,
                }
            )
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "ANCHOR",
            "start_ms": 0,
            "end_ms": 4_000,
            "source_text": "Let us go live.",
            "translated_text": "Asiye ngqo.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "REPORTER",
            "start_ms": 4_000,
            "end_ms": 10_000,
            "source_text": "Johnson's subpoena requested Task Team information.",
            "translated_text": "I-subpoena kaJohnson yafuna ulwazi ngeTask Team.",
        },
        {
            "block_id": "block_0003",
            "speaker_id": "REPORTER",
            "start_ms": 10_000,
            "end_ms": 16_000,
            "source_text": "She said she did not know it was being investigated.",
            "translated_text": "Uthe wayengazi ukuthi yayiphenywa.",
        },
    ]
    result = tiktok_editor.select_tiktok_hook(
        provider=TabloidProvider("block_0001"),
        job_id="truthful-tabloid",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=60.0,
    )

    assert result["engagement_ranking"]["winner_candidate_id"] == "candidate_2"
    assert result["selected_title_quality"]["tabloid_ready"] is True
    assert result["selected_title_quality"]["has_tabloid_punch"] is True
    assert result["selected_title_quality"]["has_curiosity_gap"] is True
    assert result["hook_text"] == (
        "UJohnson uthi wayengazi—kodwa i-subpoena yakhe iveza okunye"
    )


def test_unsupported_lying_accusation_loses_to_defensible_tabloid_hook() -> None:
    class UnsafeBaitProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0001"
            response.data["opening_payoff_block_id"] = "block_0001"
            response.data["opening_relationship"] = "standalone"
            candidates = response.data["candidates"]
            candidates[0].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002", "block_0003"],
                    "editorial_angle": "contradiction",
                    "hook_text": "UJohnson ubanjwe eqamba amanga ngeTask Team",
                    "accessible_hook_text": "UJohnson ubanjwe eqamba amanga ngeTask Team",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {
                        "visual_impact": 10,
                        "curiosity": 10,
                        "specificity": 10,
                        "stakes": 10,
                        "immediacy": 10,
                        "audience_relevance": 10,
                    },
                    "confidence": 0.99,
                }
            )
            candidates[1].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002", "block_0003"],
                    "editorial_angle": "contradiction",
                    "hook_text": "UJohnson uthi wayengazi—kodwa i-subpoena yakhe iyamphikisa",
                    "accessible_hook_text": "UJohnson uthi wayengazi—kodwa i-subpoena yakhe iyamphikisa",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {
                        "visual_impact": 7,
                        "curiosity": 7,
                        "specificity": 7,
                        "stakes": 7,
                        "immediacy": 7,
                        "audience_relevance": 7,
                    },
                    "confidence": 0.9,
                }
            )
            candidates[2].update(
                {
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002"],
                    "editorial_angle": "document_evidence",
                    "hook_text": "I-subpoena kaJohnson iveza imibuzo ngeTask Team",
                    "accessible_hook_text": "I-subpoena kaJohnson iveza imibuzo ngeTask Team",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {
                        "visual_impact": 5,
                        "curiosity": 5,
                        "specificity": 5,
                        "stakes": 5,
                        "immediacy": 5,
                        "audience_relevance": 5,
                    },
                    "confidence": 0.8,
                }
            )
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "ANCHOR",
            "start_ms": 0,
            "end_ms": 4_000,
            "source_text": "Let us go live.",
            "translated_text": "Asiye ngqo.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "REPORTER",
            "start_ms": 4_000,
            "end_ms": 10_000,
            "source_text": "Johnson's subpoena requested Task Team information.",
            "translated_text": "I-subpoena kaJohnson yafuna ulwazi ngeTask Team.",
        },
        {
            "block_id": "block_0003",
            "speaker_id": "REPORTER",
            "start_ms": 10_000,
            "end_ms": 16_000,
            "source_text": "She said she did not know it was being investigated.",
            "translated_text": "Uthe wayengazi ukuthi yayiphenywa.",
        },
    ]
    result = tiktok_editor.select_tiktok_hook(
        provider=UnsafeBaitProvider("block_0001"),
        job_id="unsafe-bait",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=60.0,
    )

    assert result["engagement_ranking"]["winner_candidate_id"] == "candidate_2"
    rejected = result["title_quality_guard"]["rejected_candidates"]
    assert rejected[0]["candidate_id"] == "candidate_1"
    assert rejected[0]["reason"] == "unsupported_accusation_language"
    assert "eqamba amanga" not in result["hook_text"]


def test_named_actor_but_reveal_formula_beats_higher_scored_direct_tabloid() -> None:
    class HybridProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0001"
            response.data["opening_payoff_block_id"] = "block_0002"
            response.data["opening_relationship"] = "setup_payoff"
            candidates = response.data["candidates"]
            candidates[0].update(
                {
                    "candidate_id": "direct_tabloid",
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002", "block_0003"],
                    "editorial_angle": "contradiction",
                    "hook_strategy": "direct_contradiction",
                    "withheld_answer": "",
                    "hook_text": "I-subpoena kaJohnson iyamphikisa ngeTask Team",
                    "accessible_hook_text": "I-subpoena kaJohnson iyamphikisa ngeTask Team",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {key: 10 for key in candidates[0]["scores"]},
                }
            )
            candidates[1].update(
                {
                    "candidate_id": "hybrid_clickbait",
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002", "block_0003"],
                    "editorial_angle": "contradiction",
                    "hook_strategy": "named_actor_but_reveal",
                    "withheld_answer": "What the subpoena requested",
                    "hook_text": "UJohnson uthi wayengazi—kodwa i-subpoena yakhe iveza okunye",
                    "accessible_hook_text": "UJohnson uthi wayengazi—kodwa i-subpoena yakhe iveza okunye",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {key: 6 for key in candidates[1]["scores"]},
                }
            )
            candidates[2].update(
                {
                    "candidate_id": "neutral",
                    "selected_block_id": "block_0002",
                    "evidence_block_ids": ["block_0002"],
                    "editorial_angle": "other",
                    "hook_strategy": "other",
                    "withheld_answer": "",
                    "hook_text": "UJohnson nePolitical Killings Task Team",
                    "accessible_hook_text": "UJohnson nePolitical Killings Task Team",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {key: 8 for key in candidates[2]["scores"]},
                }
            )
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "ANCHOR",
            "start_ms": 0,
            "end_ms": 4_000,
            "source_text": "The subpoena raises a central question.",
            "translated_text": "I-subpoena iphakamisa umbuzo omkhulu.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "REPORTER",
            "start_ms": 4_000,
            "end_ms": 10_000,
            "source_text": "Johnson's subpoena requested Task Team information.",
            "translated_text": "I-subpoena kaJohnson yafuna imininingwane yeTask Team.",
        },
        {
            "block_id": "block_0003",
            "speaker_id": "REPORTER",
            "start_ms": 10_000,
            "end_ms": 16_000,
            "source_text": "Johnson said she did not know IDAC was investigating it.",
            "translated_text": "UJohnson uthe wayengazi ukuthi i-IDAC yayiyiphenya.",
        },
    ]

    result = tiktok_editor.select_tiktok_hook(
        provider=HybridProvider("block_0001"),
        job_id="hybrid-formula",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=60.0,
    )

    assert result["engagement_ranking"]["winner_candidate_id"] == "hybrid_clickbait"
    quality = result["selected_title_quality"]
    assert quality["hybrid_formula_ready"] is True
    assert quality["named_actor_formula"] is True
    assert quality["subject_disclosed"] is True
    assert quality["explanation_withheld"] is True
    assert quality["immediate_payoff_ready"] is True
    assert result["title_quality_guard"]["preferred_formula"] == (
        "named_actor_claim_or_action_but_reveal_or_consequence"
    )


def test_pure_clickbait_that_hides_whole_subject_is_rejected() -> None:
    class PureClickbaitProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0001"
            response.data["opening_payoff_block_id"] = "block_0001"
            candidates = response.data["candidates"]
            candidates[0].update(
                {
                    "candidate_id": "pure_clickbait",
                    "selected_block_id": "block_0001",
                    "evidence_block_ids": ["block_0001"],
                    "editorial_angle": "other",
                    "hook_strategy": "other",
                    "withheld_answer": "Everything",
                    "hook_text": "Nakhu okushaqisayo okwenzekile",
                    "accessible_hook_text": "Nakhu okushaqisayo okwenzekile",
                    "title_lead": {"type": "topic", "text": ""},
                    "scores": {key: 10 for key in candidates[0]["scores"]},
                }
            )
            candidates[1].update(
                {
                    "candidate_id": "grounded_hybrid",
                    "selected_block_id": "block_0001",
                    "evidence_block_ids": ["block_0001"],
                    "editorial_angle": "decision_consequence",
                    "hook_strategy": "named_actor_but_consequence",
                    "withheld_answer": "Why resignation does not end accountability",
                    "hook_text": "UJohnson usesulile—kodwa lokhu akumkhululi",
                    "accessible_hook_text": "UJohnson usesulile—kodwa lokhu akumkhululi",
                    "title_lead": {"type": "person", "text": "Johnson"},
                    "scores": {key: 6 for key in candidates[1]["scores"]},
                }
            )
            candidates[2].update(
                {
                    "candidate_id": "neutral",
                    "selected_block_id": "block_0001",
                    "evidence_block_ids": ["block_0001"],
                    "editorial_angle": "other",
                    "hook_strategy": "other",
                    "withheld_answer": "",
                    "hook_text": "Ukusula kukaJohnson",
                    "accessible_hook_text": "Ukusula kukaJohnson",
                    "title_lead": {"type": "person", "text": "Johnson"},
                }
            )
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "GUEST",
            "start_ms": 0,
            "end_ms": 12_000,
            "source_text": "Johnson resigned but still has to account before the Commission.",
            "translated_text": "UJohnson usesulile kodwa kusamele aziphendulele eKhomishini.",
        }
    ]
    result = tiktok_editor.select_tiktok_hook(
        provider=PureClickbaitProvider("block_0001"),
        job_id="pure-clickbait",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=30.0,
    )
    assert result["engagement_ranking"]["winner_candidate_id"] == "grounded_hybrid"
    rejected = result["title_quality_guard"]["rejected_candidates"]
    pure = next(item for item in rejected if item["candidate_id"] == "pure_clickbait")
    assert pure["reason"] == "unsupported_accusation_language" or pure["reason"] == "whole_subject_withheld"
    # "okuthusayo" variants must never beat a title that names the actual subject.
    assert result["selected_title_quality"]["subject_disclosed"] is True


def test_late_payoff_candidate_loses_to_equally_grounded_early_candidate() -> None:
    class PayoffProvider(FakeProvider):
        def complete_structured(self, request):
            response = super().complete_structured(request)
            response.data["opening_boundary_block_id"] = "block_0001"
            response.data["opening_payoff_block_id"] = "block_0001"
            candidates = response.data["candidates"]
            candidates[0].update(
                {
                    "candidate_id": "late",
                    "selected_block_id": "block_0003",
                    "evidence_block_ids": ["block_0003"],
                    "editorial_angle": "named_actor_consequence",
                    "hook_strategy": "named_actor_but_consequence",
                    "withheld_answer": "The later consequence",
                    "hook_text": "UMalema uxwayisa i-EFF—kodwa lokhu kuvela kamuva",
                    "accessible_hook_text": "UMalema uxwayisa i-EFF—kodwa lokhu kuvela kamuva",
                    "title_lead": {"type": "person", "text": "Malema"},
                    "scores": {key: 10 for key in candidates[0]["scores"]},
                }
            )
            candidates[1].update(
                {
                    "candidate_id": "early",
                    "selected_block_id": "block_0001",
                    "evidence_block_ids": ["block_0001"],
                    "editorial_angle": "named_actor_consequence",
                    "hook_strategy": "named_actor_but_consequence",
                    "withheld_answer": "Why stagnation threatens the party",
                    "hook_text": "UMalema wesaba ukuma kwe-EFF—kodwa kungani?",
                    "accessible_hook_text": "UMalema wesaba ukuma kwe-EFF—kodwa kungani?",
                    "title_lead": {"type": "person", "text": "Malema"},
                    "scores": {key: 6 for key in candidates[1]["scores"]},
                }
            )
            candidates[2].update(
                {
                    "candidate_id": "neutral",
                    "selected_block_id": "block_0001",
                    "evidence_block_ids": ["block_0001"],
                    "editorial_angle": "other",
                    "hook_strategy": "other",
                    "withheld_answer": "",
                    "hook_text": "I-EFF ne-manifesto ka-2026",
                    "accessible_hook_text": "I-EFF ne-manifesto ka-2026",
                    "title_lead": {"type": "organisation", "text": "EFF"},
                }
            )
            return response

    blocks = [
        {
            "block_id": "block_0001",
            "speaker_id": "REPORTER",
            "start_ms": 0,
            "end_ms": 10_000,
            "source_text": "Malema says the EFF fears stagnation in 2026.",
            "translated_text": "UMalema uthi i-EFF yesaba ukuma ngo-2026.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "REPORTER",
            "start_ms": 10_000,
            "end_ms": 55_000,
            "source_text": "The manifesto contains several proposals.",
            "translated_text": "I-manifesto iqukethe iziphakamiso eziningi.",
        },
        {
            "block_id": "block_0003",
            "speaker_id": "REPORTER",
            "start_ms": 70_000,
            "end_ms": 80_000,
            "source_text": "A later controversy could affect the campaign.",
            "translated_text": "Impikiswano yakamuva ingaphazamisa umkhankaso.",
        },
    ]
    result = tiktok_editor.select_tiktok_hook(
        provider=PayoffProvider("block_0001"),
        job_id="payoff-latency",
        target_language="zu-ZA",
        blocks=blocks,
        seo={},
        source_duration_seconds=120.0,
        maximum_intro_cut_seconds=90.0,
    )
    assert result["engagement_ranking"]["winner_candidate_id"] == "early"
    late = next(
        item
        for item in result["title_quality_guard"]["rejected_candidates"]
        if item["candidate_id"] == "late"
    )
    assert late["reason"] == "late_title_payoff"
    assert late["payoff_start_latency_seconds"] == 70.0
