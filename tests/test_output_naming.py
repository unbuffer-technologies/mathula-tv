from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathula_tv.output_naming import (
    OutputNamingError,
    load_seo_output_context,
    publish_title_named_outputs,
    safe_title_stem,
)


def _write_seo(job_root: Path, title: str) -> Path:
    path = job_root / "translation" / "youtube_zu.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "title": title,
                "description": "Incazelo yevidiyo",
                "tags": ["Madlanga Commission", "Izindaba NgesiZulu"],
                "thumbnail_hook": "WAVUMA IPHUTHA",
                "thumbnail_subhook": "Ubufakazi buyaphikisana",
                "chapters": [
                    {"timestamp": "00:00", "title": "Isingeniso"},
                    {"timestamp": "01:23", "title": "Ubufakazi"},
                ],
                "tiktok": {
                    "caption": "Umbhalo weTikTok",
                    "hashtags": ["#MadlangaCommission", "#IzindabaNgesiZulu"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_safe_title_stem_keeps_readable_title() -> None:
    assert safe_title_stem(
        "U-Andrea Johnson Uvuma Iphutha Ebufakazini | Madlanga Commission NgesiZulu"
    ) == (
        "U-Andrea Johnson Uvuma Iphutha Ebufakazini - "
        "Madlanga Commission NgesiZulu"
    )


def test_safe_title_stem_removes_path_characters() -> None:
    stem = safe_title_stem('I-IDAC: Imibuzo / Ubufakazi? "Namuhla"')
    assert "/" not in stem
    assert ":" not in stem
    assert "?" not in stem
    assert '"' not in stem
    assert stem.startswith("I-IDAC")


def test_publish_creates_video_and_individual_seo_files(tmp_path: Path) -> None:
    job_root = tmp_path / "job"
    title = "U-Andrea Johnson Uvuma Iphutha | Madlanga Commission NgesiZulu"
    _write_seo(job_root, title)
    canonical = job_root / "direct_dub" / "final_dubbed.mp4"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"video-bytes")

    context = load_seo_output_context(job_root, "zu-ZA")
    old_package = (
        job_root
        / "output"
        / "U-Andrea Johnson Uvuma Iphutha - Madlanga Commission NgesiZulu"
        " - SEO Package.tar.gz"
    )
    old_package.parent.mkdir(parents=True, exist_ok=True)
    old_package.write_bytes(b"obsolete")

    outputs = publish_title_named_outputs(job_root, canonical, context=context)

    assert outputs.final_video.name == (
        "U-Andrea Johnson Uvuma Iphutha - Madlanga Commission NgesiZulu.mp4"
    )
    assert outputs.final_video.read_bytes() == b"video-bytes"
    assert not old_package.exists()

    assert {path.name for path in outputs.seo_files.values()} == {
        (
            "tiktok_caption_zu_U-Andrea Johnson Uvuma Iphutha - "
            "Madlanga Commission NgesiZulu.txt"
        ),
        (
            "tiktok_hashtags_zu_U-Andrea Johnson Uvuma Iphutha - "
            "Madlanga Commission NgesiZulu.txt"
        ),
        "youtube_chapters_zu.txt",
        "youtube_description_zu.txt",
        "youtube_tags_zu.txt",
        "youtube_thumbnail_zu.txt",
        "youtube_title_zu.txt",
        "youtube_zu.json",
    }
    assert outputs.seo_files["youtube_title"].read_text(encoding="utf-8").strip() == title
    assert (
        outputs.seo_files["tiktok_caption"].read_text(encoding="utf-8").strip()
        == "Umbhalo weTikTok"
    )
    assert not list((job_root / "output").glob("*SEO Package.tar.gz"))


def test_missing_seo_title_fails_before_finalization(tmp_path: Path) -> None:
    path = tmp_path / "translation" / "youtube_zu.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(OutputNamingError, match="no approved title"):
        load_seo_output_context(tmp_path, "zu-ZA")
