from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathula_tv import x_ingestion as module
from mathula_tv import youtube_captions
from mathula_tv import youtube_ingestion as shared

URL = "https://x.com/Newzroom405/status/2125522879982829740"


def test_only_a_link_to_one_post_of_x_is_an_x_url() -> None:
    assert module.is_x_url(URL)
    assert module.status_id(URL) == "2125522879982829740"
    assert module.is_x_url("https://twitter.com/Newzroom405/status/2125522879982829740?s=20")
    assert module.is_x_url("https://mobile.x.com/Newzroom405/status/2125522879982829740/video/1")
    assert module.is_x_url("https://x.com/i/status/2125522879982829740")
    assert module.is_x_url("https://x.com/i/web/status/2125522879982829740")
    assert not module.is_x_url("https://x.com/Newzroom405")                              # a profile is not a video
    assert not module.is_x_url("https://x.com/search?q=newzroom")
    assert not module.is_x_url("https://x.com/Newzroom405/status/abc")
    assert not module.is_x_url("https://notx.com/Newzroom405/status/2125522879982829740")
    assert not module.is_x_url("https://www.youtube.com/watch?v=abc123")
    assert not module.is_x_url("ftp://x.com/Newzroom405/status/2125522879982829740")
    assert not module.is_x_url("relative/video.mp4")
    assert not shared.is_youtube_url(URL)                                                # and a link is one site's or the other's


def _metadata(**extra):
    return {"id": "2125522879982829740", "title": "Newzroom Afrika - ANC says talks are done", "duration": 64.5, "webpage_url": URL, "uploader": "Newzroom Afrika",
            "uploader_id": "Newzroom405", "description": "ANC SG Fikile Mbalula on the coalition talks in Sandton.", "extractor_key": "Twitter", **extra}


def test_the_download_is_made_in_its_own_directory_and_the_manifest_keeps_where_it_came_from(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_metadata(url, **kwargs):
        seen["metadata"] = (url, kwargs)
        return _metadata()

    def fake_download(url, directory, **kwargs):
        path = directory / "source-2125522879982829740.mp4"
        path.write_bytes(b"fake-media")
        return path

    monkeypatch.setattr(module, "_read_metadata", fake_metadata)
    monkeypatch.setattr(module, "_download", fake_download)
    with module.download_x_video("https://twitter.com/Newzroom405/status/2125522879982829740?s=20", max_duration_seconds=600) as download:
        temporary = download.path
        assert temporary.is_file() and "mathula-x-" in str(temporary.parent)
        manifest = download.manifest()
        assert manifest["schema_version"] == "x-source-v1" and manifest["status_id"] == "2125522879982829740" and manifest["video_id"] == "2125522879982829740"
        assert manifest["channel"] == "Newzroom Afrika" and manifest["uploader"] == "Newzroom405"
        assert manifest["description"] == "ANC SG Fikile Mbalula on the coalition talks in Sandton."
        assert manifest["playlist_allowed"] is False and manifest["maximum_height"] == 1080 and manifest["requested_url"].startswith("https://twitter.com/")
    assert not temporary.exists()                                                       # nothing is left behind
    assert seen["metadata"][1]["site"] == "X" and "--playlist-items" in seen["metadata"][1]["extra_args"]


def test_a_post_with_several_videos_gives_its_first_and_a_post_without_one_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    playlist = {"_type": "playlist", "id": "2125522879982829740", "title": "Newzroom Afrika - thread", "webpage_url": URL, "uploader": "Newzroom Afrika", "uploader_id": "Newzroom405",
                "entries": [{"id": "2125522879982829740", "duration": 30.0, "title": "first"}, {"id": "2125522879982829740", "duration": 99.0, "title": "second"}]}
    merged = module._first_video(playlist)
    assert merged["duration"] == 30.0 and merged["uploader"] == "Newzroom Afrika" and "entries" not in merged
    monkeypatch.setattr(module, "_read_metadata", lambda url, **kw: _metadata(duration=None))
    with pytest.raises(module.XIngestionError, match="no video"):
        with module.download_x_video(URL):
            raise AssertionError("nothing should be downloaded")


def test_a_live_broadcast_and_a_video_over_the_limit_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_read_metadata", lambda url, **kw: _metadata(is_live=True))
    with pytest.raises(module.XIngestionError, match="live"):
        with module.download_x_video(URL):
            raise AssertionError("not downloaded")
    monkeypatch.setattr(module, "_read_metadata", lambda url, **kw: _metadata(duration=601))
    with pytest.raises(module.XIngestionError, match="duration limit"):
        with module.download_x_video(URL, max_duration_seconds=600):
            raise AssertionError("not downloaded")
    with pytest.raises(module.XIngestionError, match="Not a link to one X post"):
        with module.download_x_video("https://x.com/Newzroom405"):
            raise AssertionError("not downloaded")


def test_the_download_asks_yt_dlp_for_one_progressive_video_in_a_directory_of_its_own(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    actual = tmp_path / "source-2125522879982829740.mp4"
    actual.write_bytes(b"complete-media")
    captured: list[str] = []

    def fake_run_live(command, *, purpose, site):
        assert purpose == "download" and site == "X"           # a test must never reach yt-dlp and the network
        captured.extend(command)

    monkeypatch.setattr(module, "_run_live", fake_run_live)
    monkeypatch.setattr(module, "_yt_dlp_command", lambda site="YouTube": ["python", "-m", "yt_dlp"])
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    assert module._download(URL, tmp_path, cookies_file=cookies) == actual.resolve()
    assert module.X_FORMAT_SELECTOR in captured and "--playlist-items" in captured and "--windows-filenames" in captured and "--cookies" in captured
    assert str(tmp_path / "source-%(id)s.%(ext)s") in captured and captured[-1] == URL


def test_the_cookies_file_comes_from_the_environment_when_none_is_given(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cookies = tmp_path / "x-cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setenv("MATHULA_X_COOKIES_FILE", str(cookies))
    assert module._resolve_x_cookies(None) == cookies
    monkeypatch.delenv("MATHULA_X_COOKIES_FILE")
    assert module._resolve_x_cookies(None) is None


def test_the_job_records_its_x_source_and_the_posts_text_is_read_like_a_youtube_description(tmp_path: Path) -> None:
    class FakeJob:
        job_id = "job-x"
        media: dict = {}
        objects: dict = {}
        providers: dict = {}

    class FakeJobs:
        def job_dir(self, _job_id: str) -> Path:
            return tmp_path / "jobs/job-x"

        def save(self, job) -> None:
            pass

    class FakeOrchestrator:
        jobs = FakeJobs()
        gcs = None

    media = tmp_path / "source-2125522879982829740.mp4"
    media.write_bytes(b"video")
    download = module.XDownload(path=media, requested_url=URL, webpage_url=URL, status_id="2125522879982829740", title="Newzroom Afrika - ANC says talks are done", author="Newzroom Afrika",
                                author_handle="Newzroom405", duration=64.5, extractor="Twitter", downloaded_at="2026-11-25T00:00:00+00:00", text="ANC SG Fikile Mbalula on the coalition talks in Sandton.")
    job = FakeJob()
    module.record_x_submission(FakeOrchestrator(), job, download)
    saved = tmp_path / "jobs/job-x/input/x_source.json"
    assert saved.is_file() and json.loads(saved.read_text(encoding="utf-8"))["description"].startswith("ANC SG Fikile Mbalula")
    assert job.media["x_source"]["status_id"] == "2125522879982829740" and job.objects["source_url"] == URL and job.providers["source_download"] == "yt-dlp"
    context = youtube_captions.video_context(tmp_path / "jobs/job-x")                  # the readers that look for a video's own words find the post's
    assert context["channel"] == "Newzroom Afrika" and "Mbalula" in context["description"]
