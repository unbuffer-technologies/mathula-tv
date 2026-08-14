from __future__ import annotations

from pathlib import Path

import pytest

from mathula_tv import youtube_ingestion as module


def test_recognizes_supported_youtube_urls() -> None:
    assert module.is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert module.is_youtube_url("https://youtu.be/abc123")
    assert module.is_youtube_url("https://www.youtube.com/shorts/abc123")
    assert not module.is_youtube_url("https://example.com/watch?v=abc123")
    assert not module.is_youtube_url("relative/video.mp4")


def test_rejects_video_above_configured_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "_read_metadata",
        lambda _url: {
            "id": "abc123",
            "title": "Long video",
            "duration": 601,
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
        },
    )

    with pytest.raises(module.YouTubeIngestionError, match="duration limit"):
        with module.download_youtube_video(
            "https://youtu.be/abc123",
            max_duration_seconds=600,
        ):
            raise AssertionError("download should not start")


def test_download_context_returns_manifest_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_read_metadata",
        lambda _url: {
            "id": "56Jli3Rr-yU",
            "title": "Madlanga Commission testimony",
            "duration": 560.618,
            "webpage_url": "https://www.youtube.com/watch?v=56Jli3Rr-yU",
            "channel": "News channel",
            "channel_id": "channel-1",
            "extractor_key": "Youtube",
        },
    )

    def fake_download(_url: str, directory: Path) -> Path:
        path = directory / "Madlanga Commission testimony [56Jli3Rr-yU].mp4"
        path.write_bytes(b"fake-media")
        return path

    monkeypatch.setattr(module, "_download", fake_download)

    with module.download_youtube_video(
        "https://youtu.be/56Jli3Rr-yU",
        max_duration_seconds=600,
    ) as download:
        temporary_path = download.path
        assert temporary_path.is_file()
        manifest = download.manifest()
        assert manifest["video_id"] == "56Jli3Rr-yU"
        assert manifest["channel"] == "News channel"
        assert manifest["playlist_allowed"] is False
        assert manifest["maximum_height"] == 1080

    assert not temporary_path.exists()


def test_rejects_live_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "_read_metadata",
        lambda _url: {
            "id": "live123",
            "title": "Live hearing",
            "duration": 300,
            "webpage_url": "https://www.youtube.com/watch?v=live123",
            "is_live": True,
        },
    )

    with pytest.raises(module.YouTubeIngestionError, match="Live or upcoming"):
        with module.download_youtube_video("https://youtu.be/live123"):
            raise AssertionError("live stream should not be downloaded")


def test_record_submission_refreshes_gcs_manifest(tmp_path: Path) -> None:
    class FakeJob:
        job_id = "job-1"
        media: dict = {}
        objects: dict = {"manifest": "gs://bucket/mathula-tv/jobs/job-1/input/manifest.json"}
        providers: dict = {}

        def to_dict(self) -> dict:
            return {
                "job_id": self.job_id,
                "media": self.media,
                "objects": self.objects,
                "providers": self.providers,
            }

    class FakeJobs:
        def job_dir(self, _job_id: str) -> Path:
            return tmp_path / "jobs/job-1"

        def save(self, job: FakeJob) -> None:
            module._atomic_write_json(self.job_dir(job.job_id) / "manifest.json", job.to_dict())

    class FakeGCS:
        uploads: list[tuple[str, str, Path]] = []

        def upload(self, job_id: str, relative: str, source: Path) -> str:
            self.uploads.append((job_id, relative, source))
            return f"gs://bucket/mathula-tv/jobs/{job_id}/{relative}"

    class FakeOrchestrator:
        jobs = FakeJobs()
        gcs = FakeGCS()

    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"video")
    download = module.YouTubeDownload(
        path=media_path,
        requested_url="https://youtu.be/abc123",
        webpage_url="https://www.youtube.com/watch?v=abc123",
        video_id="abc123",
        title="Example",
        channel="Channel",
        channel_id="channel-id",
        duration=60.0,
        extractor="Youtube",
        downloaded_at="2026-07-25T00:00:00+00:00",
    )
    job = FakeJob()

    module.record_youtube_submission(FakeOrchestrator(), job, download)

    source_manifest = tmp_path / "jobs/job-1/input/youtube_source.json"
    assert source_manifest.is_file()
    assert job.providers["source_download"] == "yt-dlp"
    assert job.objects["youtube_source"].endswith("/input/youtube_source.json")
    uploaded_relatives = [relative for _, relative, _ in FakeOrchestrator.gcs.uploads]
    assert "input/youtube_source.json" in uploaded_relatives
    assert "input/manifest.json" in uploaded_relatives
