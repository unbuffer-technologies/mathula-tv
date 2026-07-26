from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlsplit


YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }
)

# Prefer an ordinary H.264/AAC MP4 so the existing Mathula TV media path can
# ingest the download without adding a second transcoding policy here.
YOUTUBE_FORMAT_SELECTOR = (
    "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
    "best[height<=1080][ext=mp4][vcodec^=avc1]/"
    "best[height<=1080][ext=mp4]"
)


class YouTubeIngestionError(ValueError):
    """Raised when a YouTube URL cannot be submitted safely."""


@dataclass(frozen=True)
class YouTubeDownload:
    path: Path
    requested_url: str
    webpage_url: str
    video_id: str
    title: str
    channel: str | None
    channel_id: str | None
    duration: float
    extractor: str | None
    downloaded_at: str

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "youtube-source-v1",
            "requested_url": self.requested_url,
            "webpage_url": self.webpage_url,
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "channel_id": self.channel_id,
            "duration": self.duration,
            "extractor": self.extractor,
            "downloaded_filename": self.path.name,
            "downloaded_at": self.downloaded_at,
            "playlist_allowed": False,
            "maximum_height": 1080,
        }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_youtube_submission(
    orchestrator: Any,
    job: Any,
    download: YouTubeDownload,
) -> None:
    """Persist YouTube provenance locally and refresh the durable GCS manifest."""

    record = download.manifest()
    job_dir = Path(orchestrator.jobs.job_dir(job.job_id))
    source_manifest_path = job_dir / "input/youtube_source.json"
    _atomic_write_json(source_manifest_path, record)

    job.media["youtube_source"] = record
    job.objects["source_url"] = download.webpage_url
    job.objects["youtube_source_local"] = str(source_manifest_path)
    job.providers["source_download"] = "yt-dlp"

    gcs = getattr(orchestrator, "gcs", None)
    if gcs is not None:
        job.objects["youtube_source"] = gcs.upload(
            job.job_id,
            "input/youtube_source.json",
            source_manifest_path,
        )

    orchestrator.jobs.save(job)

    if gcs is None:
        return

    local_manifest = job_dir / "manifest.json"
    if not local_manifest.is_file():
        raise YouTubeIngestionError(
            f"Cannot refresh the GCS job manifest because it is missing: {local_manifest}"
        )
    manifest_uri = gcs.upload(job.job_id, "input/manifest.json", local_manifest)
    if job.objects.get("manifest") != manifest_uri:
        job.objects["manifest"] = manifest_uri
        orchestrator.jobs.save(job)
        gcs.upload(job.job_id, "input/manifest.json", local_manifest)


def is_youtube_url(value: str | Path) -> bool:
    text = str(value).strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return False
    if parts.scheme.lower() not in {"http", "https"}:
        return False
    host = (parts.hostname or "").lower().rstrip(".")
    return host in YOUTUBE_HOSTS


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _yt_dlp_command() -> list[str]:
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    raise YouTubeIngestionError(
        "YouTube submission requires yt-dlp. Install it inside the active Mathula TV "
        "virtual environment with: python -m pip install -U yt-dlp"
    )


def _run(command: Sequence[str], *, purpose: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown yt-dlp error").strip()
        raise YouTubeIngestionError(f"YouTube {purpose} failed: {detail[-2000:]}")
    return result


def _read_metadata(url: str) -> dict[str, Any]:
    result = _run(
        [
            *_yt_dlp_command(),
            "--no-playlist",
            "--no-warnings",
            "--skip-download",
            "--dump-single-json",
            url,
        ],
        purpose="metadata lookup",
    )
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise YouTubeIngestionError("yt-dlp returned invalid metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise YouTubeIngestionError("yt-dlp returned an invalid metadata object")
    return metadata


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    max_duration_seconds: float | None,
) -> tuple[str, str, float, str]:
    live_status = str(metadata.get("live_status") or "").lower()
    if metadata.get("is_live") is True or live_status in {"is_live", "is_upcoming"}:
        raise YouTubeIngestionError(
            "Live or upcoming YouTube streams cannot be submitted as finite Mathula TV jobs"
        )

    video_id = str(metadata.get("id") or "").strip()
    title = str(metadata.get("title") or "").strip()
    webpage_url = str(metadata.get("webpage_url") or metadata.get("original_url") or "").strip()
    try:
        duration = float(metadata.get("duration") or 0)
    except (TypeError, ValueError) as exc:
        raise YouTubeIngestionError("YouTube video duration is invalid") from exc

    if not video_id:
        raise YouTubeIngestionError("YouTube metadata is missing the video ID")
    if not title:
        raise YouTubeIngestionError("YouTube metadata is missing the title")
    if not webpage_url or not is_youtube_url(webpage_url):
        raise YouTubeIngestionError("YouTube metadata is missing a valid canonical URL")
    if duration <= 0:
        raise YouTubeIngestionError("YouTube video has no finite duration")
    if max_duration_seconds is not None and duration > max_duration_seconds:
        raise YouTubeIngestionError(
            "YouTube video exceeds the configured source-duration limit: "
            f"video={duration / 60:.2f} minutes, "
            f"limit={max_duration_seconds / 60:.2f} minutes"
        )
    return video_id, title, duration, webpage_url


def _download(url: str, directory: Path) -> Path:
    output_template = str(directory / "%(title).180B [%(id)s].%(ext)s")
    result = _run(
        [
            *_yt_dlp_command(),
            "--no-playlist",
            "--no-warnings",
            "--format",
            YOUTUBE_FORMAT_SELECTOR,
            "--merge-output-format",
            "mp4",
            "--output",
            output_template,
            "--print",
            "after_move:filepath",
            url,
        ],
        purpose="download",
    )
    candidates = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not candidates:
        raise YouTubeIngestionError("yt-dlp did not report the downloaded media path")
    downloaded = candidates[-1].expanduser().resolve()
    directory = directory.resolve()
    try:
        downloaded.relative_to(directory)
    except ValueError as exc:
        raise YouTubeIngestionError("yt-dlp reported a path outside its temporary directory") from exc
    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        raise YouTubeIngestionError(f"Downloaded YouTube media is missing or empty: {downloaded}")
    if downloaded.suffix.lower() not in {".mp4", ".webm"}:
        raise YouTubeIngestionError(
            f"yt-dlp produced unsupported media: {downloaded.suffix or '<none>'}"
        )
    return downloaded


@contextmanager
def download_youtube_video(
    url: str,
    *,
    max_duration_seconds: float | None = None,
) -> Iterator[YouTubeDownload]:
    requested_url = str(url).strip()
    if not is_youtube_url(requested_url):
        raise YouTubeIngestionError(f"Not a supported YouTube URL: {requested_url}")

    metadata = _read_metadata(requested_url)
    video_id, title, duration, webpage_url = _validate_metadata(
        metadata,
        max_duration_seconds=max_duration_seconds,
    )

    with tempfile.TemporaryDirectory(prefix="mathula-youtube-") as temporary_dir:
        path = _download(webpage_url, Path(temporary_dir))
        yield YouTubeDownload(
            path=path,
            requested_url=requested_url,
            webpage_url=webpage_url,
            video_id=video_id,
            title=title,
            channel=_optional_text(metadata.get("channel") or metadata.get("uploader")),
            channel_id=_optional_text(
                metadata.get("channel_id") or metadata.get("uploader_id")
            ),
            duration=duration,
            extractor=_optional_text(
                metadata.get("extractor_key") or metadata.get("extractor")
            ),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
