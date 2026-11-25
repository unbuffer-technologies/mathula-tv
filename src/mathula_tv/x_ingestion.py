"""Submit the video of an X (Twitter) post, the way a YouTube link is submitted: yt-dlp downloads it into a directory of its own, the job is made from the file, and where it came from is kept with the job
(``input/x_source.json``, ``job.media["x_source"]``).  The post's own text is the nearest thing X has to a description: it often names the speaker, the event and the place, so it is kept in the same shape as a
YouTube description and read by the same readers (youtube_captions.video_context, the name research).  X has no caption tracks, so there are none to fetch.

Only a link to one post is taken (``x.com/<user>/status/<id>``, also on twitter.com and the mobile hosts); a profile, a search or a list is not a video.  A post with several videos gives its first.
Some posts (sensitive media, an account that hides from guests) need a signed-in session: ``MATHULA_X_COOKIES_FILE`` (or ``x_cookies_file`` in the settings) names a Netscape cookies file for yt-dlp.
"""
from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from .youtube_ingestion import (
    YouTubeIngestionError,
    _cookies_args,
    _optional_text,
    _read_metadata,
    _resolve_cookies_file,
    _resolve_downloaded_media,
    _run_live,
    _yt_dlp_command,
    record_source_submission,
    upload_text,
)

X_HOSTS = frozenset({"x.com", "www.x.com", "mobile.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"})
_STATUS_PATH = re.compile(r"^/(?:[A-Za-z0-9_]{1,15}|i(?:/web)?)/status(?:es)?/(\d{5,25})(?:/.*)?$")

# The progressive MP4 (video and audio in one file) at up to 1080p first, then separate video and audio streams to be joined, then whatever is best: the existing media path ingests an
# ordinary H.264/AAC MP4 without a second transcoding policy.
X_FORMAT_SELECTOR = (
    "best[height<=1080][ext=mp4][protocol^=http]/"
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "best[height<=1080][ext=mp4]/"
    "best[height<=1080]/best"
)


class XIngestionError(YouTubeIngestionError):
    """Raised when an X link cannot be submitted safely."""


@dataclass(frozen=True)
class XDownload:
    path: Path
    requested_url: str
    webpage_url: str
    status_id: str
    title: str
    author: str | None            # the account's display name
    author_handle: str | None     # @handle, without the @
    duration: float
    extractor: str | None
    downloaded_at: str
    text: str | None = None       # the post's own text
    tags: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            **upload_text(self.text, self.tags),
            "schema_version": "x-source-v1",
            "requested_url": self.requested_url,
            "webpage_url": self.webpage_url,
            "status_id": self.status_id,
            "video_id": self.status_id,
            "title": self.title,
            "channel": self.author,
            "uploader": self.author_handle,
            "duration": self.duration,
            "extractor": self.extractor,
            "downloaded_filename": self.path.name,
            "downloaded_at": self.downloaded_at,
            "playlist_allowed": False,
            "maximum_height": 1080,
        }


def status_id(value: str | Path) -> str | None:
    """The numeric id of the post a link points to, or None when it is not a link to one post of X."""
    text = str(value).strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"}:
        return None
    if (parts.hostname or "").lower().rstrip(".") not in X_HOSTS:
        return None
    match = _STATUS_PATH.match(parts.path or "")
    return match.group(1) if match else None


def is_x_url(value: str | Path) -> bool:
    return status_id(value) is not None


def _resolve_x_cookies(cookies_file: str | Path | None) -> Path | None:
    return _resolve_cookies_file(cookies_file or os.environ.get("MATHULA_X_COOKIES_FILE") or None)


def _first_video(metadata: dict[str, Any]) -> dict[str, Any]:
    """A post with several videos comes back from yt-dlp as a playlist: the first one stands for the post, with the post's own fields filled in from the playlist."""
    entries = [e for e in metadata.get("entries") or [] if isinstance(e, dict)]
    if not entries:
        return metadata
    merged = {k: v for k, v in metadata.items() if k != "entries"}
    merged.update({k: v for k, v in entries[0].items() if v not in (None, "")})
    return merged


def _validate_metadata(metadata: dict[str, Any], *, max_duration_seconds: float | None) -> tuple[str, str, float, str]:
    live_status = str(metadata.get("live_status") or "").lower()
    if metadata.get("is_live") is True or live_status in {"is_live", "is_upcoming"}:
        raise XIngestionError("A live X broadcast cannot be submitted as a finite Mathula TV job")
    post_id = str(metadata.get("id") or "").strip()
    title = str(metadata.get("title") or "").strip()
    webpage_url = str(metadata.get("webpage_url") or metadata.get("original_url") or "").strip()
    try:
        duration = float(metadata.get("duration") or 0)
    except (TypeError, ValueError) as exc:
        raise XIngestionError("The X video's duration is invalid") from exc
    if not post_id:
        raise XIngestionError("The X post's metadata is missing its id")
    if not title:
        title = f"X post {post_id}"
    if not webpage_url or not is_x_url(webpage_url):
        raise XIngestionError("The X post's metadata is missing a valid canonical link")
    if duration <= 0:
        raise XIngestionError("This X post has no video of a finite length (a post with only text or pictures, or one that needs a signed-in session: see MATHULA_X_COOKIES_FILE)")
    if max_duration_seconds is not None and duration > max_duration_seconds:
        raise XIngestionError(f"The X video exceeds the configured source-duration limit: video={duration / 60:.2f} minutes, limit={max_duration_seconds / 60:.2f} minutes")
    return post_id, title, duration, webpage_url


def _download(url: str, directory: Path, *, cookies_file: Path | None = None) -> Path:
    # The temporary file name does not depend on the post's text (Windows sanitising, path length): source-<id>.<ext>.
    _run_live(
        [
            *_yt_dlp_command("X"),
            "--playlist-items", "1",
            "--no-warnings",
            "--windows-filenames",
            "--newline",
            "--progress",
            "--format", X_FORMAT_SELECTOR,
            "--merge-output-format", "mp4",
            "--output", str(directory / "source-%(id)s.%(ext)s"),
            *_cookies_args(cookies_file),
            url,
        ],
        purpose="download",
        site="X",
    )
    return _resolve_downloaded_media(directory, (), site="X")


@contextmanager
def download_x_video(url: str, *, max_duration_seconds: float | None = None, cookies_file: str | Path | None = None) -> Iterator[XDownload]:
    requested_url = str(url).strip()
    if not is_x_url(requested_url):
        raise XIngestionError(f"Not a link to one X post: {requested_url}")
    cookies = _resolve_x_cookies(cookies_file)
    metadata = _first_video(_read_metadata(requested_url, cookies_file=cookies, site="X", extra_args=("--playlist-items", "1")))
    post_id, title, duration, webpage_url = _validate_metadata(metadata, max_duration_seconds=max_duration_seconds)
    with tempfile.TemporaryDirectory(prefix="mathula-x-") as temporary_dir:
        path = _download(webpage_url, Path(temporary_dir), cookies_file=cookies)
        yield XDownload(
            path=path,
            requested_url=requested_url,
            webpage_url=webpage_url,
            status_id=post_id,
            title=title,
            author=_optional_text(metadata.get("uploader") or metadata.get("channel")),
            author_handle=_optional_text(metadata.get("uploader_id") or metadata.get("channel_id")),
            duration=duration,
            extractor=_optional_text(metadata.get("extractor_key") or metadata.get("extractor")),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            text=_optional_text(metadata.get("description")),
            tags=tuple(str(t) for t in metadata.get("tags") or ()),
        )


def record_x_submission(orchestrator: Any, job: Any, download: XDownload) -> None:
    """Save where the video came from with the job (``input/x_source.json``) and refresh the durable manifest, as for a YouTube source."""
    record_source_submission(orchestrator, job, download.manifest(), file_name="x_source.json", media_key="x_source", webpage_url=download.webpage_url)
