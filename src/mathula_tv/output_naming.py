"""Create user-facing Mathula TV deliverables named with the job ID."""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

OUTPUT_SCHEMA_VERSION = "mathula-job-id-output-v1"
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_LEGACY_TIKTOK_HASHTAGS = {
    "#bantufyzulu",
    "#isizulu",
    "#mathulatv",
    "#mzansi",
    "#ngesizulu",
    "#southafrica",
}
_DISCOVERY_ACRONYM_HASHTAGS = {
    "ekurhulenimetropolitanpolicedepartment": "#EMPD",
    "independentpoliceinvestigativedirectorate": "#IPID",
    "investigatingdirectorateagainstcorruption": "#IDAC",
    "nationalprosecutingauthority": "#NPA",
    "politicalkillingstaskteam": "#PKTT",
    "southafricanpoliceservice": "#SAPS",
}
_DEFAULT_TIKTOK_ACCOUNTS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "tiktok_accounts.json"
)

class OutputNamingError(ValueError):
    """Raised when finalized outputs cannot be created safely."""

@dataclass(frozen=True)
class SEOOutputContext:
    title: str
    filename_stem: str
    language_code: str
    seo_path: Path
    seo: Mapping[str, Any]

@dataclass(frozen=True)
class NamedOutputArtifacts:
    final_video: Path
    seo_files: Mapping[str, Path]
    title: str
    filename_stem: str
    seo_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "seo_title": self.title,
            "job_id": self.filename_stem,
            "filename_stem": self.filename_stem,
            "final_video": str(self.final_video),
            "seo_files": {key: str(path) for key, path in sorted(self.seo_files.items())},
            "seo_path": str(self.seo_path),
        }

def language_code(locale: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "", str(locale).split("-", 1)[0].lower())
    return value or "target"

def safe_job_id(value: str) -> str:
    job_id = str(value).strip()
    if not _JOB_ID.fullmatch(job_id):
        raise OutputNamingError(f"Unsafe or invalid job ID: {job_id!r}")
    return job_id

def safe_title_stem(title: str, *, max_utf8_bytes: int = 180) -> str:
    """Backward-compatible alias retained for older imports; now validates a job ID."""
    del max_utf8_bytes
    return safe_job_id(title)

def load_seo_output_context(job_root: Path, target_language: str) -> SEOOutputContext:
    job_root = Path(job_root)
    job_id = safe_job_id(job_root.name)
    code = language_code(target_language)
    tiktok_path = job_root / "translation" / f"tiktok_{code}.json"
    legacy_path = job_root / "translation" / f"youtube_{code}.json"
    seo_path = tiktok_path if tiktok_path.is_file() else legacy_path
    if not seo_path.is_file():
        from .localized_seo import build_deferred_localized_seo

        seo_path = tiktok_path
        seo_path.parent.mkdir(parents=True, exist_ok=True)
        seo_path.write_text(
            json.dumps(
                build_deferred_localized_seo(
                    job_id=job_id,
                    target_language=target_language,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    try:
        seo = json.loads(seo_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutputNamingError(f"Invalid SEO metadata: {seo_path}") from exc
    if not isinstance(seo, Mapping):
        raise OutputNamingError("SEO metadata must be a JSON object")
    title = str(
        seo.get("cover_hook")
        or seo.get("title")
        or seo.get("youtube_title")
        or seo.get("primary_title")
        or ""
    ).strip()
    if not title:
        raise OutputNamingError("SEO metadata has no approved title")
    return SEOOutputContext(
        title=title,
        filename_stem=job_id,
        language_code=code,
        seo_path=seo_path,
        seo=seo,
    )

def publish_title_named_outputs(job_root: Path, canonical_video: Path, *, context: SEOOutputContext) -> NamedOutputArtifacts:
    """Publish the final hook-panel video and individual SEO files in output/."""
    job_root = Path(job_root)
    job_id = safe_job_id(job_root.name)
    canonical_video = Path(canonical_video)
    if not canonical_video.is_file() or canonical_video.stat().st_size <= 0:
        raise FileNotFoundError(canonical_video)
    output_root = job_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    final_video = output_root / f"final_dubbed_{job_id}.mp4"
    _atomic_copy(canonical_video, final_video)
    seo_files: dict[str, Path] = {}
    for key, (filename, payload) in _seo_file_payloads(context, job_id).items():
        destination = output_root / filename
        _atomic_write_bytes(destination, payload)
        seo_files[key] = destination
    for obsolete in output_root.glob("*SEO Package.tar.gz"):
        obsolete.unlink(missing_ok=True)
    return NamedOutputArtifacts(final_video, seo_files, context.title, job_id, context.seo_path)


def publish_dubbed_master_outputs(
    job_root: Path,
    canonical_video: Path,
    *,
    context: SEOOutputContext,
) -> NamedOutputArtifacts:
    """Publish an explicitly named clean dubbed master plus SEO sidecars.

    The master intentionally does not use the ``final_dubbed`` filename because
    the TikTok hook/title-panel render is the publication final.
    """
    job_root = Path(job_root)
    job_id = safe_job_id(job_root.name)
    canonical_video = Path(canonical_video)
    if not canonical_video.is_file() or canonical_video.stat().st_size <= 0:
        raise FileNotFoundError(canonical_video)
    output_root = job_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    master_video = output_root / f"dubbed_master_{job_id}.mp4"
    _atomic_copy(canonical_video, master_video)
    seo_files: dict[str, Path] = {}
    for key, (filename, payload) in _seo_file_payloads(context, job_id).items():
        destination = output_root / filename
        _atomic_write_bytes(destination, payload)
        seo_files[key] = destination
    for obsolete in output_root.glob("*SEO Package.tar.gz"):
        obsolete.unlink(missing_ok=True)
    return NamedOutputArtifacts(
        master_video,
        seo_files,
        context.title,
        job_id,
        context.seo_path,
    )


def publish_job_id_transcript_alias(job_root: Path, job_id: str | None = None) -> Path:
    """Copy the canonical transcript to its manual-handoff job-ID filename."""
    job_root = Path(job_root)
    resolved_job_id = safe_job_id(job_id or job_root.name)
    canonical = job_root / "analysis" / "transcript_en.json"
    if not canonical.is_file() or canonical.stat().st_size <= 0:
        raise FileNotFoundError(canonical)
    alias = canonical.with_name(f"transcript_en_{resolved_job_id}.json")
    _atomic_copy(canonical, alias)
    return alias

def _seo_file_payloads(context: SEOOutputContext, job_id: str) -> dict[str, tuple[str, bytes]]:
    seo = context.seo
    code = context.language_code
    suffix = f"_{job_id}"
    return {
        "source_json": (f"tiktok_seo_{code}{suffix}.json", context.seo_path.read_bytes()),
        "tiktok_caption": (f"tiktok_caption_{code}{suffix}.txt", _text_bytes(_tiktok_caption_text(seo, code))),
        "tiktok_hashtags": (f"tiktok_hashtags_{code}{suffix}.txt", _text_bytes(_tiktok_hashtags_text(seo, code))),
        "tiktok_cover": (f"tiktok_cover_{code}{suffix}.txt", _text_bytes(_thumbnail_text(seo))),
        "tiktok_search_keywords": (
            f"tiktok_search_keywords_{code}{suffix}.txt",
            _text_bytes(_search_keywords_text(seo)),
        ),
    }

def _tags_text(seo: Mapping[str, Any]) -> str:
    explicit = str(seo.get("tags_csv") or "").strip()
    if explicit: return explicit
    tags = seo.get("tags")
    if isinstance(tags, list): return ", ".join(str(x).strip() for x in tags if str(x).strip())
    return str(tags or "").strip()

def _thumbnail_text(seo: Mapping[str, Any]) -> str:
    cover_hook = str(seo.get("cover_hook") or "").strip()
    if cover_hook:
        return cover_hook
    hooks = "\n".join(
        str(value).strip()
        for value in (seo.get("thumbnail_hook"), seo.get("thumbnail_subhook"))
        if str(value or "").strip()
    )
    if hooks:
        return hooks
    return str(
        seo.get("thumbnail")
        or seo.get("youtube_thumbnail")
        or ""
    ).strip()


def _search_keywords_text(seo: Mapping[str, Any]) -> str:
    values = seo.get("search_keywords")
    if isinstance(values, list):
        return ", ".join(
            str(value).strip() for value in values if str(value).strip()
        )
    return ""

def _chapters_text(seo: Mapping[str, Any]) -> str:
    chapters = seo.get("chapters")
    if not isinstance(chapters, list): return ""
    lines=[]
    for item in chapters:
        if isinstance(item, Mapping):
            timestamp=str(item.get("timestamp") or "").strip(); title=str(item.get("title") or "").strip()
            if timestamp and title: lines.append(f"{timestamp} {title}")
    return "\n".join(lines)

def _tiktok_value(seo: Mapping[str, Any], key: str) -> str:
    value = seo.get("tiktok")
    if isinstance(value, Mapping):
        nested = str(value.get(key) or "").strip()
        if nested:
            return nested
    return str(seo.get(f"tiktok_{key}") or "").strip()

def _tiktok_hashtags_text(seo: Mapping[str, Any], code: str = "zu") -> str:
    return " ".join(_tiktok_hashtag_values(seo, code))


def _tiktok_caption_text(seo: Mapping[str, Any], code: str = "zu") -> str:
    caption = " ".join(
        token
        for token in _tiktok_value(seo, "caption").split()
        if not token.startswith("#")
    )
    hashtags = _tiktok_hashtag_values(seo, code)
    if caption and hashtags:
        return f"{caption}\n\n{' '.join(hashtags)}"
    return caption or " ".join(hashtags)


def _tiktok_hashtag_values(seo: Mapping[str, Any], code: str = "zu") -> list[str]:
    profile = _tiktok_profile(seo, code)
    explicit_candidates: list[str] = []
    explicit = str(seo.get("tiktok_hashtags_text") or "").strip()
    nested = _tiktok_value(seo, "hashtags_text")
    if explicit or nested:
        explicit_candidates.extend((explicit or nested).split())
    else:
        values = seo.get("tiktok_hashtags")
        if not isinstance(values, list):
            tiktok = seo.get("tiktok")
            values = tiktok.get("hashtags") if isinstance(tiktok, Mapping) else None
        if isinstance(values, list):
            explicit_candidates.extend(str(value).strip() for value in values)

    relevance_candidates = list(explicit_candidates)
    tags = seo.get("tags")
    if isinstance(tags, list):
        relevance_candidates.extend(_hashtag_from_seo_tag(value) for value in tags)
    search_keywords = seo.get("search_keywords")
    if isinstance(search_keywords, list):
        relevance_candidates.extend(
            _hashtag_from_seo_tag(value) for value in search_keywords
        )
    relevance_candidates.extend(profile.get("discovery_hashtags") or [])

    result: list[str] = []
    seen: set[str] = set()
    max_hashtags = min(5, max(1, int(profile.get("max_hashtags") or 5)))

    def add(value: Any) -> None:
        if len(result) >= max_hashtags:
            return
        hashtag = _canonical_discovery_hashtag(value)
        if not hashtag:
            return
        key = hashtag.casefold()
        if key in _LEGACY_TIKTOK_HASHTAGS or key in seen:
            return
        seen.add(key)
        result.append(hashtag)

    retired_brand_values = (
        profile.get("brand_hashtag"),
        profile.get("value_hashtag"),
        _hashtag_from_seo_tag(profile.get("account_name")),
    )
    for retired_brand in retired_brand_values:
        normalized_brand = _canonical_discovery_hashtag(retired_brand)
        if normalized_brand:
            seen.add(normalized_brand.casefold())
    add(profile.get("community_hashtag"))
    for value in relevance_candidates:
        add(value)
    return result


def _tiktok_profile(seo: Mapping[str, Any], code: str) -> dict[str, Any]:
    configured_path = os.getenv("MATHULA_TIKTOK_ACCOUNTS_PATH")
    path = Path(configured_path).expanduser() if configured_path else _DEFAULT_TIKTOK_ACCOUNTS_PATH
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutputNamingError(f"Invalid TikTok account registry: {path}") from exc
    accounts = registry.get("accounts")
    if not isinstance(accounts, Mapping):
        raise OutputNamingError("TikTok account registry has no accounts mapping")
    configured = accounts.get(code)
    if not isinstance(configured, Mapping):
        raise OutputNamingError(
            f"No TikTok account configured for target language code: {code}"
        )
    profile = dict(configured)
    profile["shared_hashtags"] = registry.get("shared_hashtags") or []
    profile["max_hashtags"] = registry.get("max_hashtags") or 5

    override = seo.get("tiktok_account")
    if isinstance(override, Mapping):
        profile.update(override)
    nested = seo.get("tiktok")
    if isinstance(nested, Mapping):
        for key in (
            "account_name",
            "brand_hashtag",
            "community_hashtag",
            "value_hashtag",
            "discovery_hashtags",
            "max_hashtags",
        ):
            if nested.get(key):
                profile[key] = nested[key]
    if not str(profile.get("community_hashtag") or "").strip():
        raise OutputNamingError(f"TikTok account {code!r} has no community_hashtag")
    return profile


def _canonical_discovery_hashtag(value: Any) -> str:
    hashtag = str(value or "").strip()
    if not hashtag:
        return ""
    if not hashtag.startswith("#"):
        hashtag = f"#{hashtag}"
    compact_key = re.sub(r"[^a-z0-9]+", "", hashtag.casefold())
    return _DISCOVERY_ACRONYM_HASHTAGS.get(compact_key, hashtag)


def _hashtag_from_seo_tag(value: Any) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(value or ""))
    hashtag = "#" + "".join(
        word if any(character.isupper() for character in word[1:]) else word[:1].upper() + word[1:]
        for word in words
    ) if words else ""
    return _canonical_discovery_hashtag(hashtag)

def _text_bytes(value: Any) -> bytes:
    return f"{str(value or '').strip()}\n".encode("utf-8")

def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload); os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)

def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        if source.resolve()==destination.resolve(): return
    except FileNotFoundError: pass
    temporary=destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source,temporary); os.replace(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)
