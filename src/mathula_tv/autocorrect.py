"""Production autocorrect for Mathula TV transcripts.

Design guarantees
-----------------
* Azure raw artifacts are never modified.
* The reconciled source transcript is snapshotted before autocorrect.
* The effective transcript is always rendered from that immutable snapshot.
* All state changes are transactional and recorded in PostgreSQL.
* Every apply/revert/redo is reversible.
* AI may rank supplied candidates only; it cannot rewrite the transcript.
* New AI suggestions never auto-apply.
* Only exact, previously learned user/project rules may auto-apply.
* Global learning is opt-in and never auto-promotes a term.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import unicodedata
import atexit
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - exercised by installation checks
    psycopg = None
    sql = None
    dict_row = None
    ConnectionPool = None

AUTOCORRECT_SCHEMA = "mathula-autocorrect-state-v1"
REPORT_SCHEMA = "mathula-autocorrect-report-v1"
EVENT_EXPORT_SCHEMA = "mathula-autocorrect-events-v1"
PHRASE_EXPORT_SCHEMA = "mathula-adaptive-stt-phrases-v1"
AI_REQUEST_SCHEMA = "mathula-autocorrect-ai-request-v1"
AI_RESPONSE_SCHEMA = "mathula-autocorrect-ai-response-v1"

ACTIVE_STATUSES = {"auto_applied", "confirmed", "edited"}
REVIEW_STATUSES = {"suggested", "needs_review"}
FINAL_STATUSES = ACTIVE_STATUSES | {"reverted", "dismissed", "suppressed"}

TEXT_FIELDS = ("source_text", "text", "display_text", "display", "lexical")
UNSAFE_GENERIC_ALIASES = {"cat", "ci", "mk", "khan", "johnson", "jacob"}
VALID_SCOPES = {"job", "user", "project"}


class AutocorrectError(RuntimeError):
    """Raised when an autocorrect operation would be unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


TERM_CATEGORIES = {
    "slang",
    "phrase",
    "brand",
    "place",
    "person",
    "organisation",
    "other",
}


def clean_user_term(value: str) -> str:
    """Normalise free-text corrections without destroying local spelling."""

    cleaned = unicodedata.normalize("NFKC", str(value))
    cleaned = "".join(
        character
        for character in cleaned
        if character in {"\n", "\t"} or not unicodedata.category(character).startswith("C")
    )
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        raise AutocorrectError("A new term cannot be empty")
    if len(cleaned) > 160:
        raise AutocorrectError("A new term cannot exceed 160 characters")
    if len(cleaned.split()) > 20:
        raise AutocorrectError("A new term cannot exceed 20 words")
    return cleaned


def validate_term_category(value: str) -> str:
    category = str(value or "other").strip().casefold()
    if category not in TERM_CATEGORIES:
        raise AutocorrectError(
            f"Unsupported term category {value!r}; expected one of "
            f"{sorted(TERM_CATEGORIES)}"
        )
    return category


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(
        "\0".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:24]
    return f"{prefix}_{digest}"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AutocorrectError(f"Expected a JSON object: {path}")
    return value


def _safe_name(value: str, label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise AutocorrectError(f"{label} is empty or unsafe")
    return cleaned[:160]


def _number_to_ms(value: Any, field: str) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    lowered = field.casefold()
    if "millisecond" in lowered or lowered.endswith("_ms"):
        return int(round(number))
    return int(round(number * 1000))


def _timing(record: dict[str, Any]) -> tuple[int | None, int | None]:
    start = None
    for field in (
        "start_ms",
        "offset_ms",
        "offsetMilliseconds",
        "start",
        "offset",
    ):
        if field in record:
            start = _number_to_ms(record[field], field)
            break

    end = None
    for field in ("end_ms", "end"):
        if field in record:
            end = _number_to_ms(record[field], field)
            break

    if end is None and start is not None:
        for field in (
            "duration_ms",
            "durationMilliseconds",
            "duration",
        ):
            if field in record:
                duration = _number_to_ms(record[field], field)
                if duration is not None:
                    end = start + duration
                break
    return start, end


def _confidence(record: dict[str, Any]) -> float | None:
    for field in ("confidence", "score", "probability"):
        if field not in record:
            continue
        try:
            value = float(record[field])
        except (TypeError, ValueError):
            return None
        if value > 1:
            value /= 100
        return max(0.0, min(1.0, value))
    return None


def _text_field(record: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        if isinstance(record.get(field), str):
            return field
    raise AutocorrectError(
        f"Transcript item has no supported text field: {sorted(record)}"
    )


def _item_id(record: dict[str, Any], index: int) -> str:
    for field in ("segment_id", "turn_id", "id", "phrase_id"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return f"item_{index + 1:04d}"


@dataclass(frozen=True)
class TranscriptItem:
    item_id: str
    index: int
    text_field: str
    text: str
    start_ms: int | None
    end_ms: int | None
    speaker: str | None
    source_hash: str


@dataclass(frozen=True)
class TranscriptView:
    collection_key: str
    items: tuple[TranscriptItem, ...]


def transcript_view(document: dict[str, Any]) -> TranscriptView:
    if isinstance(document.get("segments"), list):
        key = "segments"
    elif isinstance(document.get("turns"), list):
        key = "turns"
    else:
        raise AutocorrectError(
            "Transcript must contain a segments or turns list"
        )

    raw_items = document[key]
    if not raw_items:
        raise AutocorrectError("Transcript collection is empty")

    items: list[TranscriptItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise AutocorrectError(
                f"Transcript item {index} is not an object"
            )
        identifier = _item_id(raw, index)
        if identifier in seen:
            raise AutocorrectError(
                f"Duplicate transcript item ID: {identifier}"
            )
        seen.add(identifier)
        field = _text_field(raw)
        start_ms, end_ms = _timing(raw)
        speaker = raw.get("speaker") or raw.get("speaker_id")
        items.append(
            TranscriptItem(
                item_id=identifier,
                index=index,
                text_field=field,
                text=str(raw[field]),
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=str(speaker) if speaker is not None else None,
                source_hash=sha256_json(raw),
            )
        )
    return TranscriptView(key, tuple(items))


def approximate_span_timing(
    item: TranscriptItem,
    char_start: int,
    char_end: int,
) -> tuple[int | None, int | None]:
    if (
        item.start_ms is None
        or item.end_ms is None
        or not item.text
    ):
        return item.start_ms, item.end_ms
    duration = max(0, item.end_ms - item.start_ms)
    return (
        item.start_ms
        + int(round(duration * char_start / len(item.text))),
        item.start_ms
        + int(round(duration * char_end / len(item.text))),
    )


class PgSession:
    """Small compatibility wrapper around a psycopg connection."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    @staticmethod
    def translate(query: str) -> str:
        translated = query.replace("?", "%s")
        stripped = translated.lstrip()
        if stripped.upper().startswith("INSERT OR IGNORE INTO"):
            prefix_length = len(translated) - len(stripped)
            translated = (
                translated[:prefix_length]
                + re.sub(
                    r"^INSERT\s+OR\s+IGNORE\s+INTO",
                    "INSERT INTO",
                    stripped,
                    count=1,
                    flags=re.IGNORECASE,
                )
            )
            translated = translated.rstrip().rstrip(";")
            translated += " ON CONFLICT DO NOTHING"
        return translated

    def execute(
        self,
        query: str,
        parameters: Sequence[Any] | Mapping[str, Any] = (),
    ) -> Any:
        return self.connection.execute(
            self.translate(query),
            parameters,
        )


class Store:
    """Transactional PostgreSQL storage in a dedicated schema."""

    CURRENT_SCHEMA_VERSION = 2

    def __init__(
        self,
        database_url: str,
        *,
        schema: str = "mathula_autocorrect",
        min_pool_size: int = 1,
        max_pool_size: int = 5,
    ) -> None:
        if psycopg is None or ConnectionPool is None:
            raise AutocorrectError(
                "PostgreSQL support is not installed. Run "
                'pip install "psycopg[binary,pool]>=3.2,<4".'
            )
        self.database_url = str(database_url or "").strip()
        if not self.database_url:
            raise AutocorrectError(
                "MATHULA_TV_DATABASE_URL or DATABASE_URL is required"
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", schema):
            raise AutocorrectError(f"Unsafe PostgreSQL schema name: {schema!r}")
        self.schema = schema
        self.pool = ConnectionPool(
            conninfo=self.database_url,
            min_size=min_pool_size,
            max_size=max_pool_size,
            timeout=30,
            open=True,
            kwargs={"row_factory": dict_row},
        )
        atexit.register(self.close)
        self._migrate()

    def close(self) -> None:
        pool = getattr(self, "pool", None)
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass

    def _set_search_path(self, connection: Any) -> None:
        connection.execute(
            sql.SQL("SET LOCAL search_path TO {}, public").format(
                sql.Identifier(self.schema)
            )
        )

    @contextmanager
    def connect(self) -> Iterator[PgSession]:
        with self.pool.connection() as connection:
            with connection.transaction():
                self._set_search_path(connection)
                yield PgSession(connection)

    @contextmanager
    def transaction(self) -> Iterator[PgSession]:
        with self.connect() as session:
            yield session

    def _migrate(self) -> None:
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"mathula-autocorrect:{self.schema}",),
                )
                connection.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema)
                    )
                )
                self._set_search_path(connection)
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE
                            CHECK (singleton),
                        version INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                row = connection.execute(
                    "SELECT version FROM schema_migrations WHERE singleton"
                ).fetchone()
                version = int(row["version"]) if row else 0
                if version > self.CURRENT_SCHEMA_VERSION:
                    raise AutocorrectError(
                        f"Database schema {version} is newer than supported "
                        f"{self.CURRENT_SCHEMA_VERSION}"
                    )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        project_id TEXT,
                        source_path TEXT NOT NULL,
                        snapshot_path TEXT NOT NULL,
                        source_sha256 TEXT NOT NULL,
                        effective_sha256 TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS corrections (
                        correction_id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(job_id)
                            ON DELETE CASCADE,
                        item_id TEXT NOT NULL,
                        item_index INTEGER NOT NULL,
                        text_field TEXT NOT NULL,
                        char_start INTEGER NOT NULL,
                        char_end INTEGER NOT NULL,
                        heard_text TEXT NOT NULL,
                        replacement_text TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        status TEXT NOT NULL,
                        entity_id TEXT,
                        start_ms BIGINT,
                        end_ms BIGINT,
                        reason TEXT NOT NULL,
                        candidates_json TEXT NOT NULL,
                        source_item_sha256 TEXT NOT NULL,
                        user_supplied BOOLEAN NOT NULL DEFAULT FALSE,
                        term_category TEXT,
                        language TEXT NOT NULL DEFAULT 'en-ZA',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS corrections_job_status
                    ON corrections(job_id, status)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        event_id BIGSERIAL PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        correction_id TEXT,
                        action TEXT NOT NULL,
                        actor_id TEXT,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS events_job
                    ON events(job_id, event_id)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vocabulary (
                        rule_id TEXT PRIMARY KEY,
                        scope TEXT NOT NULL,
                        owner_id TEXT NOT NULL,
                        heard_text TEXT NOT NULL,
                        heard_norm TEXT NOT NULL,
                        replacement_text TEXT NOT NULL,
                        replacement_norm TEXT NOT NULL,
                        entity_id TEXT,
                        confirmations INTEGER NOT NULL DEFAULT 0,
                        reversions INTEGER NOT NULL DEFAULT 0,
                        mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        term_category TEXT,
                        language TEXT NOT NULL DEFAULT 'en-ZA',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(
                            scope,
                            owner_id,
                            heard_norm,
                            replacement_norm
                        )
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS vocabulary_lookup
                    ON vocabulary(scope, owner_id, heard_norm, status)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS suppressions (
                        scope TEXT NOT NULL,
                        owner_id TEXT NOT NULL,
                        heard_norm TEXT NOT NULL,
                        replacement_norm TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(
                            scope,
                            owner_id,
                            heard_norm,
                            replacement_norm
                        )
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS platform_evidence (
                        evidence_id TEXT PRIMARY KEY,
                        heard_norm TEXT NOT NULL,
                        replacement_norm TEXT NOT NULL,
                        heard_text TEXT NOT NULL,
                        replacement_text TEXT NOT NULL,
                        user_hash TEXT NOT NULL,
                        job_hash TEXT NOT NULL,
                        submission_kind TEXT NOT NULL,
                        term_category TEXT NOT NULL,
                        language TEXT NOT NULL,
                        domain TEXT,
                        consent_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(
                            heard_norm,
                            replacement_norm,
                            user_hash,
                            job_hash
                        )
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public_terms (
                        term_id TEXT PRIMARY KEY,
                        heard_norm TEXT NOT NULL,
                        replacement_norm TEXT NOT NULL,
                        heard_text TEXT NOT NULL,
                        replacement_text TEXT NOT NULL,
                        term_category TEXT NOT NULL,
                        language TEXT NOT NULL,
                        domain TEXT,
                        independent_users INTEGER NOT NULL DEFAULT 0,
                        distinct_jobs INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'candidate',
                        privacy_review_required BOOLEAN NOT NULL DEFAULT TRUE,
                        approved_by TEXT,
                        approved_at TEXT,
                        rejected_by TEXT,
                        rejected_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(heard_norm, replacement_norm, language)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS public_terms_status
                    ON public_terms(status, language)
                    """
                )

                now = utc_now()
                if version == 0:
                    connection.execute(
                        """
                        INSERT INTO schema_migrations(
                            singleton, version, updated_at
                        ) VALUES (TRUE, %s, %s)
                        ON CONFLICT (singleton) DO UPDATE
                        SET version = EXCLUDED.version,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (self.CURRENT_SCHEMA_VERSION, now),
                    )
                elif version < self.CURRENT_SCHEMA_VERSION:
                    # Additive migration for early private deployments.
                    connection.execute(
                        """
                        ALTER TABLE corrections
                        ADD COLUMN IF NOT EXISTS user_supplied BOOLEAN
                            NOT NULL DEFAULT FALSE
                        """
                    )
                    connection.execute(
                        """
                        ALTER TABLE corrections
                        ADD COLUMN IF NOT EXISTS term_category TEXT
                        """
                    )
                    connection.execute(
                        """
                        ALTER TABLE corrections
                        ADD COLUMN IF NOT EXISTS language TEXT
                            NOT NULL DEFAULT 'en-ZA'
                        """
                    )
                    connection.execute(
                        """
                        ALTER TABLE vocabulary
                        ADD COLUMN IF NOT EXISTS term_category TEXT
                        """
                    )
                    connection.execute(
                        """
                        ALTER TABLE vocabulary
                        ADD COLUMN IF NOT EXISTS language TEXT
                            NOT NULL DEFAULT 'en-ZA'
                        """
                    )
                    connection.execute(
                        """
                        UPDATE schema_migrations
                        SET version = %s, updated_at = %s
                        WHERE singleton
                        """,
                        (self.CURRENT_SCHEMA_VERSION, now),
                    )


def _job_dir(working_root: Path, job_id: str) -> Path:
    return Path(working_root) / "jobs" / _safe_name(job_id, "job_id")


@dataclass(frozen=True)
class JobPaths:
    job_dir: Path
    analysis: Path
    effective: Path
    snapshot: Path
    corrected: Path
    state: Path
    report: Path
    events: Path
    ai_request: Path
    ai_response: Path

    @classmethod
    def for_job(
        cls,
        working_root: Path,
        job_id: str,
        *,
        transcript_relative_path: str = "analysis/transcript_en.json",
    ) -> "JobPaths":
        job_dir = _job_dir(working_root, job_id)
        analysis = job_dir / "analysis"
        effective = job_dir / transcript_relative_path
        return cls(
            job_dir=job_dir,
            analysis=analysis,
            effective=effective,
            snapshot=analysis / "transcript_en.autocorrect_source.json",
            corrected=analysis / "corrected_transcript.json",
            state=analysis / "autocorrection_state.json",
            report=analysis / "autocorrection_report.json",
            events=analysis / "autocorrection_events.json",
            ai_request=analysis / "autocorrection_ai_request.json",
            ai_response=analysis / "autocorrection_ai_response.json",
        )


def ensure_job(
    store: Store,
    *,
    paths: JobPaths,
    job_id: str,
    user_id: str | None,
    project_id: str | None,
) -> dict[str, Any]:
    if not paths.effective.is_file():
        raise FileNotFoundError(paths.effective)

    current = load_json(paths.effective)
    transcript_view(current)
    current_hash = sha256_json(current)
    now = utc_now()

    with store.transaction() as db:
        row = db.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()

        if row is None:
            if paths.snapshot.exists():
                raise AutocorrectError(
                    "Autocorrect snapshot exists but database job state "
                    "does not. Restore the database or use reset-source."
                )
            atomic_write_json(paths.snapshot, current)
            source_hash = sha256_json(current)
            db.execute(
                """
                INSERT INTO jobs(
                    job_id, user_id, project_id, source_path,
                    snapshot_path, source_sha256, effective_sha256,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    user_id,
                    project_id,
                    str(paths.effective),
                    str(paths.snapshot),
                    source_hash,
                    current_hash,
                    now,
                    now,
                ),
            )
            db.execute(
                """
                INSERT INTO events(
                    job_id, correction_id, action, actor_id,
                    payload_json, created_at
                ) VALUES (?, NULL, 'source_snapshotted', ?, ?, ?)
                """,
                (
                    job_id,
                    user_id,
                    json.dumps(
                        {
                            "source_sha256": source_hash,
                            "source_path": str(paths.effective),
                            "snapshot_path": str(paths.snapshot),
                        },
                        sort_keys=True,
                    ),
                    now,
                ),
            )
        else:
            if not paths.snapshot.is_file():
                raise AutocorrectError(
                    f"Immutable source snapshot is missing: {paths.snapshot}"
                )
            snapshot_hash = sha256_json(load_json(paths.snapshot))
            if snapshot_hash != row["source_sha256"]:
                raise AutocorrectError(
                    "Source snapshot hash does not match database state"
                )
            allowed_hashes = {
                row["source_sha256"],
                row["effective_sha256"],
            }
            if current_hash not in allowed_hashes:
                raise AutocorrectError(
                    "transcript_en.json changed outside autocorrect. "
                    "Run reset-source before analysing the new transcript."
                )
            db.execute(
                """
                UPDATE jobs
                SET user_id = COALESCE(?, user_id),
                    project_id = COALESCE(?, project_id),
                    updated_at = ?
                WHERE job_id = ?
                """,
                (user_id, project_id, now, job_id),
            )

        refreshed = db.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return dict(refreshed)


def reset_source(
    store: Store,
    *,
    paths: JobPaths,
    job_id: str,
    actor_id: str | None,
) -> dict[str, Any]:
    if not paths.effective.is_file():
        raise FileNotFoundError(paths.effective)
    document = load_json(paths.effective)
    transcript_view(document)
    source_hash = sha256_json(document)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = paths.analysis / "autocorrect_archive" / timestamp
    archive.mkdir(parents=True, exist_ok=False)

    for path in (
        paths.snapshot,
        paths.corrected,
        paths.state,
        paths.report,
        paths.events,
        paths.ai_request,
        paths.ai_response,
    ):
        if path.exists():
            target = archive / path.name
            target.write_bytes(path.read_bytes())

    atomic_write_json(paths.snapshot, document)
    now = utc_now()
    with store.transaction() as db:
        db.execute(
            "DELETE FROM corrections WHERE job_id = ?",
            (job_id,),
        )
        row = db.execute(
            "SELECT user_id, project_id FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            db.execute(
                """
                INSERT INTO jobs(
                    job_id, user_id, project_id, source_path,
                    snapshot_path, source_sha256, effective_sha256,
                    created_at, updated_at
                ) VALUES (?, NULL, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    str(paths.effective),
                    str(paths.snapshot),
                    source_hash,
                    source_hash,
                    now,
                    now,
                ),
            )
        else:
            db.execute(
                """
                UPDATE jobs
                SET source_sha256 = ?, effective_sha256 = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (source_hash, source_hash, now, job_id),
            )
        db.execute(
            """
            INSERT INTO events(
                job_id, correction_id, action, actor_id,
                payload_json, created_at
            ) VALUES (?, NULL, 'source_reset', ?, ?, ?)
            """,
            (
                job_id,
                actor_id,
                json.dumps(
                    {
                        "source_sha256": source_hash,
                        "archive": str(archive),
                    },
                    sort_keys=True,
                ),
                now,
            ),
        )
    return {
        "job_id": job_id,
        "source_sha256": source_hash,
        "archive": str(archive),
    }


def _read_hints(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"rules": [], "review_patterns": []}
    value = load_json(path)
    return value


def _regex_flags(names: Sequence[str]) -> int:
    result = 0
    mapping = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE": re.MULTILINE,
    }
    for name in names:
        if name not in mapping:
            raise AutocorrectError(f"Unsupported regex flag: {name}")
        result |= mapping[name]
    return result


def curated_hint_auto_applies(rule: Mapping[str, Any]) -> bool:
    """Return whether an explicit configured confusion rule is authoritative.

    Rules in ``config/autocorrect_hints.json`` are curated application
    configuration, not AI suggestions. They therefore auto-apply by default.
    Set ``auto_apply`` to false (or ``review_required`` to true) for a rule
    that must remain in the human-review queue.
    """

    if bool(rule.get("review_required", False)):
        return False
    return bool(rule.get("auto_apply", True))


def registry_candidates(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    registry = load_json(path)
    entities = registry.get("entities")
    if not isinstance(entities, list):
        raise AutocorrectError(
            f"Entity registry has no entities list: {path}"
        )

    result: list[dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict) or not entity.get("active", True):
            continue
        entity_id = str(entity.get("entity_id", "")).strip()
        canonical = (
            entity.get("canonical_text")
            or entity.get("display_text")
            or entity.get("canonical")
            or entity.get("name")
        )
        if not entity_id or not canonical:
            continue
        aliases: list[str] = []
        for value in [
            *(entity.get("aliases") or []),
            *(entity.get("stt_phrases") or []),
        ]:
            cleaned = " ".join(str(value).split()).strip()
            if not cleaned:
                continue
            if (
                " " not in cleaned
                and cleaned.casefold() in UNSAFE_GENERIC_ALIASES
            ):
                continue
            aliases.append(cleaned)
        result.append(
            {
                "candidate_id": f"entity:{entity_id}",
                "replacement_text": str(canonical),
                "entity_id": entity_id,
                "source": "entity_registry",
                "scope": "registry",
                "surfaces": list(
                    dict.fromkeys([str(canonical), *aliases])
                ),
            }
        )
    return result


def _owners(
    *,
    job_id: str,
    user_id: str | None,
    project_id: str | None,
) -> list[tuple[str, str]]:
    result = [("job", job_id)]
    if user_id:
        result.append(("user", user_id))
    if project_id:
        result.append(("project", project_id))
    return result


def vocabulary_candidates(
    store: Store,
    *,
    job_id: str,
    user_id: str | None,
    project_id: str | None,
) -> list[dict[str, Any]]:
    owners = _owners(
        job_id=job_id,
        user_id=user_id,
        project_id=project_id,
    )
    clauses = " OR ".join("(scope = ? AND owner_id = ?)" for _ in owners)
    parameters: list[str] = []
    for scope, owner in owners:
        parameters.extend([scope, owner])

    with store.connect() as db:
        rows = db.execute(
            f"""
            SELECT * FROM vocabulary
            WHERE status = 'active' AND ({clauses})
            """,
            parameters,
        ).fetchall()

    return [
        {
            "candidate_id": f"vocab:{row['rule_id']}",
            "replacement_text": row["replacement_text"],
            "entity_id": row["entity_id"],
            "source": "learned_vocabulary",
            "scope": row["scope"],
            "surfaces": [row["heard_text"]],
            "confirmations": int(row["confirmations"]),
            "reversions": int(row["reversions"]),
            "mode": row["mode"],
            "rule_id": row["rule_id"],
        }
        for row in rows
    ]



def public_dictionary_candidates(
    store: Store,
    *,
    language: str = "en-ZA",
) -> list[dict[str, Any]]:
    """Approved public terms are candidates, never automatic rules."""

    with store.connect() as db:
        rows = db.execute(
            """
            SELECT * FROM public_terms
            WHERE status = 'approved' AND language = ?
            ORDER BY independent_users DESC, distinct_jobs DESC, updated_at DESC
            """,
            (language,),
        ).fetchall()

    return [
        {
            "candidate_id": f"public:{row['term_id']}",
            "replacement_text": row["replacement_text"],
            "entity_id": None,
            "source": "public_dictionary",
            "scope": "public",
            "surfaces": [row["heard_text"]],
            "independent_users": int(row["independent_users"]),
            "distinct_jobs": int(row["distinct_jobs"]),
            "term_category": row["term_category"],
            "language": row["language"],
        }
        for row in rows
    ]


def _suppressed(
    store: Store,
    *,
    heard_text: str,
    replacement_text: str,
    owners: list[tuple[str, str]],
) -> bool:
    heard = normalize(heard_text)
    replacement = normalize(replacement_text)
    with store.connect() as db:
        for scope, owner in owners:
            row = db.execute(
                """
                SELECT 1 FROM suppressions
                WHERE scope = ? AND owner_id = ?
                  AND heard_norm = ? AND replacement_norm = ?
                """,
                (scope, owner, heard, replacement),
            ).fetchone()
            if row:
                return True
    return False


def _best_candidates(
    heard_text: str,
    entries: Iterable[dict[str, Any]],
    *,
    threshold: float,
    maximum: int,
) -> list[dict[str, Any]]:
    heard = normalize(heard_text)
    found: dict[str, dict[str, Any]] = {}

    for entry in entries:
        best = 0.0
        best_surface = ""
        for surface in entry.get("surfaces", []):
            surface_normalized = normalize(str(surface))
            score = SequenceMatcher(None, heard, surface_normalized).ratio()
            if surface_normalized == heard:
                score = 1.0
            if score > best:
                best = score
                best_surface = str(surface)
        if best < threshold:
            continue

        scope_bonus = {
            "job": 0.05,
            "user": 0.04,
            "project": 0.03,
            "registry": 0.01,
        }.get(str(entry.get("scope")), 0.0)
        candidate = {
            **entry,
            "score": round(min(1.0, best + scope_bonus), 6),
            "match_surface": best_surface,
        }
        key = normalize(candidate["replacement_text"])
        existing = found.get(key)
        if existing is None or candidate["score"] > existing["score"]:
            found[key] = candidate

    return sorted(
        found.values(),
        key=lambda item: (
            -float(item["score"]),
            str(item.get("scope")),
            normalize(str(item["replacement_text"])),
        ),
    )[:maximum]


def _learned_exact_rule(
    candidates: list[dict[str, Any]],
    *,
    auto_confirmation_threshold: int,
) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        if candidate.get("source") != "learned_vocabulary":
            continue
        if float(candidate.get("score", 0)) < 1.0:
            continue
        confirmations = int(candidate.get("confirmations", 0))
        reversions = int(candidate.get("reversions", 0))
        mode = str(candidate.get("mode"))
        if reversions:
            continue
        if mode == "always" or confirmations >= auto_confirmation_threshold:
            eligible.append(candidate)

    unique = {
        normalize(str(item["replacement_text"])): item
        for item in eligible
    }
    if len(unique) == 1:
        return next(iter(unique.values()))
    return None


def _candidate_payload(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item["candidate_id"],
            "replacement_text": item["replacement_text"],
            "entity_id": item.get("entity_id"),
            "source": item.get("source"),
            "scope": item.get("scope"),
            "score": item.get("score"),
            "match_surface": item.get("match_surface"),
        }
        for item in candidates
    ]


def _upsert_correction(
    db: PgSession,
    *,
    job_id: str,
    item: TranscriptItem,
    char_start: int,
    char_end: int,
    heard_text: str,
    replacement_text: str,
    source_type: str,
    confidence: float,
    status: str,
    entity_id: str | None,
    start_ms: int | None,
    end_ms: int | None,
    reason: str,
    candidates: list[dict[str, Any]],
) -> str:
    correction_id = stable_id(
        "correction",
        job_id,
        item.item_id,
        char_start,
        char_end,
        normalize(heard_text),
    )
    now = utc_now()
    existing = db.execute(
        "SELECT status FROM corrections WHERE correction_id = ?",
        (correction_id,),
    ).fetchone()

    if existing is None:
        db.execute(
            """
            INSERT INTO corrections(
                correction_id, job_id, item_id, item_index,
                text_field, char_start, char_end, heard_text,
                replacement_text, source_type, confidence, status,
                entity_id, start_ms, end_ms, reason,
                candidates_json, source_item_sha256,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                correction_id,
                job_id,
                item.item_id,
                item.index,
                item.text_field,
                char_start,
                char_end,
                heard_text,
                replacement_text,
                source_type,
                confidence,
                status,
                entity_id,
                start_ms,
                end_ms,
                reason,
                json.dumps(_candidate_payload(candidates), sort_keys=True),
                item.source_hash,
                now,
                now,
            ),
        )
        db.execute(
            """
            INSERT INTO events(
                job_id, correction_id, action, actor_id,
                payload_json, created_at
            ) VALUES (?, ?, 'candidate_created', NULL, ?, ?)
            """,
            (
                job_id,
                correction_id,
                json.dumps(
                    {
                        "status": status,
                        "heard_text": heard_text,
                        "replacement_text": replacement_text,
                        "source_type": source_type,
                    },
                    sort_keys=True,
                ),
                now,
            ),
        )
    elif existing["status"] in REVIEW_STATUSES:
        db.execute(
            """
            UPDATE corrections
            SET replacement_text = ?, source_type = ?,
                confidence = ?, status = ?, entity_id = ?,
                start_ms = ?, end_ms = ?, reason = ?,
                candidates_json = ?, updated_at = ?
            WHERE correction_id = ?
            """,
            (
                replacement_text,
                source_type,
                confidence,
                status,
                entity_id,
                start_ms,
                end_ms,
                reason,
                json.dumps(_candidate_payload(candidates), sort_keys=True),
                now,
                correction_id,
            ),
        )
    return correction_id


def detect_candidates(
    store: Store,
    *,
    job_id: str,
    user_id: str | None,
    project_id: str | None,
    source_document: dict[str, Any],
    hints_path: Path | None,
    registry_path: Path | None,
    fuzzy_threshold: float,
    max_candidates: int,
    low_confidence_threshold: float,
    auto_confirmation_threshold: int,
) -> list[str]:
    view = transcript_view(source_document)
    hints = _read_hints(hints_path)
    vocab = vocabulary_candidates(
        store,
        job_id=job_id,
        user_id=user_id,
        project_id=project_id,
    )
    registry = registry_candidates(registry_path)
    public = public_dictionary_candidates(store, language="en-ZA")
    entries = [*vocab, *registry, *public]
    owners = _owners(
        job_id=job_id,
        user_id=user_id,
        project_id=project_id,
    )
    detected: list[tuple[TranscriptItem, int, int, str, str, str, str | None, float, str, list[dict[str, Any]]]] = []

    for rule in hints.get("rules", []):
        if not isinstance(rule, dict):
            continue
        pattern = str(rule.get("pattern", "")).strip()
        replacement = str(rule.get("replacement", "")).strip()
        if not pattern or not replacement:
            continue
        regex = re.compile(
            pattern,
            _regex_flags(rule.get("flags", ["IGNORECASE"])),
        )
        for item in view.items:
            for match in regex.finditer(item.text):
                candidates = _best_candidates(
                    match.group(0),
                    entries,
                    threshold=fuzzy_threshold,
                    maximum=max_candidates,
                )
                hint_candidate = {
                    "candidate_id": stable_id(
                        "hint",
                        rule.get("entity_id"),
                        replacement,
                    ),
                    "replacement_text": replacement,
                    "entity_id": rule.get("entity_id"),
                    "source": "confusion_hint",
                    "scope": "hint",
                    "surfaces": [match.group(0)],
                    "score": 0.99,
                    "match_surface": match.group(0),
                }
                candidates = _candidate_payload(
                    [hint_candidate, *candidates]
                )
                unique = {}
                for candidate in candidates:
                    key = normalize(candidate["replacement_text"])
                    unique.setdefault(key, candidate)
                candidates = list(unique.values())[:max_candidates]
                learned = _learned_exact_rule(
                    candidates,
                    auto_confirmation_threshold=auto_confirmation_threshold,
                )
                authoritative = curated_hint_auto_applies(rule)
                selected = hint_candidate if authoritative else (learned or candidates[0])
                source_type = (
                    "curated_hint"
                    if authoritative
                    else ("learned_exact" if learned else "confusion_hint")
                )
                if _suppressed(
                    store,
                    heard_text=match.group(0),
                    replacement_text=selected["replacement_text"],
                    owners=owners,
                ):
                    source_type = "suppressed"
                detected.append(
                    (
                        item,
                        match.start(),
                        match.end(),
                        match.group(0),
                        selected["replacement_text"],
                        source_type,
                        selected.get("entity_id"),
                        float(selected.get("score", 0.99)),
                        str(
                            rule.get("reason")
                            or (
                                "Curated exact acoustic correction"
                                if authoritative
                                else "Known acoustic confusion candidate"
                            )
                        ),
                        candidates,
                    )
                )

    for rule in hints.get("review_patterns", []):
        if not isinstance(rule, dict):
            continue
        pattern = str(rule.get("pattern", "")).strip()
        if not pattern:
            continue
        regex = re.compile(pattern, re.IGNORECASE)
        for item in view.items:
            for match in regex.finditer(item.text):
                candidates = _best_candidates(
                    match.group(0),
                    entries,
                    threshold=fuzzy_threshold,
                    maximum=max_candidates,
                )
                replacement = (
                    candidates[0]["replacement_text"]
                    if candidates
                    else match.group(0)
                )
                detected.append(
                    (
                        item,
                        match.start(),
                        match.end(),
                        match.group(0),
                        replacement,
                        "ambiguous_hint",
                        (
                            candidates[0].get("entity_id")
                            if candidates
                            else None
                        ),
                        (
                            float(candidates[0].get("score", 0))
                            if candidates
                            else 0.0
                        ),
                        str(
                            rule.get("reason")
                            or "Ambiguous phrase requires review"
                        ),
                        candidates,
                    )
                )

    # Learned exact forms are detected even when no static hint exists.
    for entry in vocab:
        for surface in entry.get("surfaces", []):
            if not surface.strip():
                continue
            regex = re.compile(
                rf"(?<!\w){re.escape(surface)}(?!\w)",
                re.IGNORECASE,
            )
            for item in view.items:
                for match in regex.finditer(item.text):
                    candidates = _best_candidates(
                        match.group(0),
                        entries,
                        threshold=fuzzy_threshold,
                        maximum=max_candidates,
                    )
                    learned = _learned_exact_rule(
                        candidates,
                        auto_confirmation_threshold=auto_confirmation_threshold,
                    )
                    if learned is None:
                        continue
                    if _suppressed(
                        store,
                        heard_text=match.group(0),
                        replacement_text=learned["replacement_text"],
                        owners=owners,
                    ):
                        status_source = "suppressed"
                    else:
                        status_source = "learned_exact"
                    detected.append(
                        (
                            item,
                            match.start(),
                            match.end(),
                            match.group(0),
                            learned["replacement_text"],
                            status_source,
                            learned.get("entity_id"),
                            1.0,
                            "Previously confirmed exact autocorrect rule",
                            candidates,
                        )
                    )

    # Optional word confidence detection.
    words: list[dict[str, Any]] = []
    if isinstance(source_document.get("words"), list):
        words.extend(
            item
            for item in source_document["words"]
            if isinstance(item, dict)
        )
    collection = source_document[view.collection_key]
    for raw in collection:
        if isinstance(raw, dict) and isinstance(raw.get("words"), list):
            words.extend(
                item
                for item in raw["words"]
                if isinstance(item, dict)
            )

    for word in words:
        confidence = _confidence(word)
        if confidence is None or confidence >= low_confidence_threshold:
            continue
        heard = ""
        for field in ("text", "word", "display", "lexical"):
            if isinstance(word.get(field), str):
                heard = str(word[field]).strip()
                break
        if len(re.sub(r"\W", "", heard)) < 3:
            continue
        word_start, word_end = _timing(word)
        containing = None
        for item in view.items:
            if (
                word_start is not None
                and item.start_ms is not None
                and item.end_ms is not None
                and item.start_ms <= word_start <= item.end_ms
            ):
                containing = item
                break
        if containing is None:
            continue
        match = re.search(
            rf"(?<!\w){re.escape(heard)}(?!\w)",
            containing.text,
            re.IGNORECASE,
        )
        if match is None:
            continue
        candidates = _best_candidates(
            heard,
            entries,
            threshold=fuzzy_threshold,
            maximum=max_candidates,
        )
        replacement = (
            candidates[0]["replacement_text"]
            if candidates
            else heard
        )
        detected.append(
            (
                containing,
                match.start(),
                match.end(),
                heard,
                replacement,
                "low_confidence",
                (
                    candidates[0].get("entity_id")
                    if candidates
                    else None
                ),
                (
                    float(candidates[0].get("score", 0))
                    if candidates
                    else 0.0
                ),
                (
                    f"Azure confidence {confidence:.3f} is below "
                    f"{low_confidence_threshold:.3f}"
                ),
                candidates,
            )
        )

    # Deduplicate overlapping spans; specific hints beat low-confidence words.
    priority = {
        "curated_hint": 110,
        "learned_exact": 100,
        "suppressed": 100,
        "confusion_hint": 90,
        "ambiguous_hint": 80,
        "low_confidence": 50,
    }
    detected.sort(
        key=lambda row: (
            row[0].index,
            row[1],
            -priority.get(row[5], 0),
            -(row[2] - row[1]),
        )
    )
    kept: list[tuple[Any, ...]] = []
    for row in detected:
        item, start, end = row[0], row[1], row[2]
        overlaps = [
            existing
            for existing in kept
            if existing[0].item_id == item.item_id
            and start < existing[2]
            and existing[1] < end
        ]
        if not overlaps:
            kept.append(row)
            continue
        best_existing = max(
            overlaps,
            key=lambda value: priority.get(value[5], 0),
        )
        if priority.get(row[5], 0) > priority.get(
            best_existing[5], 0
        ):
            kept.remove(best_existing)
            kept.append(row)

    correction_ids: list[str] = []
    with store.transaction() as db:
        for (
            item,
            char_start,
            char_end,
            heard,
            replacement,
            source_type,
            entity_id,
            confidence,
            reason,
            candidates,
        ) in kept:
            start_ms, end_ms = approximate_span_timing(
                item,
                char_start,
                char_end,
            )
            if source_type in {"curated_hint", "learned_exact"}:
                status = "auto_applied"
            elif source_type == "suppressed":
                status = "suppressed"
            elif not candidates:
                status = "needs_review"
            else:
                status = "suggested"
            correction_ids.append(
                _upsert_correction(
                    db,
                    job_id=job_id,
                    item=item,
                    char_start=char_start,
                    char_end=char_end,
                    heard_text=heard,
                    replacement_text=replacement,
                    source_type=source_type,
                    confidence=confidence,
                    status=status,
                    entity_id=entity_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    reason=reason,
                    candidates=candidates,
                )
            )
    return correction_ids


def build_ai_request(
    store: Store,
    *,
    job_id: str,
    source_document: dict[str, Any],
    maximum: int,
) -> dict[str, Any]:
    view = transcript_view(source_document)
    item_map = {item.item_id: item for item in view.items}
    with store.connect() as db:
        rows = db.execute(
            """
            SELECT * FROM corrections
            WHERE job_id = ? AND status IN ('suggested', 'needs_review')
            ORDER BY
                CASE source_type
                    WHEN 'ambiguous_hint' THEN 0
                    WHEN 'confusion_hint' THEN 1
                    ELSE 2
                END,
                confidence DESC,
                item_index,
                char_start
            LIMIT ?
            """,
            (job_id, maximum),
        ).fetchall()

    spans = []
    for row in rows:
        item = item_map[row["item_id"]]
        context = []
        for candidate_item in view.items[
            max(0, item.index - 2) : min(
                len(view.items), item.index + 3
            )
        ]:
            context.append(
                {
                    "item_id": candidate_item.item_id,
                    "speaker": candidate_item.speaker,
                    "start_ms": candidate_item.start_ms,
                    "end_ms": candidate_item.end_ms,
                    "text": candidate_item.text,
                    "target": candidate_item.item_id == item.item_id,
                }
            )
        spans.append(
            {
                "correction_id": row["correction_id"],
                "heard_text": row["heard_text"],
                "reason": row["reason"],
                "context": context,
                "candidates": json.loads(row["candidates_json"]),
            }
        )

    request = {
        "schema_version": AI_REQUEST_SCHEMA,
        "created_at": utc_now(),
        "job_id": job_id,
        "task": "Rank supplied autocorrect candidates only.",
        "rules": [
            "Never rewrite transcript text.",
            "Never invent a candidate.",
            "Choose ACCEPT_CANDIDATE, KEEP_AZURE, or ASK_USER.",
            "Use ASK_USER for emerging terms or ambiguity.",
        ],
        "spans": spans,
        "required_response": {
            "schema_version": AI_RESPONSE_SCHEMA,
            "decisions": [
                {
                    "correction_id": "supplied correction ID",
                    "decision": (
                        "ACCEPT_CANDIDATE|KEEP_AZURE|ASK_USER"
                    ),
                    "candidate_id": (
                        "required only for ACCEPT_CANDIDATE"
                    ),
                    "confidence": 0.0,
                    "reason": "short contextual reason",
                    "question": "optional user question",
                }
            ],
        },
    }
    request["request_sha256"] = sha256_json(request)
    return request


def validate_ai_response(
    request: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if response.get("schema_version") != AI_RESPONSE_SCHEMA:
        raise AutocorrectError("Unsupported AI response schema")

    allowed = {
        span["correction_id"]: {
            candidate["candidate_id"]
            for candidate in span.get("candidates", [])
        }
        for span in request.get("spans", [])
    }
    decisions = response.get("decisions")
    if not isinstance(decisions, list):
        raise AutocorrectError("AI decisions must be a list")

    validated: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise AutocorrectError("AI decision must be an object")
        correction_id = str(decision.get("correction_id", ""))
        if correction_id not in allowed or correction_id in validated:
            raise AutocorrectError(
                f"Unknown or duplicate AI correction: {correction_id}"
            )
        action = str(decision.get("decision", ""))
        if action not in {
            "ACCEPT_CANDIDATE",
            "KEEP_AZURE",
            "ASK_USER",
        }:
            raise AutocorrectError(f"Invalid AI action: {action}")
        candidate_id = decision.get("candidate_id")
        if action == "ACCEPT_CANDIDATE":
            if str(candidate_id) not in allowed[correction_id]:
                raise AutocorrectError(
                    f"AI invented candidate: {candidate_id}"
                )
        elif candidate_id is not None:
            raise AutocorrectError(
                f"{action} may not contain candidate_id"
            )
        confidence = float(decision.get("confidence", 0))
        if not 0 <= confidence <= 1:
            raise AutocorrectError("AI confidence must be between 0 and 1")
        validated[correction_id] = {
            "correction_id": correction_id,
            "decision": action,
            "candidate_id": (
                str(candidate_id) if candidate_id is not None else None
            ),
            "confidence": confidence,
            "reason": str(decision.get("reason") or ""),
            "question": str(decision.get("question") or ""),
        }

    missing = sorted(set(allowed) - set(validated))
    if missing:
        raise AutocorrectError(f"AI omitted corrections: {missing}")
    return validated


def apply_ai_advice(
    store: Store,
    *,
    job_id: str,
    advice: dict[str, dict[str, Any]],
    auto_apply_confidence: float | None = None,
    minimum_candidate_score: float = 0.75,
) -> dict[str, int]:
    """Store constrained AI advice and optionally apply only strong matches.

    The model can select only a candidate already supplied by the deterministic
    detector.  Existing callers retain advice-only behavior by leaving
    ``auto_apply_confidence`` unset.
    """

    if auto_apply_confidence is not None and not 0 <= auto_apply_confidence <= 1:
        raise AutocorrectError("AI auto-apply confidence must be between 0 and 1")
    if not 0 <= minimum_candidate_score <= 1:
        raise AutocorrectError("minimum candidate score must be between 0 and 1")

    now = utc_now()
    summary = {
        "decision_count": 0,
        "auto_applied_count": 0,
        "suggested_count": 0,
        "dismissed_count": 0,
        "review_count": 0,
    }
    with store.transaction() as db:
        for correction_id, decision in advice.items():
            row = db.execute(
                """
                SELECT * FROM corrections
                WHERE correction_id = ? AND job_id = ?
                """,
                (correction_id, job_id),
            ).fetchone()
            if row is None:
                raise AutocorrectError(
                    f"Unknown correction: {correction_id}"
                )
            candidates = json.loads(row["candidates_json"])
            selected = None
            if decision["candidate_id"]:
                selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate["candidate_id"]
                        == decision["candidate_id"]
                    ),
                    None,
                )
                if selected is None:
                    raise AutocorrectError(
                        "AI selected a candidate that disappeared"
                    )
            status = row["status"]
            replacement = row["replacement_text"]
            entity_id = row["entity_id"]
            event_action = "ai_advice"
            if decision["decision"] == "ACCEPT_CANDIDATE":
                replacement = selected["replacement_text"]
                entity_id = selected.get("entity_id")
                candidate_score = float(selected.get("score") or 0)
                strong_acceptance = bool(
                    auto_apply_confidence is not None
                    and float(decision["confidence"])
                    >= auto_apply_confidence
                    and candidate_score >= minimum_candidate_score
                    and normalize(str(row["heard_text"]))
                    != normalize(str(replacement))
                )
                if strong_acceptance:
                    status = "auto_applied"
                    event_action = "ai_autocorrect_applied"
                    summary["auto_applied_count"] += 1
                else:
                    status = "suggested"
                    summary["suggested_count"] += 1
            elif decision["decision"] == "KEEP_AZURE":
                status = "dismissed"
                summary["dismissed_count"] += 1
            elif decision["decision"] == "ASK_USER":
                status = "needs_review"
                summary["review_count"] += 1
            summary["decision_count"] += 1
            reason = (
                f"{row['reason']} | AI: {decision['reason']}"
                if decision["reason"]
                else row["reason"]
            )
            db.execute(
                """
                UPDATE corrections
                SET replacement_text = ?, entity_id = ?, status = ?,
                    confidence = ?, reason = ?, updated_at = ?
                WHERE correction_id = ?
                """,
                (
                    replacement,
                    entity_id,
                    status,
                    decision["confidence"],
                    reason,
                    now,
                    correction_id,
                ),
            )
            db.execute(
                """
                INSERT INTO events(
                    job_id, correction_id, action, actor_id,
                    payload_json, created_at
                ) VALUES (?, ?, ?, 'ai', ?, ?)
                """,
                (
                    job_id,
                    correction_id,
                    event_action,
                    json.dumps(decision, sort_keys=True),
                    now,
                ),
            )
    return summary


def _row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value["candidates"] = json.loads(value.pop("candidates_json"))
    value["active"] = value["status"] in ACTIVE_STATUSES
    return value


def list_corrections(
    store: Store,
    *,
    job_id: str,
) -> list[dict[str, Any]]:
    with store.connect() as db:
        rows = db.execute(
            """
            SELECT * FROM corrections
            WHERE job_id = ?
            ORDER BY item_index, char_start, correction_id
            """,
            (job_id,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _record_event(
    db: PgSession,
    *,
    job_id: str,
    correction_id: str | None,
    action: str,
    actor_id: str | None,
    payload: dict[str, Any],
) -> None:
    db.execute(
        """
        INSERT INTO events(
            job_id, correction_id, action, actor_id,
            payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            correction_id,
            action,
            actor_id,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            utc_now(),
        ),
    )


def _scope_owner(
    *,
    scope: str,
    job_id: str,
    user_id: str | None,
    project_id: str | None,
) -> str:
    if scope not in VALID_SCOPES:
        raise AutocorrectError(f"Invalid scope: {scope}")
    if scope == "job":
        return job_id
    if scope == "user":
        if not user_id:
            raise AutocorrectError("user scope requires user_id")
        return user_id
    if not project_id:
        raise AutocorrectError("project scope requires project_id")
    return project_id


def learn_rule(
    db: PgSession,
    *,
    heard_text: str,
    replacement_text: str,
    entity_id: str | None,
    scope: str,
    owner_id: str,
    mode: str,
    confirmed: bool,
    reverted: bool,
    term_category: str = "other",
    language: str = "en-ZA",
) -> str:
    heard_norm = normalize(heard_text)
    replacement_norm = normalize(replacement_text)
    rule_id = stable_id(
        "rule",
        scope,
        owner_id,
        heard_norm,
        replacement_norm,
    )
    now = utc_now()
    row = db.execute(
        "SELECT * FROM vocabulary WHERE rule_id = ?",
        (rule_id,),
    ).fetchone()
    if row is None:
        db.execute(
            """
            INSERT INTO vocabulary(
                rule_id, scope, owner_id, heard_text, heard_norm,
                replacement_text, replacement_norm, entity_id,
                confirmations, reversions, mode, status,
                term_category, language, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
            """,
            (
                rule_id,
                scope,
                owner_id,
                heard_text,
                heard_norm,
                replacement_text,
                replacement_norm,
                entity_id,
                1 if confirmed else 0,
                1 if reverted else 0,
                mode,
                validate_term_category(term_category),
                language,
                now,
                now,
            ),
        )
    else:
        db.execute(
            """
            UPDATE vocabulary
            SET confirmations = confirmations + ?,
                reversions = reversions + ?,
                mode = CASE
                    WHEN ? = 'always' THEN 'always'
                    ELSE mode
                END,
                entity_id = COALESCE(?, entity_id),
                term_category = COALESCE(?, term_category),
                language = COALESCE(?, language),
                updated_at = ?
            WHERE rule_id = ?
            """,
            (
                1 if confirmed else 0,
                1 if reverted else 0,
                mode,
                entity_id,
                validate_term_category(term_category),
                language,
                now,
                rule_id,
            ),
        )
    return rule_id


def _refresh_public_term(
    db: PgSession,
    *,
    heard_text: str,
    replacement_text: str,
    term_category: str,
    language: str,
    domain: str | None,
) -> str:
    heard_norm = normalize(heard_text)
    replacement_norm = normalize(replacement_text)
    term_id = stable_id(
        "public_term",
        heard_norm,
        replacement_norm,
        language,
    )
    now = utc_now()
    counts = db.execute(
        """
        SELECT
          COUNT(DISTINCT user_hash) AS independent_users,
          COUNT(DISTINCT job_hash) AS distinct_jobs
        FROM platform_evidence
        WHERE heard_norm = ? AND replacement_norm = ?
          AND language = ?
        """,
        (heard_norm, replacement_norm, language),
    ).fetchone()
    independent_users = int(counts["independent_users"] or 0)
    distinct_jobs = int(counts["distinct_jobs"] or 0)

    db.execute(
        """
        INSERT INTO public_terms(
            term_id, heard_norm, replacement_norm,
            heard_text, replacement_text, term_category,
            language, domain, independent_users, distinct_jobs,
            status, privacy_review_required, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', TRUE, ?, ?)
        ON CONFLICT (heard_norm, replacement_norm, language)
        DO UPDATE SET
            heard_text = EXCLUDED.heard_text,
            replacement_text = EXCLUDED.replacement_text,
            term_category = EXCLUDED.term_category,
            domain = COALESCE(EXCLUDED.domain, public_terms.domain),
            independent_users = EXCLUDED.independent_users,
            distinct_jobs = EXCLUDED.distinct_jobs,
            updated_at = EXCLUDED.updated_at
        """,
        (
            term_id,
            heard_norm,
            replacement_norm,
            heard_text,
            replacement_text,
            term_category,
            language,
            domain,
            independent_users,
            distinct_jobs,
            now,
            now,
        ),
    )
    return term_id


def change_correction(
    store: Store,
    *,
    job_id: str,
    correction_id: str,
    action: str,
    actor_id: str,
    replacement_text: str | None = None,
    learn_scopes: Iterable[str] = (),
    user_id: str | None = None,
    project_id: str | None = None,
    always: bool = False,
    never: bool = False,
    contribute_globally: bool = False,
    term_category: str = "other",
    language: str = "en-ZA",
    domain: str | None = None,
    consent_version: str = "v1",
) -> dict[str, Any]:
    now = utc_now()
    scopes = list(dict.fromkeys(learn_scopes))
    if invalid := sorted(set(scopes) - VALID_SCOPES):
        raise AutocorrectError(f"Invalid learning scopes: {invalid}")
    category = validate_term_category(term_category)
    language = clean_user_term(language)
    if domain is not None:
        domain = clean_user_term(domain)

    with store.transaction() as db:
        row = db.execute(
            """
            SELECT * FROM corrections
            WHERE correction_id = ? AND job_id = ?
            FOR UPDATE
            """,
            (correction_id, job_id),
        ).fetchone()
        if row is None:
            raise AutocorrectError(
                f"Unknown correction: {correction_id}"
            )

        previous_status = row["status"]
        previous_replacement = row["replacement_text"]
        user_supplied = False

        if action == "accept":
            new_status = "confirmed"
            new_replacement = previous_replacement
        elif action == "edit":
            new_replacement = clean_user_term(replacement_text or "")
            new_status = "edited"
            user_supplied = True
        elif action == "revert":
            new_replacement = previous_replacement
            new_status = "reverted"
        elif action == "redo":
            new_replacement = previous_replacement
            new_status = "confirmed"
        elif action == "dismiss":
            new_replacement = previous_replacement
            new_status = "dismissed"
        else:
            raise AutocorrectError(f"Unsupported action: {action}")

        db.execute(
            """
            UPDATE corrections
            SET replacement_text = ?, status = ?,
                user_supplied = CASE WHEN ? THEN TRUE ELSE user_supplied END,
                term_category = COALESCE(?, term_category),
                language = COALESCE(?, language),
                source_type = CASE
                    WHEN ? THEN 'user_free_text'
                    ELSE source_type
                END,
                updated_at = ?
            WHERE correction_id = ?
            """,
            (
                new_replacement,
                new_status,
                user_supplied,
                category,
                language,
                user_supplied,
                now,
                correction_id,
            ),
        )

        mode = "always" if always else "learned"
        if action in {"accept", "edit"}:
            for scope in scopes:
                owner_id = _scope_owner(
                    scope=scope,
                    job_id=job_id,
                    user_id=user_id,
                    project_id=project_id,
                )
                learn_rule(
                    db,
                    heard_text=row["heard_text"],
                    replacement_text=new_replacement,
                    entity_id=row["entity_id"],
                    scope=scope,
                    owner_id=owner_id,
                    mode=mode,
                    confirmed=True,
                    reverted=False,
                    term_category=category,
                    language=language,
                )
        elif action == "revert":
            for scope in scopes:
                owner_id = _scope_owner(
                    scope=scope,
                    job_id=job_id,
                    user_id=user_id,
                    project_id=project_id,
                )
                learn_rule(
                    db,
                    heard_text=row["heard_text"],
                    replacement_text=previous_replacement,
                    entity_id=row["entity_id"],
                    scope=scope,
                    owner_id=owner_id,
                    mode="learned",
                    confirmed=False,
                    reverted=True,
                    term_category=category,
                    language=language,
                )
                if never:
                    db.execute(
                        """
                        INSERT OR IGNORE INTO suppressions(
                            scope, owner_id, heard_norm,
                            replacement_norm, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            scope,
                            owner_id,
                            normalize(row["heard_text"]),
                            normalize(previous_replacement),
                            now,
                        ),
                    )

        public_term_id = None
        if contribute_globally:
            if action not in {"accept", "edit"}:
                raise AutocorrectError(
                    "Only confirmed corrections may be contributed"
                )
            if not user_id:
                raise AutocorrectError(
                    "Public contribution requires user_id"
                )
            salt = os.environ.get("MATHULA_TV_FEEDBACK_HASH_SALT")
            if not salt:
                raise AutocorrectError(
                    "Public contribution requires "
                    "MATHULA_TV_FEEDBACK_HASH_SALT"
                )
            user_hash = hashlib.sha256(
                f"{salt}\0{user_id}".encode("utf-8")
            ).hexdigest()
            job_hash = hashlib.sha256(
                f"{salt}\0{job_id}".encode("utf-8")
            ).hexdigest()
            evidence_id = stable_id(
                "evidence",
                normalize(row["heard_text"]),
                normalize(new_replacement),
                language,
                user_hash,
                job_hash,
            )
            db.execute(
                """
                INSERT INTO platform_evidence(
                    evidence_id, heard_norm, replacement_norm,
                    heard_text, replacement_text,
                    user_hash, job_hash, submission_kind,
                    term_category, language, domain,
                    consent_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    evidence_id,
                    normalize(row["heard_text"]),
                    normalize(new_replacement),
                    row["heard_text"],
                    new_replacement,
                    user_hash,
                    job_hash,
                    "new_term" if user_supplied else "confirmed_candidate",
                    category,
                    language,
                    domain,
                    consent_version,
                    now,
                ),
            )
            public_term_id = _refresh_public_term(
                db,
                heard_text=row["heard_text"],
                replacement_text=new_replacement,
                term_category=category,
                language=language,
                domain=domain,
            )

        _record_event(
            db,
            job_id=job_id,
            correction_id=correction_id,
            action=action,
            actor_id=actor_id,
            payload={
                "previous_status": previous_status,
                "new_status": new_status,
                "previous_replacement": previous_replacement,
                "new_replacement": new_replacement,
                "user_supplied": user_supplied,
                "term_category": category,
                "language": language,
                "domain": domain,
                "learn_scopes": scopes,
                "always": always,
                "never": never,
                "contribute_globally": contribute_globally,
                "public_term_id": public_term_id,
                "consent_version": consent_version,
            },
        )

        updated = db.execute(
            "SELECT * FROM corrections WHERE correction_id = ?",
            (correction_id,),
        ).fetchone()
        return _row_to_dict(updated)


def undo_all_auto_applied(
    store: Store,
    *,
    job_id: str,
    actor_id: str,
) -> int:
    now = utc_now()
    with store.transaction() as db:
        rows = db.execute(
            """
            SELECT correction_id FROM corrections
            WHERE job_id = ? AND status = 'auto_applied'
            """,
            (job_id,),
        ).fetchall()
        ids = [row["correction_id"] for row in rows]
        for correction_id in ids:
            db.execute(
                """
                UPDATE corrections
                SET status = 'reverted', updated_at = ?
                WHERE correction_id = ?
                """,
                (now, correction_id),
            )
            _record_event(
                db,
                job_id=job_id,
                correction_id=correction_id,
                action="revert_auto",
                actor_id=actor_id,
                payload={
                    "previous_status": "auto_applied",
                    "new_status": "reverted",
                },
            )
        _record_event(
            db,
            job_id=job_id,
            correction_id=None,
            action="undo_all_auto_applied",
            actor_id=actor_id,
            payload={"count": len(ids)},
        )
        return len(ids)


def render_effective_transcript(
    store: Store,
    *,
    paths: JobPaths,
    job_id: str,
) -> dict[str, Any]:
    source = load_json(paths.snapshot)
    view = transcript_view(source)
    item_map = {item.item_id: item for item in view.items}

    with store.connect() as db:
        rows = db.execute(
            """
            SELECT * FROM corrections
            WHERE job_id = ? AND status IN (
                'auto_applied', 'confirmed', 'edited'
            )
            ORDER BY item_index, char_start
            """,
            (job_id,),
        ).fetchall()
        events = db.execute(
            """
            SELECT * FROM events
            WHERE job_id = ?
            ORDER BY event_id
            """,
            (job_id,),
        ).fetchall()

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["item_id"], []).append(row)

    effective = copy.deepcopy(source)
    overlays: list[dict[str, Any]] = []
    for item_id, corrections in grouped.items():
        item = item_map.get(item_id)
        if item is None:
            raise AutocorrectError(
                f"Correction references missing source item: {item_id}"
            )
        raw = effective[view.collection_key][item.index]
        if sha256_json(source[view.collection_key][item.index]) != row_source_hash(corrections):
            raise AutocorrectError(
                f"Source item changed since correction detection: {item_id}"
            )
        text = str(raw[item.text_field])
        occupied: list[tuple[int, int]] = []

        for correction in sorted(
            corrections,
            key=lambda row: row["char_start"],
            reverse=True,
        ):
            start = int(correction["char_start"])
            end = int(correction["char_end"])
            if any(
                start < other_end and other_start < end
                for other_start, other_end in occupied
            ):
                raise AutocorrectError(
                    f"Overlapping active corrections in {item_id}"
                )
            occupied.append((start, end))
            actual = text[start:end]
            if normalize(actual) != normalize(
                correction["heard_text"]
            ):
                raise AutocorrectError(
                    f"Correction {correction['correction_id']} expected "
                    f"{correction['heard_text']!r}, found {actual!r}"
                )
            replacement = correction["replacement_text"]
            text = text[:start] + replacement + text[end:]
            overlays.append(
                {
                    "correction_id": correction["correction_id"],
                    "item_id": item_id,
                    "azure_text": correction["heard_text"],
                    "corrected_text": replacement,
                    "source_char_start": start,
                    "source_char_end": end,
                    "start_ms": correction["start_ms"],
                    "end_ms": correction["end_ms"],
                    "status": correction["status"],
                    "source_type": correction["source_type"],
                    "entity_id": correction["entity_id"],
                    "raw_azure_artifacts_unchanged": True,
                }
            )
        raw[item.text_field] = text

    texts = [
        str(effective[view.collection_key][item.index][item.text_field]).strip()
        for item in view.items
    ]
    effective["complete_text"] = " ".join(
        text for text in texts if text
    )
    effective["autocorrect"] = {
        "schema_version": AUTOCORRECT_SCHEMA,
        "source_snapshot": str(paths.snapshot),
        "source_sha256": sha256_json(source),
        "active_correction_count": len(overlays),
        "overlays": sorted(
            overlays,
            key=lambda item: (
                item["item_id"],
                item["source_char_start"],
            ),
        ),
        "rendered_at": utc_now(),
    }

    effective_hash = sha256_json(effective)
    atomic_write_json(paths.corrected, effective)
    atomic_write_json(paths.effective, effective)

    with store.transaction() as db:
        db.execute(
            """
            UPDATE jobs
            SET effective_sha256 = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (effective_hash, utc_now(), job_id),
        )
        _record_event(
            db,
            job_id=job_id,
            correction_id=None,
            action="transcript_rendered",
            actor_id="system",
            payload={
                "effective_sha256": effective_hash,
                "active_correction_count": len(overlays),
            },
        )

    corrections = list_corrections(store, job_id=job_id)
    status_counts: dict[str, int] = {}
    for correction in corrections:
        status_counts[correction["status"]] = (
            status_counts.get(correction["status"], 0) + 1
        )
    state = {
        "schema_version": AUTOCORRECT_SCHEMA,
        "job_id": job_id,
        "source_sha256": sha256_json(source),
        "effective_sha256": effective_hash,
        "total": len(corrections),
        "status_counts": status_counts,
        "active_count": sum(
            count
            for status, count in status_counts.items()
            if status in ACTIVE_STATUSES
        ),
        "review_count": sum(
            count
            for status, count in status_counts.items()
            if status in REVIEW_STATUSES
        ),
        "can_undo_all": status_counts.get("auto_applied", 0) > 0,
        "corrections": corrections,
        "updated_at": utc_now(),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "job_id": job_id,
        "source_artifact": str(paths.snapshot),
        "effective_artifact": str(paths.effective),
        "corrected_artifact": str(paths.corrected),
        "raw_azure_artifacts_modified": False,
        "turn_or_segment_count_preserved": True,
        "speaker_and_timing_metadata_preserved": True,
        "active_overlays": effective["autocorrect"]["overlays"],
        "status_counts": status_counts,
        "updated_at": utc_now(),
    }
    event_export = {
        "schema_version": EVENT_EXPORT_SCHEMA,
        "job_id": job_id,
        "events": [
            {
                **dict(event),
                "payload": json.loads(event["payload_json"]),
            }
            for event in events
        ],
    }
    for event in event_export["events"]:
        event.pop("payload_json", None)

    atomic_write_json(paths.state, state)
    atomic_write_json(paths.report, report)
    atomic_write_json(paths.events, event_export)
    return state


def row_source_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    values = {str(row["source_item_sha256"]) for row in rows}
    if len(values) != 1:
        raise AutocorrectError(
            "Active corrections disagree about their source item hash"
        )
    return next(iter(values))


def questions(
    store: Store,
    *,
    job_id: str,
    maximum: int,
) -> list[dict[str, Any]]:
    with store.connect() as db:
        rows = db.execute(
            """
            SELECT * FROM corrections
            WHERE job_id = ?
              AND status IN ('suggested', 'needs_review')
            ORDER BY
              CASE status WHEN 'needs_review' THEN 0 ELSE 1 END,
              CASE source_type
                WHEN 'ambiguous_hint' THEN 0
                WHEN 'confusion_hint' THEN 1
                ELSE 2
              END,
              confidence DESC,
              item_index,
              char_start
            LIMIT ?
            """,
            (job_id, maximum),
        ).fetchall()

    result = []
    for row in rows:
        correction = _row_to_dict(row)
        options = [
            {
                "option_id": candidate["candidate_id"],
                "label": candidate["replacement_text"],
                "kind": "candidate",
                "recommended": (
                    normalize(candidate["replacement_text"])
                    == normalize(correction["replacement_text"])
                ),
                "entity_id": candidate.get("entity_id"),
            }
            for candidate in correction["candidates"]
        ]
        options.extend(
            [
                {
                    "option_id": "KEEP_ORIGINAL",
                    "label": correction["heard_text"],
                    "kind": "keep_original",
                    "recommended": False,
                },
                {
                    "option_id": "OTHER",
                    "label": "Enter another word or phrase",
                    "kind": "free_text",
                    "recommended": False,
                },
                {
                    "option_id": "UNSURE",
                    "label": "I’m not sure",
                    "kind": "unsure",
                    "recommended": False,
                },
            ]
        )
        result.append(
            {
                "correction_id": correction["correction_id"],
                "question": (
                    f"What did the speaker say where Azure heard "
                    f"{correction['heard_text']!r}?"
                ),
                "start_ms": correction["start_ms"],
                "end_ms": correction["end_ms"],
                "heard_text": correction["heard_text"],
                "reason": correction["reason"],
                "options": options,
            }
        )
    return result


def export_adaptive_phrases(
    store: Store,
    *,
    job_id: str,
    user_id: str | None,
    project_id: str | None,
    maximum: int,
) -> dict[str, Any]:
    owners = _owners(
        job_id=job_id,
        user_id=user_id,
        project_id=project_id,
    )
    clauses = " OR ".join("(scope = ? AND owner_id = ?)" for _ in owners)
    parameters: list[str] = []
    for scope, owner in owners:
        parameters.extend([scope, owner])

    with store.connect() as db:
        rows = db.execute(
            f"""
            SELECT * FROM vocabulary
            WHERE status = 'active' AND ({clauses})
            ORDER BY
              CASE mode WHEN 'always' THEN 0 ELSE 1 END,
              confirmations DESC,
              updated_at DESC
            """,
            parameters,
        ).fetchall()
        public_rows = db.execute(
            """
            SELECT heard_text, replacement_text
            FROM public_terms
            WHERE status = 'approved' AND language = 'en-ZA'
            ORDER BY independent_users DESC, distinct_jobs DESC
            """
        ).fetchall()

    phrases: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for value in (row["replacement_text"], row["heard_text"]):
            key = normalize(value)
            if key and key not in seen:
                seen.add(key)
                phrases.append(value)
                if len(phrases) >= maximum:
                    break
        if len(phrases) >= maximum:
            break

    for row in public_rows:
        for value in (row["replacement_text"], row["heard_text"]):
            key = normalize(value)
            if key and key not in seen:
                seen.add(key)
                phrases.append(value)
                if len(phrases) >= maximum:
                    break
        if len(phrases) >= maximum:
            break

    return {
        "schema_version": PHRASE_EXPORT_SCHEMA,
        "job_id": job_id,
        "user_id": user_id,
        "project_id": project_id,
        "phrase_count": len(phrases),
        "phrases": phrases,
        "created_at": utc_now(),
    }


def platform_candidate_report(
    store: Store,
    *,
    minimum_users: int,
    minimum_jobs: int,
    status: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT * FROM public_terms
        WHERE 1 = 1
    """
    parameters: list[Any] = []
    if status:
        query += " AND status = ?"
        parameters.append(status)
    query += """
        ORDER BY independent_users DESC, distinct_jobs DESC, updated_at DESC
    """
    with store.connect() as db:
        rows = db.execute(query, parameters).fetchall()

    return [
        {
            **dict(row),
            "promotion_eligible": (
                int(row["independent_users"]) >= minimum_users
                and int(row["distinct_jobs"]) >= minimum_jobs
            ),
            "auto_promoted": False,
        }
        for row in rows
    ]


def moderate_public_term(
    store: Store,
    *,
    term_id: str,
    action: str,
    moderator_id: str,
    minimum_users: int = 5,
    minimum_jobs: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    if action not in {"approve", "reject"}:
        raise AutocorrectError("action must be approve or reject")
    now = utc_now()
    with store.transaction() as db:
        row = db.execute(
            "SELECT * FROM public_terms WHERE term_id = ? FOR UPDATE",
            (term_id,),
        ).fetchone()
        if row is None:
            raise AutocorrectError(f"Unknown public term: {term_id}")

        eligible = (
            int(row["independent_users"]) >= minimum_users
            and int(row["distinct_jobs"]) >= minimum_jobs
        )
        if action == "approve" and not eligible and not force:
            raise AutocorrectError(
                "Public term has insufficient independent evidence; "
                "use force only after explicit editorial review"
            )

        if action == "approve":
            db.execute(
                """
                UPDATE public_terms
                SET status = 'approved', approved_by = ?, approved_at = ?,
                    rejected_by = NULL, rejected_at = NULL,
                    privacy_review_required = FALSE, updated_at = ?
                WHERE term_id = ?
                """,
                (moderator_id, now, now, term_id),
            )
        else:
            db.execute(
                """
                UPDATE public_terms
                SET status = 'rejected', rejected_by = ?, rejected_at = ?,
                    approved_by = NULL, approved_at = NULL,
                    updated_at = ?
                WHERE term_id = ?
                """,
                (moderator_id, now, now, term_id),
            )

        _record_event(
            db,
            job_id="platform",
            correction_id=None,
            action=f"public_term_{action}",
            actor_id=moderator_id,
            payload={
                "term_id": term_id,
                "eligible": eligible,
                "force": force,
            },
        )
        updated = db.execute(
            "SELECT * FROM public_terms WHERE term_id = ?",
            (term_id,),
        ).fetchone()
        return dict(updated)
