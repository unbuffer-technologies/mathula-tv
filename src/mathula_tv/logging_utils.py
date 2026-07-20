from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SENSITIVE_KEY = re.compile(r"(key|token|secret|password|credential|signature|sig)$", re.I)
BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
URL = re.compile(r"https?://[^\s]+", re.I)


def _redact_url(value: str) -> str:
    parts = urlsplit(value)
    safe = [(k, "[REDACTED]" if SENSITIVE_KEY.search(k) or k.lower().startswith("x-goog-") else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(safe), ""))


def redact(value: Any, key: str = "") -> Any:
    if SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if not isinstance(value, str):
        return value
    text = BEARER.sub(r"\1[REDACTED]", value)
    return URL.sub(lambda match: _redact_url(match.group(0)), text)


class JsonlHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "level": record.levelname, "logger": record.name, "message": redact(record.getMessage())}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def configure_logging(path: Path | None = None, verbose: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if path:
        handlers.append(JsonlHandler(path))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers, format="%(levelname)s %(message)s", force=True)
