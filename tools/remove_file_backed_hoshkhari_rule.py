#!/usr/bin/env python3
"""Remove only the temporary file-backed rule added by the earlier bad package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


TEMP_PATTERN = r"\bhorse[\s-]+curry\b"
TEMP_REPLACEMENT = "hoshkhari"
TEMP_REASON = "Known Azure STT acoustic confusion: hoshkhari"


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def is_temporary_rule(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        str(value.get("pattern") or "") == TEMP_PATTERN
        and str(value.get("replacement") or "").casefold() == TEMP_REPLACEMENT
        and str(value.get("reason") or "") == TEMP_REASON
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    path = args.repo.resolve() / "config/autocorrect_hints.json"
    if not path.is_file():
        print(f"No hints file found: {path}")
        return 0

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    rules = value.get("rules", [])
    if not isinstance(rules, list):
        raise SystemExit(f"Expected rules to be a list in {path}")

    kept = [rule for rule in rules if not is_temporary_rule(rule)]
    removed = len(rules) - len(kept)
    if removed:
        value["rules"] = kept
        atomic_write(path, value)
    print(json.dumps({"path": str(path), "removed": removed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
