#!/usr/bin/env python3
"""Install PostgreSQL-backed production autocorrect into Mathula TV."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

FILES = [
    "src/mathula_tv/autocorrect.py",
    "src/mathula_tv/autocorrect_ai.py",
    "scripts/mathula_autocorrect.py",
    "scripts/install_production_autocorrect.py",
    "tests/test_autocorrect.py",
    "tests/test_autocorrect_postgres_live.py",
    "config/autocorrect.json",
    "config/stt_confusion_overrides.json",
    "docs/production-autocorrect.md",
]

EXPERIMENTAL = [
    "src/mathula_tv/stt_corrections.py",
    "scripts/apply_stt_corrections.py",
    "tests/test_stt_corrections.py",
    "src/mathula_tv/transcript_correction.py",
    "scripts/mathula_autocorrect.py",
    "tests/test_transcript_correction.py",
    "src/mathula_tv/adaptive_vocabulary.py",
    "src/mathula_tv/correction_review.py",
    "scripts/mathula_corrections.py",
    "tests/test_adaptive_vocabulary.py",
    "tests/test_correction_review.py",
    "config/adaptive_correction.json",
    "config/stt_confusion_overrides.json",
]


def patch_pyproject(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if 'database = ["psycopg[binary,pool]>=3.2,<4"]' in text:
        return False

    header = "[project.optional-dependencies]"
    if header not in text:
        text = text.rstrip() + (
            "\n\n[project.optional-dependencies]\n"
            'database = ["psycopg[binary,pool]>=3.2,<4"]\n'
        )
    else:
        text = text.replace(
            header,
            header
            + '\ndatabase = ["psycopg[binary,pool]>=3.2,<4"]',
            1,
        )
    path.write_text(text, encoding="utf-8")
    return True


def patch_env_example(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "MATHULA_TV_DATABASE_URL=" in text:
        return False

    block = """
# PostgreSQL-backed transcript autocorrect. This may point to the same
# PostgreSQL server used by commission-ai; Mathula TV uses a separate schema.
MATHULA_TV_DATABASE_URL=
MATHULA_TV_DATABASE_SCHEMA=mathula_autocorrect

# Random private salt used to pseudonymise opted-in public dictionary evidence.
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(48))"
MATHULA_TV_FEEDBACK_HASH_SALT=
"""
    path.write_text(text.rstrip() + "\n\n" + block.lstrip(), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument(
        "--payload",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    payload = args.payload.resolve()
    if not (repo / ".git").is_dir():
        raise SystemExit(f"Not a Git repository: {repo}")

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    allowed = set(EXPERIMENTAL)
    unexpected = []
    for line in status:
        relative = line[3:].strip()
        if relative not in allowed:
            unexpected.append(line)
    if unexpected:
        raise SystemExit(
            "Working tree has unrelated changes:\n"
            + "\n".join(unexpected)
        )

    for relative in EXPERIMENTAL:
        target = repo / relative
        if target.exists():
            target.unlink()

    for relative in FILES:
        source = payload / relative
        if not source.is_file():
            raise SystemExit(f"Payload file missing: {source}")
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    pyproject_changed = patch_pyproject(repo / "pyproject.toml")
    env_changed = patch_env_example(repo / ".env.example")

    (repo / "scripts/mathula_autocorrect.py").chmod(0o755)

    print("Installed PostgreSQL production autocorrect:")
    for relative in FILES:
        print(" -", relative)
    if pyproject_changed:
        print(" - patched pyproject.toml with the database extra")
    if env_changed:
        print(" - added PostgreSQL settings to .env.example")
    print()
    print("Install the database dependency:")
    print("  .venv/bin/python -m pip install -e '.[database]'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
