from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_package_manifest(manifest: dict[str, Any], wheel: Path) -> None:
    required = {"wheel_object", "sha256", "wheel_size", "build_timestamp", "package_version"}
    if not required.issubset(manifest):
        raise ValueError("Colab package manifest is incomplete")
    if Path(manifest["wheel_object"]).name != wheel.name:
        raise ValueError("Downloaded wheel does not match package manifest")
    if sha256_file(wheel) != manifest["sha256"]:
        raise ValueError("Downloaded wheel SHA-256 mismatch")
    if wheel.stat().st_size != manifest["wheel_size"]: raise ValueError("Downloaded wheel size mismatch")


def install_verified_wheel(manifest: dict[str, Any], wheel: Path, runner=subprocess.run) -> None:
    verify_package_manifest(manifest, wheel)
    runner([sys.executable,"-m","pip","install","--quiet","--force-reinstall","--no-deps","--no-cache-dir",str(wheel)],check=True)


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def publish_colab_package(root: Path, bucket: Any, prefix: str = "mathula-tv") -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        dist = Path(temporary)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "--wheel-dir", str(dist), str(root)], check=True)
        wheels = list(dist.glob("mathula_tv-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one Mathula TV wheel")
        wheel = wheels[0]
        wheel_name = f"{prefix.strip('/')}/runtime/{wheel.name}"
        bucket.blob(wheel_name).upload_from_filename(str(wheel), content_type="application/zip")
        manifest = {
            "schema_version": "colab-package-v1",
            "wheel_object": wheel_name,
            "sha256": sha256_file(wheel),
            "wheel_size": wheel.stat().st_size,
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(root),
            "package_version": __version__,
        }
        manifest_name = f"{prefix.strip('/')}/runtime/manifest.json"
        bucket.blob(manifest_name).upload_from_string(json.dumps(manifest, indent=2), content_type="application/json")
        manifest["manifest_object"] = manifest_name
        return manifest
