import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mathula_tv.colab_package as package


class Blob:
    def __init__(self, bucket, name): self.bucket, self.name = bucket, name
    def upload_from_filename(self, path, **kwargs): self.bucket.data[self.name] = Path(path).read_bytes()
    def upload_from_string(self, value, **kwargs): self.bucket.data[self.name] = value.encode()
class Bucket:
    def __init__(self): self.data = {}
    def blob(self, name): return Blob(self, name)


def test_publish_wheel_manifest(monkeypatch, tmp_path):
    def run(command, **kwargs):
        if "wheel" in command:
            destination = Path(command[command.index("--wheel-dir") + 1])
            (destination / "mathula_tv-0.1.3-py3-none-any.whl").write_bytes(b"wheel")
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout="abc123\n")
    monkeypatch.setattr(package.subprocess, "run", run)
    bucket = Bucket(); result = package.publish_colab_package(tmp_path, bucket)
    stored = json.loads(bucket.data["mathula-tv/runtime/manifest.json"])
    assert stored["sha256"] == hashlib.sha256(b"wheel").hexdigest()
    assert stored["git_commit"] == "abc123" and result["package_version"] == "0.1.3" and stored["wheel_size"]==5


def test_wheel_checksum_validation_and_install(tmp_path):
    wheel = tmp_path / "mathula_tv.whl"; wheel.write_bytes(b"wheel")
    manifest = {"wheel_object":"runtime/mathula_tv.whl", "sha256":hashlib.sha256(b"wheel").hexdigest(), "wheel_size":5, "build_timestamp":"now", "package_version":"1"}
    calls=[]; package.install_verified_wheel(manifest, wheel, runner=lambda command, check: calls.append(command))
    assert calls and "pip" in calls[0] and "--force-reinstall" in calls[0] and "--no-deps" in calls[0]
    manifest["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"): package.verify_package_manifest(manifest, wheel)


def test_notebook_contains_secure_executable_installation():
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/mathula_tv_colab_worker.ipynb").read_text())
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for required in ("auth.authenticate_user()", "userdata", "HUGGINGFACE_TOKEN", "package_manifest['sha256']", "pip', 'install'", "run_analysis_worker"):
        assert required in code
    assert all(any(line.strip() and not line.lstrip().startswith("#") for line in cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
