from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
NOTEBOOKS = {
    "colab": ROOT / "notebooks/mathula_tv_openvoice_colab_worker.ipynb",
    "kaggle": ROOT / "notebooks/mathula_tv_openvoice_kaggle_worker.ipynb",
}
STAGES = (
    "Runtime inspection",
    "Python and CUDA compatibility",
    "Repository checkout",
    "Dependency installation",
    "OpenVoice revision pinning",
    "Checkpoint materialisation",
    "Checkpoint hash verification",
    "Authentication",
    "GCS connectivity",
    "Job discovery",
    "Lease claim",
    "Plan download",
    "Azure source-embedding download",
    "Speaker reference and target-embedding download",
    "Input hash validation",
    "Model load",
    "Pending-turn conversion",
    "WAV validation",
    "GCS promotion",
    "Manifest creation",
    "Lease completion",
    "Server-reconciliation instructions",
    "Local sensitive-file cleanup",
)


def load_notebook(platform: str) -> dict:
    return json.loads(NOTEBOOKS[platform].read_text(encoding="utf-8"))


def notebook_code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("platform", NOTEBOOKS)
def test_openvoice_notebook_has_exact_output_free_23_stage_structure(platform):
    notebook = load_notebook(platform)
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert len(notebook["cells"]) == 1 + 2 * len(STAGES)
    assert notebook["cells"][0]["cell_type"] == "markdown"

    for index, expected_stage in enumerate(STAGES, 1):
        heading = notebook["cells"][2 * index - 1]
        code = notebook["cells"][2 * index]
        assert heading["cell_type"] == "markdown"
        assert "".join(heading["source"]).strip() == f"## {index:02d}. {expected_stage}"
        assert code["cell_type"] == "code"
        assert code["execution_count"] is None
        assert code["outputs"] == []
        ast.parse("".join(code["source"]), filename=f"{platform}-stage-{index:02d}")


@pytest.mark.parametrize("platform", NOTEBOOKS)
def test_openvoice_notebook_delegates_worker_and_storage_business_logic(platform):
    code = notebook_code(load_notebook(platform))
    required_calls = (
        "from mathula_tv.openvoice import",
        "from mathula_tv.openvoice_worker import",
        "from mathula_tv.gcs_store import",
        "OpenVoiceBackend(",
        "OpenVoiceAssetWorker(",
        "OpenVoiceWorker(",
        "GCSStore(",
        "GCSLeaseHeartbeat(",
        "store.assert_claim(",
        "store.promote(",
        "runtime_factory()",
    )
    for expected in required_calls:
        assert expected in code
    assert "class " not in code
    assert "def convert(" not in code
    assert "def extract_embedding(" not in code
    assert "omnivoice" not in code.lower()
    assert "f5_tts" not in code.lower()


@pytest.mark.parametrize("platform", NOTEBOOKS)
def test_openvoice_notebook_requires_immutable_inputs_and_live_acknowledgement(
    platform,
):
    code = notebook_code(load_notebook(platform))
    required_configuration = (
        "MATHULA_TV_REPOSITORY_COMMIT",
        "MATHULA_TV_OPENVOICE_REVISION",
        "MATHULA_TV_OPENVOICE_WORKER_MODE",
        "MATHULA_TV_OPENVOICE_ASSET_PLAN_SHA256",
        "MATHULA_TV_OPENVOICE_PLAN_SHA256",
        "MATHULA_TV_OPENVOICE_CHECKPOINTS_JSON",
        "MATHULA_TV_OPENVOICE_PYTHON_VERSION",
        "MATHULA_TV_OPENVOICE_TORCH_VERSION",
        "MATHULA_TV_OPENVOICE_RUNTIME_FACTORY",
        "MATHULA_TV_LIVE_OPERATION",
        "MATHULA_TV_LIVE_OPERATION_ACK",
        "I_ACKNOWLEDGE_LIVE_OPENVOICE_GCS_MUTATION",
    )
    for name in required_configuration:
        assert name in code
    assert 'COMMIT_RE = re.compile(r"[0-9a-f]{40}")' in code
    assert 'SHA256_RE = re.compile(r"[0-9a-f]{64}")' in code
    assert code.index("LIVE_OPERATION !=") < code.index("store.claim(")
    assert code.index("store.promote(") < code.index("store.finish_claim(")


@pytest.mark.parametrize("platform", NOTEBOOKS)
def test_openvoice_notebook_has_explicit_asset_and_conversion_modes(platform):
    code = notebook_code(load_notebook(platform))
    required_asset_contract = (
        'WORKER_MODE not in {"assets", "conversion"}',
        'LEASE_TASK = "openvoice_asset_preparation"',
        '"dubbing/openvoice/asset_plan.json"',
        '"dubbing/openvoice/asset_manifest.json"',
        "OpenVoiceAssetPlan.from_dict(plan_payload)",
        "source_embedding_manifest_path",
        "target_embedding_manifest_path",
        "candidate_manifest_path",
        "embedding_exists != embedding_manifest_exists",
        "Promoted Azure source cache must contain both embedding and manifest",
        "source_cache_pairs",
        "worker.pending_asset_ids()",
        "max_attempts_per_asset=MAX_ATTEMPTS",
        'next_subcommand = "reconcile-openvoice-assets"',
    )
    for value in required_asset_contract:
        assert value in code
    assert "backend.extract_embedding(" not in code
    assert "runtime.extract_embedding(" not in code
    assert code.index("OpenVoiceAssetWorker(") < code.index("worker.run()")


@pytest.mark.parametrize("platform", NOTEBOOKS)
def test_asset_manifest_binds_original_server_plan_before_promotion(platform):
    code = notebook_code(load_notebook(platform))
    file_binding = code.index(
        'manifest["server_plan_file_sha256"] = PLAN_SHA256'
    )
    binding_start = code.index(
        'if WORKER_MODE == "assets":\n'
        '    server_plan_artifact_sha256 = str(plan_payload.get("artifact_sha256", ""))',
        file_binding,
    )
    artifact_binding = code.index(
        'manifest["server_plan_artifact_sha256"] = server_plan_artifact_sha256',
        binding_start,
    )
    manifest_write = code.index(
        "atomic_write_json(manifest_path, manifest)", artifact_binding
    )
    manifest_hash = code.index("manifest_sha256 = sha256_file(manifest_path)")
    manifest_upload = code.index(
        "store.upload(JOB_ID, manifest_temporary, manifest_path", manifest_hash
    )
    manifest_promotion = code.index(
        "store.promote(JOB_ID, manifest_temporary, MANIFEST_RELATIVE", manifest_upload
    )

    assert "if not SHA256_RE.fullmatch(server_plan_artifact_sha256):" in code
    assert (
        code.index("sha256_file(plan_path) != PLAN_SHA256")
        < file_binding
        < binding_start
        < artifact_binding
        < manifest_write
        < manifest_hash
        < manifest_upload
        < manifest_promotion
    )


@pytest.mark.parametrize("platform", NOTEBOOKS)
def test_conversion_manifest_binds_server_plan_file_before_hashing(platform):
    code = notebook_code(load_notebook(platform))
    file_binding = code.index(
        'manifest["server_plan_file_sha256"] = PLAN_SHA256'
    )
    asset_only_branch = code.index('if WORKER_MODE == "assets":', file_binding)
    manifest_write = code.index(
        "atomic_write_json(manifest_path, manifest)", asset_only_branch
    )
    manifest_hash = code.index("manifest_sha256 = sha256_file(manifest_path)")
    manifest_promotion = code.index(
        "store.promote(JOB_ID, manifest_temporary, MANIFEST_RELATIVE",
        manifest_hash,
    )

    assert (
        code.index("sha256_file(plan_path) != PLAN_SHA256")
        < file_binding
        < asset_only_branch
        < manifest_write
        < manifest_hash
        < manifest_promotion
    )
    assert (
        'manifest["server_plan_file_sha256"] = PLAN_SHA256\n'
        'if WORKER_MODE == "assets":'
    ) in code


@pytest.mark.parametrize("platform", NOTEBOOKS)
def test_openvoice_notebook_prints_only_safe_metadata(platform):
    code = notebook_code(load_notebook(platform))
    tree = ast.parse(code)
    forbidden_print_names = {
        "CHECKPOINT_SPECS",
        "GCP_PROJECT",
        "GCS_BUCKET",
        "OPENVOICE_REPOSITORY_URL",
        "REPOSITORY_URL",
        "credential_payload",
        "credential_text",
        "gcs_client",
        "kaggle_credentials",
        "plan_payload",
        "status",
    }
    print_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert print_calls
    for call in print_calls:
        printed_names = {
            node.id for node in ast.walk(call) if isinstance(node, ast.Name)
        }
        assert printed_names.isdisjoint(forbidden_print_names)
        assert len(call.args) == 1 and isinstance(call.args[0], ast.Dict)

    lowered = code.lower()
    forbidden_literals = (
        "authorization:",
        "bearer ",
        "begin private key",
        "x-goog-signature",
        "signed_url",
        "19ba6d69f1b84132ba4f20599101834a",
    )
    for value in forbidden_literals:
        assert value not in lowered
    assert "transcript" not in lowered


def test_notebook_platform_differences_are_confined_to_runtime_and_authentication():
    colab = notebook_code(load_notebook("colab"))
    kaggle = notebook_code(load_notebook("kaggle"))
    assert "google.colab" in colab
    assert "kaggle_secrets" not in colab
    assert "/content/mathula-tv-openvoice-worker" in colab
    assert "kaggle_secrets" in kaggle
    assert "google.colab" not in kaggle
    assert "/kaggle/working/mathula-tv-openvoice-worker" in kaggle
    assert "MATHULA_TV_GCP_SERVICE_ACCOUNT_JSON" in kaggle
    assert "credential_payload.clear()" in kaggle


@pytest.mark.parametrize("platform", NOTEBOOKS)
def test_openvoice_notebook_cleans_job_local_voice_assets_after_promotion(platform):
    code = notebook_code(load_notebook(platform))
    assert "sensitive_paths.add" in code
    assert "sensitive_path.unlink()" in code
    assert code.index("store.finish_claim(") < code.index("sensitive_path.unlink()")
    assert '"local_sensitive_cleanup": "completed"' in code
