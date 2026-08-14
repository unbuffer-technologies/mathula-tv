from pathlib import Path

import pytest

from mathula_tv.azure_tts import AzureTTSIdempotencyError
from mathula_tv.production_pipeline import _synthesize_azure_candidate


def test_selective_refresh_only_retries_request_mismatch():
    sentinel = object()

    class MismatchBackend:
        def __init__(self):
            self.calls = []

        def synthesize(self, request, output_path, *, force=False):
            self.calls.append(force)
            if not force:
                raise AzureTTSIdempotencyError(
                    "different request", conflict_kind="request_mismatch"
                )
            return sentinel

    mismatch = MismatchBackend()
    result = _synthesize_azure_candidate(
        mismatch, object(), Path("candidate.wav"), force=False
    )
    assert result is sentinel
    assert mismatch.calls == [False, True]

    class CorruptBackend:
        def synthesize(self, request, output_path, *, force=False):
            raise AzureTTSIdempotencyError(
                "incomplete artifacts", conflict_kind="artifact_integrity"
            )

    with pytest.raises(AzureTTSIdempotencyError, match="incomplete artifacts"):
        _synthesize_azure_candidate(
            CorruptBackend(), object(), Path("candidate.wav"), force=False
        )
