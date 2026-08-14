from __future__ import annotations

import inspect

import pytest

from mathula_tv.cli import parser
from mathula_tv.direct_azure_dub import DirectAzureDubRenderer, DirectDubOptions


def test_cloud_timing_repair_cli_option_is_removed() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(
            ["dub-azure", "job-1", "--timing-repair-backend", "claude"]
        )


def test_direct_dub_options_have_no_cloud_timing_backend() -> None:
    options = DirectDubOptions()
    assert not hasattr(options, "timing_repair_backend")
    assert not hasattr(options, "timing_repair_batch_size")
    assert not hasattr(options, "max_timing_repair_attempts")


def test_renderer_accepts_no_timing_repair_provider() -> None:
    parameters = inspect.signature(DirectAzureDubRenderer).parameters
    assert "repair_provider_factory" not in parameters


def test_renderer_contains_no_claude_timing_repair_methods() -> None:
    assert not hasattr(DirectAzureDubRenderer, "_repair_timing_overflows_batch")
    assert not hasattr(DirectAzureDubRenderer, "_repair_timing_overflow")
