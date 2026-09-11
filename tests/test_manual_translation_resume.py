from types import SimpleNamespace

from mathula_tv.one_call_translation import _is_retryable_translation_failure


def _job(state="failed_retryable", **error):
    return SimpleNamespace(state=state, last_error=error)


def test_manual_handoff_failure_is_retryable_for_manual_translation():
    job = _job(
        stage="ai",
        code="manual_ai_response_required",
        retryable=True,
        details={"operation": "multivariant_translation"},
    )
    assert _is_retryable_translation_failure(job, "manual-chatgpt") is True


def test_manual_handoff_failure_does_not_relax_azure_translation_gate():
    job = _job(
        stage="ai",
        code="manual_ai_response_required",
        retryable=True,
        details={"operation": "multivariant_translation"},
    )
    assert _is_retryable_translation_failure(job, "azure-openai-gpt") is False


def test_unrelated_manual_ai_failure_is_not_translation_retry():
    job = _job(
        stage="ai",
        code="manual_ai_response_required",
        retryable=True,
        details={"operation": "autocorrect_name_research"},
    )
    assert _is_retryable_translation_failure(job, "manual-chatgpt") is False


def test_existing_translation_stage_retry_remains_supported():
    job = _job(stage="translation", retryable=True)
    assert _is_retryable_translation_failure(job, "azure-openai-gpt") is True


def test_non_failed_retryable_state_is_not_retry():
    job = _job(
        state="analysis_ready",
        stage="ai",
        code="manual_ai_response_required",
        retryable=True,
        details={"operation": "multivariant_translation"},
    )
    assert _is_retryable_translation_failure(job, "manual-chatgpt") is False
