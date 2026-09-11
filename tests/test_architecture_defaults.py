from pathlib import Path

from mathula_tv.ai_provider import DEFAULT_GPT_DEPLOYMENT, PRODUCTION_AI_PROVIDER
import mathula_tv.azure_stt as azure_stt
from mathula_tv.azure_stt import AzureSTTRouter
from mathula_tv.cli import parser
from mathula_tv.config import load_settings
from mathula_tv.diarization import select_authoritative


def test_production_provider_defaults_and_optional_pyannote(tmp_path, monkeypatch):
    for name in (
        "MATHULA_TV_AI_PROVIDER",
        "AZURE_AI_DEPLOYMENT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "MATHULA_TV_ENABLE_PYANNOTE_DIAGNOSTIC",
        "MATHULA_TV_OPENVOICE_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(tmp_path)
    assert settings.ai_provider == PRODUCTION_AI_PROVIDER == "azure-openai-gpt"
    assert settings.gpt_model == DEFAULT_GPT_DEPLOYMENT == "gpt-5.6-sol-1"
    assert settings.pyannote_enabled is False
    assert settings.openvoice_revision == ""  # live GPU preparation requires an immutable reviewed SHA
    # New/unvalidated fast path (see azure_tts_sdk.py) must default off -- it
    # touches real Azure billing/audio quality and hasn't been validated on a
    # real job yet, unlike enable_island_timing which defaults on.
    assert settings.enable_sdk_group_synthesis is False
    # Phase 14 (turn-level bookmark synthesis): real-job-validated twice on
    # two different jobs (2026-09-10/11) -- fixes a confirmed mouth-close drift
    # defect AND a confirmed severe-rush defect in the island-splitting
    # fallback it replaces (English-character-count-proportional windows are a
    # poor proxy for isiZulu's per-sentence expansion-ratio variance). Now the
    # default, independently toggleable from enable_sdk_group_synthesis above.
    assert settings.enable_turn_group_synthesis is True


def test_sdk_group_synthesis_env_toggle_flips_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MATHULA_TV_ENABLE_SDK_GROUP_SYNTHESIS", "1")
    settings = load_settings(tmp_path)
    assert settings.enable_sdk_group_synthesis is True


def test_turn_group_synthesis_env_toggle_can_disable_the_new_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MATHULA_TV_ENABLE_TURN_GROUP_SYNTHESIS", "0")
    settings = load_settings(tmp_path)
    assert settings.enable_turn_group_synthesis is False


def test_azure_speaker_labels_are_authoritative_when_pyannote_disagrees():
    azure = {"turns": [{"speaker": "AZURE_1", "start": 0, "end": 2}]}
    pyannote = {
        "turns": [
            {"speaker": "P0", "start": 0, "end": 1},
            {"speaker": "P1", "start": 1, "end": 2},
        ]
    }
    turns, source, warnings = select_authoritative(pyannote, azure)
    assert source == "azure" and turns == azure["turns"]
    assert "diagnostic disagreement" in warnings[0]


def test_removed_backends_are_not_cli_selectable_or_registered():
    help_text = parser().format_help().lower()
    assert "f5" not in help_text
    assert "omnivoice" not in help_text
    assert "openvoice" in help_text and "azure" in help_text


def test_no_f5_production_module_exists():
    root = Path(__file__).parents[1] / "src/mathula_tv"
    assert not any("f5" in path.name.casefold() for path in root.iterdir())


def test_azure_stt_router_uses_injected_batch_only_when_fast_is_ineligible(tmp_path, monkeypatch):
    audio = tmp_path / "large.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(azure_stt, "probe", lambda _path: {"duration": 100})
    monkeypatch.setattr(azure_stt, "FAST_MAX_BYTES", 1)

    class Backend:
        api_version = "2025-10-15"
        request_duration_seconds = 1.0

        def __init__(self, provider):
            self.provider = provider
            self.calls = 0

        def transcribe(self, path, locale):
            self.calls += 1
            return {"phrases": [{"text": "hello"}]}

    fast = Backend("azure-fast")
    batch = Backend("azure-batch")
    router = AzureSTTRouter(fast, batch)
    assert router.transcribe(audio) == {"phrases": [{"text": "hello"}]}
    assert fast.calls == 0 and batch.calls == 1 and router.provider == "azure-batch"
