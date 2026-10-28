"""The shipped .env.example must load cleanly, and blank values mean 'default'."""
from pathlib import Path

from src.core.config import Settings

EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def test_env_example_loads_without_errors():
    # Every blank `KEY=` in the example (e.g. GEMINI_TEMPERATURE=) must be accepted.
    s = Settings(_env_file=str(EXAMPLE))
    assert s.gemini_temperature is None


def test_blank_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GEMINI_TEMPERATURE", "")
    monkeypatch.setenv("TTS_VOICE_FEMALE", "")
    s = Settings(_env_file=None)
    assert s.gemini_temperature is None
    assert s.tts_voice_female == "Kore"


def test_explicit_value_still_applies(monkeypatch):
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.7")
    assert Settings(_env_file=None).gemini_temperature == 0.7
