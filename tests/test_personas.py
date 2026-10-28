from src.agents.prompts import PERSONAS, system_prompt
from src.chat import schemas
from src.core.config import settings


def test_nonstop_persona_is_general_spoken_and_grounded():
    p = system_prompt("nonstop")
    assert "NON-STOP" in p and "ANY topic" in p
    assert "no markdown" in p.lower()                 # spoken style
    assert "ACCURACY RULES" in p and "LANGUAGE RULES" in p
    assert "senior AI-SaaS software engineer" not in p  # not engineer-only


def test_every_mode_in_the_picker_exists_and_is_valid():
    for mode in ("nonstop", "executive", "coder", "architect", "security", "researcher", "writer"):
        assert mode in PERSONAS
        schemas._check_persona(mode)


def test_legacy_modes_still_work_for_old_conversations():
    for mode in ("developer", "ccna_trainer"):
        assert mode in PERSONAS


def test_nonstop_uses_the_general_chat_model():
    assert settings.model_for_persona("nonstop") == settings.chat_model
