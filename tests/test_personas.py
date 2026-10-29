import itertools

import pytest

from src.agents.prompts import PERSONAS, UNRESTRICTED_PERSONA, system_prompt
from src.chat import schemas
from src.core.config import settings

SPECIALISTS = ("executive", "coder", "architect", "security", "researcher", "writer")
PICKER = ("nonstop",) + SPECIALISTS
MODE_NAMES = {"executive": "Executive", "coder": "Coder", "architect": "Architect",
              "security": "Security", "researcher": "Researcher", "writer": "Writer"}


def test_nonstop_persona_is_general_spoken_and_grounded():
    p = system_prompt("nonstop")
    assert "NON-STOP" in p and "EVERYTHING" in p and "no topic limit" in p
    assert "no markdown" in p.lower()                 # spoken style
    assert "ACCURACY RULES" in p and "LANGUAGE RULES" in p
    assert "senior AI-SaaS software engineer" not in p  # not engineer-only


def test_nonstop_is_the_only_unrestricted_mode():
    assert UNRESTRICTED_PERSONA == "nonstop"
    assert "SCOPE:" not in system_prompt("nonstop")          # nothing to stay inside
    assert "never tell the person to switch modes" in system_prompt("nonstop")
    for mode in SPECIALISTS:
        assert "SCOPE:" in system_prompt(mode), mode


def test_every_mode_in_the_picker_exists_and_is_valid():
    for mode in PICKER:
        assert mode in PERSONAS
        schemas._check_persona(mode)


def test_legacy_modes_still_work_for_old_conversations():
    for mode in ("developer", "ccna_trainer"):
        assert mode in PERSONAS


def _own_text(mode):
    """The prompt minus the blocks every mode shares (identity, contracts, mode map)."""
    from src.agents import prompts
    p = system_prompt(mode)
    for shared in (prompts._IDENTITY, prompts._ENVIRONMENT, prompts._GROUNDING,
                   prompts._LANGUAGE, prompts._VOICE_HYGIENE, prompts._MODE_MAP):
        p = p.replace(shared, " ")
    return p


def test_each_mode_has_its_own_distinct_intent():
    own = {m: _own_text(m) for m in PICKER}
    words = {m: set(w.strip(".,;:()'\"").lower() for w in t.split()) for m, t in own.items()}
    for a, b in itertools.combinations(PICKER, 2):
        assert own[a] != own[b], (a, b)
        jaccard = len(words[a] & words[b]) / len(words[a] | words[b])
        assert jaccard < 0.45, (a, b, round(jaccard, 2))   # genuinely different briefs
        assert len(own[a]) > 400 and len(own[b]) > 400     # real briefs, not one-liners


@pytest.mark.parametrize("mode", SPECIALISTS)
def test_specialists_stay_in_their_field_and_hand_off_by_mode_name(mode):
    p = system_prompt(mode)
    assert f"in {MODE_NAMES[mode]} mode" in p
    assert "Stay inside it" in p
    others = [n for m, n in MODE_NAMES.items() if m != mode]
    # it knows the other modes by their picker names, and names at least two as hand-offs
    assert all(n in p for n in others + ["Non-stop"])
    handoffs = p.split("Typical hand-offs:")[1].split(".")[0]
    assert sum(n in handoffs for n in others + ["Non-stop"]) >= 2


@pytest.mark.parametrize("mode", PICKER + ("developer", "ccna_trainer"))
def test_every_mode_keeps_the_accuracy_and_bilingual_contracts(mode):
    p = system_prompt(mode)
    assert "ACCURACY RULES" in p and "LANGUAGE RULES" in p
    assert "AIDA" in p
    assert "Bengali" in p


def test_specialist_domains_do_not_leak_into_each_other():
    # The distinctive craft words appear only in the mode that owns them.
    assert "OWASP" in system_prompt("security") and "OWASP" not in system_prompt("writer")
    assert "subnetting" in system_prompt("security").lower()
    assert "pgvector" in system_prompt("coder") and "pgvector" not in system_prompt("security")
    assert "multi-tenancy" in system_prompt("architect") and "multi-tenancy" not in system_prompt("coder")
    assert "OKRs" in system_prompt("executive") and "OKRs" not in system_prompt("coder")
    assert "sub-questions" in system_prompt("researcher") and "sub-questions" not in system_prompt("writer")
    assert "release notes" in system_prompt("writer") and "release notes" not in system_prompt("researcher")


def test_security_mode_stays_defensive():
    p = system_prompt("security")
    assert "do NOT produce working exploits" in p or "do NOT" in p and "exploits" in p


def test_unknown_persona_falls_back_to_the_default():
    assert system_prompt("nope") == system_prompt("executive")


def test_nonstop_uses_the_general_chat_model():
    assert settings.model_for_persona("nonstop") == settings.chat_model
