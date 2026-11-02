"""Switching language mid-conversation: the reply must follow the LATEST message."""
from types import SimpleNamespace

from src.chat import service


def _sys(user_text, history=None):
    conv = SimpleNamespace(persona="executive")
    msgs = service.build_messages(conv, user_text, history or [], [], "executive")
    return msgs[0]["content"]


def test_english_message_after_bengali_history_is_told_to_answer_in_english():
    history = [{"role": "user", "content": "আমি ভালো আছি"}, {"role": "assistant", "content": "ভালো লাগলো।"}]
    prompt = _sys("What is a load balancer?", history)
    assert service._EN_NOTE in prompt
    assert service._BN_NOTE not in prompt


def test_bengali_message_still_gets_the_bengali_note_only():
    prompt = _sys("লোড ব্যালেন্সার কী?")
    assert service._BN_NOTE in prompt
    assert service._EN_NOTE not in prompt


def test_mixed_or_wordless_messages_add_no_language_note():
    from src.core.text import detect_language

    for text in ("12345 ???", "আমার Docker container কাজ করছে না"):
        assert detect_language(text) in {"unknown", "mixed"}
        prompt = _sys(text)
        assert service._EN_NOTE not in prompt
        assert service._BN_NOTE not in prompt
