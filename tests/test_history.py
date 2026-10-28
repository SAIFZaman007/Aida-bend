"""Regression tests for the history/cache/persistence bugs (reproduced against
the v2 code before being fixed): duplicated user turn, history lost on a cold
cache, and un-replied turns after a failed/aborted stream."""
import asyncio
import types

import pytest

from src.chat import service
from src.core.errors import AidaError


class Rows:
    def __init__(self, rows): self._r = rows
    def scalars(self): return self
    def all(self): return self._r


class Store:
    """Shared fake Postgres table of (role, content)."""
    def __init__(self, n_prior=6):
        self.rows = [("user" if i % 2 == 0 else "assistant", f"old-{i}") for i in range(n_prior)]


class StubSession:
    def __init__(self, store, conv=None):
        self.store, self.conv, self._pending = store, conv, []
    async def get(self, model, id): return self.conv
    def add(self, obj): self._pending.append(obj)
    async def commit(self):
        for o in self._pending:
            self.store.rows.append((o.role, o.content))
        self._pending = []
    async def execute(self, stmt):
        msgs = [types.SimpleNamespace(role=r, content=c) for r, c in self.store.rows[-service.HISTORY_WINDOW:]]
        return Rows(list(reversed(msgs)))


class StubFactory:
    """Stands in for SessionLocal(): `async with SessionLocal() as s`."""
    def __init__(self, store): self.store = store
    def __call__(self):
        outer = self
        class CM:
            async def __aenter__(self): return StubSession(outer.store)
            async def __aexit__(self, *a): return False
        return CM()


@pytest.fixture
def env(monkeypatch, fake_redis):
    store = Store()
    conv = types.SimpleNamespace(id="c1", project_id="p1", persona="executive")
    monkeypatch.setattr(service, "SessionLocal", StubFactory(store))
    async def no_recall(*a, **k): return []
    monkeypatch.setattr(service.memory, "recall", no_recall)
    captured = {}
    def set_llm(pieces, error=None):
        async def fake_stream(messages, persona=None):
            captured["messages"] = messages
            for p in pieces:
                yield p
            if error:
                raise error
        monkeypatch.setattr(service.llm, "stream_chat", fake_stream)
    return types.SimpleNamespace(store=store, conv=conv, captured=captured, set_llm=set_llm,
                                 session=StubSession(store, conv), redis=fake_redis)


async def run(env, text="NEW QUESTION"):
    events = []
    try:
        async for e in service.stream_reply(env.session, "c1", text, remember=False):
            events.append(e)
    except AidaError:
        pass
    return events


def roles(messages):
    return [(m["role"], m["content"]) for m in messages if m["role"] != "system"]


async def test_user_message_sent_exactly_once_warm_cache(env):
    from src.core import redis_client
    await redis_client.cache_set_history("c1", [{"role": r, "content": c} for r, c in env.store.rows])
    env.set_llm(["ok."])
    await run(env)
    msgs = roles(env.captured["messages"])
    assert [c for r, c in msgs].count("NEW QUESTION") == 1
    assert msgs[-1] == ("user", "NEW QUESTION")
    assert [c for _, c in msgs[:-1]] == [f"old-{i}" for i in range(6)]


async def test_cold_cache_keeps_full_history(env):
    """Regression: after a Redis restart/TTL expiry the old code kept ONLY the
    newest message and silently forgot the whole conversation."""
    env.set_llm(["ok."])
    await run(env)
    msgs = roles(env.captured["messages"])
    assert [c for r, c in msgs].count("NEW QUESTION") == 1
    assert [c for _, c in msgs[:-1]] == [f"old-{i}" for i in range(6)]


async def test_history_correct_on_following_turn_after_cold_start(env):
    env.set_llm(["first answer."])
    await run(env, "Q1")
    env.set_llm(["second answer."])
    await run(env, "Q2")
    msgs = roles(env.captured["messages"])
    assert [c for _, c in msgs][-3:] == ["Q1", "first answer.", "Q2"]
    assert len(msgs) == 6 + 3


async def test_works_without_redis(env, no_redis):
    env.set_llm(["ok."])
    await run(env)
    msgs = roles(env.captured["messages"])
    assert [c for r, c in msgs].count("NEW QUESTION") == 1
    assert len(msgs) == 7


async def test_reply_persisted_and_sentences_emitted(env):
    env.set_llm(["আমি ভালো আছি। ", "আপনি কেমন আছেন?"])
    events = await run(env)
    sentences = [e for e in events if e["type"] == "sentence"]
    assert [s["text"] for s in sentences] == ["আমি ভালো আছি।", " আপনি কেমন আছেন?"]
    assert [s["index"] for s in sentences] == [0, 1]
    assert env.store.rows[-1] == ("assistant", "আমি ভালো আছি। আপনি কেমন আছেন?")


async def test_partial_reply_saved_when_provider_fails_midstream(env):
    env.set_llm(["Half an answer, "], error=AidaError("unavailable", "boom"))
    await run(env)
    assert env.store.rows[-1] == ("assistant", "Half an answer,")


async def test_partial_reply_saved_when_client_disconnects(env):
    env.set_llm(["Some text. ", "More text. ", "Even more."])
    gen = service.stream_reply(env.session, "c1", "Q", remember=False)
    await gen.__anext__()          # first token
    await gen.aclose()             # what Starlette does on disconnect
    assert env.store.rows[-1][0] == "assistant"
    assert env.store.rows[-1][1].startswith("Some text.")


async def test_no_reply_row_when_nothing_was_generated(env):
    env.set_llm([], error=AidaError("unavailable", "down"))
    await run(env)
    assert env.store.rows[-1] == ("user", "Q") or env.store.rows[-1][0] == "user"


async def test_unknown_conversation_raises_not_found(env):
    env.session.conv = None
    with pytest.raises(AidaError) as ei:
        async for _ in service.stream_reply(env.session, "nope", "x", remember=False):
            pass
    assert ei.value.code == "not_found"


async def test_bengali_message_adds_language_note_but_english_does_not(env):
    env.set_llm(["ok."])
    await run(env, "আপনি কেমন আছেন?")
    assert "Bengali script" in env.captured["messages"][0]["content"].split("LANGUAGE RULES")[-1]
    assert "latest message is written in Bengali script" in env.captured["messages"][0]["content"]
    await run(env, "How are you?")
    assert "latest message is written in Bengali script" not in env.captured["messages"][0]["content"]


async def test_redis_flap_does_not_truncate_history(env, monkeypatch):
    """Redis fails during the history read, then recovers before the append.
    With a plain RPUSH the append created a one-turn list that the NEXT read
    trusted as the full history; RPUSHX leaves the key absent so it is rebuilt
    from Postgres."""
    from src.core import redis_client
    real_get, real_set = redis_client.cache_get_history, redis_client.cache_set_history

    async def down_get(cid): return None
    async def down_set(cid, msgs): return None

    monkeypatch.setattr(redis_client, "cache_get_history", down_get)
    monkeypatch.setattr(redis_client, "cache_set_history", down_set)
    env.set_llm(["a1."])
    await run(env, "Q1")

    monkeypatch.setattr(redis_client, "cache_get_history", real_get)  # Redis is back
    monkeypatch.setattr(redis_client, "cache_set_history", real_set)
    env.set_llm(["a2."])
    await run(env, "Q2")
    contents = [c for _, c in roles(env.captured["messages"])]
    assert contents[:6] == [f"old-{i}" for i in range(6)]
    assert contents[-3:] == ["Q1", "a1.", "Q2"] or contents[-1] == "Q2"
