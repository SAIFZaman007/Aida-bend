"""v2.5 chat behaviour: concurrent memory recall, per-mode memory, mode in the request."""
import asyncio
import time

import pytest

from src.chat import service
from test_history import Store, StubFactory, StubSession, env, roles, run  # noqa: F401  (fixtures)


def test_trivial_turns_skip_the_embedding_round_trip():
    for t in ("hi", "ok", "thanks", "yes please", "   ", "হ্যালো"):
        assert not service.wants_recall(t), t
    for t in ("how do I add an index to the messages table",
              "আপনি কেমন আছেন আজকে",
              "What is the capital of Bangladesh?"):
        assert service.wants_recall(t), t


async def test_recall_runs_while_history_loads_and_is_scoped_to_the_mode(env, monkeypatch):
    seen = {}

    async def slow_recall(session, project_id, query, k=None, persona=None):
        seen.update(project=project_id, query=query, persona=persona, session=session)
        await asyncio.sleep(0.05)
        return ["earlier note about indexes"]

    monkeypatch.setattr(service.memory, "recall", slow_recall)
    env.conv.persona = "coder"
    env.set_llm(["ok."])
    await run(env, "how do I add an index to the messages table")
    sys_msgs = [m["content"] for m in env.captured["messages"] if m["role"] == "system"]
    assert any("earlier note about indexes" in m for m in sys_msgs)
    assert seen["persona"] == "coder" and seen["project"] == "p1"
    # its OWN session: the request session is busy saving the user turn at the same time
    assert seen["session"] is not env.session


async def test_a_slow_recall_never_delays_the_model_beyond_the_budget(env, monkeypatch):
    monkeypatch.setattr(service.settings, "memory_recall_timeout", 0.2)

    async def stuck(*a, **k):
        await asyncio.sleep(30)

    monkeypatch.setattr(service.memory, "recall", stuck)
    env.set_llm(["ok."])
    t0 = time.perf_counter()
    await run(env, "how do I add an index to the messages table")
    assert time.perf_counter() - t0 < 1.0
    assert not [m for m in env.captured["messages"] if "Notes from earlier" in m["content"]]


async def test_recall_failure_is_harmless(env, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("pgvector down")

    monkeypatch.setattr(service.memory, "recall", boom)
    env.set_llm(["fine."])
    events = await run(env, "how do I add an index to the messages table")
    assert [e for e in events if e["type"] == "sentence"]


async def test_greeting_does_not_call_recall_at_all(env, monkeypatch):
    called = []

    async def spy(*a, **k):
        called.append(1)
        return []

    monkeypatch.setattr(service.memory, "recall", spy)
    env.set_llm(["Hello!"])
    await run(env, "hi")
    assert not called


async def test_the_mode_in_the_request_wins_and_is_stored(env):
    env.conv.persona = "executive"
    env.set_llm(["ok."])
    async for _ in service.stream_reply(env.session, "c1", "NEW QUESTION", remember=False, persona="security"):
        pass
    assert env.conv.persona == "security"
    assert "security mode" in env.captured["messages"][0]["content"]


async def test_unknown_mode_in_the_request_is_ignored(env):
    env.conv.persona = "writer"
    env.set_llm(["ok."])
    async for _ in service.stream_reply(env.session, "c1", "NEW QUESTION", remember=False, persona="hacker"):
        pass
    assert env.conv.persona == "writer"
    assert "writer mode" in env.captured["messages"][0]["content"]


async def test_each_mode_gets_its_own_brief(env):
    briefs = {}
    for mode in ("coder", "writer", "nonstop"):
        env.conv.persona = mode
        env.set_llm(["ok."])
        await run(env, "tell me something useful about this")
        briefs[mode] = env.captured["messages"][0]["content"]
    assert len(set(briefs.values())) == 3


async def test_exchange_is_stored_under_the_mode_it_was_said_in(env, monkeypatch):
    stored = []

    async def spy(session, project_id, text, source="note"):
        stored.append((project_id, source))

    monkeypatch.setattr(service.memory, "remember", spy)
    env.conv.persona = "architect"
    env.set_llm(["A reply. "])
    async for _ in service.stream_reply(env.session, "c1", "NEW QUESTION", remember=True):
        pass
    await asyncio.sleep(0.05)
    assert stored == [("p1", "message:architect")]


async def test_first_clause_reaches_the_client_before_the_sentence_ends(env):
    env.set_llm(["Well, after looking at the numbers you sent, the most likely cause is a missing index, which ",
                 "makes every lookup scan all rows. Add it."])
    events = await run(env, "NEW QUESTION")
    sentences = [e["text"] for e in events if e["type"] == "sentence"]
    # the clause is released while the sentence is still being written
    assert sentences[0] == "Well, after looking at the numbers you sent," and len(sentences) >= 3
    assert "".join(sentences) == "Well, after looking at the numbers you sent, the most likely cause is a missing index, which makes every lookup scan all rows. Add it."


# ---------------------------------------------------------------- recall pacing (v2.5.1)
def test_pacer_pauses_after_repeated_misses_then_probes_once():
    pacer = service.RecallPacer()
    assert pacer.active()
    pacer.miss(); pacer.miss()
    assert pacer.active()
    pacer.miss()
    assert not pacer.active()
    pacer.paused_until = time.monotonic() - 1          # the pause is over: one probe
    assert pacer.active()
    pacer.miss()                                        # still too slow: paused again at once
    assert not pacer.active()
    pacer.paused_until = 0
    pacer.hit()
    pacer.miss(); pacer.miss()
    assert pacer.active()                               # a hit resets the count


async def test_slow_recall_is_paused_so_later_turns_do_not_wait_at_all(env, monkeypatch):
    monkeypatch.setattr(service.settings, "memory_recall_timeout", 0.05)
    started = []

    async def stuck(*a, **k):
        started.append(1)
        await asyncio.sleep(30)

    monkeypatch.setattr(service.memory, "recall", stuck)
    q = "how do I add an index to the messages table"
    for _ in range(service.RecallPacer.MISS_LIMIT):
        env.set_llm(["ok."])
        await run(env, q)
    assert len(started) == service.RecallPacer.MISS_LIMIT and not service.recall_pacer.active()
    env.set_llm(["ok."])
    t0 = time.perf_counter()
    await run(env, q)
    assert len(started) == service.RecallPacer.MISS_LIMIT, "no lookup (and no embedding call) while paused"
    assert time.perf_counter() - t0 < 0.5


async def test_a_late_recall_is_logged_not_discarded_silently(env, monkeypatch, caplog):
    caplog.set_level("INFO", logger="aida.chat")
    monkeypatch.setattr(service.settings, "memory_recall_timeout", 0.05)

    async def late(*a, **k):
        await asyncio.sleep(0.2)
        return ["a note"]

    monkeypatch.setattr(service.memory, "recall", late)
    env.set_llm(["ok."])
    await run(env, "how do I add an index to the messages table")
    assert not [m for m in env.captured["messages"] if "a note" in m["content"]]
    await asyncio.sleep(0.4)
    assert any("recall was late" in r.getMessage() and "1 note" in r.getMessage() for r in caplog.records)
