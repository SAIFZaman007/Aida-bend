import pytest
from fastapi import FastAPI, Depends
from httpx import ASGITransport, AsyncClient

from src.core import ratelimit
from src.core.config import settings


def make_app():
    app = FastAPI()

    @app.get("/x", dependencies=[Depends(ratelimit.limit("t", "rate_chat_per_min", "daily_chat_cap"))])
    async def x():
        return {"ok": True}

    return app


async def hit_n(app, n, headers=None):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return [await c.get("/x", headers=headers or {}) for _ in range(n)]


@pytest.fixture(autouse=True)
def cfg(monkeypatch):
    monkeypatch.setattr(settings, "rate_chat_per_min", 3)
    monkeypatch.setattr(settings, "daily_chat_cap", 0)
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    ratelimit._memory._d.clear()


async def test_per_client_limit_with_redis(fake_redis):
    rs = await hit_n(make_app(), 5)
    assert [r.status_code for r in rs] == [200, 200, 200, 429, 429]
    body = rs[3].json()["detail"]
    assert body["code"] == "rate_limited" and rs[3].headers["retry-after"].isdigit()


async def test_falls_back_to_memory_when_redis_is_down(no_redis):
    rs = await hit_n(make_app(), 4)
    assert [r.status_code for r in rs] == [200, 200, 200, 429]


async def test_error_message_is_bengali_when_client_says_so(fake_redis):
    rs = await hit_n(make_app(), 4, headers={"X-Lang": "bn"})
    assert "অনুগ্রহ করে" in rs[3].json()["detail"]["message"]


async def test_daily_cap(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_chat_per_min", 0)   # disable per-minute
    monkeypatch.setattr(settings, "daily_chat_cap", 2)
    rs = await hit_n(make_app(), 4)
    assert [r.status_code for r in rs] == [200, 200, 429, 429]
    assert rs[2].json()["detail"]["code"] == "daily_cap"


async def test_can_be_disabled(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    rs = await hit_n(make_app(), 10)
    assert all(r.status_code == 200 for r in rs)


async def test_key_without_ttl_is_repaired(fake_redis):
    """A crash between INCR and EXPIRE must not block a client forever."""
    key = "aida:rl:t:127.0.0.1"
    await fake_redis.set(key, 99)           # no TTL
    count, ttl = await ratelimit.hit(key, 60)
    assert ttl > 0 and await fake_redis.ttl(key) > 0
