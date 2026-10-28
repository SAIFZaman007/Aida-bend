"""Test environment: set config BEFORE any src.* import; never touch the network."""
import os
import sys

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://t:t@localhost:5432/t")
os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["LLM_PROVIDER"] = "gemini"
os.environ.setdefault("RATE_LIMIT_ENABLED", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fakeredis  # noqa: E402
import pytest  # noqa: E402

from src.core import redis_client  # noqa: E402


@pytest.fixture
def fake_redis():
    """Replace the Redis client with an in-process fake."""
    old = redis_client._redis
    redis_client._redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield redis_client._redis
    redis_client._redis = old


@pytest.fixture
def no_redis(monkeypatch):
    """Simulate Redis being down (exercises every graceful-degradation path)."""
    def boom():
        raise ConnectionError("redis down")
    monkeypatch.setattr(redis_client, "get_redis", boom)
