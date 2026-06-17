"""The 'brain' — one interface over two providers, with dual-model routing.

  llm_provider = "ollama"     -> free, local (default)
  llm_provider = "anthropic"  -> Claude API

v2 additions:
 • DUAL MODEL: stream_chat() now takes an optional `persona`. Developer /
   architect / coder personas route to settings.coder_model (qwen3-coder by
   default); everything else uses settings.chat_model. This lets you run a
   strong general model for conversation and a code-specialized model for
   engineering work, side by side, with zero changes to callers that don't
   care (persona defaults to None -> general model).
 • Same pooled httpx.AsyncClient, same fast-connect/unbounded-read timeouts,
   same keep_alive — these were already correct and are preserved.
"""
import json
from collections.abc import AsyncGenerator

import httpx

from src.core.config import settings

KEEP_ALIVE = "30m"

# Reused across every request. connect fails fast; read is unbounded for streams.
_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=10.0),
    limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
)


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    return "\n\n".join(system_parts), turns


class LLM:
    def __init__(self) -> None:
        self.base = settings.ollama_base_url.rstrip("/")

    async def stream_chat(
        self, messages: list[dict], persona: str | None = None
    ) -> AsyncGenerator[str, None]:
        if persona:
            model = settings.model_for_persona(persona)
        else:
            model = settings.anthropic_model if settings.is_anthropic else settings.chat_model

        if settings.is_anthropic:
            async for piece in self._anthropic_stream(messages, model):
                yield piece
        else:
            async for piece in self._ollama_stream(messages, model):
                yield piece

    async def _ollama_stream(
        self, messages: list[dict], model: str
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
        }
        async with _client.stream("POST", f"{self.base}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break

    async def _anthropic_stream(
        self, messages: list[dict], model: str
    ) -> AsyncGenerator[str, None]:
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty. "
                "Add your key from console.anthropic.com to backend/.env."
            )
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        system, turns = _split_system(messages)
        async with client.messages.stream(
            model=model,
            max_tokens=settings.anthropic_max_tokens,
            system=system or "You are a helpful assistant.",
            messages=turns,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def complete(self, messages: list[dict], persona: str | None = None) -> str:
        out: list[str] = []
        async for piece in self.stream_chat(messages, persona=persona):
            out.append(piece)
        return "".join(out)

    async def embed(self, text: str) -> list[float] | None:
        """Embed via Ollama. Returns None (never raises) on any problem."""
        try:
            resp = await _client.post(
                f"{self.base}/api/embeddings",
                json={"model": settings.embed_model, "prompt": text, "keep_alive": KEEP_ALIVE},
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json().get("embedding")
        except Exception:
            return None

    async def aclose(self) -> None:
        await _client.aclose()


llm = LLM()
