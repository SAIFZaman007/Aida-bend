"""The 'brain' — chat + embeddings behind one interface.

  llm_provider = "gemini"     -> Google Gemini (default; chat + embeddings)
  llm_provider = "anthropic"  -> Claude for CHAT only (embeddings stay Gemini)

The public surface is unchanged from v2 (`llm.stream_chat`, `llm.complete`,
`llm.embed`, `llm.aclose`), so chat/service.py and memory/service.py need no
knowledge of the provider. Ollama was removed.

Reliability behaviour (all configurable in core/config.py):
  • First-token timeout and idle timeout — a stalled stream fails fast with a
    friendly error instead of hanging the UI.
  • Retry with exponential backoff + jitter on transient errors (429/5xx/
    timeouts) — ONLY before the first token has been produced, so a retry can
    never duplicate or splice text the user already saw.
  • Consecutive same-role turns are merged and leading assistant turns dropped,
    because providers reject or mishandle malformed turn order.
  • Provider exceptions become `AidaError` codes — raw text never reaches users.
"""
import asyncio
import logging
import math
import random
from collections.abc import AsyncGenerator

from google.genai import types

from src.core import gemini
from src.core.config import settings
from src.core.errors import AidaError

log = logging.getLogger("aida.llm")

_EMBED_MAX_CHARS = 8000


def normalize_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split out system text; return alternating user/assistant turns that
    start with a user turn (consecutive same-role turns are merged)."""
    system_parts = [m["content"] for m in messages if m["role"] == "system" and m["content"]]
    turns: list[dict] = []
    for m in messages:
        if m["role"] not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if turns and turns[-1]["role"] == m["role"]:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": m["role"], "content": content})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return "\n\n".join(system_parts), turns


def _chunk_text(chunk) -> str:
    try:
        return chunk.text or ""
    except Exception:
        return ""


def _blocked(chunk) -> bool:
    """True if a stream chunk reports a safety/policy block."""
    try:
        fb = getattr(chunk, "prompt_feedback", None)
        if fb is not None and getattr(fb, "block_reason", None):
            return True
        for cand in getattr(chunk, "candidates", None) or []:
            reason = str(getattr(cand, "finish_reason", "") or "").upper()
            if any(k in reason for k in ("SAFETY", "PROHIBITED", "BLOCKLIST", "SPII", "RECITATION")):
                return True
    except Exception:
        pass
    return False


class LLM:
    # ------------------------------------------------------------- chat
    async def stream_chat(
        self, messages: list[dict], persona: str | None = None
    ) -> AsyncGenerator[str, None]:
        model = settings.model_for_persona(persona) if persona else settings.chat_model
        attempt = 0
        while True:
            produced = False
            try:
                async for piece in self._stream_once(messages, model):
                    produced = True
                    yield piece
                return
            except Exception as exc:  # BaseException (cancel/close) passes through
                err = gemini.map_exception(exc)
                if err.code == "internal":
                    log.exception("chat stream failed (model=%s)", model)
                else:
                    log.warning("chat stream error code=%s model=%s detail=%s",
                                err.code, model, err.detail[:300])
            if produced or not err.retryable or attempt >= settings.llm_max_retries:
                raise err
            attempt += 1
            delay = min(8.0, 0.6 * (2 ** (attempt - 1))) + random.uniform(0, 0.4)
            log.info("retrying chat (attempt %d) in %.1fs", attempt, delay)
            await asyncio.sleep(delay)

    def _stream_once(self, messages: list[dict], model: str) -> AsyncGenerator[str, None]:
        if settings.is_anthropic:
            return self._anthropic_stream(messages, model)
        return self._gemini_stream(messages, model)

    async def _gemini_stream(
        self, messages: list[dict], model: str
    ) -> AsyncGenerator[str, None]:
        client = gemini.get_client()
        system, turns = normalize_turns(messages)
        if not turns:
            raise AidaError("internal", "no user turn to send")
        contents = [
            types.Content(
                role="user" if t["role"] == "user" else "model",
                parts=[types.Part(text=t["content"])],
            )
            for t in turns
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=settings.gemini_max_output_tokens,
            temperature=settings.gemini_temperature,
            thinking_config=gemini.thinking_config(settings.gemini_thinking_level),
        )
        stream = await client.aio.models.generate_content_stream(
            model=model, contents=contents, config=cfg
        )
        it = stream.__aiter__()
        got_text = False
        saw_block = False
        try:
            while True:
                timeout = settings.llm_idle_timeout if got_text else settings.llm_first_token_timeout
                try:
                    chunk = await asyncio.wait_for(it.__anext__(), timeout)
                except StopAsyncIteration:
                    break
                saw_block = saw_block or _blocked(chunk)
                text = _chunk_text(chunk)
                if text:
                    got_text = True
                    yield text
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:
                    pass
        if not got_text:
            # Nothing produced: blocked content is final; an empty reply is a
            # transient provider hiccup worth a retry.
            raise AidaError("blocked" if saw_block else "unavailable", "empty response")

    async def _anthropic_stream(
        self, messages: list[dict], model: str
    ) -> AsyncGenerator[str, None]:
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise AidaError("not_configured", "ANTHROPIC_API_KEY is empty")
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        system, turns = normalize_turns(messages)
        async with client.messages.stream(
            model=model,
            max_tokens=settings.anthropic_max_tokens,
            system=system or "You are a helpful assistant.",
            messages=turns,
        ) as stream:
            got = False
            async for text in stream.text_stream:
                got = True
                yield text
        if not got:
            raise AidaError("unavailable", "empty response")

    async def complete(self, messages: list[dict], persona: str | None = None) -> str:
        out: list[str] = []
        async for piece in self.stream_chat(messages, persona=persona):
            out.append(piece)
        return "".join(out)

    # -------------------------------------------------------- embeddings
    async def embed(self, text: str, kind: str = "document") -> list[float] | None:
        """Embed via Gemini. Returns None (never raises) on any problem.

        kind: "document" (stored memory) or "query" (recall lookup).
        """
        text = (text or "").strip()
        if not text:
            return None
        prefix = settings.embed_query_prefix if kind == "query" else settings.embed_doc_prefix
        try:
            client = gemini.get_client()
            resp = await asyncio.wait_for(
                client.aio.models.embed_content(
                    model=settings.gemini_embed_model,
                    contents=prefix + text[:_EMBED_MAX_CHARS],
                    config=types.EmbedContentConfig(
                        output_dimensionality=settings.embedding_dim
                    ),
                ),
                settings.embed_timeout,
            )
            vec = [float(x) for x in resp.embeddings[0].values]
        except Exception as exc:
            err = gemini.map_exception(exc)
            log.warning("embedding failed code=%s detail=%s", err.code, err.detail[:200])
            return None
        if len(vec) != settings.embedding_dim:
            log.error(
                "embedding dimension mismatch: got %d, EMBEDDING_DIM=%d — "
                "refusing to store/query a vector the DB column cannot hold",
                len(vec), settings.embedding_dim,
            )
            return None
        # Truncated (Matryoshka) vectors are not unit length; normalize.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def aclose(self) -> None:
        await gemini.aclose()


llm = LLM()
