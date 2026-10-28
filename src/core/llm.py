"""The 'brain' — chat + embeddings behind one interface.

  llm_provider = "gemini"     -> Google Gemini (default; chat + embeddings)
  llm_provider = "anthropic"  -> Claude for CHAT only (embeddings stay Gemini)

The public surface is unchanged from v2 (`llm.stream_chat`, `llm.complete`,
`llm.embed`, `llm.aclose`), so chat/service.py and memory/service.py need no
knowledge of the provider. Ollama was removed.

Reliability behaviour (all configurable in core/config.py):
  • First-token timeout and idle timeout — a stalled stream fails over (to a
    non-streaming call, then to GEMINI_FALLBACK_MODEL) instead of hanging the UI.
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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _Route:
    model: str
    mode: str  # "stream" (normal) | "once" (non-streaming fallback)


class LLM:
    # ------------------------------------------------------------- chat
    async def stream_chat(
        self, messages: list[dict], persona: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Yield reply text. Resilience plan, in order:

          1. stream from the primary model (retry 429/5xx a couple of times);
          2. if the stream STALLS (no first token), ask the same model without
             streaming — a different code path on the provider side;
          3. fail over to GEMINI_FALLBACK_MODEL if one is configured.

        Failover only happens before the first token, so text the user already
        saw is never repeated or spliced.
        """
        model = settings.model_for_persona(persona) if persona else settings.chat_model
        plan: list[_Route] = [_Route(model, "stream")]
        fallback = settings.fallback_model
        if fallback and fallback != model:
            plan.append(_Route(fallback, "stream"))

        last: AidaError | None = None
        while plan:
            route = plan.pop(0)
            retries = 0
            while True:
                produced = False
                try:
                    async for piece in self._run_route(messages, route):
                        produced = True
                        yield piece
                    return
                except Exception as exc:  # BaseException (cancel/close) passes through
                    if (not produced and gemini.is_thinking_rejected(exc)
                            and gemini.mark_no_thinking(route.model)):
                        continue  # self-heal: same route, no thinking_config
                    err = gemini.map_exception(exc)
                    if err.code == "internal":
                        log.exception("chat failed (model=%s mode=%s)", route.model, route.mode)
                    else:
                        log.warning("chat %s error code=%s model=%s detail=%s",
                                    route.mode, err.code, route.model, err.detail[:300])
                if produced or not err.retryable:
                    raise err
                last = err
                if err.code == "timeout":
                    # A stall will not fix itself by waiting: change approach now.
                    if route.mode == "stream" and not settings.is_anthropic:
                        plan.insert(0, _Route(route.model, "once"))
                    break
                if retries >= settings.llm_max_retries:
                    break
                retries += 1
                delay = min(8.0, 0.6 * (2 ** (retries - 1))) + random.uniform(0, 0.4)
                log.info("retrying chat (attempt %d) in %.1fs", retries, delay)
                await asyncio.sleep(delay)
            if plan:
                log.warning("failing over: %s/%s -> %s/%s",
                            route.model, route.mode, plan[0].model, plan[0].mode)
        raise last or AidaError("unavailable", "no route succeeded")

    def _run_route(self, messages: list[dict], route: _Route) -> AsyncGenerator[str, None]:
        if settings.is_anthropic:
            return self._anthropic_stream(messages, route.model)
        if route.mode == "once":
            return self._gemini_once(messages, route.model)
        return self._gemini_stream(messages, route.model)

    @staticmethod
    def _gemini_request(messages: list[dict], model: str):
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
            thinking_config=gemini.thinking_config_for(model, settings.gemini_thinking_level),
            # No tools are declared; switching AFC off removes needless SDK logic.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return contents, cfg

    async def _gemini_stream(
        self, messages: list[dict], model: str
    ) -> AsyncGenerator[str, None]:
        client = gemini.get_client()
        contents, cfg = self._gemini_request(messages, model)
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

    async def _gemini_once(
        self, messages: list[dict], model: str
    ) -> AsyncGenerator[str, None]:
        """Non-streaming fallback: one request, whole reply as a single piece."""
        client = gemini.get_client()
        contents, cfg = self._gemini_request(messages, model)
        resp = await asyncio.wait_for(
            client.aio.models.generate_content(model=model, contents=contents, config=cfg),
            settings.llm_once_timeout,
        )
        text = _chunk_text(resp)
        if not text:
            raise AidaError("blocked" if _blocked(resp) else "unavailable", "empty response")
        yield text

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
