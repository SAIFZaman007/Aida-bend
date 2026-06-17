"""Central configuration.

v2: migrated from SQLite (which was causing "database is locked" errors and
general sloppiness under bulk session load) to Postgres 18 + pgvector, and
added Redis 8 for chat-history caching / pub-sub (used for the SSE layer and
to keep token latency low). Also adds a second "coder" model slot so the
backend can route developer/architect personas to a code-specialized model
(e.g. qwen3-coder) while everything else keeps using the general chat model.

Everything is read once from the environment into a single `settings` object.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Server ---
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    cors_origins: str = (
        "http://localhost:5173,http://localhost:5174,"
        "http://127.0.0.1:5173,http://127.0.0.1:5174"
    )

    # --- Database: Postgres 18 + pgvector (required) ---
    # Example: postgresql+asyncpg://aida:password@localhost:5432/aida_db
    database_url: str = "postgresql+asyncpg://aida:aida@localhost:5432/aida_db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    # Dimensionality of the embedding model (nomic-embed-text = 768,
    # mxbai-embed-large = 1024, OpenAI text-embedding-3-small = 1536, etc.)
    embedding_dim: int = 768

    # --- Redis 8 (chat cache + pub/sub for SSE fan-out) ---
    redis_url: str = "redis://localhost:6379/0"
    redis_history_ttl: int = 60 * 60 * 6  # 6h rolling cache of recent turns
    redis_stream_channel_prefix: str = "aida:stream:"

    # --- Brain: "ollama" (free/local) or "anthropic" (Claude API) ---
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    # General-purpose chat model (executive, writer, researcher, etc.)
    chat_model: str = "llama3.2"
    # Code-specialized model used for developer / architect / coder personas.
    coder_model: str = "qwen3-coder"
    embed_model: str = "nomic-embed-text"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    # Optional: a separate Anthropic model for coding personas. Falls back to
    # anthropic_model if empty.
    anthropic_coder_model: str = ""
    anthropic_max_tokens: int = 2048

    # --- Voice: speech-to-text ---
    whisper_model: str = "base"
    # Use a faster/smaller model + fewer beams to cut STT latency. "tiny" or
    # "base" with vad_filter is a good balance for desktop use.
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1

    # --- Voice: optional natural TTS via Piper (https://github.com/rhasspy/piper) ---
    piper_enabled: bool = True
    piper_voice_female: str = "./voices/en_US-amy-medium.onnx"
    piper_voice_male: str = "./voices/en_US-ryan-medium.onnx"
    # Sample rate marks are computed at — Piper voices default to 22050Hz;
    # used to translate audio-frame timing into wall-clock ms for the
    # real-time reading-progress highlighter.
    piper_sample_rate: int = 22050

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_anthropic(self) -> bool:
        return self.llm_provider.strip().lower() == "anthropic"

    def model_for_persona(self, persona: str) -> str:
        """Pick the chat vs. coder model based on persona."""
        coder_personas = {"developer", "architect", "coder"}
        if persona in coder_personas:
            if self.is_anthropic:
                return self.anthropic_coder_model or self.anthropic_model
            return self.coder_model
        return self.anthropic_model if self.is_anthropic else self.chat_model


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
