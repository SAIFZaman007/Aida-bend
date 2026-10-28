"""Central configuration.

v2.1 — cloud brain. The local Ollama/Qwen stack and the local Whisper/Piper
voice stack were replaced by Google Gemini (chat, embeddings, speech-to-text,
text-to-speech), chosen because Gemini TTS documents Bengali ("Bangla")
support. The Anthropic chat adapter is kept as an optional alternative for
chat only.

Everything is read once from the environment into a single `settings` object.
Model IDs are env-driven on purpose: providers rename/retire models, and a
redeploy with a new env var must be enough to follow them.
"""
from functools import lru_cache

from pydantic import field_validator
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

    # --- Database: Postgres + pgvector (required) ---
    database_url: str = "postgresql+asyncpg://aida:aida@localhost:5432/aida_db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    # Must match the stored vectors. Both Gemini embedding models can emit 768
    # (Matryoshka truncation), so the existing vector(768) column is unchanged.
    # NOTE: pgvector's HNSW index on `vector` supports at most 2000 dimensions.
    embedding_dim: int = 768

    # --- Redis (chat history cache, rate limiting) ---
    redis_url: str = "redis://localhost:6379/0"
    redis_history_ttl: int = 60 * 60 * 6
    redis_stream_channel_prefix: str = "aida:stream:"

    # --- Brain ---
    # "gemini" (default) | "anthropic" (chat only; embeddings/voice stay Gemini)
    llm_provider: str = "gemini"

    gemini_api_key: str = ""
    # General-purpose chat model and optional code-specialised model for the
    # developer / architect / coder personas (empty = same as chat model).
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_coder_model: str = ""
    # Gemini 3 "thinking": low = fast conversational replies. Empty = API default.
    gemini_thinking_level: str = "low"
    # Google recommends leaving temperature at its default (1.0) for Gemini 3
    # models — lowering it can cause looping/degraded output. Leave empty unless
    # you have measured a benefit with the eval suite.
    gemini_temperature: float | None = None
    gemini_max_output_tokens: int = 2048

    gemini_embed_model: str = "gemini-embedding-2"
    # Optional text prefixes (some embedding models want task prefixes). Empty
    # by default: no format is assumed that the provider has not documented.
    embed_doc_prefix: str = ""
    embed_query_prefix: str = ""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_coder_model: str = ""
    anthropic_max_tokens: int = 2048

    # --- Reliability ---
    llm_first_token_timeout: float = 25.0  # seconds until the first token
    llm_idle_timeout: float = 30.0         # max silence between tokens
    llm_max_retries: int = 2               # transient errors, only pre-first-token
    embed_timeout: float = 10.0
    sse_heartbeat_seconds: float = 15.0

    # --- Voice: speech-to-text (Gemini audio understanding) ---
    gemini_stt_model: str = "gemini-3.8-flash"
    gemini_stt_thinking_level: str = "minimal"
    max_audio_bytes: int = 10 * 1024 * 1024

    # --- Voice: text-to-speech (Gemini TTS; auto-detects Bangla/English) ---
    gemini_tts_model: str = "gemini-3.8-flash-tts"
    tts_voice_female: str = "Kore"
    tts_voice_male: str = "Charon"
    # Optional per-language overrides (e.g. a Bangla voice from the Extended
    # Voice Library — see scripts/list_voices.py). Empty = use the above.
    tts_voice_female_bn: str = ""
    tts_voice_male_bn: str = ""
    tts_style: str = ""  # optional speech_metadata style, e.g. "calm and warm"
    tts_sample_rate: int = 24000
    tts_concurrency: int = 4
    max_tts_chars: int = 1500

    # --- Memory (long-term recall) ---
    memory_recall_k: int = 3
    memory_recall_timeout: float = 1.5     # seconds (cloud embedding needs >0.5s)
    # Cosine distance cut-off: chunks farther than this are NOT injected. Tune
    # with scripts/smoke_test.py output; too high re-introduces irrelevant
    # context, too low disables memory.
    memory_max_distance: float = 0.45

    # --- Abuse protection (the API has no login by design) ---
    rate_limit_enabled: bool = True
    rate_chat_per_min: int = 20
    rate_stt_per_min: int = 30
    rate_tts_per_min: int = 120     # one request per spoken sentence
    rate_crud_per_min: int = 120
    # Global per-day ceilings that cap worst-case API spend. 0 = unlimited.
    daily_chat_cap: int = 3000
    daily_stt_cap: int = 3000
    daily_tts_cap: int = 30000
    max_chat_chars: int = 8000

    @field_validator("llm_provider")
    @classmethod
    def _check_provider(cls, v: str) -> str:
        v = v.strip().lower()
        if v == "ollama":
            raise ValueError(
                "LLM_PROVIDER=ollama is no longer supported (local Ollama/Qwen "
                "was removed). Set LLM_PROVIDER=gemini and GEMINI_API_KEY."
            )
        if v not in {"gemini", "anthropic"}:
            raise ValueError("LLM_PROVIDER must be 'gemini' or 'anthropic'")
        return v

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_anthropic(self) -> bool:
        return self.llm_provider == "anthropic"

    @property
    def chat_model(self) -> str:
        return self.anthropic_model if self.is_anthropic else self.gemini_chat_model

    @property
    def coder_model(self) -> str:
        if self.is_anthropic:
            return self.anthropic_coder_model or self.anthropic_model
        return self.gemini_coder_model or self.gemini_chat_model

    def model_for_persona(self, persona: str) -> str:
        """Pick the chat vs. coder model based on persona."""
        if persona in {"developer", "architect", "coder"}:
            return self.coder_model
        return self.chat_model

    def tts_voice(self, gender: str, lang: str) -> str:
        male = gender == "male"
        if lang == "bn":
            override = self.tts_voice_male_bn if male else self.tts_voice_female_bn
            if override:
                return override
        return self.tts_voice_male if male else self.tts_voice_female


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
