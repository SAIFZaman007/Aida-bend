"""Google Cloud Text-to-Speech (Chirp 3 HD) over REST.

Used when TTS_PROVIDER=google_cloud and GOOGLE_TTS_API_KEY is set. It is a separate Google
product from the Gemini API: its own API enablement, its own billing and its own quota
(there is no 100-requests-a-day cap), so Bangla keeps speaking when the Gemini voice models
are out of quota. `tts.py` treats it as one more "voice model" (the pseudo-name `GCLOUD`),
so the quota gate, health ordering, fail-over to Gemini and the endpoints are unchanged.

One request per sentence (that is how the client sends text), answered as a WAV
(LINEAR16); the PCM inside is handed on in small chunks, in the same shape the Gemini
stream has. Nothing here imports `tts.py`.
"""
import base64
import io
import logging
import re
import wave

import httpx

from src.core.config import settings
from src.core.errors import AidaError

log = logging.getLogger("aida.tts.gcloud")

GCLOUD = "google-cloud:chirp3-hd"  # pseudo model name used by the gate / health tracking
_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"
_CHUNK = 16384  # bytes of PCM per yielded chunk (a multiple of 2: whole 16-bit samples)
_PER_DAY = re.compile(r"per\s*day|perday|daily", re.I)

_client: httpx.AsyncClient | None = None


def enabled() -> bool:
    """True when Google Cloud is the first-choice voice (key present, provider selected)."""
    return settings.tts_google_cloud


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.tts_first_audio_timeout, connect=3.0),
            limits=httpx.Limits(max_connections=max(4, settings.tts_concurrency * 2)),
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        c, _client = _client, None
        await c.aclose()


def voice_for(gender: str, lang: str) -> tuple[str, str]:
    """(language code, full voice name) for the speaker, e.g. ("bn-IN", "bn-IN-Chirp3-HD-Kore").

    A setting that already holds a full name (contains "Chirp3-HD") is used as it is; a bare
    name such as "Kore" is completed with the locale."""
    male = gender == "male"
    bn = lang == "bn"
    locale = (settings.google_tts_locale_bn if bn else settings.google_tts_locale_en).strip()
    name = ""
    if bn:
        name = (settings.google_tts_voice_male_bn if male else settings.google_tts_voice_female_bn).strip()
    if not name:
        name = (settings.google_tts_voice_male if male else settings.google_tts_voice_female).strip()
    if "-" in name:  # a full voice name: its own locale prefix wins
        return "-".join(name.split("-")[:2]), name
    return locale, f"{locale}-Chirp3-HD-{name}"


def _retry_after(resp: httpx.Response) -> int:
    hv = (resp.headers.get("retry-after") or "").strip()
    if hv.isdigit():
        return int(hv)
    return 900 if _PER_DAY.search(resp.text or "") else 0  # a per-day quota names no useful wait


def _google_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return f'{err.get("status", "")} {err.get("message", "")}'.strip()[:300]
    except Exception:
        return (resp.text or "")[:300]


def _raise_for(resp: httpx.Response) -> None:
    """Map a non-200 answer to the user-safe error taxonomy. Credential / configuration
    problems are logged loudly for the operator but surface as a plain "unavailable", so
    the Gemini voice (the fall-back) still gets its turn; the key is never in a message."""
    code = resp.status_code
    msg = _google_message(resp)
    if code == 429:
        raise AidaError("rate_limited", f"Google Cloud TTS 429: {msg}", retry_after=_retry_after(resp))
    if code in (400, 401, 403, 404):
        log.error("Google Cloud TTS rejected the request (HTTP %s): %s - check GOOGLE_TTS_API_KEY "
                  "(restricted to the Text-to-Speech API?), that the API is enabled and billing is "
                  "on, and the voice name / locale (python scripts/list_gcloud_voices.py).", code, msg)
        raise AidaError("unavailable", f"Google Cloud TTS {code}: {msg}")
    if code >= 500:
        raise AidaError("unavailable", f"Google Cloud TTS {code}: {msg}")
    raise AidaError("internal", f"Google Cloud TTS {code}: {msg}")


def _pcm_of(audio: bytes) -> tuple[bytes, int]:
    """(headerless 16-bit mono PCM, sample rate) from the WAV Google returns."""
    if audio[:4] != b"RIFF":  # defensive: raw PCM
        return audio[: len(audio) - (len(audio) % 2)], settings.tts_sample_rate
    with wave.open(io.BytesIO(audio), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise AidaError("unavailable", "Google Cloud TTS returned an unexpected audio format")
        return wf.readframes(wf.getnframes()), wf.getframerate()


async def fetch(text: str, gender: str, lang: str) -> tuple[bytes, int]:
    """One synthesis request -> (PCM, sample_rate). Raises AidaError."""
    locale, name = voice_for(gender, lang)
    body = {
        "input": {"text": text},
        "voice": {"languageCode": locale, "name": name},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": settings.tts_sample_rate},
    }
    try:
        resp = await _http().post(
            _ENDPOINT, json=body, headers={"X-Goog-Api-Key": settings.google_tts_api_key})
    except httpx.TimeoutException as exc:
        raise AidaError("timeout", f"Google Cloud TTS: {exc!r}") from exc
    except httpx.TransportError as exc:
        raise AidaError("unavailable", f"Google Cloud TTS: {exc!r}") from exc
    if resp.status_code != 200:
        _raise_for(resp)
    try:
        audio = base64.b64decode(resp.json()["audioContent"])
        pcm, rate = _pcm_of(audio)
    except AidaError:
        raise
    except Exception as exc:
        raise AidaError("unavailable", f"Google Cloud TTS: unreadable answer ({exc!r})") from exc
    if not pcm:
        raise AidaError("unavailable", "Google Cloud TTS returned no audio")
    return pcm, rate


def chunks(pcm: bytes):
    for i in range(0, len(pcm), _CHUNK):
        yield pcm[i : i + _CHUNK]


def wav_of(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()
