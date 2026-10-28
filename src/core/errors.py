"""User-safe error taxonomy.

Raw exception text never reaches the client (it can leak hostnames, SQL, keys).
Every failure is mapped to a stable `code`; the client gets a localized,
friendly message plus the code and a request id for log correlation.
"""

MESSAGES: dict[str, dict[str, str]] = {
    "rate_limited": {
        "en": "Too many requests right now. Please wait a few seconds and try again.",
        "bn": "এই মুহূর্তে অনুরোধ অনেক বেশি হয়ে গেছে। অনুগ্রহ করে কয়েক সেকেন্ড পরে আবার চেষ্টা করুন।",
    },
    "daily_cap": {
        "en": "Today's service capacity has been reached. Please try again later.",
        "bn": "আজকের জন্য সেবার সীমা শেষ হয়ে গেছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
    },
    "timeout": {
        "en": "The answer is taking too long. Please try again.",
        "bn": "উত্তর পেতে বেশি সময় লাগছে। অনুগ্রহ করে আবার চেষ্টা করুন।",
    },
    "unavailable": {
        "en": "The AI service is temporarily unavailable. Please try again shortly.",
        "bn": "এআই সেবাটি এই মুহূর্তে পাওয়া যাচ্ছে না। অনুগ্রহ করে একটু পরে আবার চেষ্টা করুন।",
    },
    "blocked": {
        "en": "Sorry, I can't answer that request. Please try rephrasing it.",
        "bn": "দুঃখিত, এই অনুরোধের উত্তর আমি দিতে পারছি না। অনুগ্রহ করে অন্যভাবে জিজ্ঞাসা করুন।",
    },
    "not_configured": {
        "en": "The AI service is not configured yet. Please contact the administrator.",
        "bn": "এআই সেবা এখনো কনফিগার করা হয়নি। অনুগ্রহ করে প্রশাসকের সাথে যোগাযোগ করুন।",
    },
    "too_large": {
        "en": "That input is too large. Please shorten it and try again.",
        "bn": "ইনপুটটি অনেক বড়। অনুগ্রহ করে ছোট করে আবার চেষ্টা করুন।",
    },
    "not_found": {
        "en": "That conversation could not be found.",
        "bn": "কথোপকথনটি খুঁজে পাওয়া যায়নি।",
    },
    "internal": {
        "en": "Something went wrong on our side. Please try again.",
        "bn": "একটি অপ্রত্যাশিত সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।",
    },
}

HTTP_STATUS = {
    "rate_limited": 429,
    "daily_cap": 429,
    "timeout": 504,
    "unavailable": 503,
    "blocked": 422,
    "not_configured": 503,
    "too_large": 413,
    "not_found": 404,
    "internal": 500,
}

# Failures worth retrying (only before any output was produced).
RETRYABLE = {"rate_limited", "timeout", "unavailable"}


def message_for(code: str, lang: str = "en") -> str:
    entry = MESSAGES.get(code) or MESSAGES["internal"]
    return entry.get(lang) or entry["en"]


class AidaError(Exception):
    """An error whose `code` is safe to show; `detail` is for logs only."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code if code in MESSAGES else "internal"
        self.detail = detail

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    @property
    def status(self) -> int:
        return HTTP_STATUS.get(self.code, 500)

    def message(self, lang: str = "en") -> str:
        return message_for(self.code, lang)


def request_lang(request) -> str:
    """'bn' if the client declared Bengali via the X-Lang header, else 'en'."""
    return "bn" if request.headers.get("x-lang", "").lower().startswith("bn") else "en"


def http_detail(code: str, lang: str = "en") -> dict:
    """Standard error body: {"detail": {"code", "message"}} (via HTTPException)."""
    return {"code": code, "message": message_for(code, lang)}
