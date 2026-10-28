"""Text normalization for natural-sounding speech (Bengali + English).

Mirrored by frontend/src/lib/text.js — the two MUST stay behaviourally
identical; tests/vectors/speech_vectors.json is executed by BOTH test suites.

Rules, and why they are language-aware (v2.1):
  • Stutter collapse ("soooo", "!!!", "go go go go") — never touches digits of
    ANY script. (Old JS version collapsed Bengali "৫০০০" into "৫০", changing
    the number that was spoken.)
  • Repeated-WORD collapse is ASCII-only: it must not delete deliberate Bengali
    repetition ("ধীরে ধীরে ধীরে").
  • English-only expansions (digit runs -> English cardinals, "e.g.", acronym
    letter-spelling, "percent"/"plus"...) apply ONLY when the text is
    English-dominant. Inside a Bengali sentence they used to inject English
    words ("১০০ percent", "twelve million…").
  • Bengali symbols are spoken in Bengali (% -> শতাংশ, & -> এবং, ...).
  • Digit runs that are really identifiers (leading zero, or 11+ digits: phone
    numbers, IDs) are read digit by digit, not as one huge cardinal.
"""
import re

from src.core.text import dominant_language

# --- English abbreviation / acronym expansions -----------------------------
_ABBREV_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\be\.g\.,?\s*", re.IGNORECASE | re.ASCII), "for example, "),
    (re.compile(r"\bi\.e\.,?\s*", re.IGNORECASE | re.ASCII), "that is, "),
    (re.compile(r"\betc\.", re.IGNORECASE | re.ASCII), "et cetera"),
    (re.compile(r"\bvs\.", re.IGNORECASE | re.ASCII), "versus"),
    (re.compile(r"\bw/o\b", re.IGNORECASE | re.ASCII), "without"),
    (re.compile(r"\bw/\b", re.IGNORECASE | re.ASCII), "with"),
    (re.compile(r"\bapprox\.", re.IGNORECASE | re.ASCII), "approximately"),
    (re.compile(r"\bno\.\s*(?=\d)", re.IGNORECASE | re.ASCII), "number "),
    (re.compile(r"\bDr\.", re.ASCII), "Doctor"),
    (re.compile(r"\bMr\.", re.ASCII), "Mister"),
    (re.compile(r"\bMrs\.", re.ASCII), "Missus"),
    (re.compile(r"\bMs\.", re.ASCII), "Miss"),
    (re.compile(r"\bSt\.", re.ASCII), "Street"),
]

_ACRONYM_WORDS: dict[str, str] = {
    "ASAP": "as soon as possible",
    "FAQ": "F A Q",
    "URL": "U R L",
    "API": "A P I",
    "SQL": "sequel",
    "JSON": "jay-son",
    "HTML": "H T M L",
    "CSS": "C S S",
    "CPU": "C P U",
    "GPU": "G P U",
    "RAM": "ram",
    "AI": "A I",
}

# --- Symbol -> word, per language -------------------------------------------
_SYMBOLS = {
    "en": {"&": "and", "%": "percent", "=": "equals", "+": "plus",
           "num": "number", "@": "at"},
    "bn": {"&": "এবং", "%": "শতাংশ", "=": "সমান", "+": "প্লাস",
           "num": "নম্বর", "@": "অ্যাট"},
}

_ASCII_DIGIT_RUN_RE = re.compile(r"\b[0-9]{4,}\b", re.ASCII)
# 3+ identical non-digit, non-space chars. \d is Unicode (all scripts' digits).
_REPEATED_CHAR_RE = re.compile(r"([^\d\s])\1{2,}")
# ASCII words only (see module docstring).
_REPEATED_WORD_RE = re.compile(r"\b([A-Za-z0-9_]+)(?:\s+\1\b){2,}", re.IGNORECASE | re.ASCII)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b", re.ASCII)

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = ["", "thousand", "million", "billion", "trillion"]


def _cardinal(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + _cardinal(-n)

    def three_digit(num: int) -> str:
        parts = []
        hundreds, rem = divmod(num, 100)
        if hundreds:
            parts.append(_ONES[hundreds] + " hundred")
        if rem:
            if rem < 20:
                parts.append(_ONES[rem])
            else:
                tens, ones = divmod(rem, 10)
                parts.append(_TENS[tens] + (f"-{_ONES[ones]}" if ones else ""))
        return " ".join(parts)

    groups = []
    while n > 0:
        n, rem = divmod(n, 1000)
        groups.append(rem)

    words = []
    for i in reversed(range(len(groups))):
        if groups[i] == 0:
            continue
        chunk = three_digit(groups[i])
        if _SCALES[i]:
            chunk += " " + _SCALES[i]
        words.append(chunk)
    return " ".join(words)


def _spell_number(match: "re.Match[str]") -> str:
    digits = match.group(0)
    # Identifiers (phone numbers, IDs): leading zero or very long -> digit by digit.
    if digits[0] == "0" or len(digits) >= 11:
        return " ".join(digits)
    return _cardinal(int(digits))


def normalize_for_speech(text: str, lang: str | None = None) -> str:
    """Make text read naturally by a TTS voice. `lang` ('bn'|'en') is detected
    from the text when omitted."""
    if not text:
        return ""
    lang = lang or dominant_language(text)
    t = text

    t = _REPEATED_CHAR_RE.sub(lambda m: m.group(1), t)
    t = _REPEATED_WORD_RE.sub(lambda m: m.group(1), t)

    if lang == "en":
        t = _ASCII_DIGIT_RUN_RE.sub(_spell_number, t)
        for pattern, repl in _ABBREV_MAP:
            t = pattern.sub(repl, t)
        t = _ACRONYM_RE.sub(lambda m: _ACRONYM_WORDS.get(m.group(0), m.group(0)), t)

    sym = _SYMBOLS[lang]
    t = re.sub(r"\s*&\s*", f" {sym['&']} ", t)
    t = re.sub(r"\s*%\s*", f" {sym['%']} ", t)
    t = re.sub(r"\s*=\s*", f" {sym['=']} ", t)
    t = re.sub(r"\s*\+\s*", f" {sym['+']} ", t)
    t = re.sub(r"#(\d+)", lambda m: f"{sym['num']} {m.group(1)}", t)
    t = t.replace("@", f" {sym['@']} ")

    return re.sub(r"\s+", " ", t).strip()
