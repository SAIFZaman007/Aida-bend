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
  • The assistant's own name is a NAME, not an abbreviation (v3.2): "A.I.D.A", "AIDA"
    and "Aida" are all spoken as the word "Aida" (written আইডা inside Bengali text, so
    the Bengali voice says it as a word too) instead of being read letter by letter.
  • Contact details are spoken, not spelled: "info@maktechgroup.com" -> "info at
    maktechgroup dot com", "maktechgroup.com" -> "maktechgroup dot com", a phone number
    -> its digits in groups ("+880 1897-669234" -> "plus 8 8 0, 1 8 9 7, 6 6 9 2 3 4"),
    a postal code after "Dhaka" -> digit by digit, "Rd" -> "Road".
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

# --- The assistant's name (spoken as a word) --------------------------------
_NAME_RE = re.compile(r"\bA\.I\.D\.A\b|\bAIDA\b|\bAida\b", re.ASCII)
_SPOKEN_NAME = {"en": "Aida", "bn": "আইডা"}

# --- Contact details --------------------------------------------------------
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.ASCII)
_DOMAIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|ai|app|dev|co|info|edu|gov|bd))\b"
    r"(?:/(?!\w))?", re.IGNORECASE | re.ASCII)
# digit groups of 3+ after the first ("+880 1897-669234"); dates ("2026-11-02") never match
_PHONE_RE = re.compile(r"\+?\d+(?:[ \-]\d{3,})+")
_POSTAL_RE = re.compile(r"(Dhaka|ঢাকা)([\s\-–]+)(\d{4})(?!\d)")
_ROAD_RE = re.compile(r"\bRd\b\.?", re.ASCII)

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


def _dot_words(address: str, lang: str) -> str:
    dot = " dot " if lang == "en" else " ডট "
    return address.replace(".", dot)


def _speak_email(m: "re.Match[str]", lang: str) -> str:
    local, _, domain = m.group(0).partition("@")
    at = _SYMBOLS[lang]["@"]
    return f"{_dot_words(local, lang)} {at} {_dot_words(domain, lang)}"


def _speak_phone(m: "re.Match[str]") -> str:
    text = m.group(0)
    groups = re.findall(r"\d+", text)
    if sum(len(g) for g in groups) < 9:  # not a phone number (e.g. "1 000 000")
        return text
    spoken = ", ".join(" ".join(g) for g in groups)
    return ("+" if text.startswith("+") else "") + spoken


def normalize_for_speech(text: str, lang: str | None = None) -> str:
    """Make text read naturally by a TTS voice. `lang` ('bn'|'en') is detected
    from the text when omitted."""
    if not text:
        return ""
    lang = lang or dominant_language(text)
    t = text

    t = _NAME_RE.sub(_SPOKEN_NAME[lang], t)
    t = _EMAIL_RE.sub(lambda m: _speak_email(m, lang), t)
    t = _DOMAIN_RE.sub(lambda m: _dot_words(m.group(1), lang), t)
    t = _PHONE_RE.sub(_speak_phone, t)
    t = _POSTAL_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{' '.join(m.group(3))}", t)

    t = _REPEATED_CHAR_RE.sub(lambda m: m.group(1), t)
    t = _REPEATED_WORD_RE.sub(lambda m: m.group(1), t)

    if lang == "en":
        t = _ASCII_DIGIT_RUN_RE.sub(_spell_number, t)
        for pattern, repl in _ABBREV_MAP:
            t = pattern.sub(repl, t)
        t = _ROAD_RE.sub("Road", t)
        t = _ACRONYM_RE.sub(lambda m: _ACRONYM_WORDS.get(m.group(0), m.group(0)), t)

    sym = _SYMBOLS[lang]
    t = re.sub(r"\s*&\s*", f" {sym['&']} ", t)
    t = re.sub(r"\s*%\s*", f" {sym['%']} ", t)
    t = re.sub(r"\s*=\s*", f" {sym['=']} ", t)
    t = re.sub(r"\s*\+\s*", f" {sym['+']} ", t)
    t = re.sub(r"#(\d+)", lambda m: f"{sym['num']} {m.group(1)}", t)
    t = t.replace("@", f" {sym['@']} ")

    return re.sub(r"\s+", " ", t).strip()
