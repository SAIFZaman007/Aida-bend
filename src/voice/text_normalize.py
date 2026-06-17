"""Text normalization for natural-sounding speech.

Fixes the "Avatar's voice glitches / too robotic" issue:
  • Repeated characters/digits ("11111111", "aaaaah") are collapsed or spoken
    as a group ("eleven million..." no — instead we say "1 repeated 8 times"
    is wrong too; for digit runs we read them as a NUMBER, e.g. "11111111" ->
    "eleven million one hundred eleven thousand one hundred eleven", which is
    how a human would read a long number aloud, not digit-by-digit).
  • Abbreviations and acronyms (e.g. "e.g.", "i.e.", "etc.", "API", "URL") are
    expanded to their spoken form instead of being read letter-by-letter or
    spelled out mechanically.
  • Common symbols (&, %, /, +, =, #, @) get spoken-word equivalents.

This module is used by both:
  - the backend Piper TTS endpoint (so server-side audio sounds natural), and
  - is mirrored by the frontend's sanitizeForSpeech() for the browser
    (webspeech) voice and for generating the highlighter's spoken-text view.

Keep the two implementations in sync if you add new rules.
"""
import re

# --- Abbreviation / acronym expansions -------------------------------------
# Order matters: longer / more specific patterns first.
_ABBREV_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\be\.g\.,?\s*", re.IGNORECASE), "for example, "),
    (re.compile(r"\bi\.e\.,?\s*", re.IGNORECASE), "that is, "),
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"\bw/\b", re.IGNORECASE), "with"),
    (re.compile(r"\bw/o\b", re.IGNORECASE), "without"),
    (re.compile(r"\bapprox\.", re.IGNORECASE), "approximately"),
    (re.compile(r"\bno\.\s*(?=\d)", re.IGNORECASE), "number "),
    (re.compile(r"\bDr\.", re.IGNORECASE), "Doctor"),
    (re.compile(r"\bMr\.", re.IGNORECASE), "Mister"),
    (re.compile(r"\bMrs\.", re.IGNORECASE), "Missus"),
    (re.compile(r"\bMs\.", re.IGNORECASE), "Miss"),
    (re.compile(r"\bSt\.", re.IGNORECASE), "Street"),
]

# Acronyms that should be pronounced as words ("API" -> "A P I" reads okay,
# but a few common ones sound better spoken naturally).
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

# --- Symbol -> word ----------------------------------------------------------
_SYMBOL_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"\s*%\s*"), " percent "),
    (re.compile(r"\s*=\s*"), " equals "),
    (re.compile(r"\s*\+\s*"), " plus "),
    (re.compile(r"#(\d+)"), r"number \1"),
    (re.compile(r"@"), " at "),
]

# Long digit-runs (4+ digits) get read as a number rather than letter-by-letter.
_DIGIT_RUN_RE = re.compile(r"\b\d{4,}\b")

# A run of 3+ identical non-digit characters ("soooo", "!!!", "----") is
# collapsed to a single character so it isn't spelled out.
_REPEATED_CHAR_RE = re.compile(r"([^\d\s])\1{2,}")

# A run of 3+ repeated short words/tokens ("go go go go") is collapsed to one.
_REPEATED_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b){2,}", re.IGNORECASE)


def _spell_number(match: "re.Match[str]") -> str:
    """Render a long digit run as a spoken cardinal number."""
    digits = match.group(0)
    try:
        n = int(digits)
    except ValueError:
        return digits
    return _cardinal(n)


_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]
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


def normalize_for_speech(text: str) -> str:
    """Make text read naturally by a TTS voice.

    Collapses repeated characters/words, expands abbreviations and common
    acronyms, converts symbols to words, and reads long digit runs as numbers
    instead of digit-by-digit.
    """
    if not text:
        return ""
    t = text

    # Collapse stuttered characters/words first ("11111111" handled below by
    # the digit-run rule; this targets things like "nooo" or "go go go go").
    t = _REPEATED_CHAR_RE.sub(lambda m: m.group(1), t)
    t = _REPEATED_WORD_RE.sub(lambda m: m.group(1), t)

    # Long digit runs -> spoken numbers ("11111111" -> "eleven million ...").
    t = _DIGIT_RUN_RE.sub(_spell_number, t)

    # Abbreviations.
    for pattern, repl in _ABBREV_MAP:
        t = pattern.sub(repl, t)

    # Whole-word acronyms.
    def _acro(m: "re.Match[str]") -> str:
        word = m.group(0)
        return _ACRONYM_WORDS.get(word.upper(), word)

    t = re.sub(r"\b[A-Z]{2,6}\b", _acro, t)

    # Symbols -> words.
    for pattern, repl in _SYMBOL_MAP:
        t = pattern.sub(repl, t)

    # Tidy whitespace.
    t = re.sub(r"\s+", " ", t).strip()
    return t
