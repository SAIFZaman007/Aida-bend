"""Bilingual (Bengali / English) text primitives.

Two jobs, both deterministic (no model calls):

1. LANGUAGE — `detect_language()` classifies text by Unicode script. Bengali
   script is U+0980–U+09FF; Latin is A–Z/a–z. This is what picks TTS voice
   language, number/symbol wording, and localized error messages. Romanized
   Bengali ("Banglish") is intentionally NOT guessed here: it is
   indistinguishable from English without a model, so the system prompt tells
   the LLM to understand it and reply in Bengali script.

2. SEGMENTATION — `SentenceSegmenter` splits a *streaming* reply into
   sentences. It is the single source of truth for sentence boundaries: the
   backend emits them as SSE `sentence` events and the frontend speaks/
   highlights exactly those. Properties (all unit-tested):
     • PARTITION: ''.join(all emitted sentences) == the full reply, so the
       frontend's character offsets for the reading highlighter stay exact.
     • Bengali danda `।` / double danda `॥` end a sentence.
     • `.` only ends a sentence when followed by whitespace — so 3.5, 10:30,
       example.com, https://x.y/z are never cut apart (the old regex cut
       "Version 3.5" into "Version 3." + "5"). Common abbreviations are skipped.
     • Lines inside ``` fences are flagged `code=True` (not spoken aloud).
     • A hard length cap prevents a no-punctuation stream from starving TTS.
"""
from __future__ import annotations

from dataclasses import dataclass

_BN_LO, _BN_HI = 0x0980, 0x09FF

# Sentence-ending / closing characters.
_DANDAS = "।॥"
_TERMINATORS = ".!?…"
_CLOSERS = "\"'”’)]»"
_ABBREVIATIONS = {
    "e.g", "i.e", "vs", "dr", "mr", "mrs", "ms", "st", "approx", "fig", "eq",
}

MAX_SEGMENT_CHARS = 300


# ----------------------------------------------------------------- language
def script_counts(text: str) -> tuple[int, int]:
    """Return (bengali_letters, latin_letters)."""
    bn = latin = 0
    for ch in text:
        o = ord(ch)
        if _BN_LO <= o <= _BN_HI:
            if ch.isalpha():
                bn += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            latin += 1
    return bn, latin


def detect_language(text: str) -> str:
    """'bn' | 'en' | 'mixed' | 'unknown' by letter-script share."""
    bn, latin = script_counts(text)
    total = bn + latin
    if total == 0:
        return "unknown"
    ratio = bn / total
    if ratio >= 0.6:
        return "bn"
    if ratio <= 0.1:
        return "en"
    return "mixed"


def dominant_language(text: str) -> str:
    """Always 'bn' or 'en' (ties and unknowns -> 'en')."""
    bn, latin = script_counts(text)
    return "bn" if bn > latin else "en"


# ------------------------------------------------------------- segmentation
@dataclass(frozen=True)
class Segment:
    text: str
    code: bool = False


def _abbrev_token(s: str, dot_index: int) -> str:
    j = dot_index
    while j > 0 and (s[j - 1].isalpha() or s[j - 1] == "."):
        j -= 1
    return s[j:dot_index].lower().strip(".")


def _is_abbrev(token: str, next_non_space: str) -> bool:
    """True if a '.' after `token` closes a known abbreviation (not a sentence)."""
    if token in _ABBREVIATIONS:
        return True
    if token == "no" and next_non_space.isdigit():
        return True
    if token == "etc":
        # "etc." only ends the sentence if a new sentence clearly follows.
        return next_non_space.islower()
    return False


# Tokens whose verdict depends on the character AFTER the whitespace, so a
# streaming buffer must wait for it (keeps streaming == one-shot behaviour).
_NEEDS_LOOKAHEAD = {"etc", "no"}


def _find_boundary(s: str, final: bool) -> int | None:
    """End index (exclusive) of the first complete sentence in s, else None.

    A '.'/'!'/'?' at the very end of an unfinished buffer is undecidable (it
    may be the start of "3.5"), so we wait for the next character unless
    `final` is set.
    """
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch == "\n":
            return i + 1
        if ch in _DANDAS:
            j = i + 1
            while j < n and s[j] in _DANDAS + _CLOSERS:
                j += 1
            if j >= n and not final:
                return None  # a second danda / closing quote may still arrive
            return j
        if ch in _TERMINATORS or ch == ":":
            j = i + 1
            if ch != ":":
                while j < n and s[j] in _TERMINATORS:
                    j += 1
            k = j
            while k < n and s[k] in _CLOSERS:
                k += 1
            if k >= n:
                if final:
                    return k
                return None  # undecidable until more text arrives
            nxt = s[k]
            if nxt.isspace():
                nn = k
                while nn < n and s[nn].isspace() and s[nn] != "\n":
                    nn += 1
                next_non_space = s[nn] if nn < n else ""
                if ch == "." and j == i + 1:
                    token = _abbrev_token(s, i)
                    if (
                        not next_non_space
                        and not final
                        and token in _NEEDS_LOOKAHEAD
                    ):
                        return None
                    if _is_abbrev(token, next_non_space):
                        i = j
                        continue
                return k
            i = j  # "3.5", "10:30", "a.b", "https://" -> not a boundary
            continue
        i += 1
    return None


class SentenceSegmenter:
    """Incremental segmenter. feed() tokens as they stream, flush() at the end."""

    def __init__(self) -> None:
        self._buf = ""
        self._carry = ""  # whitespace-only chunks, glued onto the next segment
        self._in_fence = False

    def _emit(self, text: str) -> Segment:
        if text.lstrip().startswith("```"):
            self._in_fence = not self._in_fence
            return Segment(text, code=True)
        return Segment(text, code=self._in_fence)

    def _drain(self, final: bool) -> list[Segment]:
        out: list[Segment] = []
        while self._buf:
            if self._in_fence:
                # Inside code: only line breaks delimit segments.
                nl = self._buf.find("\n")
                if nl == -1:
                    if not final:
                        break
                    end = len(self._buf)
                else:
                    end = nl + 1
            else:
                b = _find_boundary(self._buf, final)
                if b is None:
                    if len(self._buf) > MAX_SEGMENT_CHARS:
                        cut = self._buf.rfind(" ", 0, MAX_SEGMENT_CHARS)
                        b = cut + 1 if cut > 0 else MAX_SEGMENT_CHARS
                    elif final:
                        b = len(self._buf)
                    else:
                        break
                end = b
            chunk, self._buf = self._buf[:end], self._buf[end:]
            if not chunk.strip():
                # Never drop characters (the highlighter maps offsets onto the
                # full text) and never emit an empty sentence: carry it forward.
                self._carry += chunk
                continue
            out.append(self._emit(self._carry + chunk))
            self._carry = ""
        return out

    def feed(self, piece: str) -> list[Segment]:
        self._buf += piece
        return self._drain(final=False)

    def flush(self) -> list[Segment]:
        out = self._drain(final=True)
        if self._carry:
            out.append(Segment(self._carry, code=self._in_fence))
            self._carry = ""
        return out


def split_sentences(text: str) -> list[Segment]:
    """Non-streaming convenience wrapper (used by tests and scripts)."""
    seg = SentenceSegmenter()
    return seg.feed(text) + seg.flush()
