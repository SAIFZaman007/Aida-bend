import json
import os

import pytest

from src.core.text import (SentenceSegmenter, detect_language, dominant_language, foreign_script_share,
                           is_foreign_script, split_sentences)

with open(os.path.join(os.path.dirname(__file__), "vectors", "speech_vectors.json"), encoding="utf-8") as _f:
    VECTORS = json.load(_f)

CASES = [
    ("আমি ভালো আছি। আপনি কেমন আছেন? ধন্যবাদ।",
     ["আমি ভালো আছি।", " আপনি কেমন আছেন?", " ধন্যবাদ।"]),
    ("Version 3.5 is out. Meeting at 10:30 today.",
     ["Version 3.5 is out.", " Meeting at 10:30 today."]),
    ("Visit https://example.com/a.b now. Done.",
     ["Visit https://example.com/a.b now.", " Done."]),
    ("Use e.g. a list, etc. and Dr. Rahman said so. Next.",
     ["Use e.g. a list, etc. and Dr. Rahman said so.", " Next."]),
    ("দাম ৩.৫ টাকা। ফোন ০১৭১২৩৪৫৬৭৮।।\n\nশেষ",
     ["দাম ৩.৫ টাকা।", " ফোন ০১৭১২৩৪৫৬৭৮।।", "\n\nশেষ"]),
    ("No. 5 wins!!! Really?! Yes... ok.",
     ["No. 5 wins!!!", " Really?!", " Yes...", " ok."]),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_segments(text, expected):
    assert [s.text for s in split_sentences(text)] == expected


@pytest.mark.parametrize("text,_", CASES)
def test_partition_and_streaming_equivalence(text, _):
    """Highlighter offsets rely on: joined segments == full text, and streaming
    one character at a time yields exactly the same segments as one shot."""
    one_shot = split_sentences(text)
    assert "".join(s.text for s in one_shot) == text
    seg, out = SentenceSegmenter(), []
    for ch in text:
        out += seg.feed(ch)
    out += seg.flush()
    assert [(s.text, s.code) for s in out] == [(s.text, s.code) for s in one_shot]


def test_code_fence_flagged_and_not_split_on_dots():
    text = "Intro:\n```python\nx = 1.5\nprint(x)\n```\nAfter the code. Bye"
    segs = split_sentences(text)
    assert "".join(s.text for s in segs) == text
    flags = [(s.text.strip(), s.code) for s in segs]
    assert ("x = 1.5", True) in flags and ("print(x)", True) in flags
    assert ("After the code.", False) in flags and ("Intro:", False) in flags


def test_long_unpunctuated_stream_is_force_split():
    seg = SentenceSegmenter()
    out = seg.feed("word " * 200) + seg.flush()
    assert len(out) >= 3
    assert all(len(s.text) <= 305 for s in out)
    assert "".join(s.text for s in out) == "word " * 200


def test_trailing_whitespace_not_lost():
    segs = split_sentences("Hello.\n")
    assert "".join(s.text for s in segs) == "Hello.\n"


@pytest.mark.parametrize("case", VECTORS["detect_language"])
def test_detect_language_vectors(case):
    assert detect_language(case["in"]) == case["out"]


def test_dominant_language_defaults_to_en():
    assert dominant_language("১২৩") == "en"
    assert dominant_language("আমি") == "bn"


# ---------------------------------------------------- transcript script guard
@pytest.mark.parametrize("case", VECTORS["foreign_script"])
def test_foreign_script_vectors(case):
    assert is_foreign_script(case["in"]) is case["out"]


def test_foreign_script_share_and_allow_list():
    assert foreign_script_share("") == 0.0
    assert foreign_script_share("12 !?") == 0.0
    assert foreign_script_share("就说") == 1.0
    assert foreign_script_share("ab就说") == 0.5
    # other scripts can be allowed explicitly; an empty list disables the check
    assert not is_foreign_script("नमस्ते", ["bengali", "latin", "devanagari"])
    assert is_foreign_script("नमस्ते")
    assert not is_foreign_script("就说那个。", [])
    # names are case-insensitive
    assert not is_foreign_script("আমি", ["Bengali"])
