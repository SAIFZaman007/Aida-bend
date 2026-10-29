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


# ---- v2.5: the first clause of a long opening sentence is released early ------------
import random

from src.core.text import SentenceSegmenter, split_sentences  # noqa: E402

LONG_OPENER = ("Well, after looking at the numbers you sent, the most likely cause is a missing "
               "index on the table, which makes every lookup scan all rows. Add the index first.")


def _feed(text, step, **kw):
    seg = SentenceSegmenter(**kw)
    out = []
    for i in range(0, len(text), step):
        out += seg.feed(text[i:i + step])
    out += seg.flush()
    return out


def test_first_clause_is_released_early_when_the_opening_sentence_is_long():
    seg = SentenceSegmenter(first_clause_chars=60)
    got = seg.feed(LONG_OPENER[:75])
    assert [s.text for s in got] == ["Well, after looking at the numbers you sent,"]
    assert got[0].code is False


def test_early_release_never_changes_the_text():
    for step in (1, 3, 7, 50):
        parts = _feed(LONG_OPENER, step, first_clause_chars=60)
        assert "".join(s.text for s in parts) == LONG_OPENER
        assert len(parts) >= 3  # clause + rest of sentence + the second sentence


def test_only_the_first_segment_is_cut_at_a_clause():
    parts = _feed(LONG_OPENER + " " + LONG_OPENER, 5, first_clause_chars=60)
    texts = [s.text for s in parts]
    assert texts[0].endswith("sent,")
    # the second long sentence is NOT chopped at its commas
    assert any(t.strip().startswith("Well, after") and t.strip().endswith("all rows.") for t in texts)


def test_short_sentences_and_default_behaviour_are_untouched():
    text = "Yes, that works. Anything else, maybe?"
    assert [s.text for s in _feed(text, 4, first_clause_chars=80)] == [s.text for s in split_sentences(text)]
    # off by default: identical to the old behaviour
    assert [s.text for s in _feed(LONG_OPENER, 5)] == [s.text for s in split_sentences(LONG_OPENER)]


def test_numbers_and_tiny_clauses_are_not_cut():
    seg = SentenceSegmenter(first_clause_chars=40)
    assert seg.feed("The total comes to 1,000,000 units across all of the regions we serve") == []
    seg = SentenceSegmenter(first_clause_chars=30)
    assert seg.feed("Hi, this is a rather long opening that just keeps on going and going") == []  # "Hi," < 28 chars


def test_code_fences_are_never_cut_by_the_early_release():
    text = "```python\nprint('a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v')\n```\n"
    parts = _feed(text, 6, first_clause_chars=30)
    assert "".join(s.text for s in parts) == text
    assert all(s.code for s in parts)


def test_early_release_partition_holds_for_random_text_and_chunking():
    rng = random.Random(7)
    words = ["alpha,", "beta", "gamma;", "delta", "ε,", "আমি", "ভালো,", "আছি।", "ok.", "3.5", "x", "—", "end?"]
    for _ in range(200):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(3, 60)))
        step = rng.randint(1, 9)
        parts = _feed(text, step, first_clause_chars=rng.choice([20, 40, 80]))
        assert "".join(s.text for s in parts) == text


# ---------------------------------------------------- "no speech" phrases (v2.5)
from src.core.text import is_noise_phrase  # noqa: E402


@pytest.mark.parametrize("case", VECTORS["noise_phrase"])
def test_noise_phrase_vectors(case):
    assert is_noise_phrase(case["in"]) is case["out"], case["in"]


def test_long_sentences_are_never_noise_phrases():
    assert not is_noise_phrase("Didn't catch that " + "word " * 12)
