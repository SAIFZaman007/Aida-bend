import json
import os

import pytest

from src.voice.text_normalize import normalize_for_speech

with open(os.path.join(os.path.dirname(__file__), "vectors", "speech_vectors.json"), encoding="utf-8") as _f:
    VECTORS = json.load(_f)


@pytest.mark.parametrize("case", VECTORS["normalize"], ids=lambda c: c["id"])
def test_normalize_vectors(case):
    assert normalize_for_speech(case["in"]) == case["out"]


def test_bengali_number_is_never_altered():
    """Regression: the old JS mirror turned ৫০০০ into ৫০."""
    for n in ("৫০০০", "১০০০০", "০০০৭", "৯৯৯৯"):
        assert n in normalize_for_speech(f"দাম {n} টাকা")
