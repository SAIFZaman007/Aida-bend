"""Bilingual + grounding evaluation. Run with your real key:

    python evals/run_eval.py [--runs 3]

Defines "no hallucination" as a MEASURABLE pass rate on a golden set:
  lang      reply script matches the required language
  contains  exact token preserved (technical terms, numbers)
  abstain   unanswerable/unknowable prompts get an honest "I don't know /
            can't verify / can't access" instead of an invented answer

Each case runs --runs times (default 3) because model output varies; a case
passes only if ALL runs pass. Exit code is non-zero when a category falls below
its threshold in golden.json — wire this into CI as a release gate.

Abstention is judged by hedge/denial phrases (English + Bengali) AND the absence
of per-case fabrication markers (`forbid` in golden.json). It is a heuristic: failures should be read by a human and, if the model
was actually right, the phrase list extended — not the threshold lowered.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.prompts import system_prompt  # noqa: E402
from src.core import gemini  # noqa: E402
from src.core.config import settings  # noqa: E402
from src.core.llm import llm  # noqa: E402
from src.core.text import script_counts  # noqa: E402

# Phrases that mean "I don't know / can't do that / that doesn't exist". Kept
# specific on purpose: bare words like "don't" or "নেই" appear in confident wrong
# answers too, which would let a hallucination pass.
HEDGE = re.compile(
    r"(i (don't|do not|can't|cannot|couldn't|won't) (have|know|see|find|recognize|verify|access|confirm|give|provide|tell|summari[sz]e|share)|"
    r"there (is|are) no|"
    r"(doesn't|does not|isn't|is not) (exist|a real|a standard|part of|available)|"
    r"not (a )?(real|standard|existing)|no such|not aware|"
    r"no (record|information|evidence|access)|unable to|not (sure|certain)|unsure|"
    r"fabricat|made[- ]?up|can't (browse|look|check)|cannot (browse|look|check)|"
    r"জানি না|জানা নেই|নিশ্চিত নই|অস্তিত্ব নেই|বিদ্যমান নয়|"
    r"(খুঁজে|দেখতে|যাচাই করতে|নিশ্চিত করতে|অ্যাক্সেস করতে|জানতে|বলতে|দিতে) (পারছি না|পারি না|পাচ্ছি না|পাইনি)|"
    r"(পারছি|পারি) না|অ্যাক্সেস নেই|কোনো (তথ্য|ফাংশন|পেপার|গবেষণা)[^।]{0,30}(নেই|পাইনি|পাওয়া যায়নি)|"
    r"এমন কোনো[^।]{0,40}(নেই|পাওয়া যায়নি))",
    re.IGNORECASE,
)


def bn_ratio(t: str) -> float:
    bn, lat = script_counts(t)
    return bn / (bn + lat) if bn + lat else 0.0


def judge(case: dict, reply: str) -> bool:
    kind = case["type"]
    if kind == "lang":
        r = bn_ratio(reply)
        return r >= 0.6 if case["expect"] == "bn" else r <= 0.05
    if kind == "contains":
        return case["needle"] in reply
    if kind == "abstain":
        # Must hedge AND must not state fabricated specifics.
        if any(re.search(p, reply, re.IGNORECASE) for p in case.get("forbid", [])):
            return False
        return bool(HEDGE.search(reply))
    raise ValueError(kind)


async def main(runs: int) -> int:
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set"); return 2
    spec = json.load(open(os.path.join(os.path.dirname(__file__), "golden.json"), encoding="utf-8"))
    per_type: dict[str, list[bool]] = defaultdict(list)
    for case in spec["cases"]:
        oks, sample = [], ""
        for _ in range(runs):
            msgs = [{"role": "system", "content": system_prompt("executive")},
                    {"role": "user", "content": case["prompt"]}]
            reply = await llm.complete(msgs)
            oks.append(judge(case, reply))
            sample = sample or reply
        ok = all(oks)
        per_type[case["type"]].append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['id']} ({sum(oks)}/{runs})"
              + ("" if ok else f"\n        sample reply: {sample[:240]!r}"))
    bad = False
    print()
    for kind, results in per_type.items():
        rate = sum(results) / len(results)
        need = spec["thresholds"].get(kind, 1.0)
        flag = "OK " if rate >= need else "LOW"
        bad |= rate < need
        print(f"{flag} {kind:9s} {rate:6.1%}  (threshold {need:.0%}, {len(results)} cases)")
    await gemini.aclose()
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    sys.exit(asyncio.run(main(ap.parse_args().runs)))
