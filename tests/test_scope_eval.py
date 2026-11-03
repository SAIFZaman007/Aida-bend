"""The scope cases in evals/golden.json need a real key to run; these tests pin the judge itself."""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "evals" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def test_a_polite_decline_passes_and_an_answer_fails():
    case = {"type": "scope", "expect": "decline", "forbid": [r"\bParis\b"]}
    assert run_eval.judge(case, "That is outside my field as Aida in Coder mode. Aida mode can help with anything.")
    assert not run_eval.judge(case, "The capital of France is Paris.")
    assert not run_eval.judge(case, "Paris. Anyway, that is outside my field.")   # answered AND declined


def test_a_long_reply_is_not_a_short_decline():
    case = {"type": "scope", "expect": "decline"}
    assert not run_eval.judge(case, "That is outside my field. " + "More detail. " * 60)


def test_in_scope_questions_must_be_answered_not_declined():
    case = {"type": "scope", "expect": "answer", "require": ["Maktech"]}
    assert run_eval.judge(case, "I was developed by Maktech.")
    assert not run_eval.judge(case, "That is outside my field.")
    assert not run_eval.judge(case, "I was built by a team.")


def test_golden_scope_cases_are_well_formed_and_cover_all_six_specialists_and_nonstop_is_absent():
    spec_ = json.loads((ROOT / "evals" / "golden.json").read_text(encoding="utf-8"))
    cases = [c for c in spec_["cases"] if c["type"] == "scope"]
    assert {c["persona"] for c in cases} >= {"coder", "security", "writer", "executive", "researcher", "architect"}
    assert "nonstop" not in {c["persona"] for c in cases}
    assert {c["expect"] for c in cases} == {"decline", "answer"}
    assert spec_["thresholds"]["scope"] == 1.0
