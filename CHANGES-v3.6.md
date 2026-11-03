# Aida v3.6 — backend

- The general mode formerly called "Non-stop" is now **Aida mode** in every prompt: the product description, the mode map, the
  hand-offs of the six specialists ("... -> Aida mode"), the decline rule ("Aida mode can help with anything"), the spoken-style
  brief ("You are in AIDA mode ...") and the memory/chat comments. The persona id stays `nonstop` (stored conversations, memory
  partitions, API), and it is still the only mode without the expertise guardrails.
- Eval harness (`evals/run_eval.py`): the decline detector also recognises "Aida mode".
- Tests updated (`tests/test_personas.py`, `tests/test_scope_eval.py`): 383 passing.
