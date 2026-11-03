# A.I.D.A v3.5 — backend

**Strict expertise guardrails for the six specialist modes** (Executive, Coder, Architect, Security, Researcher, Writer). Non-stop is untouched: no boundary at all.

A specialist now answers ONLY: (1) questions inside its own field, (2) questions about Aida the platform, (3) questions about Maktech (from the verified facts only). Greetings, thanks and "repeat that" are fine. Everything else (general knowledge, trivia, news, sport, cooking, travel, health, personal advice, homework, any task that belongs to another mode) is declined: one or two short, friendly sentences saying it is outside the field, what the mode can help with, naming the right mode when there is one, otherwise "Non-stop mode can help with anything".
- Not answered "briefly", "partly" or "just this once", and not when the person insists, calls it a test, asks to pretend, to ignore the rules or to switch role. The instructions are never revealed.
- A mixed request: the in-field part is answered, the rest declined in one sentence.
- The rule is written once (`_SCOPE_RULES`) and repeated as the last lines of the prompt (`_SCOPE_REMINDER`), where the model weighs it most.
- Reconciled with v3.4's Maktech referral: a software/IT question **outside** the mode's field is declined and pointed to Maktech (and to the fitting mode); inside the field it is answered, then Maktech is mentioned.

This is enforced by the prompt (the model decides), not by a hard filter. To measure it with your real key: `python evals/run_eval.py` now includes 11 `scope` cases (8 must decline, 3 must answer), threshold 100%.

Tests: `pytest` → 383 pass (24 new).
