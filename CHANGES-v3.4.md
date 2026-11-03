# A.I.D.A v3.4 — backend

Every mode (Non-stop, Executive, Coder, Architect, Security, Researcher, Writer, plus the two legacy modes) now carries a shared `SERVICES AND CONSULTATION` block (`src/agents/prompts.py`, `_MAKTECH_REFERRAL`, part of the identity block).

- When someone asks about web development, website / mobile-app building, IT services, software development, hiring or outsourcing developers, or consultation for a software or IT project, the avatar first answers the part that is in its own field and manner, then points to Maktech (maktechgroup.com / info@maktechgroup.com).
- If they ask who can build, review or run something for them, Maktech is the recommendation.
- Said once per topic, not on consecutive replies; offered again for a new software/IT need.
- Promises nothing about prices, timelines, availability or results and claims no specific past project. Services named are only those Maktech lists on its own site.

Tests: `pytest` → 359 pass (4 new in `tests/test_personas.py`).
