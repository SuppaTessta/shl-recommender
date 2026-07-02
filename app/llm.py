"""
Thin LLM abstraction so the agent logic never talks to a provider SDK
directly. Two implementations:

- GroqClient: real inference via Groq's OpenAI-compatible endpoint
  (https://api.groq.com/openai/v1), free tier, model=llama-3.3-70b-versatile.
  Chosen over Gemini/OpenRouter free tiers for raw speed — Groq's LPU
  hardware runs well under the 30s per-call budget even for two sequential
  calls per turn, and its free tier (30 RPM / 14,400 RPD, no card) comfortably
  covers an 8-turn-capped conversation.
- MockLLMClient: returns scripted/rule-based JSON so the FastAPI pipeline
  (routing, schema validation, catalog grounding, turn-cap logic) can be
  exercised and unit-tested with zero network calls and zero API cost. This
  is what tests/test_pipeline.py runs against.

Both implementations expose the same `complete_json(system, messages,
schema_hint) -> dict` contract.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod


class LLMError(Exception):
    """Raised when the provider fails or returns unparseable output.
    Caught at the agent layer so a provider hiccup degrades to a safe
    fallback reply instead of a raw 500."""


class LLMClient(ABC):
    @abstractmethod
    def complete_json(self, system: str, user: str) -> dict:
        """Send a single-turn structured-output request; return parsed JSON.
        `user` carries the full serialized context (history + task-specific
        instructions) since Groq's JSON mode is simplest as one user turn
        rather than replaying our internal multi-message state."""


class GroqClient(LLMClient):
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str | None = None, timeout: float = 12.0):
        # Lazy import: keeps `openai` off the import path for anyone only
        # running the mock-backed test suite.
        from openai import OpenAI

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                "and export it, or pass api_key= explicitly."
            )
        self._client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        self._timeout = timeout

    def complete_json(self, system: str, user: str) -> dict:
        try:
            resp = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=1500,
                timeout=self._timeout,
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"Model returned non-JSON output: {e}") from e
        except Exception as e:  # noqa: BLE001 — provider errors are all "LLM unavailable" to us
            raise LLMError(f"Groq request failed: {e}") from e


class MockLLMClient(LLMClient):
    """Deterministic stand-in for offline pipeline testing. Behavior is
    driven by simple keyword heuristics over the (already-formatted) prompt
    text, just enough to exercise every branch in agent.py without a real
    model. Not intended to produce SHL-quality conversational judgment —
    only valid, schema-correct JSON shaped like what Groq would return, so
    that integration tests can verify the *pipeline* (schema compliance,
    history round-tripping, turn-cap handling, catalog validation) is
    robust independent of prompt/model quality."""

    REFUSAL_MARKERS = ("legally required", "am i violating", "salary negotiation", "ignore all previous")

    def __init__(self, script: list[dict] | None = None):
        # If a script is provided, pop responses off it in order (lets a
        # test pin exact outputs per call). Otherwise fall back to heuristics.
        self._script = list(script) if script else None

    def complete_json(self, system: str, user: str) -> dict:
        if self._script is not None:
            if not self._script:
                raise LLMError("MockLLMClient script exhausted")
            return self._script.pop(0)
        if "call_type: route" in user.lower():
            return self._heuristic_route(user)
        if "call_type: generate" in user.lower():
            return self._heuristic_generate(user)
        raise LLMError("MockLLMClient: unrecognized call_type in prompt")

    @classmethod
    def _heuristic_route(cls, user: str) -> dict:
        lower = user.lower()
        last_user_line = [l for l in user.splitlines() if l.startswith("USER:")]
        latest = last_user_line[-1].lower() if last_user_line else lower

        if any(m in latest for m in cls.REFUSAL_MARKERS):
            return {"intent": "refuse", "refusal_reason": "out_of_scope", "clarifying_question": None,
                    "compare_items": None, "retrieval_query": None, "excluded_names": [],
                    "test_type_focus": [], "personality_ability_hint": None, "job_level": None,
                    "language": None, "user_confirmed_shortlist": False}

        if "difference between" in latest or "what's the difference" in latest or "different from" in latest:
            # crude name extraction: split on " and "/" vs "/"between...and"
            import re as _re
            names = _re.split(r"\band\b|\bvs\.?\b", latest)
            return {"intent": "compare", "refusal_reason": None, "clarifying_question": None,
                    "compare_items": [n.strip(" ?.") for n in names if n.strip()][:3],
                    "retrieval_query": None, "excluded_names": [], "test_type_focus": [],
                    "personality_ability_hint": None, "job_level": None, "language": None,
                    "user_confirmed_shortlist": False}

        n_user_turns = user.count("USER:")
        history_len = len(user)
        if n_user_turns <= 1 and history_len < 500:
            return {"intent": "clarify", "refusal_reason": None,
                    "clarifying_question": "Could you tell me more about the role, level, and what matters most?",
                    "compare_items": None, "retrieval_query": None, "excluded_names": [],
                    "test_type_focus": [], "personality_ability_hint": None, "job_level": None,
                    "language": None, "user_confirmed_shortlist": False}

        confirmed = any(w in latest for w in ("confirmed", "that works", "perfect", "locking it in",
                                               "sounds good", "keep it as-is", "keep the shortlist"))
        excluded = []
        if "drop" in latest or "remove" in latest:
            excluded = ["Occupational Personality Questionnaire OPQ32r"]
        return {"intent": "recommend", "refusal_reason": None, "clarifying_question": None,
                "compare_items": None, "retrieval_query": user[-500:], "excluded_names": excluded,
                "test_type_focus": [], "personality_ability_hint": None, "job_level": None,
                "language": None, "user_confirmed_shortlist": confirmed}

    @staticmethod
    def _heuristic_generate(user: str) -> dict:
        import json as _json
        import re as _re
        m = _re.search(r"CANDIDATE_POOL: (\[.*?\])\n\nCONVERSATION", user, _re.DOTALL)
        urls = []
        if m:
            try:
                pool = _json.loads(m.group(1))
                urls = [p["url"] for p in pool[:6]]
            except Exception:
                urls = []
        if "INTENT: refuse" in user:
            return {"reply": "I can only help with SHL assessment selection.", "selected_urls": None}
        if "INTENT: clarify" in user:
            return {"reply": "Could you share a bit more detail?", "selected_urls": None}
        if "INTENT: compare" in user:
            return {"reply": "Here's how those two differ based on the catalog data.", "selected_urls": None}
        return {"reply": "Here's a shortlist based on your requirements.", "selected_urls": urls or None}


def get_llm_client() -> LLMClient:
    """Factory used by the FastAPI app. Falls back to the mock if no key is
    configured, so `uvicorn app.main:app` still boots (health check passes)
    in an environment without secrets — but /chat will report a clear
    degraded-mode error rather than silently mocking real user traffic."""
    if os.environ.get("GROQ_API_KEY"):
        return GroqClient()
    return MockLLMClient()
