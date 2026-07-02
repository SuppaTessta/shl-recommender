"""
Drives the REAL FastAPI app (via TestClient, real HTTP-shaped request/
response cycle, real Pydantic validation) through every user turn of all 10
gold traces, using the heuristic MockLLMClient in place of Groq. This won't
validate conversational *quality* (the mock doesn't reason like a real
model), but it validates the thing most likely to silently break in
production: does the full stack — routing, retrieval, history
reconstruction, catalog validation, schema assembly — survive a real
multi-turn conversation shaped like what the actual evaluator will send,
without crashing or emitting invalid JSON, across 8+ consecutive turns per
trace with growing history each time.

This is the test suite's answer to the assignment's explicit warning about
"insufficient evaluation rigor... testing realistic conversation patterns."
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.pop("GROQ_API_KEY", None)  # force mock backend regardless of shell env

from fastapi.testclient import TestClient

from app.agent import DEGRADED_REPLY, Agent
from app.catalog import CatalogStore
from app.llm import MockLLMClient
from app.main import app
from app.retrieval import RetrievalIndex

TRACE_DIR = Path(__file__).resolve().parent / "traces"
USER_TURN_RE = re.compile(r"\*\*User\*\*\s*\n\n>\s*(.+?)(?=\n\n\*\*Agent\*\*)", re.DOTALL)


def extract_user_turns(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [re.sub(r"\s+", " ", m.strip()) for m in USER_TURN_RE.findall(text)]


def replay_trace(client: TestClient, path: Path) -> None:
    user_turns = extract_user_turns(path)
    assert user_turns, f"{path.name}: no user turns parsed — check the trace format regex"

    history: list[dict] = []
    for i, user_text in enumerate(user_turns, start=1):
        history.append({"role": "user", "content": user_text})
        resp = client.post("/chat", json={"messages": history})

        assert resp.status_code == 200, f"{path.stem} turn {i}: HTTP {resp.status_code} — {resp.text[:300]}"
        body = resp.json()

        assert body["reply"] != DEGRADED_REPLY, (
            f"{path.stem} turn {i}: agent hit its internal-error fallback (see server logs above for "
            f"the real traceback) — this is a functional bug masked by the outer try/except, not a "
            f"legitimate degraded response"
        )

        assert "reply" in body and isinstance(body["reply"], str) and body["reply"], (
            f"{path.stem} turn {i}: empty or missing reply"
        )
        assert "recommendations" in body, f"{path.stem} turn {i}: missing recommendations key"
        recs = body["recommendations"]
        assert recs is None or (isinstance(recs, list) and 1 <= len(recs) <= 10), (
            f"{path.stem} turn {i}: recommendations violates null-or-1-to-10 contract: {recs}"
        )
        if recs:
            for r in recs:
                assert {"name", "url", "test_type"} <= r.keys(), f"{path.stem} turn {i}: malformed recommendation {r}"
                assert r["url"].startswith("https://www.shl.com/products/product-catalog/view/"), (
                    f"{path.stem} turn {i}: URL not from catalog domain: {r['url']}"
                )
        assert isinstance(body["end_of_conversation"], bool), f"{path.stem} turn {i}: end_of_conversation not bool"

        history.append({"role": "assistant", "content": body["reply"]})

        assert len(history) <= 16, (
            f"{path.stem}: exceeded a generous 16-message safety ceiling at turn {i} — "
            f"turn-cap forcing logic may not be engaging"
        )


def main():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "ok"
    print("PASS  /health")

    failures = []
    for path in sorted(TRACE_DIR.glob("C*.md"), key=lambda p: int(p.stem[1:])):
        try:
            replay_trace(client, path)
            print(f"PASS  {path.stem}  ({len(extract_user_turns(path))} user turns replayed via real HTTP)")
        except AssertionError as e:
            failures.append(str(e))
            print(f"FAIL  {path.stem}: {e}")

    print(f"\n{len(failures)} failure(s) out of 10 traces" if failures else "\nALL 10 TRACES PASSED END-TO-END")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
