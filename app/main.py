"""
FastAPI entrypoint. Catalog + retrieval index load once at import time (not
per-request) so cold start is bounded and each /chat call only pays for LLM
latency, not index rebuild. The outer try/except in /chat is the last line
of defense: whatever goes wrong internally, the response leaving this
service must always be schema-valid JSON, because a raw 500 or a malformed
body fails the hard eval regardless of how good the reasoning was upstream.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI

from .agent import Agent, DEGRADED_REPLY
from .catalog import CatalogStore
from .llm import get_llm_client
from .retrieval import RetrievalIndex
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog_raw.json"

app = FastAPI(title="SHL Assessment Recommender")

_store = CatalogStore.from_file(CATALOG_PATH)
_index = RetrievalIndex(_store)
_agent = Agent(_store, _index, get_llm_client())
logger.info("Catalog loaded: %d assessments", len(_store))
if not os.environ.get("GROQ_API_KEY"):
    logger.warning(
        "GROQ_API_KEY is not set — /chat is running on the MOCK LLM backend, which returns "
        "schema-valid but non-conversational placeholder replies. Set GROQ_API_KEY (free key at "
        "https://console.groq.com/keys) before deploying for real traffic."
    )


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return _agent.handle(request.messages)
    except Exception:  # noqa: BLE001 — last-resort guard, must never bubble as a 500
        logger.exception("Unhandled error in /chat")
        return ChatResponse(reply=DEGRADED_REPLY, recommendations=None, end_of_conversation=False)
