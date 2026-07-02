# SHL Assessment Recommender

Conversational agent that recommends SHL Individual Test Solutions through
dialogue, built for the AI Research Intern take-home assignment. FastAPI
service, stateless `/chat`, backed by TF-IDF + category retrieval over the
scraped SHL catalog and Groq's free-tier `llama-3.3-70b-versatile`.

See `APPROACH.md` for design rationale, trade-offs, and what didn't work.

## Project layout

```
app/
  catalog.py     # loads + repairs the scraped catalog, name/URL lookups
  retrieval.py   # TF-IDF index + candidate-pool construction
  history.py     # stateless state reconstruction from prior replies
  guardrails.py  # cheap pre-LLM injection filter
  prompts.py     # the two-call (route / generate) prompt templates
  llm.py         # Groq client + deterministic mock for offline testing
  agent.py       # orchestrates one /chat turn end-to-end
  schemas.py     # Pydantic request/response contracts
  main.py        # FastAPI app: GET /health, POST /chat
data/
  catalog_raw.json   # scraped SHL Individual Test Solutions catalog
tests/
  traces/             # the 10 provided gold conversation traces
  eval_retrieval.py   # offline pool-coverage measurement against the traces
  test_http_pipeline.py  # replays all 10 traces through the real FastAPI app
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY — free, no card, at
                        # https://console.groq.com/keys
```

## Run locally

```bash
export GROQ_API_KEY=gsk_...
uvicorn app.main:app --reload --port 8000
```

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders"}]
}'
```

Without `GROQ_API_KEY` set, the service still boots and `/health` still
passes, but `/chat` runs on a deterministic mock backend (a startup log
warns loudly about this) — useful for testing the pipeline without burning
API calls, not for real conversations.

## Tests

Two independent test suites, neither requires a live LLM call:

```bash
python3 tests/eval_retrieval.py      # retrieval-only pool-coverage vs. the 10 gold traces
python3 tests/test_http_pipeline.py  # full HTTP replay of all 10 traces through the real app
```

`test_http_pipeline.py` is the more important one for catching integration
bugs — it drives the actual FastAPI app over real HTTP with a scripted mock
LLM, replaying every user turn of all 10 traces and asserting schema
compliance, catalog-only URLs, and no internal errors at every step.

## Deployment (Render / Fly / Railway — any free-tier Docker host)

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... shl-recommender
```

On the platform: set `GROQ_API_KEY` as an environment variable/secret, point
it at this `Dockerfile`. No other build config needed — no model weights to
download, the catalog loads from the bundled JSON in milliseconds, so cold
start is fast well within the assignment's 2-minute health-check grace
period.
