# SHL Assessment Recommender

Conversational agent that recommends SHL Individual Test Solutions through
dialogue. FastAPI service, stateless `/chat`, backed by TF-IDF + category retrieval over the
scraped SHL catalog and Groq's free-tier `llama-3.3-70b-versatile`.

See `APPROACH.md` for design rationale, trade-offs, and what didn't work.

## Project layout

```
app/
  catalog.py     # loads the scraped catalog, name/URL lookups
  retrieval.py   # TF-IDF index + candidate-pool construction
  history.py     # stateless state reconstruction from prior replies
  guardrails.py  # cheap pre-LLM injection filter
  prompts.py     # the two-call (route / generate) prompt templates
  llm.py         # Groq client + deterministic mock for offline testing
  agent.py       # orchestrates one /chat turn end-to-end
  schemas.py     # Pydantic request/response contracts
  main.py        # FastAPI app: GET /health, POST /chat
data/
  catalog_raw.json   # scraped SHL Individual Test Solutions catalog (377 items)
results/
  eval_retrieval.txt # output of tests/eval_retrieval.py
tests/
  traces/             # the 10 provided gold conversation traces
  eval_retrieval.py   # offline pool-coverage measurement against the traces
  test_http_pipeline.py  # replays all 10 traces through the real FastAPI app (in-process TestClient)
```

## Quickstart

Requires Python 3.12 (Docker is optional).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

In a second terminal:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders"}]
}'
```

Endpoints: `GET /health` and `POST /chat` (stateless: send the full message
history each call; the reply has `reply`, `recommendations`, `end_of_conversation`).

### LLM backend

The app reads `GROQ_API_KEY` from the process environment (`.env` is not
loaded automatically). To use the real model (Groq free tier, key at
https://console.groq.com/keys), set it before starting the server:

```bash
export GROQ_API_KEY=gsk_...
```

Without a key the server still boots and uses a deterministic mock LLM
(a startup warning says so). The mock exercises the full pipeline but is not
conversational: it asks for more detail on the first turn, then returns a
heuristic shortlist. The results below were produced with the mock; live
Groq behaviour was not measured here.

## Tests

Neither script needs an API key or network access:

```bash
python tests/eval_retrieval.py      # retrieval pool-coverage vs. the 10 gold traces
python tests/test_http_pipeline.py  # replays all 10 traces through the FastAPI app (TestClient)
```

`test_http_pipeline.py` asserts schema compliance, catalog-only URLs, and
that the degraded-mode fallback reply never appears.

## Results

Measured by running the code in this repo:

- Catalog: 377 items (66 Personality & Behavior, 32 Ability & Aptitude).
- API: 2 endpoints (`GET /health`, `POST /chat`).
- End-to-end: 10/10 gold traces pass (38 user turns).
- Final-shortlist recall ("pool-coverage") against the gold traces:
  0.535 raw TF-IDF top-10, 0.782 top-20 + default companions, 0.747 for the
  production pool (about 14 items), 0.767 with hand-written query-expansion
  hints. Full output: `results/eval_retrieval.txt`.
- `/chat` median latency with the mock LLM: 39 ms (excludes real LLM latency).

## Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... shl-recommender
```

The `Dockerfile` is provided for self-hosting; it was not built or run as
part of this verification, and there is no hosted deployment of this project.

## License

MIT, see `LICENSE`.
