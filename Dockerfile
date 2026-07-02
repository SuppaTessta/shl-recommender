# Slim image; no ML model weights to bake in (TF-IDF fits in-process from
# the ~440KB catalog JSON at startup), so the image stays small and cold
# start stays comfortably under the assignment's 2-minute health-check grace
# period.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

ENV PORT=8000
EXPOSE 8000

# GROQ_API_KEY is read from the platform's env var settings at runtime, not
# baked into the image. Without it, get_llm_client() falls back to the mock
# backend — the service still boots and /health still passes, but /chat
# degrades to a fixed reply, so deployment misconfiguration is visible
# immediately rather than crash-looping.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
