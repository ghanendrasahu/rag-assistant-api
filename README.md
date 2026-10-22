# RAG Assistant API

A document-intelligence service: ingest PDF/TXT documents, embed them into a
**persistent** Chroma collection, and answer questions with source attribution
through a Groq-hosted LLM.

The store is persistent on purpose — an ephemeral `chromadb.Client()` loses
the whole index on restart and breaks under multiple workers.

## Highlights

- **Persistent vector store** — `PersistentClient` with a configurable
  `CHROMA_PERSIST_PATH` (default `data/chroma`), so the index survives
  restarts and can live on a shared volume.
- **Citation-grade answers** — every chunk returned with its source
  document and score.
- **Structured logging** — JSONL events with a request id on every call.
- **Graceful LLM degradation** — the API answers with the retrieved context
  when the LLM is unavailable, instead of failing the request.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

uvicorn rag_assistant.api.main:app --reload
```

Configuration (`.env` or environment):

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | LLM access | – |
| `CHROMA_PERSIST_PATH` | Where the index lives | `data/chroma` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | Readiness |
| `POST` | `/api/ingest` | Ingest a document (PDF/TXT) |
| `POST` | `/api/query` | Ask a question, get an answer + sources |
| `GET` | `/api/documents` | List ingested documents |

Example:

```bash
curl -X POST localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our refund policy?"}'
```

## Layout

```text
rag_assistant/
  api/main.py          FastAPI app and routes
  core/                generation, retrieval and the RAG pipeline
  services/            embeddings, LLM client, vector store
  utils/               logging, request context
tests/                 offline, deterministic tests
```

## Testing

```bash
python -m pytest tests -q     # offline; no network or API key required
ruff check rag_assistant tests
```

CI runs Black-style linting (ruff) and the test suite on every push.

## Deployment notes

- Mount the Chroma directory on a **shared volume** when running more than
  one replica, and point `CHROMA_PERSIST_PATH` at it.
- Keep `GROQ_API_KEY` in the environment (or a secret store) — never in the
  image.
