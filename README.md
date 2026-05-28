---
title: 01 RAG Chatbot
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Production RAG chatbot — FastAPI + Gradio + Groq + LangChain
---

# 📄 Production RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) system for document Q&A. Upload PDFs, DOCX, or TXT files and ask questions about them using LLMs.

[![CI](https://github.com/ankitsriv89/01-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/ankitsriv89/01-rag-chatbot/actions/workflows/ci.yml)

> **Deployed on Hugging Face Spaces** — a single Docker container runs FastAPI (`/api/v1/*`, `/docs`, `/health`) with Gradio mounted at `/` on port 7860 via `gr.mount_gradio_app`.

## Architecture

```
                  ┌─── port 7860 (single container) ───┐
User ──HTTP──▶│  Gradio UI  /                              │
                  │  FastAPI    /api/v1/*  /docs  /health     │
                  └────────────────────┬──────────────────┘
                                       ↓
                  OpenAI Embeddings  +  Groq LLM (LLaMA 3.3 70B + fallback chain)
                                       ↓
                  FAISS Vector Store (or Chroma for persistence)
```

Gradio is mounted on FastAPI via `gr.mount_gradio_app(app, demo, path="/")` so a
single Uvicorn process serves both the UI and the REST API on port 7860 — the
shape Hugging Face Spaces expects.

For local dev you can still split them (FastAPI on 8000, Gradio on 7860) by
running `uvicorn app.main:app` and `python frontend/app.py` separately with
`BACKEND_URL=http://localhost:7860`. `docker-compose up` also keeps this split.

## RAG Pipeline

| Step | What happens |
|------|-------------|
| **Ingest** | Upload PDF/DOCX/TXT → extract text → split into 1000-char chunks with 200-char overlap |
| **Embed** | Each chunk → OpenAI embeddings → 1536-dimensional vector |
| **Store** | Vectors stored in FAISS (in-memory) or Chroma (persistent) |
| **Retrieve** | User query embedded → top-4 most similar chunks fetched via cosine similarity |
| **Generate** | Retrieved chunks + question → LLaMA 3.3 70B via Groq → answer streamed back |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Backend | FastAPI + Uvicorn |
| LLM | Groq (LLaMA 3.3 70B) with 5-model fallback chain |
| Embeddings | OpenAI `text-embedding-3-small` (default) or HuggingFace `all-MiniLM-L6-v2` (free/local) |
| Vector Store | FAISS (dev) / Chroma (prod) |
| Orchestration | LangChain 1.3 (LCEL) |
| Frontend | Gradio 6.x |
| Containerisation | Docker + docker-compose |
| CI | GitHub Actions |

## Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone https://github.com/ankitsriv89/01-rag-chatbot
cd 01-rag-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# GROQ_API_KEY=gsk_...         (get free at console.groq.com)
# OPENAI_API_KEY=sk-...        (for embeddings — set EMBEDDING_PROVIDER=openai)
#
# To use free local embeddings instead (requires AVX2-capable CPU):
# EMBEDDING_PROVIDER=huggingface
# Model downloads ~90MB on first use, then cached.
```

### 3. Run the combined app (single process)

```bash
uvicorn app.main:app --reload --port 7860
# UI:      http://localhost:7860/
# Docs:    http://localhost:7860/docs
# Health:  http://localhost:7860/health
```

### Optional: split FastAPI and Gradio for development

```bash
# Terminal A
uvicorn app.main:app --reload --port 8000
# Terminal B
BACKEND_URL=http://localhost:7860 python frontend/app.py
# UI at http://localhost:7860, API at http://localhost:7860/docs
```

## Docker

```bash
cp .env.example .env   # fill in your keys
docker-compose up --build
# App: http://localhost:7860/ (UI), /docs (API), /health
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/upload` | Upload & index a document |
| `POST` | `/api/v1/query` | Ask a question (streaming or JSON) |
| `GET` | `/api/v1/status` | Vector store + LLM status |
| `DELETE` | `/api/v1/clear` | Reset vector store |
| `GET` | `/docs` | Interactive API docs (Swagger UI) |

### Example: Upload a document

```bash
curl -X POST http://localhost:7860/api/v1/upload \
  -F "file=@report.pdf"
```

### Example: Query (non-streaming with sources)

```bash
curl -X POST http://localhost:7860/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main finding?", "stream": false}'
```

### Example: Query (streaming SSE)

```bash
curl -X POST http://localhost:7860/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the document", "stream": true}'
```

## Running Tests

```bash
# With the project venv activated:
GROQ_API_KEY=gsk_test OPENAI_API_KEY=sk-test \
  python -m pytest tests/ -v --tb=short
```

All 19 tests pass. Tests mock all external APIs — no real keys needed.

## Groq Model Fallback Chain

The system automatically tries models in order if a rate limit or outage occurs:

```
llama-3.3-70b-versatile   ← primary (best quality)
llama-3.1-70b-versatile   ← backup 70B
llama3-70b-8192           ← stable alias
mixtral-8x7b-32768        ← 32k context (good for long docs)
gemma2-9b-it              ← Google Gemma 2
llama-3.1-8b-instant      ← fastest, always available
```

Override the primary model with `GROQ_MODEL=<model-id>` in `.env`.

## Deployment

### Hugging Face Spaces (this repo)

The Space [anksriv/01-rag-chatbot](https://huggingface.co/spaces/anksriv/01-rag-chatbot)
deploys directly from this repo:

1. **Docker SDK** declared in the README frontmatter (`sdk: docker`, `app_port: 7860`)
2. **One container** built from this `Dockerfile` runs FastAPI + Gradio on port 7860
3. **Secrets** set in Space Settings (NOT committed):
   - `GROQ_API_KEY` — required
   - `OPENAI_API_KEY` — required when `EMBEDDING_PROVIDER=openai`
   - any other vars from `.env.example` you want to override

Push to the `main` branch of the Space remote to trigger a rebuild.

### Cloud backend (GCP Cloud Run / AWS ECS)

```bash
docker build -t rag-chatbot .
docker tag rag-chatbot gcr.io/your-project/rag-chatbot
docker push gcr.io/your-project/rag-chatbot
# Deploy via GCP Console or `gcloud run deploy --port 7860`
```

## Embedding Provider

| Provider | Model | Cost | Requires | Notes |
|----------|-------|------|----------|-------|
| `openai` (default) | text-embedding-3-small | ~$0.00002/1K tokens | `OPENAI_API_KEY` | Fast, cloud, works on any CPU |
| `huggingface` | all-MiniLM-L6-v2 | Free | AVX2 CPU | 90MB one-time download, runs offline |

Set `EMBEDDING_PROVIDER=huggingface` in `.env` to switch. Note: HuggingFace requires a CPU with AVX2 instruction support (Intel Haswell 2013+ / AMD Ryzen+).

## Key Concepts Demonstrated

- **RAG architecture**: retrieval-augmented generation end-to-end
- **Vector embeddings**: semantic search with cosine similarity
- **LangChain LCEL**: composable chain pipelines with `|` operator
- **Async FastAPI**: non-blocking I/O for LLM calls
- **SSE streaming**: real-time token streaming via Server-Sent Events
- **Pydantic v2**: schema-first API design with auto-validation
- **Production fallback**: multi-model resilience on pay-as-you-go APIs
- **Docker multi-stage builds**: minimal production images
- **CI/CD**: GitHub Actions with pytest coverage gates
