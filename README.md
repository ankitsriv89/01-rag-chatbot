# 📄 Production RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) system for document Q&A. Upload PDFs, DOCX, or TXT files and ask questions about them using LLMs.

[![CI](https://github.com/your-username/01-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/01-rag-chatbot/actions/workflows/ci.yml)

## Architecture

```
User → Gradio UI (port 7860)
            ↓ HTTP
      FastAPI Backend (port 8000)
            ↓               ↓
   OpenAI Embeddings    Groq LLM (LLaMA 3.3 70B)
   text-embedding-3-small    + fallback chain
            ↓
      FAISS Vector Store
      (or Chroma for persistence)
```

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
git clone https://github.com/your-username/01-rag-chatbot
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

### 3. Start the backend

```bash
uvicorn app.main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### 4. Start the frontend (new terminal)

```bash
pip install gradio httpx
BACKEND_URL=http://localhost:8000 python frontend/app.py
# UI at http://localhost:7860
```

## Docker (Recommended)

```bash
cp .env.example .env   # fill in your keys
docker-compose up --build
# Backend: http://localhost:8000/docs
# Frontend: http://localhost:7860
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
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@report.pdf"
```

### Example: Query (non-streaming with sources)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main finding?", "stream": false}'
```

### Example: Query (streaming SSE)

```bash
curl -X POST http://localhost:8000/api/v1/query \
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

### Hugging Face Spaces (frontend)

1. Create a new Space with Gradio SDK
2. Upload `frontend/app.py` and a `requirements.txt` with `gradio httpx`
3. Add `BACKEND_URL` as a Space Secret pointing to your deployed backend

### Cloud backend (GCP Cloud Run / AWS ECS)

```bash
docker build -t rag-chatbot .
docker tag rag-chatbot gcr.io/your-project/rag-chatbot
docker push gcr.io/your-project/rag-chatbot
# Deploy via GCP Console or `gcloud run deploy`
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
