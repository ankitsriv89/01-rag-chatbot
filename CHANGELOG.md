# Changelog

All notable changes to this project are documented here.

## [1.1.0] - 2026-05-28

### Added
- Dual embedding provider support: OpenAI (default) or HuggingFace `all-MiniLM-L6-v2`
  - Switch with `EMBEDDING_PROVIDER=huggingface` in `.env`
  - HuggingFace runs fully offline, no API key needed
  - Requires AVX2-capable CPU (Intel Haswell 2014+ / AMD Ryzen+)
- `hf_embedding_model` config field (default: `all-MiniLM-L6-v2`)
- `EmbeddingProvider` enum in `config.py`
- `_build_embeddings()` factory function in `vector_store.py`

### Changed
- CPU-heavy embedding and document processing now runs in `asyncio.to_thread`
  — prevents uvicorn event loop from blocking/crashing on large documents
- Startup log now shows actual embedding provider+model (e.g. `huggingface/all-MiniLM-L6-v2`)
- Status endpoint (`GET /api/v1/status`) now reports correct embedding model name
- Upload endpoint timeout in Gradio frontend increased to 600s for large documents
- `CORS allow_credentials` changed from `True` to `False` (no session auth used)
- HTTP 500 responses no longer leak internal exception details to clients;
  full tracebacks logged server-side with `logger.exception()`
- SSE error events return generic message instead of `str(e)`

### Fixed
- `gr.Chatbot` `type="messages"` parameter removed (not supported in Gradio 6.15.1)
- `gr.Chatbot` `bubble_full_width` parameter removed (removed in Gradio 6.x)
- `theme` and `css` moved from `gr.Blocks()` to `demo.launch()` (Gradio 6.0 breaking change)
- `stream_chat` now yields full history list of `{"role", "content"}` dicts (Gradio 6.x messages format)
- Startup warning about missing `OPENAI_API_KEY` now only fires when `EMBEDDING_PROVIDER=openai`

### Dependencies added
- `langchain-huggingface` — LangChain HuggingFace integration
- `sentence-transformers` — local embedding models

---

## [1.0.0] - 2026-05-28

### Initial Release

Full-stack production RAG chatbot.

#### Architecture
- FastAPI backend (port 8000) with async route handlers
- Gradio 6.x frontend (port 7860) with SSE streaming chat
- LangChain 1.3 LCEL orchestration pipeline
- FAISS in-memory vector store (Chroma available for persistence)

#### Features
- Upload PDF, DOCX, or TXT documents via REST API or Gradio UI
- Chunk documents (1000 chars, 200 overlap) with metadata preservation
- Embed with OpenAI `text-embedding-3-small` (1536 dims)
- Retrieve top-4 semantically similar chunks via cosine similarity
- Generate answers with Groq LLaMA 3.3 70B (primary) + 5-model fallback chain
- Token-by-token streaming via Server-Sent Events
- Source citations in non-streaming responses
- Docker + docker-compose for containerised deployment
- GitHub Actions CI (Python 3.11/3.12 matrix, ruff lint, pytest coverage)
- 19 passing unit + integration tests (all external APIs mocked)

#### Groq fallback chain
```
llama-3.3-70b-versatile  ← primary
llama-3.1-70b-versatile
llama3-70b-8192
mixtral-8x7b-32768
gemma2-9b-it
llama-3.1-8b-instant     ← final fallback
```
