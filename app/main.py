"""
FastAPI Application Entry Point
================================
Wires together the app: middleware, routers, startup/shutdown events, health check.

CONCEPT — Application Lifecycle:
  FastAPI has lifespan events: startup and shutdown.
  Startup = initialize expensive resources once (DB connections, model loading).
  Shutdown = clean up gracefully (close connections, flush logs).
  Using @asynccontextmanager is the modern FastAPI pattern (replaces @app.on_event).

CONCEPT — CORS (Cross-Origin Resource Sharing):
  Browsers block requests from one domain to another by default.
  Our Gradio frontend (port 7860) talks to FastAPI (port 8000) — different origins.
  CORSMiddleware tells the browser "these origins are allowed to call this API".
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.config import settings
from app.api.routes import router


# ── Logging Setup ──────────────────────────────────────────────────────────────
# Loguru intercepts all logging and formats it with colors + structured output.
# Remove the default handler, add our configured one.
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.debug else "INFO",
    colorize=True,
)


# ── Lifespan Context Manager ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before yield, shutdown logic after yield.
    FastAPI calls this automatically when the server starts and stops.
    """
    # ── Startup ──
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"LLM Provider : {settings.llm_provider.value}")
    logger.info(f"Vector Store : {settings.vector_store_type.value}")
    emb_info = (
        f"huggingface/{settings.hf_embedding_model}"
        if settings.embedding_provider.value == "huggingface"
        else f"openai/{settings.embedding_model}"
    )
    logger.info(f"Embedding    : {emb_info}")
    logger.info(f"Debug mode   : {settings.debug}")

    # Validate that required API keys are set before accepting traffic
    if settings.llm_provider.value == "groq" and not settings.groq_api_key:
        logger.error("GROQ_API_KEY is not set. Set it in your .env file.")
        raise RuntimeError("Missing GROQ_API_KEY")

    if settings.embedding_provider.value == "openai" and not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set but embedding_provider=openai — embeddings will fail.")

    logger.info("Application startup complete. Ready to accept requests.")

    yield  # Application runs here

    # ── Shutdown ──
    logger.info("Application shutting down. Cleaning up resources...")


# ── FastAPI App Instance ───────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Production RAG Chatbot API

Upload documents (PDF, DOCX, TXT) and ask questions about them using LLMs.

### Features
- **Document Ingestion**: Upload → chunk → embed → store in vector DB
- **Semantic Retrieval**: Find relevant context using cosine similarity
- **LLM Generation**: Answer using Groq (LLaMA 3) or OpenAI (GPT-4o)
- **Streaming**: Token-by-token streaming via Server-Sent Events
- **Source Citations**: Every answer includes which document chunks it used

### Providers
- **LLM**: Groq (LLaMA 3.3 70B) — primary | OpenAI (GPT-4o) — fallback
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector Store**: FAISS (dev) / Chroma (production)
    """,
    docs_url="/docs",         # Swagger UI at /docs
    redoc_url="/redoc",       # ReDoc UI at /redoc
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ── CORS Middleware ────────────────────────────────────────────────────────────
# CONCEPT: In production, replace ["*"] with your actual frontend domain.
# ["*"] means any origin can call this API — fine for dev/demo, not for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Replace with ["https://yourapp.hf.space"] in prod
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────
# Mount all routes from routes.py under the /api/v1 prefix.
# Versioning (/v1/) lets you release breaking changes as /v2/ without breaking clients.
app.include_router(router, prefix="/api/v1", tags=["RAG Pipeline"])


# ── Root Health Check ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Simple health check — load balancers and uptime monitors ping this."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check for monitoring systems (Prometheus, Datadog, etc.)."""
    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider.value,
        "vector_store": settings.vector_store_type.value,
    }
