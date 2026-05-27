"""
Configuration Management
========================
Uses Pydantic Settings to load all config from environment variables or a .env file.

WHY this pattern matters in production:
- Secrets never appear in source code
- Different values per environment (dev/staging/prod) without code changes
- Validation at startup — app crashes immediately if a required key is missing
  rather than failing silently on the first API call
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers — extend this as you add more."""
    OPENAI = "openai"
    GROQ = "groq"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"       # text-embedding-3-small — requires OPENAI_API_KEY
    HUGGINGFACE = "huggingface"  # all-MiniLM-L6-v2 — free, runs locally


class VectorStoreType(str, Enum):
    """Supported vector stores — FAISS for dev, Chroma for production persistence."""
    FAISS = "faiss"
    CHROMA = "chroma"


class Settings(BaseSettings):
    """
    Central settings class. All values come from environment variables.
    Pydantic automatically reads from a .env file if present.

    Field() lets us set defaults and descriptions — this doubles as documentation.
    """

    # ── App Identity ────────────────────────────────────────────────────────
    app_name: str = Field(default="RAG Chatbot", description="Application name")
    app_version: str = Field(default="1.0.0", description="API version")
    debug: bool = Field(default=False, description="Enable debug mode (verbose logging)")

    # ── LLM Provider Selection ───────────────────────────────────────────────
    # Change LLM_PROVIDER=openai in .env to switch without touching code
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description="Which LLM provider to use: 'groq' or 'openai'"
    )

    # ── API Keys ─────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key (for GPT-4o + embeddings)")
    groq_api_key: str = Field(default="", description="Groq API key (for LLaMA 3 — free tier)")

    # ── Model Selection ──────────────────────────────────────────────────────
    # Primary Groq model — override with GROQ_MODEL in .env
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Primary Groq model ID")

    # Fallback chain tried in order when the primary Groq model fails (rate limit / outage).
    # CONCEPT — Fallback strategy: each model in the list is tried until one succeeds.
    # 70B → 70B-alt → Mixtral (long context) → 8B (fastest, always available)
    groq_fallback_models: list[str] = Field(
        default=[
            "llama-3.1-70b-versatile",   # Same tier, different checkpoint
            "llama3-70b-8192",            # Stable alias — 8k context
            "mixtral-8x7b-32768",         # 32k context — good for long documents
            "gemma2-9b-it",               # Google Gemma 2 — solid mid-tier
            "llama-3.1-8b-instant",       # Fastest, lowest cost — final fallback
        ],
        description="Ordered list of Groq models to try if the primary fails"
    )

    # OpenAI models: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model ID")

    # Embedding provider — switch with EMBEDDING_PROVIDER in .env
    # 'openai' requires OPENAI_API_KEY; 'huggingface' is free and runs locally
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.HUGGINGFACE,
        description="Embedding provider: 'openai' or 'huggingface'"
    )
    # OpenAI embedding model (used only when embedding_provider=openai)
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    # HuggingFace embedding model (used when embedding_provider=huggingface)
    # all-MiniLM-L6-v2: 384 dims, ~90MB download, fast CPU inference, no API key
    hf_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model name"
    )

    # ── RAG Pipeline Parameters ──────────────────────────────────────────────
    # CONCEPT: Chunking splits documents into overlapping pieces.
    # chunk_size = how many characters per chunk
    # chunk_overlap = how many chars the next chunk repeats from the previous
    # Overlap ensures context isn't lost at chunk boundaries.
    chunk_size: int = Field(default=1000, description="Document chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between consecutive chunks")

    # How many chunks to retrieve per query (k in top-k retrieval)
    # More = more context but higher token cost and potential noise
    retrieval_top_k: int = Field(default=4, description="Number of chunks to retrieve")

    # ── Vector Store ─────────────────────────────────────────────────────────
    vector_store_type: VectorStoreType = Field(
        default=VectorStoreType.FAISS,
        description="Vector store backend: 'faiss' (in-memory) or 'chroma' (persistent)"
    )
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory to persist Chroma vector store"
    )

    # ── API Rate Limiting ────────────────────────────────────────────────────
    # Prevents abuse and controls costs in production
    max_requests_per_minute: int = Field(default=20, description="Rate limit per client IP")
    max_file_size_mb: int = Field(default=50, description="Max upload file size in MB")

    # ── Pydantic Config ──────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",          # Read from .env file automatically
        env_file_encoding="utf-8",
        case_sensitive=False,     # LLM_PROVIDER and llm_provider both work
        extra="ignore",           # Ignore unknown env vars (don't crash)
    )


# Singleton: import `settings` everywhere instead of creating new instances
# This ensures config is loaded once at startup, not on every request
settings = Settings()
