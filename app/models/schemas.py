"""
Pydantic Schemas (API Data Contracts)
======================================
Defines the shape of all data flowing in and out of the API.

CONCEPT — Why Pydantic schemas?
  FastAPI uses these models to:
  1. Validate incoming request bodies (wrong types → 422 error, auto-generated)
  2. Serialize outgoing responses (Python objects → JSON, auto-generated)
  3. Generate OpenAPI docs at /docs (free documentation, auto-generated)

  This pattern is called "schema-first" design. The schema IS the contract
  between your API and its clients (frontend, other services, etc.).
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def utcnow() -> datetime:
    """Timezone-aware UTC now — replaces the deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


# ── Upload Endpoint ────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned after a successful document upload and indexing."""
    success: bool
    filename: str
    chunks_created: int = Field(description="Number of text chunks indexed in the vector store")
    total_chunks_in_store: int = Field(description="Total chunks across all uploaded documents")
    message: str

    model_config = {"json_schema_extra": {
        "example": {
            "success": True,
            "filename": "annual_report.pdf",
            "chunks_created": 48,
            "total_chunks_in_store": 48,
            "message": "Document indexed successfully. Ready to answer questions."
        }
    }}


# ── Query Endpoint ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="The question to ask about the uploaded documents"
    )
    stream: bool = Field(
        default=True,
        description="If True, response streams token-by-token. If False, returns complete JSON answer."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What were the key financial results in Q3?",
            "stream": False
        }
    }}


class SourceChunk(BaseModel):
    """A single retrieved document chunk used as evidence for the answer."""
    content: str = Field(description="The text of this chunk")
    source_filename: str = Field(description="Which uploaded file this came from")
    page: Optional[int] = Field(default=None, description="Page number (for PDFs)")
    relevance_score: float = Field(description="Cosine similarity score (0-1, higher = more relevant)")
    chunk_index: int = Field(description="Position of this chunk in the full document")


class QueryResponse(BaseModel):
    """Returned after a successful (non-streaming) query."""
    question: str
    answer: str
    sources: List[SourceChunk] = Field(description="The document chunks used to generate this answer")
    model_used: str = Field(description="Which LLM generated this answer")
    timestamp: datetime = Field(default_factory=utcnow)

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What is the company's revenue?",
            "answer": "According to the annual report (page 12), total revenue was $4.2B in FY2024.",
            "sources": [
                {
                    "content": "Total revenue for FY2024 was $4.2 billion...",
                    "source_filename": "annual_report.pdf",
                    "page": 12,
                    "relevance_score": 0.91,
                    "chunk_index": 34
                }
            ],
            "model_used": "llama-3.3-70b-versatile",
            "timestamp": "2024-01-15T10:30:00+00:00"
        }
    }}


# ── Status Endpoint ────────────────────────────────────────────────────────────

class StoreStatus(BaseModel):
    """Current state of the vector store — shown on the /status endpoint."""
    is_ready: bool = Field(description="True if at least one document has been uploaded")
    document_count: int = Field(description="Total chunks indexed in the vector store")
    vector_store_type: str = Field(description="Backend being used: faiss or chroma")
    llm_provider: str = Field(description="Active LLM provider: groq or openai")
    llm_model: str = Field(description="Specific model being used for generation")
    embedding_model: str = Field(description="Model used for creating embeddings")


# ── Error Response ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error format — returned for all 4xx/5xx responses."""
    error: str = Field(description="Human-readable error message")
    detail: Optional[str] = Field(default=None, description="Technical details for debugging")
    timestamp: datetime = Field(default_factory=utcnow)
