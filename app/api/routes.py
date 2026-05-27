"""
FastAPI Route Handlers
======================
Thin API layer: validate request → call core logic → return structured response.

CONCEPT — REST API design:
  POST /upload   → Upload and index a document
  POST /query    → Ask a question (streaming or non-streaming)
  GET  /status   → Check vector store state and system config
  DELETE /clear  → Wipe the vector store

CONCEPT — Async route handlers:
  All handlers are `async def`. While awaiting LLM calls (1-5s),
  the FastAPI event loop handles other incoming requests — no thread blocking.

CONCEPT — Streaming (SSE):
  StreamingResponse with media_type="text/event-stream" pushes
  data: {json}\n\n lines to the browser as tokens arrive.
  This is how ChatGPT-style "typing" effect works.
"""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger

from app.config import settings
from app.core.document_processor import process_upload
from app.core.vector_store import vector_store_manager
from app.core.rag_chain import invoke_with_fallback, stream_with_fallback
from app.models.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StoreStatus,
    ErrorResponse,
)

router = APIRouter()


# ── POST /upload ───────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    description="Accepts PDF, DOCX, or TXT. Chunks the text, embeds with OpenAI, stores in vector DB.",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported file type or empty document"},
        413: {"model": ErrorResponse, "description": "File exceeds size limit"},
        500: {"model": ErrorResponse, "description": "Processing or embedding error"},
    }
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT file to index")
):
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    file_bytes = await file.read()

    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {len(file_bytes)/1024/1024:.1f}MB. Limit is {settings.max_file_size_mb}MB."
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    logger.info(f"Upload received: '{file.filename}' ({len(file_bytes)/1024:.1f} KB)")

    try:
        chunks = process_upload(file_bytes, file.filename)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text extracted. Is it a scanned/image-only PDF?"
            )

        total_in_store = vector_store_manager.add_documents(chunks)

        return UploadResponse(
            success=True,
            filename=file.filename,
            chunks_created=len(chunks),
            total_chunks_in_store=total_in_store,
            message="Document indexed successfully. Ready to answer questions."
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Upload failed for '{file.filename}'")
        raise HTTPException(status_code=500, detail="Processing error. See server logs.")


# ── POST /query ────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    summary="Ask a question about uploaded documents",
    description=(
        "Retrieves relevant chunks from the vector store, then generates an answer "
        "using the configured LLM. Set `stream=true` for token-by-token SSE streaming. "
        "Set `stream=false` for a complete JSON response with source citations."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "No documents uploaded yet"},
        500: {"model": ErrorResponse, "description": "LLM generation error"},
    }
)
async def query_documents(request: QueryRequest):
    if not vector_store_manager.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload at least one document first."
        )

    retriever = vector_store_manager.get_retriever()

    # ── Streaming mode ─────────────────────────────────────────────────────────
    if request.stream:
        async def token_generator() -> AsyncGenerator[str, None]:
            """
            Yields SSE-formatted events as tokens arrive from the LLM.

            CONCEPT — SSE format:
              Each event is: "data: <json_string>\n\n"
              The double newline signals end of one event to the browser.
              Frontend reads these with EventSource API or httpx streaming.
            """
            try:
                async for token, model in stream_with_fallback(request.question, retriever):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception:
                logger.exception("Streaming error")
                yield f"data: {json.dumps({'error': 'Generation error. See server logs.'})}\n\n"

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",         # Disable Nginx buffering
                "Access-Control-Allow-Origin": "*",
            }
        )

    # ── Non-streaming mode with source citations ───────────────────────────────
    try:
        # Get chunks with similarity scores for citation display
        docs_with_scores = vector_store_manager.similarity_search(request.question)

        # Generate answer with automatic model fallback
        answer, model_used = await invoke_with_fallback(request.question, retriever)

        sources = [
            SourceChunk(
                content=doc.page_content[:500],
                source_filename=doc.metadata.get("source_filename", "unknown"),
                page=doc.metadata.get("page"),
                relevance_score=round(float(score), 4),
                chunk_index=doc.metadata.get("chunk_index", 0),
            )
            for doc, score in docs_with_scores
        ]

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            model_used=model_used,
        )

    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Generation error. See server logs.")


# ── GET /status ────────────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=StoreStatus,
    summary="System status",
    description="Returns vector store stats and active LLM/embedding configuration.",
)
async def get_status():
    model = (
        settings.groq_model
        if settings.llm_provider.value == "groq"
        else settings.openai_model
    )
    return StoreStatus(
        is_ready=vector_store_manager.is_ready,
        document_count=vector_store_manager.document_count,
        vector_store_type=settings.vector_store_type.value,
        llm_provider=settings.llm_provider.value,
        llm_model=model,
        embedding_model=(
            f"huggingface/{settings.hf_embedding_model}"
            if settings.embedding_provider.value == "huggingface"
            else settings.embedding_model
        ),
    )


# ── DELETE /clear ──────────────────────────────────────────────────────────────

@router.delete(
    "/clear",
    summary="Clear all indexed documents",
    description="Resets the vector store. All documents must be re-uploaded.",
)
async def clear_store():
    vector_store_manager.clear()
    return {"success": True, "message": "Vector store cleared."}
