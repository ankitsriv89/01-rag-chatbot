"""
Vector Store Manager
====================
Handles embedding documents and storing/querying them in a vector database.

CONCEPT — What is an embedding?
  text-embedding-3-small converts any text into a 1536-dimensional vector.
  Example: "The cat sat on the mat" → [0.02, -0.15, 0.33, ... 1536 numbers]
  Semantically similar sentences have vectors that are geometrically close.

CONCEPT — FAISS vs Chroma:
  FAISS (Facebook AI Similarity Search):
    - In-memory, blazing fast, no persistence between restarts
    - Best for: development, single-session use cases

  Chroma:
    - Persists to disk, survives restarts
    - Best for: production where documents shouldn't be re-uploaded every time

CONCEPT — Cosine Similarity:
  The most common metric for comparing embeddings.
  Measures the angle between two vectors (not distance).
  Score of 1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite.
"""

from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from loguru import logger

from app.config import settings, VectorStoreType, EmbeddingProvider


def _build_embeddings():
    """
    Return the configured embeddings model.

    HuggingFace (default): free, local, no API key — downloads ~90MB on first use.
    OpenAI: paid, cloud-based — requires OPENAI_API_KEY in .env.
    """
    if settings.embedding_provider == EmbeddingProvider.HUGGINGFACE:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Using HuggingFace embeddings: {settings.hf_embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Using OpenAI embeddings: {settings.embedding_model}")
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )


class VectorStoreManager:
    """
    Manages the lifecycle of the vector store:
    - Creating embeddings from document chunks
    - Storing vectors
    - Retrieving relevant chunks for a query

    Follows a simple class pattern so the vector store persists
    across multiple API requests (in-memory singleton).
    """

    def __init__(self):
        self._embeddings = _build_embeddings()
        self._store = None   # Populated after first document upload
        self._doc_count = 0  # Track how many chunks are stored

    @property
    def is_ready(self) -> bool:
        """Returns True if at least one document has been indexed."""
        return self._store is not None

    def add_documents(self, chunks: List[Document]) -> int:
        """
        Embed a list of document chunks and add them to the vector store.

        FLOW:
          1. Each chunk's text is sent to the configured embeddings model
          2. Returns dense vectors (384-dim for HuggingFace, 1536-dim for OpenAI)
          3. Vectors + original text stored in FAISS/Chroma

        Args:
            chunks: Document chunks from document_processor.py

        Returns:
            Total number of chunks now in the store.
        """
        if not chunks:
            logger.warning("add_documents called with empty chunk list")
            return self._doc_count

        model_name = settings.hf_embedding_model if settings.embedding_provider == EmbeddingProvider.HUGGINGFACE else settings.embedding_model
        logger.info(f"Embedding {len(chunks)} chunks using '{model_name}'...")

        if self._store is None:
            # First upload — create the vector store from scratch
            self._store = self._create_store(chunks)
        else:
            # Subsequent uploads — add to the existing store
            self._store.add_documents(chunks)

        self._doc_count += len(chunks)
        logger.info(f"Vector store now contains {self._doc_count} chunks")
        return self._doc_count

    def _create_store(self, chunks: List[Document]):
        """
        Create a new vector store from initial document chunks.
        Called only once (on first document upload).
        """
        if settings.vector_store_type == VectorStoreType.FAISS:
            # from_documents() calls the embeddings API and builds the index in one step
            return FAISS.from_documents(chunks, self._embeddings)

        elif settings.vector_store_type == VectorStoreType.CHROMA:
            # Chroma persists to disk at chroma_persist_dir
            return Chroma.from_documents(
                chunks,
                self._embeddings,
                persist_directory=settings.chroma_persist_dir,
            )

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Find the top-k most semantically similar chunks to the query.

        CONCEPT — Similarity search steps:
          1. Query text is embedded using the same model as documents
          2. Cosine similarity computed between query vector and all stored vectors
          3. Top-k highest-scoring chunks returned

        Args:
            query: The user's question.
            k: Number of chunks to return. Defaults to settings.retrieval_top_k.

        Returns:
            List of (Document, similarity_score) tuples, sorted by score descending.
            Score is between 0 and 1 (higher = more relevant).

        Raises:
            RuntimeError: if no documents have been indexed yet.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Vector store is empty. Please upload at least one document first."
            )

        top_k = k or settings.retrieval_top_k

        # similarity_search_with_score returns (Document, score) tuples
        results = self._store.similarity_search_with_score(query, k=top_k)

        logger.debug(
            f"Retrieved {len(results)} chunks for query: '{query[:60]}...' "
            f"(scores: {[round(s, 3) for _, s in results]})"
        )
        return results

    def get_retriever(self):
        """
        Return a LangChain-compatible retriever object.

        LangChain chains expect a retriever interface (not raw similarity_search).
        This bridges our vector store to LangChain's LCEL (LangChain Expression Language).
        """
        if not self.is_ready:
            raise RuntimeError("Vector store is empty. Upload documents first.")

        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_top_k},
        )

    def clear(self):
        """Reset the vector store (useful for testing or a 'clear all docs' endpoint)."""
        self._store = None
        self._doc_count = 0
        logger.info("Vector store cleared")

    @property
    def document_count(self) -> int:
        return self._doc_count


# Module-level singleton — shared across all API requests in one server process
# In production with multiple workers, use a persistent store (Chroma/Pinecone)
vector_store_manager = VectorStoreManager()
