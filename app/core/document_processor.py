"""
Document Processor
==================
Handles loading and chunking of uploaded documents.

CONCEPT — Why chunking?
  LLMs have a context window limit (e.g. 8k tokens for LLaMA 3).
  A 50-page PDF has ~25,000 tokens — far too large to send whole.
  Solution: split into small overlapping chunks, store all of them,
  retrieve only the relevant ones at query time.

CONCEPT — Chunk overlap:
  If chunk 1 ends at char 1000 and chunk 2 starts at char 800,
  they share 200 chars of overlap. This prevents a sentence from
  being split mid-thought across two chunks that never appear together.
"""

import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from loguru import logger

from app.config import settings


# Supported file types and their corresponding LangChain loaders
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from disk using the appropriate loader for its file type.

    Args:
        file_path: Absolute path to the file on disk.

    Returns:
        List of LangChain Document objects. Each Document has:
          - page_content: the raw text
          - metadata: dict with source, page number, etc.

    Raises:
        ValueError: if the file extension is not supported.
    """
    extension = Path(file_path).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    loader_class = SUPPORTED_EXTENSIONS[extension]
    loader = loader_class(file_path)

    # .load() reads the entire file and returns a list of Documents
    # PDFs return one Document per page; TXT returns one Document total
    documents = loader.load()

    logger.info(f"Loaded {len(documents)} page(s) from '{Path(file_path).name}'")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller overlapping chunks for embedding.

    CONCEPT — RecursiveCharacterTextSplitter:
      Tries to split on paragraph breaks (\n\n) first, then sentences (\n),
      then words (" "), then characters. This keeps semantic units together
      as long as possible, only breaking at finer boundaries when needed.

    Args:
        documents: Raw documents from load_document().

    Returns:
        List of smaller Document chunks, each with inherited metadata
        plus a 'chunk_index' field added for traceability.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,          # Max chars per chunk (from config)
        chunk_overlap=settings.chunk_overlap,    # Overlap chars between chunks
        length_function=len,                     # Use character count (not token count)
        separators=["\n\n", "\n", ". ", " ", ""],  # Try these split points in order
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for debugging and citation purposes
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    logger.info(
        f"Split {len(documents)} document(s) into {len(chunks)} chunks "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks


def save_upload_to_temp(file_bytes: bytes, filename: str) -> str:
    """
    Save uploaded file bytes to a temporary file on disk.

    FastAPI receives uploads as bytes in memory. LangChain loaders need
    a file path on disk, so we write it to a temp file first.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to preserve the extension).

    Returns:
        Path to the temporary file. Caller is responsible for deleting it.
    """
    extension = Path(filename).suffix
    # delete=False: we need the file to persist after this function returns
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    logger.debug(f"Saved upload '{filename}' to temp path: {tmp_path}")
    return tmp_path


def process_upload(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Full pipeline: bytes → load → chunk → return chunks.

    This is the single entry point called by the API route.
    Handles temp file cleanup automatically.

    Args:
        file_bytes: Raw bytes from the uploaded file.
        filename: Original filename with extension.

    Returns:
        List of text chunks ready for embedding.
    """
    tmp_path = save_upload_to_temp(file_bytes, filename)

    try:
        documents = load_document(tmp_path)
        chunks = chunk_documents(documents)

        # Tag every chunk with the original filename for citation in answers
        for chunk in chunks:
            chunk.metadata["source_filename"] = filename

        return chunks

    finally:
        # Always delete temp file, even if an exception occurs
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"Cleaned up temp file: {tmp_path}")
