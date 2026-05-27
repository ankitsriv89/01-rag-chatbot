"""
Unit Tests — Document Processor
=================================
Tests chunking logic without needing API keys or a running server.

CONCEPT — Unit tests vs integration tests:
  Unit tests: test one function in isolation, mock all external dependencies.
  Integration tests: test multiple components together with real dependencies.

  These are unit tests — they run fast (< 1 second) and need no credentials.
  We mock the file loaders so we don't need real PDF files either.

Run with:  pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.core.document_processor import chunk_documents, save_upload_to_temp, process_upload


class TestChunkDocuments:
    """Tests for the chunking function — most important RAG component."""

    def test_chunks_single_document(self):
        """A document longer than chunk_size should produce multiple chunks."""
        # Create a fake document with 3000 chars (3x the default chunk_size of 1000)
        long_text = "A" * 3000
        docs = [Document(page_content=long_text, metadata={"source": "test.pdf"})]

        chunks = chunk_documents(docs)

        # Should produce at least 2 chunks
        assert len(chunks) > 1

    def test_chunks_preserve_metadata(self):
        """Metadata from original documents should be preserved in chunks."""
        docs = [Document(
            page_content="Some test content " * 100,
            metadata={"source": "test.pdf", "page": 1}
        )]

        chunks = chunk_documents(docs)

        # All chunks should keep the original metadata
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 1

    def test_chunks_add_index_metadata(self):
        """chunk_documents should add chunk_index and chunk_total to each chunk."""
        docs = [Document(page_content="word " * 500, metadata={})]

        chunks = chunk_documents(docs)

        assert "chunk_index" in chunks[0].metadata
        assert "chunk_total" in chunks[0].metadata
        assert chunks[0].metadata["chunk_total"] == len(chunks)

    def test_empty_document_list(self):
        """Empty input should return empty output without errors."""
        chunks = chunk_documents([])
        assert chunks == []

    def test_short_document_stays_single_chunk(self):
        """A document shorter than chunk_size should remain as one chunk."""
        short_text = "This is a short document."
        docs = [Document(page_content=short_text, metadata={})]

        chunks = chunk_documents(docs)

        assert len(chunks) == 1
        assert short_text in chunks[0].page_content


class TestSaveUploadToTemp:
    """Tests for the temp file save utility."""

    def test_creates_file_with_correct_extension(self, tmp_path):
        """Saved temp file should have the same extension as the original filename."""
        content = b"fake pdf content"

        tmp_file = save_upload_to_temp(content, "report.pdf")

        assert tmp_file.endswith(".pdf")

    def test_file_content_is_preserved(self):
        """Bytes written to temp file should match what was passed in."""
        import os
        content = b"hello world test content"

        tmp_file = save_upload_to_temp(content, "test.txt")

        try:
            with open(tmp_file, "rb") as f:
                assert f.read() == content
        finally:
            os.remove(tmp_file)


class TestProcessUpload:
    """Integration-style tests for the full upload pipeline (loader mocked)."""

    @patch("app.core.document_processor.load_document")
    def test_adds_source_filename_to_metadata(self, mock_load):
        """Each chunk should have source_filename in metadata for citations."""
        mock_load.return_value = [
            Document(page_content="chunk content " * 100, metadata={})
        ]

        chunks = process_upload(b"fake bytes", "my_report.pdf")

        for chunk in chunks:
            assert chunk.metadata["source_filename"] == "my_report.pdf"

    @patch("app.core.document_processor.load_document")
    def test_cleanup_temp_file(self, mock_load, tmp_path):
        """Temp file should be deleted after processing, even on success."""
        import os
        from app.core.document_processor import save_upload_to_temp

        mock_load.return_value = [Document(page_content="x " * 200, metadata={})]

        # Track which temp path was created
        created_paths = []
        original_save = save_upload_to_temp

        def tracking_save(b, f):
            path = original_save(b, f)
            created_paths.append(path)
            return path

        with patch("app.core.document_processor.save_upload_to_temp", side_effect=tracking_save):
            process_upload(b"data", "test.txt")

        # Temp file should have been deleted
        for path in created_paths:
            assert not os.path.exists(path), f"Temp file not cleaned up: {path}"
