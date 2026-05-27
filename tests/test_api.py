"""
API Integration Tests
======================
Tests FastAPI endpoints using TestClient (no real server needed).

CONCEPT — TestClient:
  FastAPI's TestClient wraps the app in a test context.
  It sends real HTTP requests to your route handlers but in-process —
  no network, no server startup needed. Tests run in milliseconds.

CONCEPT — Mocking external services:
  We mock vector_store_manager and the LLM calls so tests don't need
  API keys or GPU. This is called "dependency injection" testing.

Run with:  pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from io import BytesIO

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test the root and health check endpoints."""

    def test_root_returns_healthy(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


class TestStatusEndpoint:
    """Test the /api/v1/status endpoint."""

    def test_status_returns_store_info(self):
        response = client.get("/api/v1/status")
        assert response.status_code == 200

        data = response.json()
        assert "is_ready" in data
        assert "document_count" in data
        assert "llm_provider" in data
        assert "llm_model" in data
        assert "embedding_model" in data


class TestUploadEndpoint:
    """Test the /api/v1/upload endpoint."""

    @patch("app.api.routes.process_upload")
    @patch("app.api.routes.vector_store_manager")
    def test_upload_pdf_success(self, mock_store, mock_process):
        """A valid PDF upload should return 201 with chunk stats."""
        from langchain_core.documents import Document

        # Mock the processing pipeline
        mock_chunks = [Document(page_content=f"chunk {i}", metadata={}) for i in range(10)]
        mock_process.return_value = mock_chunks
        mock_store.add_documents.return_value = 10

        fake_pdf = BytesIO(b"%PDF-1.4 fake content")
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["chunks_created"] == 10
        assert data["filename"] == "test.pdf"

    def test_upload_no_file_returns_422(self):
        """Missing file field should return 422 Unprocessable Entity."""
        response = client.post("/api/v1/upload")
        assert response.status_code == 422

    @patch("app.api.routes.process_upload")
    def test_upload_unsupported_type_returns_400(self, mock_process):
        """Unsupported file extension should return 400."""
        mock_process.side_effect = ValueError("Unsupported file type: '.exe'")

        response = client.post(
            "/api/v1/upload",
            files={"file": ("malware.exe", BytesIO(b"bad"), "application/octet-stream")},
        )
        assert response.status_code == 400


class TestQueryEndpoint:
    """Test the /api/v1/query endpoint."""

    @patch("app.api.routes.vector_store_manager")
    def test_query_without_documents_returns_400(self, mock_store):
        """Query before uploading any docs should return 400."""
        mock_store.is_ready = False

        response = client.post(
            "/api/v1/query",
            json={"question": "What is the main topic?", "stream": False},
        )
        assert response.status_code == 400
        assert "No documents" in response.json()["detail"]

    def test_query_too_short_question_returns_422(self):
        """Question shorter than min_length=3 should return 422."""
        response = client.post(
            "/api/v1/query",
            json={"question": "Hi", "stream": False},
        )
        assert response.status_code == 422

    @patch("app.api.routes.vector_store_manager")
    @patch("app.api.routes.invoke_with_fallback", new_callable=AsyncMock)
    def test_query_returns_answer_with_sources(self, mock_invoke, mock_store):
        """Successful non-streaming query should return answer + sources."""
        from langchain_core.documents import Document

        mock_store.is_ready = True
        mock_store.get_retriever.return_value = MagicMock()
        mock_store.similarity_search.return_value = [
            (Document(
                page_content="Revenue was $4.2B",
                metadata={"source_filename": "report.pdf", "chunk_index": 0}
            ), 0.92)
        ]
        mock_invoke.return_value = ("Revenue was $4.2B in FY2024.", "llama-3.3-70b-versatile")

        response = client.post(
            "/api/v1/query",
            json={"question": "What was the revenue?", "stream": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["model_used"] == "llama-3.3-70b-versatile"


class TestClearEndpoint:
    """Test the /api/v1/clear endpoint."""

    @patch("app.api.routes.vector_store_manager")
    def test_clear_returns_success(self, mock_store):
        response = client.delete("/api/v1/clear")
        assert response.status_code == 200
        assert response.json()["success"] is True
        mock_store.clear.assert_called_once()
