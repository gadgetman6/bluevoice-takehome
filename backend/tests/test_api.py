"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
import tempfile
import os
from pathlib import Path
import fitz  # PyMuPDF

from ..src.backend_api.api.app import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        # Create PDF with test content
        doc = fitz.open()
        page = doc.new_page()
        
        # Add content that we can query later
        page.insert_text(
            (50, 50),
            "Artificial Intelligence (AI) is transforming industries.\n" +
            "Machine learning models can process natural language.\n" +
            "Deep learning is a subset of machine learning.",
            fontsize=11
        )
        
        # Save and close
        doc.save(f.name)
        doc.close()
        
        yield f.name
        
        # Cleanup
        Path(f.name).unlink()


def test_upload_document(client, sample_pdf):
    """Test document upload endpoint."""
    # Upload file
    with open(sample_pdf, 'rb') as f:
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "metadata" in data
    assert data["metadata"]["filename"] == "test.pdf"
    assert data["metadata"]["total_pages"] == 1
    assert data["metadata"]["total_chunks"] > 0
    
    return data["document_id"]


def test_chat_endpoint(client, sample_pdf):
    """Test chat endpoint with document query."""
    # First upload a document
    document_id = test_upload_document(client, sample_pdf)
    
    # Test queries
    queries = [
        {
            "query": "What is transforming industries?",
            "expected_content": "artificial intelligence",
            "expected_page": 1
        },
        {
            "query": "What can machine learning models do?",
            "expected_content": "natural language",
            "expected_page": 1
        },
        {
            "query": "What is a subset of machine learning?",
            "expected_content": "deep learning",
            "expected_page": 1
        }
    ]
    
    for test_case in queries:
        response = client.post(
            "/chat",
            json={
                "query": test_case["query"],
                "document_id": document_id
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "answer" in data
        assert "source_pages" in data
        assert "confidence" in data
        assert "timestamp" in data
        
        # Check content
        assert test_case["expected_content"].lower() in data["answer"].lower()
        assert test_case["expected_page"] in data["source_pages"]
        assert 0 <= data["confidence"] <= 1


def test_error_handling(client):
    """Test API error handling."""
    # Test invalid document ID
    response = client.post(
        "/chat",
        json={
            "query": "test query",
            "document_id": "nonexistent-id"
        }
    )
    assert response.status_code == 404
    
    # Test missing file
    response = client.post("/documents/upload")
    assert response.status_code == 422
    
    # Test non-PDF file
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        f.write(b"This is not a PDF")
        f.seek(0)
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", f, "text/plain")}
        )
    assert response.status_code == 500
    
    # Test empty query
    response = client.post(
        "/chat",
        json={
            "query": "",
            "document_id": "test-id"
        }
    )
    assert response.status_code == 422


def test_concurrent_queries(client, sample_pdf):
    """Test handling multiple concurrent queries."""
    # Upload document
    document_id = test_upload_document(client, sample_pdf)
    
    # Make concurrent requests
    import asyncio
    import httpx
    
    async def make_request(query: str):
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/chat",
                json={
                    "query": query,
                    "document_id": document_id
                }
            )
            return response.json()
    
    # Run concurrent queries
    queries = [
        "What is AI?",
        "Explain machine learning",
        "What is deep learning?"
    ]
    
    async def run_concurrent_queries():
        tasks = [make_request(q) for q in queries]
        return await asyncio.gather(*tasks)
    
    responses = asyncio.run(run_concurrent_queries())
    
    # Verify all responses
    assert len(responses) == len(queries)
    for response in responses:
        assert "answer" in response
        assert "source_pages" in response
        assert "confidence" in response
