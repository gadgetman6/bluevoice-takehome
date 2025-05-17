"""
Global fixtures that stub out anything that might hit the network.
"""
import pytest
from fastapi.testclient import TestClient
from backend.src.backend_api.api.app import app

#
# ▸ No-op stubs for all expensive services -----------------------------
#
@pytest.fixture(autouse=True)
def _stub_everything(monkeypatch):
    # vector store
    monkeypatch.setattr(
        "backend.src.backend_api.vector_store.store.VectorStore",
        lambda *_, **__: None,
        raising=False,
    )
    # embeddings service
    monkeypatch.setattr(
        "backend.src.backend_api.vector_store.embeddings.EmbeddingsService",
        lambda *_, **__: None,
        raising=False,
    )
    # processor
    monkeypatch.setattr(
        "backend.src.backend_api.document_processor.processor.DocumentProcessor",
        lambda *_, **__: None,
        raising=False,
    )
    # LLM service
    monkeypatch.setattr(
        "backend.src.backend_api.llm_service.service.LLMService",
        lambda *_, **__: None,
        raising=False,
    )
    # (add more stubs here if new deps appear)

@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient (shared by tests)."""
    return TestClient(app)