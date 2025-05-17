"""Integration tests for vector store."""
import pytest
import tempfile
import shutil
from pathlib import Path
import asyncio

from ..src.vector_store.store import VectorStore


@pytest.fixture(scope="function")
async def vector_store():
    """Create a temporary vector store for testing."""
    # Create temporary directory for ChromaDB
    temp_dir = tempfile.mkdtemp()
    
    # Initialize store
    store = VectorStore(persist_dir=temp_dir)
    
    yield store
    
    # Cleanup
    await asyncio.sleep(0.1)  # Ensure ChromaDB has finished writing
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_add_and_search_chunks(vector_store):
    """Test adding chunks and searching them."""
    # Test data
    document_id = "test-doc"
    chunks = [
        {
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"page": 1, "source": "test.pdf"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"page": 2, "source": "test.pdf"}
        },
        {
            "content": "Python is a popular programming language.",
            "metadata": {"page": 3, "source": "test.pdf"}
        }
    ]

    # Add chunks
    await vector_store.add_chunks(document_id, chunks)

    # Test exact match search
    results = await vector_store.search(
        query="What animal jumps over the dog?",
        document_id=document_id,
        limit=1
    )
    assert len(results) == 1
    assert "fox" in results[0]["content"].lower()
    assert results[0]["metadata"]["page"] == 1

    # Test semantic search
    results = await vector_store.search(
        query="Tell me about AI",
        document_id=document_id,
        limit=1
    )
    assert len(results) == 1
    assert "machine learning" in results[0]["content"].lower()
    assert results[0]["metadata"]["page"] == 2

    # Test multiple results
    results = await vector_store.search(
        query="What technologies are mentioned?",
        document_id=document_id,
        limit=2
    )
    assert len(results) == 2
    contents = [r["content"].lower() for r in results]
    assert any("python" in c for c in contents)
    assert any("machine learning" in c for c in contents)


@pytest.mark.asyncio
async def test_search_with_document_filter(vector_store):
    """Test searching with document ID filtering."""
    # Add chunks for two different documents
    chunks_doc1 = [{
        "content": "Document 1 contains unique information.",
        "metadata": {"page": 1, "source": "doc1.pdf"}
    }]
    chunks_doc2 = [{
        "content": "Document 2 has different content.",
        "metadata": {"page": 1, "source": "doc2.pdf"}
    }]

    await vector_store.add_chunks("doc1", chunks_doc1)
    await vector_store.add_chunks("doc2", chunks_doc2)

    # Search in specific document
    results = await vector_store.search(
        query="What information is available?",
        document_id="doc1"
    )
    assert len(results) > 0
    assert all("doc1.pdf" in r["metadata"]["source"] for r in results)


@pytest.mark.asyncio
async def test_search_relevance_ordering(vector_store):
    """Test that search results are ordered by relevance."""
    # Add chunks with varying relevance
    chunks = [
        {
            "content": "Cats are common household pets.",
            "metadata": {"page": 1, "source": "test.pdf"}
        },
        {
            "content": "Dogs are known as man's best friend.",
            "metadata": {"page": 2, "source": "test.pdf"}
        },
        {
            "content": "Cats and dogs are both domesticated animals.",
            "metadata": {"page": 3, "source": "test.pdf"}
        }
    ]
    await vector_store.add_chunks("test-doc", chunks)

    # Search for cat-related content
    results = await vector_store.search(
        query="Tell me about cats",
        document_id="test-doc",
        limit=3
    )

    # Verify ordering by checking similarity scores
    assert len(results) == 3
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)
    assert "cats" in results[0]["content"].lower()


@pytest.mark.asyncio
async def test_empty_results(vector_store):
    """Test searching with no relevant results."""
    # Add chunks
    chunks = [{
        "content": "This is a test document.",
        "metadata": {"page": 1, "source": "test.pdf"}
    }]
    await vector_store.add_chunks("test-doc", chunks)

    # Search for unrelated content
    results = await vector_store.search(
        query="quantum physics theoretical frameworks",
        document_id="test-doc"
    )
    
    # Verify low similarity scores
    assert len(results) > 0
    assert all(r["similarity"] < 0.5 for r in results)
