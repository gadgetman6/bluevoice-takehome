"""Tests for document processor."""
import pytest
from pathlib import Path
import tempfile
import pymupdf  # PyMuPDF

from backend.src.backend_api.document_processor.processor import DocumentProcessor

@pytest.fixture
def processor():
    """Create a document processor instance."""
    return DocumentProcessor()

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        # Create a new PDF with PyMuPDF
        doc = pymupdf.open()
        page = doc.new_page()
        
        # Add some test content
        page.insert_text(
            (50, 50),
            "This is a test document.\nIt contains multiple lines of text.\n" * 100,
            fontsize=11
        )

        page2 = doc.new_page()
        page2.insert_text(
            (50, 200),
            "This is the second page.\nIt also contains multiple lines of text.\n" * 100,
            fontsize=11
        )
        
        # Save the PDF
        doc.save(f.name)
        doc.close()
        
        yield f.name
        
        # Cleanup
        Path(f.name).unlink()

async def test_process_document(processor, sample_pdf):
    """Test document processing."""
    # Process the document
    result = await processor.process_document(sample_pdf)
    
    # Check basic structure
    assert "document_id" in result
    assert "chunks" in result
    assert "metadata" in result
    
    # Check metadata
    assert result["metadata"]["filename"] == Path(sample_pdf).name
    assert result["metadata"]["total_pages"] == 2
    assert result["metadata"]["total_chunks"] > 0
    
    # Check chunks
    assert len(result["chunks"]) > 0
    for chunk in result["chunks"]:
        assert "content" in chunk
        assert "metadata" in chunk
        assert "page" in chunk["metadata"]

async def test_process_nonexistent_file(processor):
    """Test handling of nonexistent files."""
    with pytest.raises(FileNotFoundError):
        await processor.process_document("nonexistent.pdf")

async def test_process_non_pdf(processor):
    """Test handling of non-PDF files."""
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        f.write(b"This is not a PDF")
        f.flush()
        
        with pytest.raises(ValueError) as exc_info:
            await processor.process_document(f.name)
        
        assert "Only PDF files are supported" in str(exc_info.value)

async def test_chunk_size_and_overlap(processor, sample_pdf):
    """Test document chunking with different sizes."""
    # Process with default settings
    default_result = await processor.process_document(sample_pdf)
    default_chunks = len(default_result["chunks"])
    
    # Process with smaller chunk size
    small_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    small_result = await small_processor.process_document(sample_pdf)
    small_chunks = len(small_result["chunks"])
    
    # Smaller chunks should result in more chunks
    assert small_chunks > default_chunks

async def test_metadata_extraction(processor, sample_pdf):
    """Test metadata extraction."""
    result = await processor.process_document(sample_pdf)
    
    assert isinstance(result["metadata"], dict)
    assert all(key in result["metadata"] for key in ["filename", "total_pages", "total_chunks"])
    assert isinstance(result["metadata"]["total_pages"], int)
    assert isinstance(result["metadata"]["total_chunks"], int)
    assert result["metadata"]["total_chunks"] > 0
