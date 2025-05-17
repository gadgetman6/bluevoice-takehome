"""Tests for LLM service."""
import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime

from ..src.llm_service.service import LLMService
from ..src.llm_service.models import ChatResponse


@pytest.fixture
def mock_vertex_ai():
    """Mock Vertex AI initialization."""
    with patch('vertexai.init') as mock_init:
        yield mock_init


@pytest.fixture
def mock_generative_model():
    """Mock GenerativeModel."""
    with patch('vertexai.generative_models.GenerativeModel') as mock_model:
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def llm_service(mock_vertex_ai, mock_generative_model):
    """Create LLM service with mocked dependencies."""
    return LLMService(
        project_id="test-project",
        location="us-west1"
    )


@pytest.mark.asyncio
async def test_generate_response_success(llm_service, mock_generative_model):
    """Test successful response generation."""
    # Mock data
    query = "What is the main topic?"
    context_chunks = [
        {
            "content": "The main topic is AI and machine learning.",
            "metadata": {"page": 1}
        }
    ]
    
    # Mock response
    mock_response = Mock()
    mock_response.text = json.dumps({
        "answer": "The main topic is AI and machine learning.",
        "source_pages": [1],
        "confidence": 0.95
    })
    mock_generative_model.generate_content.return_value = mock_response

    # Generate response
    response = await llm_service.generate_response(query, context_chunks)

    # Verify response structure
    assert isinstance(response, dict)
    assert "answer" in response
    assert "source_pages" in response
    assert "confidence" in response

    # Validate with Pydantic model
    chat_response = ChatResponse(**response)
    assert chat_response.answer == "The main topic is AI and machine learning."
    assert chat_response.source_pages == [1]
    assert chat_response.confidence == 0.95
    assert isinstance(chat_response.timestamp, datetime)

    # Verify prompt construction
    prompt_call = mock_generative_model.generate_content.call_args[0][0]
    assert query in prompt_call
    assert context_chunks[0]["content"] in prompt_call
    assert "Page 1" in prompt_call


@pytest.mark.asyncio
async def test_generate_response_invalid_json(llm_service, mock_generative_model):
    """Test handling of invalid JSON response."""
    # Mock invalid JSON response
    mock_response = Mock()
    mock_response.text = "Invalid JSON"
    mock_generative_model.generate_content.return_value = mock_response

    # Generate response
    response = await llm_service.generate_response("query", [])

    # Verify fallback response
    assert response == {
        "answer": "Invalid JSON",
        "source_pages": [],
        "confidence": 0.0
    }

    # Validate with Pydantic model
    chat_response = ChatResponse(**response)
    assert chat_response.answer == "Invalid JSON"
    assert chat_response.source_pages == []
    assert chat_response.confidence == 0.0


@pytest.mark.asyncio
async def test_generate_response_error(llm_service, mock_generative_model):
    """Test error handling."""
    # Mock API error
    mock_generative_model.generate_content.side_effect = Exception("API Error")

    # Verify error is raised
    with pytest.raises(Exception) as exc_info:
        await llm_service.generate_response("query", [])
    assert str(exc_info.value) == "API Error"


@pytest.mark.asyncio
async def test_prompt_construction(llm_service, mock_generative_model):
    """Test prompt construction with multiple context chunks."""
    # Mock data
    query = "What are the key points?"
    context_chunks = [
        {
            "content": "First key point.",
            "metadata": {"page": 1}
        },
        {
            "content": "Second key point.",
            "metadata": {"page": 2}
        }
    ]
    
    # Mock response
    mock_response = Mock()
    mock_response.text = json.dumps({
        "answer": "The key points are...",
        "source_pages": [1, 2],
        "confidence": 0.9
    })
    mock_generative_model.generate_content.return_value = mock_response

    # Generate response
    await llm_service.generate_response(query, context_chunks)

    # Verify prompt construction
    prompt = mock_generative_model.generate_content.call_args[0][0]
    assert query in prompt
    assert "First key point." in prompt
    assert "Second key point." in prompt
    assert "Page 1" in prompt
    assert "Page 2" in prompt
    assert "JSON" in prompt  # Verify JSON formatting instruction
