"""Pydantic models for LLM service."""
from typing import List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class DocumentMetadata(BaseModel):
    """Metadata about a processed document."""
    filename: str = Field(..., description="Original filename")
    total_pages: int = Field(..., description="Total number of pages")
    total_chunks: int = Field(..., description="Total number of text chunks")


class DocumentResponse(BaseModel):
    """Response from document processing."""
    document_id: str = Field(..., description="Unique identifier for the document")
    metadata: DocumentMetadata = Field(..., description="Document metadata")


class ChatResponse(BaseModel):
    """Response from the LLM."""
    answer: str = Field(
        ...,
        description="The answer to the query",
        min_length=1
    )
    source_pages: List[int] = Field(
        ...,
        description="Page numbers where information was found",
        min_items=1
    )
    confidence: float = Field(
        ...,
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the response was generated"
    )

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1 and rounded."""
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return round(v, 3)

    @field_validator('source_pages')
    @classmethod
    def validate_pages(cls, v: List[int]) -> List[int]:
        """Ensure page numbers are positive."""
        if not all(page > 0 for page in v):
            raise ValueError('Page numbers must be positive')
        return sorted(list(set(v)))  # Remove duplicates and sort
