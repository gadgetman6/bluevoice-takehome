"""FastAPI application for document Q&A."""
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os
from dotenv import load_dotenv
import logging

from ..document_processor.processor import DocumentProcessor
from ..vector_store.store import VectorStore
from ..llm_service.service import LLMService
from ..llm_service.models import ChatResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Q&A API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
processor = DocumentProcessor()
vector_store = VectorStore()
llm_service = LLMService(
    project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location="us-west1"  # Using us-west1 for lower latency
)

@app.post("/documents/upload")
async def upload_document(file: UploadFile) -> Dict[str, Any]:
    """
    Upload and process a document.

    Args:
        file: PDF document to process

    Returns:
        Document ID and metadata
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process document
        result = await processor.process_document(temp_path)
        
        # Store chunks in vector store
        await vector_store.add_chunks(
            result["document_id"],
            result["chunks"]
        )

        # Cleanup temp file
        os.remove(temp_path)

        return {
            "document_id": result["document_id"],
            "metadata": result["metadata"]
        }

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(
    query: str,
    document_id: str
) -> ChatResponse:
    """
    Chat with a document.

    Args:
        query: User's question
        document_id: ID of the document to query

    Returns:
        Structured response with answer and metadata
    """
    try:
        # Get relevant chunks from vector store
        chunks = await vector_store.search(query, document_id)

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant content found"
            )

        # Generate response
        response = await llm_service.generate_response(query, chunks)
        
        # Validate response with Pydantic model
        return ChatResponse(**response)

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
