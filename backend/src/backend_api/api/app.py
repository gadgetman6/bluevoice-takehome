"""FastAPI application for document Q&A."""

import asyncio
import uuid
from backend_api.api.events.bus import EventBus
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    Request,
    UploadFile,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import logging

from sse_starlette import EventSourceResponse

from ..document_processor.processor import DocumentProcessor
from ..vector_store.store import VectorStore
from ..llm_service.service import LLMService
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat request model."""

    query: str
    document_id: str


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
llm_service = LLMService()


async def get_client_id(x_client_id: str = Header(None)):
    if not x_client_id:
        raise HTTPException(400, "X-Client-Id header required")
    return x_client_id


async def _vectorize_and_store(
    doc_id: str,
    chunks: list[dict],
    client_id: str,
):
    """Embed chunks and add them to the vector DB; push SSE when done."""
    try:
        await vector_store.add_chunks(doc_id, chunks)  # <-- long step
        await EventBus.push(
            client_id,
            "indexed",
            {
                "document_id": doc_id,
                "status": "complete",
            },
        )
    except Exception as exc:
        logger.exception("vectorization failed")
        await EventBus.push(client_id, "error", f"vectorize: {exc}")


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,  # ← inject FastAPI helper
    client_id: str = Depends(get_client_id),
):
    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        # 1) save upload to disk
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # 2) parse / chunk (you await this so you can push 'ready')
        result = await processor.process_document(temp_path)

        await EventBus.push(
            client_id,
            "ready",
            {
                "document_id": result["document_id"],
                "metadata": result["metadata"],
            },
        )

        # run vectorization in background so we don't block the request
        background_tasks.add_task(
            _vectorize_and_store,
            result["document_id"],
            result["chunks"],
            client_id,
        )

        return {"status": "processing"}  # POST returns immediately

    except Exception as exc:
        logger.exception("upload failed")
        await EventBus.push(client_id, "error", str(exc))
        raise HTTPException(500, "Upload failed")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat")
async def chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(get_client_id),
):
    chunks = await vector_store.search(body.query, body.document_id)

    async def push_tokens():
        try:
            async for token in llm_service.stream_answer(body.query, chunks):
                await EventBus.push(client_id, "chat", token)
        except Exception as exc:
            await EventBus.push(client_id, "error", str(exc))

    # run in background so /chat returns 202 immediately
    background_tasks.add_task(push_tokens)
    return {"status": "generating"}


@app.get("/events/stream")
async def stream_events(client_id: str, request: Request):
    """
    Long-lived SSE connection. Client supplies clientId query parameter.
    """
    queue = EventBus.queue(client_id)

    async def event_gen():
        while True:
            if await request.is_disconnected():
                break
            payload = await queue.get()
            yield payload

    return EventSourceResponse(event_gen(), ping=True)
