"""LLM service using Google's Gemini model."""

import re
from typing import Dict, Any, List, AsyncGenerator
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.generative_models._generative_models import (
    HarmCategory,
    HarmBlockThreshold,
)
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMService:
    """Service for generating responses using Gemini."""

    def __init__(
        self,
        location: str = "us-central1",  # Using us-west1 for lower latency
        model_name: str = "gemini-2.5-flash-preview-04-17",  # Using Gemini 2.5 Flash
    ):
        """
        Initialize the LLM service.

        Args:
            api_key: Google Cloud API key
            location: Model location (us-west1 for lower latency)
            model_name: Name of the model to use
        """
        self.location = location
        self.model_name = model_name

        # Initialize Vertex AI with API key
        vertexai.init(location=location)

        # Configure safety settings to handle content appropriately
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }

        # Initialize model with safety settings
        self.model = GenerativeModel(model_name, safety_settings=self.safety_settings)
        logger.info(f"Initialized LLM service with model {model_name}")

    def _build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a prompt for the LLM.

        Args:
            query: User's question
            context_chunks: Relevant document chunks

        Returns:
            Formatted prompt string
        """
        # Combine context chunks into a single context string
        context = "\n\n".join(
            [
                f"[Page {chunk['metadata']['page']}] {chunk['content']}"
                for chunk in context_chunks
            ]
        )

        # Build the prompt
        prompt = f"""Based on the following context, answer the question and format the response as JSON with the following structure:
        {{
            "answer": "Your detailed answer here",
            "source_pages": [page numbers where information was found],
            "confidence": confidence score between 0 and 1
        }}

        Context:
        {context}

        Question: {query}"""

        return prompt

    async def stream_answer(
        self, query: str, context_chunks: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Yield answer text chunks; stop early if confidence is low."""
        prompt = self._build_prompt(query, context_chunks)

        responses = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "confidence": {"type": "number"},
                        "answer": {"type": "string"},
                        "source_pages": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["confidence", "answer", "source_pages"],
                },
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=1024,
            ),
            stream=True,
        )

        buf, inside_answer, checked_conf = "", False, False

        full_json = ""

        for chunk in responses:
            # ----- pull the delta text out of the chunk -----
            delta = "".join(
                part.text
                for cand in chunk.candidates
                for part in cand.content.parts
                if hasattr(part, "text")
            )
            buf += delta
            full_json += delta

            # ── 1. early confidence check ──────────────────
            if not checked_conf and '"confidence"' in buf:
                try:
                    conf = json.loads(buf + "}")["confidence"]
                    checked_conf = True
                    if conf < 0.30:               # your threshold
                        responses.close()         # stop the server-side stream
                        yield "[confidence too low]\n"
                        return
                except json.JSONDecodeError:
                    pass                          # not enough yet

            # ── 2. stream the “answer” value to caller ─────
            if not inside_answer:
                m = re.search(r'"answer"\s*:\s*"', buf)
                if m:
                    inside_answer = True
                    yield buf[m.end():]           # text after the opening quote
                    buf = ""
            else:
                end = buf.find('",')              # naive “end of answer” detector
                if end != -1:
                    yield buf[:end]
                    buf  = buf[end + 2 :]         # keep trailing bytes (meta)
                    inside_answer = False
                else:
                    yield buf
                    buf = ""

        # ── 3. send trailing metadata (pages + conf) ──────
        print('full_json:', full_json)
        meta = json.loads(full_json)
        yield f"\n\nSource pages: {meta['source_pages']}, Confidence: {meta['confidence']})"
