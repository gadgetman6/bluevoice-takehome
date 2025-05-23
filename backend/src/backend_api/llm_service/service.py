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

END_QUOTE_RE = re.compile(r'(?<!\\)"\s*,')

ANSWER_FIELD_RE = re.compile(
    r'"answer"\s*:\s*"'  # opening quote
    r'(?P<ans>(?:[^"\\]|\\.)*?)'  # non-greedy match for answer content
    r'"\s*,',  # closing quote
    re.IGNORECASE | re.DOTALL,
)

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

        # Initialize model with safety settings
        self.model = GenerativeModel(model_name)
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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield answer chunks as JSON objects; stop early if confidence is low."""
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
            ),
            stream=True,
        )

        buf = ""
        checked_conf = False
        inside_answer = False
        current_answer = ""

        for chunk in responses:
            # Get delta text from chunk
            delta = "".join(
                part.text
                for cand in chunk.candidates
                for part in cand.content.parts
                if hasattr(part, "text")
            )
            buf += delta
            current_answer += delta

            # Check the confidence value
            if not checked_conf and '"confidence"' in buf:
                try:
                    partial_json = json.loads(buf + "}")
                    conf = partial_json["confidence"]
                    checked_conf = True
                    if conf < 0.30:  # confidence threshold
                        responses.close()
                        yield {"type": "error", "content": "Confidence too low"}
                        return
                except json.JSONDecodeError:
                    pass

            # Extract and stream the answer
            if not inside_answer:
                # If the entire answer is available, yield it
                m = ANSWER_FIELD_RE.search(buf)
                if m:
                    yield {
                        "type": "chunk",
                        "content": m.group("ans"),
                    }  # text after the opening quote
                    buf = buf[m.end() :]
                # Else, if we just see the start of the answer field
                else:
                    # If we have a partial answer, yield it
                    partial_match = re.search(r'"answer"\s*:\s*"', buf)
                    if partial_match:
                        inside_answer = True
                        start = partial_match.end()
                        yield {
                            "type": "chunk",
                            "content": buf[start:],
                        }
                        buf = ""
            else:
                end_match = END_QUOTE_RE.search(buf)  # end of answer field
                if end_match:
                    end = end_match.start()
                    yield {
                        "type": "chunk",
                        "content": buf[:end],
                    }
                    buf = buf[end:]  # keep trailing bytes
                    inside_answer = False
                else:
                    yield {
                        "type": "chunk",
                        "content": buf[:-1],
                    }
                    buf = buf[-1:]  # keep trailing bytes

        # Attempt to parse the final buffer as JSON
        try:
            # Attempt to parse the buffer as JSON
            parsed = json.loads(current_answer)
        except json.JSONDecodeError:
            # If parsing fails, try again with a cleaned buffer
            yield {"type": "error", "content": "Could not get source pages"}
            return

        # If we have a complete answer, yield final response
        if all(k in parsed for k in ["answer", "confidence", "source_pages"]):
            yield {
                "type": "final",
                "content": {
                    "answer": parsed["answer"],
                    "confidence": parsed["confidence"],
                    "source_pages": parsed["source_pages"],
                },
            }
            return
