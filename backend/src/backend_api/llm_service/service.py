"""LLM service using Google's Gemini model."""

import re
from typing import Dict, Any, List, AsyncGenerator
import ijson
# from ijson.common import ObjectBuilder # Not used
import asyncio # May be needed for async generator bridge
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

# END_QUOTE_RE and ANSWER_FIELD_RE are removed as per plan

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
        """Yield answer chunks as JSON objects using ijson; stop early if confidence is low."""
        prompt = self._build_prompt(query, context_chunks)

        # Configure model generation parameters
        generation_params = GenerationConfig(
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
        )

        responses = self.model.generate_content(
            prompt, generation_config=generation_params, stream=True
        )

        async def llm_text_stream():
            """Async generator to yield text chunks from LLM responses."""
            try:
                async for chunk in responses:
                    if chunk.candidates:
                        for cand in chunk.candidates:
                            if cand.content and cand.content.parts:
                                for part in cand.content.parts:
                                    if hasattr(part, "text") and part.text:
                                        yield part.text.encode('utf-8')
            except Exception as e:
                logger.error(f"Error in LLM response stream: {e}")
                pass

        class AsyncGeneratorReader:
            """Wraps an async generator of byte chunks to provide an async read() interface."""
            def __init__(self, agen):
                self._agen = agen
                self._buffer = bytearray()
                self._eof = False

            async def read(self, size=-1):
                if size == -1: # Read until EOF
                    while not self._eof:
                        await self._load_more_data()
                    data = self._buffer
                    self._buffer = bytearray()
                    return bytes(data)

                while len(self._buffer) < size and not self._eof:
                    await self._load_more_data()
                
                data = self._buffer[:size] # Corrected self_buffer to self._buffer
                self._buffer = self._buffer[size:]
                return bytes(data)

            async def _load_more_data(self):
                if self._eof:
                    return
                try:
                    chunk = await self._agen.__anext__()
                    if chunk:
                        self._buffer.extend(chunk)
                    else: # Should not happen with well-behaved generators yielding non-empty bytes
                        self._eof = True 
                except StopAsyncIteration:
                    self._eof = True
        
        ijson_input_stream = AsyncGeneratorReader(llm_text_stream())

        confidence_value = None
        confidence_checked = False
        answer_parts = []
        source_pages_value = []
        
        parsing_answer_string = False
        parsing_source_pages_array = False
        
        final_parsed_data = {
            "answer": None,
            "confidence": None,
            "source_pages": None,
        }

        logger.debug(f"Initializing ijson parsing with stream type: {type(ijson_input_stream)}")
        try:
            async for path, event, value in ijson.parse_async(ijson_input_stream):
                logger.debug(f"RAW IJSON EVENT: path='{path}', event='{event}', value_repr='{repr(value)[:60]}'")

                if path == "confidence" and event == "number":
                    confidence_value = value
                    final_parsed_data["confidence"] = confidence_value
                    confidence_checked = True
                    if confidence_value < 0.30:
                        logger.info(f"Confidence {confidence_value} is below threshold 0.30.")
                        # It's important to close the response stream if we exit early.
                        # The `responses` object from VertexAI SDK is an AsyncIterable.
                        # There isn't a direct 'close()' method on the async iterator itself.
                        # It should be garbage collected, or if it's based on an underlying http stream,
                        # that stream should be managed by the SDK when iteration stops.
                        # For now, we assume breaking the loop is sufficient.
                        yield {"type": "error", "content": "Confidence too low"}
                        return

                elif path == "answer":
                    # For logging, decode byte value if it's bytes, or display as is.
                    log_value = value
                    if isinstance(value, bytes):
                        try:
                            log_value = value.decode('utf-8', errors='replace')
                        except: # Fallback if decode fails for some reason
                            pass 
                    log_value = value
                    if isinstance(value, bytes):
                        try:
                            log_value = value.decode('utf-8', errors='replace')
                        except: 
                            pass 
                    logger.debug(f"ANSWER PATH: event='{event}', value_repr='{repr(log_value)[:50]}'")

                    # If ijson gives a direct 'string' event for the 'answer' path
                    if event == "string":
                        if isinstance(value, bytes):
                            decoded_value = value.decode('utf-8')
                        else: # it's already a string
                            decoded_value = value
                        
                        yield {"type": "chunk", "content": decoded_value}
                        answer_parts.append(decoded_value) 
                        logger.debug(f"ANSWER PATH: Appended to answer_parts: '{decoded_value}'")
                        # Update final_parsed_data as parts come in, assuming ijson might chunk a very long string
                        final_parsed_data["answer"] = "".join(answer_parts) 
                        logger.debug(f"ANSWER PATH: Updated final_parsed_data['answer'] = '{final_parsed_data['answer']}'")
                    # Note: The parsing_answer_string flag and start_string/end_string events for 'answer'
                    # seem to be bypassed by ijson's behavior for this simple string field,
                    # so they are effectively not used for 'answer' path with this simplified logic.

                elif path == "source_pages":
                    # For logging, decode byte value if it's bytes, or display as is.
                    log_value_sp = value
                    if isinstance(value, bytes): # Should not be bytes for source_pages numbers, but good practice
                        try:
                            log_value_sp = value.decode('utf-8', errors='replace')
                        except:
                            pass
                    logger.debug(f"SOURCE_PAGES PATH: event='{event}', value_repr='{repr(log_value_sp)}'")
                    if event == "start_array":
                        parsing_source_pages_array = True
                        source_pages_value = [] 
                        logger.debug("SOURCE_PAGES PATH: Started parsing source_pages array.")
                    elif event == "end_array":
                        parsing_source_pages_array = False
                        final_parsed_data["source_pages"] = source_pages_value
                        logger.debug(f"SOURCE_PAGES PATH: Ended source_pages array. Collected: {source_pages_value}")
                
                elif parsing_source_pages_array and event == "number": 
                    if path.startswith("source_pages.") and path.endswith(".item"): 
                         source_pages_value.append(value)
                         logger.debug(f"SOURCE_PAGES PATH: Appended page number: {value}")

        except ijson.common.IncompleteJSONError as e:
            logger.error(f"Incomplete JSON response from LLM: {e}", exc_info=True)
            if not confidence_checked: # If confidence wasn't even checked, it's a fundamental parsing issue early on
                yield {"type": "error", "content": f"Failed to parse initial JSON: {e}"}
            elif confidence_value is not None and confidence_value < 0.30: # Low confidence error already yielded or would be
                pass # Error already handled or will be by confidence check post-loop
            else: # Confidence was ok or not determined yet but stream failed later for other parts
                 yield {"type": "error", "content": f"Could not get source pages due to incomplete JSON: {e}"}
            return
        except Exception as e: 
            logger.error(f"Error processing LLM stream with ijson: {e}", exc_info=True)
            yield {"type": "error", "content": "Error processing LLM response"}
            return
        finally:
            logger.debug("Finished processing ijson events or an exception occurred.")
            pass

        logger.debug(f"Post-loop: confidence_value={confidence_value}, answer_parts={answer_parts}, final_parsed_data={final_parsed_data}")

        if confidence_value is None: 
            # This case might be hit if IncompleteJSONError happened before confidence was parsed.
            # The IncompleteJSONError yield above is more specific.
            if not any(res.get("type") == "error" for res in []): # poor way to check if error already yielded
                 yield {"type": "error", "content": "Could not determine confidence"}
            return

        if final_parsed_data["answer"] is None and answer_parts: # If end_string for answer was missed
            final_parsed_data["answer"] = "".join(answer_parts)
            logger.debug(f"Reconstructed answer post-loop: {final_parsed_data['answer']}")
            
        if (
            final_parsed_data["answer"] is not None
            and final_parsed_data["confidence"] is not None
            and final_parsed_data["source_pages"] is not None # This was "Present"
        ):
            yield {
                "type": "final",
                "content": {
                    "answer": final_parsed_data["answer"],
                    "confidence": final_parsed_data["confidence"],
                    "source_pages": final_parsed_data["source_pages"],
                },
            }
        else:
            # This case handles if the stream finished but some parts were missing,
            # and confidence was okay.
            logger.warning(f"LLM stream finished but required data incomplete. Confidence: {final_parsed_data['confidence']}, Answer: {'Present' if final_parsed_data['answer'] else 'Missing'}, Source Pages: {'Present' if final_parsed_data['source_pages'] is not None else 'Missing'}")
            yield {"type": "error", "content": "Could not get source pages"}
