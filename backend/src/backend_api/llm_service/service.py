"""LLM service using Google's Gemini model."""
from typing import Dict, Any, List
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.generative_models._generative_models import HarmCategory, HarmBlockThreshold
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating responses using Gemini."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-west1",  # Using us-west1 for lower latency
        model_name: str = "gemini-2.5-flash"  # Using Gemini 2.5 Flash
    ):
        """
        Initialize the LLM service.

        Args:
            project_id: Google Cloud project ID
            location: Model location (us-west1 for lower latency)
            model_name: Name of the model to use
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        # Configure safety settings to handle content appropriately
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }

        # Initialize model with safety settings
        self.model = GenerativeModel(
            model_name,
            safety_settings=self.safety_settings
        )
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
        context = "\n\n".join([
            f"[Page {chunk['metadata']['page']}] {chunk['content']}"
            for chunk in context_chunks
        ])

        # Build the prompt
        prompt = f"""Based on the following context, answer the question and format the response as JSON with the following structure:
        {{
            "answer": "Your detailed answer here",
            "source_pages": [page numbers where information was found],
            "confidence": confidence score between 0 and 1
        }}

        Context:
        {context}

        Question: {query}

        Answer:"""

        return prompt

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a response in JSON format.

        Args:
            query: User's question
            context_chunks: Relevant document chunks

        Returns:
            Response in JSON format with answer and metadata

        Note: Uses carefully tuned parameters for optimal document Q&A:
        - temperature=0.1: Low temperature for focused, factual responses
        - top_p=0.8: Narrower nucleus sampling keeps responses on-topic
        - candidate_count=1: Single best response for lower latency
        - frequency_penalty=0.1: Slight penalty to reduce repetition
        - presence_penalty=0.0: No push for new topics in factual Q&A
        """
        try:
            # Build prompt
            prompt = self._build_prompt(query, context_chunks)

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.1,  # low randomness → factual, still natural
                    top_p=0.8,  # narrower nucleus keeps it on-topic
                    candidate_count=1,  # single best hypothesis keeps latency low
                    frequency_penalty=0.1,
                    presence_penalty=0.0,  # no push for new topics
                    max_output_tokens=1024,  # plenty for an answer + citations
                    stop_sequences=["\n\nContext:"]  # cuts off if model tries to spill prompt
                )
            )

            # Parse and validate JSON response
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return {
                    "answer": response.text,
                    "source_pages": [],
                    "confidence": 0.0
                }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
