"""OpenAI embeddings service."""
from typing import List
import openai
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingsService:
    """Service for generating embeddings using OpenAI."""

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embeddings service.

        Args:
            model: OpenAI embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(
            model=model,
            dimensions=1536,  # Dimensionality of text-embedding-3-small
            show_progress_bar=True
        )
        logger.info(f"Initialized embeddings service with model {model}")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings as float arrays
        """
        try:
            # Generate embeddings
            embeddings = await self.embeddings.aembed_documents(texts)
            
            # Convert to native Python lists for ChromaDB
            return [embedding.tolist() for embedding in embeddings]

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def get_query_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Text to embed

        Returns:
            Query embedding as float array
        """
        try:
            # Generate embedding
            embedding = await self.embeddings.aembed_query(text)
            
            # Convert to native Python list for ChromaDB
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
