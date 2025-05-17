"""Vector store implementation using ChromaDB with OpenAI embeddings."""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import logging
from pathlib import Path

from .embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for document chunks using ChromaDB with OpenAI embeddings."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings service
        self.embeddings = EmbeddingsService(model=embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized vector store at {persist_dir} with {embedding_model}")

    async def add_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Add document chunks to the vector store.

        Args:
            document_id: ID of the document
            chunks: List of document chunks with content and metadata
        """
        try:
            # Extract texts and metadata
            texts = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]

            # Generate embeddings using OpenAI
            embeddings = await self.embeddings.get_embeddings(texts)

            # Add to collection with embeddings
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(chunks)} chunks for document {document_id}")

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            raise

    async def search(
        self,
        query: str,
        document_id: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            document_id: Optional document ID to restrict search
            limit: Maximum number of results

        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        try:
            # Generate query embedding using OpenAI
            query_embedding = await self.embeddings.get_query_embedding(query)

            # Prepare where clause if document_id is provided
            where = {"source": {"$contains": document_id}} if document_id else None

            # Query the collection using the embedding
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            chunks = []
            for i in range(len(results['ids'][0])):
                chunks.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                })

            # Sort by similarity (highest first)
            chunks.sort(key=lambda x: x['similarity'], reverse=True)

            return chunks

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset the vector store by deleting all data."""
        try:
            self.client.reset()
            logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
            raise
