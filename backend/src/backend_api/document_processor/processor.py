"""Document processing module for handling PDF documents."""
import re
from typing import List, Dict, Any
import unicodedata
import pymupdf  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean the text by removing unwanted characters and formatting.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    # Remove unwanted characters and formatting
    text = re.sub(r"^\s*Page \d+\s*$", "", text, flags=re.MULTILINE)
    # 2. undo hyphenated line‐wraps
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # 3. collapse excess whitespace
    text = re.sub(r"\s+\n", "\n", text)
    # 4. normalise unicode (ﬁ → fi, etc.)
    text = unicodedata.normalize("NFKC", text)
    return text.strip()

class DocumentProcessor:
    """Handles document loading, processing, and chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict containing document ID and chunks

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a PDF
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not path.suffix.lower() == '.pdf':
                raise ValueError("Only PDF files are supported")

            # Generate unique document ID
            document_id = str(uuid.uuid4())

            # Load PDF with PyMuPDF
            doc = pymupdf.open(file_path)
            chunks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                text = clean_text(text)
                
                # Process and chunk the page text
                page_chunks = self.text_splitter.split_text(text)
                chunks.extend([
                    {
                        'content': chunk,
                        'metadata': {
                            'page': page_num + 1,
                            'source': file_path,
                            'total_pages': len(doc)
                        }
                    }
                    for chunk in page_chunks
                ])

            return {
                'document_id': document_id,
                'chunks': chunks,
                'metadata': {
                    'filename': path.name,
                    'total_pages': len(doc),
                    'total_chunks': len(chunks)
                }
            }

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def get_relevant_chunks(
        self,
        document_id: str,
        query: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get relevant document chunks for a query.

        Args:
            document_id: ID of the document
            query: User's question
            k: Number of chunks to retrieve

        Returns:
            List of relevant document chunks

        Note: This is a placeholder. The actual implementation will use
        the vector store to retrieve relevant chunks.
        """
        # TODO: Implement vector store retrieval
        return []
