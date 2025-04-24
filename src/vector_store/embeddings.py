"""
Embeddings module.

This module handles the creation of vector embeddings for text chunks using
OpenAI's embedding models.
"""

import logging
from typing import Dict, List, Union

from langchain_openai import OpenAIEmbeddings
from ..config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using OpenAI's embedding models.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, api_key: str = OPENAI_API_KEY):
        """
        Initialize the embedding generator with the specified model.
        
        Args:
            model_name: Name of the OpenAI embedding model to use
            api_key: OpenAI API key
        """
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
        logger.info(f"Initialized embedding generator with model: {model_name}")
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts asynchronously.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings asynchronously")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
            
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for a single query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            List[float]: Embedding vector for the query
        """
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
            
    def process_chunks(self, chunks: List[Dict]) -> Dict[str, Union[List[Dict], List[List[float]]]]:
        """
        Process a list of text chunks and generate embeddings for each.
        
        Args:
            chunks: List of text chunk dictionaries with text and metadata
                [
                    {
                        "text": str,
                        "metadata": Dict
                    },
                    ...
                ]
                
        Returns:
            Dict: Dictionary containing chunks and their embeddings
                {
                    "chunks": List[Dict],
                    "embeddings": List[List[float]]
                }
        """
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        return {
            "chunks": chunks,
            "embeddings": embeddings
        }