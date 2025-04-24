"""
Vector store module.

This module handles storage and retrieval of document embeddings using Chroma DB.
"""

import logging
import uuid
from typing import Dict, List, Optional, Union, Any

import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.schema import Document

from src.config import CHROMA_DB_DIR, DISTANCE_METRIC, SEARCH_TOP_K
from src.vector_store.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages storage and retrieval of document embeddings using Chroma DB.
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator, collection_name: str = "fortune500_docs"):
        """
        Initialize the vector store with an embedding generator.
        
        Args:
            embedding_generator: The embedding generator to use
            collection_name: Name of the Chroma collection to use
        """
        self.embedding_generator = embedding_generator
        self.collection_name = collection_name
        self.db_path = CHROMA_DB_DIR
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize or get collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": DISTANCE_METRIC}
            )
            logger.info(f"Initialized Chroma collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error initializing Chroma collection: {e}")
            raise
            
        # Initialize langchain wrapper for more advanced functionality
        self.langchain_db = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_generator.embeddings
        )
            
    def add_documents(self, chunks: List[Dict]) -> List[str]:
        """
        Add document chunks to the vector store.
        
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
            List[str]: List of document IDs
        """
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Extract text and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} documents to vector store")
        return ids
        
    def search(self, query: str, k: int = SEARCH_TOP_K, 
               filter_criteria: Optional[Dict] = None) -> List[Dict]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_criteria: Optional filter criteria for metadata
            
        Returns:
            List[Dict]: List of search results with text, metadata, and score
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Prepare filter
        where_document = None
        if filter_criteria:
            where_document = filter_criteria
            
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_document
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i, (document, metadata, score) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1 - score if DISTANCE_METRIC == "cosine" else score
                
                formatted_results.append({
                    "text": document,
                    "metadata": metadata,
                    "score": similarity
                })
                
        logger.info(f"Found {len(formatted_results)} results for query")
        return formatted_results
        
    def search_with_langchain(self, query: str, k: int = SEARCH_TOP_K,
                              filter_criteria: Optional[Dict] = None) -> List[Document]:
        """
        Search for documents using LangChain's retriever interface.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_criteria: Optional filter criteria for metadata
            
        Returns:
            List[Document]: List of langchain Document objects
        """
        # Create retriever with filter if needed
        retriever = self.langchain_db.as_retriever(
            search_kwargs={"k": k, "filter": filter_criteria}
        )
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Found {len(docs)} documents with LangChain retriever")
        
        return docs
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict: Statistics about the collection
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "path": str(self.db_path)
        }
        
    def delete_collection(self) -> None:
        """
        Delete the collection and all its data.
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise