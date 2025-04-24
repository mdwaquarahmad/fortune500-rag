"""
Tests for the vector store module.

This module contains tests for the embedding generator and vector store.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import numpy as np

from src.vector_store.embeddings import EmbeddingGenerator
from src.vector_store.store import VectorStore

class MockEmbeddings:
    """Mock embeddings class for testing."""
    
    def embed_documents(self, texts):
        """Mock embedding generation."""
        # Return a mock embedding for each text
        return [
            [0.1] * 10 for _ in texts
        ]
    
    def embed_query(self, query):
        """Mock query embedding generation."""
        return [0.1] * 10

class TestEmbeddingGenerator:
    """Tests for the EmbeddingGenerator class."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Create a mock embeddings instance
        mock_embeddings = MockEmbeddings()
        
        # Patch the OpenAIEmbeddings to return our mock
        self.embeddings_patcher = patch(
            'src.vector_store.embeddings.OpenAIEmbeddings',
            return_value=mock_embeddings
        )
        self.mock_openai_embeddings = self.embeddings_patcher.start()
        
        # Create the embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model_name="test-model",
            api_key="test-key"
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.embeddings_patcher.stop()
    
    def test_initialization(self):
        """Test initialization of the embedding generator."""
        # Check that OpenAIEmbeddings was called with the right parameters
        self.mock_openai_embeddings.assert_called_once_with(
            model="test-model",
            openai_api_key="test-key"
        )
    
    def test_generate_embeddings(self):
        """Test generating embeddings for documents."""
        texts = ["This is a test", "Another test document"]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Check results
        assert len(embeddings) == len(texts)
        for emb in embeddings:
            assert len(emb) == 10
    
    def test_generate_query_embedding(self):
        """Test generating an embedding for a query."""
        query = "Test query"
        
        # Generate embedding
        embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Check result
        assert len(embedding) == 10
    
    def test_process_chunks(self):
        """Test processing chunks and generating embeddings."""
        chunks = [
            {"text": "Chunk 1", "metadata": {"source": "doc1.pdf"}},
            {"text": "Chunk 2", "metadata": {"source": "doc2.pdf"}}
        ]
        
        # Process chunks
        result = self.embedding_generator.process_chunks(chunks)
        
        # Check results
        assert "chunks" in result
        assert "embeddings" in result
        assert len(result["chunks"]) == len(chunks)
        assert len(result["embeddings"]) == len(chunks)

class TestVectorStore:
    """Tests for the VectorStore class."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Create a temporary directory for the vector store
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock embedding generator
        self.mock_embedding_generator = MagicMock()
        self.mock_embedding_generator.embeddings = MockEmbeddings()
        self.mock_embedding_generator.generate_embeddings.return_value = [
            [0.1] * 10 for _ in range(2)
        ]
        self.mock_embedding_generator.generate_query_embedding.return_value = [0.1] * 10
        
        # Patch chromadb
        self.mock_collection = MagicMock()
        self.mock_client = MagicMock()
        self.mock_client.get_or_create_collection.return_value = self.mock_collection
        
        self.chroma_client_patcher = patch(
            'src.vector_store.store.chromadb.PersistentClient',
            return_value=self.mock_client
        )
        self.mock_chromadb_client = self.chroma_client_patcher.start()
        
        self.chroma_patcher = patch('src.vector_store.store.Chroma')
        self.mock_chroma = self.chroma_patcher.start()
        
        # Configure the patched environment variables
        self.env_patcher = patch.dict('os.environ', {
            'CHROMA_DB_DIR': self.temp_dir
        })
        self.env_patcher.start()
        
        # Create the vector store
        self.vector_store = VectorStore(
            embedding_generator=self.mock_embedding_generator,
            collection_name="test_collection"
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.chroma_client_patcher.stop()
        self.chroma_patcher.stop()
        self.env_patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of the vector store."""
        # Check that the Chroma client was initialized correctly
        self.mock_chromadb_client.assert_called_once()
        
        # Check that the collection was created
        self.mock_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"hnsw:space": "cosine"}
        )
    
    def test_add_documents(self):
        """Test adding documents to the vector store."""
        chunks = [
            {"text": "Chunk 1", "metadata": {"source": "doc1.pdf"}},
            {"text": "Chunk 2", "metadata": {"source": "doc2.pdf"}}
        ]
        
        # Add documents
        ids = self.vector_store.add_documents(chunks)
        
        # Check that embeddings were generated
        self.mock_embedding_generator.generate_embeddings.assert_called_once_with(
            ["Chunk 1", "Chunk 2"]
        )
        
        # Check that documents were added to the collection
        self.mock_collection.add.assert_called_once()
        add_args = self.mock_collection.add.call_args[1]
        
        assert len(add_args["ids"]) == 2
        assert add_args["documents"] == ["Chunk 1", "Chunk 2"]
        assert add_args["metadatas"] == [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
        assert len(add_args["embeddings"]) == 2
        
        # Check return value
        assert len(ids) == 2
    
    def test_search(self):
        """Test searching for similar documents."""
        # Configure mock response
        self.mock_collection.query.return_value = {
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]],
            "distances": [[0.2, 0.3]]
        }
        
        # Search for documents
        results = self.vector_store.search("Test query")
        
        # Check that query embedding was generated
        self.mock_embedding_generator.generate_query_embedding.assert_called_once_with(
            "Test query"
        )
        
        # Check that collection query was called
        self.mock_collection.query.assert_called_once()
        query_args = self.mock_collection.query.call_args[1]
        
        assert len(query_args["query_embeddings"]) == 1
        assert query_args["n_results"] == 5  # Default value
        
        # Check results
        assert len(results) == 2
        assert results[0]["text"] == "Document 1"
        assert results[0]["metadata"] == {"source": "doc1.pdf"}
        assert results[0]["score"] == 0.8  # 1 - 0.2
        
        assert results[1]["text"] == "Document 2"
        assert results[1]["metadata"] == {"source": "doc2.pdf"}
        assert results[1]["score"] == 0.7  # 1 - 0.3
    
    def test_search_with_filter(self):
        """Test searching with metadata filters."""
        # Configure mock response
        self.mock_collection.query.return_value = {
            "documents": [["Document 1"]],
            "metadatas": [[{"source": "doc1.pdf", "company": "Amazon"}]],
            "distances": [[0.2]]
        }
        
        # Search with filter
        filter_criteria = {"company": "Amazon"}
        results = self.vector_store.search("Test query", filter_criteria=filter_criteria)
        
        # Check that collection query was called with filter
        self.mock_collection.query.assert_called_once()
        query_args = self.mock_collection.query.call_args[1]
        
        assert query_args["where"] == filter_criteria
        
        # Check results
        assert len(results) == 1
        assert results[0]["metadata"]["company"] == "Amazon"
    
    def test_get_stats(self):
        """Test getting statistics about the vector store."""
        # Configure mock response
        self.mock_collection.count.return_value = 10
        
        # Get stats
        stats = self.vector_store.get_stats()
        
        # Check that count was called
        self.mock_collection.count.assert_called_once()
        
        # Check results
        assert stats["collection_name"] == "test_collection"
        assert stats["document_count"] == 10