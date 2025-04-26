"""
Tests for the document processor module.

This module contains tests for the document loader, text extractor, and chunker.
"""

import os
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.document_processor.loader import DocumentLoader
from src.document_processor.text_extractor import TextExtractor
from src.document_processor.chunker import TextChunker
from src.config import SUPPORTED_FILE_TYPES

class TestDocumentLoader:
    """Tests for the DocumentLoader class."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Create a temporary directory for uploads
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a loader with the temporary directory
        self.loader = DocumentLoader()
        self.loader.upload_dir = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_validate_file_type(self):
        """Test validation of file types."""
        # Test valid file types
        for file_type, extensions in SUPPORTED_FILE_TYPES.items():
            for ext in extensions:
                filename = f"test{ext}"
                is_valid, detected_type = self.loader.validate_file_type(filename)
                assert is_valid is True
                assert detected_type == file_type
        
        # Test invalid file type
        is_valid, _ = self.loader.validate_file_type("test.xyz")
        assert is_valid is False
    
    def test_extract_metadata(self):
        """Test metadata extraction from filenames."""
        # Test company and year
        metadata = self.loader.extract_metadata("Amazon_2023.pdf")
        assert metadata["company"] == "Amazon"
        assert metadata["year"] == "2023"
        
        # Test company with spaces
        metadata = self.loader.extract_metadata("Microsoft-Corporation_2022.docx")
        assert metadata["company"] == "Microsoft Corporation"
        assert metadata["year"] == "2022"
        
        # Test source is included
        metadata = self.loader.extract_metadata("Apple_2023.pdf")
        assert metadata["source"] == "Apple_2023.pdf"

class TestTextChunker:
    """Tests for the TextChunker class."""
    
    def setup_method(self):
        """Set up the test environment."""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        # Create a long text
        text = "This is a test. " * 20  # 300 characters
        
        # Chunk the text
        chunks = self.chunker._chunk_text(text)
        
        # Should split into chunks
        assert len(chunks) > 1
        
        # Each chunk should be less than or equal to chunk_size
        for chunk in chunks:
            assert len(chunk) <= 100
    
    def test_chunk_documents(self):
        """Test document chunking functionality."""
        # Create test documents
        text_blocks = [
            {
                "text": "This is a test. " * 20,
                "metadata": {"source": "doc1.pdf", "page": 1}
            },
            {
                "text": "Another test document. " * 15,
                "metadata": {"source": "doc2.pdf", "page": 1}
            }
        ]
        
        # Chunk the documents
        chunked_blocks = self.chunker.chunk_documents(text_blocks)
        
        # Should have created multiple chunks
        assert len(chunked_blocks) > 2
        
        # Metadata should be preserved and enhanced
        for block in chunked_blocks:
            assert "source" in block["metadata"]
            assert "chunk" in block["metadata"]
            assert "total_chunks" in block["metadata"]