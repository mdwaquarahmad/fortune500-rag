"""
Text chunking module.

This module provides functionality to split text into semantically meaningful chunks
for effective embedding and retrieval.
"""

import logging
import re
from typing import Dict, List, Optional

from ..config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Chunks text documents into smaller pieces for embedding and retrieval.
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize text chunker with specified chunk size and overlap.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_documents(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Process a list of text blocks and chunk each one appropriately.
        
        Args:
            text_blocks: List of dictionaries containing text and metadata
                [
                    {
                        "text": str,
                        "metadata": Dict
                    },
                    ...
                ]
                
        Returns:
            List[Dict]: List of chunked text blocks with preserved metadata
        """
        chunked_blocks = []
        
        for block in text_blocks:
            text = block.get("text", "")
            metadata = block.get("metadata", {})
            
            # Skip empty blocks
            if not text.strip():
                continue
                
            # Chunk the text
            chunks = self._chunk_text(text)
            
            # Create new text blocks with chunks
            for i, chunk in enumerate(chunks):
                chunked_blocks.append({
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk": i + 1,
                        "total_chunks": len(chunks)
                    }
                })
                
        logger.info(f"Created {len(chunked_blocks)} chunks from {len(text_blocks)} text blocks")
        return chunked_blocks
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on semantic boundaries.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # If text is smaller than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]
            
        # Try to split on paragraph boundaries first
        chunks = self._split_by_paragraph(text)
        
        # If chunks are still too large, split by sentence
        chunks = self._ensure_max_chunk_size(chunks, self._split_by_sentence)
        
        # If chunks are still too large, split by character
        chunks = self._ensure_max_chunk_size(chunks, self._split_by_character)
        
        return chunks
    
    def _split_by_paragraph(self, text: str) -> List[str]:
        """
        Split text by paragraph boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of paragraphs
        """
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Remove empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Combine short paragraphs if needed
        return self._recombine_small_chunks(paragraphs)
    
    def _split_by_sentence(self, text: str) -> List[str]:
        """
        Split text by sentence boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of sentences
        """
        # Basic sentence splitting pattern
        # This is a simple approach - for better results, consider using nltk or spacy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Recombine short sentences
        return self._recombine_small_chunks(sentences)
    
    def _split_by_character(self, text: str) -> List[str]:
        """
        Split text by character count as a last resort.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Take a chunk of size chunk_size
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            
            # Move start position accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else end
            
        return chunks
    
    def _ensure_max_chunk_size(self, chunks: List[str], split_function) -> List[str]:
        """
        Ensure all chunks are below the maximum chunk size.
        
        Args:
            chunks: List of text chunks
            split_function: Function to use for further splitting
            
        Returns:
            List[str]: List of properly sized chunks
        """
        result = []
        
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                result.append(chunk)
            else:
                # Split chunk further
                smaller_chunks = split_function(chunk)
                result.extend(smaller_chunks)
                
        return result
    
    def _recombine_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Recombine small chunks to reduce the total number of chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List[str]: List of recombined chunks
        """
        result = []
        current_chunk = ""
        
        for chunk in chunks:
            # If adding the chunk would exceed the size limit
            if len(current_chunk) + len(chunk) > self.chunk_size:
                # Save the current chunk if not empty
                if current_chunk:
                    result.append(current_chunk)
                    
                # Start a new chunk
                current_chunk = chunk
            else:
                # Add to current chunk with a space if needed
                if current_chunk:
                    current_chunk += " " + chunk
                else:
                    current_chunk = chunk
                    
        # Add the last chunk if not empty
        if current_chunk:
            result.append(current_chunk)
            
        return result