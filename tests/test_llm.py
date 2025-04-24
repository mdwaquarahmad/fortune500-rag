"""
Tests for the LLM integration module.

This module contains tests for prompt templates and response generation.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.llm.prompt_templates import (
    QA_PROMPT, 
    FALLBACK_PROMPT, 
    format_document_context
)
from src.llm.response_generator import ResponseGenerator

class TestPromptTemplates:
    """Tests for the prompt template module."""
    
    def test_format_document_context(self):
        """Test formatting document context for prompts."""
        # Create test documents
        docs = [
            {
                "text": "Amazon reported revenue of $1 billion.",
                "metadata": {
                    "company": "Amazon",
                    "year": "2023",
                    "source": "Amazon_2023.pdf",
                    "page": 10
                }
            },
            {
                "text": "Microsoft invested in cloud infrastructure.",
                "metadata": {
                    "company": "Microsoft",
                    "year": "2022",
                    "source": "Microsoft_2022.pdf",
                    "section": "Cloud Computing"
                }
            }
        ]
        
        # Format the context
        context = format_document_context(docs)
        
        # Check that the context contains the document text
        assert "Amazon reported revenue of $1 billion" in context
        assert "Microsoft invested in cloud infrastructure" in context
        
        # Check that the context contains the source information
        assert "Company: Amazon" in context
        assert "Year: 2023" in context
        assert "Page: 10" in context
        
        assert "Company: Microsoft" in context
        assert "Year: 2022" in context
        assert "Section: Cloud Computing" in context
    
    def test_qa_prompt(self):
        """Test the QA prompt template."""
        # Create a test prompt
        prompt = QA_PROMPT.format(
            system_prompt="You are a helpful assistant.",
            context="Amazon reported revenue of $1 billion in 2023.",
            question="What was Amazon's revenue in 2023?"
        )
        
        # Check that the prompt contains the expected components
        assert "You are a helpful assistant." in prompt
        assert "Amazon reported revenue of $1 billion in 2023." in prompt
        assert "What was Amazon's revenue in 2023?" in prompt
    
    def test_fallback_prompt(self):
        """Test the fallback prompt template."""
        # Create a test prompt
        prompt = FALLBACK_PROMPT.format(
            system_prompt="You are a helpful assistant.",
            question="What was Amazon's revenue in 2023?"
        )
        
        # Check that the prompt contains the expected components
        assert "You are a helpful assistant." in prompt
        assert "What was Amazon's revenue in 2023?" in prompt
        assert "I don't have relevant context information" in prompt

class TestResponseGenerator:
    """Tests for the ResponseGenerator class."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Create a mock LLM
        self.mock_response = MagicMock()
        self.mock_response.content = "Test response"
        
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = self.mock_response
        
        # Patch ChatOpenAI to return our mock
        self.llm_patcher = patch(
            'src.llm.response_generator.ChatOpenAI',
            return_value=self.mock_llm
        )
        self.mock_chat_openai = self.llm_patcher.start()
        
        # Create the response generator
        self.response_generator = ResponseGenerator(
            model_name="gpt-4o",
            temperature=0.1
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.llm_patcher.stop()
    
    def test_initialization(self):
        """Test initialization of the response generator."""
        # Check that ChatOpenAI was called with the right parameters
        self.mock_chat_openai.assert_called_once_with(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key=None  # None in tests because we're mocking
        )
    
    def test_generate_response(self):
        """Test generating a response with retrieved documents."""
        # Create test documents
        retrieved_docs = [
            {
                "text": "Amazon reported revenue of $1 billion in 2023.",
                "metadata": {
                    "company": "Amazon",
                    "year": "2023",
                    "source": "Amazon_2023.pdf"
                }
            }
        ]
        
        # Generate a response
        result = self.response_generator.generate_response(
            query="What was Amazon's revenue in 2023?",
            retrieved_docs=retrieved_docs
        )
        
        # Check that the LLM was called
        self.mock_llm.invoke.assert_called_once()
        
        # Check the result
        assert result["response"] == "Test response"
        assert result["query"] == "What was Amazon's revenue in 2023?"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["company"] == "Amazon"
    
    def test_generate_fallback_response(self):
        """Test generating a fallback response when no documents are retrieved."""
        # Generate a fallback response
        result = self.response_generator._generate_fallback_response(
            query="What was Amazon's revenue in 2023?"
        )
        
        # Check that the LLM was called with a fallback prompt
        self.mock_llm.invoke.assert_called_once()
        
        # Check the result
        assert result["response"] == "Test response"
        assert result["query"] == "What was Amazon's revenue in 2023?"
        assert len(result["sources"]) == 0
        assert result["fallback"] is True
    
    def test_generate_response_with_filters(self):
        """Test generating a response with metadata filters."""
        # Create test documents
        retrieved_docs = [
            {
                "text": "Amazon reported revenue of $1 billion in 2023.",
                "metadata": {
                    "company": "Amazon",
                    "year": "2023",
                    "source": "Amazon_2023.pdf"
                }
            }
        ]
        
        # Create filter info
        filter_info = {
            "company": "Amazon",
            "year": "2023",
            "applied": True
        }
        
        # Generate a response
        result = self.response_generator.generate_response(
            query="What was Amazon's revenue in 2023?",
            retrieved_docs=retrieved_docs,
            filter_info=filter_info
        )
        
        # Check the result
        assert result["response"] == "Test response"
        assert "filters" in result
        assert result["filters"]["company"] == "Amazon"
        assert result["filters"]["year"] == "2023"
    
    def test_error_handling(self):
        """Test error handling in response generation."""
        # Configure the mock to raise an exception
        self.mock_llm.invoke.side_effect = Exception("Test error")
        
        # Create test documents
        retrieved_docs = [
            {
                "text": "Amazon reported revenue of $1 billion in 2023.",
                "metadata": {
                    "company": "Amazon",
                    "year": "2023",
                    "source": "Amazon_2023.pdf"
                }
            }
        ]
        
        # Generate a response
        result = self.response_generator.generate_response(
            query="What was Amazon's revenue in 2023?",
            retrieved_docs=retrieved_docs
        )
        
        # Check that the result contains an error
        assert "error" in result
        assert "Test error" in result["error"]