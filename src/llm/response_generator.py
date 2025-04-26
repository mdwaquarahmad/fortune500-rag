"""
Response generator module.

This module handles interactions with the LLM to generate responses
based on retrieved context and user queries.
"""

import logging
import re
from typing import Dict, List, Optional, Any

from langchain_openai import ChatOpenAI
from langchain.schema import Document

from ..config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT
from .prompt_templates import (
    QA_PROMPT, 
    FALLBACK_PROMPT, 
    FILTER_EXPLANATION_PROMPT, 
    format_document_context
)

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates responses using an LLM based on retrieved context and user queries.
    """
    
    def __init__(self, model_name: str = LLM_MODEL, temperature: float = TEMPERATURE):
        """
        Initialize the response generator with the specified LLM model.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature parameter for response generation
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY
        )
        
        logger.info(f"Initialized response generator with model: {model_name}")
    
    def generate_response(self, query: str, retrieved_docs: List[Dict], 
                          filter_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: User query string
            retrieved_docs: List of retrieved documents with text and metadata
            filter_info: Optional information about applied filters
            
        Returns:
            Dict: Generated response with additional information
        """
        # Check if we have any relevant documents
        if not retrieved_docs:
            return self._generate_fallback_response(query, filter_info)
            
        # Format the context from retrieved documents
        context = format_document_context(retrieved_docs)
        
        # Create the prompt
        prompt = QA_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            question=query
        )
        
        # Generate response
        try:
            logger.info(f"Generating response for query: {query}")
            response = self.llm.invoke(prompt)
            logger.info(f"Raw LLM response received: {response}")
            logger.info(f"Response content type: {type(response.content)}")
            logger.info(f"Response content: {response.content[:100]}...")  # Log first 100 chars
            
            # Standardize financial notation in response
            response_text = self._standardize_financial_notation(response.content)
            
            # Create the result object
            result = {
                "response": response_text,
                "sources": [doc.get("metadata", {}) for doc in retrieved_docs],
                "query": query
            }
            
            # Add filter information if provided
            if filter_info:
                result["filters"] = filter_info
                
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I'm sorry, I encountered an error while generating the response.",
                "error": str(e),
                "query": query
            }
    
    def _standardize_financial_notation(self, text: str) -> str:
        """
        Standardize financial notation to ensure consistent formatting.
        
        Args:
            text: The text containing financial information
            
        Returns:
            str: Text with standardized financial notation
        """
        # Fix billions notation
        # Convert "X billion" to "$XB"
        text = re.sub(r'(\$?\s*\d+(?:\.\d+)?)\s*billion', r'$\1B', text, flags=re.IGNORECASE)
        # Ensure dollar sign for "XB"
        text = re.sub(r'(\d+)B\b', r'$\1B', text)
        
        # Fix millions notation
        # Convert "X million" to "$XM"
        text = re.sub(r'(\$?\s*\d+(?:\.\d+)?)\s*million', r'$\1M', text, flags=re.IGNORECASE)
        # Ensure dollar sign for "XM"
        text = re.sub(r'(\d+)M\b', r'$\1M', text)
        
        # Fix ranges formatting
        # Ensure proper spacing and formatting in ranges like "$XB to $YB"
        text = re.sub(r'(\$\d+[BM])\s+to\s+(\$?\d+[BM])', r'\1 to \2', text)
        # Add dollar sign to second part of range if missing
        text = re.sub(r'(\$\d+[BM])\s+to\s+(\d+[BM])', r'\1 to $\2', text)
        
        # Fix year-over-year notation
        text = re.sub(r'([Yy]ear[-\s]over[-\s][Yy]ear)\s*\(\s*"?YoY"?\s*\)', r'\1 (YoY)', text)
        
        # Fix percentage formatting
        text = re.sub(r'(\d+)\s*%', r'\1%', text)
        
        # Fix spacing around dollar signs
        text = re.sub(r'\$\s+(\d+)', r'$\1', text)
        
        # Fix combined notation issues (e.g., "514billionin2022")
        text = re.sub(r'(\d+)billion(?:in|to)(\d+)', r'$\1B in \2', text)
        text = re.sub(r'(\d+)million(?:in|to)(\d+)', r'$\1M in \2', text)
        
        # Fix formatting errors with asterisks and italics
        text = re.sub(r'(\d+)\*B\*', r'$\1B', text)
        text = re.sub(r'(\d+)\*M\*', r'$\1M', text)
        text = re.sub(r'\*billion\*', r'billion', text)
        text = re.sub(r'\*million\*', r'million', text)
        
        # Fix specific formatting issues with "to" in ranges
        text = re.sub(r'to\*', r'to ', text)
        
        return text
    
    def generate_response_from_langchain_docs(self, query: str, 
                                             docs: List[Document],
                                             filter_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a response based on the query and LangChain documents.
        
        Args:
            query: User query string
            docs: List of LangChain Document objects
            filter_info: Optional information about applied filters
            
        Returns:
            Dict: Generated response with additional information
        """
        # Convert LangChain documents to our format
        retrieved_docs = []
        for doc in docs:
            retrieved_docs.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
            
        # Use the standard generate_response method
        return self.generate_response(query, retrieved_docs, filter_info)
    
    def _generate_fallback_response(self, query: str, 
                                   filter_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a fallback response when no relevant context is available.
        
        Args:
            query: User query string
            filter_info: Optional information about applied filters
            
        Returns:
            Dict: Generated fallback response
        """
        try:
            logger.info(f"Generating fallback response for query with no relevant context: {query}")
            
            # If filters were applied, use the filter explanation prompt
            if filter_info and filter_info.get("applied", False):
                filters_text = ", ".join([
                    f"{key}: {value}" for key, value in filter_info.items() 
                    if key != "applied"
                ])
                
                prompt = FILTER_EXPLANATION_PROMPT.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=query,
                    filters=filters_text
                )
            else:
                # Otherwise use the standard fallback prompt
                prompt = FALLBACK_PROMPT.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=query
                )
            
            # Generate fallback response
            response = self.llm.invoke(prompt)
            
            # Standardize financial notation in fallback response
            response_text = self._standardize_financial_notation(response.content)
            
            # Create the result object
            result = {
                "response": response_text,
                "sources": [],
                "query": query,
                "fallback": True
            }
            
            # Add filter information if provided
            if filter_info:
                result["filters"] = filter_info
                
            return result
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return {
                "response": "I apologize, but I don't have enough information to answer your question about Fortune 500 annual reports. Please try a different question or consider uploading relevant documents.",
                "error": str(e),
                "query": query,
                "fallback": True
            }