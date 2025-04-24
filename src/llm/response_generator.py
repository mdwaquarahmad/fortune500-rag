"""
Response generator module.

This module handles interactions with the LLM to generate responses
based on retrieved context and user queries.
"""

import logging
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
            response_text = response.content
            
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
            response_text = response.content
            
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
