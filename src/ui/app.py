"""
Streamlit application for the Fortune 500 RAG Chatbot.

This module provides a web-based user interface for document uploading,
question asking, and viewing responses.
"""

import os
import logging
import streamlit as st
from pathlib import Path
import time
from datetime import datetime
import pandas as pd

from src.document_processor.loader import DocumentLoader
from src.document_processor.text_extractor import TextExtractor
from src.document_processor.chunker import TextChunker
from src.vector_store.embeddings import EmbeddingGenerator
from src.vector_store.store import VectorStore
from src.llm.response_generator import ResponseGenerator
from src.config import STREAMLIT_TITLE, STREAMLIT_DESCRIPTION, UPLOAD_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration globally
st.set_page_config(
    page_title="Fortune 500 RAG Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processor" not in st.session_state:
    st.session_state.document_processor = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_generator" not in st.session_state:
    st.session_state.embedding_generator = None
if "response_generator" not in st.session_state:
    st.session_state.response_generator = None
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False

def initialize_components():
    """Initialize all RAG components."""
    # Initialize document processing components
    if not st.session_state.document_processor:
        st.session_state.document_processor = {
            "loader": DocumentLoader(),
            "extractor": TextExtractor(),
            "chunker": TextChunker()
        }
    
    # Initialize embedding generator
    if not st.session_state.embedding_generator:
        with st.spinner("Initializing embedding model..."):
            st.session_state.embedding_generator = EmbeddingGenerator()
    
    # Initialize vector store
    if not st.session_state.vector_store:
        with st.spinner("Initializing vector database..."):
            st.session_state.vector_store = VectorStore(
                st.session_state.embedding_generator
            )
    
    # Initialize response generator
    if not st.session_state.response_generator:
        with st.spinner("Initializing LLM..."):
            st.session_state.response_generator = ResponseGenerator()

def process_uploaded_file(uploaded_file):
    """
    Process an uploaded file and add it to the vector store.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    """
    try:
        # Set processing status
        st.session_state.processing_status = f"Processing {uploaded_file.name}..."
        
        # Save the uploaded file
        loader = st.session_state.document_processor["loader"]
        file_path = loader.save_uploaded_file(uploaded_file, uploaded_file.name)
        
        # Load file info
        file_info = loader.load_file(file_path)
        
        # Extract text
        extractor = st.session_state.document_processor["extractor"]
        text_blocks = extractor.extract_text(file_info)
        
        # Chunk text
        chunker = st.session_state.document_processor["chunker"]
        chunks = chunker.chunk_documents(text_blocks)
        
        # Add to vector store
        vector_store = st.session_state.vector_store
        doc_ids = vector_store.add_documents(chunks)
        
        # Update status
        st.session_state.processing_status = f"Successfully processed {uploaded_file.name}"
        st.session_state.files_uploaded = True
        
        return True
    except Exception as e:
        # Update status with error
        st.session_state.processing_status = f"Error processing {uploaded_file.name}: {str(e)}"
        logger.error(f"Error processing file: {e}", exc_info=True)
        return False

def deduplicate_sources(sources):
    """
    Remove duplicate sources based on document name.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        List: Deduplicated sources
    """
    if not sources:
        return []
        
    unique_sources = []
    seen_documents = set()
    
    for source in sources:
        doc_name = source.get("source", "")
        if doc_name and doc_name not in seen_documents:
            seen_documents.add(doc_name)
            unique_sources.append(source)
    
    return unique_sources[:3]  # Limit to top 3 sources for simplicity

def ask_question(question):
    """
    Process a user question and generate a response.
    """
    if not question.strip():
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    try:
        # Search for relevant documents
        with st.spinner("Searching for relevant information..."):
            vector_store = st.session_state.vector_store
            search_results = vector_store.search(question)
        
        # Generate response
        with st.spinner("Generating response..."):
            response_generator = st.session_state.response_generator
            result = response_generator.generate_response(
                question,
                search_results
            )
        
        # Deduplicate sources
        if "sources" in result:
            result["sources"] = deduplicate_sources(result["sources"])
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": result["response"],
            "sources": result.get("sources", [])
        })
        logger.info(f"Added response to chat history")
        
    except Exception as e:
        # Add error message to chat history
        logger.error(f"Error processing question: {e}", exc_info=True)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"I'm sorry, I encountered an error: {str(e)}"
        })

def clear_chat():
    """Clear the chat history."""
    st.session_state.chat_history = []

def check_files_available():
    """Check if files are available for querying."""
    if "document_processor" in st.session_state and st.session_state.document_processor:
        loader = st.session_state.document_processor["loader"]
        files = loader.get_file_list()
        return len(files) > 0
    return False

def main():
    # Create a sidebar for document upload and management
    with st.sidebar:
        st.title("Upload Documents")
        st.write("Upload Fortune 500 Annual Reports")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            accept_multiple_files=True,
            type=["pdf", "docx", "pptx", "png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
        # Process uploaded files
        if uploaded_files:
            # Initialize components
            initialize_components()
            
            # Process each file
            for uploaded_file in uploaded_files:
                process_uploaded_file(uploaded_file)
                
        # Show processing status if any
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        # Chat options
        st.divider()
        if st.button("Clear Conversation"):
            clear_chat()
    
    # Main chat interface
    st.title("Fortune 500 RAG Chatbot")
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"].replace("$", "\$"))
    
    # Chat input
    files_available = check_files_available()
    
    # Query input
    placeholder = "Ask a question about Fortune 500 annual reports..."
    user_input = st.chat_input(placeholder, disabled=not files_available)
    
    if not files_available:
        st.info("📤 Please upload documents to start asking questions")
    
    if user_input:
        # Initialize components if needed
        if not st.session_state.vector_store:
            initialize_components()
            
        # Process the question
        ask_question(user_input)
        st.rerun()

if __name__ == "__main__":
    main()