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

# Initialize session state variables
def init_session_state():
    """Initialize Streamlit session state variables."""
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
        st.session_state.processing_status = f"Successfully processed {uploaded_file.name} ({len(chunks)} chunks)"
        time.sleep(2)  # Show success message briefly
        st.session_state.processing_status = None
        
        return True
    except Exception as e:
        # Update status with error
        st.session_state.processing_status = f"Error processing {uploaded_file.name}: {str(e)}"
        logger.error(f"Error processing file: {e}", exc_info=True)
        return False

def get_available_metadata_filters():
    """
    Get available metadata filters from the uploaded documents.
    
    Returns:
        Dict: Dictionary of available filter options
    """
    loader = st.session_state.document_processor["loader"]
    files = loader.get_file_list()
    
    companies = set()
    years = set()
    
    for file in files:
        metadata = file.get("metadata", {})
        if "company" in metadata:
            companies.add(metadata["company"])
        if "year" in metadata:
            years.add(metadata["year"])
    
    return {
        "companies": sorted(list(companies)),
        "years": sorted(list(years))
    }

def ask_question(question, company_filter=None, year_filter=None):
    """
    Process a user question and generate a response.
    
    Args:
        question: User's question
        company_filter: Optional company filter
        year_filter: Optional year filter
    """
    if not question.strip():
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Prepare filter criteria
    filter_criteria = {}
    filter_info = {"applied": False}
    
    if company_filter:
        filter_criteria["company"] = company_filter
        filter_info["company"] = company_filter
        filter_info["applied"] = True
    
    if year_filter:
        filter_criteria["year"] = year_filter
        filter_info["year"] = year_filter
        filter_info["applied"] = True
    
    try:
        # Search for relevant documents
        with st.spinner("Searching for relevant information..."):
            vector_store = st.session_state.vector_store
            search_results = vector_store.search(
                question,
                filter_criteria=filter_criteria if filter_criteria else None
            )
        
        # Generate response
        with st.spinner("Generating response..."):
            response_generator = st.session_state.response_generator
            result = response_generator.generate_response(
                question,
                search_results,
                filter_info if filter_info["applied"] else None
            )
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": result["response"],
            "sources": result.get("sources", [])
        })
        logger.info(f"Added assistant response to chat history: {result['response'][:100]}...")
    except Exception as e:
        # Add error message to chat history
        logger.error(f"Error processing question: {e}", exc_info=True)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"I'm sorry, I encountered an error: {str(e)}"
        })

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []

def export_chat_history():
    """Export chat history to a CSV file."""
    if not st.session_state.chat_history:
        st.warning("No chat history to export.")
        return
    
    # Convert chat history to DataFrame
    data = []
    for msg in st.session_state.chat_history:
        data.append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    df = pd.DataFrame(data)
    
    # Create CSV download link
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Chat History",
        data=csv,
        file_name="fortune500_chat_history.csv",
        mime="text/csv"
    )

def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title=STREAMLIT_TITLE,
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Page header
    st.title(STREAMLIT_TITLE)
    st.write(STREAMLIT_DESCRIPTION)
    
    # Create sidebar
    with st.sidebar:
        st.header("Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Fortune 500 Annual Reports",
            accept_multiple_files=True,
            type=["pdf", "docx", "pptx", "png", "jpg", "jpeg"]
        )
        
        # Process uploaded files
        if uploaded_files:
            # Initialize components if not already done
            initialize_components()
            
            # Process each file
            for uploaded_file in uploaded_files:
                process_uploaded_file(uploaded_file)
        
        # Display processing status
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        # Separator
        st.divider()
        
        # Display available files
        st.subheader("Available Documents")
        
        if "document_processor" in st.session_state and st.session_state.document_processor:
            loader = st.session_state.document_processor["loader"]
            files = loader.get_file_list()
            
            if files:
                for file in files:
                    st.write(f"📄 {file['name']}")
            else:
                st.write("No documents uploaded yet.")
                
        # Database stats
        if "vector_store" in st.session_state and st.session_state.vector_store:
            st.divider()
            st.subheader("Vector Database Stats")
            stats = st.session_state.vector_store.get_stats()
            st.write(f"Documents indexed: {stats['document_count']}")
        
        # Clear chat button
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                clear_chat_history()
        with col2:
            if st.button("Export Chat"):
                export_chat_history()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    # Metadata filters
    with col2:
        st.subheader("Filters")
        company_filter = None
        year_filter = None
        
        if "document_processor" in st.session_state and st.session_state.document_processor:
            filters = get_available_metadata_filters()
            
            if filters["companies"]:
                company_filter = st.selectbox(
                    "Company",
                    ["All"] + filters["companies"]
                )
                if company_filter == "All":
                    company_filter = None
            
            if filters["years"]:
                year_filter = st.selectbox(
                    "Year",
                    ["All"] + filters["years"]
                )
                if year_filter == "All":
                    year_filter = None
    
    # Chat interface
    with col1:
        st.subheader("Ask Questions About Fortune 500 Annual Reports")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        
                        # Display sources if available
                        if "sources" in message and message["sources"]:
                            with st.expander("Sources"):
                                for i, source in enumerate(message["sources"]):
                                    st.write(f"Source {i+1}:")
                                    for key, value in source.items():
                                        st.write(f"- {key}: {value}")
        
        # Question input
        user_question = st.chat_input("Ask a question about Fortune 500 annual reports")
        if user_question:
            # Initialize components if not already done
            initialize_components()
            
            # Process the question
            ask_question(user_question, company_filter, year_filter)
        
        # Initial instruction if no chat history
        if not st.session_state.chat_history:
            st.info("Upload Fortune 500 annual reports and ask questions about them. The AI will retrieve relevant information and provide answers based on the documents.")

if __name__ == "__main__":
    main()