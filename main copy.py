"""
Main entry point for the Fortune 500 RAG Chatbot application.

This script initializes and runs the Streamlit application.
"""
from dotenv import load_dotenv
load_dotenv()
import os            
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


import os
import logging
import streamlit as st
import subprocess
import sys
from pathlib import Path

from src.utils.helpers import setup_logger

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("Error: requirements.txt not found.")
            return False
            
        # Try to import key dependencies
        import streamlit
        import langchain
        import openai
        import chromadb
        import PyPDF2
        
        # Check for optional OCR dependency
        try:
            import pytesseract
        except ImportError:
            print("Warning: pytesseract not installed. OCR functionality will be limited.")
            
        return True
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please install all required dependencies with: pip install -r requirements.txt")
        return False

def check_environment():
    """
    Check if the environment is properly configured.
    
    Returns:
        bool: True if environment is properly configured, False otherwise
    """
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not found.")
        print("Please set your OpenAI API key in a .env file or environment variables.")
        return False
        
    # Check if necessary directories exist
    for directory in ["uploads", "chroma_db"]:
        os.makedirs(directory, exist_ok=True)
        
    return True

def run_app():
    """
    Run the Streamlit application.
    """
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "src" / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"Error: Application file not found at {app_path}")
        return False
        
    print(f"Starting Streamlit application from {app_path}")
    
    # Set the PYTHONPATH to include the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Run the Streamlit app using the streamlit module API
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", str(app_path), "--server.headless", "true"]
    sys.exit(stcli.main())

def main():
    """
    Main entry point for the application.
    """
    # Set up logging
    setup_logger()
    
    # Check dependencies and environment
    if not check_dependencies():
        return
        
    if not check_environment():
        return
        
    # Run the application
    run_app()

if __name__ == "__main__":
    main()