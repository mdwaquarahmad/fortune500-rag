"""
Main entry point for the Fortune 500 RAG Chatbot application.

This script initializes and runs the Streamlit application. It handles environment setup,
dependency checking, and application startup.
"""

import os
import sys
import logging
from pathlib import Path
import importlib

# Set up environment
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Configure OpenAI API
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import utilities
from src.utils.helpers import setup_logger

# Constants
REQUIRED_DIRECTORIES = ["uploads", "chroma_db"]
REQUIRED_PACKAGES = [
    ("streamlit", "Streamlit UI framework"),
    ("langchain", "LangChain library"),
    ("openai", "OpenAI API client"),
    ("chromadb", "ChromaDB vector database"),
    ("pypdf", "PDF processing library")
]
OPTIONAL_PACKAGES = [
    ("pytesseract", "OCR processing")
]

def check_dependencies() -> bool:
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
        
        # Check required packages
        missing_packages = []
        for package_name, description in REQUIRED_PACKAGES:
            try:
                importlib.import_module(package_name)
            except ImportError:
                missing_packages.append(f"{package_name} ({description})")
        
        if missing_packages:
            print("Error: Missing required dependencies:")
            for pkg in missing_packages:
                print(f"  - {pkg}")
            print("Please install all required dependencies with: pip install -r requirements.txt")
            return False
        
        # Check optional packages
        for package_name, description in OPTIONAL_PACKAGES:
            try:
                importlib.import_module(package_name)
            except ImportError:
                print(f"Warning: Optional dependency {package_name} ({description}) not installed.")
                print(f"Some functionality may be limited.")
        
        return True
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False

def check_environment() -> bool:
    """
    Check if the environment is properly configured.
    
    Returns:
        bool: True if environment is properly configured, False otherwise
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not found or empty.")
        print("Please set your OpenAI API key in a .env file or environment variables.")
        return False
    
    # Check and create necessary directories
    for directory in REQUIRED_DIRECTORIES:
        try:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Confirmed directory exists: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    
    return True

def setup_python_path():
    """
    Ensure the current directory is in the Python path.
    This allows modules to be imported correctly.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"Added {current_dir} to Python path")

def run_app() -> bool:
    """
    Run the Streamlit application.
    
    Returns:
        bool: True if application started successfully, False otherwise
    """
    try:
        # Get the path to the Streamlit app
        app_path = Path(__file__).parent / "src" / "ui" / "app.py"
        
        if not app_path.exists():
            print(f"Error: Application file not found at {app_path}")
            return False
        
        print(f"Starting Streamlit application from {app_path}")
        
        # Ensure Python path is set up correctly
        setup_python_path()
        
        # Run the Streamlit app
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", str(app_path), "--server.headless", "true"]
        sys.exit(stcli.main())
    except Exception as e:
        print(f"Error starting application: {e}")
        return False

def main():
    """
    Main entry point for the application.
    
    Handles initialization, dependency checking, and application startup.
    """
    # Set up logging first
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Starting Fortune 500 RAG Chatbot")
    
    print("=" * 80)
    print("Fortune 500 RAG Chatbot")
    print("=" * 80)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        logger.error("Dependency check failed")
        return
    print("All required dependencies are installed.")
    
    # Check environment
    print("\nChecking environment...")
    if not check_environment():
        logger.error("Environment check failed")
        return
    print("Environment is properly configured.")
    
    # Run the application
    print("\nStarting application...")
    run_app()

if __name__ == "__main__":
    main()