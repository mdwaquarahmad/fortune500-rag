"""
Configuration module for the Fortune 500 RAG Chatbot.

This module loads environment variables and provides configuration
settings for different components of the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", BASE_DIR / "chroma_db"))

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables or .env file")

# Model configurations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# OCR Configuration
TESSERACT_PATH = os.getenv("TESSERACT_PATH", None)
if TESSERACT_PATH:
    os.environ["TESSDATA_PREFIX"] = TESSERACT_PATH

# Vector DB settings
DISTANCE_METRIC = "cosine"
SEARCH_TOP_K = 5

# LLM settings
TEMPERATURE = 0.1
MAX_TOKENS = 1000
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context from Fortune 500 company annual reports.
Answer questions truthfully based only on the provided context. If the information is not in the context, say that you don't have enough information.
Always cite the source of your information by mentioning the company name and year."""

# File type configurations
SUPPORTED_FILE_TYPES = {
    "pdf": [".pdf"],
    "word": [".docx", ".doc"],
    "powerpoint": [".pptx", ".ppt"],
    "image": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
}

# Get a flat list of all supported extensions
SUPPORTED_EXTENSIONS = [ext for exts in SUPPORTED_FILE_TYPES.values() for ext in exts]

# Streamlit settings
STREAMLIT_TITLE = "Fortune 500 RAG Chatbot"
STREAMLIT_DESCRIPTION = "Ask questions about Fortune 500 company annual reports"