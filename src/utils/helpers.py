"""
Helper utilities for the Fortune 500 RAG Chatbot.

This module provides common utility functions used across the application.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

def setup_logger(log_level=logging.INFO):
    """
    Set up and configure logging.
    
    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def extract_company_from_filename(filename: str) -> Optional[str]:
    """
    Extract company name from filename using pattern matching.
    
    Args:
        filename: Name of the file
        
    Returns:
        Optional[str]: Extracted company name or None
    """
    # Remove file extension
    base_name = os.path.splitext(filename)[0]
    
    # Try different patterns
    
    # Pattern 1: CompanyName_Year
    match = re.match(r'^([A-Za-z0-9\s]+)_(\d{4}).*$', base_name)
    if match:
        return match.group(1).replace('_', ' ').strip()
    
    # Pattern 2: CompanyName-Year
    match = re.match(r'^([A-Za-z0-9\s]+)-(\d{4}).*$', base_name)
    if match:
        return match.group(1).replace('-', ' ').strip()
    
    # Pattern 3: AnnualReport_CompanyName_Year
    match = re.match(r'^AnnualReport[_-]([A-Za-z0-9\s]+)[_-](\d{4}).*$', base_name)
    if match:
        return match.group(1).replace('_', ' ').strip()
    
    # No match found
    return None

def extract_year_from_filename(filename: str) -> Optional[str]:
    """
    Extract year from filename using pattern matching.
    
    Args:
        filename: Name of the file
        
    Returns:
        Optional[str]: Extracted year or None
    """
    # Look for 4-digit year pattern
    match = re.search(r'(\d{4})', filename)
    if match:
        return match.group(1)
    
    return None

def sanitize_text(text: str) -> str:
    """
    Clean and sanitize text for better processing.
    
    Args:
        text: Input text string
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'[\r\n]+', '\n', text)
    
    return text.strip()

def format_metadata_for_display(metadata: Dict) -> str:
    """
    Format metadata dictionary into a readable string.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        str: Formatted metadata string
    """
    if not metadata:
        return "No metadata available"
    
    formatted_parts = []
    
    # Order keys for consistent display
    important_keys = ["company", "year", "source", "page", "section"]
    
    # Add important keys first if they exist
    for key in important_keys:
        if key in metadata:
            formatted_parts.append(f"{key.capitalize()}: {metadata[key]}")
    
    # Add any remaining keys
    for key, value in metadata.items():
        if key not in important_keys:
            formatted_parts.append(f"{key.capitalize()}: {value}")
    
    return " | ".join(formatted_parts)

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List[List[Any]]: List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def measure_processing_time(func):
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Function wrapper
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

def save_json(data: Any, file_path: str):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON data: {e}")
        raise

def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Any: Loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        raise