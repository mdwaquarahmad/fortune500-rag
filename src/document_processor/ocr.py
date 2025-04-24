"""
OCR module for extracting text from images.

This module uses Tesseract OCR to extract text from various image formats.
"""

import logging
import os
from typing import Dict, List

import pytesseract
from PIL import Image

from ..config import TESSERACT_PATH

logger = logging.getLogger(__name__)

# Configure Tesseract path if provided
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_from_image(file_path: str, metadata: Dict) -> List[Dict]:
    """
    Extract text from an image file using Tesseract OCR.
    
    Args:
        file_path: Path to the image file
        metadata: Document metadata
        
    Returns:
        List[Dict]: List of extracted text blocks with metadata
    """
    text_blocks = []
    
    try:
        # Open image with PIL
        image = Image.open(file_path)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(image)
        
        # Clean and process text
        text = text.strip()
        
        if text:
            text_blocks.append({
                "text": text,
                "metadata": {
                    **metadata,
                    "processed_with": "tesseract_ocr"
                }
            })
            
        logger.info(f"Successfully extracted {len(text)} characters from image {file_path}")
    except Exception as e:
        logger.error(f"Error extracting text from image {file_path}: {e}")
        raise
        
    return text_blocks

def extract_text_from_pdf_with_ocr(file_path: str, metadata: Dict, pages=None) -> List[Dict]:
    """
    Extract text from a PDF file using OCR.
    This is useful for scanned PDFs where normal text extraction fails.
    
    Args:
        file_path: Path to the PDF file
        metadata: Document metadata
        pages: List of page numbers to process (None = all pages)
        
    Returns:
        List[Dict]: List of extracted text blocks with metadata
    """
    try:
        # Check if pdf2image is available
        from pdf2image import convert_from_path
    except ImportError:
        logger.error("pdf2image library not installed. Required for PDF OCR processing.")
        raise ImportError("pdf2image library required for PDF OCR processing. Install with 'pip install pdf2image'")
    
    text_blocks = []
    
    try:
        # Convert PDF pages to images
        images = convert_from_path(file_path, dpi=300, first_page=pages[0] if pages else None, 
                                 last_page=pages[-1] if pages else None)
        
        # Process each page
        for i, image in enumerate(images):
            page_num = pages[i] if pages else i + 1
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image)
            text = text.strip()
            
            if text:
                text_blocks.append({
                    "text": text,
                    "metadata": {
                        **metadata,
                        "page": page_num,
                        "processed_with": "tesseract_ocr"
                    }
                })
                
        logger.info(f"Successfully extracted text from {len(images)} PDF pages using OCR")
    except Exception as e:
        logger.error(f"Error extracting text from PDF with OCR {file_path}: {e}")
        raise
        
    return text_blocks