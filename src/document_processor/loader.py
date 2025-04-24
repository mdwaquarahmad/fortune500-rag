"""
Document loader module.

This module handles loading of different document types and routes them
to the appropriate text extractor.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, BinaryIO, Tuple

from ..config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, SUPPORTED_FILE_TYPES

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Handles loading of documents from various sources and formats.
    """
    
    def __init__(self):
        """Initialize the document loader."""
        self.upload_dir = UPLOAD_DIR
        
    def save_uploaded_file(self, file: BinaryIO, filename: str) -> Path:
        """
        Save an uploaded file to the uploads directory.
        
        Args:
            file: The file object
            filename: The name of the file
            
        Returns:
            Path: The path to the saved file
        """
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file path
        file_path = self.upload_dir / filename
        
        # Write file
        with open(file_path, 'wb') as f:
            f.write(file.read())
        
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    
    def validate_file_type(self, filename: str) -> Tuple[bool, str]:
        """
        Validate if the file type is supported.
        
        Args:
            filename: The name of the file
            
        Returns:
            Tuple[bool, str]: (is_valid, file_type)
        """
        # Get file extension
        _, extension = os.path.splitext(filename.lower())
        
        # Check if extension is supported
        if extension not in SUPPORTED_EXTENSIONS:
            return False, ""
        
        # Determine file type
        file_type = next(
            (ftype for ftype, exts in SUPPORTED_FILE_TYPES.items() if extension in exts),
            ""
        )
        
        return True, file_type
    
    def extract_metadata(self, filename: str) -> Dict[str, str]:
        """
        Extract metadata from filename.
        This is a simple implementation - in a real system, you might use
        NER or other techniques to extract company names, dates, etc.
        
        Args:
            filename: The name of the file
            
        Returns:
            Dict: Metadata extracted from filename
        """
        # Strip extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Simple parsing - assumes filename format like "Company_Year"
        parts = name_without_ext.split('_')
        
        metadata = {
            "source": filename,
        }
        
        # Try to extract company and year if possible
        if len(parts) >= 1:
            metadata["company"] = parts[0].replace('-', ' ')
        
        if len(parts) >= 2 and parts[1].isdigit():
            metadata["year"] = parts[1]
            
        return metadata
    
    def get_file_list(self) -> List[Dict]:
        """
        Get list of all files in the upload directory.
        
        Returns:
            List[Dict]: List of file information including path, type, and metadata
        """
        files = []
        
        for file_path in self.upload_dir.glob('*'):
            if file_path.is_file():
                is_valid, file_type = self.validate_file_type(file_path.name)
                
                if is_valid:
                    metadata = self.extract_metadata(file_path.name)
                    files.append({
                        "path": str(file_path),
                        "name": file_path.name,
                        "type": file_type,
                        "metadata": metadata
                    })
                    
        return files
    
    def load_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Load a file and prepare it for text extraction.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict: File information including path, type, and metadata
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        is_valid, file_type = self.validate_file_type(file_path.name) 
        
        if not is_valid:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        metadata = self.extract_metadata(file_path.name)
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "type": file_type,
            "metadata": metadata
        }