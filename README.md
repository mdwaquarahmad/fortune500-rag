# Fortune 500 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python that answers questions based on Fortune 500 company annual reports. The system processes multiple document formats, including PDF, DOCX, PPTX, and images (via OCR), to provide accurate and contextual responses using OpenAI's GPT-4o.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)

## Overview

This RAG chatbot system combines document processing, vector embeddings, and large language model capabilities to create an intelligent document-based question answering system. The chatbot:

1. Ingests and processes documents in multiple formats
2. Extracts, chunks, and vectorizes text content
3. Stores vector embeddings with metadata in a vector database
4. Retrieves relevant content when questions are asked
5. Generates contextual, accurate responses using OpenAI's GPT-4o
6. Presents information through an intuitive Streamlit interface

## System Architecture

### High-Level Design (HLD)

![System Architecture Diagram](docs/images/architecture_diagram.png)

### Component Breakdown

1. **Document Processor**
   - Handles ingestion of PDF, DOCX, PPTX, and image files
   - Extracts text using specialized extractors for each file type
   - Implements OCR for image-based documents
   - Chunks text appropriately for vector embedding
   - Extracts and applies metadata (company, year, document type)

2. **Vector Database**
   - Generates embeddings using OpenAI's embedding model
   - Stores vector embeddings in Chroma DB
   - Manages indexing for efficient retrieval
   - Implements similarity search functionality

3. **LLM Integration**
   - Constructs effective prompts using retrieved context
   - Integrates with OpenAI GPT-4o API
   - Implements fallback strategies for ambiguous queries
   - Formats and refines LLM responses

4. **User Interface**
   - Provides document upload capabilities for multiple file types
   - Implements chat interface for question input and response display
   - Offers metadata filtering options
   - Displays processing status and progress indicators

### Data Flow

1. User uploads documents through the Streamlit interface
2. System processes and chunks documents by paragraph
3. Text chunks are converted to vector embeddings
4. Embeddings are stored in Chroma DB with metadata
5. User asks a question through the interface
6. Question is embedded and relevant chunks are retrieved
7. Retrieved context is used to construct a prompt for GPT-4o
8. LLM generates a response based on the provided context
9. Response is displayed to the user with source attributions

## Features

- **Multi-format Document Processing**: Handles PDF, DOCX, PPTX, and images
- **OCR Integration**: Extracts text from images and scanned documents
- **Intelligent Chunking**: Splits documents into semantically meaningful chunks
- **Metadata Tagging**: Associates company name, year, and document type with chunks
- **Vector Similarity Search**: Retrieves the most relevant content for each query
- **Contextual Response Generation**: Uses GPT-4o to produce accurate, helpful answers
- **Source Attribution**: Cites the specific document sources used in responses
- **Intuitive User Interface**: Clean Streamlit interface for document upload and interaction
- **Metadata Filtering**: Allows users to filter queries by company or year

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fortune500-rag.git
   cd fortune500-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR (for image processing):
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Starting the Application

Run the Streamlit application:
```bash
streamlit run src/ui/app.py
```

The application will be available at `http://localhost:8501`.

### Uploading Documents

1. Use the file upload section to add documents (PDF, DOCX, PPTX, or images)
2. The system will process the documents and display progress
3. Once processed, documents will be available for querying

### Asking Questions

1. Type your question in the input field
2. Optionally, use metadata filters to narrow the search scope
3. Submit your question and view the generated response
4. The response will include citations to the source documents

### Example Queries

- "What were Amazon's total revenue and net income in 2023?"
- "How does Microsoft describe its cloud strategy in recent reports?"
- "Compare the R&D investments of Apple and Google in the last fiscal year."
- "What are the major risks mentioned in Tesla's annual report?"

## Components

### Document Processor

The document processor handles multiple file formats:

- **PDF**: Using PyPDF2 for text extraction
- **DOCX**: Using python-docx for structured document parsing
- **PPTX**: Using python-pptx for presentation content extraction
- **Images**: Using Tesseract OCR via pytesseract

Text is chunked using a paragraph-based strategy with overlap to maintain context across chunk boundaries.

### Vector Store

The system uses:

- **Embeddings**: OpenAI's text-embedding model
- **Vector Database**: Chroma DB for storing and retrieving embeddings
- **Similarity Metric**: Cosine similarity for matching queries to content

### LLM Integration

- **Model**: OpenAI GPT-4o
- **Prompt Engineering**: Carefully designed prompts that include:
  - Task description
  - Retrieved context
  - Query
  - Response format instructions
- **Fallback Strategy**: Graceful handling of queries without relevant context

### User Interface

Built with Streamlit, the UI provides:

- File upload with drag-and-drop support
- Chat-style interface for questions and answers
- Document processing status indicators
- Metadata filter controls
- Source attribution display

## Evaluation

The system has been designed with the following evaluation criteria in mind:

### Code Quality
- Clean, modular architecture with separation of concerns
- Consistent code style and comprehensive documentation
- Error handling and edge case management

### RAG Architecture
- Effective document chunking strategy
- Optimized embedding generation
- Accurate retrieval flow with relevance ranking

### Use of LLM
- Well-engineered prompts with appropriate context injection
- Fallback handling for queries without relevant context
- Response formatting for readability

### End-to-End Functionality
- Seamless document ingestion and processing
- Fast and accurate question answering
- Intuitive user experience

### Error Handling
- Graceful handling of malformed documents
- Support for empty queries and edge cases
- Clear error messaging for users

### Tech Stack
- Appropriate use of LangChain, Chroma, and OpenAI
- Integration of specialized libraries for document processing
- Efficient code with minimal dependencies

### Interface
- Clean, intuitive Streamlit UI
- Responsive design with progress indicators
- Support for various question types

### Testing
- Unit tests for core components
- Integration tests for end-to-end functionality
- Example query test suite

## Future Improvements

Potential enhancements for future versions:

1. **Advanced Metadata Extraction**: Automatic company and date recognition
2. **Multi-Model Support**: Option to switch between different LLMs
3. **Conversation Memory**: Support for follow-up questions and conversation history
4. **Advanced OCR**: Improved image and table processing
5. **Performance Optimization**: Caching and parallel processing for faster document ingestion
6. **Citation Enhancement**: Direct linking to document sections in responses
7. **User Authentication**: Secure access controls for sensitive documents
8. **Export Functionality**: Options to export chat histories and response reports
9. **Multilingual Support**: Process and respond to questions in multiple languages

## License

MIT

## Acknowledgments

- This project uses OpenAI's GPT-4o for text generation
- Built with LangChain for RAG pipeline orchestration
- Utilizes Chroma DB for vector storage