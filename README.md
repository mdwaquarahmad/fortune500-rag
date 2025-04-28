# Fortune 500 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python that answers user questions based on the content of multiple Fortune 500 company annual reports. The system processes various document formats (PDF, DOCX, PPTX, and images via OCR) to provide accurate and contextual responses using OpenAI's GPT-4o.

## Table of Contents

- Overview
- System Architecture
- Features
- Installation
- Usage
- Components
- Evaluation Strategy
- Project Structure
- Technical Decisions
- Challenges and Solutions
- Future Improvements
- License
- Acknowledgments

## Overview

This RAG chatbot system combines document processing, vector embeddings, and large language model capabilities to create an intelligent document-based question answering system. The chatbot:

1. Ingests and processes documents in multiple formats (PDF, DOCX, PPTX, images)
2. Extracts, chunks, and vectorizes text content
3. Stores vector embeddings with metadata in a vector database
4. Retrieves relevant content when questions are asked
5. Generates contextual, accurate responses using OpenAI's GPT-4o
6. Presents information through an intuitive Streamlit interface

## System Architecture

### High-Level Design (HLD)

![System Architecture Diagram](docs/images/architecture_diagram.png)

### Component Breakdown

1. Document Processor
   - Handles ingestion of PDF, DOCX, PPTX, and image files
   - Extracts text using specialized extractors for each file type
   - Implements OCR for image-based documents
   - Chunks text appropriately for vector embedding
   - Extracts and applies metadata (company, year, document type)

2. Vector Database
   - Generates embeddings using OpenAI's embedding model
   - Stores vector embeddings in Chroma DB
   - Manages indexing for efficient retrieval
   - Implements similarity search functionality

3. LLM Integration
   - Constructs effective prompts using retrieved context
   - Integrates with OpenAI GPT-4o API
   - Implements fallback strategies for ambiguous queries
   - Formats and refines LLM responses

4. User Interface
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

- Multi-format Document Processing: Handles PDF, DOCX, PPTX, and images
- OCR Integration: Extracts text from images and scanned documents
- Intelligent Chunking: Splits documents into semantically meaningful chunks
- Metadata Tagging: Associates company name, year, and document type with chunks
- Vector Similarity Search: Retrieves the most relevant content for each query
- Contextual Response Generation: Uses GPT-4o to produce accurate, helpful answers
- Source Attribution: Cites the specific document sources used in responses
- Intuitive User Interface: Clean Streamlit interface for document upload and interaction
- Metadata Filtering: Allows users to filter queries by company or year
- Comprehensive Evaluation System: Built-in tools to measure accuracy, completeness, and latency

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key
- Tesseract OCR (for image processing)

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
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=openai_api_key
   ```

6. Create necessary directories:
   ```bash
   mkdir -p uploads chroma_db
   ```

## Usage

### Starting the Application

You can start the application using either of these methods:

1. Run using the main script:
   ```bash
   python main.py
   ```

2. Or run directly with Streamlit:
   ```bash
   streamlit run src/ui/app.py
   ```

The application will be available at `http://localhost:8501`.

### Uploading Documents

1. Use the file upload section in the sidebar to add documents (PDF, DOCX, PPTX, or images)
2. Drag and drop files or click "Browse files" to select from your file system
3. The system will process the documents and display progress in the sidebar
4. Once processed, documents will be available for querying
5. Uploaded documents are listed in the sidebar for reference

### Asking Questions

1. Type your question in the chat input field at the bottom of the main panel
2. Press Enter or click the send button to submit your question
3. The system will:
   - Search for relevant information in your documents
   - Generate a response based on the retrieved content
   - Display the answer with proper formatting for financial data
   - Include citations to the source documents

### Example Queries

- "What was Amazon's total revenue in 2023?"
- "How did Amazon's North America segment perform in 2023?"
- "What was Amazon's operating income in 2023?"
- "How did AWS perform in 2023?"
- "How did the International segment perform in 2023?"
- "What were the key achievements for Amazon in 2023?"
- "What is Amazon's strategy for AWS going forward?"

### Using the Evaluation System

The application includes a comprehensive evaluation system to measure performance:

1. Run a basic evaluation:
   ```bash
   python evaluation/run_evaluation.py
   ```

2. Generate a detailed evaluation report:
   ```bash
   python evaluation/run_evaluation.py --report
   ```

3. Evaluate specific questions:
   ```bash
   python evaluation/run_evaluation.py --single-question 0 --report
   ```

4. View evaluation results in the `evaluation/results` directory

## Components

### Document Processor

The document processor handles multiple file formats using specialized libraries:

- PDF: Uses PyPDF2 for text extraction, with fallback to OCR for scanned documents
- DOCX: Uses python-docx for structured document parsing, preserving paragraph and table structure
- PPTX: Uses python-pptx for presentation content extraction
- Images: Uses Tesseract OCR via pytesseract with preprocessing for optimal text recognition

Text chunking strategies:
- Primary approach: Paragraph-based chunking with overlap
- Fallback approaches: Sentence-level chunking or character-level chunking when needed
- Configurable chunk size and overlap via environment variables

### Vector Store

The vector database implementation:

- Embedding Model: OpenAI's text-embedding-3-small for efficient, high-quality embeddings
- Vector Database: Chroma DB for persistent storage and efficient retrieval
- Similarity Metric: Cosine similarity for semantic matching
- Retrieval Strategy: Returns top k most relevant chunks, with configurable k parameter
- Metadata Filtering: Support for filtering by company, year, and document type

### LLM Integration

LLM implementation details:

- Model: OpenAI GPT-4o for high-quality, contextual responses
- Prompt Engineering: Carefully designed prompts with:
  - System-level instructions for response format and style
  - Persona definition as a financial analyst
  - Retrieved context with source attribution
  - Query with any special instructions
- Context Management: Optimized for GPT-4o's context window
- Financial Data Formatting: Special handling for consistent financial notation

### User Interface

The Streamlit UI provides:

- Sidebar: For document management and upload
- Main Panel: For chat history and question answering
- File Upload: Drag-and-drop or browser-based file selection
- Chat Interface: Natural conversational interface for questions
- Processing Indicators: Status messages and progress information
- Error Handling: User-friendly error messages

## Evaluation Strategy

The system includes a comprehensive evaluation framework (in the `evaluation/` directory) that measures:

1. Latency: Timing for document retrieval and response generation
2. Accuracy: How well responses match expected answers from ground truth
3. Completeness: Coverage of key facts in generated responses

The evaluation system features:
- LLM-based evaluation: Using GPT-4o to assess response quality
- Visualization tools: For latency and quality metrics
- Detailed reports: Markdown reports with comprehensive analysis
- Customizable test sets: Support for custom test questions

Run evaluations with:
```bash
python evaluation/run_evaluation.py --report
```

## Project Structure

```
fortune500-rag/
├── docs/
│   ├── images/
│   │   └── architecture_diagram.png
│   └── hld.md
├── evaluation/
│   ├── results/
│   ├── custom_test_questions.json
│   ├── evaluator.py
│   ├── metrics.py
│   ├── run_evaluation.py
│   └── test_data.py
├── src/
│   ├── document_processor/
│   │   ├── chunker.py
│   │   ├── loader.py
│   │   ├── ocr.py
│   │   └── text_extractor.py
│   ├── llm/
│   │   ├── prompt_templates.py
│   │   └── response_generator.py
│   ├── ui/
│   │   └── app.py
│   ├── utils/
│   │   └── helpers.py
│   ├── vector_store/
│   │   ├── embeddings.py
│   │   └── store.py
│   └── config.py
├── tests/
│   ├── test_document_processor.py
│   ├── test_llm.py
│   └── test_vector_store.py
├── uploads/
├── chroma_db/
├── .env.example
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Technical Decisions

### Language and Framework Choices

- Python: Chosen for its rich ecosystem of libraries for NLP and document processing
- LangChain: Provides effective abstractions for RAG pipeline components
- Streamlit: Enables rapid UI development with Python, ideal for this application
- Chroma DB: Lightweight vector database with good performance and no external dependencies

### Embedding and LLM Selection

- OpenAI Embeddings: Selected for high-quality semantic search capabilities
- GPT-4o: Chosen for its superior reasoning capabilities and understanding of financial data

### Document Processing Strategy

- Chunking Strategy: Paragraph-based chunking balances context preservation with retrieval precision
- OCR Integration: Adds support for scanned documents and images, enhancing versatility

### Prompt Engineering Decisions

- Financial Focus: Prompts designed specifically for financial document analysis
- Source Attribution: Explicit instructions to cite sources enhance transparency
- Financial Notation Standardization: Consistent formatting of monetary values

## Challenges and Solutions

### Challenge 1: Handling Multiple Document Formats
Solution: Implemented modular document processors with specialized extractors for each format

### Challenge 2: Financial Data Formatting
Solution: Created custom prompt engineering and post-processing to ensure consistent notation

### Challenge 3: OCR Quality
Solution: Implemented preprocessing steps for images and fallback strategies for low-quality scans

### Challenge 4: Response Quality Evaluation
Solution: Developed a comprehensive evaluation framework with LLM-based assessment

## Future Improvements

Potential enhancements for future versions:

1. Advanced Metadata Extraction: Automatic company and date recognition using NER
2. Multi-Model Support: Option to switch between different LLMs (Claude, Llama, etc.)
3. Conversation Memory: Support for follow-up questions and conversation history
4. Advanced OCR: Improved image and table processing with layout analysis
5. Performance Optimization: Caching and parallel processing for faster document ingestion
6. Citation Enhancement: Direct linking to document sections in responses
7. User Authentication: Secure access controls for sensitive documents
8. Export Functionality: Options to export chat histories and response reports
9. Multilingual Support: Process and respond to questions in multiple languages

## License

MIT

## Acknowledgments

- This project uses OpenAI's GPT-4o for text generation
- Built with LangChain for RAG pipeline orchestration
- Utilizes Chroma DB for vector storage
- Uses Streamlit for the user interface and matplotlib for visualization
- Thanks to the maintainers of pypdf, python-docx, python-pptx, and pytesseract