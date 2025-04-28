# Fortune 500 RAG Chatbot - High-Level Design (HLD)

## 1. Introduction

This document outlines the high-level design of the Fortune 500 RAG Chatbot system. The system is designed to answer user questions based on the content of multiple Fortune 500 company annual reports in various formats (PDF, DOCX, PPTX, and images) using Retrieval-Augmented Generation (RAG).

### 1.1 System Goals

- Process multiple document formats (PDF, DOCX, PPTX, images)
- Extract, chunk, and vectorize text content
- Store and retrieve relevant information efficiently
- Generate accurate answers using the OpenAI GPT-4o model
- Provide a user-friendly interface for document uploading and querying
- Evaluate system performance, accuracy, and reliability

## 2. System Architecture

The system follows a modular architecture with several distinct components that work together to implement the RAG pattern. The architecture is designed to be extensible and maintainable.

### 2.1 Component Overview

```
+------------------+    +-----------------+    +----------------+    +----------------+
|                  |    |                 |    |                |    |                |
|  Document        |    |  Vector         |    |  LLM           |    |  User          |
|  Processor       |--->|  Store          |--->|  Integration   |--->|  Interface     |
|                  |    |                 |    |                |    |                |
+------------------+    +-----------------+    +----------------+    +----------------+
        |                       |                     |                     |
        v                       v                     v                     v
+------------------+    +-----------------+    +----------------+    +----------------+
|  - Load docs     |    |  - Generate     |    |  - Create      |    |  - Upload docs |
|  - Extract text  |    |    embeddings   |    |    prompts     |    |  - Ask         |
|  - Process OCR   |    |  - Store in     |    |  - Generate    |    |    questions   |
|  - Chunk text    |    |    Chroma DB    |    |    responses   |    |  - View        |
|  - Add metadata  |    |  - Retrieve     |    |  - Format      |    |    responses   |
|                  |    |    similar docs |    |    output      |    |                |
+------------------+    +-----------------+    +----------------+    +----------------+
                                                                            |
                                                                            v
                                                                     +----------------+
                                                                     |                |
                                                                     |  Evaluation    |
                                                                     |  System        |
                                                                     |                |
                                                                     +----------------+
                                                                     |  - Measure     |
                                                                     |    latency     |
                                                                     |  - Assess      |
                                                                     |    accuracy    |
                                                                     |  - Evaluate    |
                                                                     |    completeness|
                                                                     |  - Visualize   |
                                                                     |    performance |
                                                                     +----------------+
```

### 2.2 Data Flow

1. Document Ingestion:
   - User uploads documents through the UI
   - System saves documents to storage
   - Documents are loaded and validated

2. Text Extraction and Processing:
   - Text is extracted from various document formats
   - OCR is applied to images if needed
   - Text is cleaned and preprocessed
   - Text is divided into semantic chunks

3. Embedding and Indexing:
   - Text chunks are converted to vector embeddings
   - Embeddings are stored in the vector database
   - Metadata (company, year, etc.) is associated with each chunk

4. Query Processing:
   - User enters a question
   - Question is converted to a vector embedding
   - Similar document chunks are retrieved based on embedding similarity
   - Metadata filters are applied if specified

5. Response Generation:
   - Retrieved context is formatted
   - Prompt is constructed with context and question
   - LLM generates a response based on the prompt
   - Response is displayed to the user with source citations

6. System Evaluation:
   - Performance metrics are collected during operation
   - Test questions are evaluated against ground truth
   - LLM-based evaluations assess response quality
   - Visualizations and reports are generated

## 3. Component Design

### 3.1 Document Processor

#### 3.1.1 Document Loader
- Handles file uploads and storage
- Validates file types and formats
- Extracts basic metadata from filenames
- Manages the document repository

#### 3.1.2 Text Extractor
- Extracts text from PDF files using PyPDF2
- Extracts text from DOCX files using python-docx
- Extracts text from PPTX files using python-pptx
- Delegates to OCR module for image-based content

#### 3.1.3 OCR Module
- Processes images using Tesseract OCR
- Extracts text from scanned PDFs
- Handles image preprocessing for better OCR results

#### 3.1.4 Text Chunker
- Splits text into semantically meaningful chunks
- Implements various chunking strategies (paragraph, sentence, etc.)
- Maintains context across chunk boundaries
- Optimizes chunk size for embedding quality

### 3.2 Vector Store

#### 3.2.1 Embedding Generator
- Generates embeddings using OpenAI's embedding model
- Processes batches of text chunks efficiently
- Handles API rate limiting and retries
- Optimizes for cost and performance

#### 3.2.2 Vector Database
- Stores embeddings in Chroma DB
- Implements vector similarity search
- Manages metadata filtering
- Provides efficient retrieval of relevant chunks

### 3.3 LLM Integration

#### 3.3.1 Prompt Templates
- Defines templates for different query scenarios
- Formats retrieved context for optimal LLM input
- Implements system prompts and instructions
- Handles different response formats

#### 3.3.2 Response Generator
- Connects to OpenAI API with GPT-4o model
- Implements context window management
- Handles fallback strategies for edge cases
- Formats and post-processes LLM responses

### 3.4 User Interface

#### 3.4.1 Streamlit Application
- Provides document upload functionality
- Implements chat interface for questions
- Displays responses with source citations
- Offers metadata filtering options
- Shows processing status and progress indicators

### 3.5 Evaluation System

#### 3.5.1 Metrics Collection
- Measures latency for retrieval and generation
- Tracks accuracy against ground truth
- Evaluates completeness of responses
- Calculates overall performance metrics

#### 3.5.2 LLM-Based Evaluation
- Uses OpenAI's GPT-4o to evaluate response quality
- Assesses factual accuracy of responses
- Evaluates completeness of information
- Determines relevance to the original question

#### 3.5.3 Reporting and Visualization
- Generates comprehensive evaluation reports
- Creates visualizations of performance metrics
- Provides insights for system improvement
- Tracks performance changes over time

## 4. Technology Stack

### 4.1 Core Technologies

- Python 3.9+: Primary programming language
- LangChain: Framework for RAG pipeline
- OpenAI API: For embeddings and LLM capabilities
- Chroma DB: Vector database for storing embeddings
- Streamlit: Web interface framework

### 4.2 Document Processing

- PyPDF2: PDF text extraction
- python-docx: DOCX document parsing
- python-pptx: PowerPoint presentation parsing
- Tesseract OCR: Image-to-text conversion
- Pillow: Image processing

### 4.3 Storage and Retrieval

- Chroma DB: Persistent vector database
- OpenAI Embeddings: Text embedding model
- LangChain Retrievers: Similarity search and retrieval

### 4.4 Development and Testing

- pytest: Testing framework
- python-dotenv: Environment configuration
- tqdm: Progress indication

### 4.5 Evaluation and Visualization

- Matplotlib: Data visualization
- Pandas: Data analysis and manipulation
- OpenAI GPT-4o: LLM-based evaluation
- Custom metrics: Latency and accuracy calculations

## 5. Design Decisions

### 5.1 Document Chunking Strategy

We chose a paragraph-based chunking strategy with overlap to balance several factors:

1. Semantic Coherence: Paragraphs generally contain complete thoughts
2. Context Preservation: Overlap ensures context spans chunks
3. Vector Quality: Paragraphs provide enough context for meaningful embeddings
4. Retrieval Efficiency: Paragraph-sized chunks allow for precise retrieval

The chunker implements a multi-level approach:
- First attempts to split by paragraphs
- Falls back to sentence splitting if paragraphs are too large
- As a last resort, splits by character count

### 5.2 Vector Database Selection

Chroma DB was selected for the following reasons:

1. Ease of Integration: Well-supported by LangChain
2. Persistence: Supports disk-based storage
3. Metadata Filtering: Strong support for metadata queries
4. Performance: Good balance of speed and accuracy
5. Simplicity: No need for external databases or services

### 5.3 LLM Selection

OpenAI's GPT-4o was selected for response generation:

1. Quality: State-of-the-art reasoning and language capabilities
2. Context Length: Supports longer context windows
3. API Stability: Well-documented, reliable API
4. Cost-Effectiveness: Good balance of quality and cost

### 5.4 User Interface

Streamlit was chosen for the user interface:

1. Rapid Development: Fast implementation of interactive elements
2. Python Integration: Seamless integration with backend code
3. File Handling: Built-in support for file uploads
4. Responsive Design: Adapts to different screen sizes
5. Rich Components: Built-in support for chat interfaces

### 5.5 Evaluation Approach

Our evaluation system was designed based on these considerations:

1. Comprehensive Metrics: Measuring both technical performance (latency) and output quality
2. LLM-Based Evaluation: Using GPT-4o to assess quality aspects that are difficult to quantify programmatically
3. Visualization: Providing visual representations of system performance
4. Integration: Reusing the same components as the main application to ensure evaluation accuracy

## 6. Performance Considerations

### 6.1 Embedding Generation

- Batching document chunks for efficient API usage
- Caching embeddings to avoid regeneration
- Using appropriate embedding model for balance of quality and cost

### 6.2 Vector Search

- Optimizing chunk size for retrieval quality
- Tuning the number of retrieved chunks (k)
- Implementing metadata filters to narrow search space

### 6.3 Response Generation

- Optimizing prompt design for context utilization
- Managing LLM parameters (temperature, max tokens)
- Implementing caching for repeated queries

### 6.4 Evaluation Performance

- Selective use of LLM-based evaluation to manage API costs
- Efficient handling of financial notation in responses
- Optimized visualization generation for large result sets

## 7. Scalability

The current design can be scaled in several ways:

### 7.1 Vertical Scaling

- Processing larger documents
- Handling more complex queries
- Supporting more document formats

### 7.2 Horizontal Scaling

- Moving to distributed vector databases
- Implementing document processing queues
- Adding user authentication and multi-user support

## 8. Future Enhancements

### 8.1 Advanced Features

- Conversation memory for follow-up questions
- Multi-modal document understanding (tables, charts)
- Advanced metadata extraction using NER
- Cross-document reasoning and comparison

### 8.2 Performance Improvements

- Parallel document processing
- Hybrid search (keyword + vector)
- Embedding model fine-tuning
- Query optimization

### 8.3 Evaluation Enhancements

- Automated benchmark testing 
- Comparative evaluation across different LLM models
- User feedback integration into evaluation metrics
- Real-time performance monitoring dashboard

## 9. Conclusion

This high-level design provides a blueprint for a robust RAG chatbot system capable of processing, indexing, and retrieving information from Fortune 500 annual reports in various formats. The modular architecture allows for easy extension and maintenance, while the chosen technology stack provides a balance of functionality, performance, and development speed. The integrated evaluation system ensures ongoing quality assessment and continuous improvement of the system's capabilities.