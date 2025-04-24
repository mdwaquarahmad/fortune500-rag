"""
Prompt template module.

This module defines the prompt templates used for interacting with the LLM.
"""

from langchain.prompts import PromptTemplate
from ..config import SYSTEM_PROMPT

# Base system prompt for the LLM
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT

# Template for generating responses from retrieved context
QA_TEMPLATE = """
{system_prompt}

CONTEXT INFORMATION:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer the user's question based only on the provided context.
2. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer this question." and suggest what information might help.
3. Always cite the source of your information by referencing the company name and year from the metadata.
4. Provide a concise, accurate response that directly addresses the user's question.
5. Format your response in a clear, readable way.
6. If the question is about financial data, include specific numbers and percentages from the context when available.

YOUR ANSWER:
"""

QA_PROMPT = PromptTemplate(
    input_variables=["system_prompt", "context", "question"],
    template=QA_TEMPLATE
)

# Template for handling queries without relevant context
FALLBACK_TEMPLATE = """
{system_prompt}

USER QUESTION:
{question}

INSTRUCTIONS:
1. The user's question is about Fortune 500 company annual reports, but I don't have relevant context information to answer it specifically.
2. Provide a general response that acknowledges the limitations and suggests how the user might refine their question.
3. If appropriate, explain what types of information are typically found in annual reports that might be relevant to their query.

YOUR ANSWER:
"""

FALLBACK_PROMPT = PromptTemplate(
    input_variables=["system_prompt", "question"],
    template=FALLBACK_TEMPLATE
)

# Template for metadata filtering explanation
FILTER_EXPLANATION_TEMPLATE = """
{system_prompt}

USER QUESTION:
{question}

INSTRUCTIONS:
1. The user has applied the following filters to their search: {filters}
2. Explain how these filters were applied to narrow down the search results.
3. If the filters resulted in no relevant context being found, suggest how they might adjust their filters.

YOUR ANSWER:
"""

FILTER_EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["system_prompt", "question", "filters"],
    template=FILTER_EXPLANATION_TEMPLATE
)

def format_document_context(retrieved_docs):
    """
    Format a list of retrieved documents into a string context for the prompt.
    
    Args:
        retrieved_docs: List of retrieved document chunks with text and metadata
        
    Returns:
        str: Formatted context string
    """
    context_elements = []
    
    for i, doc in enumerate(retrieved_docs):
        # Extract text and metadata
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        # Format source information
        source_info = []
        if "company" in metadata:
            source_info.append(f"Company: {metadata['company']}")
        if "year" in metadata:
            source_info.append(f"Year: {metadata['year']}")
        if "source" in metadata:
            source_info.append(f"Document: {metadata['source']}")
        if "page" in metadata:
            source_info.append(f"Page: {metadata['page']}")
        if "section" in metadata:
            source_info.append(f"Section: {metadata['section']}")
            
        # Combine source information
        source_text = ", ".join(source_info)
        
        # Format document with source
        doc_text = f"[DOCUMENT {i+1}] {text}\n[SOURCE: {source_text}]\n"
        context_elements.append(doc_text)
    
    # Join all documents
    return "\n".join(context_elements)