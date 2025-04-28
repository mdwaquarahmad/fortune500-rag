"""
Test data for evaluating the Fortune 500 RAG Chatbot.

This module contains a set of test questions and expected answers
for evaluating the performance of the RAG system.
"""

# Test questions with expected answers
# These questions should be based on the actual content of documents
# that have been processed by the system

TEST_QUESTIONS = [
    {
        "question": "What was Amazon's total revenue in 2023?",
        "expected_answer": "Amazon's total revenue in 2023 was $575 billion, which represents a 12% increase from the previous year ($514 billion in 2022).",
        "ground_truth_facts": [
            "Amazon's total revenue was $575 billion in 2023",
            "Revenue increased by 12% year-over-year",
            "Previous year (2022) revenue was $514 billion"
        ],
        "document_sources": ["Amazon-com-Inc-2023-Annual-Report.pdf"],
        "category": "financial_metrics"
    },
    {
        "question": "How did Amazon's North America segment perform in 2023?",
        "expected_answer": "Amazon's North America segment revenue grew by 12% year-over-year, increasing from $316 billion to $353 billion in 2023.",
        "ground_truth_facts": [
            "North America revenue was $353 billion in 2023",
            "North America revenue grew by 12% year-over-year",
            "Previous year revenue for North America was $316 billion"
        ],
        "document_sources": ["Amazon-com-Inc-2023-Annual-Report.pdf"],
        "category": "segment_performance"
    },
    {
        "question": "What was Amazon's operating income in 2023?",
        "expected_answer": "Amazon's operating income in 2023 was $36.9 billion (operating margin of 6.4%), which represents a 201% increase from $12.2 billion (operating margin of 2.4%) in 2022.",
        "ground_truth_facts": [
            "Operating income was $36.9 billion in 2023",
            "Operating margin was 6.4% in 2023",
            "Operating income increased by 201% year-over-year",
            "Previous year (2022) operating income was $12.2 billion",
            "Previous year operating margin was 2.4%"
        ],
        "document_sources": ["Amazon-com-Inc-2023-Annual-Report.pdf"],
        "category": "financial_metrics"
    },
    {
        "question": "How did AWS perform in 2023?",
        "expected_answer": "AWS revenue rose by 13% year-over-year, from $80 billion to $91 billion in 2023. The growth was primarily driven by increased customer usage, although it was partially offset by pricing changes due to long-term customer contracts.",
        "ground_truth_facts": [
            "AWS revenue was $91 billion in 2023",
            "AWS revenue grew by 13% year-over-year",
            "Previous year (2022) AWS revenue was $80 billion",
            "Growth was driven by increased customer usage",
            "Growth was partially offset by pricing changes due to long-term contracts"
        ],
        "document_sources": ["Amazon-com-Inc-2023-Annual-Report.pdf"],
        "category": "segment_performance"
    },
    {
        "question": "How did Amazon's International segment perform in 2023?",
        "expected_answer": "Amazon's International segment revenue grew by 11% year-over-year, increasing from $118 billion to $131 billion in 2023.",
        "ground_truth_facts": [
            "International revenue was $131 billion in 2023",
            "International revenue grew by 11% year-over-year",
            "Previous year (2022) International revenue was $118 billion"
        ],
        "document_sources": ["Amazon-com-Inc-2023-Annual-Report.pdf"],
        "category": "segment_performance"
    }
]
