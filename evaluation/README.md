# RAG Chatbot Evaluation System

This module provides tools for evaluating the performance, accuracy, and completeness of the Fortune 500 RAG Chatbot system. It implements a comprehensive evaluation strategy that measures various metrics and generates detailed reports.

## Features

- Latency Measurement: Tracks processing time for retrieval and response generation
- Accuracy Evaluation: Measures how well the system answers questions compared to ground truth
- Completeness Assessment: Evaluates what percentage of key facts are included in responses
- LLM-Based Evaluation: Uses an LLM to assess response quality on multiple dimensions
- Visualization Generation: Creates charts and graphs of performance metrics
- Detailed Reports: Generates comprehensive evaluation reports

## Directory Structure

- `evaluator.py`: Main evaluation logic and metrics collection
- `metrics.py`: Implementation of specific evaluation metrics
- `test_data.py`: Test questions with expected answers
- `run_evaluation.py`: Command-line script to run evaluations
- `results/`: Directory where evaluation results are stored

## Usage

### Basic Usage

To run a complete evaluation using the default test questions:

```bash
python evaluation/run_evaluation.py
```


## Understanding Evaluation Results

The evaluation system generates several types of metrics:

1. Latency Metrics: Time taken for retrieval and generation
2. Fact Coverage: Percentage of ground truth facts included in the answer
3. LLM Evaluation Scores (if enabled):
   - Factual Accuracy: Correctness of factual claims (0-10)
   - Completeness: Coverage of key information (0-10)
   - Relevance: How directly the answer addresses the question (0-10)
   - Overall Score: Average of the above metrics (0-10)

Results are saved as JSON files in the output directory and include both individual and aggregate metrics.

## Visualizations

The evaluation system generates several visualizations:

1. Latency Breakdown: Bar chart showing retrieval vs. generation time with average latency line
2. LLM Evaluation: Bar chart showing LLM evaluation scores by question

These visualizations are saved in the `results/visualizations/` directory.

## Financial Notation Handling

The evaluation system includes robust handling of financial notation, ensuring that:

- Dollar amounts are properly formatted in reports
- Financial figures with dollar signs, commas, and decimals are correctly displayed
- Alternative notations ($B for billions, $M for millions) are properly normalized for comparison

## Integration with RAG System

The evaluation system directly integrates with the main RAG chatbot components:

- Uses the same document processor for file handling
- Connects to the same vector store for retrieval
- Employs the same response generator used in the main application

This ensures that the evaluation results accurately reflect the actual performance of the production system.

## Performance Considerations

For optimal performance when running evaluations:

1. Ensure sufficient API quota is available if using LLM evaluation
2. Pre-process and index documents before running evaluations
3. For large-scale evaluations, consider batching questions or using the `--single-question` option

## Interpreting Results

When analyzing evaluation results, consider:

- Latency: Lower is better, under 5 seconds is considered good for a complete RAG pipeline
- Factual Accuracy: Scores above 8 indicate high-quality responses
- Completeness: Scores above 8 indicate comprehensive answers
- Relevance: Scores above 8 indicate highly targeted responses
- Overall Score: Scores above 8 indicate excellent overall performance