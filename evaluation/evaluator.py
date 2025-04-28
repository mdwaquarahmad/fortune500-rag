"""
Evaluator module for the Fortune 500 RAG Chatbot.

This module provides the main evaluation logic for the RAG system,
combining various metrics to assess performance, accuracy, and completeness.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import re

from src.document_processor.loader import DocumentLoader
from src.vector_store.embeddings import EmbeddingGenerator
from src.vector_store.store import VectorStore
from src.llm.response_generator import ResponseGenerator
from evaluation.metrics import (
    calculate_latency,
    calculate_fact_coverage,
    evaluate_with_llm_with_retry,
    calculate_aggregate_metrics,
    clean_text_for_comparison
)
from evaluation.test_data import TEST_QUESTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluates the RAG chatbot system."""
    
    def __init__(self, 
                 test_questions: List[Dict] = None,
                 results_dir: str = "evaluation/results",
                 use_llm_evaluation: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            test_questions: List of test questions with expected answers
            results_dir: Directory to save evaluation results
            use_llm_evaluation: Whether to use LLM-based evaluation
        """
        self.test_questions = test_questions or TEST_QUESTIONS
        self.results_dir = Path(results_dir)
        self.use_llm_evaluation = use_llm_evaluation
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize system components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize the RAG system components for evaluation."""
        logger.info("Initializing RAG system components...")
        
        # Document processor components
        self.loader = DocumentLoader()
        
        # Vector store components
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(self.embedding_generator)
        
        # Response generator
        self.response_generator = ResponseGenerator()
        
        logger.info("Components initialized successfully.")
    
    def evaluate_question(self, 
                          question_data: Dict,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a single question.
        
        Args:
            question_data: Question data including the question and expected answer
            save_results: Whether to save results to file
            
        Returns:
            Dict: Evaluation results
        """
        question = question_data["question"]
        expected_answer = question_data.get("expected_answer", "")
        ground_truth_facts = question_data.get("ground_truth_facts", [])
        category = question_data.get("category", "general")
        
        logger.info(f"Evaluating question: {question}")
        
        # Step 1: Measure retrieval latency and get search results
        retrieval_metrics = calculate_latency(
            self.vector_store.search,
            question
        )
        search_results = retrieval_metrics["result"]
        retrieval_latency = retrieval_metrics["latency_seconds"]
        
        # Step 2: Measure response generation latency
        generation_metrics = calculate_latency(
            self.response_generator.generate_response,
            question, search_results
        )
        result = generation_metrics["result"]
        generation_latency = generation_metrics["latency_seconds"]
        
        # Extract response and source information
        response = result["response"]
        sources = result.get("sources", [])
        
        # Step 3: Calculate fact coverage
        fact_coverage = calculate_fact_coverage(response, ground_truth_facts)
        
        # Step 4: Use LLM to evaluate response quality (if enabled)
        llm_evaluation = None
        if self.use_llm_evaluation and expected_answer:
            llm_evaluation = evaluate_with_llm_with_retry(
                response=response,
                question=question,
                expected_answer=expected_answer
            )
        
        # Compile evaluation results
        evaluation_result = {
            "question": question,
            "expected_answer": expected_answer,
            "actual_response": response,
            "latency": {
                "retrieval_seconds": retrieval_latency,
                "generation_seconds": generation_latency,
                "total_seconds": retrieval_latency + generation_latency
            },
            "fact_coverage": fact_coverage,
            "sources": sources,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        
        if llm_evaluation:
            evaluation_result["llm_evaluation"] = llm_evaluation
        
        # Save results if requested
        if save_results:
            self._save_result(evaluation_result)
        
        return evaluation_result
    
    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate the RAG system on all test questions.
        
        Returns:
            Dict: Aggregated evaluation results
        """
        logger.info(f"Starting evaluation on {len(self.test_questions)} test questions...")
        
        results = []
        for question_data in self.test_questions:
            result = self.evaluate_question(question_data)
            results.append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = calculate_aggregate_metrics(results)
        
        # Save aggregate results
        self._save_aggregate_results(aggregate_metrics, results)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        logger.info("Evaluation completed successfully.")
        return {
            "aggregate_metrics": aggregate_metrics,
            "individual_results": results
        }
    
    def _save_result(self, result: Dict):
        """Save an individual evaluation result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_id = result["question"][:20].replace(" ", "_").lower()
        
        filename = f"{timestamp}_{question_id}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as file:
            json.dump(result, file, indent=2)
    
    def _clean_response_text(self, text: str) -> str:
        """Clean response text for display in reports."""
        # Replace multiple dollar signs with a single one
        cleaned = re.sub(r'\$+\.?\$+', '$', text)  # Match patterns like "$$.$"
        cleaned = re.sub(r'\${2,}', '$', cleaned)  # Match sequences of 2+ dollar signs
        
        # Fix comma formatting in financial numbers
        cleaned = re.sub(r'\$(\d+),(\d+)', r'$\1,\2', cleaned)
        
        return cleaned
    
    def _save_aggregate_results(self, 
                               aggregate_metrics: Dict, 
                               individual_results: List[Dict]):
        """Save aggregate evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"{timestamp}_aggregate_results.json"
        
        # Create the full result object
        full_results = {
            "aggregate_metrics": aggregate_metrics,
            "individual_results": individual_results,
            "timestamp": datetime.now().isoformat(),
            "test_count": len(individual_results)
        }
        
        with open(filepath, 'w') as file:
            json.dump(full_results, file, indent=2)
        
        logger.info(f"Saved aggregate results to {filepath}")
    
    def _generate_visualizations(self, results: List[Dict]):
        """Generate visualizations of evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a directory for visualizations
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Latency Visualization
        self._plot_latency(results, viz_dir / f"{timestamp}_latency.png")
        
        # LLM Evaluation Visualization (if available)
        if all("llm_evaluation" in result for result in results):
            self._plot_llm_evaluation(results, viz_dir / f"{timestamp}_llm_evaluation.png")
        
        logger.info(f"Generated visualizations in {viz_dir}")
    
    def _plot_latency(self, results: List[Dict], filepath: Path):
        """Plot latency metrics."""
        retrieval_times = [r["latency"]["retrieval_seconds"] for r in results]
        generation_times = [r["latency"]["generation_seconds"] for r in results]
        total_times = [r["latency"]["total_seconds"] for r in results]
        
        question_labels = [f"Q{i+1}" for i in range(len(results))]
        
        plt.figure(figsize=(12, 6))
        
        # Create the stacked bar chart
        plt.bar(question_labels, retrieval_times, label='Retrieval')
        plt.bar(question_labels, generation_times, bottom=retrieval_times, label='Generation')
        
        plt.xlabel('Questions')
        plt.ylabel('Time (seconds)')
        plt.title('RAG System Latency Breakdown')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Add average line with text annotation
        avg_total = sum(total_times) / len(total_times)
        plt.axhline(y=avg_total, color='r', linestyle='--', label=f'Avg: {avg_total:.2f}s')
        plt.text(len(question_labels)-0.5, avg_total+0.1, f'Average: {avg_total:.2f}s', 
                color='r', ha='right', va='bottom')
        
        # Add the legend again to include the average line
        plt.legend()
        
        # Save the figure
        plt.savefig(filepath, dpi=300)
        plt.close()
    
    def _plot_llm_evaluation(self, results: List[Dict], filepath: Path):
        """Plot LLM evaluation metrics."""
        # Extract metrics
        questions = [f"Q{i+1}" for i in range(len(results))]
        
        metrics = {
            'Factual Accuracy': [],
            'Completeness': [],
            'Relevance': [],
            # Removed conciseness score
            'Overall': []
        }
        
        for result in results:
            if "llm_evaluation" in result:
                eval_data = result["llm_evaluation"]
                metrics['Factual Accuracy'].append(eval_data.get('factual_accuracy_score', 0))
                metrics['Completeness'].append(eval_data.get('completeness_score', 0))
                metrics['Relevance'].append(eval_data.get('relevance_score', 0))
                # Removed conciseness score
                metrics['Overall'].append(eval_data.get('overall_score', 0))
        
        # Create a DataFrame
        df = pd.DataFrame(metrics, index=questions)
        
        # Plot
        plt.figure(figsize=(12, 8))
        df.plot(kind='bar', ax=plt.gca())
        plt.xlabel('Questions')
        plt.ylabel('Score (0-10)')
        plt.title('LLM Evaluation Scores by Question')
        
        # Use horizontal legend above the plot to avoid overlap
        plt.legend(title='Metrics', loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                ncol=5, fancybox=True, shadow=True)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(filepath, dpi=300)
        plt.close()