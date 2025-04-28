#!/usr/bin/env python
"""
Run the evaluation for the Fortune 500 RAG Chatbot.

This script runs the evaluation process and generates a report
of the system's performance, accuracy, and completeness.
"""

import os
import sys
import logging
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# Ensure the project root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.evaluator import RAGEvaluator
from evaluation.test_data import TEST_QUESTIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run evaluation for the Fortune 500 RAG Chatbot')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation/results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--no-llm-eval',
        action='store_true',
        help='Disable LLM-based evaluation'
    )
    
    parser.add_argument(
        '--questions',
        type=str,
        help='Path to JSON file with custom test questions'
    )
    
    parser.add_argument(
        '--single-question',
        type=int,
        help='Index of a single question to evaluate (zero-based)'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate a detailed evaluation report'
    )
    
    return parser.parse_args()

def load_custom_questions(file_path):
    """Load custom test questions from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            questions = json.load(file)
        logger.info(f"Loaded {len(questions)} custom test questions")
        return questions
    except Exception as e:
        logger.error(f"Error loading custom questions: {e}")
        logger.info("Using default test questions instead")
        return TEST_QUESTIONS

def clean_text_for_report(text):
    """Clean text for display in reports."""
    if not text:
        return ""
        
    # Fix different dollar sign patterns
    # First handle the specific pattern "$number.$number"
    cleaned = re.sub(r'\$(\d+)\.?\$(\d+)', r'$\1.\2', text)
    
    # Handle multiple dollar signs
    cleaned = re.sub(r'\${2,}', '$', cleaned)
    
    # Handle "$number,$number"
    cleaned = re.sub(r'\$(\d+),\$(\d+)', r'$\1,\2', cleaned)
    
    # Handle any remaining patterns with adjacent dollar signs
    cleaned = re.sub(r'\$\s*\$', '$', cleaned)
    
    return cleaned

def generate_report(evaluation_results, output_dir):
    """Generate a human-readable evaluation report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"{timestamp}_evaluation_report.md"
    
    aggregate = evaluation_results["aggregate_metrics"]
    results = evaluation_results["individual_results"]
    
    with open(report_path, 'w') as file:
        # Write report header
        file.write("# Fortune 500 RAG Chatbot Evaluation Report\n\n")
        file.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Write summary section
        file.write("## Summary\n\n")
        file.write(f"- **Total Questions Evaluated**: {aggregate['total_questions']}\n")
        
        # Write LLM evaluation section if available
        if "llm_evaluation_averages" in aggregate and aggregate["llm_evaluation_averages"]:
            file.write("\n### LLM Evaluation Metrics\n\n")
            # Add latency to LLM metrics
            file.write(f"- **Average Latency**: {aggregate['average_latency_seconds']:.3f} seconds\n")
            
            llm_metrics = aggregate["llm_evaluation_averages"]
            
            for metric, value in llm_metrics.items():
                if (isinstance(value, (int, float)) and 
                    "error" not in metric.lower() and 
                    "conciseness" not in metric.lower()):
                    file.write(f"- **{metric.replace('_', ' ').title()}**: {value:.2f}/10\n")
        
        # Add visualizations section
        file.write("\n## Visualizations\n\n")
        file.write("Visualization images are available in the 'visualizations' directory.\n\n")
        
        # Write detailed results section
        file.write("\n## Detailed Results\n\n")
        
        for i, result in enumerate(results):
            file.write(f"### Question {i+1}: {result['question']}\n\n")
            
            # Expected vs. Actual
            file.write("**Expected Answer:**\n")
            file.write(f"```\n{result['expected_answer']}\n```\n\n")
            
            # Clean the response text for the report
            cleaned_response = clean_text_for_report(result['actual_response'])
            
            file.write("**Actual Response:**\n")
            file.write(f"```\n{cleaned_response}\n```\n\n")
            
            # Latency
            file.write("**Latency:**\n")
            file.write(f"- Retrieval: {result['latency']['retrieval_seconds']:.3f} seconds\n")
            file.write(f"- Generation: {result['latency']['generation_seconds']:.3f} seconds\n")
            file.write(f"- Total: {result['latency']['total_seconds']:.3f} seconds\n\n")
            
            # LLM evaluation if available
            if "llm_evaluation" in result:
                file.write("**LLM Evaluation:**\n")
                llm_eval = result['llm_evaluation']
                
                for metric, value in llm_eval.items():
                    if (isinstance(value, (int, float)) and 
                        "error" not in metric.lower() and 
                        "conciseness" not in metric.lower()):
                        file.write(f"- {metric.replace('_', ' ').title()}: {value}/10\n")
                
                # Add explanation if available
                if "explanation" in llm_eval:
                    file.write(f"\n*Explanation: {llm_eval['explanation']}*\n")
                    
                # Add errors if any
                if "factual_errors" in llm_eval and llm_eval["factual_errors"]:
                    file.write("\n**Factual Errors Identified:**\n")
                    for error in llm_eval["factual_errors"]:
                        file.write(f"- {error}\n")
                
                # Add missing information if any
                if "missing_information" in llm_eval and llm_eval["missing_information"]:
                    file.write("\n**Missing Information:**\n")
                    for missing in llm_eval["missing_information"]:
                        file.write(f"- {missing}\n")
                        
                file.write("\n")
            
            # Sources
            if result['sources']:
                file.write("**Sources:**\n")
                for j, source in enumerate(result['sources'][:3]):  # Limit to first 3 sources
                    file.write(f"- Source {j+1}: {source}\n")
                file.write("\n")
            
            file.write("---\n\n")
    
    logger.info(f"Evaluation report generated: {report_path}")
    return report_path

def main():
    """Run the evaluation and generate results."""
    args = parse_arguments()
    
    # Set up the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load custom questions if specified
    test_questions = TEST_QUESTIONS
    if args.questions:
        test_questions = load_custom_questions(args.questions)
    
    # Initialize the evaluator
    evaluator = RAGEvaluator(
        test_questions=test_questions,
        results_dir=output_dir,
        use_llm_evaluation=not args.no_llm_eval
    )
    
    # Run evaluation
    if args.single_question is not None:
        # Evaluate a single question
        if 0 <= args.single_question < len(test_questions):
            question_data = test_questions[args.single_question]
            logger.info(f"Evaluating single question: {question_data['question']}")
            
            result = evaluator.evaluate_question(question_data)
            evaluation_results = {
                "aggregate_metrics": {
                    "total_questions": 1,
                    "average_latency_seconds": result["latency"]["total_seconds"],
                    "average_fact_coverage_percentage": result["fact_coverage"]["coverage_percentage"]
                },
                "individual_results": [result]
            }
        else:
            logger.error(f"Invalid question index: {args.single_question}")
            sys.exit(1)
    else:
        # Evaluate all questions
        logger.info("Running evaluation on all test questions...")
        evaluation_results = evaluator.evaluate_all()
    
    # Generate report if requested
    if args.report:
        report_path = generate_report(evaluation_results, output_dir)
        logger.info(f"Detailed evaluation report saved to: {report_path}")
    
    # Print summary to console
    aggregate = evaluation_results["aggregate_metrics"]
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions: {aggregate['total_questions']}")
    
    if "llm_evaluation_averages" in aggregate and aggregate["llm_evaluation_averages"]:
        print("\nLLM Evaluation Metrics:")
        # Print latency here under LLM metrics
        print(f"- Average Latency: {aggregate['average_latency_seconds']:.3f} seconds")
        
        # Filter out conciseness score
        for metric, value in aggregate["llm_evaluation_averages"].items():
            if (isinstance(value, (int, float)) and 
                "error" not in metric.lower() and 
                "conciseness" not in metric.lower()):
                print(f"- {metric.replace('_', ' ').title()}: {value:.2f}/10")
    
    print("\nEvaluation results saved to:", output_dir)
    print("="*60)

if __name__ == "__main__":
    main()