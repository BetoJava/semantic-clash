"""
Processing module: compute distances and generate report.
"""

import argparse

from .report import generate_report, generate_inter_h2_report


def main():
    """CLI entry point for report generation."""
    parser = argparse.ArgumentParser(description="Generate similarity report from vector database")
    parser.add_argument("--input", required=True, help="Path to vector database (base path, without extension)")
    parser.add_argument("--output", required=True, help="Path to output CSV report file")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top overlaps to include (default: 100)")
    parser.add_argument("--min-similarity", type=float, default=0.0, help="Minimum similarity score to include (default: 0.0)")
    
    args = parser.parse_args()
    
    try:
        generate_report(args.input, args.output, args.top_k, args.min_similarity)
        print(f"\nSuccess! Report generated at: {args.output}")
    except Exception as e:
        print(f"\nError during report generation: {e}")
        raise


def main_inter_h2():
    """CLI entry point for inter-H2 report generation."""
    parser = argparse.ArgumentParser(description="Generate inter-H2 similarity reports from vector database")
    parser.add_argument("--input", required=True, help="Path to vector database (base path, without extension)")
    parser.add_argument("--output-dir", required=True, help="Directory to save output reports")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top pairs per H2 to include (default: 100)")
    parser.add_argument("--min-similarity", type=float, default=0.0, help="Minimum similarity score to include (default: 0.0)")
    parser.add_argument("--exclude-threshold", type=float, default=0.999, help="Exclude pairs with similarity > this threshold (default: 0.999)")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Threshold for redundancy score calculation (default: 0.5)")
    parser.add_argument("--score-exponent", type=float, default=2.0, help="Exponent for redundancy score calculation (default: 2.0)")
    
    args = parser.parse_args()
    
    try:
        generate_inter_h2_report(
            args.input,
            args.output_dir,
            args.top_k,
            args.min_similarity,
            args.exclude_threshold,
            args.score_threshold,
            args.score_exponent
        )
        print(f"\nSuccess! Reports generated in: {args.output_dir}")
    except Exception as e:
        print(f"\nError during inter-H2 report generation: {e}")
        raise

