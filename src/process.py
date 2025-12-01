"""
Processing module: compute distances and generate report.
"""

import argparse

from .report import generate_report


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

