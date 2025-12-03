"""
Main entry point for Semantic Clash application.
"""

import argparse
import sys
from pathlib import Path

from src.preprocess import preprocess
from src.report import generate_report, generate_inter_h2_report


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic Clash: Detect semantic overlaps in DOCX documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess a DOCX file
  python main.py --mode preprocess --input document.docx --output data/
  
  # Generate similarity report
  python main.py --mode process --input data/document_vector_db --output data/report.txt
  
  # Generate inter-H2 similarity reports
  python main.py --mode process-inter-h2 --input data/document_vector_db --output data/inter_h2_reports/
        """
    )
    
    parser.add_argument(
        "--mode",
        required=True,
        choices=["preprocess", "process", "process-inter-h2"],
        help="Processing mode: 'preprocess' for DOCX parsing/embedding, 'process' for report generation, 'process-inter-h2' for inter-H2 report generation"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input path: DOCX file (preprocess mode) or vector DB base path (process mode)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output path: directory (preprocess/process-inter-h2 mode) or CSV file path (process mode)"
    )
    
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model name (preprocess mode only, default: Qwen/Qwen3-Embedding-0.6B)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top overlaps to include in report (process/process-inter-h2 mode, default: 100)"
    )
    
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum similarity score to include (process/process-inter-h2 mode, default: 0.0)"
    )
    
    parser.add_argument(
        "--exclude-threshold",
        type=float,
        default=0.999,
        help="Exclude pairs with similarity > this threshold (process-inter-h2 mode only, default: 0.999)"
    )
    
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Threshold for redundancy score calculation (process-inter-h2 mode only, default: 0.5)"
    )
    
    parser.add_argument(
        "--score-exponent",
        type=float,
        default=2.0,
        help="Exponent for redundancy score calculation (process-inter-h2 mode only, default: 2.0)"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for embedding (preprocess mode only, default: cpu)"
    )
    
    parser.add_argument(
        "--chunking",
        default="naive",
        choices=["naive", "linebreak"],
        help="Chunking strategy: 'naive' (by word count) or 'linebreak' (by newlines, preprocess mode only, default: naive)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "preprocess":
            print("=" * 60)
            print("PREPROCESS MODE")
            print("=" * 60)
            
            # Validate input
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"Error: Input file not found: {input_path}")
                sys.exit(1)
            
            if not input_path.suffix.lower() == ".docx":
                print(f"Warning: Input file does not have .docx extension: {input_path}")
            
            # Run preprocessing
            db_path = preprocess(args.input, args.output, args.model, args.device, args.chunking)
            
            print("\n" + "=" * 60)
            print("PREPROCESSING COMPLETE")
            print("=" * 60)
            print(f"Vector database saved to: {db_path}")
            print(f"\nNext step: Run with --mode process to generate report")
            print(f"  python main.py --mode process --input {db_path} --output {Path(args.output) / 'report.csv'}")
            
        elif args.mode == "process":
            print("=" * 60)
            print("PROCESS MODE")
            print("=" * 60)
            
            # Validate input (check if .npy and .json exist)
            input_path = Path(args.input)
            npy_path = input_path.with_suffix('.npy')
            json_path = input_path.with_suffix('.json')
            
            if not npy_path.exists():
                print(f"Error: Vector database embeddings not found: {npy_path}")
                sys.exit(1)
            
            if not json_path.exists():
                print(f"Error: Vector database metadata not found: {json_path}")
                sys.exit(1)
            
            # Generate report
            generate_report(
                args.input,
                args.output,
                top_k=args.top_k,
                min_similarity=args.min_similarity
            )
            
            print("\n" + "=" * 60)
            print("REPORT GENERATION COMPLETE")
            print("=" * 60)
            print(f"Report saved to: {args.output}")
        
        elif args.mode == "process-inter-h2":
            print("=" * 60)
            print("PROCESS INTER-H2 MODE")
            print("=" * 60)
            
            # Validate input (check if .npy and .json exist)
            input_path = Path(args.input)
            npy_path = input_path.with_suffix('.npy')
            json_path = input_path.with_suffix('.json')
            
            if not npy_path.exists():
                print(f"Error: Vector database embeddings not found: {npy_path}")
                sys.exit(1)
            
            if not json_path.exists():
                print(f"Error: Vector database metadata not found: {json_path}")
                sys.exit(1)
            
            # Generate inter-H2 report
            generate_inter_h2_report(
                args.input,
                args.output,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
                exclude_threshold=args.exclude_threshold,
                score_threshold=args.score_threshold,
                score_exponent=args.score_exponent
            )
            
            print("\n" + "=" * 60)
            print("INTER-H2 REPORT GENERATION COMPLETE")
            print("=" * 60)
            print(f"Reports saved to: {args.output}")
            print(f"  - Summary: {Path(args.output) / 'summary.csv'}")
            print(f"  - Individual reports: {Path(args.output) / 'h2_reports'}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

