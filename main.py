"""
Main entry point for Semantic Clash application.
"""

import argparse
import sys
from pathlib import Path

from src.preprocess import preprocess
from src.process import generate_report


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
        """
    )
    
    parser.add_argument(
        "--mode",
        required=True,
        choices=["preprocess", "process"],
        help="Processing mode: 'preprocess' for DOCX parsing/embedding, 'process' for report generation"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input path: DOCX file (preprocess mode) or vector DB base path (process mode)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output path: directory (preprocess mode) or TXT file path (process mode)"
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
        help="Number of top overlaps to include in report (process mode only, default: 100)"
    )
    
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum similarity score to include (process mode only, default: 0.0)"
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
            db_path = preprocess(args.input, args.output, args.model)
            
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

