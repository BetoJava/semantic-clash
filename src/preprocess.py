"""
Preprocessing pipeline: parsing, chunking, embedding, and saving.
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

from .docx_parser import parse_docx
from .chunking import NaiveChunkingStrategy, LineBreakChunkingStrategy, ChunkingStrategy
from .embedding import Embedder
from .vector_db import VectorDB


def preprocess(
    docx_path: str, 
    output_dir: str, 
    model_name: str = "Qwen/Qwen3-Embedding-0.6B", 
    device: str = "cpu",
    chunking_strategy: str = "naive"
) -> str:
    """
    Complete preprocessing pipeline: parse docx, chunk, embed, and save.
    
    Args:
        docx_path: Path to the input DOCX file
        output_dir: Directory to save output files
        model_name: Name of the embedding model to use
        device: Device to use ('cpu' or 'cuda')
        chunking_strategy: Chunking strategy to use ('naive' or 'linebreak', default: 'naive')
        
    Returns:
        Path to the saved vector database
    """
    docx_path_obj = Path(docx_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    if not docx_path_obj.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path_obj}")
    
    print(f"Starting preprocessing pipeline for: {docx_path_obj}")
    
    total_start_time = time.perf_counter()
    
    # Step 1: Parse DOCX to JSON
    print("\n[1/4] Parsing DOCX file...")
    step_start = time.perf_counter()
    elements = parse_docx(str(docx_path_obj))
    step_time = time.perf_counter() - step_start
    print(f"Parsed {len(elements)} elements (Time: {step_time:.2f}s)")
    
    # Save intermediate JSON (optional, for debugging)
    json_path = output_dir_obj / f"{docx_path_obj.stem}_parsed.json"
    from .docx_parser import save_to_json
    save_start = time.perf_counter()
    save_to_json(elements, str(json_path))
    save_time = time.perf_counter() - save_start
    print(f"Saved parsed JSON to: {json_path} (Time: {save_time:.2f}s)")
    
    # Step 2: Chunk the elements
    print("\n[2/4] Chunking elements...")
    step_start = time.perf_counter()
    
    # Select chunking strategy
    strategy: ChunkingStrategy
    if chunking_strategy == "linebreak":
        strategy = LineBreakChunkingStrategy()
        print(f"Using LineBreakChunkingStrategy")
    else:  # default to "naive"
        strategy = NaiveChunkingStrategy(target_words=50)
        print(f"Using NaiveChunkingStrategy (target_words=50)")
    
    chunks = strategy.chunk(elements)
    step_time = time.perf_counter() - step_start
    print(f"Created {len(chunks)} chunks (Time: {step_time:.2f}s)")
    
    # Step 3: Embed chunks
    print("\n[3/4] Embedding chunks...")
    step_start = time.perf_counter()
    embedder = Embedder(model_name=model_name, device=device)
    embeddings = embedder.embed_chunks(chunks)
    step_time = time.perf_counter() - step_start
    print(f"Embedding complete (Time: {step_time:.2f}s)")
    
    # Step 4: Create and save vector database
    print("\n[4/4] Saving vector database...")
    step_start = time.perf_counter()
    
    # Prepare metadata (extract text and metadata from chunks)
    chunks_metadata = []
    for chunk in chunks:
        metadata = {
            "text": chunk["text"],
            **chunk.get("metadata", {})
        }
        chunks_metadata.append(metadata)
    
    vector_db = VectorDB(embeddings, chunks_metadata)
    
    # Save vector DB
    db_path = output_dir_obj / f"{docx_path_obj.stem}_vector_db"
    vector_db.save(str(db_path))
    
    step_time = time.perf_counter() - step_start
    print(f"Vector database saved (Time: {step_time:.2f}s)")
    
    total_time = time.perf_counter() - total_start_time
    
    print(f"\nPreprocessing complete! (Total time: {total_time:.2f}s)")
    print(f"Vector database saved to: {db_path}")
    print(f"  - Embeddings: {db_path.with_suffix('.npy')}")
    print(f"  - Metadata: {db_path.with_suffix('.json')}")
    
    return str(db_path)


def main():
    """CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess DOCX file: parse, chunk, embed, and save")
    parser.add_argument("--input", required=True, help="Path to input DOCX file")
    parser.add_argument("--output", required=True, help="Output directory for processed files")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model name")
    parser.add_argument("--chunking", default="naive", choices=["naive", "linebreak"], 
                       help="Chunking strategy: 'naive' (by word count) or 'linebreak' (by newlines, default: naive)")
    
    args = parser.parse_args()
    
    try:
        db_path = preprocess(args.input, args.output, args.model, chunking_strategy=args.chunking)
        print(f"\nSuccess! Vector database ready at: {db_path}")
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        raise

