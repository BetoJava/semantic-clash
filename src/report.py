import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import numpy as np

from .vector_db import VectorDB


def _load_and_compute_similarities(vector_db_path: str) -> Tuple[VectorDB, np.ndarray]:
    """
    Load vector database and compute similarity matrix.
    
    Args:
        vector_db_path: Path to the vector database
        
    Returns:
        Tuple of (vector_db, similarity_matrix)
    """
    print(f"Loading vector database from: {vector_db_path}")
    step_start = time.perf_counter()
    vector_db = VectorDB.load(vector_db_path)
    step_time = time.perf_counter() - step_start
    print(f"Database loaded (Time: {step_time:.2f}s)")
    
    n_vectors = len(vector_db)
    print(f"\nComputing pairwise similarities for {n_vectors} vectors...")
    
    step_start = time.perf_counter()
    similarity_matrix = vector_db.compute_all_distances()
    step_time = time.perf_counter() - step_start
    print(f"Similarity matrix computed (Time: {step_time:.2f}s)")
    
    return vector_db, similarity_matrix


def _extract_pairs(
    similarity_matrix: np.ndarray,
    min_similarity: float = 0.0,
    pair_filter: Optional[Callable[[int, int], bool]] = None
) -> List[Tuple[int, int, float]]:
    """
    Extract pairs from similarity matrix with optional filtering.
    
    Args:
        similarity_matrix: Similarity matrix (numpy array)
        min_similarity: Minimum similarity score to include
        pair_filter: Optional function to filter pairs (returns True to include, False to exclude)
        
    Returns:
        List of (i, j, similarity) tuples
    """
    print(f"\nExtracting pairs...")
    step_start = time.perf_counter()
    pairs = []
    n_vectors = similarity_matrix.shape[0]
    
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            similarity = float(similarity_matrix[i, j])
            if similarity >= min_similarity:
                # Apply filter if provided
                if pair_filter is None or pair_filter(i, j):
                    pairs.append((i, j, similarity))
    
    step_time = time.perf_counter() - step_start
    print(f"Found {len(pairs)} pairs with similarity >= {min_similarity} (Time: {step_time:.2f}s)")
    
    return pairs


def _sort_and_select_top_k(pairs: List[Tuple[int, int, float]], top_k: int) -> List[Tuple[int, int, float]]:
    """
    Sort pairs by similarity and select top K.
    
    Args:
        pairs: List of (i, j, similarity) tuples
        top_k: Number of top pairs to select
        
    Returns:
        Top K pairs sorted by similarity (descending)
    """
    print(f"\nSorting and selecting top {top_k} pairs...")
    step_start = time.perf_counter()
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:top_k]
    step_time = time.perf_counter() - step_start
    print(f"Top {len(top_pairs)} pairs selected (Time: {step_time:.2f}s)")
    
    return top_pairs


def _write_csv_report(
    output_path: Path,
    vector_db: VectorDB,
    top_pairs: List[Tuple[int, int, float]]
) -> None:
    """
    Write CSV report file.
    
    Args:
        output_path: Path to output CSV file
        vector_db: Vector database
        top_pairs: List of (i, j, similarity) tuples to write
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write CSV header
        f.write("score;chunk_text_1;chunk_text_2\n")
        
        for i, j, score in top_pairs:
            chunk1 = vector_db.get_metadata(i)
            chunk2 = vector_db.get_metadata(j)
            
            text1 = chunk1.get("text", "")
            text2 = chunk2.get("text", "")
            
            # Format CSV with semicolon separator: score;chunk_text_1;chunk_text_2
            # Replace newlines with spaces and escape semicolons by quoting if needed
            text1_clean = text1.replace('\n', ' ').replace('\r', ' ').strip()
            text2_clean = text2.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Escape semicolons and quotes by wrapping in quotes if needed
            if ';' in text1_clean or '"' in text1_clean:
                text1_clean = '"' + text1_clean.replace('"', '""') + '"'
            if ';' in text2_clean or '"' in text2_clean:
                text2_clean = '"' + text2_clean.replace('"', '""') + '"'
            
            # Truncate very long texts (optional, for readability)
            max_length = 500
            if len(text1_clean) > max_length:
                text1_clean = text1_clean[:max_length] + "..."
            if len(text2_clean) > max_length:
                text2_clean = text2_clean[:max_length] + "..."
            
            line = f"{score:.6f};{text1_clean};{text2_clean}\n"
            f.write(line)


def generate_report(
    vector_db_path: str,
    output_csv_path: str,
    top_k: int = 200,
    min_similarity: float = 0.0
) -> None:
    """
    Generate a report of the top K most similar chunk pairs.
    
    Args:
        vector_db_path: Path to the vector database (base path, without extension)
        output_csv_path: Path to save the output CSV report
        top_k: Number of top overlaps to include
        min_similarity: Minimum similarity score to include (default: 0.0, include all)
    """
    total_start_time = time.perf_counter()
    
    # Load and compute similarities
    vector_db, similarity_matrix = _load_and_compute_similarities(vector_db_path)
    
    # Extract pairs (no filter)
    pairs = _extract_pairs(similarity_matrix, min_similarity)
    
    # Sort and select top K
    top_pairs = _sort_and_select_top_k(pairs, top_k)
    
    # Generate report
    print(f"\nGenerating report...")
    step_start = time.perf_counter()
    output_path = Path(output_csv_path)
    _write_csv_report(output_path, vector_db, top_pairs)
    step_time = time.perf_counter() - step_start
    
    total_time = time.perf_counter() - total_start_time
    
    print(f"Report saved to: {output_path} (Time: {step_time:.2f}s)")
    print(f"\nTotal pairs in report: {len(top_pairs)}")
    
    if top_pairs:
        print(f"Highest similarity: {top_pairs[0][2]:.6f}")
        print(f"Lowest similarity (in top {top_k}): {top_pairs[-1][2]:.6f}")
    
    print(f"\nTotal processing time: {total_time:.2f}s")


def generate_report_filtered(
    vector_db_path: str,
    output_csv_path: str,
    exclude_level: str = "h3",
    top_k: int = 200,
    min_similarity: float = 0.0
) -> None:
    """
    Generate a report of the top K most similar chunk pairs, excluding pairs
    from the same hierarchical section.
    
    Args:
        vector_db_path: Path to the vector database (base path, without extension)
        output_csv_path: Path to save the output CSV report
        exclude_level: Hierarchical level to exclude matches from ("h1", "h2", or "h3")
        top_k: Number of top overlaps to include
        min_similarity: Minimum similarity score to include (default: 0.0, include all)
    """
    if exclude_level not in ["h1", "h2", "h3"]:
        raise ValueError(f"exclude_level must be 'h1', 'h2', or 'h3', got '{exclude_level}'")
    
    total_start_time = time.perf_counter()
    
    # Load and compute similarities
    vector_db, similarity_matrix = _load_and_compute_similarities(vector_db_path)
    
    # Create filter function based on exclude_level
    def pair_filter(i: int, j: int) -> bool:
        """Filter out pairs from the same hierarchical section."""
        chunk1 = vector_db.get_metadata(i)
        chunk2 = vector_db.get_metadata(j)
        
        # Get the ID at the specified level for both chunks
        id1 = chunk1.get(exclude_level)
        id2 = chunk2.get(exclude_level)
        
        # Exclude if both have the same ID (same section) and ID is not None
        if id1 is not None and id2 is not None and id1 == id2:
            return False
        
        return True
    
    # Extract pairs with filter
    pairs = _extract_pairs(similarity_matrix, min_similarity, pair_filter)
    
    # Sort and select top K
    top_pairs = _sort_and_select_top_k(pairs, top_k)
    
    # Generate report
    print(f"\nGenerating report (excluding pairs from same {exclude_level.upper()} section)...")
    step_start = time.perf_counter()
    output_path = Path(output_csv_path)
    _write_csv_report(output_path, vector_db, top_pairs)
    step_time = time.perf_counter() - step_start
    
    total_time = time.perf_counter() - total_start_time
    
    print(f"Report saved to: {output_path} (Time: {step_time:.2f}s)")
    print(f"\nTotal pairs in report: {len(top_pairs)}")
    print(f"Filtered by: excluding pairs from same {exclude_level.upper()} section")
    
    if top_pairs:
        print(f"Highest similarity: {top_pairs[0][2]:.6f}")
        print(f"Lowest similarity (in top {top_k}): {top_pairs[-1][2]:.6f}")
    
    print(f"\nTotal processing time: {total_time:.2f}s")
