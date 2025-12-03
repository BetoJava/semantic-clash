import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict
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


def _group_chunks_by_h2(vector_db: VectorDB) -> Dict[str, List[int]]:
    """
    Group chunk indices by their h2_id.
    
    Args:
        vector_db: Vector database
        
    Returns:
        Dictionary mapping h2_id to list of chunk indices
    """
    h2_groups: Dict[str, List[int]] = {}
    
    for idx in range(len(vector_db)):
        metadata = vector_db.get_metadata(idx)
        h2_id = metadata.get("h2_id")
        
        # Use "None" as key for chunks without h2_id
        key = str(h2_id) if h2_id is not None else "None"
        
        if key not in h2_groups:
            h2_groups[key] = []
        h2_groups[key].append(idx)
    
    return h2_groups


def _compute_intra_h2_pairs(
    vector_db: VectorDB,
    similarity_matrix: np.ndarray,
    h2_groups: Dict[str, List[int]],
    min_similarity: float = 0.0,
    exclude_threshold: float = 0.999
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Extract pairs from the same H2 section, excluding scores above threshold.
    
    Args:
        vector_db: Vector database
        similarity_matrix: Similarity matrix
        h2_groups: Dictionary mapping h2_id to list of chunk indices
        min_similarity: Minimum similarity score to include
        exclude_threshold: Exclude pairs with similarity > this threshold (default: 0.999)
        
    Returns:
        Dictionary mapping h2_id to list of (i, j, similarity) tuples
    """
    print(f"\nExtracting intra-H2 pairs (excluding similarity > {exclude_threshold})...")
    step_start = time.perf_counter()
    h2_pairs: Dict[str, List[Tuple[int, int, float]]] = {}
    
    # Process each H2 group
    for h2_id, chunk_indices in h2_groups.items():
        if h2_id == "None":
            continue  # Skip chunks without h2_id
        
        pairs_for_h2 = []
        n_chunks = len(chunk_indices)
        
        # Compare chunks within this H2
        for idx_i in range(n_chunks):
            for idx_j in range(idx_i + 1, n_chunks):
                i = chunk_indices[idx_i]
                j = chunk_indices[idx_j]
                similarity = float(similarity_matrix[i, j])
                
                # Exclude scores above threshold (near-identical texts)
                if similarity > exclude_threshold:
                    continue
                
                # Only include pairs above min_similarity
                if similarity < min_similarity:
                    continue
                
                pairs_for_h2.append((i, j, similarity))
        
        if pairs_for_h2:
            h2_pairs[h2_id] = pairs_for_h2
    
    step_time = time.perf_counter() - step_start
    total_pairs = sum(len(pairs) for pairs in h2_pairs.values())
    print(f"Found {total_pairs} intra-H2 pairs across {len(h2_pairs)} H2 sections (Time: {step_time:.2f}s)")
    
    return h2_pairs


def _compute_h2_redundancy_scores(
    h2_pairs: Dict[str, List[Tuple[int, int, float]]],
    threshold: float = 0.5,
    exponent: float = 2.0
) -> Dict[str, Tuple[float, int]]:
    """
    Compute redundancy scores for each H2 section based on intra-H2 pairs.
    Score formula: sum(max(0, similarity - threshold) ** exponent)
    
    Args:
        h2_pairs: Dictionary mapping h2_id to list of (i, j, similarity) tuples
        threshold: Threshold for score calculation (default: 0.5)
        exponent: Exponent for score calculation (default: 2.0)
        
    Returns:
        Dictionary mapping h2_id to (redundancy_score, pair_count) tuple
    """
    h2_scores: Dict[str, Tuple[float, int]] = {}
    
    for h2_id, pairs in h2_pairs.items():
        score = 0.0
        for i, j, similarity in pairs:
            # Compute contribution: max(0, similarity - threshold) ** exponent
            contribution = max(0.0, similarity - threshold) ** exponent
            score += contribution
        
        h2_scores[h2_id] = (score, len(pairs))
    
    return h2_scores


def _write_intra_h2_csv_report(
    output_path: Path,
    vector_db: VectorDB,
    pairs: List[Tuple[int, int, float]]
) -> None:
    """
    Write CSV report file for a specific H2 section with intra-H2 pairs.
    
    Args:
        output_path: Path to output CSV file
        vector_db: Vector database
        pairs: List of (i, j, similarity) tuples for this H2
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write CSV header
        f.write("score;chunk_text_1;chunk_text_2\n")
        
        for i, j, score in pairs:
            chunk1 = vector_db.get_metadata(i)
            chunk2 = vector_db.get_metadata(j)
            
            text1 = chunk1.get("text", "")
            text2 = chunk2.get("text", "")
            
            # Clean and escape text
            text1_clean = text1.replace('\n', ' ').replace('\r', ' ').strip()
            text2_clean = text2.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Escape semicolons and quotes
            if ';' in text1_clean or '"' in text1_clean:
                text1_clean = '"' + text1_clean.replace('"', '""') + '"'
            if ';' in text2_clean or '"' in text2_clean:
                text2_clean = '"' + text2_clean.replace('"', '""') + '"'
            
            # Truncate very long texts
            max_length = 500
            if len(text1_clean) > max_length:
                text1_clean = text1_clean[:max_length] + "..."
            if len(text2_clean) > max_length:
                text2_clean = text2_clean[:max_length] + "..."
            
            line = f"{score:.6f};{text1_clean};{text2_clean}\n"
            f.write(line)


def _write_summary_csv(
    output_path: Path,
    h2_scores: Dict[str, Tuple[float, int]],
    h2_groups: Dict[str, List[int]]
) -> None:
    """
    Write summary CSV report with H2 sections ordered by redundancy score.
    
    Args:
        output_path: Path to output CSV file
        h2_scores: Dictionary mapping h2_id to (redundancy_score, pair_count) tuple
        h2_groups: Dictionary mapping h2_id to list of chunk indices
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort H2 sections by redundancy score (descending)
    sorted_h2 = sorted(h2_scores.items(), key=lambda x: x[1][0], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write CSV header
        f.write("h2_id;redundancy_score;pair_count;chunk_count\n")
        
        for h2_id, (score, pair_count) in sorted_h2:
            chunk_count = len(h2_groups.get(h2_id, []))
            line = f"{h2_id};{score:.6f};{pair_count};{chunk_count}\n"
            f.write(line)


def generate_inter_h2_report(
    vector_db_path: str,
    output_dir: str,
    top_k: int = 100,
    min_similarity: float = 0.0,
    exclude_threshold: float = 0.999,
    score_threshold: float = 0.5,
    score_exponent: float = 2.0
) -> None:
    """
    Generate intra-H2 similarity reports.
    Creates individual CSV reports for each H2 section and a summary report.
    Compares chunks within the same H2 section to find redundant content.
    
    Args:
        vector_db_path: Path to the vector database (base path, without extension)
        output_dir: Directory to save output reports
        top_k: Number of top pairs per H2 to include
        min_similarity: Minimum similarity score to include
        exclude_threshold: Exclude pairs with similarity > this threshold (default: 0.999)
        score_threshold: Threshold for redundancy score calculation (default: 0.5)
        score_exponent: Exponent for redundancy score calculation (default: 2.0)
    """
    total_start_time = time.perf_counter()
    
    # Load and compute similarities
    vector_db, similarity_matrix = _load_and_compute_similarities(vector_db_path)
    
    # Group chunks by H2
    print(f"\nGrouping chunks by H2...")
    step_start = time.perf_counter()
    h2_groups = _group_chunks_by_h2(vector_db)
    # Filter out "None" group
    h2_groups_with_id = {k: v for k, v in h2_groups.items() if k != "None"}
    step_time = time.perf_counter() - step_start
    print(f"Found {len(h2_groups_with_id)} H2 sections (Time: {step_time:.2f}s)")
    
    if not h2_groups_with_id:
        print("\nNo H2 sections found. Exiting.")
        return
    
    # Extract intra-H2 pairs (excluding scores > exclude_threshold)
    h2_pairs = _compute_intra_h2_pairs(vector_db, similarity_matrix, h2_groups_with_id, min_similarity, exclude_threshold)
    
    if not h2_pairs:
        print("\nNo intra-H2 pairs found. Exiting.")
        return
    
    # Compute redundancy scores
    print(f"\nComputing redundancy scores (threshold={score_threshold}, exponent={score_exponent})...")
    step_start = time.perf_counter()
    h2_scores = _compute_h2_redundancy_scores(h2_pairs, score_threshold, score_exponent)
    step_time = time.perf_counter() - step_start
    print(f"Computed scores for {len(h2_scores)} H2 sections (Time: {step_time:.2f}s)")
    
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    h2_reports_dir = output_dir_path / "h2_reports"
    h2_reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual reports for each H2
    print(f"\nGenerating individual H2 reports...")
    step_start = time.perf_counter()
    report_count = 0
    for h2_id, pairs_for_h2 in h2_pairs.items():
        # Sort pairs by similarity and take top K
        pairs_for_h2.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs_for_h2[:top_k]
        
        if not top_pairs:
            continue
        
        # Generate filename: 1_<h2_id>.csv, 2_<h2_id>.csv, etc.
        report_count += 1
        filename = f"{report_count}_{h2_id}.csv"
        report_path = h2_reports_dir / filename
        
        _write_intra_h2_csv_report(report_path, vector_db, top_pairs)
    
    step_time = time.perf_counter() - step_start
    print(f"Generated {report_count} individual reports (Time: {step_time:.2f}s)")
    
    # Generate summary report
    print(f"\nGenerating summary report...")
    step_start = time.perf_counter()
    summary_path = output_dir_path / "summary.csv"
    _write_summary_csv(summary_path, h2_scores, h2_groups_with_id)
    step_time = time.perf_counter() - step_start
    print(f"Summary report saved (Time: {step_time:.2f}s)")
    
    total_time = time.perf_counter() - total_start_time
    
    print(f"\nSummary report saved to: {summary_path}")
    print(f"Individual reports saved to: {h2_reports_dir}")
    print(f"Total H2 sections with intra-H2 similarities: {len(h2_scores)}")
    
    if h2_scores:
        max_score_h2 = max(h2_scores.items(), key=lambda x: x[1][0])
        print(f"Highest redundancy score: {max_score_h2[1][0]:.6f} (H2: {max_score_h2[0]})")
    
    print(f"\nTotal processing time: {total_time:.2f}s")
