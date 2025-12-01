import time
from pathlib import Path

from .vector_db import VectorDB


def generate_report(
    vector_db_path: str,
    output_csv_path: str,
    top_k: int = 100,
    min_similarity: float = 0.0
) -> None:
    """
    Generate a report of the top K most similar chunk pairs.
    
    Args:
        vector_db_path: Path to the vector database (base path, without extension)
        output_csv_path: Path to save the output TXT report
        top_k: Number of top overlaps to include
        min_similarity: Minimum similarity score to include (default: 0.0, include all)
    """
    total_start_time = time.perf_counter()
    
    # Step 1: Load vector database
    print(f"Loading vector database from: {vector_db_path}")
    step_start = time.perf_counter()
    vector_db = VectorDB.load(vector_db_path)
    step_time = time.perf_counter() - step_start
    print(f"Database loaded (Time: {step_time:.2f}s)")
    
    n_vectors = len(vector_db)
    print(f"\nComputing pairwise similarities for {n_vectors} vectors...")
    
    # Step 2: Compute all pairwise similarities
    step_start = time.perf_counter()
    similarity_matrix = vector_db.compute_all_distances()
    step_time = time.perf_counter() - step_start
    print(f"Similarity matrix computed (Time: {step_time:.2f}s)")
    
    # Step 3: Extract pairs
    print(f"\nExtracting pairs...")
    step_start = time.perf_counter()
    pairs = []
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            similarity = similarity_matrix[i, j]
            if similarity >= min_similarity:
                pairs.append((i, j, similarity))
    step_time = time.perf_counter() - step_start
    print(f"Found {len(pairs)} pairs with similarity >= {min_similarity} (Time: {step_time:.2f}s)")
    
    # Step 4: Sort and select top K
    print(f"\nSorting and selecting top {top_k} pairs...")
    step_start = time.perf_counter()
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:top_k]
    step_time = time.perf_counter() - step_start
    print(f"Top {len(top_pairs)} pairs selected (Time: {step_time:.2f}s)")
    
    # Step 5: Generate report
    print(f"\nGenerating report...")
    step_start = time.perf_counter()
    output_path = Path(output_csv_path)
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
    
    step_time = time.perf_counter() - step_start
    total_time = time.perf_counter() - total_start_time
    
    print(f"Report saved to: {output_path} (Time: {step_time:.2f}s)")
    print(f"\nTotal pairs in report: {len(top_pairs)}")
    
    if top_pairs:
        print(f"Highest similarity: {top_pairs[0][2]:.6f}")
        print(f"Lowest similarity (in top {top_k}): {top_pairs[-1][2]:.6f}")
    
    print(f"\nTotal processing time: {total_time:.2f}s")
