"""
Vector database module for storing and managing embeddings.
Uses NumPy for efficient storage and retrieval.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class VectorDB:
    """Vector database for storing embeddings and metadata."""
    
    def __init__(self, embeddings: Optional[np.ndarray] = None, chunks_metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize VectorDB.
        
        Args:
            embeddings: Numpy array of shape (n_vectors, embedding_dim)
            chunks_metadata: List of metadata dictionaries for each vector
        """
        self.embeddings = embeddings if embeddings is not None else np.array([])
        self.chunks_metadata = chunks_metadata if chunks_metadata is not None else []
        
        if embeddings is not None and chunks_metadata is not None:
            if len(embeddings) != len(chunks_metadata):
                raise ValueError("Number of embeddings must match number of metadata entries")
    
    def save(self, filepath: str) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            filepath: Base path for saving (will create .npy and .json files)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as .npy
        embeddings_path = filepath.with_suffix('.npy')
        np.save(embeddings_path, self.embeddings)
        print(f"Saved embeddings to {embeddings_path}")
        
        # Save metadata as .json
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VectorDB':
        """
        Load embeddings and metadata from disk.
        
        Args:
            filepath: Base path for loading (will load .npy and .json files)
            
        Returns:
            VectorDB instance
        """
        filepath = Path(filepath)
        
        # Load embeddings
        embeddings_path = filepath.with_suffix('.npy')
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings from {embeddings_path}, shape: {embeddings.shape}")
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            chunks_metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}, {len(chunks_metadata)} entries")
        
        return cls(embeddings, chunks_metadata)
    
    def add_vector(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        Add a vector and its metadata to the database.
        
        Args:
            embedding: Embedding vector of shape (embedding_dim,)
            metadata: Metadata dictionary for this vector
            
        Returns:
            Index of the added vector
        """
        embedding = np.array(embedding)
        
        # Ensure embedding is 1D
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        # Check dimension consistency
        if len(self.embeddings) > 0:
            expected_dim = self.embeddings.shape[1]
            if embedding.shape[0] != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {embedding.shape[0]}, "
                    f"expected {expected_dim}"
                )
        
        # Add embedding
        if len(self.embeddings) == 0:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
        
        # Add metadata
        self.chunks_metadata.append(metadata)
        
        return len(self.embeddings) - 1
    
    def remove_vector(self, index: int) -> None:
        """
        Remove a vector and its metadata by index.
        
        Args:
            index: Index of the vector to remove
        """
        if index < 0 or index >= len(self.embeddings):
            raise IndexError(f"Index {index} out of range [0, {len(self.embeddings)})")
        
        # Remove embedding
        self.embeddings = np.delete(self.embeddings, index, axis=0)
        
        # Remove metadata
        self.chunks_metadata.pop(index)
    
    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Returns a value between -1 and 1, where 1 means identical.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (higher = more similar)
        """
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector shape mismatch: {vec1.shape} vs {vec2.shape}")
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        return float(similarity)
    
    def compute_all_distances(self) -> np.ndarray:
        """
        Compute pairwise cosine similarities for all vectors in the database.
        
        Returns:
            Symmetric matrix of shape (n_vectors, n_vectors) with similarity scores
        """
        if len(self.embeddings) == 0:
            return np.array([])
        
        # Use sklearn for efficient computation
        similarity_matrix = cosine_similarity(self.embeddings)
        
        return similarity_matrix
    
    def get_vector(self, index: int) -> np.ndarray:
        """
        Get a vector by index.
        
        Args:
            index: Index of the vector
            
        Returns:
            Embedding vector
        """
        if index < 0 or index >= len(self.embeddings):
            raise IndexError(f"Index {index} out of range [0, {len(self.embeddings)})")
        
        return self.embeddings[index]
    
    def get_metadata(self, index: int) -> Dict[str, Any]:
        """
        Get metadata by index.
        
        Args:
            index: Index of the vector
            
        Returns:
            Metadata dictionary
        """
        if index < 0 or index >= len(self.chunks_metadata):
            raise IndexError(f"Index {index} out of range [0, {len(self.chunks_metadata)})")
        
        return self.chunks_metadata[index]
    
    def __len__(self) -> int:
        """Return the number of vectors in the database."""
        return len(self.embeddings)
    
    def __repr__(self) -> str:
        """String representation of the VectorDB."""
        return f"VectorDB(n_vectors={len(self)}, embedding_dim={self.embeddings.shape[1] if len(self.embeddings) > 0 else 0})"

