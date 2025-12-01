"""
Embedding module using Qwen3-Embedding-0.6B model.
Provides text vectorization capabilities.
"""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    """Wrapper for sentence transformer embedding model."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cpu"):
        """
        Initialize the embedder with a specific model.
        
        Args:
            model_name: Name of the HuggingFace model
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        # Ensure CPU is used
        if device == "cpu":
            torch.set_num_threads(1)  # Optimize for CPU
        
        print(f"Loading embedding model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        print("Model loaded successfully.")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed a list of text chunks into vectors.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            Numpy array of shape (n_chunks, embedding_dim)
        """
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings into vectors.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        # Qwen3-Embedding-0.6B has 1024 dimensions by default
        # But it can be configured, so we check the model config
        try:
            return self.model.get_sentence_embedding_dimension()
        except:
            # Fallback to default
            return 1024

