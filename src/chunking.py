"""
Chunking strategies for text processing.
Provides base class and naive implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import uuid


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of document elements into smaller pieces.
        
        Args:
            elements: List of document elements with 'text', 'type', 'title_level', etc.
            
        Returns:
            List of chunks with 'text' and 'metadata' keys
        """
        pass


class NaiveChunkingStrategy(ChunkingStrategy):
    """
    Naive chunking strategy: creates chunks of approximately 50 words,
    respecting natural separators (sentences, paragraphs).
    """
    
    def __init__(self, target_words: int = 50, min_words: int = 10, max_words: int = 100):
        """
        Initialize the naive chunking strategy.
        
        Args:
            target_words: Target number of words per chunk
            min_words: Minimum words per chunk (merge if too small)
            max_words: Maximum words per chunk (split if too large)
        """
        self.target_words = target_words
        self.min_words = min_words
        self.max_words = max_words
    
    def chunk(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk elements into approximately target_words-sized chunks.
        
        Args:
            elements: List of document elements
            
        Returns:
            List of chunks with text and metadata
        """
        chunks = []
        current_chunk_text = []
        current_chunk_words = 0
        current_metadata = {
            "source_types": [],
            "title_levels": [],
            "element_indices": [],
            "h1_id": None,  # Track hierarchical IDs
            "h2_id": None,
            "h3_id": None
        }
        
        for idx, element in enumerate(elements):
            text = element.get("text", "")
            element_type = element.get("type", "paragraph")
            title_level = element.get("title_level")
            
            # Get hierarchical IDs from element
            h1_id = element.get("h1_id")
            h2_id = element.get("h2_id")
            h3_id = element.get("h3_id")
            
            # Update current hierarchical IDs (use latest from element)
            if h1_id is not None:
                current_metadata["h1_id"] = h1_id
            if h2_id is not None:
                current_metadata["h2_id"] = h2_id
            if h3_id is not None:
                current_metadata["h3_id"] = h3_id
            
            # Skip empty elements
            if not text.strip():
                continue
            
            # Split text into sentences for better chunking
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # If adding this sentence would exceed max_words, finalize current chunk
                if current_chunk_words + sentence_words > self.max_words and current_chunk_text:
                    chunk = self._create_chunk(current_chunk_text, current_metadata, idx)
                    chunks.append(chunk)
                    current_chunk_text = []
                    current_chunk_words = 0
                    # Preserve hierarchical IDs when resetting metadata
                    current_metadata = {
                        "source_types": [],
                        "title_levels": [],
                        "element_indices": [],
                        "h1_id": current_metadata.get("h1_id"),
                        "h2_id": current_metadata.get("h2_id"),
                        "h3_id": current_metadata.get("h3_id")
                    }
                
                # Add sentence to current chunk
                current_chunk_text.append(sentence)
                current_chunk_words += sentence_words
                
                # Update metadata
                if element_type not in current_metadata["source_types"]:
                    current_metadata["source_types"].append(element_type)
                if title_level and title_level not in current_metadata["title_levels"]:
                    current_metadata["title_levels"].append(title_level)
                if idx not in current_metadata["element_indices"]:
                    current_metadata["element_indices"].append(idx)
                
                # If we've reached target size, finalize chunk
                if current_chunk_words >= self.target_words:
                    chunk = self._create_chunk(current_chunk_text, current_metadata, idx)
                    chunks.append(chunk)
                    current_chunk_text = []
                    current_chunk_words = 0
                    # Preserve hierarchical IDs when resetting metadata
                    current_metadata = {
                        "source_types": [],
                        "title_levels": [],
                        "element_indices": [],
                        "h1_id": current_metadata.get("h1_id"),
                        "h2_id": current_metadata.get("h2_id"),
                        "h3_id": current_metadata.get("h3_id")
                    }
        
        # Add remaining chunk if it meets minimum size
        if current_chunk_text:
            word_count = sum(len(s.split()) for s in current_chunk_text)
            if word_count >= self.min_words:
                chunk = self._create_chunk(current_chunk_text, current_metadata, len(elements) - 1)
                chunks.append(chunk)
            elif chunks:
                # Merge small remaining chunk with last chunk
                last_chunk = chunks[-1]
                last_chunk["text"] += " " + " ".join(current_chunk_text)
                last_chunk["metadata"]["word_count"] = len(last_chunk["text"].split())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using natural separators.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Split on sentence endings (. ! ?) followed by space or newline
        sentences = re.split(r'([.!?]+\s+)', text)
        
        # Recombine sentences with their punctuation
        result = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and re.match(r'[.!?]+\s+', sentences[i + 1]):
                result.append(sentences[i] + sentences[i + 1])
                i += 2
            elif sentences[i].strip():
                result.append(sentences[i])
                i += 1
            else:
                i += 1
        
        # Filter out empty sentences
        return [s.strip() for s in result if s.strip()]
    
    def _create_chunk(self, chunk_text: List[str], metadata: Dict[str, Any], element_idx: int) -> Dict[str, Any]:
        """
        Create a chunk dictionary from text parts and metadata.
        Generates a UUID4 for each chunk and includes hierarchical IDs.
        
        Args:
            chunk_text: List of text parts (sentences)
            metadata: Metadata dictionary
            element_idx: Index of the source element
            
        Returns:
            Chunk dictionary with UUID and hierarchical IDs
        """
        full_text = " ".join(chunk_text)
        word_count = len(full_text.split())
        
        # Generate UUID4 for this chunk
        chunk_id = str(uuid.uuid4())
        
        return {
            "text": full_text,
            "metadata": {
                **metadata,
                "chunk_id": chunk_id,  # UUID4 for this chunk
                "h1_id": metadata.get("h1_id"),  # H1 ID (can be None)
                "h2_id": metadata.get("h2_id"),  # H2 ID (can be None)
                "h3_id": metadata.get("h3_id"),  # H3 ID (can be None)
                "word_count": word_count,
                "char_count": len(full_text)
            }
        }


class LineBreakChunkingStrategy(ChunkingStrategy):
    """
    Line break chunking strategy: creates chunks by splitting on newlines.
    Each non-empty line becomes a separate chunk.
    """
    
    def chunk(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk elements by splitting on newlines.
        
        Args:
            elements: List of document elements
            
        Returns:
            List of chunks with text and metadata
        """
        chunks = []
        
        for idx, element in enumerate(elements):
            text = element.get("text", "")
            element_type = element.get("type", "paragraph")
            title_level = element.get("title_level")
            
            # Get hierarchical IDs from element
            h1_id = element.get("h1_id")
            h2_id = element.get("h2_id")
            h3_id = element.get("h3_id")
            
            # Skip empty elements
            if not text.strip():
                continue
            
            # Split text by newlines
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Create metadata for this chunk
                metadata = {
                    "source_types": [element_type],
                    "title_levels": [title_level] if title_level else [],
                    "element_indices": [idx],
                    "h1_id": h1_id,
                    "h2_id": h2_id,
                    "h3_id": h3_id
                }
                
                # Create chunk
                chunk = self._create_chunk([line], metadata, idx)
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, chunk_text: List[str], metadata: Dict[str, Any], element_idx: int) -> Dict[str, Any]:
        """
        Create a chunk dictionary from text parts and metadata.
        Generates a UUID4 for each chunk and includes hierarchical IDs.
        
        Args:
            chunk_text: List of text parts (lines)
            metadata: Metadata dictionary
            element_idx: Index of the source element
            
        Returns:
            Chunk dictionary with UUID and hierarchical IDs
        """
        full_text = " ".join(chunk_text)
        word_count = len(full_text.split())
        
        # Generate UUID4 for this chunk
        chunk_id = str(uuid.uuid4())
        
        return {
            "text": full_text,
            "metadata": {
                **metadata,
                "chunk_id": chunk_id,  # UUID4 for this chunk
                "h1_id": metadata.get("h1_id"),  # H1 ID (can be None)
                "h2_id": metadata.get("h2_id"),  # H2 ID (can be None)
                "h3_id": metadata.get("h3_id"),  # H3 ID (can be None)
                "word_count": word_count,
                "char_count": len(full_text)
            }
        }
