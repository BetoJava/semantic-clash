"""
Chunking strategies for text processing.
Provides base class and naive implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re


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
            "element_indices": []
        }
        
        for idx, element in enumerate(elements):
            text = element.get("text", "")
            element_type = element.get("type", "paragraph")
            title_level = element.get("title_level")
            
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
                    current_metadata = {
                        "source_types": [],
                        "title_levels": [],
                        "element_indices": []
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
                    current_metadata = {
                        "source_types": [],
                        "title_levels": [],
                        "element_indices": []
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
        
        Args:
            chunk_text: List of text parts (sentences)
            metadata: Metadata dictionary
            element_idx: Index of the source element
            
        Returns:
            Chunk dictionary
        """
        full_text = " ".join(chunk_text)
        word_count = len(full_text.split())
        
        return {
            "text": full_text,
            "metadata": {
                **metadata,
                "word_count": word_count,
                "char_count": len(full_text)
            }
        }

