"""Short-term memory system for recent recognitions."""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

class ShortTermMemory:
    """Memory system that stores recent recognitions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize short-term memory.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self.max_size = config.get('stm_size', 100)
        self.decay_factor = config.get('memory_decay_factor', 0.95)
        
        # Initialize memory
        self.memory = deque(maxlen=self.max_size)  # (track_id, embedding, identity, confidence, timestamp)
        
    def add(self, track_id: int, embedding: np.ndarray, 
           identity: str, confidence: float):
        """Add recognition result to memory.
        
        Args:
            track_id: Track ID
            embedding: Face embedding
            identity: Recognized identity
            confidence: Recognition confidence
        """
        # Add to memory with current timestamp
        self.memory.append((track_id, embedding, identity, confidence, time.time()))
        
    def query(self, embedding: np.ndarray, 
             track_id: Optional[int] = None) -> Tuple[str, float]:
        """Query memory for a matching face.
        
        Args:
            embedding: Face embedding to query
            track_id: Track ID for track-based matching
            
        Returns:
            Tuple of (identity, confidence)
        """
        best_identity = "Unknown"
        best_confidence = 0.0
        
        # Check recent recognitions, starting from most recent
        for mem_track_id, mem_embedding, mem_identity, mem_confidence, mem_time in reversed(self.memory):
            # Track matching gives a big boost in confidence
            track_match_factor = 1.5 if track_id is not None and track_id == mem_track_id else 1.0
            
            # Time decay - more recent matches have higher weight
            time_factor = max(0.5, min(1.0, 1.0 - (time.time() - mem_time)/10.0))
            
            # Calculate embedding similarity
            similarity = float(np.dot(embedding, mem_embedding))
            
            # Calculate combined confidence
            combined_confidence = similarity * mem_confidence * time_factor * track_match_factor
            
            # Check if this is the best match so far
            if combined_confidence > best_confidence and similarity > 0.5:
                best_identity = mem_identity
                best_confidence = combined_confidence
                
        return best_identity, min(best_confidence, 1.0)  # Cap at 1.0