"""Unified memory query system."""

import numpy as np
from typing import Dict, Any, Tuple

class MemorySystem:
    """Unified memory system combining short and medium term memory."""
    
    def __init__(self, short_term_memory, medium_term_memory, config: Dict[str, Any]):
        """Initialize memory system.
        
        Args:
            short_term_memory: ShortTermMemory instance
            medium_term_memory: MediumTermMemory instance
            config: Memory configuration
        """
        self.stm = short_term_memory
        self.mtm = medium_term_memory
        self.config = config
        
    def query(self, embedding: np.ndarray, track_id=None) -> Tuple[str, float]:
        """Query memory for face identification.
        
        Args:
            embedding: Face embedding
            track_id: Track ID (optional)
            
        Returns:
            Tuple of (identity, confidence)
        """
        # Query short-term memory first (recency-weighted)
        stm_identity, stm_conf = self.stm.query(embedding, track_id)
        
        # Query medium-term memory for persistent variations
        mtm_identity, mtm_conf = self.mtm.query(embedding)
        
        # Determine best match from memory systems
        if stm_conf > 0.7:  # Strong recent match
            return stm_identity, stm_conf
        elif mtm_conf > 0.8:  # Strong variation match
            return mtm_identity, mtm_conf
        elif stm_conf > 0.6 and mtm_conf > 0.6 and stm_identity == mtm_identity:
            # Both systems agree with moderate confidence
            combined_conf = (stm_conf * 0.6) + (mtm_conf * 0.4)
            return stm_identity, combined_conf
        
        # No strong memory match
        if stm_conf > mtm_conf:
            return stm_identity, stm_conf
        else:
            return mtm_identity, mtm_conf
            
    def update(self, track_id: int, embedding: np.ndarray, 
              identity: str, confidence: float):
        """Update memory with new recognition result.
        
        Args:
            track_id: Track ID
            embedding: Face embedding
            identity: Recognized identity
            confidence: Recognition confidence
        """
        # Update short-term memory with current recognition
        self.stm.add(track_id, embedding, identity, confidence)
        
        # Update medium-term memory if confidence is high enough
        if confidence > 0.75:
            self.mtm.update(identity, embedding, confidence)