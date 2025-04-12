"""Medium-term memory system for identity variations."""

import numpy as np
from typing import Dict, Any, List, Tuple

class MediumTermMemory:
    """Memory system that stores identity variations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize medium-term memory.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self.max_variations = config.get('mtm_max_variations', 5)
        
        # Initialize memory - dictionary of identity -> list of (embedding, confidence, count)
        self.memory = {}
        
    def update(self, identity: str, embedding: np.ndarray, confidence: float):
        """Update medium-term memory with new embedding.
        
        Args:
            identity: Identity label
            embedding: Face embedding
            confidence: Recognition confidence
        """
        # Skip unknown identities
        if identity == "Unknown":
            return
            
        # Initialize entry if not exists
        if identity not in self.memory:
            self.memory[identity] = []
            
        # Find if this embedding is similar to any existing variation
        best_similarity = 0.0
        best_idx = -1
        
        for i, (var_embedding, var_conf, count) in enumerate(self.memory[identity]):
            similarity = float(np.dot(embedding, var_embedding))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
                
        # Check if similar to existing variation
        if best_similarity > 0.8 and best_idx >= 0:
            # Update existing entry with weighted average
            old_embedding, old_conf, count = self.memory[identity][best_idx]
            
            # Weighted average of embeddings
            updated_embedding = old_embedding * 0.7 + embedding * 0.3
            normalized_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            
            # Update with higher confidence and increased count
            self.memory[identity][best_idx] = (
                normalized_embedding, 
                max(old_conf, confidence), 
                count + 1
            )
        else:
            # Add as new variation if space available
            if len(self.memory[identity]) < self.max_variations:
                self.memory[identity].append((embedding.copy(), confidence, 1))
            else:
                # Replace lowest count variation
                lowest_idx = min(range(len(self.memory[identity])), 
                                key=lambda i: self.memory[identity][i][2])
                self.memory[identity][lowest_idx] = (embedding.copy(), confidence, 1)
                
    def query(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Query medium-term memory for a matching face.
        
        Args:
            embedding: Face embedding to query
            
        Returns:
            Tuple of (identity, confidence)
        """
        best_identity = "Unknown"
        best_confidence = 0.0
        
        # Check each identity
        for identity, variations in self.memory.items():
            # Check each variation
            for var_embedding, var_conf, count in variations:
                # Calculate similarity
                similarity = float(np.dot(embedding, var_embedding))
                
                # Calculate combined confidence (similarity, variation confidence, count factor)
                count_factor = min(1.0, count/5.0)  # More seen variations are more reliable
                combined_confidence = similarity * var_conf * count_factor
                
                # Update best match
                if combined_confidence > best_confidence and similarity > 0.5:
                    best_identity = identity
                    best_confidence = combined_confidence
                    
        return best_identity, min(best_confidence, 1.0)  # Cap at 1.0