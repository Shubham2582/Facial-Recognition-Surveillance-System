"""Re-identification manager for face tracking."""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

class ReIDManager:
    """Re-identification manager for face tracking."""
    
    def __init__(self, config: Dict[str, Any], recognizer):
        """Initialize Re-ID manager.
        
        Args:
            config: Re-ID configuration
            recognizer: Face recognizer instance
        """
        self.config = config
        self.recognizer = recognizer
        self.reid_threshold = config.get('reid_threshold', 0.5)
        self.reid_interval = config.get('reid_interval', 10)
        
        # Initialize track database
        self.track_database = {}  # track_id -> embedding
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def add_track(self, track_id: int, embedding: np.ndarray) -> None:
        """Add track to Re-ID database.
        
        Args:
            track_id: Track ID
            embedding: Face embedding
        """
        self.track_database[track_id] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
        
    def remove_track(self, track_id: int) -> None:
        """Remove track from Re-ID database.
        
        Args:
            track_id: Track ID
        """
        if track_id in self.track_database:
            del self.track_database[track_id]
            
    def should_reid(self, track) -> bool:
        """Check if track should be re-identified.
        
        Args:
            track: Track object
            
        Returns:
            True if track should be re-identified
        """
        # Always re-ID if track is lost
        if track.get('lost', False):
            return True
            
        # Re-ID based on interval
        if track.get('frames_since_reid', 0) >= self.reid_interval:
            return True
            
        # Re-ID based on priority
        if track.get('priority', 0) >= 7:
            if track.get('frames_since_reid', 0) >= self.reid_interval // 2:
                return True
                
        return False
        
    def match_track(self, embedding: np.ndarray, 
                   min_similarity: float = 0.7) -> Tuple[int, float]:
        """Match embedding with known tracks.
        
        Args:
            embedding: Face embedding
            min_similarity: Minimum similarity threshold
            
        Returns:
            Tuple of (track_id, similarity)
        """
        best_match = -1
        best_similarity = 0.0
        
        for track_id, data in self.track_database.items():
            track_embedding = data['embedding']
            
            # Calculate similarity
            similarity = self.recognizer.compare_embeddings(embedding, track_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = track_id
                
        # Check threshold
        if best_similarity < min_similarity:
            return -1, 0.0
            
        return best_match, best_similarity
        
    def update_track(self, track_id: int, embedding: np.ndarray) -> None:
        """Update track embedding.
        
        Args:
            track_id: Track ID
            embedding: New face embedding
        """
        if track_id in self.track_database:
            old_embedding = self.track_database[track_id]['embedding']
            
            # Weighted average of embeddings
            updated_embedding = old_embedding * 0.7 + embedding * 0.3
            normalized_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            
            self.track_database[track_id] = {
                'embedding': normalized_embedding,
                'timestamp': time.time()
            }
        else:
            self.add_track(track_id, embedding)
            
    def clean_old_tracks(self, max_age: float = 300.0) -> None:
        """Remove old tracks from database.
        
        Args:
            max_age: Maximum track age in seconds
        """
        now = time.time()
        
        # Find old tracks
        old_tracks = []
        for track_id, data in self.track_database.items():
            if now - data['timestamp'] > max_age:
                old_tracks.append(track_id)
                
        # Remove old tracks
        for track_id in old_tracks:
            self.remove_track(track_id)
            
        if old_tracks:
            self.logger.debug(f"Removed {len(old_tracks)} old tracks from Re-ID database")