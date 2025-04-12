"""Embedding delta caching implementation."""

import time
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional

class EmbeddingCache:
    """Cache for face embeddings to avoid redundant computation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize embedding cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache_timeout = config.get('cache_timeout', 5.0)
        self.appearance_threshold = config.get('appearance_threshold', 0.8)
        self.priority_update_interval = config.get('priority_update_interval', 2.0)
        
        # Initialize cache
        self.cache = {}  # track_id -> (embedding, timestamp, histogram)
        
    def get(self, track_id: int) -> Optional[np.ndarray]:
        """Get cached embedding for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Cached embedding or None if not available
        """
        if track_id in self.cache:
            return self.cache[track_id][0]  # Return embedding
        return None
        
    def update(self, track_id: int, embedding: np.ndarray, face_image: np.ndarray):
        """Update cache with new embedding and face image.
        
        Args:
            track_id: Track ID
            embedding: Face embedding
            face_image: Face image for histogram calculation
        """
        hist = self._compute_histogram(face_image)
        self.cache[track_id] = (embedding, time.time(), hist)
        
    def should_update(self, track, current_face: Optional[np.ndarray] = None) -> bool:
        """Determine if embedding should be updated.
        
        Args:
            track: Track object
            current_face: Current face image
            
        Returns:
            True if embedding should be updated
        """
        track_id = track.track_id
        
        # Always compute new embedding for tracks not in cache
        if track_id not in self.cache:
            return True
            
        # Unpack cached data
        _, timestamp, cached_hist = self.cache[track_id]
        
        # Always update if cache is too old
        if time.time() - timestamp > self.cache_timeout:
            return True
            
        # If face image is provided, check for appearance change
        if current_face is not None:
            current_hist = self._compute_histogram(current_face)
            hist_similarity = self._compare_histograms(cached_hist, current_hist)
            
            # Update if appearance changed significantly
            if hist_similarity < self.appearance_threshold:
                return True
                
        # Update more frequently for high-priority tracks
        if track.priority >= 7 and time.time() - timestamp > self.priority_update_interval:
            return True
            
        return False
        
    def _compute_histogram(self, face_image: np.ndarray) -> np.ndarray:
        """Compute color histogram from face image.
        
        Args:
            face_image: Face image
            
        Returns:
            Color histogram
        """
        # Convert to HSV for better color representation
        try:
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            
            # Compute histogram with 8x8x8 bins
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                                [0, 180, 0, 256, 0, 256])
            
            # Normalize histogram
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist
        except Exception as e:
            # Return a placeholder histogram in case of error
            return np.ones((8, 8, 8), dtype=np.float32)
        
    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms and return similarity score.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            
        Returns:
            Similarity score (0-1)
        """
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)