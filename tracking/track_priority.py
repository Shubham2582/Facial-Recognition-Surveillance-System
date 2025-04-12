"""Track priority calculation for face tracking."""

import time
from typing import Dict, Any
import logging

class TrackPriority:
    """Track priority calculation for face tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize track priority calculator.
        
        Args:
            config: Priority configuration
        """
        self.config = config
        self.watchlist_boost = config.get('watchlist_boost', 5)
        self.recognition_boost = config.get('recognition_boost', 2)
        self.size_boost = config.get('size_boost', 1)
        self.age_boost = config.get('age_boost', 0.5)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def calculate_priority(self, track: Dict[str, Any]) -> float:
        """Calculate track priority score.
        
        Args:
            track: Track information
            
        Returns:
            Priority score (0-10)
        """
        # Start with base priority
        priority = 5.0
        
        # Extract detection information
        detection = track.get('detection', {})
        identity = detection.get('identity', 'Unknown')
        confidence = detection.get('confidence', 0.0)
        bbox = track.get('bbox', None)
        
        # Boost for recognized faces
        if identity != 'Unknown' and confidence > 0.6:
            priority += self.recognition_boost
            
        # Boost for watchlist matches
        if identity in self.config.get('watchlist', []):
            priority += self.watchlist_boost
            
        # Boost for larger faces (likely closer to camera)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face_area = (x2 - x1) * (y2 - y1)
            
            # Normalize area to 0-1 range assuming max size of 300x300
            normalized_area = min(1.0, face_area / 90000.0)
            
            # Add size boost
            priority += normalized_area * self.size_boost
            
        # Boost for track age (stability)
        if 'created_at' in track:
            track_age = time.time() - track['created_at']
            age_factor = min(1.0, track_age / 10.0)  # Cap at 10 seconds
            
            # Add age boost
            priority += age_factor * self.age_boost
            
        # Clamp priority to valid range
        priority = max(0.0, min(10.0, priority))
        
        return priority
        
    def get_priority_class(self, priority: float) -> str:
        """Get priority classification.
        
        Args:
            priority: Priority score
            
        Returns:
            Priority class string
        """
        if priority >= 8.0:
            return "high"
        elif priority >= 5.0:
            return "medium"
        else:
            return "low"
            
    def update_track_priority(self, track: Dict[str, Any]) -> Dict[str, Any]:
        """Update track with calculated priority.
        
        Args:
            track: Track information
            
        Returns:
            Updated track information
        """
        # Calculate priority
        priority = self.calculate_priority(track)
        
        # Update track
        track['priority'] = priority
        track['priority_class'] = self.get_priority_class(priority)
        
        return track
        
    def prioritize_tracks(self, tracks: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Update priorities for all tracks.
        
        Args:
            tracks: Dictionary of track_id -> track
            
        Returns:
            Updated tracks dictionary
        """
        for track_id, track in tracks.items():
            tracks[track_id] = self.update_track_priority(track)
            
        return tracks