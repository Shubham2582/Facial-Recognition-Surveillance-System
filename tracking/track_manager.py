"""Track lifecycle management for face tracking."""

import time
import logging
from typing import Dict, Any, List, Optional

class TrackManager:
    """Track lifecycle management for face tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize track manager.
        
        Args:
            config: Track manager configuration
        """
        self.config = config
        self.max_tracks = config.get('max_tracks', 100)
        self.track_timeout = config.get('track_timeout', 5.0)  # seconds
        self.track_cleanup_interval = config.get('track_cleanup_interval', 10.0)  # seconds
        
        # Initialize track database
        self.tracks = {}  # track_id -> track data
        self.next_track_id = 0
        
        # Initialize timestamps
        self.last_cleanup = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def create_track(self, detection: Dict[str, Any]) -> int:
        """Create new track for detection.
        
        Args:
            detection: Detection information
            
        Returns:
            Track ID
        """
        # Check if cleanup needed
        self._check_cleanup()
        
        # Generate track ID
        track_id = self.next_track_id
        self.next_track_id += 1
        
        # Create track
        self.tracks[track_id] = {
            'track_id': track_id,
            'bbox': detection.get('bbox', None),
            'detection': detection,
            'created_at': time.time(),
            'last_updated': time.time(),
            'frames_tracked': 1,
            'lost': False,
            'lost_count': 0,
            'priority': 5  # Default priority
        }
        
        self.logger.debug(f"Created track {track_id}")
        return track_id
        
    def update_track(self, track_id: int, detection: Dict[str, Any]) -> bool:
        """Update existing track with new detection.
        
        Args:
            track_id: Track ID
            detection: Detection information
            
        Returns:
            Success flag
        """
        if track_id not in self.tracks:
            return False
            
        # Update track
        track = self.tracks[track_id]
        track['bbox'] = detection.get('bbox', track['bbox'])
        track['detection'] = detection
        track['last_updated'] = time.time()
        track['frames_tracked'] += 1
        track['lost'] = False
        track['lost_count'] = 0
        
        return True
        
    def mark_lost(self, track_id: int) -> bool:
        """Mark track as lost.
        
        Args:
            track_id: Track ID
            
        Returns:
            Success flag
        """
        if track_id not in self.tracks:
            return False
            
        # Mark as lost
        track = self.tracks[track_id]
        track['lost'] = True
        track['lost_count'] += 1
        
        return True
        
    def get_track(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get track information.
        
        Args:
            track_id: Track ID
            
        Returns:
            Track information or None if not found
        """
        return self.tracks.get(track_id, None)
        
    def remove_track(self, track_id: int) -> bool:
        """Remove track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Success flag
        """
        if track_id in self.tracks:
            del self.tracks[track_id]
            return True
        return False
        
    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Get all active tracks.
        
        Returns:
            List of track information
        """
        return list(self.tracks.values())
        
    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Get active (not lost) tracks.
        
        Returns:
            List of active track information
        """
        return [track for track in self.tracks.values() if not track['lost']]
        
    def get_lost_tracks(self) -> List[Dict[str, Any]]:
        """Get lost tracks.
        
        Returns:
            List of lost track information
        """
        return [track for track in self.tracks.values() if track['lost']]
        
    def _check_cleanup(self) -> None:
        """Check if track cleanup is needed."""
        now = time.time()
        
        # Check if cleanup interval has passed
        if now - self.last_cleanup < self.track_cleanup_interval:
            return
            
        self.last_cleanup = now
        
        # Find expired tracks
        expired_tracks = []
        for track_id, track in self.tracks.items():
            # Check if track is expired
            if now - track['last_updated'] > self.track_timeout:
                expired_tracks.append(track_id)
                
        # Remove expired tracks
        for track_id in expired_tracks:
            self.remove_track(track_id)
            
        # Check if we need to remove old tracks due to capacity
        if len(self.tracks) > self.max_tracks:
            # Sort tracks by last updated time
            sorted_tracks = sorted(self.tracks.items(), 
                                key=lambda x: x[1]['last_updated'])
            
            # Remove oldest tracks
            to_remove = len(self.tracks) - self.max_tracks
            for i in range(to_remove):
                if i < len(sorted_tracks):
                    self.remove_track(sorted_tracks[i][0])
                    
        if expired_tracks:
            self.logger.debug(f"Removed {len(expired_tracks)} expired tracks")