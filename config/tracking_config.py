"""Tracking configuration."""

# ByteTrack parameters
TRACKING_CONFIG = {
    'track_threshold': 0.5,  # Tracking confidence threshold
    'track_buffer': 30,  # Frames to keep track after disappearance
    'match_threshold': 0.8,  # IoU matching threshold
    'frame_rate': 30,  # Frame rate for Kalman filter
}

# Re-ID parameters
REID_CONFIG = {
    'reid_interval': 10,  # Check for re-ID every N frames for lost tracks
    'reid_threshold': 0.5,  # Similarity threshold for re-ID
}

# Track priority calculation
PRIORITY_CONFIG = {
    'watchlist_boost': 5,  # Priority boost for watchlist matches
    'recognition_boost': 2,  # Priority boost for recognized faces
    'size_boost': 1,  # Priority boost for larger faces
    'age_boost': 0.5,  # Priority boost per second of track age
}