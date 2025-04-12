"""Face recognition configuration."""

# Recognition settings
RECOGNITION_CONFIG = {
    'threshold': 0.5,  # Similarity threshold for recognition
    'use_largest_face': True,  # Use largest face in database images
    'embedding_size': 512,  # Size of face embeddings
}

# Database building settings
DATABASE_CONFIG = {
    'image_size': (112, 112),  # Standard size for ArcFace
    'one_face_per_person': True,  # Use only one face per person in database
}

# Embedding cache configuration
CACHE_CONFIG = {
    'cache_timeout': 5.0,  # Seconds before forced cache update
    'appearance_threshold': 0.8,  # Histogram similarity threshold
    'priority_update_interval': 2.0,  # Update interval for high priority tracks
}