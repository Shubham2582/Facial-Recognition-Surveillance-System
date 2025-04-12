"""System-wide configuration parameters."""

# Enable all optimization features
# In config/system_config.py, modify FEATURE_FLAGS:
# Feature flags - Re-enable one at a time
FEATURE_FLAGS = {
    'use_dynamic_resolution': False,  # Try enabling this first
    'use_embedding_cache': True,      # Already enabled
    'use_memory_system': True,        # Already enabled
    'use_gpu_indexing': False,        # Keep disabled if CUDA isn't available
    'use_roi_management': False,      # Try enabling second
    'use_tracking': False,            # Try enabling third
    'use_incremental_learning': False, # Try enabling last
}



# Memory system configuration
MEMORY_CONFIG = {
    'stm_size': 100,  # Short-term memory size
    'mtm_max_variations': 5,  # Max appearance variations per identity
    'memory_decay_factor': 0.95,  # Time-based memory decay
}

# Camera configuration
CAMERA_CONFIG = {
    'input_width': 1280,
    'input_height': 720,
    'fps': 30,
}

# Database configuration
DATABASE_CONFIG = {
    'database_dir': r"F:\projects\face_recognition_system\celeb_images",
    'use_faiss': False,  # Set to False since CUDA is not available
    'index_type': 'flat',  # 'flat' or 'ivfpq'
}




# System parameters
SYSTEM_CONFIG = {
    'det_size': (640, 640),
    'ctx_id': 0,  # GPU ID
    'recognition_threshold': 0.5,
    'detection_interval': 1,  # Detect faces every N frames
    'max_faces_per_frame': 20,
    'log_level': 'INFO',
    'input_width': 1280,
    'input_height': 720,
    'fps': 30,
    'database_dir': r"F:\projects\face_recognition_system\celeb_images",
}
