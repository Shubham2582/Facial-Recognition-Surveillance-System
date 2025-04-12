# config/detection_config.py
DETECTION_CONFIG = {
    'model_name': 'buffalo_l',
    'det_size': (640, 640),
    'providers': ['CPUExecutionProvider'],
    
    # Add proper configuration for resolution scaling
    'resolution_config': {
        'bg_resolution': (480, 320),
        'face_padding': 20,
    },
    
    # Add proper configuration for ROI management
    'roi_config': {
        'grid_size': (4, 4),
        'attention_decay': 0.9,
        'min_face_size': (30, 30),
    }
}