"""Model warm-up utilities for optimized inference."""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional

def warmup_onnx_model(session, input_shape: Dict[str, Any], 
                    input_name: str, iterations: int = 3) -> None:
    """Warm up an ONNX model to prevent cold-start delays.
    
    Args:
        session: ONNX session
        input_shape: Input shape dictionary
        input_name: Input tensor name
        iterations: Number of warm-up iterations
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Warming up ONNX model with shape {input_shape}")
    
    # Create dummy input
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    
    # Run warm-up iterations
    start_time = time.time()
    try:
        for i in range(iterations):
            _ = session.run(None, {input_name: dummy_input})
    except Exception as e:
        logger.error(f"Error during model warm-up: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"Model warm-up complete in {elapsed:.3f}s")
    
def warmup_detector(detector, frame_size: Optional[tuple] = None, iterations: int = 3) -> None:
    """Warm up face detector.
    
    Args:
        detector: Face detector instance
        frame_size: Frame size for warm-up (width, height)
        iterations: Number of warm-up iterations
    """
    logger = logging.getLogger(__name__)
    logger.info("Warming up face detector")
    
    # Use default size if not specified
    if frame_size is None:
        frame_size = (640, 480)
        
    # Create dummy frame
    dummy_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Run warm-up iterations
    start_time = time.time()
    try:
        for i in range(iterations):
            _ = detector.detect(dummy_frame)
    except Exception as e:
        logger.error(f"Error during detector warm-up: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"Detector warm-up complete in {elapsed:.3f}s")
    
def warmup_recognizer(recognizer, face_size: Optional[tuple] = None, iterations: int = 3) -> None:
    """Warm up face recognizer.
    
    Args:
        recognizer: Face recognizer instance
        face_size: Face size for warm-up (width, height)
        iterations: Number of warm-up iterations
    """
    logger = logging.getLogger(__name__)
    logger.info("Warming up face recognizer")
    
    # Use default size if not specified
    if face_size is None:
        face_size = (112, 112)
        
    # Create dummy face
    dummy_face = np.zeros((face_size[1], face_size[0], 3), dtype=np.uint8)
    
    # Run warm-up iterations
    start_time = time.time()
    try:
        for i in range(iterations):
            _ = recognizer.get_embedding(dummy_face)
    except Exception as e:
        logger.error(f"Error during recognizer warm-up: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"Recognizer warm-up complete in {elapsed:.3f}s")
    
def warmup_pipeline(engine, frame_size: Optional[tuple] = None, iterations: int = 3) -> None:
    """Warm up entire processing pipeline.
    
    Args:
        engine: SurveillanceEngine instance
        frame_size: Frame size for warm-up (width, height)
        iterations: Number of warm-up iterations
    """
    logger = logging.getLogger(__name__)
    logger.info("Warming up processing pipeline")
    
    # Use default size if not specified
    if frame_size is None:
        frame_size = (640, 480)
        
    # Create dummy frame
    dummy_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Run warm-up iterations
    start_time = time.time()
    try:
        for i in range(iterations):
            _ = engine.process_frame(dummy_frame)
    except Exception as e:
        logger.error(f"Error during pipeline warm-up: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"Pipeline warm-up complete in {elapsed:.3f}s")