"""RetinaFace detector using InsightFace."""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import List, Tuple, Dict, Any, Optional

class FaceDetector:
    """Face detector using InsightFace's implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize face detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.det_size = config.get('det_size', (640, 640))
        self.model_name = config.get('model_name', 'buffalo_l')
        self.providers = config.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.ctx_id = config.get('ctx_id', 0)
        
        # Initialize InsightFace detector
        self.detector = FaceAnalysis(name=self.model_name, providers=self.providers)
        self.detector.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        
        # Warm up the model
        self._warmup()
        
    def _warmup(self):
        """Warm up the model with a dummy image."""
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detector.get(dummy_image)
        
    def detect(self, frame: np.ndarray, 
           roi: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Detect faces in frame."""
        # Apply ROI mask if provided
        if roi is not None:
            # Create a copy of the frame for processing
            processed_frame = frame.copy()
            
            # Apply ROI mask - set non-ROI areas to black
            try:
                processed_frame = cv2.bitwise_and(processed_frame, processed_frame, mask=roi)
            except Exception as e:
                print(f"Error applying ROI mask: {e}")
                processed_frame = frame  # Fallback to original frame
        else:
            processed_frame = frame
            
        # Detect faces using InsightFace
        faces = self.detector.get(processed_frame)
        
        # Convert to standard format for our system
        results = []
        for face in faces:
            # Extract bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Create standard result format
            result = {
                'bbox': (x1, y1, x2, y2),
                'landmarks': face.landmark,
                'embedding': face.normed_embedding,
                'det_score': face.det_score,
                'gender': face.gender,
                'age': face.age
            }
            results.append(result)
            
        # REMOVE THE PROBLEMATIC DEBUGGING CODE HERE
        # Just return the results directly
        return results
    
    def get_app(self):
        """Get the InsightFace app instance for direct use.
        
        Returns:
            FaceAnalysis instance
        """
        return self.detector