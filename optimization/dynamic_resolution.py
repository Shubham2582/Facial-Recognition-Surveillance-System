"""Dynamic resolution scaling for efficient processing."""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

class DynamicResolutionScaler:
    """Dynamic resolution scaling for efficient processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize dynamic resolution scaler.
        
        Args:
            config: Resolution scaling configuration
        """
        self.config = config
        self.bg_resolution = config.get('bg_resolution', (480, 320))
        self.face_padding = config.get('face_padding', 20)
        self.face_regions = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def process(self, frame: np.ndarray, 
               face_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Process frame with dynamic resolution scaling.
        
        Args:
            frame: Input frame
            face_regions: List of face bounding boxes (x1, y1, x2, y2)
            
        Returns:
            Tuple of (processed frame, scale factors)
        """
        if face_regions is not None:
            self.face_regions = face_regions
            
        # Get current frame dimensions
        h, w = frame.shape[:2]
        
        # Calculate background scale factors
        bg_h, bg_w = self.bg_resolution
        scale_x = bg_w / w
        scale_y = bg_h / h
        
        # Downsample the entire frame first
        bg_frame = cv2.resize(frame, (bg_w, bg_h))
        result = bg_frame.copy()
        
        # Process each face region at high resolution
        for region in self.face_regions:
            x1, y1, x2, y2 = region
            
            # Add padding
            x1 = max(0, x1 - self.face_padding)
            y1 = max(0, y1 - self.face_padding)
            x2 = min(w, x2 + self.face_padding)
            y2 = min(h, y2 + self.face_padding)
            
            # Extract high-res face region
            face_roi = frame[y1:y2, x1:x2]
            
            # Map to low-resolution coordinates
            lr_x1, lr_y1 = int(x1 * scale_x), int(y1 * scale_y)
            lr_x2, lr_y2 = int(x2 * scale_x), int(y2 * scale_y)
            
            # Ensure valid dimensions
            lr_w, lr_h = lr_x2 - lr_x1, lr_y2 - lr_y1
            if lr_w <= 0 or lr_h <= 0:
                continue
                
            # Resize face to fit low-res position
            try:
                resized_face = cv2.resize(face_roi, (lr_w, lr_h))
                # Replace region in low-res frame
                result[lr_y1:lr_y2, lr_x1:lr_x2] = resized_face
            except Exception as e:
                self.logger.error(f"Error in dynamic resolution scaling: {e}")
        
        # Return processed frame and scaling factors for coordinate mapping
        return result, (scale_x, scale_y)
        
    def set_face_regions(self, regions: List[Tuple[int, int, int, int]]) -> None:
        """Update face regions for next frame processing.
        
        Args:
            regions: List of face bounding boxes (x1, y1, x2, y2)
        """
        self.face_regions = regions
        
    def update_resolution(self, load_factor: float) -> None:
        """Adjust background resolution based on system load.
        
        Args:
            load_factor: System load factor (0-1)
        """
        if load_factor > 0.8:  # High load
            self.bg_resolution = (320, 240)
        elif load_factor > 0.6:  # Medium load
            self.bg_resolution = (480, 320)
        else:  # Low load
            self.bg_resolution = (640, 480)
            
        self.logger.debug(f"Updated background resolution to {self.bg_resolution}")