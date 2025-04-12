# optimization/roi_manager.py
import cv2
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import logging

class ROIManager:
    """Region of interest management for facial detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ROI manager.
        
        Args:
            config: ROI configuration
        """
        self.config = config
        self.grid_size = config.get('grid_size', (4, 4))
        self.attention_decay = config.get('attention_decay', 0.9)
        
        # Initialize ROI grid
        self.attention_map = None
        self.frame_shape = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def initialize(self, frame_shape: Tuple[int, int, int]) -> None:
        """Initialize ROI grid for frame size.
        
        Args:
            frame_shape: Frame shape (height, width, channels)
        """
        # Store frame shape
        self.frame_shape = frame_shape
        
        # Initialize attention map
        self.attention_map = np.ones(self.grid_size, dtype=np.float32)
        
        # Initialize frame counter for startup phase
        self.frame_count = 0
        
        self.logger.debug(f"ROI manager initialized with frame shape: {frame_shape}")
            
    def update(self, detections: List[Dict[str, Any]]) -> None:
        """Update ROI based on detections.
        
        Args:
            detections: List of detection information
        """
        if self.frame_shape is None:
            return
            
        # Apply decay to current attention map
        self.attention_map *= self.attention_decay
        
        # Increase attention for cells with detections
        for det in detections:
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                
                # Convert to grid coordinates
                grid_x1 = int(x1 / self.frame_shape[1] * self.grid_size[1])
                grid_y1 = int(y1 / self.frame_shape[0] * self.grid_size[0])
                grid_x2 = int(x2 / self.frame_shape[1] * self.grid_size[1])
                grid_y2 = int(y2 / self.frame_shape[0] * self.grid_size[0])
                
                # Clamp to valid range
                grid_x1 = max(0, min(grid_x1, self.grid_size[1] - 1))
                grid_y1 = max(0, min(grid_y1, self.grid_size[0] - 1))
                grid_x2 = max(0, min(grid_x2, self.grid_size[1] - 1))
                grid_y2 = max(0, min(grid_y2, self.grid_size[0] - 1))
                
                # Update attention map
                for y in range(grid_y1, grid_y2 + 1):
                    for x in range(grid_x1, grid_x2 + 1):
                        self.attention_map[y, x] += 1.0
                        
        # Normalize attention map to 0-1 range
        if np.max(self.attention_map) > 0:
            self.attention_map = self.attention_map / np.max(self.attention_map)
            
    def get_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get ROI mask for frame."""
        try:
            # Initialize ROI if needed - important for first frame or if dimensions change
            if self.frame_shape is None or self.frame_shape[0] != frame.shape[0] or self.frame_shape[1] != frame.shape[1]:
                self.initialize(frame.shape)
                
            # Create mask with exactly the same shape as the input frame
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # If attention map is not initialized, return a full mask
            if self.attention_map is None:
                mask.fill(255)  # Full attention
                return mask
                
            # Convert attention map to mask
            cell_h = max(1, h // self.grid_size[0])
            cell_w = max(1, w // self.grid_size[1])
            
            for y in range(min(self.grid_size[0], h)):
                for x in range(min(self.grid_size[1], w)):
                    if self.attention_map[y, x] > 0.2:  # Threshold for attention
                        y1 = min(h-1, y * cell_h)
                        x1 = min(w-1, x * cell_w)
                        y2 = min(h, (y + 1) * cell_h)
                        x2 = min(w, (x + 1) * cell_w)
                        
                        if y1 >= y2 or x1 >= x2:
                            continue
                            
                        # Set cell to active
                        value = max(1, min(255, int(self.attention_map[y, x] * 255)))
                        mask[y1:y2, x1:x2] = value
                        
            # For the first few frames or if no attention, use full frame
            if np.max(mask) == 0 or self.frame_count < 10:
                mask.fill(255)
                
            self.frame_count += 1
            return mask
        except Exception as e:
            self.logger.error(f"Error generating ROI mask: {e}")
            # Return a full mask in case of error
            return np.full(frame.shape[:2], 255, dtype=np.uint8)