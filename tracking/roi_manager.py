"""Region of interest management for facial detection."""

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
        self.frame_shape = frame_shape
        self.attention_map = np.ones(self.grid_size, dtype=np.float32)
        
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
        """Get ROI mask for frame.
        
        Args:
            frame: Input frame
            
        Returns:
            ROI mask (0-255 uint8)
        """
        # Initialize ROI if needed
        if self.frame_shape is None or self.frame_shape[0] != frame.shape[0] or self.frame_shape[1] != frame.shape[1]:
            self.initialize(frame.shape)
            
        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Convert attention map to mask
        h, w = frame.shape[:2]
        cell_h = h // self.grid_size[0]
        cell_w = w // self.grid_size[1]
        
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if self.attention_map[y, x] > 0.2:  # Threshold for attention
                    y1 = y * cell_h
                    x1 = x * cell_w
                    y2 = min(h, (y + 1) * cell_h)
                    x2 = min(w, (x + 1) * cell_w)
                    
                    # Set cell to active
                    value = int(self.attention_map[y, x] * 255)
                    mask[y1:y2, x1:x2] = value
                    
        return mask
        
    def get_attention_heatmap(self, frame_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Get visualization of attention heatmap.
        
        Args:
            frame_shape: Target frame shape (height, width)
            
        Returns:
            Visualization image
        """
        if self.attention_map is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # Use current shape if not specified
        if frame_shape is None:
            if self.frame_shape is None:
                return np.zeros((100, 100, 3), dtype=np.uint8)
            frame_shape = (self.frame_shape[0], self.frame_shape[1])
            
        # Create heatmap
        heatmap = cv2.resize(self.attention_map, (frame_shape[1], frame_shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Convert to color visualization
        heatmap_img = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        
        # Apply jet colormap
        for i in range(3):
            if i == 0:  # Blue channel
                heatmap_img[:, :, i] = np.clip(4 * (0.75 - heatmap), 0, 1) * 255
            elif i == 1:  # Green channel
                heatmap_img[:, :, i] = np.clip(4 * np.abs(heatmap - 0.5) - 1, 0, 1) * 255
            else:  # Red channel
                heatmap_img[:, :, i] = np.clip(4 * (heatmap - 0.25), 0, 1) * 255
                
        return heatmap_img