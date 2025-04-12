"""Attention-based prioritization for frame processing."""

import numpy as np
import cv2
import time
from typing import Dict, Any, List, Tuple, Optional
import logging

class AttentionManager:
    """Attention-based prioritization for frame processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize attention manager.
        
        Args:
            config: Attention configuration
        """
        self.config = config
        self.motion_weight = config.get('motion_weight', 0.6)
        self.history_weight = config.get('history_weight', 0.3)
        self.prediction_weight = config.get('prediction_weight', 0.1)
        self.decay_factor = config.get('decay_factor', 0.95)
        self.grid_size = config.get('grid_size', (8, 8))
        
        # Initialize attention maps
        self.activity_map = None
        self.history_map = None
        self.prediction_map = None
        self.last_frame = None
        self.frame_shape = None
        
        # Historical patterns
        self.historical_patterns = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def initialize(self, frame_shape: Tuple[int, int, int]) -> None:
        """Initialize attention maps for frame size.
        
        Args:
            frame_shape: Frame shape (height, width, channels)
        """
        self.frame_shape = frame_shape
        self.activity_map = np.ones(self.grid_size, dtype=np.float32)
        self.history_map = np.zeros(self.grid_size, dtype=np.float32)
        self.prediction_map = np.zeros(self.grid_size, dtype=np.float32)
        self.historical_patterns = np.zeros(self.grid_size, dtype=np.float32)
        
    def update(self, frame: np.ndarray, 
              detections: Optional[List[Dict[str, Any]]] = None) -> None:
        """Update attention maps based on frame and detections.
        
        Args:
            frame: Current frame
            detections: List of detection information
        """
        if self.frame_shape is None or self.frame_shape[0] != frame.shape[0] or self.frame_shape[1] != frame.shape[1]:
            self.initialize(frame.shape)
            
        # Apply decay to current attention maps
        self.activity_map *= self.decay_factor
        self.history_map *= 0.99  # Slower decay for history
        
        # Calculate motion
        if self.last_frame is not None:
            # Convert to grayscale
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray_current, gray_last)
            
            # Threshold
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Calculate motion per grid cell
            h, w = motion_mask.shape
            cell_h = h // self.grid_size[0]
            cell_w = w // self.grid_size[1]
            
            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    y1 = y * cell_h
                    x1 = x * cell_w
                    y2 = min(h, (y + 1) * cell_h)
                    x2 = min(w, (x + 1) * cell_w)
                    
                    # Calculate motion percentage in cell
                    cell_motion = np.mean(motion_mask[y1:y2, x1:x2]) / 255.0
                    
                    # Update activity map
                    self.activity_map[y, x] += cell_motion * self.motion_weight
            
        # Update history based on detections
        if detections is not None:
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
                    
                    # Update history map
                    for y in range(grid_y1, grid_y2 + 1):
                        for x in range(grid_x1, grid_x2 + 1):
                            self.history_map[y, x] += 1.0
                            
                            # Update historical patterns (long-term memory)
                            self.historical_patterns[y, x] = self.historical_patterns[y, x] * 0.999 + 0.001
        
        # Update prediction map based on historical patterns
        self.prediction_map = self.historical_patterns.copy()
        
        # Add neighboring spread activation
        kernel = np.array([[0.1, 0.2, 0.1], 
                          [0.2, 0.0, 0.2], 
                          [0.1, 0.2, 0.1]])
                          
        # Apply convolution for spread activation
        prediction_spread = cv2.filter2D(self.prediction_map, -1, kernel)
        self.prediction_map += prediction_spread * 0.5
        
        # Store current frame for next update
        self.last_frame = frame.copy()
        
    def get_attention_map(self) -> np.ndarray:
        """Get combined attention map.
        
        Returns:
            Combined attention map (0-1 float32)
        """
        if self.activity_map is None:
            return np.ones(self.grid_size, dtype=np.float32)
            
        # Combine maps
        combined_map = (
            self.activity_map * self.motion_weight +
            self.history_map * self.history_weight +
            self.prediction_map * self.prediction_weight
        )
        
        # Normalize
        if np.max(combined_map) > 0:
            combined_map = combined_map / np.max(combined_map)
            
        return combined_map
        
    def get_frame_priority(self, frame: np.ndarray) -> float:
        """Calculate frame priority based on attention.
        
        Args:
            frame: Input frame
            
        Returns:
            Priority score (0-10)
        """
        # Update attention maps
        self.update(frame)
        
        # Calculate priority
        combined_attention = self.get_attention_map()
        
        # Base priority on sum of attention
        attention_sum = np.sum(combined_attention)
        
        # Convert to priority score (0-10)
        max_possible = np.prod(self.grid_size)
        priority = 10.0 * min(1.0, attention_sum / max_possible)
        
        return priority
        
    def get_attention_heatmap(self, frame_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Get visualization of attention heatmap.
        
        Args:
            frame_shape: Target frame shape (height, width)
            
        Returns:
            Visualization image
        """
        combined_map = self.get_attention_map()
        
        if combined_map is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # Use current shape if not specified
        if frame_shape is None:
            if self.frame_shape is None:
                return np.zeros((100, 100, 3), dtype=np.uint8)
            frame_shape = (self.frame_shape[0], self.frame_shape[1])
            
        # Create heatmap
        heatmap = cv2.resize(combined_map, (frame_shape[1], frame_shape[0]), 
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