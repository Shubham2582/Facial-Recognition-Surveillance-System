# optimization/attention_manager.py
"""Attention-based frame prioritization system."""
import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class AttentionManager:
    """Manages attention mechanisms for optimizing processing resources."""
    
    def __init__(self, frame_size: Tuple[int, int], grid_size: Tuple[int, int] = (4, 4)):
        """
        Initialize attention manager.
        
        Args:
            frame_size: Frame dimensions (width, height)
            grid_size: Attention grid size (cols, rows)
        """
        self.frame_size = frame_size
        self.grid_size = grid_size
        
        # Attention map
        self.attention_map = np.ones(grid_size, dtype=np.float32)
        
        # Previous frame for motion detection
        self.prev_frame = None
        
        # Activity history
        self.activity_history = np.zeros(grid_size, dtype=np.float32)
        
        # Cell dimensions
        self.cell_width = frame_size[0] // grid_size[0]
        self.cell_height = frame_size[1] // grid_size[1]
        
        # Update counter
        self.update_count = 0
        
        logger.info(f"Attention manager initialized with grid {grid_size}")
        
    def update(self, frame: np.ndarray, face_detections: List[List[float]] = None):
        """
        Update attention map with new frame and detections.
        
        Args:
            frame: Current frame
            face_detections: Optional list of face detections [x1, y1, x2, y2, ...]
        """
        self.update_count += 1
        
        # Motion-based attention
        self._update_motion_attention(frame)
        
        # Detection-based attention
        if face_detections:
            self._update_detection_attention(face_detections)
            
        # Apply decay
        self._apply_attention_decay()
        
        # Apply spatial spreading
        self._apply_spatial_spreading()
        
    def _update_motion_attention(self, frame: np.ndarray):
        """
        Update attention map based on motion.
        
        Args:
            frame: Current frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Skip if first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            return
            
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray, self.prev_frame)
        _, motion_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
        
        # Process each cell
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                # Cell region
                y1 = y * self.cell_height
                y2 = min((y + 1) * self.cell_height, self.frame_size[1])
                x1 = x * self.cell_width
                x2 = min((x + 1) * self.cell_width, self.frame_size[0])
                
                # Calculate motion in cell
                cell_motion = motion_mask[y1:y2, x1:x2]
                if cell_motion.size > 0:
                    motion_ratio = np.count_nonzero(cell_motion) / cell_motion.size
                    
                    # Update attention map
                    self.attention_map[y, x] += motion_ratio * 2.0
                    
        # Update previous frame
        self.prev_frame = gray
        
    def _update_detection_attention(self, detections: List[List[float]]):
        """
        Update attention map based on face detections.
        
        Args:
            detections: List of face detections [x1, y1, x2, y2, ...]
        """
        # Process each detection
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[:4])
            
            # Calculate affected cells
            cell_x1 = max(0, x1 // self.cell_width)
            cell_y1 = max(0, y1 // self.cell_height)
            cell_x2 = min(self.grid_size[0] - 1, x2 // self.cell_width)
            cell_y2 = min(self.grid_size[1] - 1, y2 // self.cell_height)
            
            # Update attention for affected cells
            for y in range(cell_y1, cell_y2 + 1):
                for x in range(cell_x1, cell_x2 + 1):
                    # Add detection boost
                    self.attention_map[y, x] += 3.0
                    
                    # Update activity history
                    self.activity_history[y, x] += 0.2
                    
    def _apply_attention_decay(self):
        """Apply decay to attention map."""
        # Decay factor
        decay = 0.95
        
        # Apply decay
        self.attention_map *= decay
        self.activity_history *= 0.99
        
        # Add periodic boost to ensure full frame coverage
        if self.update_count % 30 == 0:
            self.attention_map += 0.2
            
    def _apply_spatial_spreading(self):
        """Apply spatial spreading to attention map."""
        # Copy current map
        attention_copy = self.attention_map.copy()
        
        # Apply spreading
        kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.0, 0.2],
            [0.1, 0.2, 0.1]
        ])
        
        # Convolution (spreading)
        for y in range(1, self.grid_size[1] - 1):
            for x in range(1, self.grid_size[0] - 1):
                spreading = 0.0
                
                # Apply kernel
                for ky in range(3):
                    for kx in range(3):
                        if kernel[ky, kx] > 0:
                            spreading += attention_copy[y + ky - 1, x + kx - 1] * kernel[ky, kx]
                            
                self.attention_map[y, x] += spreading
                
    def get_priority_regions(self, count: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        Get high-priority regions based on attention map.
        
        Args:
            count: Number of regions to return
            
        Returns:
            List of regions as (x1, y1, x2, y2)
        """
        # Flatten attention map
        flat_attention = self.attention_map.flatten()
        
        # Get indices of top cells
        top_indices = np.argsort(flat_attention)[-count:]
        
        # Convert to 2D indices
        top_cells = [
            (idx % self.grid_size[0], idx // self.grid_size[0])
            for idx in top_indices
        ]
        
        # Convert to regions
        regions = []
        for x, y in top_cells:
            x1 = x * self.cell_width
            y1 = y * self.cell_height
            x2 = min((x + 1) * self.cell_width, self.frame_size[0])
            y2 = min((y + 1) * self.cell_height, self.frame_size[1])
            
            regions.append((x1, y1, x2, y2))
            
        return regions
        
    def get_frame_priority(self, frame: np.ndarray) -> float:
        """
        Calculate overall frame priority.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame priority (1-10)
        """
        # Update attention map
        self.update(frame)
        
        # Calculate priority based on attention map
        max_attention = np.max(self.attention_map)
        mean_attention = np.mean(self.attention_map)
        
        # Combine max and mean
        priority = (max_attention * 0.7 + mean_attention * 0.3) * 2.0
        
        # Add periodic boost
        if self.update_count % 10 == 0:
            priority += 2.0
            
        # Ensure valid range
        return max(1.0, min(10.0, priority))
        
    def get_attention_heatmap(self) -> np.ndarray:
        """
        Get visualization of attention map.
        
        Returns:
            Colorized attention heatmap
        """
        # Normalize attention map
        normalized = self.attention_map.copy()
        min_val = np.min(normalized)
        max_val = np.max(normalized)
        
        if max_val > min_val:
            normalized = (normalized - min_val) / (max_val - min_val)
        else:
            normalized.fill(0.5)
            
        # Resize to frame size
        heatmap = cv2.resize(
            normalized, 
            (self.frame_size[0], self.frame_size[1]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply colormap
        heatmap_colored = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        return heatmap_colored