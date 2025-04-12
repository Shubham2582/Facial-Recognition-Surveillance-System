"""Frame prioritization for efficient processing."""

import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import queue

class FramePrioritizer:
    """Frame prioritization for efficient processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize frame prioritizer.
        
        Args:
            config: Prioritization configuration
        """
        self.config = config
        self.motion_weight = config.get('motion_weight', 0.7)
        self.prediction_weight = config.get('prediction_weight', 0.3)
        self.boost_interval = config.get('boost_interval', 30)
        self.queue_size = config.get('queue_size', 10)
        
        # Initialize priority queue
        self.frame_queue = queue.PriorityQueue(maxsize=self.queue_size)
        
        # Initialize state
        self.frame_count = 0
        self.last_frame = None
        self.attention_manager = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def set_attention_manager(self, attention_manager) -> None:
        """Set attention manager for more intelligent prioritization.
        
        Args:
            attention_manager: AttentionManager instance
        """
        self.attention_manager = attention_manager
        
    def calculate_priority(self, frame: np.ndarray) -> float:
        """Calculate frame priority score.
        
        Args:
            frame: Input frame
            
        Returns:
            Priority score (0-10)
        """
        self.frame_count += 1
        
        # Use attention manager if available
        if self.attention_manager is not None:
            return self.attention_manager.get_frame_priority(frame)
            
        # Calculate motion score
        motion_score = 0.0
        if self.last_frame is not None:
            # Convert to grayscale
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray_current, gray_last)
            
            # Calculate motion percentage
            motion_score = np.mean(diff) / 255.0
            
        # Calculate predicted activity (placeholder)
        predicted_activity = 0.2  # Default value
        
        # Add periodic boost to ensure processing of low-activity frames
        periodic_boost = 3.0 if self.frame_count % self.boost_interval == 0 else 0.0
        
        # Calculate final priority
        priority = (
            motion_score * self.motion_weight * 10.0 +
            predicted_activity * self.prediction_weight * 10.0 +
            periodic_boost
        )
        
        # Clamp to valid range
        priority = max(1.0, min(10.0, priority))
        
        # Store current frame for next calculation
        self.last_frame = frame.copy()
        
        return priority
        
    def put(self, frame: np.ndarray, timestamp: float) -> bool:
        """Put frame into priority queue.
        
        Args:
            frame: Input frame
            timestamp: Frame timestamp
            
        Returns:
            Success flag
        """
        # Calculate priority
        priority = self.calculate_priority(frame)
        
        # Create queue item
        item = (10.0 - priority, timestamp, frame)  # Lower value = higher priority
        
        try:
            self.frame_queue.put(item, block=False)
            return True
        except queue.Full:
            return False
            
    def get(self, timeout: Optional[float] = 0.1) -> Optional[Tuple[np.ndarray, float, float]]:
        """Get highest priority frame from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (frame, timestamp, priority) or None if queue is empty
        """
        try:
            neg_priority, timestamp, frame = self.frame_queue.get(timeout=timeout)
            priority = 10.0 - neg_priority
            return frame, timestamp, priority
        except queue.Empty:
            return None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get prioritizer statistics.
        
        Returns:
            Prioritizer statistics
        """
        return {
            'queue_size': self.frame_queue.qsize(),
            'frame_count': self.frame_count
        }