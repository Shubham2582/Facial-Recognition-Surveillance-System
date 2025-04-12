"""Frame acquisition and management for video processing."""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import Dict, Any, Optional, Tuple, List

class FrameManager:
    """Frame acquisition and management for video processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize frame manager.
        
        Args:
            config: Frame manager configuration
        """
        self.config = config
        self.source = config.get('source', 0)
        self.width = config.get('width', 1280)
        self.height = config.get('height', 720)
        self.fps = config.get('fps', 30)
        self.buffer_size = config.get('buffer_size', 10)
        
        # Initialize frame buffer
        self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
        
        # Initialize capture
        self.cap = None
        self.running = False
        self.thread = None
        
        # Initialize stats
        self.frame_count = 0
        self.start_time = 0
        self.actual_fps = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> bool:
        """Start frame acquisition.
        
        Returns:
            Success flag
        """
        try:
            # Open capture
            if isinstance(self.source, str) and self.source.lower().endswith(('.mp4', '.avi', '.mov')):
                self.cap = cv2.VideoCapture(self.source)
            else:
                try:
                    source_idx = int(self.source)
                    self.cap = cv2.VideoCapture(source_idx)
                except ValueError:
                    self.cap = cv2.VideoCapture(self.source)
                    
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                return False
                
            # Set properties for camera sources
            if not isinstance(self.source, str) or not self.source.lower().endswith(('.mp4', '.avi', '.mov')):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
            # Start acquisition thread
            self.running = True
            self.thread = threading.Thread(target=self._acquisition_thread, daemon=True)
            self.thread.start()
            
            self.logger.info(f"Frame acquisition started from source: {self.source}")
            return True
        except Exception as e:
            self.logger.error(f"Error starting frame acquisition: {e}")
            return False
            
    def stop(self) -> None:
        """Stop frame acquisition."""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
                
        self.logger.info("Frame acquisition stopped")
        
    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, float]]:
        """Get frame from buffer.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (frame, timestamp) or None if timeout
        """
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _acquisition_thread(self) -> None:
        """Frame acquisition thread."""
        self.start_time = time.time()
        self.frame_count = 0
        
        while self.running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    # End of video or error
                    if isinstance(self.source, str) and self.source.lower().endswith(('.mp4', '.avi', '.mov')):
                        # End of video file, restart
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.logger.info("Video file restarted")
                        continue
                    else:
                        # Camera error
                        self.logger.error("Error reading frame from camera")
                        time.sleep(1.0)
                        continue
                        
                # Get timestamp
                timestamp = time.time()
                
                # Update stats
                self.frame_count += 1
                elapsed = timestamp - self.start_time
                if elapsed >= 1.0:
                    self.actual_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = timestamp
                    
                # Add to buffer
                try:
                    self.frame_buffer.put((frame, timestamp), timeout=0.1)
                except queue.Full:
                    # Buffer full, skip frame
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error in frame acquisition: {e}")
                time.sleep(0.1)
                
    def get_stats(self) -> Dict[str, Any]:
        """Get acquisition statistics.
        
        Returns:
            Acquisition statistics
        """
        return {
            'source': self.source,
            'width': self.width,
            'height': self.height,
            'target_fps': self.fps,
            'actual_fps': self.actual_fps,
            'buffer_size': self.buffer_size,
            'buffer_used': self.frame_buffer.qsize()
        }