"""Camera input management for surveillance system."""

import cv2
import numpy as np
import threading
import time
import queue
import logging
from typing import Dict, Any, List, Tuple, Optional

class CameraManager:
    """Multi-camera input management."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize camera manager.
        
        Args:
            config: Camera configuration
        """
        self.config = config
        self.camera_configs = config.get('cameras', [])
        self.buffer_size = config.get('buffer_size', 10)
        
        # Initialize cameras
        self.cameras = {}
        self.camera_threads = {}
        self.camera_buffers = {}
        self.camera_stats = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def initialize_cameras(self) -> bool:
        """Initialize all cameras.
        
        Returns:
            Success flag
        """
        success = True
        
        for cam_config in self.camera_configs:
            cam_id = cam_config.get('id', f"cam_{len(self.cameras)}")
            source = cam_config.get('source', 0)
            
            # Initialize camera
            if not self.add_camera(cam_id, source, cam_config):
                success = False
                
        return success
        
    def add_camera(self, camera_id: str, source, config: Dict[str, Any]) -> bool:
        """Add camera to manager.
        
        Args:
            camera_id: Camera ID
            source: Camera source (index or URL)
            config: Camera configuration
            
        Returns:
            Success flag
        """
        try:
            # Initialize camera
            if isinstance(source, str) and source.lower().endswith(('.mp4', '.avi', '.mov')):
                cap = cv2.VideoCapture(source)
            else:
                try:
                    source_idx = int(source)
                    cap = cv2.VideoCapture(source_idx)
                except ValueError:
                    cap = cv2.VideoCapture(source)
                    
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera source: {source}")
                return False
                
            # Set camera properties
            width = config.get('width', 1280)
            height = config.get('height', 720)
            fps = config.get('fps', 30)
            
            if not isinstance(source, str) or not source.lower().endswith(('.mp4', '.avi', '.mov')):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                
            # Create frame buffer
            buffer = queue.Queue(maxsize=self.buffer_size)
            
            # Store camera
            self.cameras[camera_id] = cap
            self.camera_buffers[camera_id] = buffer
            self.camera_stats[camera_id] = {
                'source': source,
                'width': width,
                'height': height,
                'fps': fps,
                'frames_captured': 0,
                'actual_fps': 0,
                'start_time': time.time(),
                'last_frame_time': 0
            }
            
            # Start acquisition thread
            thread = threading.Thread(
                target=self._acquisition_thread,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            
            self.camera_threads[camera_id] = thread
            
            self.logger.info(f"Added camera {camera_id} from source {source}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding camera {camera_id}: {e}")
            return False
            
    def remove_camera(self, camera_id: str) -> bool:
        """Remove camera from manager.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Success flag
        """
        if camera_id not in self.cameras:
            return False
            
        # Stop acquisition thread
        if camera_id in self.camera_threads:
            thread = self.camera_threads[camera_id]
            thread.join(timeout=1.0)
            del self.camera_threads[camera_id]
            
        # Release camera
        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
            
        # Clear buffer
        if camera_id in self.camera_buffers:
            buffer = self.camera_buffers[camera_id]
            while not buffer.empty():
                try:
                    buffer.get_nowait()
                except queue.Empty:
                    break
            del self.camera_buffers[camera_id]
            
        # Remove stats
        if camera_id in self.camera_stats:
            del self.camera_stats[camera_id]
            
        self.logger.info(f"Removed camera {camera_id}")
        return True
        
    def get_frame(self, camera_id: str, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Get frame from camera.
        
        Args:
            camera_id: Camera ID
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (frame, timestamp) or None if timeout
        """
        if camera_id not in self.camera_buffers:
            return None
            
        try:
            return self.camera_buffers[camera_id].get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_all_frames(self, timeout: float = 0.1) -> Dict[str, Tuple[np.ndarray, float]]:
        """Get frames from all cameras.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Dictionary of camera_id -> (frame, timestamp)
        """
        frames = {}
        
        for camera_id in self.camera_buffers:
            frame_data = self.get_frame(camera_id, timeout)
            if frame_data is not None:
                frames[camera_id] = frame_data
                
        return frames
        
    def stop_all(self) -> None:
        """Stop all cameras."""
        camera_ids = list(self.cameras.keys())
        
        for camera_id in camera_ids:
            self.remove_camera(camera_id)
            
        self.logger.info("Stopped all cameras")
        
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get camera statistics.
        
        Returns:
            Dictionary of camera statistics
        """
        stats = {}
        
        for camera_id, camera_stat in self.camera_stats.items():
            stats[camera_id] = {
                'source': camera_stat['source'],
                'width': camera_stat['width'],
                'height': camera_stat['height'],
                'target_fps': camera_stat['fps'],
                'actual_fps': camera_stat['actual_fps'],
                'frames_captured': camera_stat['frames_captured'],
                'buffer_used': self.camera_buffers[camera_id].qsize() if camera_id in self.camera_buffers else 0,
                'buffer_size': self.buffer_size,
                'last_frame_time': camera_stat['last_frame_time']
            }
            
        return stats
        
    def _acquisition_thread(self, camera_id: str) -> None:
        """Frame acquisition thread for camera.
        
        Args:
            camera_id: Camera ID
        """
        if camera_id not in self.cameras:
            return
            
        cap = self.cameras[camera_id]
        buffer = self.camera_buffers[camera_id]
        stats = self.camera_stats[camera_id]
        
        start_time = time.time()
        frame_count = 0
        
        while camera_id in self.cameras:
            try:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    # End of video or error
                    if isinstance(stats['source'], str) and stats['source'].lower().endswith(('.mp4', '.avi', '.mov')):
                        # End of video file, restart
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.logger.info(f"Video file restarted for camera {camera_id}")
                        continue
                    else:
                        # Camera error
                        self.logger.error(f"Error reading frame from camera {camera_id}")
                        time.sleep(1.0)
                        continue
                        
                # Get timestamp
                timestamp = time.time()
                
                # Update stats
                frame_count += 1
                stats['frames_captured'] += 1
                stats['last_frame_time'] = timestamp
                
                elapsed = timestamp - start_time
                if elapsed >= 1.0:
                    stats['actual_fps'] = frame_count / elapsed
                    frame_count = 0
                    start_time = timestamp
                    
                # Add to buffer
                try:
                    buffer.put((frame, timestamp), timeout=0.1)
                except queue.Full:
                    # Buffer full, skip frame
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error in camera {camera_id} acquisition: {e}")
                time.sleep(0.1)