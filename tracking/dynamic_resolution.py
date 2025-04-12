"""Dynamic resolution tracking for face tracking."""
import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DynamicResolutionTracker:
    """
    Tracks faces with dynamic resolution adaptation.
    Adjusts tracking resolution based on face size and importance.
    """
    
    def __init__(self, high_res: Tuple[int, int] = (1080, 1920),
               low_res: Tuple[int, int] = (480, 854),
               mid_res: Tuple[int, int] = (720, 1280)):
        """
        Initialize dynamic resolution tracker.
        
        Args:
            high_res: High resolution dimensions (height, width)
            low_res: Low resolution dimensions (height, width)
            mid_res: Medium resolution dimensions (height, width)
        """
        self.high_res = high_res
        self.low_res = low_res
        self.mid_res = mid_res
        
        # Resolution adjustment factor
        self.resolution_factor = 1.0
        
        # Performance metrics
        self.metrics = {
            'high_res_usage': 0,
            'mid_res_usage': 0,
            'low_res_usage': 0,
            'processing_time': 0.0,
            'resolution_switches': 0,
        }
        
        # Adaptive thresholds
        self.min_face_size_for_high_res = 100
        self.max_face_count_for_high_res = 5
        
        logger.info(f"Dynamic resolution tracker initialized with resolutions: "
                  f"high={high_res}, mid={mid_res}, low={low_res}")
        
    def adjust_resolution(self, frame: np.ndarray, face_detections: List[List[float]],
                        system_load: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Adjust frame resolution based on faces and system load.
        
        Args:
            frame: Input frame
            face_detections: List of face detections [x1, y1, x2, y2, ...]
            system_load: System load factor (0-1)
            
        Returns:
            Tuple of (resized_frame, resolution_factor)
        """
        start_time = time.time()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Count faces and calculate average size
        face_count = len(face_detections)
        
        if face_count > 0:
            face_sizes = [
                (det[2] - det[0]) * (det[3] - det[1])
                for det in face_detections
                if len(det) >= 4
            ]
            avg_face_size = sum(face_sizes) / len(face_sizes) if face_sizes else 0
            max_face_size = max(face_sizes) if face_sizes else 0
        else:
            avg_face_size = 0
            max_face_size = 0
            
        # Determine target resolution based on faces and system load
        prev_factor = self.resolution_factor
        
        if face_count == 0:
            # No faces - use low resolution
            target_res = self.low_res
            self.resolution_factor = min(self.low_res[0] / max(1, height),
                                      self.low_res[1] / max(1, width))
            self.metrics['low_res_usage'] += 1
            
        elif face_count > self.max_face_count_for_high_res or system_load > 0.8:
            # Many faces or high system load - use medium resolution
            target_res = self.mid_res
            self.resolution_factor = min(self.mid_res[0] / max(1, height),
                                      self.mid_res[1] / max(1, width))
            self.metrics['mid_res_usage'] += 1
            
        elif max_face_size > self.min_face_size_for_high_res * self.min_face_size_for_high_res:
            # Large faces - use high resolution
            target_res = self.high_res
            self.resolution_factor = min(self.high_res[0] / max(1, height),
                                      self.high_res[1] / max(1, width))
            self.metrics['high_res_usage'] += 1
            
        else:
            # Default - use medium resolution
            target_res = self.mid_res
            self.resolution_factor = min(self.mid_res[0] / max(1, height),
                                      self.mid_res[1] / max(1, width))
            self.metrics['mid_res_usage'] += 1
            
        # Check if resolution changed
        if abs(prev_factor - self.resolution_factor) > 0.05:
            self.metrics['resolution_switches'] += 1
            
        # Resize frame if needed
        if abs(self.resolution_factor - 1.0) > 0.05:
            new_width = int(width * self.resolution_factor)
            new_height = int(height * self.resolution_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height))
        else:
            resized_frame = frame.copy()
            
        # Update processing time
        self.metrics['processing_time'] = time.time() - start_time
        
        return resized_frame, self.resolution_factor
        
    def process_regions(self, frame: np.ndarray, 
                      regions: List[Tuple[int, int, int, int]]) -> Dict[int, np.ndarray]:
        """
        Process specific regions at high resolution.
        
        Args:
            frame: Input frame
            regions: List of regions as (x1, y1, x2, y2)
            
        Returns:
            Dictionary of region_index -> high_res_region
        """
        high_res_regions = {}
        
        # Process each region
        for i, (x1, y1, x2, y2) in enumerate(regions):
            # Extract region
            region = frame[y1:y2, x1:x2]
            
            # Skip invalid regions
            if region.size == 0:
                continue
                
            # Use high resolution for small regions
            if region.shape[0] < 100 or region.shape[1] < 100:
                # Scale factor
                scale = 2.0
                
                # Resize to high resolution
                high_res = cv2.resize(region, (0, 0), fx=scale, fy=scale)
                high_res_regions[i] = high_res
            else:
                # Use original resolution
                high_res_regions[i] = region
                
        return high_res_regions
        
    def scale_detections(self, detections: List[List[float]], 
                       scale_factor: float) -> List[List[float]]:
        """
        Scale face detections to original resolution.
        
        Args:
            detections: List of detections [x1, y1, x2, y2, score, ...]
            scale_factor: Resolution scale factor
            
        Returns:
            Scaled detections
        """
        if scale_factor == 1.0:
            return detections
            
        scaled_detections = []
        
        for det in detections:
            if len(det) < 4:
                continue
                
            # Scale coordinates
            x1, y1, x2, y2 = det[:4]
            scaled_det = [
                x1 / scale_factor,
                y1 / scale_factor,
                x2 / scale_factor,
                y2 / scale_factor
            ]
            
            # Add additional info
            if len(det) > 4:
                scaled_det.extend(det[4:])
                
            scaled_detections.append(scaled_det)
            
        return scaled_detections
        
    def get_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Metrics dictionary
        """
        total_usage = max(1, sum([
            self.metrics['high_res_usage'],
            self.metrics['mid_res_usage'],
            self.metrics['low_res_usage']
        ]))
        
        # Calculate percentages
        metrics = self.metrics.copy()
        metrics['high_res_percentage'] = (metrics['high_res_usage'] / total_usage) * 100
        metrics['mid_res_percentage'] = (metrics['mid_res_usage'] / total_usage) * 100
        metrics['low_res_percentage'] = (metrics['low_res_usage'] / total_usage) * 100
        
        return metrics
        
    def adaptive_resize(self, frame: np.ndarray, face_boxes: List[List[float]],
                      target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize frame with adaptive resolution for faces.
        
        Args:
            frame: Input frame
            face_boxes: List of face boxes [x1, y1, x2, y2, ...]
            target_size: Optional target size
            
        Returns:
            Resized frame with high-res faces
        """
        if not face_boxes or target_size is None:
            return frame
            
        # Get current size
        h, w = frame.shape[:2]
        
        # Target size
        target_h, target_w = target_size
        
        # Skip if already at target size
        if h == target_h and w == target_w:
            return frame
            
        # Create low-res frame
        low_res = cv2.resize(frame, (target_w, target_h))
        
        # Calculate scale factors
        scale_h = target_h / h
        scale_w = target_w / w
        
        # Process each face
        for box in face_boxes:
            if len(box) < 4:
                continue
                
            # Get coordinates
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Extract face region
            face = frame[y1:y2, x1:x2]
            
            # Calculate size in low-res
            lr_x1 = int(x1 * scale_w)
            lr_y1 = int(y1 * scale_h)
            lr_x2 = int(x2 * scale_w)
            lr_y2 = int(y2 * scale_h)
            
            # Resize face to fit low-res position
            face_size = (lr_x2 - lr_x1, lr_y2 - lr_y1)
            
            if face_size[0] > 0 and face_size[1] > 0 and face.size > 0:
                try:
                    # Resize face
                    resized_face = cv2.resize(face, face_size)
                    
                    # Insert into low-res frame
                    low_res[lr_y1:lr_y2, lr_x1:lr_x2] = resized_face
                    
                except Exception as e:
                    logger.warning(f"Error resizing face: {e}")
                    
        return low_res