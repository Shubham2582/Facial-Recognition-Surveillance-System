# utils/visualization.py
import cv2
import numpy as np
import time
from typing import Dict, Any

class Visualizer:
    """Visualization utilities for facial recognition system."""
    
    def __init__(self, config):
        """Initialize visualizer."""
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.5
        self.text_thickness = 1
        self.box_thickness = 2
        


    def draw_detections(self, frame, detections):
        """Draw face detections on frame."""
        result = frame.copy()
        
        for det in detections:
            try:
                if 'bbox' in det:
                    # Extract bounding box
                    x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                    
                    # Determine color based on recognition
                    identity = det.get('identity', 'Unknown')
                    confidence = det.get('confidence', 0.0)
                    
                    if identity == 'Unknown':
                        if confidence > 0.25:  # Near match
                            color = (0, 165, 255)  # Orange for near matches
                        else:
                            color = (0, 0, 255)  # Red for unknown
                    else:
                        color = (0, 255, 0)  # Green for known
                        
                    # Draw bounding box
                    cv2.rectangle(result, (x1, y1), (x2, y2), color, self.box_thickness)
                    
                    # Draw label background
                    label = f"{identity}: {confidence:.2f}"  # Only show identity and confidence
                    text_size = cv2.getTextSize(label, self.font, self.text_scale, self.text_thickness)[0]
                    cv2.rectangle(result, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(result, label, (x1, y1 - 5), self.font, self.text_scale, (255, 255, 255), self.text_thickness)
                    
                    # Draw track ID if available
                    if 'track_id' in det:
                        track_id = det['track_id']
                        track_label = f"ID: {track_id}"
                        cv2.putText(result, track_label, (x1, y2 + 15), self.font, self.text_scale, color, self.text_thickness)
            except Exception as e:
                print(f"Error drawing detection: {e}")
                continue
        
        return result
    def create_dashboard(self, frame, stats):
        """Create dashboard with performance statistics."""
        result = frame.copy()
        
        # Draw dashboard background
        h, w = result.shape[:2]
        cv2.rectangle(result, (10, 10), (250, 120), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (250, 120), (255, 255, 255), 1)
        
        # Draw stats
        stats_text = [
            f"FPS: {stats.get('fps', 0):.1f}",
            f"Faces: {stats.get('face_count', 0)}",
            f"Known: {stats.get('known_count', 0)}",
            f"Processing: {stats.get('processing_time', 0)*1000:.1f} ms"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(result, text, (20, 35 + i * 20), self.font, self.text_scale, (255, 255, 255), self.text_thickness)
            
        return result
        
    def draw_time(self, frame):
        """Draw current time on frame."""
        result = frame.copy()
        
        # Get current time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Draw time text
        cv2.putText(result, current_time, (10, result.shape[0] - 10), self.font, self.text_scale, (255, 255, 255), self.text_thickness)
        
        return result