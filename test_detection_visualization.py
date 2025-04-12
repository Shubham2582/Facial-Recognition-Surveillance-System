# test_detection_visualization.py
import cv2
import logging
import os
import time
import numpy as np
from models.detection.retinaface import FaceDetector
from utils.visualization import Visualizer
from config.detection_config import DETECTION_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detection_viz_test")

def main():
    # Initialize detector
    detector = FaceDetector(DETECTION_CONFIG)
    
    # Initialize visualizer
    visualizer = Visualizer({})
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Process frames
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Get direct detections from detector
        faces = detector.detect(frame)
        
        # Log detection info
        logger.info(f"Frame {frame_count}: Detected {len(faces)} faces")
        if faces:
            bbox = faces[0]['bbox']
            logger.info(f"First face bbox: {bbox}")
        
        # Add dummy identity and confidence for visualization
        for face in faces:
            face['identity'] = 'Person'
            face['confidence'] = 1.0
        
        # Visualize detections
        result = visualizer.draw_detections(frame, faces)
        
        # Draw FPS
        fps_text = f"Frame: {frame_count}"
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Detection Test", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()