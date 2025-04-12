# test_roi.py
import cv2
import numpy as np
import time
from models.detection.retinaface import FaceDetector
from optimization.roi_manager import ROIManager
from config.detection_config import DETECTION_CONFIG
from config.system_config import SYSTEM_CONFIG

# Initialize components
detector = FaceDetector(DETECTION_CONFIG)
roi_manager = ROIManager(DETECTION_CONFIG["roi_config"])

# Open video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get ROI mask
    roi_mask = roi_manager.get_roi_mask(frame)
    
    # Detect faces using ROI mask
    faces = detector.detect(frame, roi_mask)
    
    # Update ROI based on detections
    roi_manager.update(faces)
    
    # Draw results
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Visualize ROI heatmap
    attention_map = roi_manager.get_attention_heatmap(frame.shape[:2])
    
    # Create visualization with ROI overlay
    alpha = 0.4
    overlay = cv2.addWeighted(frame, 1-alpha, attention_map, alpha, 0)
    
    cv2.imshow('ROI Management', overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()