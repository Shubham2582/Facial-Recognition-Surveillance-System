# debug_roi.py
import cv2
import numpy as np
import logging
import os
from models.detection.retinaface import FaceDetector
from optimization.roi_manager import ROIManager
from config.detection_config import DETECTION_CONFIG

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_roi")

def main():
    # Initialize detector
    detector = FaceDetector(DETECTION_CONFIG)
    
    # Initialize ROI manager
    roi_manager = ROIManager(DETECTION_CONFIG.get('roi_config', {
        'grid_size': (4, 4),
        'attention_decay': 0.9,
        'min_face_size': (30, 30)
    }))
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
        
    # Process a few frames
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get ROI mask
        logger.debug(f"Frame shape: {frame.shape}")
        roi_mask = roi_manager.get_roi_mask(frame)
        logger.debug(f"ROI mask shape: {roi_mask.shape}, dtype: {roi_mask.dtype}")
        
        # Try using the mask
        try:
            # Check if mask is valid for bitwise_and
            if roi_mask.shape[:2] != frame.shape[:2]:
                logger.error(f"Mask shape {roi_mask.shape[:2]} doesn't match frame shape {frame.shape[:2]}")
                
            # Check mask type
            if roi_mask.dtype != np.uint8:
                logger.error(f"Mask has wrong dtype: {roi_mask.dtype}")
                
            # Try the operation
            result = cv2.bitwise_and(frame, frame, mask=roi_mask)
            logger.debug("bitwise_and operation succeeded")
            
            # Display images for visual confirmation
            cv2.imshow("Original Frame", frame)
            cv2.imshow("ROI Mask", roi_mask)
            cv2.imshow("Result", result)
            cv2.waitKey(1000)
            
        except Exception as e:
            logger.error(f"Error applying mask: {e}")
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()