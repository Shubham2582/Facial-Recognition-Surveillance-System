# test_resolution_roi.py
import cv2
import logging
import time
from models.detection.retinaface import FaceDetector
from optimization.roi_manager import ROIManager
from optimization.dynamic_resolution import DynamicResolutionScaler
from config.detection_config import DETECTION_CONFIG

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("resolution_roi_test")

def main():
    # Initialize components
    detector = FaceDetector(DETECTION_CONFIG)
    
    roi_config = DETECTION_CONFIG.get('roi_config', {
        'grid_size': (4, 4),
        'attention_decay': 0.9,
        'min_face_size': (30, 30)
    })
    
    resolution_config = DETECTION_CONFIG.get('resolution_config', {
        'bg_resolution': (480, 320),
        'face_padding': 20,
    })
    
    roi_manager = ROIManager(roi_config)
    resolution_manager = DynamicResolutionScaler(resolution_config)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Process frames
    frame_count = 0
    
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        logger.debug(f"Processing frame {frame_count}")
        
        # Step 1: Apply dynamic resolution
        logger.debug(f"Original frame shape: {frame.shape}")
        processed_frame, scale_info = resolution_manager.process(frame, [])
        logger.debug(f"After resolution scaling: {processed_frame.shape}, scale: {scale_info}")
        
        # Step 2: Generate ROI mask for the PROCESSED frame
        roi_mask = roi_manager.get_roi_mask(processed_frame)
        logger.debug(f"ROI mask shape: {roi_mask.shape}, dtype: {roi_mask.dtype}")
        
        # Step 3: Apply ROI mask
        try:
            masked_frame = cv2.bitwise_and(processed_frame, processed_frame, mask=roi_mask)
            logger.debug("Applied ROI mask successfully")
        except Exception as e:
            logger.error(f"Failed to apply ROI mask: {e}")
            continue
            
        # Step 4: Detect faces
        try:
            faces = detector.detect(processed_frame, roi_mask)
            logger.debug(f"Detected {len(faces)} faces")
        except Exception as e:
            logger.error(f"Failed to detect faces: {e}")
            continue
            
        # Display results
        # Original frame
        cv2.imshow("Original", frame)
        
        # Processed frame (after resolution scaling)
        cv2.imshow("Processed", processed_frame)
        
        # ROI mask
        cv2.imshow("ROI Mask", roi_mask)
        
        # Masked frame
        cv2.imshow("Masked Frame", masked_frame)
        
        # Wait for key press
        key = cv2.waitKey(100)
        if key & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()