# test_recognition_accuracy.py
import cv2
import numpy as np
import logging
import os
import time
from models.detection.retinaface import FaceDetector
from models.recognition.arcface import FaceRecognizer
from database.face_database import FaceDatabase
from config.system_config import SYSTEM_CONFIG
from config.detection_config import DETECTION_CONFIG
from config.recognition_config import RECOGNITION_CONFIG

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("recognition_test")

def main():
    # Initialize components
    logger.info("Initializing face detector...")
    detector = FaceDetector(DETECTION_CONFIG)
    app = detector.get_app()
    
    logger.info("Initializing face recognizer...")
    recognizer = FaceRecognizer(app, RECOGNITION_CONFIG)
    
    logger.info("Loading database...")
    database = FaceDatabase(SYSTEM_CONFIG, app)
    db = database.get_database()
    
    logger.info(f"Database loaded with {len(db['identities'])} identities")
    
    # Track recognition accuracy
    matches = 0
    near_matches = 0
    non_matches = 0
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        faces = app.get(frame)
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # Process each face
        for face in faces:
            # Extract face info
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Get embedding (already normalized)
            embedding = face.normed_embedding
            
            # Try direct identification against database
            identity, confidence = recognizer.identify_face(embedding, db)
            
            # Track statistics
            if identity != "Unknown":
                matches += 1
                color = (0, 255, 0)  # Green for match
            elif confidence > 0.25:  # Track near matches
                near_matches += 1
                color = (0, 165, 255)  # Orange for near match
                logger.debug(f"Near match: {confidence:.4f}")
            else:
                non_matches += 1
                color = (0, 0, 255)  # Red for non-match
            
            # Draw face box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{identity}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Draw statistics
        total = max(1, matches + near_matches + non_matches)
        match_rate = matches / total * 100
        stats_text = [
            f"Matches: {matches} ({match_rate:.1f}%)",
            f"Near Matches: {near_matches} ({near_matches/total*100:.1f}%)",
            f"Non-Matches: {non_matches} ({non_matches/total*100:.1f}%)",
            f"Threshold: {RECOGNITION_CONFIG['threshold']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(vis_frame, text, (10, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow("Recognition Test", vis_frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    logger.info("=== Recognition Statistics ===")
    logger.info(f"Matches: {matches}")
    logger.info(f"Near Matches: {near_matches}")
    logger.info(f"Non-Matches: {non_matches}")
    logger.info(f"Match Rate: {match_rate:.1f}%")

if __name__ == "__main__":
    main()