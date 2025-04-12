# simple_main.py
import cv2
import logging
import time
import numpy as np
from models.detection.retinaface import FaceDetector
from models.recognition.arcface import FaceRecognizer
from database.face_database import FaceDatabase
from utils.visualization import Visualizer
from config.detection_config import DETECTION_CONFIG
from config.recognition_config import RECOGNITION_CONFIG
from config.system_config import SYSTEM_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_main")

def main():
    # Initialize components
    logger.info("Initializing detector...")
    detector = FaceDetector(DETECTION_CONFIG)
    app = detector.get_app()
    
    logger.info("Initializing recognizer...")
    recognizer = FaceRecognizer(app, RECOGNITION_CONFIG)
    
    logger.info("Loading database...")
    database = FaceDatabase(SYSTEM_CONFIG, app)
    db = database.get_database()
    
    logger.info(f"Database loaded with {len(db['identities'])} identities")
    
    logger.info("Initializing visualizer...")
    visualizer = Visualizer({})
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    logger.info("Starting processing...")
    
    while True:
        # Measure timing
        loop_start = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            break
            
        frame_count += 1
        
        # Detect faces - directly with InsightFace for simplicity
        try:
            faces = app.get(frame)
            logger.info(f"Frame {frame_count}: Detected {len(faces)} faces")
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            faces = []
        
        # Process each face
        recognized_faces = []
        for face in faces:
            try:
                # Convert to our format
                bbox = face.bbox.astype(int)
                embedding = face.normed_embedding
                
                # Get identity
                identity, confidence = recognizer.identify_face(embedding, db)
                
                # Create result
                result = {
                    'bbox': tuple(bbox),
                    'identity': identity,
                    'confidence': confidence
                }
                
                recognized_faces.append(result)
                
                logger.info(f"Face: {identity}, confidence: {confidence:.2f}")
            except Exception as e:
                logger.error(f"Error processing face: {e}")
        
        # Calculate FPS
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time) if end_time > start_time else 0
            start_time = end_time
        
        # Draw results
        try:
            result_frame = visualizer.draw_detections(frame, recognized_faces)
            
            # Add simple stats
            cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            result_frame = frame
        
        # Display
        cv2.imshow("Simple Face Recognition", result_frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Limit processing speed
        loop_time = time.time() - loop_start
        logger.debug(f"Frame processing time: {loop_time*1000:.1f}ms")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()