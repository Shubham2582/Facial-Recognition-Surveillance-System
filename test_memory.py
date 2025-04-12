# test_memory.py
import cv2
import time
from models.detection.retinaface import FaceDetector
from models.recognition.arcface import FaceRecognizer
from database.face_database import FaceDatabase
from memory.short_term_memory import ShortTermMemory
from memory.medium_term_memory import MediumTermMemory
from memory.memory_query import MemorySystem
from config.system_config import SYSTEM_CONFIG
from config.detection_config import DETECTION_CONFIG
from config.recognition_config import RECOGNITION_CONFIG

# Initialize components
detector = FaceDetector(DETECTION_CONFIG)
app = detector.get_app()
recognizer = FaceRecognizer(app, RECOGNITION_CONFIG)
database = FaceDatabase(SYSTEM_CONFIG, app)

# Initialize memory systems
stm = ShortTermMemory(SYSTEM_CONFIG)
mtm = MediumTermMemory(SYSTEM_CONFIG)
memory_system = MemorySystem(stm, mtm, SYSTEM_CONFIG)

# Open video
cap = cv2.VideoCapture(0)  # Use webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Detect faces
    faces = detector.detect(frame)
    
    for face in faces:
        track_id = 0  # Simplified for testing
        embedding = face['embedding']
        
        # First check memory
        memory_identity, memory_conf = memory_system.query(embedding, track_id)
        
        # If memory doesn't have strong match, check database
        if memory_conf < 0.65:
            db_identity, db_conf = recognizer.identify_face(embedding, database.get_database())
            
            # Use the stronger result
            if db_conf > memory_conf:
                identity, confidence = db_identity, db_conf
            else:
                identity, confidence = memory_identity, memory_conf
        else:
            identity, confidence = memory_identity, memory_conf
        
        # Update memory
        memory_system.update(track_id, embedding, identity, confidence)
        
        # Draw results
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{identity}: {confidence:.2f}", 
                   (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Memory Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()