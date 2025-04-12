# test_recognition.py
import cv2
from models.detection.retinaface import FaceDetector
from models.recognition.arcface import FaceRecognizer
from database.face_database import FaceDatabase
from config.system_config import SYSTEM_CONFIG
from config.detection_config import DETECTION_CONFIG
from config.recognition_config import RECOGNITION_CONFIG

# Initialize components
detector = FaceDetector(DETECTION_CONFIG)
app = detector.get_app()
recognizer = FaceRecognizer(app, RECOGNITION_CONFIG)
database = FaceDatabase(SYSTEM_CONFIG, app)

# Test on an image
image = cv2.imread('test_images/sample1.jpg')
faces = detector.detect(image)

for face in faces:
    identity, confidence = recognizer.identify_face(face['embedding'], database.get_database())
    print(f"Detected: {identity} with confidence {confidence:.2f}")
    
    # Draw results
    x1, y1, x2, y2 = face['bbox']
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, f"{identity}: {confidence:.2f}", 
               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Recognition', image)
cv2.waitKey(0)