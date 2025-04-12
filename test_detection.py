# test_detection.py
import cv2
from models.detection.retinaface import FaceDetector
from config.detection_config import DETECTION_CONFIG

# Initialize detector
detector = FaceDetector(DETECTION_CONFIG)

# Test on an image
image = cv2.imread('test_images/sample1.jpg')
faces = detector.detect(image)
print(f"Detected {len(faces)} faces")

# Draw results
for face in faces:
    x1, y1, x2, y2 = face['bbox']
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv2.imshow('Detections', image)
cv2.waitKey(0)