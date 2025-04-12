# test_dynamic_resolution.py
import cv2
import time
from models.detection.retinaface import FaceDetector
from optimization.dynamic_resolution import DynamicResolutionScaler
from config.detection_config import DETECTION_CONFIG
from config.system_config import SYSTEM_CONFIG

# Initialize components
detector = FaceDetector(DETECTION_CONFIG)
resolution_scaler = DynamicResolutionScaler(DETECTION_CONFIG["resolution_config"])

# Open video
cap = cv2.VideoCapture(0)

# Initialize variables
face_regions = []
fps_list = []
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Apply dynamic resolution scaling
    processed_frame, scale_factors = resolution_scaler.process(frame, face_regions)
    
    # Time detection
    t1 = time.time()
    faces = detector.detect(processed_frame)
    detection_time = time.time() - t1
    
    # Scale bounding boxes back to original coordinates
    face_regions = []
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        x1 = int(x1 / scale_factors[0])
        y1 = int(y1 / scale_factors[1])
        x2 = int(x2 / scale_factors[0])
        y2 = int(y2 / scale_factors[1])
        face['bbox'] = (x1, y1, x2, y2)
        face_regions.append((x1, y1, x2, y2))
        
        # Draw results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Calculate FPS
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = 30 / elapsed
        fps_list.append(fps)
        print(f"FPS: {fps:.2f}, Detection time: {detection_time*1000:.2f}ms")
        start_time = time.time()
    
    # Draw FPS
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Dynamic Resolution', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()