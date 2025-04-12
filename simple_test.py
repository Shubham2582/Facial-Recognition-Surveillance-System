# simple_test.py
import cv2
from models.detection.retinaface import FaceDetector
from config.detection_config import DETECTION_CONFIG

def main():
    # Initialize detector
    detector = FaceDetector(DETECTION_CONFIG)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        faces = detector.detect(frame)
        print(f"Detected {len(faces)} faces")
        
        # Draw faces
        for face in faces:
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Show result
        cv2.imshow("Simple Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()