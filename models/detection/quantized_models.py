"""INT8 quantized face detection models."""

import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import Dict, Any, List, Tuple, Optional

class QuantizedDetector:
    """INT8 quantized RetinaFace detector for optimized inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize quantized detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_path = config.get('quantized_model_path', 'models/quantized/det_10g_int8.onnx')
        self.input_size = config.get('det_size', (640, 640))
        self.logger = logging.getLogger(__name__)
        
        try:
            # Create optimized session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_cpu_mem_arena = False
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=config.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            )
            
            # Get model details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.logger.info(f"Loaded quantized detection model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading quantized model: {e}")
            self.session = None
            
    def detect(self, frame: np.ndarray, 
              threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Detect faces using quantized model.
        
        Args:
            frame: Input frame
            threshold: Detection confidence threshold
            
        Returns:
            List of detected faces
        """
        if self.session is None:
            return []
            
        try:
            # Preprocess image
            img = self._preprocess(frame)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: img})
            
            # Parse results
            boxes, landmarks, scores = self._parse_outputs(outputs, frame.shape[:2])
            
            # Filter by threshold
            valid_indices = np.where(scores > threshold)[0]
            
            # Format results
            results = []
            for i in valid_indices:
                x1, y1, x2, y2 = boxes[i].astype(int)
                result = {
                    'bbox': (x1, y1, x2, y2),
                    'landmarks': landmarks[i],
                    'det_score': float(scores[i])
                }
                results.append(result)
                
            return results
        except Exception as e:
            self.logger.error(f"Error during quantized detection: {e}")
            return []
            
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for quantized model.
        
        Args:
            img: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize image
        resized = cv2.resize(img, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and transpose
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))  # CHW format
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
        
    def _parse_outputs(self, outputs: List[np.ndarray], 
                      original_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse model outputs.
        
        Args:
            outputs: Model outputs
            original_size: Original image size (height, width)
            
        Returns:
            Tuple of (boxes, landmarks, scores)
        """
        # Extract outputs
        boxes = outputs[0][0]  # Shape: [N, 4]
        landmarks = outputs[1][0]  # Shape: [N, 10]
        scores = outputs[2][0]  # Shape: [N]
        
        # Convert to original image coordinates
        h, w = original_size
        input_h, input_w = self.input_size
        
        scale_x = w / input_w
        scale_y = h / input_h
        
        # Scale boxes
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        
        # Scale landmarks
        landmarks[:, 0::2] *= scale_x
        landmarks[:, 1::2] *= scale_y
        
        # Reshape landmarks to [N, 5, 2]
        landmarks = landmarks.reshape(-1, 5, 2)
        
        return boxes, landmarks, scores