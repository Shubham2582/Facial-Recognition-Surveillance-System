"""Optimized recognition models using INT8 quantization."""

import onnxruntime as ort
import numpy as np
import cv2
import logging
from typing import Dict, Any, Optional

class QuantizedRecognizer:
    """INT8 quantized ArcFace model for optimized recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize quantized recognizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_path = config.get('quantized_model_path', 'models/quantized/w600k_r50_int8.onnx')
        self.input_size = config.get('image_size', (112, 112))
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
            self.output_name = self.session.get_outputs()[0].name
            
            self.logger.info(f"Loaded quantized recognition model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading quantized recognition model: {e}")
            self.session = None
            
    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding for a face image.
        
        Args:
            face_img: Face image (already cropped)
            
        Returns:
            Normalized embedding vector
        """
        if self.session is None:
            return None
            
        try:
            # Preprocess face
            processed_face = self._preprocess(face_img)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: processed_face})
            
            # Get embedding
            embedding = outputs[0][0]
            
            # Normalize embedding
            normed_embedding = embedding / np.linalg.norm(embedding)
            
            return normed_embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding from quantized model: {e}")
            return None
            
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess face for quantized model.
        
        Args:
            img: Input face image
            
        Returns:
            Preprocessed face
        """
        # Resize image
        resized = cv2.resize(img, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 127.5 - 1.0
        
        # Transpose to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched