"""INT8 quantization for deep learning models."""

import os
import numpy as np
import logging
import onnx
from typing import Dict, Any, List, Optional

class ModelQuantizer:
    """Model quantizer for ONNX models to INT8 precision."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model quantizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check if onnxruntime-extensions is available
        try:
            import onnxruntime_extensions
            self.have_extensions = True
            self.logger.info("onnxruntime-extensions available for quantization")
        except ImportError:
            self.have_extensions = False
            self.logger.warning("onnxruntime-extensions not available, using basic quantization")
        
    def quantize_model(self, input_model_path: str, output_model_path: str,
                      calibration_data: Optional[List[np.ndarray]] = None) -> bool:
        """Quantize ONNX model to INT8.
        
        Args:
            input_model_path: Path to input ONNX model
            output_model_path: Path to output quantized model
            calibration_data: List of input data for calibration
            
        Returns:
            Success flag
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            
            # Quantize model
            self.logger.info(f"Quantizing model: {input_model_path} -> {output_model_path}")
            quantize_dynamic(
                model_input=input_model_path,
                model_output=output_model_path,
                per_channel=False,
                reduce_range=True,
                weight_type=QuantType.QInt8
            )
            
            self.logger.info(f"Model quantized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error quantizing model: {e}")
            return False
            
    def quantize_with_calibration(self, input_model_path: str, output_model_path: str,
                                calibration_data: List[np.ndarray]) -> bool:
        """Quantize ONNX model to INT8 with calibration data.
        
        Args:
            input_model_path: Path to input ONNX model
            output_model_path: Path to output quantized model
            calibration_data: List of input data for calibration
            
        Returns:
            Success flag
        """
        if not self.have_extensions:
            self.logger.warning("Calibration-based quantization requires onnxruntime-extensions")
            return self.quantize_model(input_model_path, output_model_path)
            
        try:
            from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
            from onnxruntime.quantization.calibrate import CalibrationDataReader
            
            # Create calibration data reader
            class CustomCalibrationDataReader(CalibrationDataReader):
                def __init__(self, data_list, input_name):
                    self.data_list = data_list
                    self.input_name = input_name
                    self.index = 0
                    
                def get_next(self):
                    if self.index >= len(self.data_list):
                        return None
                    input_data = self.data_list[self.index]
                    self.index += 1
                    return {self.input_name: input_data}
                    
                def rewind(self):
                    self.index = 0
            
            # Get input name
            model = onnx.load(input_model_path)
            input_name = model.graph.input[0].name
            
            # Create calibration data reader
            calibration_reader = CustomCalibrationDataReader(calibration_data, input_name)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            
            # Quantize model with calibration
            self.logger.info(f"Quantizing model with calibration: {input_model_path} -> {output_model_path}")
            quantize_static(
                model_input=input_model_path,
                model_output=output_model_path,
                calibration_data_reader=calibration_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
                optimize_model=True
            )
            
            self.logger.info(f"Model quantized with calibration successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error quantizing model with calibration: {e}")
            return False