# optimize_models.py
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorRT")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True):
    """
    Build TensorRT engine from ONNX model
    """
    logger.info(f"Building TensorRT engine: {os.path.basename(onnx_file_path)}")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_file_path, 'rb') as model:
        success = parser.parse(model.read())
        if not success:
            for error in range(parser.num_errors):
                logger.error(f"ONNX parsing error: {parser.get_error(error)}")
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
        
    logger.info(f"TensorRT engine saved to: {engine_file_path}")
    return engine

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models/trt", exist_ok=True)
    
    # Convert detector model
    det_onnx_path = "models/detection/det_10g.onnx"
    det_engine_path = "models/trt/det_10g.engine"
    
    if os.path.exists(det_onnx_path):
        build_engine(det_onnx_path, det_engine_path)
    else:
        logger.error(f"Detection model not found: {det_onnx_path}")
    
    # Convert recognition model
    rec_onnx_path = "models/recognition/w600k_r50.onnx"
    rec_engine_path = "models/trt/w600k_r50.engine"
    
    if os.path.exists(rec_onnx_path):
        build_engine(rec_onnx_path, rec_engine_path)
    else:
        logger.error(f"Recognition model not found: {rec_onnx_path}")

if __name__ == "__main__":
    main()
    