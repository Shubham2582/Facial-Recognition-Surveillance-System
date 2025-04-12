# main.py
import cv2
import argparse
import logging
import os
import time
import numpy as np

# Import configuration
from config.system_config import SYSTEM_CONFIG, FEATURE_FLAGS
from config.detection_config import DETECTION_CONFIG
from config.recognition_config import RECOGNITION_CONFIG
from config.tracking_config import TRACKING_CONFIG

# Import core components
from models.detection.retinaface import FaceDetector
from models.recognition.arcface import FaceRecognizer
from database.face_database import FaceDatabase
from core.engine import SurveillanceEngine
from utils.visualization import Visualizer

# Import optional components based on feature flags
if FEATURE_FLAGS.get('use_memory_system', True):
    from memory.short_term_memory import ShortTermMemory
    from memory.medium_term_memory import MediumTermMemory
    from memory.memory_query import MemorySystem

if FEATURE_FLAGS.get('use_embedding_cache', True):
    from memory.embedding_cache import EmbeddingCache

if FEATURE_FLAGS.get('use_tracking', True):
    from tracking.byte_tracker import ByteTracker

if FEATURE_FLAGS.get('use_dynamic_resolution', True):
    from optimization.dynamic_resolution import DynamicResolutionScaler

if FEATURE_FLAGS.get('use_roi_management', True):
    from optimization.roi_manager import ROIManager

if FEATURE_FLAGS.get('use_gpu_indexing', True):
    from database.faiss_index import GPUAcceleratedIndex

def setup_logging():
    """Set up logging configuration."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'surveillance_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=getattr(logging, SYSTEM_CONFIG.get('log_level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def build_engine():
    """Build and initialize surveillance engine with all components."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing surveillance system...")
    
    # Initialize InsightFace detector
    logger.info("Initializing face detector...")
    detector = FaceDetector(DETECTION_CONFIG)
    app = detector.get_app()
    
    # Initialize face recognizer
    logger.info("Initializing face recognizer...")
    recognizer = FaceRecognizer(app, RECOGNITION_CONFIG)
    
    # Initialize face database
    logger.info("Initializing face database...")
    database = FaceDatabase(SYSTEM_CONFIG, app)
    
    # Initialize optional components
    memory_system = None
    embedding_cache = None
    tracker = None
    resolution_manager = None
    roi_manager = None
    
    # Initialize memory systems if enabled
    if FEATURE_FLAGS.get('use_memory_system', True):
        logger.info("Initializing memory systems...")
        short_term_memory = ShortTermMemory(SYSTEM_CONFIG)
        medium_term_memory = MediumTermMemory(SYSTEM_CONFIG)
        memory_system = MemorySystem(short_term_memory, medium_term_memory, SYSTEM_CONFIG)
    
    # Initialize embedding cache if enabled
    if FEATURE_FLAGS.get('use_embedding_cache', True):
        logger.info("Initializing embedding cache...")
        embedding_cache = EmbeddingCache(RECOGNITION_CONFIG.get('cache_config', {}))
    
    # Initialize tracking if enabled
    if FEATURE_FLAGS.get('use_tracking', True):
        logger.info("Initializing face tracker...")
        tracker = ByteTracker(TRACKING_CONFIG)
    
    # Initialize dynamic resolution scaling if enabled
    if FEATURE_FLAGS.get('use_dynamic_resolution', True):
        logger.info("Initializing dynamic resolution scaler...")
        resolution_manager = DynamicResolutionScaler(DETECTION_CONFIG.get('resolution_config', {}))
    
    # Initialize ROI management if enabled
    if FEATURE_FLAGS.get('use_roi_management', True):
        logger.info("Initializing ROI manager...")
        roi_manager = ROIManager(DETECTION_CONFIG.get('roi_config', {}))
    
    # Initialize surveillance engine
    logger.info("Initializing surveillance engine...")
    engine = SurveillanceEngine(
        SYSTEM_CONFIG,
        detector,
        recognizer,
        database.get_database(),
        memory_system,
        embedding_cache,
        tracker,
        resolution_manager,
        roi_manager
    )
    
    logger.info("Surveillance system initialized successfully")
    return engine, database

def process_video(video_source, engine, visualizer, save_output=False):
    """Process video stream with surveillance engine."""
    logger = logging.getLogger(__name__)
    
    # Open video source
    if isinstance(video_source, str) and os.path.isfile(video_source):
        logger.info(f"Opening video file: {video_source}")
        cap = cv2.VideoCapture(video_source)
    else:
        try:
            source = int(video_source)
        except ValueError:
            source = video_source
        logger.info(f"Opening camera: {source}")
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        return
    
    # Set camera properties if using camera
    if isinstance(video_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, SYSTEM_CONFIG.get('input_width', 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SYSTEM_CONFIG.get('input_height', 720))
        cap.set(cv2.CAP_PROP_FPS, SYSTEM_CONFIG.get('fps', 30))
    
    # Setup video writer if saving output
    writer = None
    if save_output:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'processed_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        logger.info(f"Saving output to: {output_path}")
    
    # Process frames
    logger.info("Starting video processing...")
    frame_count = 0
    total_processing_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or error reading frame")
                break
            
            frame_count += 1
            
            # Process frame
            detections, process_time, fps = engine.process_frame(frame)
            print(f"Got {len(detections)} detections, fps={fps}, time={process_time*1000:.1f}ms")
            # Make absolutely sure we're using detections
            if not detections:
                print("No detections in this frame")
            else:
                print(f"First detection: identity={detections[0].get('identity', 'Unknown')}, bbox={detections[0].get('bbox')}")
            
            # Debug detection results
            logger.debug(f"Frame {frame_count}: {len(detections)} detections")
            
            if len(detections) > 0:
                logger.debug(f"First detection: {detections[0].get('identity', 'Unknown')}, " 
                            f"confidence: {detections[0].get('confidence', 0.0):.2f}")
            
            # Ensure detections have proper format
            valid_detections = []
            for det in detections:
                if 'bbox' in det and isinstance(det['bbox'], tuple) and len(det['bbox']) == 4:
                    valid_detections.append(det)
                else:
                    logger.warning(f"Invalid detection format: {det}")
            
            # Add print statement before visualization
            print(f"First detection: {detections[0] if detections else 'None'}")
            
            # Draw results
            result_frame = visualizer.draw_detections(frame, valid_detections)
            
            # Add dashboard with stats
            stats = {
                'fps': fps,
                'face_count': len(valid_detections),
                'known_count': sum(1 for d in valid_detections if d.get('identity', 'Unknown') != 'Unknown'),
                'processing_time': process_time
            }
            result_frame = visualizer.create_dashboard(result_frame, stats)
            
            # Draw time
            result_frame = visualizer.draw_time(result_frame)
            
            # Add debug information directly on frame
            debug_info = [
                f"Frame: {frame_count}",
                f"Detections: {len(detections)}",
                f"FPS: {fps:.1f}",
                f"Processing: {process_time*1000:.1f}ms"
            ]
            for i, text in enumerate(debug_info):
                cv2.putText(result_frame, text, (10, 30 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Facial Recognition Surveillance', result_frame)
            
            # Save frame if enabled
            if writer is not None:
                writer.write(result_frame)
            
            # Track processing time
            total_processing_time += process_time
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested exit")
                break
                
            # Log progress periodically
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames. Avg processing time: {total_processing_time/frame_count*1000:.1f}ms")
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.exception(f"Error processing video: {e}")
    finally:
        # Clean up
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
        avg_time = total_processing_time / max(1, frame_count)
        logger.info(f"Processing complete. {frame_count} frames processed in {total_processing_time:.2f}s (avg: {avg_time*1000:.1f}ms/frame)")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Facial Recognition Surveillance System')
    parser.add_argument('--source', default=0, help='Video source (file path or camera index)')
    parser.add_argument('--build-db', action='store_true', help='Rebuild face database')
    parser.add_argument('--save', action='store_true', help='Save processed video')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Facial Recognition Surveillance System")
    
    try:
        # Build engine
        engine, database = build_engine()
        
        # Initialize visualization
        visualizer = Visualizer({})
        
        # Rebuild database if requested
        if args.build_db:
            logger.info("Rebuilding face database...")
            database.build_database()
            logger.info("Database rebuild complete")
        
        # Process video
        process_video(args.source, engine, visualizer, args.save)
        
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
    
    logger.info("Facial Recognition Surveillance System shut down")

if __name__ == "__main__":
    main()