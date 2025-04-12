"""Main processing engine for facial recognition surveillance system."""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple, Optional

class SurveillanceEngine:
    """Main processing engine for facial recognition surveillance system."""
    
    def __init__(self, config, detector, recognizer, database,
                memory_system=None, embedding_cache=None, tracker=None, 
                resolution_manager=None, roi_manager=None):
        """Initialize surveillance engine.
        
        Args:
            config: System configuration
            detector: Face detector instance
            recognizer: Face recognizer instance
            database: Face database
            memory_system: Memory system instance (optional)
            embedding_cache: Embedding cache instance (optional)
            tracker: Face tracker instance (optional)
            resolution_manager: Dynamic resolution manager (optional)
            roi_manager: ROI manager (optional)
        """
        self.config = config
        self.detector = detector
        self.recognizer = recognizer
        self.database = database
        self.memory_system = memory_system
        self.embedding_cache = embedding_cache
        self.tracker = tracker
        self.resolution_manager = resolution_manager
        self.roi_manager = roi_manager
        
        # Configure engine parameters
        self.detection_interval = config.get('detection_interval', 3)
        self.recognition_threshold = config.get('recognition_threshold', 0.35)  # Lower for one-shot learning
        self.max_faces = config.get('max_faces_per_frame', 20)
        
        # Initialize counters
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.processed_frames = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Surveillance engine initialized")
        
    def process_frame(self, frame):
        """Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (detections, process_time, fps)
        """
        self.frame_count += 1
        # Add at beginning of method
        print(f"Engine processing frame {self.frame_count}")
        
        start_time = time.time()
        
        # Initialize result
        results = []
        
        # Create a copy to avoid modifying the original
        original_frame = frame.copy()
        
        # Apply dynamic resolution scaling if available
        if self.resolution_manager is not None:
            # Get face regions from previous detections if available
            face_regions = []
            if hasattr(self, 'previous_results') and self.previous_results:
                face_regions = [det['bbox'] for det in self.previous_results if 'bbox' in det]
                
            # Apply resolution scaling
            processed_frame, scale_info = self.resolution_manager.process(original_frame, face_regions)
        else:
            processed_frame = original_frame
            scale_info = (1.0, 1.0)
        
        # Apply ROI management if available - IMPORTANT: Apply to the PROCESSED frame
        roi_mask = None
        if self.roi_manager is not None:
            try:
                # Generate mask for the PROCESSED frame size (after scaling)
                roi_mask = self.roi_manager.get_roi_mask(processed_frame)
                
                # Safety checks
                if roi_mask.shape[:2] != processed_frame.shape[:2]:
                    self.logger.warning(f"ROI mask shape mismatch: {roi_mask.shape[:2]} vs {processed_frame.shape[:2]}")
                    roi_mask = cv2.resize(roi_mask, (processed_frame.shape[1], processed_frame.shape[0]))
                
                if roi_mask.dtype != np.uint8:
                    self.logger.warning(f"ROI mask wrong dtype: {roi_mask.dtype}")
                    roi_mask = roi_mask.astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Error generating ROI mask: {e}")
                roi_mask = None
        
        # Run face detection on interval or first frame
        should_detect = (self.frame_count % self.detection_interval == 0) or (self.frame_count == 1)
        
        if should_detect:
            # Detect faces using the processed frame and ROI mask
            try:
                detected_faces = self.detector.detect(processed_frame, roi_mask)
                print(f"Engine detected {len(detected_faces)} faces")
            except Exception as e:
                self.logger.error(f"Error in face detection: {e}")
                detected_faces = []
            
            # Add debug logging for detected faces
            self.logger.debug(f"Detected {len(detected_faces)} faces")
            if detected_faces:
                self.logger.debug(f"First detection: {detected_faces[0].get('bbox')}")
            
            # Limit number of faces to process
            if len(detected_faces) > self.max_faces:
                # Sort by detection score and take top faces
                detected_faces = sorted(detected_faces, 
                                       key=lambda x: x.get('det_score', 0), reverse=True)[:self.max_faces]
            
            # Update tracking if available
            if self.tracker is not None:
                try:
                    tracked_faces = self.tracker.update(detected_faces, processed_frame)
                except Exception as e:
                    self.logger.error(f"Error updating tracking: {e}")
                    tracked_faces = detected_faces
            else:
                tracked_faces = detected_faces
                
            # Process each face
            for face in tracked_faces:
                # Get face info
                if 'bbox' not in face:
                    continue
                    
                bbox = face['bbox']
                
                # Assign track ID
                if 'track_id' in face:
                    track_id = face['track_id']
                else:
                    track_id = -1
                    
                # Extract face image from processed frame
                try:
                    x1, y1, x2, y2 = [int(c) for c in bbox]
                    
                    # Validate coordinates
                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > processed_frame.shape[1] or y2 > processed_frame.shape[0]:
                        self.logger.warning(f"Invalid face bbox: {bbox}")
                        continue
                        
                    face_img = processed_frame[y1:y2, x1:x2]
                    
                    # Check if face image is valid
                    if face_img.size == 0:
                        self.logger.warning(f"Empty face image from bbox: {bbox}")
                        continue
                except Exception as e:
                    self.logger.error(f"Error extracting face image: {e}")
                    continue
                
                # Handle embedding - either from detection, cache, or compute new
                embedding = None
                
                # Get from detection result if available
                if 'embedding' in face and face['embedding'] is not None:
                    embedding = face['embedding']
                
                # Check cache if tracking is enabled
                elif self.tracker is not None and self.embedding_cache is not None and track_id != -1:
                    try:
                        cached_embedding = self.embedding_cache.get(track_id)
                        
                        if cached_embedding is not None and not self.embedding_cache.should_update(face, face_img):
                            embedding = cached_embedding
                        else:
                            # Compute new embedding
                            embedding = self.recognizer.get_embedding(face_img)
                            
                            # Update cache
                            if embedding is not None and track_id != -1:
                                self.embedding_cache.update(track_id, embedding, face_img)
                    except Exception as e:
                        self.logger.error(f"Error with embedding cache: {e}")
                        # Fallback to direct embedding computation
                        embedding = self.recognizer.get_embedding(face_img)
                else:
                    # Compute embedding directly
                    embedding = self.recognizer.get_embedding(face_img)
                
                # Identify face if embedding is available
                identity = "Unknown"
                confidence = 0.0
                
                if embedding is not None:
                    # Check memory system first if available
                    if self.memory_system is not None:
                        try:
                            mem_identity, mem_conf = self.memory_system.query(embedding, track_id)
                            
                            # If memory doesn't have strong match, query database
                            if mem_conf < 0.4:  # Lower threshold from 0.65 to 0.4
                                db_identity, db_conf = self.recognizer.identify_face(embedding, self.database)
                                
                                # Use the stronger result
                                if db_conf > mem_conf:
                                    identity, confidence = db_identity, db_conf
                                else:
                                    identity, confidence = mem_identity, mem_conf
                            else:
                                identity, confidence = mem_identity, mem_conf
                                
                            # Update memory systems with this recognition
                            if confidence > 0.3:  # Lower threshold from 0.5 to 0.3
                                self.memory_system.update(track_id, embedding, identity, confidence)
                        except Exception as e:
                            self.logger.error(f"Error with memory system: {e}")
                            # Fallback to direct database check
                            identity, confidence = self.recognizer.identify_face(embedding, self.database)
                    else:
                        # Direct database check
                        identity, confidence = self.recognizer.identify_face(embedding, self.database)
                    
                    # Log successful recognitions
                    if identity != "Unknown":
                        self.logger.info(f"Recognized: {identity} with confidence {confidence:.4f}")
                    elif confidence > 0.25:  # Log near misses
                        self.logger.debug(f"Near miss: confidence {confidence:.4f} below threshold {self.recognition_threshold}")
                
                # Scale bounding box back to original coordinates if needed
                if scale_info != (1.0, 1.0):
                    x1, y1, x2, y2 = face['bbox']
                    face['bbox'] = (
                        int(x1 / scale_info[0]),
                        int(y1 / scale_info[1]),
                        int(x2 / scale_info[0]),
                        int(y2 / scale_info[1])
                    )
                
                # Add identification result to face info
                face['identity'] = identity
                face['confidence'] = confidence
                
                # Add to results
                results.append(face)
                
            # Save results for next frame's ROI optimization
            self.previous_results = results
        else:
            # If not detecting this frame, use previous results (if tracking isn't being used)
            if self.tracker is None and hasattr(self, 'previous_results'):
                results = self.previous_results
        
        # Update ROI manager if available
        if self.roi_manager is not None and len(results) > 0:
            try:
                self.roi_manager.update(results)
            except Exception as e:
                self.logger.error(f"Error updating ROI manager: {e}")
        
        # Calculate FPS
        self.processed_frames += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.fps = self.processed_frames
            self.processed_frames = 0
            self.last_fps_time = time.time()
        
        process_time = time.time() - start_time
        
        # Add debug logging for return results
        self.logger.debug(f"Returning {len(results)} processed detections")
        if results:
            self.logger.debug(f"First result: id={results[0].get('identity', 'Unknown')}, bbox={results[0].get('bbox')}")
        
        print(f"Engine returning {len(results)} results")
        return results, process_time, self.fps