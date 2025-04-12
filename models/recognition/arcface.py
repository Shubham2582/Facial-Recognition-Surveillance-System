"""ArcFace recognition using InsightFace."""

import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional

class FaceRecognizer:
    """Face recognizer using InsightFace's ArcFace implementation."""
    
    def __init__(self, detector_app, config: Dict[str, Any]):
        """Initialize face recognizer.
        
        Args:
            detector_app: InsightFace FaceAnalysis instance
            config: Configuration dictionary
        """
        self.config = config
        self.threshold = config.get('threshold', 0.35)  # Lower threshold for one-shot learning
        self.detector_app = detector_app
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding for a face image.
        
        Args:
            face_img: Face image (already cropped)
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Process with InsightFace
            faces = self.detector_app.get(face_img)
            
            if not faces:
                self.logger.debug("No face found in provided face image")
                return None
                
            # Get largest face if multiple are detected
            if len(faces) > 1:
                largest_face = max(faces, key=lambda x: 
                                  (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            else:
                largest_face = faces[0]
                
            # Return normalized embedding
            embedding = largest_face.normed_embedding
            
            # Double-check normalization
            norm = np.linalg.norm(embedding)
            if norm < 0.95 or norm > 1.05:
                self.logger.debug(f"Re-normalizing embedding (norm: {norm:.4f})")
                embedding = embedding / norm
                
            return embedding
        
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return None
            
    def compare_embeddings(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Compare two embeddings and return similarity score.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure embeddings are normalized
        if np.linalg.norm(embedding1) < 0.95 or np.linalg.norm(embedding1) > 1.05:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            
        if np.linalg.norm(embedding2) < 0.95 or np.linalg.norm(embedding2) > 1.05:
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
        
    def identify_face(self, embedding: np.ndarray, 
                     database: Dict[str, Any]) -> Tuple[str, float]:
        """Identify a face against a database.
        
        Args:
            embedding: Face embedding
            database: Database of embeddings
            
        Returns:
            Tuple of (identity, confidence)
        """
        if "embeddings" not in database or len(database["embeddings"]) == 0:
            self.logger.warning("Empty database or missing embeddings")
            return "Unknown", 0.0
            
        # Find best match
        best_match_idx = -1
        best_match_score = -1.0
        all_scores = []  # For debugging
        
        for i, db_embedding in enumerate(database["embeddings"]):
            score = self.compare_embeddings(embedding, db_embedding)
            all_scores.append((database["identities"][i], score))
            
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i
                
        # Debug: Log top 3 matches
        top_matches = sorted(all_scores, key=lambda x: x[1], reverse=True)[:3]
        self.logger.debug(f"Top 3 matches: {top_matches}")
        
        # Check if score is above threshold
        if best_match_score >= self.threshold and best_match_idx >= 0:
            identity = database["identities"][best_match_idx]
            self.logger.debug(f"Match found: {identity} with confidence {best_match_score:.4f}")
            return identity, best_match_score
        else:
            self.logger.debug(f"No match above threshold. Best score: {best_match_score:.4f} (threshold: {self.threshold})")
            return "Unknown", best_match_score