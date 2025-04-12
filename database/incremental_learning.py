"""Database updating through incremental learning."""

import numpy as np
import time
import cv2
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

class IncrementalLearning:
    """Incremental learning system for face recognition database."""
    
    def __init__(self, config: Dict[str, Any], database, recognizer):
        """Initialize incremental learning.
        
        Args:
            config: Learning configuration
            database: Face database instance
            recognizer: Face recognizer instance
        """
        self.config = config
        self.database = database
        self.recognizer = recognizer
        self.confidence_threshold = config.get('confidence_threshold', 0.9)
        self.max_samples_per_identity = config.get('max_samples_per_identity', 5)
        self.update_interval = config.get('update_interval', 3600)  # seconds
        
        # Initialize sample storage
        self.samples = {}  # identity -> list of (image, embedding, quality)
        
        # Initialize last update time
        self.last_update_time = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def add_sample(self, identity: str, face_image: np.ndarray, 
                 embedding: np.ndarray, confidence: float) -> bool:
        """Add face sample for incremental learning.
        
        Args:
            identity: Identity label
            face_image: Face image
            embedding: Face embedding
            confidence: Recognition confidence
            
        Returns:
            Success flag
        """
        # Skip unknown identities
        if identity == "Unknown":
            return False
            
        # Skip low confidence recognitions
        if confidence < self.confidence_threshold:
            return False
            
        # Ensure quality of sample
        quality_score = self._calculate_quality(face_image)
        if quality_score < 0.5:  # Minimum quality threshold
            return False
            
        # Initialize identity if needed
        if identity not in self.samples:
            self.samples[identity] = []
            
        # Check if we have too many samples
        if len(self.samples[identity]) >= self.max_samples_per_identity:
            # Find lowest quality sample
            min_idx = -1
            min_quality = 1.0
            
            for i, (_, _, q) in enumerate(self.samples[identity]):
                if q < min_quality:
                    min_quality = q
                    min_idx = i
                    
            # Replace if new sample is better
            if min_quality < quality_score:
                self.samples[identity][min_idx] = (face_image, embedding, quality_score)
                return True
            else:
                return False
                
        # Add new sample
        self.samples[identity].append((face_image, embedding, quality_score))
        return True
        
    def check_update(self) -> bool:
        """Check if database update should be performed.
        
        Returns:
            True if update should be performed
        """
        # Check update interval
        if time.time() - self.last_update_time < self.update_interval:
            return False
            
        # Check if we have enough samples
        if len(self.samples) < 3:  # Minimum identities
            return False
            
        # Check if we have enough samples per identity
        samples_per_identity = [len(samples) for samples in self.samples.values()]
        if sum(1 for count in samples_per_identity if count >= 3) < 3:
            return False
            
        return True
        
    def update_database(self) -> bool:
        """Update face database with collected samples.
        
        Returns:
            Success flag
        """
        if not self.samples:
            return False
            
        try:
            # Update database for each identity
            identities_updated = 0
            
            for identity, samples in self.samples.items():
                if not samples:
                    continue
                    
                # Average embeddings
                embeddings = [s[1] for s in samples]
                avg_embedding = np.mean(embeddings, axis=0)
                
                # Normalize
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm
                    
                # Update database
                self.database.add_face(identity, avg_embedding)
                identities_updated += 1
                
            # Reset samples
            self.samples = {}
            
            # Update last update time
            self.last_update_time = time.time()
            
            self.logger.info(f"Updated database with {identities_updated} identities through incremental learning")
            return True
        except Exception as e:
            self.logger.error(f"Error updating database: {e}")
            return False
            
    def _calculate_quality(self, face_image: np.ndarray) -> float:
        """Calculate quality score for face image.
        
        Args:
            face_image: Face image
            
        Returns:
            Quality score (0-1)
        """
        # Check image size
        h, w = face_image.shape[:2]
        size_score = min(1.0, (h * w) / (112 * 112))
        
        # Check blur
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 100)
        
        # Calculate overall score
        quality_score = (size_score * 0.3) + (blur_score * 0.7)
        
        return quality_score
        
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics.
        
        Returns:
            Learning statistics
        """
        stats = {
            'identities': len(self.samples),
            'total_samples': sum(len(samples) for samples in self.samples.values()),
            'samples_per_identity': {identity: len(samples) for identity, samples in self.samples.items()},
            'time_since_update': time.time() - self.last_update_time,
            'next_update_in': max(0, self.update_interval - (time.time() - self.last_update_time))
        }
        
        return stats