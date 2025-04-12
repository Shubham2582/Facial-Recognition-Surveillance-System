# database/face_database.py
import os
import cv2
import numpy as np
import pickle
import logging
from typing import Dict, Any, List, Optional
import time

class FaceDatabase:
    """Face database management system."""
    
    def __init__(self, config: Dict[str, Any], detector):
        """Initialize face database.
        
        Args:
            config: Database configuration
            detector: Face detector instance (InsightFace app instance)
        """
        self.config = config
        self.detector = detector
        self.database_dir = config.get('database_dir', 'face_data')
        self.db_path = os.path.join(self.database_dir, 'face_database.pkl')
        
        # Initialize database
        self.database = {"embeddings": [], "identities": []}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load database if exists
        self.load_database()
        
    def load_database(self):
        """Load face database from file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.database = pickle.load(f)
                self.logger.info(f"Loaded database with {len(self.database['identities'])} identities")
            except Exception as e:
                self.logger.error(f"Error loading database: {e}")
                self.database = {"embeddings": [], "identities": []}
        else:
            self.logger.warning("Database file not found, creating new database")
            
    def build_database(self):
        """Build face database from images."""
        start_time = time.time()
        self.logger.info("Building face database...")
        
        # Reset database
        self.database = {"embeddings": [], "identities": []}
        
        # Check if database directory exists
        if not os.path.exists(self.database_dir):
            self.logger.error(f"Database directory {self.database_dir} does not exist")
            return
            
        # Process each person directory
        people_processed = 0
        total_images = 0
        successful_images = 0
        
        for person_dir in os.listdir(self.database_dir):
            person_path = os.path.join(self.database_dir, person_dir)
            
            # Skip non-directories and special files
            if not os.path.isdir(person_path) or person_dir.startswith('.'):
                continue
                
            self.logger.info(f"Processing person: {person_dir}")
            
            # Get all image files
            images = [f for f in os.listdir(person_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not images:
                self.logger.warning(f"No images found for {person_dir}")
                continue
                
            # Process all images for this person
            valid_embeddings = []
            
            for img_file in images:
                image_path = os.path.join(person_path, img_file)
                total_images += 1
                
                img = cv2.imread(image_path)
                if img is None:
                    self.logger.warning(f"Failed to read image {image_path}")
                    continue
                    
                # Use InsightFace for processing
                try:
                    faces = self.detector.get(img)
                    
                    if not faces:
                        self.logger.warning(f"No face detected in {image_path}")
                        continue
                        
                    # Get largest face if multiple are detected
                    if len(faces) > 1:
                        largest_face = max(faces, key=lambda x: 
                                        (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                        self.logger.debug(f"Multiple faces in {img_file}, using largest")
                    else:
                        largest_face = faces[0]
                        
                    # Get normalized embedding
                    embedding = largest_face.normed_embedding
                    
                    # Verify embedding is normalized
                    norm = np.linalg.norm(embedding)
                    if norm < 0.95 or norm > 1.05:
                        self.logger.warning(f"Embedding not properly normalized: {norm:.4f}")
                        embedding = embedding / norm  # Normalize it
                    
                    valid_embeddings.append(embedding)
                    successful_images += 1
                    self.logger.debug(f"Processed {img_file} successfully")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    continue
                    
            # Only add to database if we have valid embeddings
            if valid_embeddings:
                # Create single representative embedding by averaging
                avg_embedding = np.mean(valid_embeddings, axis=0)
                
                # Normalize the average embedding
                norm = np.linalg.norm(avg_embedding)
                avg_embedding = avg_embedding / norm
                
                # Add to database
                self.database["embeddings"].append(avg_embedding)
                self.database["identities"].append(person_dir)
                people_processed += 1
                
                self.logger.info(f"Added {person_dir} with {len(valid_embeddings)} images")
            
        self.logger.info(f"Database built with {people_processed} identities from {successful_images}/{total_images} images")
        self.logger.info(f"Database building took {time.time() - start_time:.2f} seconds")
        
        # Save database
        self.save_database()
        
    def save_database(self):
        """Save face database to file."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.database, f)
            self.logger.info(f"Saved database with {len(self.database['identities'])} identities")
        except Exception as e:
            self.logger.error(f"Error saving database: {e}")
            
    def add_face(self, identity: str, embedding: np.ndarray):
        """Add face to database.
        
        Args:
            identity: Identity label
            embedding: Face embedding
        """
        # Check if identity already exists
        if identity in self.database["identities"]:
            idx = self.database["identities"].index(identity)
            self.database["embeddings"][idx] = embedding
            self.logger.info(f"Updated existing identity: {identity}")
        else:
            self.database["embeddings"].append(embedding)
            self.database["identities"].append(identity)
            self.logger.info(f"Added new identity: {identity}")
            
        # Save database
        self.save_database()
        
    def remove_face(self, identity: str):
        """Remove face from database.
        
        Args:
            identity: Identity label
        """
        if identity in self.database["identities"]:
            idx = self.database["identities"].index(identity)
            self.database["embeddings"].pop(idx)
            self.database["identities"].pop(idx)
            self.logger.info(f"Removed identity: {identity}")
            
            # Save database
            self.save_database()
        else:
            self.logger.warning(f"Identity not found: {identity}")
            
    def get_database(self):
        """Get current database.
        
        Returns:
            Database dictionary
        """
        return self.database