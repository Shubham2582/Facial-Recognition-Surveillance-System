# build_database.py
import logging
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from config.system_config import SYSTEM_CONFIG, DATABASE_CONFIG  # Make sure DATABASE_CONFIG is imported
from config.detection_config import DETECTION_CONFIG
from database.face_database import FaceDatabase
from database.faiss_index import GPUAcceleratedIndex

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatabaseBuilder")

def build_database(data_dir, app):
    """Build face database from all images in person folders.
    
    Args:
        data_dir: Path to database directory
        app: InsightFace FaceAnalysis instance
        
    Returns:
        Database dictionary
    """
    logger.info(f"Building database from directory: {data_dir}")
    database = {"embeddings": [], "identities": []}
    
    # Process each person directory
    person_count = 0
    total_images = 0
    
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        
        # Skip non-directories and special files
        if not os.path.isdir(person_path) or person_dir.startswith('.'):
            continue
            
        logger.info(f"Processing person: {person_dir}")
        
        # Get all image files
        images = [f for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            logger.warning(f"No images found for {person_dir}")
            continue
            
        # Process all images for this person
        valid_embeddings = []
        
        for img_file in images:
            image_path = os.path.join(person_path, img_file)
            logger.debug(f"Processing image: {img_file}")
            
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to read image {image_path}")
                continue
                
            # Use InsightFace for consistent processing
            faces = app.get(img)
            
            if not faces:
                logger.warning(f"No face detected in {image_path}")
                continue
                
            # Get largest face if multiple are detected
            if len(faces) > 1:
                largest_face = max(faces, key=lambda x: 
                                  (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                logger.debug(f"Multiple faces found in {img_file}, using largest")
            else:
                largest_face = faces[0]
                
            # Get normalized embedding
            embedding = largest_face.normed_embedding
            valid_embeddings.append(embedding)
            total_images += 1
            
        # Only add to database if we have valid embeddings
        if valid_embeddings:
            # Create single representative embedding by averaging
            # This approach creates a centroid in the embedding space
            avg_embedding = np.mean(valid_embeddings, axis=0)
            
            # Normalize the average embedding
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            # Add to database
            database["embeddings"].append(avg_embedding)
            database["identities"].append(person_dir)
            person_count += 1
            
            logger.info(f"Added {person_dir} with {len(valid_embeddings)} images")
        
    logger.info(f"Database built with {person_count} identities from {total_images} images")
    return database

def main():
    # Initialize InsightFace
    logger.info("Initializing InsightFace...")
    app = FaceAnalysis(name=DETECTION_CONFIG['model_name'], 
                     providers=DETECTION_CONFIG['providers'])
    app.prepare(ctx_id=SYSTEM_CONFIG['ctx_id'], det_size=DETECTION_CONFIG['det_size'])
    
    # Get database directory from config
    data_dir = DATABASE_CONFIG.get('database_dir', 'face_data')
    logger.info(f"Building face database from {data_dir}...")
    
    # Build database using custom function
    database = build_database(data_dir, app)
    
    # Create database directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save database directly
    db_path = os.path.join(data_dir, 'face_database.pkl')
    with open(db_path, 'wb') as f:
        pickle.dump(database, f)
    logger.info(f"Saved database with {len(database['identities'])} identities")
    
    # Initialize FAISS index if needed
    if DATABASE_CONFIG.get('use_faiss', True):
        logger.info("Building GPU-accelerated index...")
        faiss_index = GPUAcceleratedIndex(DATABASE_CONFIG)
        faiss_index.build_index(database)
    
    logger.info(f"Database built and saved successfully with {len(database['identities'])} identities")

if __name__ == "__main__":
    main()