"""Histogram calculation utilities for face comparison."""

import cv2
import numpy as np
from typing import Tuple

def compute_histogram(face_image: np.ndarray) -> np.ndarray:
    """Compute color histogram from face image.
    
    Args:
        face_image: Face image
        
    Returns:
        Color histogram
    """
    try:
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        
        # Compute histogram with 8x8x8 bins
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                        [0, 180, 0, 256, 0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except Exception as e:
        # Return a placeholder histogram in case of error
        return np.ones((8, 8, 8), dtype=np.float32)
        
def compare_histograms(hist1: np.ndarray, hist2: np.ndarray, 
                     method: int = cv2.HISTCMP_CORREL) -> float:
    """Compare two histograms and return similarity score.
    
    Args:
        hist1: First histogram
        hist2: Second histogram
        method: Comparison method
        
    Returns:
        Similarity score (0-1 for HISTCMP_CORREL)
    """
    similarity = cv2.compareHist(hist1, hist2, method)
    
    # For methods where smaller value means more similar
    if method in [cv2.HISTCMP_CHISQR, cv2.HISTCMP_BHATTACHARYYA]:
        # Convert to similarity score (0-1)
        return 1.0 - min(1.0, similarity)
    
    # For methods where larger value means more similar
    return max(0.0, similarity)
    
def compute_lbp_histogram(face_image: np.ndarray) -> np.ndarray:
    """Compute LBP (Local Binary Pattern) histogram for texture analysis.
    
    Args:
        face_image: Face image
        
    Returns:
        LBP histogram
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Simple LBP implementation
        lbp = np.zeros_like(gray)
        h, w = gray.shape
        
        # Iterate through image (excluding borders)
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                
                # Compare with 8 neighbors
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                
                lbp[i, j] = code
                
        # Compute histogram
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except Exception as e:
        # Return a placeholder histogram in case of error
        return np.ones(256, dtype=np.float32)
        
def compute_appearance_signature(face_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute combined appearance signature (color and texture).
    
    Args:
        face_image: Face image
        
    Returns:
        Tuple of (color histogram, LBP histogram)
    """
    color_hist = compute_histogram(face_image)
    lbp_hist = compute_lbp_histogram(face_image)
    
    return color_hist, lbp_hist
    
def compare_appearance(face1: np.ndarray, face2: np.ndarray) -> float:
    """Compare appearance of two face images.
    
    Args:
        face1: First face image
        face2: Second face image
        
    Returns:
        Similarity score (0-1)
    """
    # Compute histograms
    color_hist1 = compute_histogram(face1)
    color_hist2 = compute_histogram(face2)
    
    lbp_hist1 = compute_lbp_histogram(face1)
    lbp_hist2 = compute_lbp_histogram(face2)
    
    # Compare histograms
    color_similarity = compare_histograms(color_hist1, color_hist2)
    texture_similarity = compare_histograms(lbp_hist1, lbp_hist2)
    
    # Combine similarities (weighted)
    similarity = color_similarity * 0.6 + texture_similarity * 0.4
    
    return similarity