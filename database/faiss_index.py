"""GPU-accelerated indexing using FAISS."""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional

class GPUAcceleratedIndex:
    """GPU-accelerated face database indexing using FAISS."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FAISS index.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.embedding_dim = config.get('embedding_size', 512)
        self.index_type = config.get('index_type', 'flat')
        self.identity_map = {}  # Maps index positions to identities
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize FAISS
        try:
            import faiss
            self.faiss = faiss
            self.res = faiss.StandardGpuResources()
            self.index = self._create_index()
            self.logger.info("FAISS GPU index initialized")
        except ImportError:
            self.logger.error("FAISS not installed, falling back to linear search")
            self.faiss = None
            self.index = None
            
    def _create_index(self):
        """Create FAISS index based on configuration.
        
        Returns:
            FAISS index
        """
        if self.faiss is None:
            return None
            
        if self.index_type == 'flat':
            # Simple flat index for small datasets
            return self.faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
        elif self.index_type == 'ivfpq':
            # IVF with Product Quantization for larger datasets
            quantizer = self.faiss.IndexFlatIP(self.embedding_dim)
            index = self.faiss.IndexIVFPQ(quantizer, self.embedding_dim, 100, 8, 8)
            index.nprobe = 10  # Number of clusters to search
            index = self.faiss.index_cpu_to_gpu(self.res, 0, index)
            return index
        else:
            self.logger.warning(f"Unknown index type {self.index_type}, using flat index")
            return self.faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
            
    def build_index(self, database: Dict[str, Any]):
        """Build index from database.
        
        Args:
            database: Face database dictionary
        """
        if self.faiss is None or self.index is None:
            return
            
        # Reset index
        self.index = self._create_index()
        self.identity_map = {}
        
        if len(database['embeddings']) == 0:
            self.logger.warning("Empty database, index not built")
            return
            
        # Convert embeddings to numpy array
        embeddings = np.array(database['embeddings']).astype('float32')
        
        # Normalize embeddings
        self.faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Build identity map
        for i, identity in enumerate(database['identities']):
            self.identity_map[i] = identity
            
        self.logger.info(f"Built FAISS index with {len(self.identity_map)} identities")
        
    def search(self, embedding: np.ndarray, k: int = 1) -> Tuple[str, float]:
        """Search for closest match to embedding.
        
        Args:
            embedding: Face embedding
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (identity, confidence)
        """
        if self.faiss is None or self.index is None or self.index.ntotal == 0:
            return "Unknown", 0.0
            
        # Convert to numpy array and normalize
        query = np.array([embedding]).astype('float32')
        self.faiss.normalize_L2(query)
        
        # Search index
        distances, indices = self.index.search(query, k)
        
        # Check if match found
        if indices[0][0] == -1 or indices[0][0] >= len(self.identity_map):
            return "Unknown", 0.0
            
        # Get best match
        best_idx = indices[0][0]
        confidence = float(distances[0][0])  # Convert from numpy type
        
        # Return identity and confidence
        identity = self.identity_map.get(best_idx, "Unknown")
        return identity, confidence
        
    def add_face(self, identity: str, embedding: np.ndarray):
        """Add face to index.
        
        Args:
            identity: Identity label
            embedding: Face embedding
        """
        if self.faiss is None or self.index is None:
            return
            
        # Convert to numpy array and normalize
        embedding_np = np.array([embedding]).astype('float32')
        self.faiss.normalize_L2(embedding_np)
        
        # Add to index
        idx = self.index.ntotal
        self.index.add(embedding_np)
        self.identity_map[idx] = identity