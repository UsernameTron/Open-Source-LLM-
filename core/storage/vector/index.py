"""
Specialized indexing strategies for text embeddings.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import faiss
from dataclasses import dataclass
from core.metrics import performance_monitor
from core.storage.types import VectorConfig

logger = logging.getLogger(__name__)

@dataclass
class IndexStats:
    """Statistics about the index."""
    size: int
    dimension: int
    index_type: str
    memory_usage: int  # bytes
    is_trained: bool

class VectorIndex:
    def __init__(self, config: VectorConfig):
        """
        Initialize vector index.
        
        Args:
            config: Vector configuration
        """
        self.config = config
        self._index = None
        self._init_index()
        logger.info(f"Initialized {config.index_type} index")
        
    @performance_monitor
    def _init_index(self):
        """Initialize FAISS index based on configuration."""
        try:
            if self.config.index_type == "hnsw":
                self._create_hnsw_index()
            elif self.config.index_type == "ivf":
                self._create_ivf_index()
            elif self.config.index_type == "flat":
                self._create_flat_index()
            else:
                raise ValueError(f"Unsupported index type: {self.config.index_type}")
                
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}")
            raise
            
    def _create_hnsw_index(self):
        """Create HNSW index optimized for text embeddings."""
        if self.config.metric == "cosine":
            # Normalize vectors for cosine similarity
            self._index = faiss.IndexHNSWFlat(
                self.config.dimension,
                self.config.index_m
            )
            self._index.hnsw.efConstruction = self.config.index_ef_construction
            self._index.hnsw.efSearch = self.config.index_ef_search
            
        elif self.config.metric == "euclidean":
            self._index = faiss.IndexHNSWL2(
                self.config.dimension,
                self.config.index_m
            )
            
        else:
            raise ValueError(f"Unsupported metric: {self.config.metric}")
            
    def _create_ivf_index(self):
        """Create IVF index for large-scale search."""
        # Quantizer for IVF
        quantizer = faiss.IndexFlatL2(self.config.dimension)
        
        # Number of centroids (adjust based on dataset size)
        n_centroids = max(int(np.sqrt(self.config.batch_size)), 1)
        
        self._index = faiss.IndexIVFFlat(
            quantizer,
            self.config.dimension,
            n_centroids,
            faiss.METRIC_L2
        )
        
    def _create_flat_index(self):
        """Create flat index for exact search."""
        if self.config.metric == "cosine":
            self._index = faiss.IndexFlatIP(self.config.dimension)
        else:
            self._index = faiss.IndexFlatL2(self.config.dimension)
            
    @performance_monitor
    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to index.
        
        Args:
            vectors: Vectors to add [n_vectors, dimension]
        """
        try:
            if vectors.shape[1] != self.config.dimension:
                raise ValueError(
                    f"Vector dimension {vectors.shape[1]} does not match "
                    f"index dimension {self.config.dimension}"
                )
                
            # Normalize vectors for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(vectors)
                
            self._index.add(vectors)
            logger.info(f"Added {len(vectors)} vectors to index")
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {str(e)}")
            raise
            
    @performance_monitor
    def search(
        self,
        query: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector [1, dimension]
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        try:
            if query.shape[1] != self.config.dimension:
                raise ValueError(
                    f"Query dimension {query.shape[1]} does not match "
                    f"index dimension {self.config.dimension}"
                )
                
            # Normalize query for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(query)
                
            distances, indices = self._index.search(query, k)
            return distances, indices
            
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            raise
            
    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        return IndexStats(
            size=self._index.ntotal,
            dimension=self.config.dimension,
            index_type=self.config.index_type,
            memory_usage=self._estimate_memory_usage(),
            is_trained=getattr(self._index, "is_trained", True)
        )
        
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of the index in bytes."""
        # Base memory for vectors
        vector_memory = (
            self._index.ntotal *  # Number of vectors
            self.config.dimension *  # Dimension
            4  # 4 bytes per float32
        )
        
        # Additional memory for index structures
        if self.config.index_type == "hnsw":
            # HNSW graph structure
            graph_memory = (
                self._index.ntotal *  # Number of vectors
                self.config.index_m *  # Neighbors per node
                8  # 8 bytes per pointer
            )
            return vector_memory + graph_memory
            
        elif self.config.index_type == "ivf":
            # IVF centroids and lists
            n_centroids = self._index.nlist
            centroid_memory = (
                n_centroids *
                self.config.dimension *
                4  # 4 bytes per float32
            )
            return vector_memory + centroid_memory
            
        else:
            return vector_memory
            
    def save(self, path: str) -> None:
        """Save index to disk."""
        try:
            faiss.write_index(self._index, path)
            logger.info(f"Saved index to {path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
            
    def load(self, path: str) -> None:
        """Load index from disk."""
        try:
            self._index = faiss.read_index(path)
            logger.info(f"Loaded index from {path}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
