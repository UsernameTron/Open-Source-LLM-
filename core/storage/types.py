"""
Type definitions for vector storage framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np

@dataclass
class VectorConfig:
    """Configuration for vector storage."""
    dimension: int
    metric: str = "cosine"  # cosine, euclidean, dot_product
    index_type: str = "hnsw"  # hnsw, flat, ivf
    cache_size: int = 1000
    batch_size: int = 100
    
    # DuckDB specific settings
    db_path: str = "vectors.db"
    table_name: str = "vectors"
    
    # Index settings
    index_m: int = 16  # HNSW M parameter
    index_ef_construction: int = 200  # HNSW ef_construction parameter
    index_ef_search: int = 100  # HNSW ef parameter for search

@dataclass
class SearchConfig:
    """Configuration for hybrid search."""
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    top_k: int = 10
    min_score: float = 0.0
    
    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75
    
    # Cache settings
    use_cache: bool = True
    cache_ttl: int = 3600  # seconds

@dataclass
class VectorMetadata:
    """Metadata for stored vectors."""
    id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=float)

@dataclass
class SearchResult:
    """Search result with combined score."""
    id: str
    text: str
    semantic_score: float
    bm25_score: float
    combined_score: float
    metadata: Dict = field(default_factory=dict)
    
    @property
    def normalized_score(self) -> float:
        """Get normalized combined score."""
        return (self.combined_score + 1) / 2  # Scale to [0,1]
