"""
DuckDB-based vector storage implementation with hybrid search capabilities.
"""

import logging
import time
from typing import List, Optional, Tuple, Dict
import numpy as np
import duckdb
import faiss
from rank_bm25 import BM25Okapi
from dataclasses import asdict
from core.metrics import performance_monitor
from core.storage.types import VectorConfig, SearchConfig, VectorMetadata, SearchResult

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, config: VectorConfig):
        """
        Initialize vector store with DuckDB backend.
        
        Args:
            config: Vector storage configuration
        """
        self.config = config
        self._conn = None
        self._index = None
        self._bm25 = None
        self._text_corpus = []
        self._setup_storage()
        self._init_indices()
        logger.info(f"Initialized vector store with dimension {config.dimension}")
        
    @performance_monitor
    def _setup_storage(self):
        """Setup DuckDB storage."""
        try:
            self._conn = duckdb.connect(self.config.db_path)
            
            # Create vectors table if not exists
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    id VARCHAR PRIMARY KEY,
                    vector DOUBLE[],
                    text VARCHAR,
                    metadata JSON,
                    timestamp DOUBLE
                )
            """)
            
            # Create index on id
            self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_id 
                ON {self.config.table_name}(id)
            """)
            
            logger.info("Successfully setup DuckDB storage")
            
        except Exception as e:
            logger.error(f"Error setting up storage: {str(e)}")
            raise
            
    @performance_monitor
    def _init_indices(self):
        """Initialize FAISS index and BM25."""
        try:
            # Create FAISS index
            if self.config.metric == "cosine":
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
                
            # Load existing vectors into index
            vectors = self._load_vectors()
            if len(vectors) > 0:
                self._index.add(vectors)
                
            # Initialize BM25
            self._text_corpus = self._load_text_corpus()
            self._bm25 = BM25Okapi([text.split() for text in self._text_corpus])
            
            logger.info("Successfully initialized indices")
            
        except Exception as e:
            logger.error(f"Error initializing indices: {str(e)}")
            raise
            
    def _load_vectors(self) -> np.ndarray:
        """Load existing vectors from storage."""
        result = self._conn.execute(f"""
            SELECT vector 
            FROM {self.config.table_name}
            ORDER BY timestamp ASC
        """).fetchall()
        
        if result:
            return np.array([r[0] for r in result])
        return np.array([])
        
    def _load_text_corpus(self) -> List[str]:
        """Load text corpus for BM25."""
        result = self._conn.execute(f"""
            SELECT text
            FROM {self.config.table_name}
            ORDER BY timestamp ASC
        """).fetchall()
        
        if result:
            return [r[0] for r in result]
        return []
        
    @performance_monitor
    async def store(
        self,
        vectors: np.ndarray,
        metadata_list: List[VectorMetadata]
    ) -> None:
        """
        Store vectors with metadata.
        
        Args:
            vectors: Array of vectors to store [n_vectors, dimension]
            metadata_list: List of metadata for each vector
        """
        try:
            if len(vectors) != len(metadata_list):
                raise ValueError("Number of vectors and metadata entries must match")
                
            # Add to FAISS index
            self._index.add(vectors)
            
            # Store in DuckDB
            timestamp = time.time()
            for vector, metadata in zip(vectors, metadata_list):
                self._conn.execute(f"""
                    INSERT INTO {self.config.table_name}
                    (id, vector, text, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metadata.id,
                    vector.tolist(),
                    metadata.text,
                    asdict(metadata),
                    timestamp
                ))
                
            # Update BM25 index
            new_texts = [m.text for m in metadata_list]
            self._text_corpus.extend(new_texts)
            self._bm25 = BM25Okapi([text.split() for text in self._text_corpus])
            
            logger.info(f"Successfully stored {len(vectors)} vectors")
            
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            raise
            
    @performance_monitor
    async def search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and BM25 scores.
        
        Args:
            query_vector: Query vector
            query_text: Query text for BM25
            config: Search configuration
            
        Returns:
            List of search results
        """
        try:
            config = config or SearchConfig()
            
            # Get semantic search scores
            semantic_scores, indices = self._semantic_search(
                query_vector,
                config.top_k
            )
            
            # Get BM25 scores
            bm25_scores = self._bm25.get_scores(query_text.split())
            
            # Combine scores
            results = []
            for i, (score, idx) in enumerate(zip(semantic_scores, indices)):
                # Get metadata
                metadata = self._get_metadata_by_index(idx)
                
                # Calculate combined score
                bm25_score = bm25_scores[idx]
                combined_score = (
                    config.semantic_weight * score +
                    config.bm25_weight * bm25_score
                )
                
                # Create search result
                result = SearchResult(
                    id=metadata.id,
                    text=metadata.text,
                    semantic_score=float(score),
                    bm25_score=float(bm25_score),
                    combined_score=float(combined_score),
                    metadata=metadata.metadata
                )
                
                if result.normalized_score >= config.min_score:
                    results.append(result)
                    
            # Sort by combined score
            results.sort(key=lambda x: x.combined_score, reverse=True)
            
            logger.info(f"Successfully performed hybrid search with {len(results)} results")
            return results[:config.top_k]
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise
            
    def _semantic_search(
        self,
        query_vector: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform semantic search using FAISS."""
        # Reshape query vector if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Perform search
        scores, indices = self._index.search(query_vector, k)
        return scores[0], indices[0]
        
    def _get_metadata_by_index(self, idx: int) -> VectorMetadata:
        """Get metadata for vector at given index."""
        result = self._conn.execute(f"""
            SELECT id, text, metadata, timestamp
            FROM {self.config.table_name}
            ORDER BY timestamp ASC
            LIMIT 1 OFFSET {idx}
        """).fetchone()
        
        return VectorMetadata(
            id=result[0],
            text=result[1],
            metadata=result[2],
            timestamp=result[3]
        )
        
    async def delete(self, ids: List[str]) -> None:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
        """
        try:
            # Delete from DuckDB
            placeholders = ", ".join(["?" for _ in ids])
            self._conn.execute(f"""
                DELETE FROM {self.config.table_name}
                WHERE id IN ({placeholders})
            """, ids)
            
            # Rebuild indices
            self._init_indices()
            
            logger.info(f"Successfully deleted {len(ids)} vectors")
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise
            
    def __del__(self):
        """Cleanup resources."""
        if self._conn:
            self._conn.close()
