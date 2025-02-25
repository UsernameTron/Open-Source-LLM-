"""
Hybrid search implementation combining BM25 and semantic similarity.
"""

import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from core.metrics import performance_monitor
from core.storage.types import SearchConfig, SearchResult, VectorMetadata

logger = logging.getLogger(__name__)

@dataclass
class SearchStats:
    """Statistics about the search operation."""
    semantic_time_ms: float
    bm25_time_ms: float
    total_time_ms: float
    num_results: int
    avg_score: float

class HybridSearch:
    def __init__(self, config: SearchConfig):
        """
        Initialize hybrid search.
        
        Args:
            config: Search configuration
        """
        self.config = config
        self._bm25 = None
        self._text_corpus = []
        self._metadata_map: Dict[int, VectorMetadata] = {}
        logger.info("Initialized hybrid search")
        
    @performance_monitor
    def update_corpus(
        self,
        texts: List[str],
        metadata_list: List[VectorMetadata]
    ) -> None:
        """
        Update text corpus and BM25 index.
        
        Args:
            texts: List of text documents
            metadata_list: List of metadata for each document
        """
        try:
            # Update text corpus
            self._text_corpus = texts
            
            # Update metadata map
            self._metadata_map = {
                i: metadata for i, metadata in enumerate(metadata_list)
            }
            
            # Create BM25 index
            tokenized_corpus = [text.split() for text in texts]
            self._bm25 = BM25Okapi(
                tokenized_corpus,
                k1=self.config.k1,
                b=self.config.b
            )
            
            logger.info(f"Updated corpus with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Error updating corpus: {str(e)}")
            raise
            
    @performance_monitor
    def search(
        self,
        query_text: str,
        semantic_scores: np.ndarray,
        indices: np.ndarray
    ) -> Tuple[List[SearchResult], SearchStats]:
        """
        Perform hybrid search combining BM25 and semantic similarity.
        
        Args:
            query_text: Text query for BM25
            semantic_scores: Semantic similarity scores from vector search
            indices: Indices from vector search
            
        Returns:
            Tuple of (search results, search statistics)
        """
        try:
            import time
            start_time = time.time()
            
            # Get BM25 scores
            bm25_start = time.time()
            bm25_scores = self._bm25.get_scores(query_text.split())
            bm25_time = (time.time() - bm25_start) * 1000
            
            # Combine scores and create results
            results = []
            total_score = 0
            
            for i, (score, idx) in enumerate(zip(semantic_scores, indices)):
                # Get metadata
                metadata = self._metadata_map[idx]
                
                # Calculate combined score
                bm25_score = bm25_scores[idx]
                combined_score = (
                    self.config.semantic_weight * score +
                    self.config.bm25_weight * bm25_score
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
                
                if result.normalized_score >= self.config.min_score:
                    results.append(result)
                    total_score += result.normalized_score
                    
            # Sort by combined score
            results.sort(key=lambda x: x.combined_score, reverse=True)
            results = results[:self.config.top_k]
            
            # Calculate statistics
            total_time = (time.time() - start_time) * 1000
            semantic_time = total_time - bm25_time
            avg_score = total_score / len(results) if results else 0
            
            stats = SearchStats(
                semantic_time_ms=semantic_time,
                bm25_time_ms=bm25_time,
                total_time_ms=total_time,
                num_results=len(results),
                avg_score=avg_score
            )
            
            logger.info(
                f"Hybrid search completed in {total_time:.2f}ms with "
                f"{len(results)} results"
            )
            return results, stats
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            raise
            
    def _normalize_bm25_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize BM25 scores to [0,1] range."""
        if len(scores) == 0:
            return scores
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.ones_like(scores)
            
        return (scores - min_score) / (max_score - min_score)
        
    @performance_monitor
    def explain_result(
        self,
        result: SearchResult,
        query_text: str
    ) -> Dict:
        """
        Explain the scoring for a search result.
        
        Args:
            result: Search result to explain
            query_text: Original query text
            
        Returns:
            Dictionary with score explanation
        """
        try:
            # Get token contributions
            query_tokens = query_text.split()
            doc_tokens = result.text.split()
            
            token_scores = []
            for token in query_tokens:
                if token in doc_tokens:
                    score = self._bm25.get_scores([token])[0]
                    token_scores.append({
                        'token': token,
                        'score': float(score)
                    })
            
            # Sort by score
            token_scores.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                'semantic': {
                    'score': result.semantic_score,
                    'weight': self.config.semantic_weight,
                    'contribution': result.semantic_score * self.config.semantic_weight
                },
                'bm25': {
                    'score': result.bm25_score,
                    'weight': self.config.bm25_weight,
                    'contribution': result.bm25_score * self.config.bm25_weight,
                    'token_scores': token_scores
                },
                'combined': {
                    'score': result.combined_score,
                    'normalized_score': result.normalized_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error explaining search result: {str(e)}")
            raise
