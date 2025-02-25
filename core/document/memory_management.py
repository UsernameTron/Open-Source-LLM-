"""Memory management utilities for document processing."""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content."""
    content: str
    start_offset: int
    end_offset: int
    metadata: Dict[str, Any]

class MemoryManager:
    """Manages memory usage during document processing."""
    
    def __init__(
        self,
        max_chunk_size: int = 5 * 1024 * 1024,  # 5MB default
        overlap_size: int = 1024,  # 1KB overlap
        max_memory_usage: float = 0.8  # 80% of available memory
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.max_memory_usage = max_memory_usage
        
    def create_overlapping_chunks(
        self,
        document_path: str
    ) -> List[DocumentChunk]:
        """Create overlapping chunks from a document."""
        try:
            document_size = os.path.getsize(document_path)
            chunks = []
            
            if document_size > self.max_chunk_size:
                with open(document_path, 'r') as f:
                    start_offset = 0
                    
                    while start_offset < document_size:
                        # Calculate chunk boundaries
                        end_offset = min(
                            start_offset + self.max_chunk_size,
                            document_size
                        )
                        
                        # Read chunk with overlap
                        f.seek(start_offset)
                        content = f.read(end_offset - start_offset)
                        
                        # Create chunk
                        chunk = DocumentChunk(
                            content=content,
                            start_offset=start_offset,
                            end_offset=end_offset,
                            metadata={
                                'is_first': start_offset == 0,
                                'is_last': end_offset == document_size
                            }
                        )
                        chunks.append(chunk)
                        
                        # Move to next chunk with overlap
                        start_offset = end_offset - self.overlap_size
                        
                logger.info(
                    f"Split document into {len(chunks)} chunks "
                    f"with {self.overlap_size}B overlap"
                )
                
            else:
                # Small document, process as single chunk
                with open(document_path, 'r') as f:
                    content = f.read()
                    chunks.append(DocumentChunk(
                        content=content,
                        start_offset=0,
                        end_offset=document_size,
                        metadata={
                            'is_first': True,
                            'is_last': True
                        }
                    ))
                logger.info("Processing document as single chunk")
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating document chunks: {str(e)}")
            raise
            
    def merge_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]],
        merge_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Merge results from multiple chunks."""
        try:
            if not chunk_results:
                return {}
                
            if len(chunk_results) == 1:
                return chunk_results[0]
                
            # Default to simple concatenation
            if not merge_strategy:
                merged = {}
                for result in chunk_results:
                    for key, value in result.items():
                        if key not in merged:
                            merged[key] = []
                        if isinstance(value, list):
                            merged[key].extend(value)
                        else:
                            merged[key].append(value)
                return merged
                
            # Custom merge strategies can be added here
            if merge_strategy == "weighted":
                return self._weighted_merge(chunk_results)
            elif merge_strategy == "hierarchical":
                return self._hierarchical_merge(chunk_results)
                
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            
        except Exception as e:
            logger.error(f"Error merging chunk results: {str(e)}")
            raise
            
    def _weighted_merge(
        self,
        chunk_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge results with weights based on chunk size."""
        total_size = sum(
            len(result.get('content', ''))
            for result in chunk_results
        )
        
        merged = {}
        for result in chunk_results:
            weight = len(result.get('content', '')) / total_size
            for key, value in result.items():
                if key not in merged:
                    merged[key] = 0
                merged[key] += value * weight
                
        return merged
        
    def _hierarchical_merge(
        self,
        chunk_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge results maintaining hierarchical structure."""
        # Implement hierarchical merging logic here
        pass
