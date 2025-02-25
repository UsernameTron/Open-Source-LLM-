"""
Sectional sentiment analysis for documents.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from transformers import pipeline
from core.metrics import performance_monitor
from .structure import DocumentStructure, Section, Element

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes sentiment in document sections."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        batch_size: int = 8,
        device: str = "mps"  # Use Metal Performance Shaders for M4
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
            device: Device to run model on
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._pipeline = None
        logger.info(f"Initialized sentiment analyzer with {model_name}")
        
    @property
    def pipeline(self):
        """Lazy load sentiment pipeline."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device
            )
        return self._pipeline
        
    @performance_monitor
    async def analyze_document(
        self,
        document: DocumentStructure,
        min_text_length: int = 10
    ) -> DocumentStructure:
        """
        Analyze sentiment throughout document.
        
        Args:
            document: Document to analyze
            min_text_length: Minimum text length for analysis
            
        Returns:
            Document with sentiment scores
        """
        try:
            # Get all sections
            sections = document.get_all_sections()
            
            # Analyze sections in batches
            for i in range(0, len(sections), self.batch_size):
                batch = sections[i:i + self.batch_size]
                
                # Get text content for each section
                texts = []
                valid_sections = []
                
                for section in batch:
                    text = section.get_text_content()
                    if len(text) >= min_text_length:
                        texts.append(text)
                        valid_sections.append(section)
                        
                if not texts:
                    continue
                    
                # Run sentiment analysis
                results = self.pipeline(texts)
                
                # Update section sentiment scores
                for section, result in zip(valid_sections, results):
                    score = (
                        1.0 if result['label'] == 'POSITIVE'
                        else 0.0
                    )
                    section.sentiment_score = score
                    
                    # Update element scores
                    for element in section.elements:
                        if (
                            element.type == "text" and
                            len(element.content) >= min_text_length
                        ):
                            element_result = self.pipeline(
                                element.content
                            )[0]
                            element.sentiment_score = (
                                1.0 if element_result['label'] == 'POSITIVE'
                                else 0.0
                            )
                            element.confidence = element_result['score']
                            
            # Calculate document-level sentiment
            section_scores = [
                s.sentiment_score for s in sections
                if s.sentiment_score is not None
            ]
            if section_scores:
                document.sentiment_score = np.mean(section_scores)
                
            return document
            
        except Exception as e:
            logger.error(f"Error analyzing document sentiment: {str(e)}")
            raise
            
    @performance_monitor
    async def analyze_sections(
        self,
        sections: List[Section],
        min_text_length: int = 10
    ) -> List[Section]:
        """
        Analyze sentiment for a list of sections.
        
        Args:
            sections: List of sections to analyze
            min_text_length: Minimum text length for analysis
            
        Returns:
            Sections with sentiment scores
        """
        try:
            # Process sections in batches
            for i in range(0, len(sections), self.batch_size):
                batch = sections[i:i + self.batch_size]
                
                # Get text content
                texts = []
                valid_sections = []
                
                for section in batch:
                    text = section.get_text_content()
                    if len(text) >= min_text_length:
                        texts.append(text)
                        valid_sections.append(section)
                        
                if not texts:
                    continue
                    
                # Run sentiment analysis
                results = self.pipeline(texts)
                
                # Update sentiment scores
                for section, result in zip(valid_sections, results):
                    score = (
                        1.0 if result['label'] == 'POSITIVE'
                        else 0.0
                    )
                    section.sentiment_score = score
                    
            return sections
            
        except Exception as e:
            logger.error(f"Error analyzing section sentiment: {str(e)}")
            raise
            
    @performance_monitor
    async def get_sentiment_summary(
        self,
        document: DocumentStructure
    ) -> Dict:
        """
        Get sentiment summary statistics.
        
        Args:
            document: Analyzed document
            
        Returns:
            Dictionary of sentiment statistics
        """
        try:
            sections = document.get_all_sections()
            scores = [
                s.sentiment_score for s in sections
                if s.sentiment_score is not None
            ]
            
            if not scores:
                return {
                    'error': 'No sentiment scores available'
                }
                
            return {
                'document_score': document.sentiment_score,
                'num_sections': len(scores),
                'mean_score': np.mean(scores),
                'std_dev': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'positive_sections': sum(1 for s in scores if s > 0.5),
                'negative_sections': sum(1 for s in scores if s <= 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error generating sentiment summary: {str(e)}")
            raise
