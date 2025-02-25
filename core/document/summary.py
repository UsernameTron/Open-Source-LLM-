"""
Hierarchical document summarization.
"""

import logging
from typing import List, Dict, Optional
from transformers import pipeline
from core.metrics import performance_monitor
from .structure import DocumentStructure, Section

logger = logging.getLogger(__name__)

class HierarchicalSummarizer:
    """Generates hierarchical summaries of documents."""
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        batch_size: int = 4,
        device: str = "mps",  # Use Metal Performance Shaders for M4
        max_length: int = 150,
        min_length: int = 50
    ):
        """
        Initialize summarizer.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
            device: Device to run model on
            max_length: Maximum summary length
            min_length: Minimum summary length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.min_length = min_length
        self._pipeline = None
        logger.info(f"Initialized summarizer with {model_name}")
        
    @property
    def pipeline(self):
        """Lazy load summarization pipeline."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "summarization",
                model=self.model_name,
                device=self.device
            )
        return self._pipeline
        
    @performance_monitor
    async def summarize_document(
        self,
        document: DocumentStructure,
        max_section_length: Optional[int] = None
    ) -> DocumentStructure:
        """
        Generate hierarchical summary of document.
        
        Args:
            document: Document to summarize
            max_section_length: Optional maximum section length
            
        Returns:
            Document with summaries
        """
        try:
            # Summarize sections bottom-up
            for section in reversed(document.get_all_sections()):
                # Combine subsection summaries with section content
                content = section.get_text_content()
                
                if section.subsections:
                    subsection_summaries = [
                        s.summary for s in section.subsections
                        if s.summary
                    ]
                    if subsection_summaries:
                        content += "\n\nSubsection Summaries:\n"
                        content += "\n".join(subsection_summaries)
                        
                # Check content length
                if max_section_length:
                    content = content[:max_section_length]
                    
                # Generate summary
                if len(content) > self.min_length:
                    summary = self.pipeline(
                        content,
                        max_length=self.max_length,
                        min_length=self.min_length,
                        do_sample=False
                    )[0]['summary_text']
                    
                    section.summary = summary
                    
            # Generate document summary
            document_content = document.get_text_content()
            if len(document_content) > self.min_length:
                document.summary = self.pipeline(
                    document_content,
                    max_length=self.max_length * 2,
                    min_length=self.min_length * 2,
                    do_sample=False
                )[0]['summary_text']
                
            return document
            
        except Exception as e:
            logger.error(f"Error generating document summary: {str(e)}")
            raise
            
    @performance_monitor
    async def summarize_sections(
        self,
        sections: List[Section],
        max_length: Optional[int] = None
    ) -> List[Section]:
        """
        Summarize a list of sections.
        
        Args:
            sections: Sections to summarize
            max_length: Optional maximum length per section
            
        Returns:
            Sections with summaries
        """
        try:
            # Process sections in batches
            for i in range(0, len(sections), self.batch_size):
                batch = sections[i:i + self.batch_size]
                
                # Get content for each section
                texts = []
                valid_sections = []
                
                for section in batch:
                    content = section.get_text_content()
                    if max_length:
                        content = content[:max_length]
                        
                    if len(content) > self.min_length:
                        texts.append(content)
                        valid_sections.append(section)
                        
                if not texts:
                    continue
                    
                # Generate summaries
                summaries = self.pipeline(
                    texts,
                    max_length=self.max_length,
                    min_length=self.min_length,
                    do_sample=False
                )
                
                # Update sections
                for section, summary in zip(valid_sections, summaries):
                    section.summary = summary['summary_text']
                    
            return sections
            
        except Exception as e:
            logger.error(f"Error summarizing sections: {str(e)}")
            raise
            
    @performance_monitor
    async def get_summary_tree(
        self,
        document: DocumentStructure
    ) -> Dict:
        """
        Get tree of document summaries.
        
        Args:
            document: Summarized document
            
        Returns:
            Dictionary with summary tree
        """
        try:
            def section_to_dict(section: Section) -> Dict:
                return {
                    'title': section.title,
                    'summary': section.summary,
                    'subsections': [
                        section_to_dict(s) for s in section.subsections
                        if s.summary
                    ]
                }
                
            return {
                'title': document.title,
                'summary': document.summary,
                'sections': [
                    section_to_dict(s) for s in document.sections
                    if s.summary
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating summary tree: {str(e)}")
            raise
