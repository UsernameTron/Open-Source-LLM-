"""
Main document processing pipeline.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import aiofiles
from core.metrics import performance_monitor
from .structure import (
    DocumentStructure,
    StructureExtractor
)
from .sentiment import SentimentAnalyzer
from .summary import HierarchicalSummarizer

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents with structure extraction, sentiment, and summaries."""
    
    def __init__(
        self,
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        summary_model: str = "facebook/bart-large-cnn",
        device: str = "mps",  # Use Metal Performance Shaders for M4
        batch_size: int = 8
    ):
        """
        Initialize document processor.
        
        Args:
            sentiment_model: Model for sentiment analysis
            summary_model: Model for summarization
            device: Device to run models on
            batch_size: Batch size for inference
        """
        self.extractor = StructureExtractor()
        self.sentiment_analyzer = SentimentAnalyzer(
            model_name=sentiment_model,
            device=device,
            batch_size=batch_size
        )
        self.summarizer = HierarchicalSummarizer(
            model_name=summary_model,
            device=device,
            batch_size=batch_size
        )
        logger.info("Initialized document processor")
        
    @performance_monitor
    async def process_file(
        self,
        file_path: str,
        analyze_sentiment: bool = True,
        generate_summaries: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process document file.
        
        Args:
            file_path: Path to document file
            analyze_sentiment: Whether to analyze sentiment
            generate_summaries: Whether to generate summaries
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with processing results
        """
        try:
            path = Path(file_path)
            
            # Read file content
            async with aiofiles.open(path, 'rb') as f:
                content = await f.read()
                
            # Extract structure based on file type
            suffix = path.suffix.lower()
            if suffix == '.pdf':
                document = self.extractor.extract_from_pdf(content)
            elif suffix == '.docx':
                document = self.extractor.extract_from_docx(content)
            elif suffix in ['.html', '.htm']:
                document = self.extractor.extract_from_html(
                    content.decode('utf-8')
                )
            elif suffix == '.csv':
                document = self.extractor.extract_from_csv(content)
            elif suffix == '.md':
                document = self.extractor.extract_from_markdown(content)
            elif suffix == '.txt':
                document = self.extractor.extract_from_text(content)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
                
            # Analyze sentiment
            if analyze_sentiment:
                document = await self.sentiment_analyzer.analyze_document(
                    document,
                    **kwargs
                )
                
            # Generate summaries
            if generate_summaries:
                document = await self.summarizer.summarize_document(
                    document,
                    **kwargs
                )
                
            # Return results
            return {
                'structure': document.get_structure_tree(),
                'sentiment': (
                    await self.sentiment_analyzer.get_sentiment_summary(document)
                    if analyze_sentiment else None
                ),
                'summaries': (
                    await self.summarizer.get_summary_tree(document)
                    if generate_summaries else None
                )
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
            
    @performance_monitor
    async def process_text(
        self,
        text: str,
        analyze_sentiment: bool = True,
        generate_summaries: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process text content directly.
        
        Args:
            text: Text content to process
            analyze_sentiment: Whether to analyze sentiment
            generate_summaries: Whether to generate summaries
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Create simple document structure
            document = DocumentStructure(title="Text Document")
            
            # Add single section with content
            from .structure import Section, Element, ElementType
            section = Section(
                title="Main Content",
                level=1,
                elements=[
                    Element(
                        type=ElementType.TEXT,
                        content=text
                    )
                ]
            )
            document.add_section(section)
            
            # Process document
            if analyze_sentiment:
                document = await self.sentiment_analyzer.analyze_document(
                    document,
                    **kwargs
                )
                
            if generate_summaries:
                document = await self.summarizer.summarize_document(
                    document,
                    **kwargs
                )
                
            # Return results
            return {
                'structure': document.get_structure_tree(),
                'sentiment': (
                    await self.sentiment_analyzer.get_sentiment_summary(document)
                    if analyze_sentiment else None
                ),
                'summaries': (
                    await self.summarizer.get_summary_tree(document)
                    if generate_summaries else None
                )
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
