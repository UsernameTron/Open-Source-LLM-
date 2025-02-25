"""
Document structure representation and extraction.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import csv
import io
import pandas as pd

logger = logging.getLogger(__name__)

class ElementType(Enum):
    """Types of document elements."""
    TEXT = "text"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    EQUATION = "equation"
    REFERENCE = "reference"

@dataclass
class Element:
    """Base class for document elements."""
    type: ElementType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sentiment_score: Optional[float] = None
    confidence: Optional[float] = None

@dataclass
class Section:
    """Document section with hierarchical structure."""
    title: str
    level: int
    elements: List[Element] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    sentiment_score: Optional[float] = None
    
    def add_element(self, element: Element):
        """Add element to section."""
        self.elements.append(element)
        
    def add_subsection(self, section: 'Section'):
        """Add subsection."""
        self.subsections.append(section)
        
    def get_text_content(self) -> str:
        """Get all text content in section."""
        content = [self.title]
        
        # Add element content
        for element in self.elements:
            if element.type in [ElementType.TEXT, ElementType.HEADING]:
                content.append(element.content)
                
        # Add subsection content
        for subsection in self.subsections:
            content.append(subsection.get_text_content())
            
        return "\n\n".join(content)

@dataclass
class DocumentStructure:
    """Complete document structure."""
    title: str
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    sentiment_score: Optional[float] = None
    
    def add_section(self, section: Section):
        """Add top-level section."""
        self.sections.append(section)
        
    def get_section_by_title(
        self,
        title: str,
        case_sensitive: bool = False
    ) -> Optional[Section]:
        """Find section by title."""
        def search_sections(sections: List[Section]) -> Optional[Section]:
            for section in sections:
                # Check title match
                section_title = section.title
                search_title = title
                if not case_sensitive:
                    section_title = section_title.lower()
                    search_title = search_title.lower()
                    
                if section_title == search_title:
                    return section
                    
                # Search subsections
                result = search_sections(section.subsections)
                if result:
                    return result
            return None
            
        return search_sections(self.sections)
        
    def get_all_sections(self) -> List[Section]:
        """Get flattened list of all sections."""
        sections = []
        
        def collect_sections(section_list: List[Section]):
            for section in section_list:
                sections.append(section)
                collect_sections(section.subsections)
                
        collect_sections(self.sections)
        return sections
        
    def get_text_content(self) -> str:
        """Get all document text content."""
        content = [self.title]
        
        for section in self.sections:
            content.append(section.get_text_content())
            
        return "\n\n".join(content)
        
    def get_structure_tree(self) -> Dict:
        """Get tree representation of document structure."""
        def section_to_dict(section: Section) -> Dict:
            return {
                'title': section.title,
                'level': section.level,
                'num_elements': len(section.elements),
                'sentiment': section.sentiment_score,
                'subsections': [
                    section_to_dict(s) for s in section.subsections
                ]
            }
            
        return {
            'title': self.title,
            'num_sections': len(self.sections),
            'sentiment': self.sentiment_score,
            'sections': [section_to_dict(s) for s in self.sections]
        }

class StructureExtractor:
    """Extracts document structure from various formats."""
    
    def __init__(self):
        """Initialize structure extractor."""
        self._current_section = None
        self._current_level = 0
        logger.info("Initialized structure extractor")
        
    def extract_from_pdf(
        self,
        pdf_content: bytes,
        **kwargs
    ) -> DocumentStructure:
        """Extract structure from PDF content."""
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer, LTChar
            
            document = DocumentStructure(title="")
            current_section = None
            
            # Extract pages
            for page_layout in extract_pages(pdf_content):
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text = element.get_text().strip()
                        if not text:
                            continue
                            
                        # Detect headings by font size
                        max_font_size = 0
                        for text_line in element:
                            for char in text_line:
                                if isinstance(char, LTChar):
                                    max_font_size = max(
                                        max_font_size,
                                        char.size
                                    )
                                    
                        # Create section or add content
                        if max_font_size > 12:  # Heading threshold
                            level = 1 if max_font_size > 16 else 2
                            
                            if level == 1:
                                # Top-level section
                                current_section = Section(
                                    title=text,
                                    level=level
                                )
                                document.add_section(current_section)
                            else:
                                # Subsection
                                subsection = Section(
                                    title=text,
                                    level=level
                                )
                                if current_section:
                                    current_section.add_subsection(subsection)
                                current_section = subsection
                        else:
                            # Regular text content
                            text_element = Element(
                                type=ElementType.TEXT,
                                content=text
                            )
                            if current_section:
                                current_section.add_element(text_element)
                                
            # Set document title from first heading
            if document.sections:
                document.title = document.sections[0].title
                
            return document
            
        except Exception as e:
            logger.error(f"Error extracting PDF structure: {str(e)}")
            raise
            
    def extract_from_docx(
        self,
        docx_content: bytes,
        **kwargs
    ) -> DocumentStructure:
        """Extract structure from DOCX content."""
        try:
            from docx import Document
            
            doc = Document(docx_content)
            document = DocumentStructure(title="")
            current_section = None
            
            # Process paragraphs
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                    
                # Check if heading
                if paragraph.style.name.startswith('Heading'):
                    level = int(paragraph.style.name[-1])
                    
                    if level == 1:
                        # Top-level section
                        current_section = Section(
                            title=text,
                            level=level
                        )
                        document.add_section(current_section)
                    else:
                        # Subsection
                        subsection = Section(
                            title=text,
                            level=level
                        )
                        if current_section:
                            current_section.add_subsection(subsection)
                        current_section = subsection
                else:
                    # Regular text
                    text_element = Element(
                        type=ElementType.TEXT,
                        content=text
                    )
                    if current_section:
                        current_section.add_element(text_element)
                        
            # Set document title
            if document.sections:
                document.title = document.sections[0].title
                
            return document
            
        except Exception as e:
            logger.error(f"Error extracting DOCX structure: {str(e)}")
            raise
            
    def extract_from_html(
        self,
        html_content: str,
        **kwargs
    ) -> DocumentStructure:
        """Extract structure from HTML content."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            document = DocumentStructure(title="")
            current_section = None
            
            # Get title
            title_tag = soup.find('title')
            if title_tag:
                document.title = title_tag.text.strip()
                
            # Process headings and content
            for element in soup.find_all(['h1', 'h2', 'h3', 'p']):
                if element.name.startswith('h'):
                    level = int(element.name[1])
                    text = element.text.strip()
                    
                    if level == 1:
                        current_section = Section(
                            title=text,
                            level=level
                        )
                        document.add_section(current_section)
                    else:
                        subsection = Section(
                            title=text,
                            level=level
                        )
                        if current_section:
                            current_section.add_subsection(subsection)
                        current_section = subsection
                else:
                    # Paragraph content
                    text = element.text.strip()
                    if text:
                        text_element = Element(
                            type=ElementType.TEXT,
                            content=text
                        )
                        if current_section:
                            current_section.add_element(text_element)
                            
            return document
            
        except Exception as e:
            logger.error(f"Error extracting HTML structure: {str(e)}")
            raise
            
    def extract_from_csv(
        self,
        csv_content: bytes,
        **kwargs
    ) -> DocumentStructure:
        """Extract structure from CSV content."""
        try:
            # Read CSV content
            df = pd.read_csv(io.BytesIO(csv_content))
            document = DocumentStructure(title="CSV Document")
            
            # Create sections for each column
            for column in df.columns:
                section = Section(
                    title=f"Column: {column}",
                    level=1
                )
                
                # Add column statistics
                stats = df[column].describe()
                stats_text = "\n".join([
                    f"{stat}: {value}"
                    for stat, value in stats.items()
                ])
                
                section.add_element(Element(
                    type=ElementType.TEXT,
                    content=f"Statistics:\n{stats_text}"
                ))
                
                # Add sample values
                sample_values = df[column].head().tolist()
                sample_text = "\n".join([
                    f"- {value}" for value in sample_values
                ])
                
                section.add_element(Element(
                    type=ElementType.TEXT,
                    content=f"Sample Values:\n{sample_text}"
                ))
                
                document.add_section(section)
                
            # Add summary section
            summary_section = Section(
                title="Dataset Summary",
                level=1
            )
            
            summary_section.add_element(Element(
                type=ElementType.TEXT,
                content=f"Total Rows: {len(df)}\nTotal Columns: {len(df.columns)}"
            ))
            
            document.add_section(summary_section)
            return document
            
        except Exception as e:
            logger.error(f"Error extracting CSV structure: {str(e)}")
            raise
            
    def extract_from_markdown(
        self,
        md_content: bytes,
        **kwargs
    ) -> DocumentStructure:
        """Extract structure from Markdown content."""
        try:
            import markdown
            from bs4 import BeautifulSoup
            
            # Convert markdown to HTML
            html = markdown.markdown(md_content.decode('utf-8'))
            
            # Parse HTML structure
            soup = BeautifulSoup(html, 'html.parser')
            document = DocumentStructure(title="")
            current_section = None
            
            # Get title from first heading
            first_heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if first_heading:
                document.title = first_heading.text.strip()
            
            # Process elements
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'code']):
                if element.name.startswith('h'):
                    # Handle heading
                    level = int(element.name[1])
                    text = element.text.strip()
                    
                    if level == 1:
                        current_section = Section(
                            title=text,
                            level=level
                        )
                        document.add_section(current_section)
                    else:
                        subsection = Section(
                            title=text,
                            level=level
                        )
                        if current_section:
                            current_section.add_subsection(subsection)
                        current_section = subsection
                else:
                    # Handle content elements
                    if current_section:
                        element_type = ElementType.TEXT
                        if element.name == 'code':
                            element_type = ElementType.CODE
                        elif element.name in ['ul', 'ol']:
                            element_type = ElementType.LIST
                            
                        current_section.add_element(Element(
                            type=element_type,
                            content=element.text.strip()
                        ))
                        
            return document
            
        except Exception as e:
            logger.error(f"Error extracting Markdown structure: {str(e)}")
            raise
            
    def extract_from_text(
        self,
        text_content: bytes,
        **kwargs
    ) -> DocumentStructure:
        """Extract structure from plain text content."""
        try:
            # Decode text content
            text = text_content.decode('utf-8')
            
            # Create document
            document = DocumentStructure(title="Text Document")
            current_section = None
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                # Try to detect section breaks
                lines = paragraph.split('\n')
                first_line = lines[0].strip()
                
                # Heuristic: If line is short and followed by empty line,
                # treat as heading
                if (
                    len(first_line) < 100 and
                    len(lines) > 1 and
                    not lines[1].strip() and
                    not first_line.endswith('.') and
                    not first_line.endswith('?')
                ):
                    # Create new section
                    if not document.title:
                        document.title = first_line
                    else:
                        current_section = Section(
                            title=first_line,
                            level=1
                        )
                        document.add_section(current_section)
                else:
                    # Add as content
                    if not current_section:
                        current_section = Section(
                            title="Main Content",
                            level=1
                        )
                        document.add_section(current_section)
                        
                    current_section.add_element(Element(
                        type=ElementType.TEXT,
                        content=paragraph
                    ))
                    
            return document
            
        except Exception as e:
            logger.error(f"Error extracting text structure: {str(e)}")
            raise
