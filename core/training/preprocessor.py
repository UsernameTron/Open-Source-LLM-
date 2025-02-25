"""Data preprocessing for fine-tuning."""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import Counter
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class TextPattern:
    """Represents a detected text pattern."""
    pattern_type: str
    content: str
    frequency: int
    context: Optional[str] = None

@dataclass
class ProcessedExample:
    """Processed training example."""
    original_text: str
    structural_patterns: List[TextPattern]
    linguistic_devices: List[TextPattern]
    transitions: List[TextPattern]
    citations: List[TextPattern]
    one_liners: List[TextPattern]
    metadata: Dict[str, Any]

class DataPreprocessor:
    """Preprocesses training data for fine-tuning."""
    
    def __init__(self):
        """Initialize preprocessor."""
        # Load NLP models
        self.nlp = spacy.load("en_core_web_lg")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Download required NLTK data
        nltk.download("punkt")
        
        # Initialize pattern matchers
        self._init_patterns()
        
    def _init_patterns(self):
        """Initialize pattern matching rules."""
        self.transition_markers = [
            "but", "however", "yet", "still",
            "meanwhile", "conversely", "instead",
            "on the other hand", "in contrast",
            "despite", "nevertheless"
        ]
        
        self.citation_patterns = [
            r"\d+%",  # Percentage patterns
            r"according to .+",  # Attribution patterns
            r"(?:reports?|studies?|research|data) shows?",  # Research references
            r"\[[\d,]+\]",  # Citation brackets
        ]
        
        self.one_liner_markers = [
            "One-Liner:",
            "The bottom line:",
            "Key takeaway:",
            "In summary:",
            "TL;DR:"
        ]
        
    def process_example(self, text: str) -> ProcessedExample:
        """Process a single training example."""
        try:
            # Basic text cleanup
            text = self._clean_text(text)
            
            # Process with spaCy
            doc = self.nlp(text)
            
            return ProcessedExample(
                original_text=text,
                structural_patterns=self._extract_structural_patterns(doc),
                linguistic_devices=self._extract_linguistic_devices(doc),
                transitions=self._extract_transitions(doc),
                citations=self._extract_citations(doc),
                one_liners=self._extract_one_liners(doc),
                metadata={
                    "length": len(text),
                    "sentences": len(list(doc.sents)),
                    "paragraphs": text.count("\n\n") + 1
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing example: {str(e)}")
            raise
            
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = text.strip()
        return text
        
    def _extract_structural_patterns(self, doc) -> List[TextPattern]:
        """Extract common structural patterns."""
        patterns = []
        
        # Detect section headers
        for sent in doc.sents:
            if sent.text.isupper() or sent.text.endswith(":"):
                patterns.append(TextPattern(
                    pattern_type="header",
                    content=sent.text,
                    frequency=1
                ))
                
        # Detect bullet points
        bullet_points = re.findall(r"[-•*]\s+.+", doc.text)
        if bullet_points:
            patterns.append(TextPattern(
                pattern_type="bullet_list",
                content="\n".join(bullet_points),
                frequency=len(bullet_points)
            ))
            
        return patterns
        
    def _extract_linguistic_devices(self, doc) -> List[TextPattern]:
        """Extract recurring linguistic devices."""
        devices = []
        
        # Detect rhetorical questions
        questions = [sent.text for sent in doc.sents 
                    if sent.text.endswith("?")]
        if questions:
            devices.append(TextPattern(
                pattern_type="rhetorical_question",
                content="\n".join(questions),
                frequency=len(questions)
            ))
            
        # Detect metaphors (simplified)
        for sent in doc.sents:
            if "like" in sent.text.lower() or "as" in sent.text.lower():
                devices.append(TextPattern(
                    pattern_type="metaphor",
                    content=sent.text,
                    frequency=1
                ))
                
        return devices
        
    def _extract_transitions(self, doc) -> List[TextPattern]:
        """Extract transition phrases."""
        transitions = []
        
        for marker in self.transition_markers:
            matches = re.finditer(
                f"\\b{marker}\\b",
                doc.text,
                re.IGNORECASE
            )
            for match in matches:
                # Get context around transition
                start = max(0, match.start() - 50)
                end = min(len(doc.text), match.end() + 50)
                context = doc.text[start:end]
                
                transitions.append(TextPattern(
                    pattern_type="transition",
                    content=marker,
                    frequency=1,
                    context=context
                ))
                
        return transitions
        
    def _extract_citations(self, doc) -> List[TextPattern]:
        """Extract statistical citations."""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, doc.text)
            for match in matches:
                # Get context around citation
                start = max(0, match.start() - 50)
                end = min(len(doc.text), match.end() + 50)
                context = doc.text[start:end]
                
                citations.append(TextPattern(
                    pattern_type="citation",
                    content=match.group(),
                    frequency=1,
                    context=context
                ))
                
        return citations
        
    def _extract_one_liners(self, doc) -> List[TextPattern]:
        """Extract one-liner conclusions."""
        one_liners = []
        
        # Split into sentences
        sentences = sent_tokenize(doc.text)
        
        # Look for marked one-liners
        for marker in self.one_liner_markers:
            for sent in sentences:
                if marker.lower() in sent.lower():
                    one_liners.append(TextPattern(
                        pattern_type="one_liner",
                        content=sent,
                        frequency=1
                    ))
                    
        # Also detect potential unmarked one-liners (last sentences of sections)
        sections = doc.text.split("\n\n")
        for section in sections:
            if section.strip():
                last_sent = sent_tokenize(section)[-1]
                if len(last_sent) < 150:  # Reasonable length for one-liner
                    one_liners.append(TextPattern(
                        pattern_type="potential_one_liner",
                        content=last_sent,
                        frequency=1
                    ))
                    
        return one_liners
        
    def analyze_corpus_patterns(
        self,
        examples: List[str]
    ) -> Dict[str, Counter]:
        """Analyze patterns across entire corpus."""
        pattern_counts = {
            "structural": Counter(),
            "linguistic": Counter(),
            "transitions": Counter(),
            "citations": Counter(),
            "one_liners": Counter()
        }
        
        for text in examples:
            processed = self.process_example(text)
            
            # Update counts
            for pattern in processed.structural_patterns:
                pattern_counts["structural"][pattern.content] += 1
                
            for pattern in processed.linguistic_devices:
                pattern_counts["linguistic"][pattern.content] += 1
                
            for pattern in processed.transitions:
                pattern_counts["transitions"][pattern.content] += 1
                
            for pattern in processed.citations:
                pattern_counts["citations"][pattern.content] += 1
                
            for pattern in processed.one_liners:
                pattern_counts["one_liners"][pattern.content] += 1
                
        return pattern_counts
