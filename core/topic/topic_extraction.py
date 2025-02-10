"""Topic extraction module for the inference engine."""
from typing import List
import re
from collections import Counter

def extract_topics(text: str) -> List[str]:
    """Extract topics from the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of extracted topics
    """
    # Simple keyword extraction
    words = re.findall(r'\b\w+\b', text.lower())
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Get top 3 most frequent words as topics
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(3)]
