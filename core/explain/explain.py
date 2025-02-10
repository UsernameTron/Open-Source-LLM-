"""Explanation generation module for the inference engine."""
from typing import Dict, Any

def generate_explanation(text: str, sentiment_info: dict) -> str:
    """Generate a simple explanation for the sentiment analysis result.
    
    Args:
        text: Input text that was analyzed
        sentiment_info: Dictionary containing sentiment analysis results
        
    Returns:
        String containing explanation
    """
    sentiment = sentiment_info.get('prediction', 'Unknown')
    confidence = sentiment_info.get('confidence', 0.0)
    
    return f"The text was classified as {sentiment} with {confidence:.1%} confidence based on sentiment analysis."
