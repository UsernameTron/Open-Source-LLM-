#!/usr/bin/env python
"""
verify_diversity.py

A simple script to verify that the sentiment model is producing diverse predictions
"""

import argparse
import torch
import logging
import json
from pathlib import Path
from collections import Counter
import requests  # Added for API testing
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# List of test sentences that should cover different sentiments
TEST_SENTENCES = [
    "This product exceeded all my expectations! Absolutely fantastic.",
    "I love how well this works, best purchase I've made all year.",
    "Terrible experience, would not recommend to anyone.",
    "This is the worst product I've ever used, complete waste of money.",
    "The product works as expected, neither good nor bad.",
    "It arrived on time and functions adequately.",
    "While the design is nice, the functionality is lacking.",
    "Great features but the customer service was disappointing."
]

def test_model_diversity():
    """Test the local model diversity."""
    try:
        from transformers import RobertaTokenizer
        from core.sentiment.model import SentimentClassifier
        
        logger.info("=== Testing Sentiment Model Diversity ===")
        
        # Load model
        model_dir = Path("./models/sentiment")
        logger.info(f"Model directory: {model_dir}")
        
        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device for Apple M4 Pro")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        # Load config
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)
            logger.info(f"Loaded config: {config}")
        
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )
        logger.info(f"Loaded tokenizer from {model_dir}")
        
        # Initialize model
        model = SentimentClassifier(
            num_classes=config["architecture"]["classifier_output_dim"]
        )
        
        # Load model weights
        model.load_state_dict(torch.load(
            model_dir / "model.pth",
            map_location=device
        ))
        logger.info(f"Loaded model weights from {model_dir / 'model.pth'}")
        
        # Move model to device
        model.to(device)
        model.eval()
        
        # Set prediction classes
        sentiment_labels = ["Negative", "Neutral", "Positive"]
        
        # Test on different sentences
        logger.info(f"Testing sentiment model diversity on {len(TEST_SENTENCES)} test cases")
        
        predictions = []
        all_probs = []
        
        for text in TEST_SENTENCES:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
            
            # Get sentiment class and confidence
            probs = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()
            
            # Get readable sentiment label
            sentiment = sentiment_labels[predicted_class]
            predictions.append(sentiment)
            
            # Convert to probabilities dict for logging
            probs_dict = {
                sentiment_labels[i]: probs[i].item() 
                for i in range(len(sentiment_labels))
            }
            all_probs.append(probs_dict)
            
            # Log prediction
            logger.info(f"Text: {text}")
            logger.info(f"Prediction: {sentiment} (Confidence: {confidence:.4f})")
            logger.info(f"All probabilities: {probs_dict}")
            logger.info("-" * 80)
        
        # Analyze diversity
        unique_predictions = len(set(predictions))
        logger.info(f"\nUnique sentiment predictions: {unique_predictions} of {len(sentiment_labels)}")
        
        # Count occurrences of each sentiment
        sentiment_counts = Counter(predictions)
        logger.info(f"Sentiment distribution: {list(sentiment_counts.items())}")
        
        # Calculate average probabilities
        avg_probs = {}
        for label in sentiment_labels:
            avg_probs[label] = sum(p[label] for p in all_probs) / len(all_probs)
        logger.info(f"Average probabilities: {avg_probs}")
        
        # Check if model is producing diverse predictions
        if unique_predictions >= 3:
            logger.info("✅ Model is producing diverse predictions!")
            return True
        else:
            logger.warning("⚠️ WARNING: Model is not producing diverse predictions!")
            logger.error("\n❌ Failed! The model is not producing diverse predictions.")
            return False
            
    except Exception as e:
        logger.error(f"Error testing model diversity: {e}")
        return False

def test_api_diversity(api_url):
    """Test the API diversity."""
    try:
        logger.info("=== Testing Sentiment API Diversity ===")
        logger.info(f"API URL: {api_url}")
        
        # Call API health check
        try:
            response = requests.get(f"{api_url}/health")
            response.raise_for_status()
            logger.info(f"API health check: {response.json()}")
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
        
        # Prepare request
        request_data = {"texts": TEST_SENTENCES}
        
        # Call API
        response = requests.post(
            f"{api_url}/predict",
            json=request_data
        )
        response.raise_for_status()
        
        # Get predictions
        result = response.json()
        predictions = result["results"]
        
        # Log each prediction
        sentiments = []
        for i, pred in enumerate(predictions):
            sentiments.append(pred["sentiment"])
            logger.info(f"Text: {pred['text']}")
            logger.info(f"Prediction: {pred['sentiment']} (Confidence: {pred['confidence']:.4f})")
            logger.info(f"All probabilities: {pred['scores']}")
            logger.info("-" * 80)
        
        # Analyze diversity
        unique_predictions = len(set(sentiments))
        logger.info(f"\nUnique sentiment predictions: {unique_predictions} of 3")
        
        # Count occurrences of each sentiment
        sentiment_counts = Counter(sentiments)
        logger.info(f"Sentiment distribution: {list(sentiment_counts.items())}")
        
        # Check if API is producing diverse predictions
        if unique_predictions >= 3:
            logger.info("✅ API is producing diverse predictions!")
            return True
        else:
            logger.warning("⚠️ WARNING: API is not producing diverse predictions!")
            logger.error("\n❌ Failed! The API is not producing diverse predictions.")
            return False
            
    except Exception as e:
        logger.error(f"Error testing API diversity: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify diversity of sentiment predictions")
    parser.add_argument("--api", action="store_true", help="Test API instead of local model")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of sentiment API")
    args = parser.parse_args()
    
    if args.api:
        # Test API diversity
        success = test_api_diversity(args.api_url)
    else:
        # Test local model diversity
        success = test_model_diversity()
    
    # Exit with appropriate code
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
