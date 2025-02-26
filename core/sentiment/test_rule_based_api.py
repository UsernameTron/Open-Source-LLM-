#!/usr/bin/env python
"""
test_rule_based_api.py

Script to verify that the rule-based sentiment API produces diverse predictions.
"""

import argparse
import requests
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import subprocess
import sys
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test sentences with expected diverse sentiments
TEST_SENTENCES = [
    # Positive examples
    "This product is amazing! I absolutely love it.",
    "The service was excellent, I'm very happy with my experience.",
    "I'm impressed by the quality and speed of delivery.",
    "This works perfectly and exceeds my expectations.",
    "The performance is outstanding, best purchase I've made.",
    
    # Negative examples
    "This product is terrible, worst purchase I've ever made.",
    "The customer service was awful and completely unhelpful.",
    "I hate this product, it completely failed to work.",
    "Very disappointed with the quality, it broke after one use.",
    "The experience was frustrating and a complete waste of money.",
    
    # Neutral examples
    "The product works as expected, nothing special.",
    "It's an average product that does what it should.",
    "The service was adequate, neither great nor terrible.",
    "This meets the basic requirements but doesn't stand out.",
    "It functions adequately for the price point."
]

def start_api_server():
    """Start the API server as a subprocess."""
    logger.info("Starting API server...")
    process = subprocess.Popen(
        [sys.executable, "-m", "core.sentiment.rule_based_api"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(2)
    
    # Check if process is running
    if process.poll() is not None:
        _, stderr = process.communicate()
        logger.error(f"Failed to start API server: {stderr.decode()}")
        raise RuntimeError("API server failed to start")
    
    logger.info("API server started successfully")
    return process

def stop_api_server(process):
    """Stop the API server subprocess."""
    logger.info("Stopping API server...")
    process.terminate()
    process.wait()
    logger.info("API server stopped")

def test_api_health(api_url):
    """Test API health endpoint."""
    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()
        logger.info(f"API health check successful: {response.json()}")
        return response.json()
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        raise

def test_api_predictions(api_url):
    """Test API predictions and analyze diversity."""
    try:
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
        
        # Analyze results
        sentiments = [p["sentiment"] for p in predictions]
        sentiment_counts = Counter(sentiments)
        
        # Log results
        logger.info(f"API returned {len(predictions)} predictions")
        logger.info(f"Model version: {result.get('model_version', 'unknown')}")
        logger.info(f"Processing time: {result.get('processing_time', 0):.6f} seconds")
        logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
        logger.info(f"Unique sentiment predictions: {len(sentiment_counts)} of 3")
        
        # Check diversity
        if len(sentiment_counts) >= 3:
            logger.info("✅ API is producing diverse sentiment predictions")
        else:
            logger.warning("⚠️ API is not producing all 3 sentiment classes")
        
        return predictions
    except Exception as e:
        logger.error(f"API prediction test failed: {str(e)}")
        raise

def visualize_predictions(predictions):
    """Visualize prediction distributions."""
    # Extract probabilities for each sentiment class
    neg_scores = [p["scores"]["Negative"] for p in predictions]
    neu_scores = [p["scores"]["Neutral"] for p in predictions]
    pos_scores = [p["scores"]["Positive"] for p in predictions]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot probability distributions
    plt.subplot(2, 1, 1)
    plt.bar(range(len(predictions)), neg_scores, label="Negative", alpha=0.7, color="red")
    plt.bar(range(len(predictions)), neu_scores, bottom=neg_scores, label="Neutral", alpha=0.7, color="gray")
    plt.bar(
        range(len(predictions)), 
        pos_scores, 
        bottom=np.array(neg_scores) + np.array(neu_scores), 
        label="Positive", 
        alpha=0.7, 
        color="green"
    )
    plt.xlabel("Test Sentences")
    plt.ylabel("Probability")
    plt.title("Sentiment Probabilities for Each Test Sentence")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Plot sentiment distribution
    plt.subplot(2, 1, 2)
    sentiments = [p["sentiment"] for p in predictions]
    sentiment_counts = Counter(sentiments)
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=["red", "gray", "green"])
    plt.xlabel("Sentiment Class")
    plt.ylabel("Count")
    plt.title("Distribution of Sentiment Predictions")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plots_dir = Path("./plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "sentiment_distribution.png")
    logger.info(f"Saved visualization to {plots_dir / 'sentiment_distribution.png'}")
    
    # Show plot if running in interactive mode
    if os.environ.get("DISPLAY"):
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test rule-based sentiment API for diversity")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of sentiment API")
    parser.add_argument("--start-server", action="store_true", help="Start API server locally")
    args = parser.parse_args()
    
    # Start API server if requested
    process = None
    if args.start_server:
        process = start_api_server()
    
    try:
        # Run tests
        logger.info(f"Testing API at {args.api_url}")
        test_api_health(args.api_url)
        predictions = test_api_predictions(args.api_url)
        visualize_predictions(predictions)
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        sys.exit(1)
    finally:
        # Stop API server if we started it
        if process:
            stop_api_server(process)

if __name__ == "__main__":
    main()
