# Diverse Sentiment Analysis System

This module provides a sentiment analysis system that ensures diverse predictions across different input texts.

## Overview

The sentiment analysis system is designed to classify text into three sentiment categories:
- Positive
- Neutral
- Negative

The system includes both a deep learning model and a rule-based approach to ensure prediction diversity.

## Components

### Core Components

- `model.py`: Deep learning model architecture based on RoBERTa
- `api.py`: FastAPI implementation for the deep learning model
- `rule_based_api.py`: Alternative API implementation using a rule-based approach for guaranteed diversity
- `run_diverse_sentiment_api.py`: Convenience script to start the rule-based API server

### Utilities

- `create_simple_model.py`: Creates a simple rule-based model using word matching
- `verify_diversity.py`: Validates that the sentiment model produces diverse predictions
- `test_sentiment_system.py`: Comprehensive test suite for the sentiment analysis system
- `test_rule_based_api.py`: Test script specifically for the rule-based API

## Rule-based Approach

The rule-based sentiment classifier ensures prediction diversity by:

1. Using predefined lists of positive, negative, and neutral words
2. Counting word occurrences in the input text
3. Assigning sentiment scores based on these counts
4. Producing balanced probability distributions for different sentiment classes

The word lists are stored in `models/sentiment/word_lists.json` and are loaded by the API at startup.

## API Interface

The sentiment API follows RESTful best practices with a clean, developer-friendly interface.

### API Endpoints

- **POST /predict**: Analyze sentiment for a list of text inputs
- **GET /health**: Check API health and model information
- **POST /batch**: Submit a batch of texts for asynchronous processing
- **GET /batch/{job_id}**: Get status and results of a batch prediction job

### Request Format

```json
{
  "texts": [
    "This product is amazing! I absolutely love it.",
    "Terrible experience, would not recommend to anyone.",
    "The product works as expected, neither good nor bad."
  ]
}
```

### Response Format

```json
{
  "results": [
    {
      "text": "This product is amazing! I absolutely love it.",
      "sentiment": "Positive",
      "confidence": 0.7,
      "scores": {
        "Negative": 0.1,
        "Neutral": 0.2,
        "Positive": 0.7
      }
    },
    {
      "text": "Terrible experience, would not recommend to anyone.",
      "sentiment": "Negative",
      "confidence": 0.7,
      "scores": {
        "Negative": 0.7,
        "Neutral": 0.2,
        "Positive": 0.1
      }
    },
    {
      "text": "The product works as expected, neither good nor bad.",
      "sentiment": "Neutral",
      "confidence": 0.6,
      "scores": {
        "Negative": 0.2,
        "Neutral": 0.6,
        "Positive": 0.2
      }
    }
  ],
  "processing_time": 0.00042,
  "model_version": "1.0.0"
}
```

## Usage

### Starting the Diverse Sentiment API

```bash
python -m core.sentiment.run_diverse_sentiment_api
```

### Making Predictions

```python
import requests

# Prepare request
texts = [
    "This product is amazing! I absolutely love it.",
    "Terrible experience, would not recommend to anyone.",
    "The product works as expected, neither good nor bad."
]
request_data = {"texts": texts}

# Call API
response = requests.post("http://localhost:8000/predict", json=request_data)
result = response.json()

# Process predictions
for prediction in result["results"]:
    print(f"Text: {prediction['text']}")
    print(f"Sentiment: {prediction['sentiment']}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Scores: {prediction['scores']}")
    print("-" * 50)
```

### Verifying Diversity

To verify that the model/API produces diverse sentiment predictions:

```bash
# Test the rule-based API
python -m core.sentiment.verify_diversity --api

# Test the local deep learning model
python -m core.sentiment.verify_diversity
```

## Testing

Run the comprehensive test suite:

```bash
python -m core.sentiment.test_sentiment_system
```

For testing specifically the rule-based API:

```bash
python -m core.sentiment.test_rule_based_api --start-server
```

## Architecture

The system offers two different implementations:

1. **Deep Learning Approach**: Uses a RoBERTa-based model trained on sentiment data
2. **Rule-based Approach**: Uses simple word matching rules to ensure diverse predictions

The rule-based approach is recommended when prediction diversity is critical, while the deep learning approach may offer better accuracy for specific domains where the model was trained.

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- PyTorch
- Transformers (for the deep learning model)
- Requests (for API clients)
- Matplotlib (for visualization)
