# Sentiment API Launch Instructions

This document provides instructions for running the improved Sentiment Analysis API with the new RESTful interface.

## Quick Start

The simplest way to launch the API is by using the desktop launcher:

1. Double-click the `run_sentiment_api.command` icon on your Desktop.
2. The API server will start and display information about the available endpoints.
3. By default, the API will run on `http://localhost:8000`.

## Manual Launch

To manually start the API server:

```bash
# Navigate to the project directory
cd /Users/cpconnor/CascadeProjects/llm-engine

# Start the server
python -m core.sentiment.run_diverse_sentiment_api
```

## Verifying the API

To verify that the API is working correctly:

```bash
# Test basic functionality
python -m core.sentiment.verify_diversity --api

# Run comprehensive tests
python -m core.sentiment.test_rule_based_api

# Verify API format compliance
python -m core.sentiment.verify_api_format
```

## Using the Sample Clients

The project includes sample clients to demonstrate API usage:

```bash
# Simple prediction client
python -m core.sentiment.sample_client

# Batch processing client
python -m core.sentiment.sample_batch_client
```

## API Endpoints

The API provides the following endpoints:

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model": {
    "type": "rule_based",
    "version": "1.0.0",
    ...
  }
}
```

### Sentiment Prediction

```
POST /predict
```

Request:
```json
{
  "texts": [
    "This product is amazing!",
    "Terrible experience, would not recommend."
  ]
}
```

Response:
```json
{
  "results": [
    {
      "text": "This product is amazing!",
      "sentiment": "Positive",
      "confidence": 0.7,
      "scores": {
        "Negative": 0.1,
        "Neutral": 0.2,
        "Positive": 0.7
      }
    },
    {
      "text": "Terrible experience, would not recommend.",
      "sentiment": "Negative",
      "confidence": 0.7,
      "scores": {
        "Negative": 0.7,
        "Neutral": 0.2,
        "Positive": 0.1
      }
    }
  ],
  "processing_time": 0.00042,
  "model_version": "1.0.0"
}
```

### Batch Processing

```
POST /batch
```

Request:
```json
{
  "texts": [
    "This product is amazing!",
    "Terrible experience, would not recommend."
  ]
}
```

Response:
```json
{
  "job_id": "265e096a-f4d4-4f34-987a-7ad4d3fd1bc5",
  "status": "pending",
  "model_version": "1.0.0"
}
```

### Batch Status

```
GET /batch/{job_id}
```

Response:
```json
{
  "job_id": "265e096a-f4d4-4f34-987a-7ad4d3fd1bc5",
  "status": "complete",
  "model_version": "1.0.0",
  "results": [
    {
      "text": "This product is amazing!",
      "sentiment": "Positive",
      "confidence": 0.7,
      "scores": {
        "Negative": 0.1,
        "Neutral": 0.2,
        "Positive": 0.7
      }
    },
    {
      "text": "Terrible experience, would not recommend.",
      "sentiment": "Negative",
      "confidence": 0.7,
      "scores": {
        "Negative": 0.7,
        "Neutral": 0.2,
        "Positive": 0.1
      }
    }
  ]
}
```

## Troubleshooting

If you encounter issues:

1. **Port Conflict**: If port 8000 is already in use:
   ```bash
   # Check if port is in use
   lsof -i :8000
   
   # Kill the process using the port
   kill -9 <PID>
   ```

2. **Server Not Starting**: Check for error messages in the console output.

3. **API Request Failures**: Verify that the server is running using the health check endpoint:
   ```bash
   curl http://localhost:8000/health
   ```
