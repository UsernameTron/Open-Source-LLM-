#!/usr/bin/env python
"""
rule_based_api.py

A FastAPI implementation for sentiment analysis using a rule-based approach
to ensure diverse predictions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import time
import logging
import os
import json
from pathlib import Path
import uuid
import asyncio

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define sentiment labels
SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis with rule-based approach",
    version="1.0.0"
)

# FastAPI Pydantic models for request/response
class PredictRequest(BaseModel):
    texts: List[str]

class Prediction(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: Dict[str, float]

class PredictResponse(BaseModel):
    results: List[Prediction]
    processing_time: float
    model_version: str

class BatchSubmitResponse(BaseModel):
    job_id: str
    status: str

class BatchStatusResponse(BaseModel):
    job_id: str
    status: str
    total: int
    completed: int
    results: Optional[List[Prediction]] = None
    model_version: Optional[str] = "1.0.0"

# Global variables
word_lists = {}
batch_jobs = {}

def load_word_lists():
    """Load sentiment word lists."""
    global word_lists
    
    try:
        # Load word lists from JSON file
        word_lists_path = Path("./models/sentiment/word_lists.json")
        if word_lists_path.exists():
            with open(word_lists_path, "r") as f:
                word_lists = json.load(f)
            logger.info(f"Loaded word lists with {len(word_lists['positive_words'])} positive, "
                      f"{len(word_lists['negative_words'])} negative, and "
                      f"{len(word_lists['neutral_words'])} neutral words")
        else:
            # Default word lists if file doesn't exist
            word_lists = {
                "positive_words": [
                    "good", "great", "excellent", "amazing", "love", "wonderful", 
                    "fantastic", "awesome", "best", "perfect", "happy", "pleased"
                ],
                "negative_words": [
                    "bad", "worst", "terrible", "awful", "poor", "hate", "disappointing",
                    "horrible", "useless", "waste", "failed", "annoying"
                ],
                "neutral_words": [
                    "ok", "okay", "fine", "average", "expected", "adequate", "reasonable",
                    "standard", "normal", "usual", "basic", "acceptable"
                ]
            }
            logger.warning("Word lists file not found, using default word lists")
    except Exception as e:
        logger.error(f"Error loading word lists: {e}")
        # Fallback to empty lists
        word_lists = {
            "positive_words": [],
            "negative_words": [],
            "neutral_words": []
        }
    
    return word_lists

def predict_sentiment(text):
    """
    Predict sentiment using rule-based approach.
    
    Args:
        text (str): Input text to classify
        
    Returns:
        Tuple of (sentiment_label, confidence, scores)
    """
    global word_lists
    
    try:
        # Convert to lowercase for matching
        text = text.lower()
        
        # Count word occurrences
        pos_count = sum(1 for word in word_lists["positive_words"] if word in text)
        neg_count = sum(1 for word in word_lists["negative_words"] if word in text)
        neu_count = sum(1 for word in word_lists["neutral_words"] if word in text)
        
        # If no sentiment words found, default to neutral
        if pos_count == 0 and neg_count == 0 and neu_count == 0:
            scores = [0.2, 0.6, 0.2]  # [negative, neutral, positive]
            predicted_class = 1  # Neutral
            
        # If multiple types of sentiment words, use the highest count
        elif pos_count > neg_count and pos_count > neu_count:
            scores = [0.1, 0.2, 0.7]  # [negative, neutral, positive]
            predicted_class = 2  # Positive
        elif neg_count > pos_count and neg_count > neu_count:
            scores = [0.7, 0.2, 0.1]  # [negative, neutral, positive]
            predicted_class = 0  # Negative
        elif neu_count > pos_count and neu_count > neg_count:
            scores = [0.2, 0.6, 0.2]  # [negative, neutral, positive]
            predicted_class = 1  # Neutral
        
        # If tie between positive and negative, check intensity
        elif pos_count == neg_count:
            # Check for intensifiers
            if any(word in text for word in ["very", "really", "absolutely", "extremely"]):
                if "not" in text or "don't" in text or "doesn't" in text:
                    scores = [0.7, 0.2, 0.1]  # [negative, neutral, positive]
                    predicted_class = 0  # Negative with intensifier
                else:
                    scores = [0.1, 0.2, 0.7]  # [negative, neutral, positive]
                    predicted_class = 2  # Positive with intensifier
            else:
                scores = [0.3, 0.4, 0.3]  # [negative, neutral, positive]
                predicted_class = 1  # Mixed sentiment, slightly neutral
        
        # Fallback to neutral
        else:
            scores = [0.2, 0.6, 0.2]  # [negative, neutral, positive]
            predicted_class = 1  # Neutral
        
        # Get sentiment label and confidence
        sentiment_label = SENTIMENT_LABELS[predicted_class]
        confidence = scores[predicted_class]
        
        # Log prediction details
        logger.info(f"Predicted sentiment: {sentiment_label} with confidence: {confidence:.4f}")
        
        # Convert scores to dictionary
        scores_dict = {
            "Negative": scores[0],
            "Neutral": scores[1],
            "Positive": scores[2]
        }
        
        return sentiment_label, confidence, scores_dict
            
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        # Return neutral in case of error
        return "Neutral", 0.6, {"Negative": 0.2, "Neutral": 0.6, "Positive": 0.2}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict sentiment for a list of texts.
    """
    # Start timer
    start_time = time.time()
    
    predictions = []
    
    try:
        for text in request.texts:
            # Get sentiment prediction
            sentiment, confidence, scores = predict_sentiment(text)
            
            # Create prediction object
            prediction = Prediction(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                scores=scores
            )
            
            predictions.append(prediction)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = PredictResponse(
            results=predictions,
            processing_time=processing_time,
            model_version="1.0.0"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=BatchSubmitResponse)
async def batch_predict(request: PredictRequest, background_tasks: BackgroundTasks) -> BatchSubmitResponse:
    """
    Submit a batch of texts for sentiment analysis.
    
    Returns a job ID that can be used to check the status and retrieve results.
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    batch_jobs[job_id] = {
        "status": "processing",
        "total": len(request.texts),
        "completed": 0,
        "results": [],
        "model_version": "1.0.0"
    }
    
    # Process batch in background
    background_tasks.add_task(process_batch, job_id, request.texts)
    
    # Return job ID
    return BatchSubmitResponse(
        job_id=job_id,
        status="processing"
    )

@app.get("/batch/{job_id}", response_model=BatchStatusResponse)
async def batch_status(job_id: str) -> BatchStatusResponse:
    """
    Get the status and results of a batch prediction job.
    """
    # Check if job exists
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get job status
    job = batch_jobs[job_id]
    
    # Create response
    response = BatchStatusResponse(
        job_id=job_id,
        status=job["status"],
        total=job["total"],
        completed=job["completed"],
        model_version=job.get("model_version", "1.0.0")
    )
    
    # Include results if job is complete
    if job["status"] == "complete":
        response.results = job["results"]
    
    return response

async def process_batch(job_id: str, texts: List[str]):
    """
    Process a batch of texts in the background.
    """
    try:
        for i, text in enumerate(texts):
            # Get sentiment prediction
            sentiment, confidence, scores = predict_sentiment(text)
            
            # Create prediction object
            prediction = Prediction(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                scores=scores
            )
            
            # Update job status
            batch_jobs[job_id]["results"].append(prediction)
            batch_jobs[job_id]["completed"] = i + 1
            
            # Simulate processing time for demonstration
            await asyncio.sleep(0.1)
        
        # Mark job as complete
        batch_jobs[job_id]["status"] = "complete"
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        batch_jobs[job_id]["status"] = "error"
        batch_jobs[job_id]["error"] = str(e)

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns status of the API and model information.
    """
    # Get model information
    model_info = {
        "type": "rule_based",
        "word_lists": {
            "positive_count": len(word_lists.get("positive_words", [])),
            "negative_count": len(word_lists.get("negative_words", [])),
            "neutral_count": len(word_lists.get("neutral_words", []))
        },
        "version": "1.0.0"
    }
    
    # Try to load config file
    config_path = Path("./models/sentiment/config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            model_info["config"] = config
        except:
            # Config loading failed but API is still working
            model_info["config"] = "Failed to load config"
    
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "model": model_info
    }

# On startup, load word lists
@app.on_event("startup")
async def startup_event():
    """Load resources when API starts."""
    # Load word lists
    load_word_lists()
    logger.info("API started successfully")

# For local testing
if __name__ == "__main__":
    import uvicorn
    
    # Load word lists immediately for testing
    load_word_lists()
    
    # Run API
    uvicorn.run(app, host="0.0.0.0", port=8000)
