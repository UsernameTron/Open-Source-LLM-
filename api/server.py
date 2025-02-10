from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
import asyncio
from typing import List, Dict, Any

# Add parent directory to path to import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.inference.engine import InferenceEngine

app = FastAPI(title="Sentiment Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine
inference_engine = None

class SentimentRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    global inference_engine
    inference_engine = InferenceEngine(
        model_path="mock_model",  # Using mock model for testing
        tokenizer_name=None  # No tokenizer needed for mock
    )

@app.on_event("shutdown")
async def shutdown_event():
    global inference_engine
    if inference_engine:
        await inference_engine.cleanup()

@app.post("/analyze")
async def analyze_text(request: SentimentRequest) -> Dict[str, Any]:
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = await inference_engine.analyze_text(request.text)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to get analysis results")
        
        # Ensure we're returning a proper JSON response
        response = {
            'prediction': result.get('prediction', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
            'explanation': result.get('explanation', ''),
            'text': request.text
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=True)
