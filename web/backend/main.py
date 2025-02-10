from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
from core.inference.engine import InferenceEngine, InferenceConfig

app = FastAPI(title="LLM Engine API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine
config = InferenceConfig()
engine = InferenceEngine(model_path="path/to/model", tokenizer_name="model_name", config=config)

class InferenceRequest(BaseModel):
    text: str
    config: Dict[str, Any] = None

class MetricsResponse(BaseModel):
    latency: float
    throughput: float
    accuracy: float
    confidence: float

@app.post("/api/infer")
async def infer(request: InferenceRequest):
    try:
        result = await engine.infer_async(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics() -> MetricsResponse:
    try:
        metrics = engine._collect_resource_metrics()
        return MetricsResponse(
            latency=metrics.batch_latency,
            throughput=metrics.throughput,
            accuracy=0.95,  # Replace with actual accuracy metrics
            confidence=0.88  # Replace with actual confidence metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    return {
        "batch_size": config.max_batch_size,
        "confidence_threshold": config.confidence_threshold,
        "temperature": config.temperature,
        "ensemble_size": config.ensemble_size
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
