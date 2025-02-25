from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import List
from core.inference.engine import InferenceEngine, InferenceConfig
from core.monitoring import monitor
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the inference engine with configuration
config = InferenceConfig(
    num_threads=4,
    max_batch_size=32,
    min_batch_size=1,
    metrics_window=100,
    cache_size=1000,
    cache_ttl_seconds=3600,
    adaptive_batching=False
)

try:
    engine = InferenceEngine(
        config=config,
        model_path="models/sentiment_model_metal.mlpackage",
        tokenizer_name="distilbert-base-uncased"
    )
    logger.info("Inference engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize inference engine: {e}")
    raise

from core.file_processor import FileProcessor

class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        try:
            response = await call_next(request)
            monitor.log_request_end(
                endpoint=str(request.url.path),
                start_time=start_time,
                status='success'
            )
            return response
        except Exception as e:
            monitor.log_request_end(
                endpoint=str(request.url.path),
                start_time=start_time,
                status='error',
                error=e
            )
            raise

app = FastAPI(title="LLM Inference Engine")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def analyze_text(request: TextRequest):
    logger = logging.getLogger("text_analysis")
    try:
        logger.info("Running text analysis")
        result = await engine.infer(request.text)
        logger.info(f"Analysis complete: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    logger = logging.getLogger("file_analysis")
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Validate file type
        allowed_types = ['.txt', '.json']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_types:
            logger.warning(f"Invalid file type: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Read file content
        content = await file.read()
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="File must be a valid UTF-8 encoded text file"
            )
            
        # Validate content
        if not text.strip():
            logger.warning("Empty file content")
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
            
        logger.info("Running inference on file content")
        result = await engine.infer(text)
        logger.info(f"Inference complete: {result}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze file: {str(e)}"
        )

@app.get("/api/metrics")
async def get_metrics():
    try:
        metrics = {
            "active_requests": engine._current_batch_size,
            "total_requests": engine._processed_requests,
            "average_latency": engine._current_latency,
            "throughput": engine._current_throughput
        }
        return {"metrics": metrics, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
async def get_metrics():
    try:
        metrics = engine.get_performance_metrics()
        # Add monitoring metrics
        monitoring_metrics = {
            "recent_errors": monitor.get_recent_errors(limit=5),
            "error_count": len(monitor.recent_errors),
            "active_requests": monitor.active_requests._value.get(),
            "total_requests": monitor.request_counter._value.get(),
        }
        metrics.update({"monitoring": monitoring_metrics})
        return metrics
    except Exception as e:
        monitor.log_request_end(
            endpoint="/api/metrics",
            start_time=time.time(),
            status="error",
            error=e
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/errors")
async def get_errors():
    try:
        return {
            "errors": monitor.get_recent_errors(limit=50)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export-errors")
async def export_errors():
    try:
        report_path = monitor.export_error_report()
        return {"report_path": report_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    start_time = time.time()
    try:
        saved_files = []
        for file in files:
            file_path = UPLOADS_DIR / file.filename
            with file_path.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_files.append(file.filename)
        response = {"status": "success", "processed_files": saved_files}
        
        monitor.log_request_end(
            endpoint="/api/upload",
            start_time=start_time,
            status="success"
        )
        return response
    except Exception as e:
        error_msg = str(e)
        monitor.log_request_end(
            endpoint="/api/upload",
            start_time=start_time,
            status="error",
            error=e
        )
        raise HTTPException(status_code=500, detail=error_msg)

        
        return {"uploaded_files": saved_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

if __name__ == "__main__":
    config = uvicorn.Config(
        "app:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        timeout_keep_alive=300,  # Increase keep-alive timeout
        limit_concurrency=100,   # Limit concurrent connections
        backlog=128,            # Connection queue size
        timeout_notify=30,      # Timeout for notifying workers
        workers=4               # Number of worker processes
    )
    server = uvicorn.Server(config)
    server.run()
