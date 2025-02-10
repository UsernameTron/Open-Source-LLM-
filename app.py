from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
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
from unittest.mock import Mock
from core.inference.engine import InferenceEngine, InferenceConfig

# Initialize the inference engine with configuration
config = InferenceConfig(
    min_batch_size=1,
    max_batch_size=16,
    target_latency_ms=100.0,
    max_queue_size=100,
    cache_size=1000,
    num_threads=2
)

# Initialize engine with mock model for testing
engine = InferenceEngine(
    model_path=Mock(),  # Mock model for testing
    tokenizer_name="bert-base-uncased",
    config=config
)
from core.file_processor import FileProcessor
from core.monitoring import monitor

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

# Initialize inference engine
engine = InferenceEngine(
    model_path="path/to/your/model",  # You'll need to update this
    tokenizer_name="bert-base-uncased"  # Update based on your model
)

class TextRequest(BaseModel):
    text: str
    priority: int = 0

@app.post("/api/infer")
async def infer(request: TextRequest):
    start_time = time.time()
    try:
        if isinstance(engine.model, Mock):
            # For mock model, return a simple response
            response = {
                "result": {
                    "text": request.text,
                    "prediction": "This is a mock prediction for: " + request.text,
                    "confidence": 0.95
                },
                "status": "success"
            }
        else:
            # For real model, use the inference engine
            result = await engine.infer_async(request.text, request.priority)
            response = {"result": result, "status": "success"}
        
        monitor.log_request_end(
            endpoint="/api/infer",
            start_time=start_time,
            status="success"
        )
        return response
    except Exception as e:
        error_msg = str(e)
        monitor.log_request_end(
            endpoint="/api/infer",
            start_time=start_time,
            status="error",
            error=e,
            input_data={"text": request.text, "priority": request.priority}
        )
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/explain")
async def explain(request: TextRequest):
    start_time = time.time()
    try:
        result = await engine.explain(request.text)
        monitor.log_request_end(
            endpoint="/api/explain",
            start_time=start_time,
            status="success"
        )
        return {"explanation": result, "status": "success"}
    except Exception as e:
        monitor.log_request_end(
            endpoint="/api/explain",
            start_time=start_time,
            status="error",
            error=e,
            input_data={"text": request.text}
        )
        raise HTTPException(status_code=500, detail=str(e))

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
        if isinstance(engine.model, Mock):
            processed_files = [file.filename for file in files]
            response = {"status": "success", "processed_files": processed_files}
        else:
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

@app.get("/")
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Inference Engine</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <link rel="stylesheet" href="/static/style.css">
        <script src="/static/main.js" defer></script>
    </head>
    <body class="bg-gray-100">
        <div id="app" class="container mx-auto px-4 py-8">
            <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
                <h1 class="text-3xl font-bold mb-4">LLM Inference Engine</h1>
                
                <!-- Input Section -->
                <div class="mb-8">
                    <div class="mb-4">
                        <h3 class="text-lg font-semibold mb-2">Text Input</h3>
                        <textarea 
                            v-model="inputText" 
                            class="w-full p-2 border rounded-lg mb-4"
                            rows="4"
                            placeholder="Enter text for inference..."></textarea>
                        <div class="flex gap-4">
                            <button 
                                @click="runInference"
                                class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                                :disabled="loading || !inputText">
                                {{ loading ? 'Processing...' : 'Run Inference' }}
                            </button>
                            <button 
                                @click="getExplanation"
                                class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
                                :disabled="loading || !inputText">
                                {{ loading ? 'Processing...' : 'Get Explanation' }}
                            </button>
                        </div>
                    </div>
                    
                    <div class="mb-4">
                        <h3 class="text-lg font-semibold mb-2">File Upload</h3>
                        <div class="flex items-center gap-4">
                            <input 
                                type="file" 
                                ref="fileInput"
                                @change="uploadFiles"
                                multiple
                                class="hidden"
                                accept=".txt,.pdf,.doc,.docx"
                            >
                            <button 
                                @click="$refs.fileInput.click()"
                                class="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600">
                                Choose Files
                            </button>
                            <button 
                                v-if="selectedFiles.length"
                                @click="uploadFiles"
                                class="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600">
                                Upload {{ selectedFiles.length }} File(s)
                            </button>
                        </div>
                        <div v-if="selectedFiles.length" class="mt-2">
                            <div v-for="file in selectedFiles" :key="file.name" class="text-sm text-gray-600">
                                {{ file.name }} ({{ (file.size / 1024).toFixed(1) }} KB)
                            </div>
                        </div>
                        <div v-if="uploadedFiles.length" class="mt-4">
                            <h4 class="font-medium mb-2">Uploaded Files:</h4>
                            <div v-for="file in uploadedFiles" :key="file.filename" class="text-sm text-gray-600">
                                {{ file.filename }}
                            </div>
                        </div>
                    </div>
                    
                    <div class="flex gap-4">
                        <button 
                            @click="runInference" 
                            class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                            Run Inference
                        </button>
                        <button 
                            @click="getExplanation" 
                            class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
                            Get Explanation
                        </button>
                    </div>
                </div>

                <!-- Results Section -->
                <div v-if="result" class="mb-8">
                    <h2 class="text-xl font-bold mb-2">Results</h2>
                    <div class="bg-gray-50 p-4 rounded-lg">
                        <pre class="whitespace-pre-wrap">{{ JSON.stringify(result, null, 2) }}</pre>
                    </div>
                </div>

                <!-- Metrics Section -->
                <div class="mb-8">
                    <h2 class="text-xl font-bold mb-2">Performance Metrics</h2>
                    <div id="metricsChart" class="w-full h-64"></div>
                </div>

                <!-- Error Monitoring Section -->
                <div class="mb-8">
                    <h2 class="text-xl font-bold mb-2">Error Monitoring</h2>
                    <div class="bg-white rounded-lg shadow p-4">
                        <div class="flex justify-between items-center mb-4">
                            <div>
                                <span class="text-sm font-medium text-gray-500">Active Requests:</span>
                                <span class="ml-2 text-lg font-bold" v-text="activeRequests"></span>
                            </div>
                            <div>
                                <span class="text-sm font-medium text-gray-500">Total Errors:</span>
                                <span class="ml-2 text-lg font-bold text-red-500" v-text="errorCount"></span>
                            </div>
                            <button 
                                @click="exportErrorReport"
                                class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                                Export Error Report
                            </button>
                        </div>

                        <!-- Recent Errors Table -->
                        <div class="overflow-x-auto">
                            <table class="min-w-full divide-y divide-gray-200">
                                <thead class="bg-gray-50">
                                    <tr>
                                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Message</th>
                                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Endpoint</th>
                                    </tr>
                                </thead>
                                <tbody class="bg-white divide-y divide-gray-200">
                                    <tr v-for="error in recentErrors" :key="error.timestamp" class="hover:bg-gray-50">
                                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {{ new Date(error.timestamp).toLocaleString() }}
                                        </td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-red-500">
                                            {{ error.error_type }}
                                        </td>
                                        <td class="px-6 py-4 text-sm text-gray-900">
                                            {{ error.error_message }}
                                        </td>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {{ error.endpoint }}
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const { createApp } = Vue

            createApp({
                data() {
                    return {
                        inputText: '',
                        result: null,
                        metricsInterval: null,
                        selectedFiles: [],
                        uploadedFiles: [],
                        activeRequests: 0,
                        errorCount: 0,
                        recentErrors: [],
                        errorMonitoringInterval: null
                    }
                },
                methods: {
                    handleFileSelect(event) {
                        this.selectedFiles = Array.from(event.target.files)
                    },
                    async uploadFiles() {
                        try {
                            const formData = new FormData()
                            this.selectedFiles.forEach(file => {
                                formData.append('files', file)
                            })
                            
                            const response = await fetch('/api/upload', {
                                method: 'POST',
                                body: formData
                            })
                            
                            const result = await response.json()
                            this.uploadedFiles = result.uploaded_files
                            this.selectedFiles = []
                            this.$refs.fileInput.value = ''
                            
                            // Extract text from uploaded files and add to input
                            if (result.uploaded_files && result.uploaded_files.length > 0) {
                                // You can add file processing logic here if needed
                                console.log('Files uploaded successfully:', result.uploaded_files)
                            }
                        } catch (error) {
                            console.error('Error uploading files:', error)
                            alert('Error uploading files: ' + error.message)
                        }
                    },
                    async runInference() {
                        try {
                            const response = await fetch('/api/infer', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({
                                    text: this.inputText
                                })
                            })
                            this.result = await response.json()
                        } catch (error) {
                            console.error('Error:', error)
                            alert('Error running inference: ' + error.message)
                        }
                    },
                    async getExplanation() {
                        try {
                            const response = await fetch('/api/explain', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({
                                    text: this.inputText
                                })
                            })
                            this.result = await response.json()
                        } catch (error) {
                            console.error('Error:', error)
                            alert('Error getting explanation: ' + error.message)
                        }
                    },
                    async updateMetrics() {
                        try {
                            const response = await fetch('/api/metrics')
                            const metrics = await response.json()
                            
                            this.activeRequests = metrics.monitoring.active_requests
                            this.errorCount = metrics.monitoring.error_count
                            
                            const data = [{
                                x: [new Date()],
                                y: [metrics.throughput],
                                name: 'Throughput',
                                type: 'scatter'
                            }, {
                                x: [new Date()],
                                y: [metrics.latency],
                                name: 'Latency',
                                type: 'scatter'
                            }]
                            
                            Plotly.newPlot('metricsChart', data)
                        } catch (error) {
                            console.error('Error updating metrics:', error)
                        }
                    },
                    
                    async updateErrors() {
                        try {
                            const response = await fetch('/api/errors')
                            const data = await response.json()
                            this.recentErrors = data.errors
                        } catch (error) {
                            console.error('Error updating errors:', error)
                        }
                    },
                    
                    async exportErrorReport() {
                        try {
                            const response = await fetch('/api/export-errors', {
                                method: 'POST'
                            })
                            const data = await response.json()
                            alert(`Error report exported to: ${data.report_path}`)
                        } catch (error) {
                            console.error('Error exporting report:', error)
                            alert('Error exporting report: ' + error.message)
                        }
                    },
                    
                    async updateMetricsOld() {
                        try {
                            const response = await fetch('/api/metrics')
                            const metrics = await response.json()
                            
                            const data = [{
                                x: [new Date()],
                                y: [metrics.throughput],
                                name: 'Throughput',
                                type: 'scatter'
                            }, {
                                x: [new Date()],
                                y: [metrics.latency],
                                name: 'Latency',
                                type: 'scatter'
                            }]
                            
                            Plotly.newPlot('metricsChart', data)
                        } catch (error) {
                            console.error('Error updating metrics:', error)
                        }
                    }
                },
                mounted() {
                    this.updateMetrics()
                    this.metricsInterval = setInterval(this.updateMetrics, 5000)
                    this.updateErrors()
                    this.errorMonitoringInterval = setInterval(this.updateErrors, 2000)
                },
                beforeUnmount() {
                    if (this.metricsInterval) {
                        clearInterval(this.metricsInterval)
                    }
                    if (this.errorMonitoringInterval) {
                        clearInterval(this.errorMonitoringInterval)
                    }
                }
            }).mount('#app')
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
