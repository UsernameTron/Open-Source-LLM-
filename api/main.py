import os
import tempfile
import time
import asyncio
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple, Union
import logging
from core.inference.engine import InferenceEngine, InferenceConfig
import duckdb
from pathlib import Path
import aiofiles
import json
import io
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from PyPDF2 import PdfReader
from core.metrics import metrics_tracker  # Import metrics_tracker for consistent metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Engine API",
    description="Secure LLM inference engine with advanced monitoring",
    version="1.0.0",
    docs_url=None if os.getenv("ENVIRONMENT") == "production" else "/docs",
    redoc_url=None if os.getenv("ENVIRONMENT") == "production" else "/redoc"
)

# Security middleware
from core.security import SecurityMiddleware, verify_api_key
app.add_middleware(SecurityMiddleware)

# CORS configuration with secure defaults
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in 
                  os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Error-Code"],
    max_age=3600
)

# Mount static files with security headers
app.mount("/static", 
          StaticFiles(directory="api/static"), 
          name="static")

# Initialize templates
templates = Jinja2Templates(directory="api/templates")

# Global security dependency
def get_auth_dependency():
    if os.getenv("ENVIRONMENT") == "production":
        return [Depends(verify_api_key)]
    return []

# Add CSP middleware
app.add_middleware(CSPMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Mount templates directory
templates = Jinja2Templates(directory="api/templates")

# Mount static files directory
app.mount("/static", StaticFiles(directory="api/static"), name="static")

class TextInput(SecureInput):
    """Secure text input model with validation."""
    explain: bool = False
    max_length: int = 5000  # Maximum text length
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Sample text for analysis",
                "explain": True
            }
        }

class InferenceResponse(BaseModel):
    output: Dict[str, Any]
    metrics: Dict[str, Any]
    explanation: Optional[Dict[str, Any]] = None

# Initialize database
db_path = Path("storage/results.db")
db_path.parent.mkdir(exist_ok=True)
def get_db():
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS inference_results_id_seq;
            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY DEFAULT(nextval('inference_results_id_seq')),
                text TEXT NOT NULL,
                prediction JSON,
                explanation JSON,
                metrics JSON,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        yield conn
    finally:
        conn.close()

# Create FastAPI dependency
db_dependency = Depends(get_db)

# Initialize inference engine
engine = None

@app.on_event("startup")
async def startup():
    # Initialize the database on startup
    db = next(get_db())
    db.close()

    # Initialize inference engine with a proper sentiment analysis model
    global engine
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Load the sentiment analysis model with specific configuration
        logger.info("Loading sentiment analysis model...")
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        
        # Clear any existing cached files
        import shutil
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), 'sentiment_model_cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        
        # Load model with fresh configuration
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5,  # 5 classes (1-5 stars)
            cache_dir=cache_dir,
            force_download=True,  # Force fresh download
            local_files_only=False
        )
        
        # Move model to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        model = model.to(device)
        model.eval()
        
        # Reset model state
        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        
        # Clear any cached states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Test model with sample inputs
        logger.info("Testing model with sample inputs...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sample_texts = [
            "This is amazing!",  # Positive
            "This is terrible!",  # Negative
            "This is okay."  # Neutral
        ]
        
        for text in sample_texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                logger.info(f"Sample text: {text}")
                logger.info(f"Raw logits: {logits[0].tolist()}")
                logger.info(f"Probabilities: {probs[0].tolist()}")
        
        # Verify model configuration
        logger.info(f"Model config: {model.config}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Wrap the model to match our expected interface
        class SentimentModel:
            def __init__(self, model, device):
                self.model = model
                self.device = device
                self.model.eval()  # Ensure model is in eval mode
                logger.info(f"Model loaded successfully on {device}")
                
                # Test the model
                self._test_model()
            
            def _test_model(self):
                """Run a test prediction to verify model behavior."""
                test_text = "This is a test."
                logger.info(f"Running test prediction on: {test_text}")
                
                # Create test input
                tokens = tokenizer(
                    test_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Run test prediction
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    logger.info(f"Test prediction - Raw logits: {logits[0].tolist()}")
                    logger.info(f"Test prediction - Probabilities: {probs[0].tolist()}")
                
            def __call__(self, input_ids=None, attention_mask=None):
                try:
                    # Log input state
                    logger.info(f"Input shape: {input_ids.shape}")
                    logger.info(f"Input device: {input_ids.device}")
                    logger.info(f"First few input tokens: {input_ids[0][:10].tolist()}")
                    
                    # Ensure inputs are on the correct device
                    if input_ids.device != self.device:
                        input_ids = input_ids.to(self.device)
                    if attention_mask.device != self.device:
                        attention_mask = attention_mask.to(self.device)
                    
                    # Run model inference
                    with torch.no_grad():
                        # Ensure inputs are in the correct format
                        if len(input_ids.shape) == 1:
                            input_ids = input_ids.unsqueeze(0)
                        if len(attention_mask.shape) == 1:
                            attention_mask = attention_mask.unsqueeze(0)
                        
                        logger.info(f"Model input shapes - IDs: {input_ids.shape}, Mask: {attention_mask.shape}")
                        
                        # Get model outputs
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                        
                        # Extract logits
                        logits = outputs.logits
                        
                        # Apply softmax to get probabilities
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        
                        logger.info(f"Raw logits: {logits[0].tolist()}")
                        logger.info(f"Probabilities: {probs[0].tolist()}")
                        
                        return outputs
                        # Log input shapes after correction
                        logger.info(f"Final input shapes - IDs: {input_ids.shape}, Mask: {attention_mask.shape}")
                        
                        # Get model outputs
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                        
                        # Extract logits
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                        
                        # Log raw outputs
                        logger.info(f"Raw logits shape: {logits.shape}")
                        logger.info(f"Sample raw logits: {logits[0].tolist()}")
                        
                        # Create a custom output object
                        return type('ModelOutput', (), {
                            'logits': logits,
                            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
                        })
                        
                except Exception as e:
                    logger.error(f"Error during model inference: {str(e)}")
                    logger.error(f"Input IDs shape: {input_ids.shape if input_ids is not None else None}")
                    logger.error(f"Attention mask shape: {attention_mask.shape if attention_mask is not None else None}")
                    raise
            
            def get_spec(self):
                return {
                    'model_name': model_name,
                    'device': str(self.device),
                    'num_labels': 5,
                    'config': self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else str(self.model.config)
                }
        
        # Initialize the inference engine with the wrapped model
        engine = InferenceEngine(
            model_path=SentimentModel(model, device),  # Pass device to model wrapper
            tokenizer_name="nlptown/bert-base-multilingual-uncased-sentiment",
            config=InferenceConfig(
                min_batch_size=1,
                max_batch_size=8,
                cache_size=1024,
                num_threads=4,
                target_latency_ms=100.0,
                temperature=0.7,  # Adjusted for more precise predictions
                confidence_threshold=0.5  # Lower threshold for more sensitivity
            )
        )
        logger.info("Inference engine initialized successfully")
        
        logger.info("Sentiment analysis model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/inference")
async def process_input(input_data: TextInput):
    try:
        input_text = input_data.text
        
        if not input_text:
            raise HTTPException(status_code=400, detail="No input provided")
            
        try:
            # Create TextInput object with explanations enabled
            input_data = TextInput(text=input_text, explain=True)
            
            # Use analyze_text endpoint
            result = await analyze_text(input_data, db=next(get_db()))
            
            return result
            
            # Ensure we have a valid response structure with defaults
            response = {
                "output": {
                    "text": input_text,
                    "prediction": prediction.get('prediction', 1),  # Default to neutral
                    "confidence": prediction.get('confidence', 0.0),
                    "probabilities": prediction.get('probabilities', [0.0, 1.0, 0.0])  # Default to neutral
                },
                "metrics": {
                    "latency_ms": metrics.get("latency_ms", 0.0),
                    "throughput": metrics.get("throughput", 0.0),
                    "gpu_utilization": metrics.get("gpu_utilization", 0.0),
                    "memory_utilization": metrics.get("memory_utilization", 0.0)
                }
            }
            
            # Log the response for debugging
            logger.info(f"Inference response: {response}")
            
            return response
        except Exception as e:
            logger.error(f"Error in inference endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class CSVOptions(BaseModel):
    text_column: Optional[str] = None  # Column name containing the text
    column_index: Optional[int] = None  # Column index (0-based) containing the text
    batch_size: int = 100  # Number of rows to process at once

class FileUploadRequest(BaseModel):
    csv_options: Optional[CSVOptions] = None

async def process_file_content(file_content: bytes, filename: str, upload_options: FileUploadRequest) -> Tuple[str, List[str]]:
    """Process a single file's content and return texts to analyze"""
    texts = []
    if filename.endswith('.pdf'):
        # Handle PDF
        pdf = PdfReader(io.BytesIO(file_content))
        texts = [page.extract_text() for page in pdf.pages]
    elif filename.endswith('.txt'):
        # Handle text file
        texts = [file_content.decode('utf-8')]
    elif filename.endswith('.csv'):
        # Handle CSV file with enhanced options
        import pandas as pd
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Determine which column to use
        if upload_options.csv_options:
            if upload_options.csv_options.text_column:
                if upload_options.csv_options.text_column not in df.columns:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column {upload_options.csv_options.text_column} not found in CSV. Available columns: {', '.join(df.columns)}"
                    )
                series = df[upload_options.csv_options.text_column]
            elif upload_options.csv_options.column_index is not None:
                if upload_options.csv_options.column_index >= len(df.columns):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column index {upload_options.csv_options.column_index} out of range. CSV has {len(df.columns)} columns"
                    )
                series = df.iloc[:, upload_options.csv_options.column_index]
            else:
                # Default to first column
                series = df.iloc[:, 0]
        else:
            # Default to first column
            series = df.iloc[:, 0]
        
        # Process in batches
        batch_size = upload_options.csv_options.batch_size if upload_options.csv_options else 100
        texts = [series[i:i + batch_size].astype(str).tolist() 
                for i in range(0, len(series), batch_size)]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for {filename}. Only PDF, TXT, and CSV files are supported."
        )
    
    return filename, texts

async def process_batch(batch: Union[str, List[str]], engine: InferenceEngine) -> Dict[str, Any]:
    """Process a single batch of text"""
    if isinstance(batch, list):
        batch_text = "\n".join(batch)
    else:
        batch_text = batch
        
    prediction = await engine.infer_async(batch_text, explain=True)
    
    # Generate concise explanation based on probabilities
    probabilities = prediction.get('probabilities', [])
    labels = ['Negative', 'Neutral', 'Positive']
    explanation = []
    
    if probabilities:
        # Find the highest and second highest probabilities
        sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
        top_label = labels[sorted_probs[0][0]]
        top_prob = sorted_probs[0][1]
        second_label = labels[sorted_probs[1][0]]
        second_prob = sorted_probs[1][1]
        
        # Calculate difference from second highest
        diff = (top_prob - second_prob) * 100
        
        # Generate concise explanation
        if diff < 10:
            explanation.append(f"Marginally {top_label.lower()} ({top_prob*100:.1f}%), with {second_label.lower()} close behind ({second_prob*100:.1f}%)")
        elif diff < 30:
            explanation.append(f"Moderately {top_label.lower()} ({top_prob*100:.1f}%), with some {second_label.lower()} elements ({second_prob*100:.1f}%)")
        else:
            explanation.append(f"Strongly {top_label.lower()} ({top_prob*100:.1f}%), significantly outweighing {second_label.lower()} ({second_prob*100:.1f}%)")
    
    return {
        'output': {
            'prediction': prediction.get('prediction'),
            'confidence': prediction.get('confidence'),
            'probabilities': probabilities,
            'explanation': explanation
        },
        'text': batch_text[:1000] + '...' if len(batch_text) > 1000 else batch_text
    }

async def process_single_file(file: UploadFile, upload_options: FileUploadRequest, engine: InferenceEngine) -> Dict[str, Any]:
    """Process a single file completely"""
    try:
        content = b""
        # Read file in chunks
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            if len(content) + len(chunk) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail=f"File {file.filename} too large (max 10MB)")
            content += chunk

        filename, texts = await process_file_content(content, file.filename, upload_options)
        
        # Process batches in parallel
        batch_results = await asyncio.gather(*[process_batch(batch, engine) for batch in texts])
        
        return {
            'filename': filename,
            'results': batch_results,
            'metrics': engine.get_performance_metrics()
        }
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise

@app.post("/upload", dependencies=get_auth_dependency())
async def upload_file(
    request: Request,
    files: List[UploadFile] = File(...),
    options: str = Form(default='{}')
):
    try:
        # Parse options
        upload_options = FileUploadRequest(**json.loads(options))
        
        # Process all files in parallel
        start_time = time.time()
        results = await asyncio.gather(
            *[process_single_file(file, upload_options, engine) for file in files],
            return_exceptions=True
        )
        
        # Check for errors
        errors = []
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    'filename': files[i].filename,
                    'error': str(result)
                })
            else:
                valid_results.append(result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'files': valid_results,
            'errors': errors if errors else None,
            'total_files': len(files),
            'successful_files': len(valid_results),
            'failed_files': len(errors),
            'total_processing_time': processing_time,
            'average_time_per_file': processing_time / len(files) if files else 0
        }
        
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=InferenceResponse)
async def analyze_text(input_data: TextInput, db: duckdb.DuckDBPyConnection = db_dependency):
    try:
        # Validate input
        if not isinstance(input_data.text, str):
            raise HTTPException(status_code=400, detail="Text input must be a string")
            
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text input must be a non-empty string")
            
        if len(input_data.text) > 10000:  # Arbitrary limit
            raise HTTPException(status_code=400, detail="Text input is too long (max 10000 characters)")
        
        logger.info(f"Processing text: {input_data.text[:100]}...")
        
        try:
            # Perform inference
            try:
                prediction = await engine.infer_async(input_data.text)
                logger.info(f"Raw prediction: {prediction}")
                
                # Validate prediction structure
                if not isinstance(prediction, dict):
                    logger.error(f"Invalid prediction structure: {type(prediction)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid response type from inference engine: {type(prediction)}"
                    )
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")
            
            # Get explanation if requested
            explanation = None
            if input_data.explain:
                try:
                    explanation = await engine.explain(input_data.text)
                except Exception as e:
                    logger.warning(f"Failed to generate explanation: {e}")
                    explanation = {"error": str(e)}
            
            # Log prediction details
            logger.info(f"Prediction keys: {prediction.keys()}")
            if 'debug_info' in prediction:
                logger.info(f"Debug info: {prediction['debug_info']}")
            
            # Generate explanation based on probabilities
            probabilities = prediction.get('probabilities', [])
            explanation = {'details': {}}
            if probabilities:
                labels = ['Negative', 'Neutral', 'Positive']
                sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
                top_label = labels[sorted_probs[0][0]]
                top_prob = sorted_probs[0][1]
                second_label = labels[sorted_probs[1][0]]
                second_prob = sorted_probs[1][1]
                
                # Calculate difference from second highest
                diff = (top_prob - second_prob) * 100
                
                # Generate concise explanation
                explanation['details']['top_label'] = top_label
                explanation['details']['top_probability'] = top_prob
                explanation['details']['second_label'] = second_label
                explanation['details']['second_probability'] = second_prob
                
                if diff < 10:
                    explanation['summary'] = f"Marginally {top_label.lower()} ({top_prob*100:.1f}%), with {second_label.lower()} close behind ({second_prob*100:.1f}%)"
                elif diff < 30:
                    explanation['summary'] = f"Moderately {top_label.lower()} ({top_prob*100:.1f}%), with some {second_label.lower()} elements ({second_prob*100:.1f}%)"
                else:
                    explanation['summary'] = f"Strongly {top_label.lower()} ({top_prob*100:.1f}%), significantly outweighing {second_label.lower()} ({second_prob*100:.1f}%)"
            
            # Convert string prediction to numeric
            prediction_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            pred_str = prediction.get('prediction')
            pred_num = prediction_map.get(pred_str) if isinstance(pred_str, str) else pred_str
            
            # Format response with detailed logging
            response = {
                'output': {
                    'text': input_data.text,
                    'prediction': pred_num,
                    'prediction_label': pred_str if isinstance(pred_str, str) else list(prediction_map.keys())[pred_str],
                    'confidence': prediction.get('confidence'),
                    'probabilities': prediction.get('probabilities', []),
                    'explanation': explanation['summary'] if explanation else None
                },
                'metrics': {
                    'latency_ms': prediction.get('performance', {}).get('latency_ms', 0),
                    'throughput': prediction.get('performance', {}).get('throughput', 0),
                    'gpu_utilization': prediction.get('performance', {}).get('gpu_utilization', 0.5),
                    'memory_utilization': prediction.get('performance', {}).get('memory_utilization', 0.3)
                }
            }
            
            # Log the formatted response
            logger.info(f"Formatted response: {response}")
            
            # Validate required fields with detailed error messages
            if not prediction or not isinstance(prediction, dict) or 'prediction' not in prediction:
                error_msg = f"Invalid or missing prediction in response. Available data: {prediction}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )
            
            # Store result in database
            try:
                db.execute("""
                    INSERT INTO inference_results (text, prediction, metrics, explanation)
                    VALUES (?, ?, ?, ?)
                """, [input_data.text, json.dumps(response['output']), 
                       json.dumps(response['metrics']), 
                       json.dumps(explanation['details'] if explanation else None)])
                db.commit()
            except Exception as db_error:
                logger.error(f"Database error: {db_error}")
                # Continue even if DB storage fails
            
            return InferenceResponse(
                output=response['output'],
                metrics=response['metrics'],
                explanation=explanation if explanation else None
            )
            
        except Exception as inference_error:
            logger.error(f"Inference error: {inference_error}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process text. Please try again later."
            )
            
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
        
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

@app.get("/stats")
async def get_stats(db: duckdb.DuckDBPyConnection = db_dependency):
    try:
        # Get basic statistics
        stats = db.execute("""
            SELECT 
                COUNT(*) as total_requests,
                AVG(EPOCH(CURRENT_TIMESTAMP) - EPOCH(timestamp)) as avg_age_seconds
            FROM inference_results
        """).fetchone()
        
        # Get performance metrics - use the last metrics to ensure consistency
        metrics = metrics_tracker.get_last_metrics()
        
        return {
            "total_requests": stats[0],
            "avg_request_age_seconds": stats[1] if stats[1] is not None else 0,
            **metrics
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
