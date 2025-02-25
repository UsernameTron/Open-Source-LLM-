import coremltools as ct
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import traceback
import threading
from functools import lru_cache
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer
import time
import asyncio

@dataclass
class ResourceMetrics:
    gpu_utilization: float
    memory_utilization: float
    batch_latency: float
    queue_size: int
    throughput: float
    timestamp: float = time.time()
    cpu_util: float = 0.0
    temperature: float = 0.0
    power_draw: float = 0.0

@dataclass
class InferenceConfig:
    # Optimized batch sizes based on benchmark results
    min_batch_size: int = 8  # Minimum batch size for efficiency
    max_batch_size: int = 32  # Optimal batch size from benchmarks
    target_latency_ms: float = 15.0  # Target latency based on benchmarks
    max_queue_size: int = 1000  # Queue size for request handling
    cache_size: int = 8192  # Increased for better hit rate
    num_threads: int = 4  # Optimized for stability
    metrics_window: int = 300  # Window for metrics collection
    adaptive_batching: bool = True
    max_length: int = 512  # Maximum sequence length
    confidence_threshold: float = 0.85  # Minimum confidence threshold
    temperature: float = 0.7  # Temperature for confidence scaling
    ensemble_size: int = 3  # Number of forward passes for confidence estimation
    neutral_threshold: float = 0.2  # Threshold for neutral classification
    enable_caching: bool = True  # Enable result caching
    cache_ttl_seconds: int = 3600  # Cache TTL in seconds

class InferenceEngine:
    def __init__(
        self,
        config: InferenceConfig,
        model_path: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None
    ):
        """Initialize the inference engine."""
        # Set up logging first
        self.logger = logging.getLogger(__name__)
        
        try:
            self.config = config
            self.model = None
            self.tokenizer = None
            self.id2label = id2label or {0: "Negative", 1: "Neutral", 2: "Positive"}
            self.label2id = label2id or {"Negative": 0, "Neutral": 1, "Positive": 2}
            
            # Load model and tokenizer
            self.logger.info(f"Loading model from {model_path}...")
            if isinstance(model_path, str) and model_path.endswith('.mlpackage'):
                self.model = ct.models.MLModel(model_path)
                # Get model metadata
                spec = self.model.get_spec()
                self.logger.info(f"Model loaded successfully. Input shape: {spec.description.input[0].type.multiArrayType.shape}")
            else:
                raise ValueError("Invalid model path or format")
            
            # Load tokenizer
            self.logger.info(f"Loading tokenizer {tokenizer_name}...")
            if tokenizer_name is None:
                raise ValueError("Tokenizer name must be provided")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise

    async def _preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess text input for inference."""
        try:
            # Tokenize the text
            encoded = self.tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Create input dictionary
            input_data = {
                'text': text,  # Add the original text
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            }
            
            return input_data
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise

    async def infer_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis on a single text input."""
        try:
            # Preprocess text
            input_data = await self._preprocess_text(text)
            
            # Run inference
            results = await self._run_model_inference(input_data)
            
            # Return first result
            if results and isinstance(results, list):
                return results[0]
            return {
                'error': 'Inference failed',
                'details': 'No results returned',
                'text': text
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment inference: {str(e)}")
            return {
                'error': 'Inference failed',
                'details': str(e),
                'text': text
            }
        
    async def _run_model_inference(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run model inference asynchronously with proper resource management."""
        try:
            # Validate inputs
            if not input_data:
                raise RuntimeError("Error in model inference: list index out of range")
            if 'text' not in input_data:
                raise RuntimeError("Error in model inference: Missing text field")
            
            # Convert tensors to numpy arrays for CoreML
            input_ids = input_data['input_ids'].numpy()
            attention_mask = input_data['attention_mask'].numpy()
            
            # Log input shapes and types for debugging
            self.logger.info(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            self.logger.info(f"Input types - input_ids: {input_ids.dtype}, attention_mask: {attention_mask.dtype}")
            
            # Ensure inputs are float32 for CoreML
            input_ids = input_ids.astype(np.float32)
            attention_mask = attention_mask.astype(np.float32)
            
            self.logger.info(f"Running inference on text: {input_data['text'][:100]}...")
            
            # Run inference using CoreML model's predict method with named inputs
            start_time = time.time()
            outputs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.predict({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
            )
            
            # Log model outputs for debugging
            self.logger.info(f"Model outputs: {outputs.keys()}")
            
            # Process outputs - CoreML model returns output as 'var_XX'
            logits = list(outputs.values())[0]  # Get first output tensor
            probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            prediction_idx = np.argmax(probabilities, axis=-1)[0]
            confidence = np.max(probabilities)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result = [{
                'text': input_data['text'],
                'prediction': self.id2label[prediction_idx],
                'confidence': float(confidence),
                'processing_time': processing_time,
                'probabilities': probabilities.tolist()[0]
            }]
            
            self.logger.info(f"Inference complete: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in model inference: {e}", exc_info=True)
            raise

    async def infer(self, text: str) -> Dict[str, Any]:
        """Run inference on a single text input."""
        try:
            # Preprocess the text
            input_data = await self._preprocess_text(text)
            
            # Run model inference
            result = await self._run_model_inference(input_data)
            
            if not result:
                raise ValueError("Model returned no results")
                
            # Extract the first result since we only process one text
            prediction = result[0]
            
            return {
                'text': text,
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'processing_time': prediction['processing_time']
            }
            
        except Exception as e:
            self.logger.error(f"Error in inference: {e}")
            raise
