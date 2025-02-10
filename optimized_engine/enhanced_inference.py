"""
Enhanced CoreML inference engine with optimized performance, confidence calibration,
and advanced monitoring capabilities.
"""
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer
import time
import logging
import queue
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import re
import json
from pathlib import Path
import prometheus_client as prom
from datetime import datetime
import emoji
import traceback
import psutil
import collections
from logging.handlers import RotatingFileHandler

# Configure logging with more detail
log_formatter = logging.Formatter(
    '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s'
)

# File handler with rotation
log_file = Path('logs/inference_engine.log')
log_file.parent.mkdir(exist_ok=True)
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
LATENCY_HISTOGRAM = prom.Histogram(
    'inference_latency_ms',
    'Inference latency in milliseconds',
    ['batch_size', 'priority']
)
CONFIDENCE_HISTOGRAM = prom.Histogram(
    'prediction_confidence',
    'Model prediction confidence',
    ['dataset_type']
)
QUEUE_DEPTH_GAUGE = prom.Gauge(
    'inference_queue_depth',
    'Current inference queue depth'
)
BATCH_SIZE_GAUGE = prom.Gauge(
    'current_batch_size',
    'Current batch size being used'
)

@dataclass
class InferenceRequest:
    """Container for inference requests with metadata."""
    text: str
    priority: int
    timestamp: float
    request_id: str
    dataset_type: str = 'standard'  # standard, mixed, neutral, edge_case
    
    def __lt__(self, other):
        # Lower priority number means higher priority
        if self.priority != other.priority:
            return self.priority < other.priority
        # If priorities are equal, use timestamp (FIFO)
        return self.timestamp < other.timestamp

@dataclass
class InferenceResult:
    """Container for inference results with performance metrics."""
    text: str
    label: str
    confidence: float
    latency_ms: float
    batch_size: int
    queue_depth: int
    request_id: str
    temperature: float

class EnhancedInferenceConfig:
    """Configuration for the enhanced inference engine."""
    def __init__(self):
        # Batch processing configuration (optimized based on benchmarks)
        self.min_batch_size: int = 8  # Increased based on performance tests
        self.max_batch_size: int = 16  # Reduced to maintain consistent latency
        self.target_latency_ms: float = 20.0  # Adjusted for realistic target
        self.max_queue_size: int = 1000
        
        # Confidence thresholds (tuned for better neutral detection)
        self.confidence_threshold: float = 0.6  # Lowered for better neutral detection
        self.neutral_threshold: float = 0.2  # Confidence difference threshold for neutral
        self.min_confidence: float = 0.4  # Minimum confidence for any prediction
        
        # Temperature scaling
        self.temperature: float = 1.2  # Optimal from temperature scaling tests
        
        # System configuration
        self.metrics_window: int = 100
        self.num_threads: int = min(os.cpu_count() * 2, 8)
        self.metrics_export_path: str = "metrics"
        self.max_length: int = 512
        
        # Dynamic batching thresholds (adjusted for optimal throughput)
        self.low_queue_threshold: int = 5   # Trigger small batches sooner
        self.high_queue_threshold: int = 20 # Earlier transition to max batch

class InferenceError(Exception):
    """Base class for inference engine errors."""
    pass

class ModelLoadError(InferenceError):
    """Error loading model or tokenizer."""
    pass

class BatchProcessingError(InferenceError):
    """Error processing a batch of requests."""
    pass

class MetricsError(InferenceError):
    """Error collecting or processing metrics."""
    pass

class EnhancedInferenceEngine:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        config: Optional[EnhancedInferenceConfig] = None
    ):
        """Initialize the enhanced inference engine."""
        self.config = config or EnhancedInferenceConfig()
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model = ct.models.MLModel(model_path)
        self.model.compute_units = ct.ComputeUnit.ALL
        
        logger.info(f"Loading tokenizer {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize queues and thread pool
        self.request_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        # Initialize temperature
        self.temperature = self.config.temperature
        
        # Initialize monitoring
        self.metrics_path = Path(self.config.metrics_export_path)
        self.metrics_path.mkdir(exist_ok=True)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_metrics,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Temperature scaling parameters (to be calibrated)
        self.temperature = self.config.temperature
        
        # Preprocessing patterns
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Remove special characters while preserving structure
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _get_optimal_batch_size(self, queue_depth: int) -> int:
        """Determine optimal batch size based on queue depth."""
        if queue_depth < self.config.low_queue_threshold:
            return self.config.min_batch_size
        elif queue_depth < self.config.high_queue_threshold:
            return 16  # Balanced mode
        else:
            return self.config.max_batch_size
    
    def _apply_temperature_scaling(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def _process_batch(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Process a batch of requests with enhanced monitoring."""
        start_time = time.time()
        batch_size = len(requests)
        queue_depth = self.request_queue.qsize()
        
        # Update monitoring
        BATCH_SIZE_GAUGE.set(batch_size)
        QUEUE_DEPTH_GAUGE.set(queue_depth)
        
        try:
            # Preprocess texts
            texts = [self._preprocess_text(req.text) for req in requests]
            
            # Tokenize
            tokens = self.tokenizer(
                texts,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            
            # Prepare inputs
            inputs = {
                "input_ids": tokens["input_ids"].astype(np.int32),
                "attention_mask": tokens["attention_mask"].astype(np.int32)
            }
            
            # Run inference
            predictions = self.model.predict(inputs)
            logits = predictions.get('linear_37', predictions)
            
            # Apply temperature scaling
            scaled_logits = self._apply_temperature_scaling(logits)
            
            # Process results with improved neutral detection
            results = []
            for i, request in enumerate(requests):
                # Get logits for current text
                text_logits = scaled_logits[i] if isinstance(scaled_logits, np.ndarray) else scaled_logits
                
                # Apply softmax
                exp_logits = np.exp(text_logits - np.max(text_logits))
                probabilities = exp_logits / exp_logits.sum()
                
                # Get top 2 predictions and confidences
                top2_indices = np.argsort(probabilities)[-2:]
                confidence1 = float(probabilities[top2_indices[-1]])
                confidence2 = float(probabilities[top2_indices[-2]])
                
                # Determine label based on confidence spread
                confidence_spread = confidence1 - confidence2
                if confidence1 < self.config.min_confidence:
                    label = 'neutral'
                    confidence = 1.0 - confidence1  # High confidence in neutrality
                elif confidence_spread < self.config.neutral_threshold:
                    label = 'neutral'
                    confidence = 1.0 - confidence_spread  # High confidence when spread is small
                else:
                    predicted_class = top2_indices[-1]
                    label = 'positive' if predicted_class == 1 else 'negative'
                    confidence = confidence1
                
                # Calculate latency
                latency = (time.time() - start_time) * 1000 / batch_size
                
                # Create result
                result = InferenceResult(
                    text=request.text,
                    label=label,
                    confidence=confidence,
                    latency_ms=latency,
                    batch_size=batch_size,
                    queue_depth=queue_depth,
                    request_id=request.request_id,
                    temperature=self.temperature
                )
                
                # Update metrics
                LATENCY_HISTOGRAM.labels(
                    batch_size=str(batch_size),
                    priority=str(request.priority)
                ).observe(latency)
                
                CONFIDENCE_HISTOGRAM.labels(
                    dataset_type=request.dataset_type
                ).observe(confidence)
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _get_latency_stats(self) -> Dict[str, float]:
        """Calculate comprehensive latency statistics."""
        try:
            samples = [s.value for s in LATENCY_HISTOGRAM.collect()[0].samples]
            if not samples:
                return {}
            
            return {
                'mean': float(np.mean(samples)),
                'p50': float(np.percentile(samples, 50)),
                'p95': float(np.percentile(samples, 95)),
                'p99': float(np.percentile(samples, 99)),
                'min': float(np.min(samples)),
                'max': float(np.max(samples))
            }
        except Exception as e:
            logger.error(f"Error calculating latency stats: {e}\n{traceback.format_exc()}")
            return {}
    
    def _get_confidence_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive confidence statistics."""
        try:
            samples = [s.value for s in CONFIDENCE_HISTOGRAM.collect()[0].samples]
            if not samples:
                return {}
            
            low_conf_samples = [s for s in samples if s < self.config.confidence_threshold]
            
            return {
                'mean': float(np.mean(samples)),
                'median': float(np.median(samples)),
                'std': float(np.std(samples)),
                'min': float(np.min(samples)),
                'max': float(np.max(samples)),
                'low_confidence_count': len(low_conf_samples),
                'low_confidence_mean': float(np.mean(low_conf_samples)) if low_conf_samples else 0
            }
        except Exception as e:
            logger.error(f"Error calculating confidence stats: {e}\n{traceback.format_exc()}")
            return {}
    
    def _get_resource_stats(self) -> Dict[str, Any]:
        """Get system resource utilization statistics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            stats = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'thread_count': threading.active_count()
            }
            
            # Add GPU stats if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    stats.update({
                        'gpu_utilization': gpu.load * 100,
                        'gpu_memory_percent': gpu.memoryUtil * 100
                    })
            except ImportError:
                pass
            
            return stats
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}\n{traceback.format_exc()}")
            return {}
    
    def _normalize_confidence(self, prob_dist: np.ndarray) -> float:
        """Normalize confidence scores using softmax distribution properties."""
        # Sort probabilities in descending order
        sorted_probs = np.sort(prob_dist)[::-1]
        
        # Calculate confidence based on probability gap and distribution
        prob_gap = sorted_probs[0] - sorted_probs[1]
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        max_entropy = -np.log(1/len(prob_dist))
        
        # Combine metrics for final confidence
        confidence = (prob_gap * (1 - entropy/max_entropy))**0.5
        return np.clip(confidence, 0, 1)
    
    def _get_dynamic_temperature(self, latency: float) -> float:
        """Dynamically adjust temperature based on latency and performance."""
        base_temp = self.config.temperature
        
        # Increase temperature if latency is too high
        if latency > self.config.latency_threshold:
            temp_scale = min(latency / self.config.latency_threshold, 2.0)
            return base_temp * temp_scale
        
        return base_temp
    
    def _log_batch_stats(self, results: List[InferenceResult]):
        """Log detailed statistics for the processed batch."""
        confidences = [r.confidence for r in results]
        latencies = [r.latency for r in results]
        sentiments = [r.sentiment for r in results]
        
        stats = {
            'batch_size': len(results),
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'mean_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'sentiment_distribution': Counter(sentiments)
        }
        
        logger.info(f"Batch Stats: {json.dumps(stats, indent=2)}")
    
    def _monitor_metrics(self):
        """Monitor and export metrics with enhanced error handling and adaptive thresholds."""
        metrics_buffer = collections.deque(maxlen=100)  # Keep last 100 measurements
        last_export_time = time.time()
        export_interval = 10  # Export every 10 seconds
        
        while True:
            try:
                current_time = time.time()
                
                # Collect current metrics
                queue_size = self.request_queue.qsize()
                batch_size = BATCH_SIZE_GAUGE._value.get() if hasattr(BATCH_SIZE_GAUGE, '_value') else 0
                
                # Update metrics buffer with moving averages
                metrics_buffer.setdefault('queue_depth', []).append(queue_size)
                metrics_buffer.setdefault('batch_size', []).append(batch_size)
                
                # Keep buffer size manageable
                window = self.config.metrics_window
                for key in metrics_buffer:
                    if len(metrics_buffer[key]) > window:
                        metrics_buffer[key] = metrics_buffer[key][-window:]
                
                # Export metrics periodically
                if current_time - last_export_time >= export_interval:
                    current_metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'queue_depth': {
                            'current': queue_size,
                            'mean': np.mean(metrics_buffer['queue_depth']),
                            'max': max(metrics_buffer['queue_depth'])
                        },
                        'batch_size': {
                            'current': batch_size,
                            'mean': np.mean(metrics_buffer['batch_size']),
                            'min': min(metrics_buffer['batch_size'])
                        },
                        'temperature': self.temperature,
                        'latency_stats': self._get_latency_stats(),
                        'confidence_stats': self._get_confidence_stats()
                    }
                    
                    # Save metrics
                    metrics_file = self.metrics_path / 'current_metrics.json'
                    temp_file = metrics_file.with_suffix('.tmp')
                    try:
                        with open(temp_file, 'w') as f:
                            json.dump(current_metrics, f, indent=2)
                        temp_file.replace(metrics_file)  # Atomic update
                    except Exception as e:
                        logger.error(f"Failed to save metrics: {e}")
                    
                    # Check for alerts
                    self._check_alerts(current_metrics)
                    last_export_time = current_time
                
                time.sleep(0.1)  # More frequent updates, less CPU intensive
                
            except Exception as e:
                logger.error(f"Metrics monitoring failed: {e}")
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics for alert conditions with improved error handling and context."""
        try:
            # Latency alerts
            latency_stats = metrics.get('latency_stats', {})
            p95_latency = latency_stats.get('p95', 0)
            if p95_latency > self.config.target_latency_ms:
                logger.warning(
                    f"High latency detected:\n"
                    f"  - P95 latency: {p95_latency:.2f}ms (target: {self.config.target_latency_ms}ms)\n"
                    f"  - Current batch size: {metrics['batch_size']['current']}\n"
                    f"  - Queue depth: {metrics['queue_depth']['current']}"
                )
            
            # Queue depth alerts
            queue_depth = metrics['queue_depth']['current']
            if queue_depth > self.config.max_queue_size * 0.9:  # Alert at 90% capacity
                logger.warning(
                    f"High queue depth detected:\n"
                    f"  - Current depth: {queue_depth} (max: {self.config.max_queue_size})\n"
                    f"  - Mean depth: {metrics['queue_depth']['mean']:.1f}\n"
                    f"  - Batch size: {metrics['batch_size']['current']}"
                )
            
            # Confidence alerts
            conf_stats = metrics.get('confidence_stats', {})
            low_conf_count = conf_stats.get('low_confidence_count', 0)
            if low_conf_count > 0:
                logger.warning(
                    f"Low confidence predictions detected:\n"
                    f"  - Count: {low_conf_count}\n"
                    f"  - Mean confidence: {conf_stats.get('mean', 0):.2%}\n"
                    f"  - Min confidence: {conf_stats.get('min', 0):.2%}"
                )
            
            # Resource alerts
            if psutil.cpu_percent() > 80:
                logger.warning(f"High CPU usage: {psutil.cpu_percent()}%")
            if psutil.virtual_memory().percent > 80:
                logger.warning(f"High memory usage: {psutil.virtual_memory().percent}%")
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def infer_async(
        self,
        text: str,
        priority: int = 0,
        dataset_type: str = 'standard'
    ) -> str:
        """Asynchronous inference with priority queuing."""
        request = InferenceRequest(
            text=text,
            priority=priority,
            timestamp=time.time(),
            request_id=f"req_{int(time.time()*1000)}",
            dataset_type=dataset_type
        )
        
        self.request_queue.put((priority, request))
        return request.request_id
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[InferenceResult]:
        """Get result for a specific request."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Process queue
            if self.request_queue.qsize() > 0:
                batch_size = self._get_optimal_batch_size(self.request_queue.qsize())
                batch = []
                
                # Collect batch
                while len(batch) < batch_size and not self.request_queue.empty():
                    _, request = self.request_queue.get()
                    batch.append(request)
                
                # Process batch
                if batch:
                    results = self._process_batch(batch)
                    
                    # Check if our request is in results
                    for result in results:
                        if result.request_id == request_id:
                            return result
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
        
        return None  # Timeout
    
    def infer(self, text: str, priority: int = 0) -> InferenceResult:
        """Synchronous inference."""
        request_id = self.infer_async(text, priority)
        result = self.get_result(request_id)
        if result is None:
            raise TimeoutError("Inference request timed out")
        return result
