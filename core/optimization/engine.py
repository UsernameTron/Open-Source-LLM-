import numpy as np
import coremltools as ct
from typing import Dict, List, Tuple, Optional
import threading
import queue
import time
import logging
from dataclasses import dataclass
from core.monitoring.metrics import MetricsCollector, InferenceMetrics
import psutil

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    min_batch_size: int = 1
    max_batch_size: int = 128
    target_latency_ms: float = 50
    max_queue_size: int = 1000
    adaptive_batching: bool = True
    confidence_threshold: float = 0.9
    memory_threshold: float = 0.9
    gpu_target_utilization: float = 0.8

class OptimizedInferenceEngine:
    def __init__(
        self,
        model_path: str,
        config: OptimizationConfig = None
    ):
        self.model = ct.models.MLModel(model_path)
        self.config = config or OptimizationConfig()
        self.metrics_collector = MetricsCollector()
        
        # Request queue for batching
        self.request_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.results_map: Dict[int, Dict] = {}
        self.next_request_id = 0
        self.request_lock = threading.Lock()
        
        # Start batch processing thread
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processing_loop)
        self.batch_thread.daemon = True
        self.batch_thread.start()
        
    def predict(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform prediction with automatic batching and optimization.
        Returns prediction and confidence score.
        """
        with self.request_lock:
            request_id = self.next_request_id
            self.next_request_id += 1
            
        # Add request to queue
        self.request_queue.put({
            'id': request_id,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'timestamp': time.time()
        })
        
        # Wait for results
        while request_id not in self.results_map:
            time.sleep(0.001)
            
        result = self.results_map.pop(request_id)
        return result['prediction'], result['confidence']
        
    def _batch_processing_loop(self):
        """Main loop for batch processing."""
        while self.running:
            batch = self._collect_batch()
            if not batch:
                time.sleep(0.001)
                continue
                
            # Process batch
            start_time = time.time()
            try:
                predictions, confidences = self._process_batch(batch)
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                gpu_util = self._get_gpu_utilization()
                memory_util = psutil.virtual_memory().percent / 100
                
                metrics = InferenceMetrics(
                    latency_ms=latency_ms,
                    batch_size=len(batch),
                    confidence_scores=confidences.tolist(),
                    gpu_utilization=gpu_util,
                    memory_utilization=memory_util
                )
                self.metrics_collector.add_inference_metrics(metrics)
                
                # Store results
                for i, request in enumerate(batch):
                    self.results_map[request['id']] = {
                        'prediction': predictions[i],
                        'confidence': confidences[i]
                    }
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Return error results
                for request in batch:
                    self.results_map[request['id']] = {
                        'prediction': np.zeros((2,)),
                        'confidence': 0.0
                    }
                    
    def _collect_batch(self) -> List[Dict]:
        """Collect requests into a batch based on current conditions."""
        if self.request_queue.empty():
            return []
            
        # Determine optimal batch size
        current_batch_size = self._get_optimal_batch_size()
        
        # Collect batch
        batch = []
        try:
            while len(batch) < current_batch_size and not self.request_queue.empty():
                batch.append(self.request_queue.get_nowait())
        except queue.Empty:
            pass
            
        return batch
        
    def _process_batch(self, batch: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch of requests."""
        # Process each input individually to maintain CoreML's expected shapes
        all_logits = []
        all_confidences = []
        
        for request in batch:
            # Run inference
            outputs = self.model.predict({
                'input_ids': request['input_ids'],
                'attention_mask': request['attention_mask']
            })
            
            logits = outputs['linear_37']
            probabilities = self._softmax(logits)
            confidence = float(np.max(probabilities))
            
            all_logits.append(logits)
            all_confidences.append(confidence)
        
        return np.array(all_logits), np.array(all_confidences)
        
    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on current conditions."""
        if not self.config.adaptive_batching:
            return self.config.max_batch_size
            
        # Get current metrics
        memory_util = psutil.virtual_memory().percent / 100
        gpu_util = self._get_gpu_utilization()
        
        # Adjust batch size based on resource utilization
        if memory_util > self.config.memory_threshold:
            return self.config.min_batch_size
        
        if gpu_util < self.config.gpu_target_utilization:
            return min(
                self.config.max_batch_size,
                self._get_current_batch_size() * 2
            )
        
        return self._get_current_batch_size()
        
    def _get_current_batch_size(self) -> int:
        """Get current batch size from recent metrics."""
        recent_metrics = list(self.metrics_collector.metrics_window)[-10:]
        if not recent_metrics:
            return self.config.min_batch_size
        return int(np.median([m.batch_size for m in recent_metrics]))
        
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        # This is a placeholder - implement actual GPU monitoring
        # For Metal GPU, you might need to use IOKit or other Mac-specific APIs
        return 0.8  # Placeholder
        
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each set of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
        
    def __del__(self):
        self.running = False
        if hasattr(self, 'batch_thread'):
            self.batch_thread.join()
