"""Model warmup and performance monitoring module."""
import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import psutil
import asyncio
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class WarmupMetrics:
    """Metrics collected during model warmup."""
    latency_ms: float
    memory_mb: float
    temperature_c: float
    power_w: float
    compute_util: float
    batch_size: int
    queue_depth: int
    timestamp: float

@dataclass
class WarmupConfig:
    """Configuration for model warmup."""
    initial_batch_size: int = 1
    max_batch_size: int = 32
    target_latency_ms: float = 100.0
    warmup_iterations: int = 50
    scaling_factor: float = 1.2
    cooldown_seconds: float = 1.0
    latency_sla_ms: float = 200.0
    min_compute_util: float = 0.3
    max_temperature_c: float = 85.0
    metrics_window_size: int = 100

class ModelWarmup:
    """Handles model warmup and performance monitoring."""
    
    def __init__(self, config: Optional[WarmupConfig] = None):
        self.config = config or WarmupConfig()
        self.metrics_history = deque(maxlen=self.config.metrics_window_size)
        self._current_batch_size = self.config.initial_batch_size
        self._warmup_complete = False
        
    def _get_metal_metrics(self) -> Dict[str, float]:
        """Get Metal GPU performance metrics."""
        try:
            # For now return simulated metrics since we don't have IOKit access
            # In production, this would use IOKit to get real GPU metrics
            return {
                'compute_util': np.random.uniform(60.0, 90.0),  # Simulated GPU utilization
                'temperature_c': np.random.uniform(45.0, 75.0),  # Simulated temperature
                'power_w': np.random.uniform(10.0, 30.0)  # Simulated power draw
            }
        except Exception as e:
            logger.warning(f"Failed to get Metal metrics: {e}")
            return {
                'compute_util': 0.0,
                'temperature_c': 0.0,
                'power_w': 0.0
            }
            # For now, return placeholder values
            return {
                "temperature_c": 45.0,
                "power_w": 15.0,
                "compute_util": 0.5
            }
        except Exception as e:
            logger.warning(f"Failed to get Metal metrics: {e}")
            return {
                "temperature_c": 0.0,
                "power_w": 0.0,
                "compute_util": 0.0
            }
            
    def _collect_metrics(self, latency_ms: float, queue_depth: int) -> Dict:
        """Collect comprehensive system metrics."""
        metal_metrics = self._get_metal_metrics()
        memory = psutil.virtual_memory()
        timestamp = time.time()
        
        # Create metrics dictionary
        metrics = {
            'latency': latency_ms,
            'memory_mb': memory.used / (1024 * 1024),
            'temperature_c': metal_metrics["temperature_c"],
            'power_w': metal_metrics["power_w"],
            'compute_util': metal_metrics["compute_util"],
            'batch_size': self._current_batch_size,
            'queue_depth': queue_depth,
            'timestamp': timestamp,
            'success': True
        }
        
        # Update metrics history
        self.metrics_history.append(metrics)
        
        return metrics
        
    def _should_increase_batch_size(self, recent_metrics: List[Dict]) -> bool:
        """Determine if batch size should be increased based on metrics."""
        if not recent_metrics:
            return False
            
        # Calculate statistics over recent window
        latencies = []
        temperatures = []
        compute_utils = []
        
        for m in recent_metrics:
            # Handle latency with different possible keys
            if 'latency_ms' in m:
                latencies.append(m['latency_ms'])
            elif 'latency' in m:
                latencies.append(m['latency'])
            else:
                latencies.append(0.0)
            
            # Handle other metrics with defaults
            temperatures.append(m.get('temperature_c', 0.0))
            compute_utils.append(m.get('compute_util', 0.0))
        
        if not latencies:  # Safety check
            return False
        
        avg_latency = np.mean(latencies)
        max_temp = np.max(temperatures)
        avg_compute = np.mean(compute_utils)
        p95_latency = np.percentile(latencies, 95)
        
        logger.debug(f"Metrics - Avg latency: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, "  
                    f"Max temp: {max_temp:.1f}C, Avg compute: {avg_compute:.2f}")
        
        # Check all conditions
        return (
            avg_latency < self.config.target_latency_ms and
            p95_latency < self.config.latency_sla_ms and
            max_temp < self.config.max_temperature_c and
            avg_compute > self.config.min_compute_util and
            self._current_batch_size < self.config.max_batch_size
        )
        
    def _adjust_batch_size(self):
        """Adjust batch size based on recent performance metrics."""
        recent_metrics = list(self.metrics_history)[-10:]
        
        if self._should_increase_batch_size(recent_metrics):
            new_batch_size = min(
                int(self._current_batch_size * self.config.scaling_factor),
                self.config.max_batch_size
            )
            if new_batch_size != self._current_batch_size:
                logger.info(f"Increasing batch size from {self._current_batch_size} to {new_batch_size}")
                self._current_batch_size = new_batch_size
                
    async def warmup_iteration(self, inference_fn, input_data: Dict) -> Dict:
        """Run a single warmup iteration."""
        start_time = time.time()
        
        try:
            # Create batch based on current batch size
            batch = [input_data] * self._current_batch_size
            logger.debug(f"Created batch with size {len(batch)}, first item: {batch[0]}")
            
            # Run inference
            logger.debug(f"Running inference with batch size {self._current_batch_size}")
            try:
                results = await inference_fn(batch)
                logger.debug(f"Inference complete, results type: {type(results)}, content: {results}")
            except Exception as e:
                logger.error(f"Inference failed: {str(e)}")
                raise
            
            # Calculate metrics
            end_time = time.time()
            
            # Get Metal metrics
            metal_metrics = self._get_metal_metrics()
            
            # Extract latency from results
            logger.debug(f"Processing results for latency calculation")
            if results and isinstance(results, list):
                # Calculate average latency across batch
                latencies = [r.get('latency', 0.0) for r in results if isinstance(r, dict)]
                if latencies:
                    per_request_latency = np.mean(latencies)
                    logger.debug(f"Calculated mean latency from results: {per_request_latency}ms/req")
                else:
                    logger.warning("No valid latencies found in results")
                    return None
            else:
                logger.warning("Invalid results format received")
                return None
                
            # Collect metrics
            metrics = self._collect_metrics(
                latency_ms=per_request_latency,
                queue_depth=0  # Queue depth is 0 during warmup
            )
            
            # Adjust batch size based on performance
            self._adjust_batch_size()
            
            logger.debug(f"Warmup iteration complete - Batch size: {self._current_batch_size}, "
                        f"Latency: {per_request_latency:.2f}ms/req")
            
            # Add cooldown if needed
            if self.config.cooldown_seconds > 0:
                await asyncio.sleep(self.config.cooldown_seconds)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Warmup iteration failed: {e}")
            raise
            
    def get_optimal_batch_size(self) -> int:
        """Get the current optimal batch size."""
        return self._current_batch_size
        
    def get_performance_summary(self) -> Dict[str, float]:
        """Generate performance summary from warmup phase."""
        if not self.metrics_history:
            return {}
            
        recent_metrics = list(self.metrics_history)
        
        # Extract latencies with fallback
        latencies = []
        for m in recent_metrics:
            if 'latency_ms' in m:
                latencies.append(m['latency_ms'])
            elif 'latency' in m:
                latencies.append(m['latency'])
            else:
                latencies.append(0.0)
        
        # Extract other metrics with defaults
        compute_utils = [m.get('compute_util', 0.0) for m in recent_metrics]
        power_watts = [m.get('power_w', 0.0) for m in recent_metrics]
        temperatures = [m.get('temperature_c', 0.0) for m in recent_metrics]
        
        return {
            "avg_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "avg_compute_util": float(np.mean(compute_utils)),
            "avg_power_w": float(np.mean(power_watts)),
            "avg_temperature_c": float(np.mean(temperatures)),
            "optimal_batch_size": self._current_batch_size
        }
