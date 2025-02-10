"""Advanced metrics collection and monitoring for batch inference."""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque
import numpy as np
import logging
from datetime import datetime
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RequestTrace:
    request_id: str
    priority: int
    batch_size: int
    queue_time: float
    processing_time: float
    total_latency: float
    timestamp: float = field(default_factory=time.time)
    batch_id: Optional[str] = None

@dataclass
class BatchMetrics:
    batch_id: str
    size: int
    processing_time: float
    gpu_utilization: float
    memory_utilization: float
    queue_size: int
    priorities: List[int]
    timestamp: float = field(default_factory=time.time)

class MetricsCollector:
    def __init__(self, metrics_window: int = 1000, export_path: str = "metrics"):
        self.metrics_window = metrics_window
        self.export_path = Path(export_path)
        self.export_path.mkdir(exist_ok=True)
        
        # Metrics storage
        self._request_traces: Deque[RequestTrace] = deque(maxlen=metrics_window)
        self._batch_metrics: Deque[BatchMetrics] = deque(maxlen=metrics_window)
        self._lock = threading.Lock()
        
        # Real-time statistics
        self._current_stats = {
            "avg_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "avg_batch_size": 0.0,
            "throughput": 0.0,
            "gpu_utilization": 0.0,
            "memory_utilization": 0.0
        }
        
        # Start periodic export
        self._start_metrics_export()

    def add_request_trace(self, trace: RequestTrace):
        """Add a new request trace to the collector."""
        with self._lock:
            self._request_traces.append(trace)
            self._update_stats()

    def add_batch_metrics(self, metrics: BatchMetrics):
        """Add new batch metrics to the collector."""
        with self._lock:
            self._batch_metrics.append(metrics)
            self._update_stats()

    def _update_stats(self):
        """Update real-time statistics."""
        if not self._request_traces:
            return

        # Calculate latency statistics
        latencies = [t.total_latency for t in self._request_traces]
        self._current_stats.update({
            "avg_latency": np.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99)
        })

        # Calculate batch statistics
        if self._batch_metrics:
            recent_batches = list(self._batch_metrics)
            self._current_stats.update({
                "avg_batch_size": np.mean([b.size for b in recent_batches]),
                "gpu_utilization": np.mean([b.gpu_utilization for b in recent_batches]),
                "memory_utilization": np.mean([b.memory_utilization for b in recent_batches])
            })

        # Calculate throughput (requests/second)
        window_time = self._request_traces[-1].timestamp - self._request_traces[0].timestamp
        if window_time > 0:
            self._current_stats["throughput"] = len(self._request_traces) / window_time

    def get_current_stats(self) -> Dict:
        """Get the current statistics."""
        with self._lock:
            return dict(self._current_stats)

    def export_metrics(self):
        """Export metrics to disk."""
        timestamp = datetime.now().strftime("%Y%m%d")
        export_file = self.export_path / f"metrics_{timestamp}.json"
        
        with self._lock:
            metrics_data = {
                "request_traces": [{
                    "request_id": t.request_id,
                    "priority": t.priority,
                    "batch_size": t.batch_size,
                    "queue_time": t.queue_time,
                    "processing_time": t.processing_time,
                    "total_latency": t.total_latency,
                    "timestamp": t.timestamp,
                    "batch_id": t.batch_id
                } for t in self._request_traces],
                "batch_metrics": [{
                    "batch_id": b.batch_id,
                    "size": b.size,
                    "processing_time": b.processing_time,
                    "gpu_utilization": b.gpu_utilization,
                    "memory_utilization": b.memory_utilization,
                    "queue_size": b.queue_size,
                    "priorities": b.priorities,
                    "timestamp": b.timestamp
                } for b in self._batch_metrics]
            }
        
        with open(export_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Exported metrics to {export_file}")

    def _start_metrics_export(self):
        """Start periodic metrics export."""
        def export_loop():
            while True:
                time.sleep(60)  # Export every minute
                try:
                    self.export_metrics()
                except Exception as e:
                    logger.error(f"Failed to export metrics: {e}")

        thread = threading.Thread(target=export_loop, daemon=True)
        thread.start()
