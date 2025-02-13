import time
import threading
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Deque, Optional
import json
import logging

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
    throughput: float
    avg_latency_ms: float
    p95_latency_ms: float
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    timestamp: float = field(default_factory=time.time)

class MetricsCollector:
    def __init__(self, metrics_window: int = 1000, export_path: str = "metrics"):
        self.metrics_window = metrics_window
        self.export_path = Path(export_path)
        self.export_path.mkdir(exist_ok=True)
        
        # Thread-safe metrics storage
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
        
        self.last_report_time = time.time()
        self.report_interval = 60  # seconds
        
        # Start periodic export
        self._start_metrics_export()
        
    def add_request_trace(self, trace: RequestTrace) -> None:
        """Add a request trace to the collector."""
        with self._lock:
            self._request_traces.append(trace)
            self._update_stats()

    def add_batch_metrics(self, metrics: BatchMetrics) -> None:
        """Add batch metrics to the collector."""
        with self._lock:
            self._batch_metrics.append(metrics)
            self._update_stats()

    def _update_stats(self) -> None:
        """Update real-time statistics."""
        if not self._request_traces:
            return

        latencies = [trace.total_latency for trace in self._request_traces]
        self._current_stats.update({
            "avg_latency": np.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "avg_batch_size": np.mean([trace.batch_size for trace in self._request_traces])
        })

        if self._batch_metrics:
            self._current_stats.update({
                "gpu_utilization": np.mean([m.gpu_utilization for m in self._batch_metrics]),
                "memory_utilization": np.mean([m.memory_utilization for m in self._batch_metrics])
            })

    def _start_metrics_export(self) -> None:
        """Start periodic metrics export."""
        def export_loop():
            while True:
                self._export_metrics()
                time.sleep(self.report_interval)

        thread = threading.Thread(target=export_loop, daemon=True)
        thread.start()

    def _export_metrics(self) -> None:
        """Export metrics to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = self.export_path / f"metrics_{timestamp}.json"
            
            with self._lock:
                metrics_data = {
                    "timestamp": time.time(),
                    "stats": self._current_stats,
                    "request_traces": [vars(trace) for trace in self._request_traces],
                    "batch_metrics": [vars(metrics) for metrics in self._batch_metrics]
                }
            
            with open(export_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.info(f"Metrics exported to {export_file}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def get_current_stats(self) -> Dict:
        """Get current statistics."""
        with self._lock:
            return self._current_stats.copy()
